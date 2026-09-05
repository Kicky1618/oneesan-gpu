#pragma push_macro("main")
#undef main
#define main two_cell_component_sliding_microprobe_main_unused
#include "two_cell_component_sliding_microprobe.cu"
#pragma pop_macro("main")

namespace {

constexpr int PRIMITIVE_CHUNK = 256;

struct PrimitiveLut {
    std::vector<std::uint32_t> value;
    Rank offset[oneesan::twocell::kMaxWidth + 1]{};
};

PrimitiveLut build_primitive_lut(int W, const RankTables& tables) {
    PrimitiveLut lut;
    const int label_len = W - 2;
    for (int occupied = 1; occupied <= label_len; occupied += 2) {
        lut.offset[occupied] = lut.value.size();
        const std::uint32_t compact_support = oneesan::twocell::low_mask(occupied);
        const Rank pc = tables.primitive[occupied][1];
        for (Rank r = 0; r < pc; ++r) {
            lut.value.push_back(oneesan::twocell::primitive_left_unrank(
                compact_support, occupied, occupied, r, tables));
        }
    }
    return lut;
}

__device__ __forceinline__ std::uint32_t deposit_left_warp(
    std::uint32_t support,
    std::uint32_t compact_left,
    int len
) {
    const int lane = threadIdx.x & 31;
    const bool occupied = lane < len && ((support >> lane) & 1u);
    const int ordinal = __popc(support & oneesan::twocell::low_mask(lane));
    const bool is_left = occupied && ((compact_left >> ordinal) & 1u);
    return __ballot_sync(0xffffffffu, is_left) & oneesan::twocell::low_mask(len);
}

__global__ void two_cell_component_tiled_kernel(
    const std::uint32_t* __restrict__ input,
    std::uint32_t* __restrict__ output,
    unsigned long long* __restrict__ owner,
    const std::uint32_t* __restrict__ primitive_lut,
    Rank supports,
    Rank primitive_count,
    Rank state_count,
    int occupied_count,
    int W,
    int i,
    std::uint32_t mod,
    unsigned long long* processed,
    int* error
) {
    __shared__ PackedKey sh_src[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ PackedKey sh_dst[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_value[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_support[WARPS_PER_BLOCK];
    __shared__ int sh_ns[WARPS_PER_BLOCK];
    __shared__ int sh_nd[WARPS_PER_BLOCK];
    __shared__ int sh_in_ones[WARPS_PER_BLOCK];
    __shared__ int sh_out_ones[WARPS_PER_BLOCK];
    __shared__ Rank sh_in_base[WARPS_PER_BLOCK];
    __shared__ Rank sh_out_base[WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank chunks = (primitive_count + PRIMITIVE_CHUNK - 1) / PRIMITIVE_CHUNK;
    const Rank tiles = supports * chunks;
    const Rank first = Rank(blockIdx.x) * WARPS_PER_BLOCK + Rank(warp);
    const Rank stride = Rank(gridDim.x) * WARPS_PER_BLOCK;
    const int label_len = W - 2;

    for (Rank tile = first; tile < tiles; tile += stride) {
        const Rank support_rank = tile / chunks;
        const Rank chunk = tile - support_rank * chunks;
        const Rank primitive_begin = chunk * PRIMITIVE_CHUNK;
        Rank primitive_end = primitive_begin + PRIMITIVE_CHUNK;
        if (primitive_end > primitive_count) primitive_end = primitive_count;

        if (lane == 0) {
            const std::uint32_t support = oneesan::twocell::support_unrank(
                label_len, occupied_count, support_rank, TC_RANK_TABLES);
            sh_support[warp] = support;
            const ComponentBlocks blocks = oneesan::twocell::component_blocks(
                support, W, i, TC_RANK_TABLES);
            sh_in_ones[warp] = blocks.input_ones;
            sh_out_ones[warp] = blocks.output_ones;
            sh_in_base[warp] = blocks.input_base;
            sh_out_base[warp] = blocks.output_base;
        }
        __syncwarp();

        const std::uint32_t support = sh_support[warp];
        for (Rank pr = primitive_begin; pr < primitive_end; ++pr) {
            std::uint32_t compact_left = 0;
            if (lane == 0) compact_left = primitive_lut[pr];
            compact_left = __shfl_sync(0xffffffffu, compact_left, 0);
            const std::uint32_t left = deposit_left_warp(support, compact_left, label_len);

            if (lane == 0) {
                sh_ns[warp] = 0;
                sh_nd[warp] = 0;
                const PackedWord label{
                    support, left, static_cast<std::uint8_t>(label_len)};
                const auto src = oneesan::twocell::direct_component_sources(label, W, i);
                if (src.overflow || src.size <= 0 || src.size > MAX_STATES) {
                    set_error(error, 131);
                } else {
                    sh_ns[warp] = src.size;
                    for (int s = 0; s < src.size; ++s) sh_src[warp][s] = src.value[s];
                    for (int s = 0; s < src.size; ++s) {
                        const auto edges = oneesan::twocell::K_step(src.value[s], W, i);
                        if (edges.overflow) {
                            set_error(error, 132);
                            break;
                        }
                        for (int e = 0; e < edges.size; ++e) {
                            bool seen = false;
                            for (int d = 0; d < sh_nd[warp]; ++d)
                                seen |= oneesan::twocell::equal(
                                    sh_dst[warp][d], edges.value[e]);
                            if (seen) continue;
                            if (sh_nd[warp] >= MAX_STATES) {
                                set_error(error, 133);
                                break;
                            }
                            sh_dst[warp][sh_nd[warp]++] = edges.value[e];
                        }
                    }
                    if (sh_nd[warp] != sh_ns[warp]) set_error(error, 134);
                    atomicAdd(processed, 1ULL);
                }
            }
            __syncwarp();

            const int ns = sh_ns[warp];
            const int nd = sh_nd[warp];
            if (lane < ns) {
                const Rank r = shared_rank(
                    sh_src[warp][lane], W, i - 1,
                    sh_in_ones[warp], sh_in_base[warp]);
                if (r >= state_count) {
                    sh_value[warp][lane] = 0;
                    set_error(error, 135);
                } else {
                    sh_value[warp][lane] = input[r];
                }
            }
            __syncwarp();

            if (lane < nd) {
                const PackedKey mine = sh_dst[warp][lane];
                unsigned long long acc = 0;
                for (int s = 0; s < ns; ++s) {
                    const auto edges = oneesan::twocell::K_step(sh_src[warp][s], W, i);
                    for (int e = 0; e < edges.size; ++e)
                        if (oneesan::twocell::equal(edges.value[e], mine))
                            acc += sh_value[warp][s];
                }
                const Rank r = shared_rank(
                    mine, W, i,
                    sh_out_ones[warp], sh_out_base[warp]);
                if (r >= state_count) {
                    set_error(error, 136);
                } else {
                    const unsigned long long component_id =
                        (static_cast<unsigned long long>(support_rank) << 32) |
                        static_cast<unsigned long long>(pr);
                    const unsigned long long empty = ~0ULL;
                    const unsigned long long previous = atomicCAS(owner + r, empty, component_id);
                    if (previous != empty && previous != component_id) set_error(error, 137);
                    output[r] = static_cast<std::uint32_t>(acc % mod);
                }
            }
            __syncwarp();
        }
    }
}

Rank sector_tiles(int W, int occupied, const RankTables& tables) {
    const Rank supports = tables.choose[W - 2][occupied];
    const Rank pc = tables.primitive[occupied][1];
    return supports * ((pc + PRIMITIVE_CHUNK - 1) / PRIMITIVE_CHUNK);
}

Rank total_tiles(int W, const RankTables& tables) {
    Rank z = 0;
    for (int occupied = 1; occupied <= W - 2; occupied += 2)
        z += sector_tiles(W, occupied, tables);
    return z;
}

void run_tiled_position(
    int W,
    int i,
    const RankTables& tables,
    const PrimitiveLut& host_lut,
    std::uint32_t mod,
    unsigned requested_blocks
) {
    const Rank states = tables.suffix[W - 5][0];
    const Rank components = oneesan::twocell::component_label_count(W, tables);

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(W + 1));
    for (int n = 1; n <= W; ++n) words[n] = gen_words(n);
    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    std::vector<std::uint32_t> reference(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + Rank(i) * 17ULL) % (mod - 1ULL)));

    for (const Key& s : q_basis(W, i, words)) {
        const Rank sr = oneesan::twocell::rank_state(device_key(s), W, i - 1, tables);
        const std::uint32_t value = input[static_cast<std::size_t>(sr)];
        for (const auto& [d, c] : K_basis(s, W, i)) {
            if (c != 1) std::exit(138);
            const Rank dr = oneesan::twocell::rank_state(device_key(d), W, i, tables);
            reference[static_cast<std::size_t>(dr)] =
                add_mod(reference[static_cast<std::size_t>(dr)], value, mod);
        }
    }

    std::uint32_t* d_input = nullptr;
    std::uint32_t* d_output = nullptr;
    std::uint32_t* d_lut = nullptr;
    unsigned long long* d_owner = nullptr;
    unsigned long long* d_processed = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_input, states * sizeof(std::uint32_t)), "tiled alloc input");
    ck(cudaMalloc(&d_output, states * sizeof(std::uint32_t)), "tiled alloc output");
    ck(cudaMalloc(&d_lut, host_lut.value.size() * sizeof(std::uint32_t)), "tiled alloc lut");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "tiled alloc owner");
    ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "tiled alloc processed");
    ck(cudaMalloc(&d_error, sizeof(int)), "tiled alloc error");
    ck(cudaMemcpy(d_input, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "tiled copy input");
    ck(cudaMemcpy(d_lut, host_lut.value.data(),
                  host_lut.value.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "tiled copy lut");
    ck(cudaMemset(d_output, 0, states * sizeof(std::uint32_t)), "tiled zero output");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "tiled clear owner");
    ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "tiled zero processed");
    ck(cudaMemset(d_error, 0, sizeof(int)), "tiled zero error");

    cudaEvent_t begin{}, end{};
    ck(cudaEventCreate(&begin), "tiled event begin");
    ck(cudaEventCreate(&end), "tiled event end");
    ck(cudaEventRecord(begin), "tiled record begin");

    for (int occupied = 1; occupied <= W - 2; occupied += 2) {
        const Rank supports = tables.choose[W - 2][occupied];
        const Rank pc = tables.primitive[occupied][1];
        const Rank tiles = sector_tiles(W, occupied, tables);
        const Rank one_pass_blocks = (tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned blocks = static_cast<unsigned>(std::max<Rank>(
            1, std::min<Rank>(requested_blocks, one_pass_blocks)));
        two_cell_component_tiled_kernel<<<blocks, THREADS>>>(
            d_input, d_output, d_owner,
            d_lut + host_lut.offset[occupied],
            supports, pc, states, occupied, W, i, mod,
            d_processed, d_error);
        ck(cudaGetLastError(), "tiled launch sector");
    }

    ck(cudaEventRecord(end), "tiled record end");
    ck(cudaEventSynchronize(end), "tiled sync end");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, begin, end), "tiled elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    std::vector<unsigned long long> owner(static_cast<std::size_t>(states));
    unsigned long long processed = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_output, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
       "tiled copy output");
    ck(cudaMemcpy(owner.data(), d_owner, states * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "tiled copy owner");
    ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost),
       "tiled copy processed");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "tiled copy error");

    if (error || processed != components) {
        std::cerr << "FAIL tiled two-cell error=" << error
                  << " processed=" << processed << " expected=" << components << '\n';
        std::exit(139);
    }
    for (Rank r = 0; r < states; ++r) {
        if (owner[static_cast<std::size_t>(r)] == std::numeric_limits<unsigned long long>::max() ||
            output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL tiled arithmetic W=" << W << " i=" << i
                      << " rank=" << r << '\n';
            std::exit(140);
        }
    }

    std::cout << "two-cell-component-tiled-microprobe"
              << " W=" << W
              << " i=" << i
              << " states=" << states
              << " components=" << components
              << " primitive_chunk=" << PRIMITIVE_CHUNK
              << " tiles=" << total_tiles(W, tables)
              << " components_per_tile="
              << double(components) / double(total_tiles(W, tables))
              << " primitive_lut_KiB="
              << double(host_lut.value.size() * sizeof(std::uint32_t)) / 1024.0
              << " kernel_ms=" << ms
              << " support_unrank_amortized=1 block_base_amortized=1"
              << " arithmetic=OK\n";

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_lut);
    cudaFree(d_owner);
    cudaFree(d_processed);
    cudaFree(d_error);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const unsigned blocks = argc > 2
        ? static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10)) : 4096u;
    const std::uint32_t mod = argc > 3
        ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth || blocks == 0 || mod < 3) return 2;

    const RankTables tables = oneesan::twocell::make_rank_tables();
    const PrimitiveLut lut = build_primitive_lut(W, tables);
    const Rank states = tables.suffix[W - 5][0];
    const Rank components = oneesan::twocell::component_label_count(W, tables);
    const Rank tiles = total_tiles(W, tables);

    if (plan_only) {
        std::cout << "two-cell-component-tiled-microprobe-plan"
                  << " W=" << W
                  << " states=" << states
                  << " components=" << components
                  << " primitive_chunk=" << PRIMITIVE_CHUNK
                  << " tiles=" << tiles
                  << " components_per_tile=" << double(components) / double(tiles)
                  << " support_unrank_reduction=" << double(components) / double(tiles)
                  << " primitive_lut_MiB="
                  << double(lut.value.size() * sizeof(std::uint32_t)) / double(1ULL << 20)
                  << " rank_table_KiB=" << double(sizeof(tables)) / 1024.0
                  << " global_component_table_bytes=0 permutation_table_bytes=0"
                  << "\n";
        for (int occupied = 1; occupied <= W - 2; occupied += 2) {
            const Rank supports = tables.choose[W - 2][occupied];
            const Rank pc = tables.primitive[occupied][1];
            std::cout << "sector occupied=" << occupied
                      << " supports=" << supports
                      << " primitive=" << pc
                      << " components=" << supports * pc
                      << " tiles=" << sector_tiles(W, occupied, tables)
                      << "\n";
        }
        return 0;
    }

    if (W > 11) {
        std::cerr << "execution mode is intentionally limited to W<=11; use --plan-only above that\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "tiled device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "tiled set device");
    install_tables(tables);

    for (int i = 1; i <= W - 5; ++i)
        run_tiled_position(W, i, tables, lut, mod, blocks);
    std::cout << "ALL_OK two_cell_component_tiled_cuda=1 W=" << W << '\n';
    return 0;
}
