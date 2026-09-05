#pragma push_macro("main")
#undef main
#define main two_cell_component_tiled_edges_microprobe_main_unused
#include "two_cell_component_tiled_edges_microprobe.cu"
#pragma pop_macro("main")

namespace {

__global__ void two_cell_component_reuse_kernel(
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
    unsigned long long* primitive_scans,
    int* error
) {
    __shared__ PackedKey sh_src[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ PackedKey sh_edge[WARPS_PER_BLOCK][MAX_STATES][3];
    __shared__ std::uint32_t sh_value[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint64_t sh_primitive[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint8_t sh_edge_n[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_support[WARPS_PER_BLOCK];
    __shared__ int sh_ns[WARPS_PER_BLOCK];
    __shared__ int sh_retained[WARPS_PER_BLOCK];
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
            std::uint32_t compact_left = lane == 0 ? primitive_lut[pr] : 0;
            compact_left = __shfl_sync(0xffffffffu, compact_left, 0);
            const std::uint32_t left = deposit_left_warp(support, compact_left, label_len);

            if (lane == 0) {
                const PackedWord label{support, left, static_cast<std::uint8_t>(label_len)};
                const auto src = oneesan::twocell::direct_component_sources(label, W, i);
                sh_retained[warp] = oneesan::twocell::symbol(label, i) != oneesan::twocell::TC_N;
                sh_ns[warp] = 0;
                if (src.overflow || src.size <= 0 || src.size > MAX_STATES) {
                    set_error(error, 151);
                } else {
                    sh_ns[warp] = src.size;
                    for (int s = 0; s < src.size; ++s) sh_src[warp][s] = src.value[s];
                    atomicAdd(processed, 1ULL);
                }
            }
            __syncwarp();

            const int ns = sh_ns[warp];
            if (lane < ns) {
                const PackedKey source = sh_src[warp][lane];
                Rank primitive = pr;
                const bool reuse_label = sh_retained[warp] && lane < 3;
                if (!reuse_label) {
                    const int len = source.type ? W - 2 : W - 1;
                    primitive = oneesan::twocell::primitive_rank(
                        source.support, source.left, len, TC_RANK_TABLES);
                    atomicAdd(primitive_scans, 1ULL);
                }
                sh_primitive[warp][lane] = primitive;

                const Rank r = oneesan::twocell::rank_with_block_primitive(
                    source, i - 1, sh_in_ones[warp], sh_in_base[warp],
                    primitive, TC_RANK_TABLES);
                if (r >= state_count) {
                    sh_value[warp][lane] = 0;
                    sh_edge_n[warp][lane] = 0;
                    set_error(error, 152);
                } else {
                    sh_value[warp][lane] = input[r];
                    const auto edges = oneesan::twocell::K_step(source, W, i);
                    if (edges.overflow || edges.size > 3) {
                        sh_edge_n[warp][lane] = 0;
                        set_error(error, 153);
                    } else {
                        sh_edge_n[warp][lane] = static_cast<std::uint8_t>(edges.size);
                        for (int e = 0; e < edges.size; ++e)
                            sh_edge[warp][lane][e] = edges.value[e];
                    }
                }
            }
            __syncwarp();

            if (lane < ns) {
                const PackedKey mine = oneesan::twocell::recouple_coordinate(
                    sh_src[warp][lane], i);
                unsigned long long acc = 0;
                for (int s = 0; s < ns; ++s) {
                    const int ne = sh_edge_n[warp][s];
                    for (int e = 0; e < ne; ++e)
                        if (oneesan::twocell::equal(sh_edge[warp][s][e], mine))
                            acc += sh_value[warp][s];
                }

                // recouple_coordinate only moves a C vacancy across one occupied
                // symbol, so the compact primitive L/R word is unchanged.
                const Rank primitive = sh_primitive[warp][lane];
                const Rank r = oneesan::twocell::rank_with_block_primitive(
                    mine, i, sh_out_ones[warp], sh_out_base[warp],
                    primitive, TC_RANK_TABLES);
                if (r >= state_count) {
                    set_error(error, 154);
                } else {
                    const unsigned long long component_id =
                        (static_cast<unsigned long long>(support_rank) << 32) |
                        static_cast<unsigned long long>(pr);
                    const unsigned long long empty = ~0ULL;
                    const unsigned long long previous = atomicCAS(owner + r, empty, component_id);
                    if (previous != empty && previous != component_id) set_error(error, 155);
                    output[r] = static_cast<std::uint32_t>(acc % mod);
                }
            }
            __syncwarp();
        }
    }
}

void run_reuse_position(
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
            if (c != 1) std::exit(156);
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
    unsigned long long* d_primitive_scans = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_input, states * sizeof(std::uint32_t)), "reuse alloc input");
    ck(cudaMalloc(&d_output, states * sizeof(std::uint32_t)), "reuse alloc output");
    ck(cudaMalloc(&d_lut, host_lut.value.size() * sizeof(std::uint32_t)), "reuse alloc lut");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "reuse alloc owner");
    ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "reuse alloc processed");
    ck(cudaMalloc(&d_primitive_scans, sizeof(unsigned long long)), "reuse alloc primitive scans");
    ck(cudaMalloc(&d_error, sizeof(int)), "reuse alloc error");
    ck(cudaMemcpy(d_input, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "reuse copy input");
    ck(cudaMemcpy(d_lut, host_lut.value.data(),
                  host_lut.value.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "reuse copy lut");
    ck(cudaMemset(d_output, 0, states * sizeof(std::uint32_t)), "reuse zero output");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "reuse clear owner");
    ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "reuse zero processed");
    ck(cudaMemset(d_primitive_scans, 0, sizeof(unsigned long long)), "reuse zero primitive scans");
    ck(cudaMemset(d_error, 0, sizeof(int)), "reuse zero error");

    cudaEvent_t begin{}, end{};
    ck(cudaEventCreate(&begin), "reuse event begin");
    ck(cudaEventCreate(&end), "reuse event end");
    ck(cudaEventRecord(begin), "reuse record begin");
    for (int occupied = 1; occupied <= W - 2; occupied += 2) {
        const Rank supports = tables.choose[W - 2][occupied];
        const Rank pc = tables.primitive[occupied][1];
        const Rank tiles = sector_tiles(W, occupied, tables);
        const Rank one_pass_blocks = (tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned blocks = static_cast<unsigned>(std::max<Rank>(
            1, std::min<Rank>(requested_blocks, one_pass_blocks)));
        two_cell_component_reuse_kernel<<<blocks, THREADS>>>(
            d_input, d_output, d_owner,
            d_lut + host_lut.offset[occupied],
            supports, pc, states, occupied, W, i, mod,
            d_processed, d_primitive_scans, d_error);
        ck(cudaGetLastError(), "reuse launch sector");
    }
    ck(cudaEventRecord(end), "reuse record end");
    ck(cudaEventSynchronize(end), "reuse sync end");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, begin, end), "reuse elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    std::vector<unsigned long long> owner(static_cast<std::size_t>(states));
    unsigned long long processed = 0;
    unsigned long long primitive_scans = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_output, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
       "reuse copy output");
    ck(cudaMemcpy(owner.data(), d_owner, states * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "reuse copy owner");
    ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost),
       "reuse copy processed");
    ck(cudaMemcpy(&primitive_scans, d_primitive_scans, sizeof(primitive_scans), cudaMemcpyDeviceToHost),
       "reuse copy primitive scans");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "reuse copy error");
    if (error || processed != components) {
        std::cerr << "FAIL reuse error=" << error << " processed=" << processed
                  << " expected=" << components << '\n';
        std::exit(157);
    }
    for (Rank r = 0; r < states; ++r) {
        if (owner[static_cast<std::size_t>(r)] == std::numeric_limits<unsigned long long>::max() ||
            output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL reuse arithmetic W=" << W << " i=" << i
                      << " rank=" << r << '\n';
            std::exit(158);
        }
    }

    std::cout << "two-cell-component-reuse-microprobe"
              << " W=" << W
              << " i=" << i
              << " states=" << states
              << " components=" << components
              << " primitive_chunk=" << PRIMITIVE_CHUNK
              << " tiles=" << total_tiles(W, tables)
              << " kernel_ms=" << ms
              << " primitive_scans=" << primitive_scans
              << " baseline_primitive_scans=" << (2 * states)
              << " destination_primitive_scans=0"
              << " destination_list_build=0"
              << " K_evaluations_per_source=1 arithmetic=OK\n";

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_lut);
    cudaFree(d_owner);
    cudaFree(d_processed);
    cudaFree(d_primitive_scans);
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
    const Rank states = tables.suffix[W - 5][0];
    const Rank components = oneesan::twocell::component_label_count(W, tables);
    const Rank tiles = total_tiles(W, tables);
    if (plan_only) {
        Rank lut_entries = 0;
        for (int occupied = 1; occupied <= W - 2; occupied += 2)
            lut_entries += tables.primitive[occupied][1];
        Rank source_scan_upper = states;
        if (W >= 5) {
            // Interior two-cell component counts: singleton=M_{W-3},
            // triple=M_{W-2}-2M_{W-3}, deep=M_{W-3}. The first three
            // coordinates of every retained (triple/deep) component use the
            // label primitive rank directly.
            source_scan_upper = states; // exact formula printed for W=28 below
        }
        std::cout << "two-cell-component-reuse-microprobe-plan"
                  << " W=" << W
                  << " states=" << states
                  << " components=" << components
                  << " primitive_chunk=" << PRIMITIVE_CHUNK
                  << " tiles=" << tiles
                  << " components_per_tile=" << double(components) / double(tiles)
                  << " primitive_lut_MiB="
                  << double(lut_entries * sizeof(std::uint32_t)) / double(1ULL << 20)
                  << " destination_primitive_scans=0 destination_list_build=0"
                  << " recoupling=A_identity+C_xN_swap"
                  << "\n";
        if (W == 28) {
            const Rank m26 = 47337954326ULL;
            const Rank m25 = 16626415975ULL;
            const Rank scan = states - 3 * (m26 - m25);
            std::cout << "W=28 primitive_scan_upper=" << scan
                      << " baseline=" << (2 * states)
                      << " reduction_lower_bound=" << double(2 * states) / double(scan)
                      << "\n";
        }
        return 0;
    }

    if (W > 11) {
        std::cerr << "execution mode is intentionally limited to W<=11; use --plan-only above that\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "reuse device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "reuse set device");
    install_tables(tables);
    const PrimitiveLut lut = build_primitive_lut(W, tables);
    for (int i = 1; i <= W - 5; ++i)
        run_reuse_position(W, i, tables, lut, mod, blocks);
    std::cout << "ALL_OK two_cell_component_reuse_cuda=1 W=" << W << '\n';
    return 0;
}
