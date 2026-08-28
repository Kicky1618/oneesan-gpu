#pragma push_macro("main")
#undef main
#define main two_cell_component_stationary_inplace_microprobe_main_unused
#include "two_cell_component_stationary_inplace_microprobe.cu"
#pragma pop_macro("main")

namespace {

__global__ void two_cell_stationary_tiledbase_kernel(
    std::uint32_t* __restrict__ values,
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
    unsigned long long* support_base_builds,
    int* error
) {
    __shared__ PackedKey sh_src[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ PackedKey sh_edge[WARPS_PER_BLOCK][MAX_STATES][3];
    __shared__ std::uint32_t sh_value[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint8_t sh_edge_n[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_support[WARPS_PER_BLOCK];
    __shared__ oneesan::twocell::StationaryComponentBases sh_bases[WARPS_PER_BLOCK];
    __shared__ int sh_ns[WARPS_PER_BLOCK];

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
            sh_bases[warp] = oneesan::twocell::stationary_component_bases(
                support, W, i, TC_RANK_TABLES, TC_STATIONARY_TABLES);
            atomicAdd(support_base_builds,
                      static_cast<unsigned long long>(sh_bases[warp].count));
        }
        __syncwarp();

        const std::uint32_t support = sh_support[warp];
        const auto bases = sh_bases[warp];
        for (Rank pr = primitive_begin; pr < primitive_end; ++pr) {
            std::uint32_t compact_left = lane == 0 ? primitive_lut[pr] : 0;
            compact_left = __shfl_sync(0xffffffffu, compact_left, 0);
            const std::uint32_t left = deposit_left_warp(support, compact_left, label_len);

            if (lane == 0) {
                const PackedWord label{support, left, static_cast<std::uint8_t>(label_len)};
                const auto src = oneesan::twocell::direct_component_sources(label, W, i);
                sh_ns[warp] = 0;
                if (src.overflow || src.size <= 0 || src.size > MAX_STATES) {
                    set_error(error, 171);
                } else {
                    sh_ns[warp] = src.size;
                    for (int s = 0; s < src.size; ++s) sh_src[warp][s] = src.value[s];
                    atomicAdd(processed, 1ULL);
                }
            }
            __syncwarp();

            const int ns = sh_ns[warp];
            Rank my_rank = 0;
            if (lane < ns) {
                const PackedKey source = sh_src[warp][lane];
                Rank primitive = pr;
                const bool reuse_label = bases.retained &&
                    (lane < 3 || (lane == 3 && bases.xN_deep));
                if (!reuse_label) {
                    const int len = source.type ? W - 2 : W - 1;
                    primitive = oneesan::twocell::primitive_rank(
                        source.support, source.left, len, TC_RANK_TABLES);
                    atomicAdd(primitive_scans, 1ULL);
                }

                my_rank = oneesan::twocell::stationary_component_source_base(
                    bases, lane) + primitive;
                if (my_rank >= state_count) {
                    sh_value[warp][lane] = 0;
                    sh_edge_n[warp][lane] = 0;
                    set_error(error, 172);
                } else {
                    sh_value[warp][lane] = values[my_rank];
                    const auto edges = oneesan::twocell::K_step(source, W, i);
                    if (edges.overflow || edges.size > 3) {
                        sh_edge_n[warp][lane] = 0;
                        set_error(error, 173);
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

                const unsigned long long component_id =
                    (static_cast<unsigned long long>(support_rank) << 32) |
                    static_cast<unsigned long long>(pr);
                const unsigned long long empty = ~0ULL;
                const unsigned long long previous = atomicCAS(
                    owner + my_rank, empty, component_id);
                if (previous != empty && previous != component_id)
                    set_error(error, 174);
                values[my_rank] = static_cast<std::uint32_t>(acc % mod);
            }
            __syncwarp();
        }
    }
}

void run_stationary_tiledbase_position(
    int W,
    int i,
    const RankTables& rt,
    const StationaryRankTables& st,
    const PrimitiveLut& host_lut,
    std::uint32_t mod,
    unsigned requested_blocks
) {
    const Rank states = st.total[W];
    const Rank components = oneesan::twocell::component_label_count(W, rt);

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(W + 1));
    for (int n = 1; n <= W; ++n) words[n] = gen_words(n);
    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    std::vector<std::uint32_t> reference(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + Rank(i) * 17ULL) % (mod - 1ULL)));
    for (const Key& s : q_basis(W, i, words)) {
        const Rank sr = oneesan::twocell::stationary_rank(
            device_key(s), W, i, rt, st);
        const std::uint32_t value = input[static_cast<std::size_t>(sr)];
        for (const auto& [d, c] : K_basis(s, W, i)) {
            if (c != 1) std::exit(175);
            const Rank dr = oneesan::twocell::stationary_rank(
                device_key(d), W, i + 1, rt, st);
            reference[static_cast<std::size_t>(dr)] =
                add_mod(reference[static_cast<std::size_t>(dr)], value, mod);
        }
    }

    std::uint32_t* d_values = nullptr;
    std::uint32_t* d_lut = nullptr;
    unsigned long long* d_owner = nullptr;
    unsigned long long* d_processed = nullptr;
    unsigned long long* d_primitive_scans = nullptr;
    unsigned long long* d_support_base_builds = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)), "tiledbase alloc values");
    ck(cudaMalloc(&d_lut, host_lut.value.size() * sizeof(std::uint32_t)), "tiledbase alloc lut");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "tiledbase alloc owner");
    ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "tiledbase alloc processed");
    ck(cudaMalloc(&d_primitive_scans, sizeof(unsigned long long)), "tiledbase alloc primitive scans");
    ck(cudaMalloc(&d_support_base_builds, sizeof(unsigned long long)), "tiledbase alloc support builds");
    ck(cudaMalloc(&d_error, sizeof(int)), "tiledbase alloc error");
    ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "tiledbase copy input");
    ck(cudaMemcpy(d_lut, host_lut.value.data(),
                  host_lut.value.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "tiledbase copy lut");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "tiledbase clear owner");
    ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "tiledbase zero processed");
    ck(cudaMemset(d_primitive_scans, 0, sizeof(unsigned long long)), "tiledbase zero primitive scans");
    ck(cudaMemset(d_support_base_builds, 0, sizeof(unsigned long long)), "tiledbase zero support builds");
    ck(cudaMemset(d_error, 0, sizeof(int)), "tiledbase zero error");

    cudaEvent_t begin{}, end{};
    ck(cudaEventCreate(&begin), "tiledbase event begin");
    ck(cudaEventCreate(&end), "tiledbase event end");
    ck(cudaEventRecord(begin), "tiledbase record begin");
    for (int occupied = 1; occupied <= W - 2; occupied += 2) {
        const Rank supports = rt.choose[W - 2][occupied];
        const Rank pc = rt.primitive[occupied][1];
        const Rank tiles = sector_tiles(W, occupied, rt);
        const Rank one_pass_blocks = (tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned blocks = static_cast<unsigned>(std::max<Rank>(
            1, std::min<Rank>(requested_blocks, one_pass_blocks)));
        two_cell_stationary_tiledbase_kernel<<<blocks, THREADS>>>(
            d_values, d_owner, d_lut + host_lut.offset[occupied],
            supports, pc, states, occupied, W, i, mod,
            d_processed, d_primitive_scans, d_support_base_builds, d_error);
        ck(cudaGetLastError(), "tiledbase launch sector");
    }
    ck(cudaEventRecord(end), "tiledbase record end");
    ck(cudaEventSynchronize(end), "tiledbase sync end");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, begin, end), "tiledbase elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    unsigned long long processed = 0;
    unsigned long long primitive_scans = 0;
    unsigned long long support_base_builds = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
       "tiledbase copy output");
    ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost),
       "tiledbase copy processed");
    ck(cudaMemcpy(&primitive_scans, d_primitive_scans, sizeof(primitive_scans), cudaMemcpyDeviceToHost),
       "tiledbase copy primitive scans");
    ck(cudaMemcpy(&support_base_builds, d_support_base_builds, sizeof(support_base_builds), cudaMemcpyDeviceToHost),
       "tiledbase copy support builds");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "tiledbase copy error");
    if (error || processed != components) {
        std::cerr << "FAIL tiledbase error=" << error << " processed=" << processed
                  << " expected=" << components << '\n';
        std::exit(176);
    }
    for (Rank r = 0; r < states; ++r) {
        if (output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL tiledbase arithmetic W=" << W << " i=" << i
                      << " rank=" << r << '\n';
            std::exit(177);
        }
    }

    std::cout << "two-cell-stationary-tiledbase-microprobe"
              << " W=" << W
              << " i=" << i
              << " states=" << states
              << " components=" << components
              << " kernel_ms=" << ms
              << " primitive_scans=" << primitive_scans
              << " support_base_builds=" << support_base_builds
              << " support_rank_inside_component=0"
              << " value_vectors=1 destination_rank_calls=0 arithmetic=OK\n";

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(d_values);
    cudaFree(d_lut);
    cudaFree(d_owner);
    cudaFree(d_processed);
    cudaFree(d_primitive_scans);
    cudaFree(d_support_base_builds);
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

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    const Rank states = st.total[W];
    const Rank components = oneesan::twocell::component_label_count(W, rt);
    const Rank tiles = total_tiles(W, rt);
    if (plan_only) {
        Rank lut_entries = 0;
        for (int occupied = 1; occupied <= W - 2; occupied += 2)
            lut_entries += rt.primitive[occupied][1];
        std::cout << "two-cell-stationary-tiledbase-microprobe-plan"
                  << " W=" << W
                  << " states=" << states
                  << " components=" << components
                  << " tiles=" << tiles
                  << " components_per_tile=" << double(components) / double(tiles)
                  << " support_base_builds_upper=" << (5 * tiles)
                  << " support_rank_inside_component=0"
                  << " value_vectors=1 destination_rank_calls=0"
                  << " primitive_lut_MiB="
                  << double(lut_entries * sizeof(std::uint32_t)) / double(1ULL << 20)
                  << "\n";
        if (W == 28) {
            const Rank m26 = 47337954326ULL;
            const Rank m25 = 16626415975ULL;
            const Rank m24 = 5850674704ULL;
            const Rank scan = states - 3 * (m26 - m25) - (m25 - m24);
            std::cout << "W=28 primitive_scan_upper=" << scan
                      << " baseline_source_plus_destination=" << (2 * states)
                      << " reduction_lower_bound=" << double(2 * states) / double(scan)
                      << " one_vector_GiB="
                      << double(states * 4ULL) / double(1ULL << 30)
                      << "\n";
        }
        return 0;
    }

    if (W > 11) {
        std::cerr << "execution mode is intentionally limited to W<=11; use --plan-only above that\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "tiledbase device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "tiledbase set device");
    install_tables(rt);
    install_stationary_tables(st);
    const PrimitiveLut lut = build_primitive_lut(W, rt);
    for (int i = 1; i <= W - 5; ++i)
        run_stationary_tiledbase_position(W, i, rt, st, lut, mod, blocks);
    std::cout << "ALL_OK two_cell_stationary_tiledbase_cuda=1 W=" << W << '\n';
    return 0;
}
