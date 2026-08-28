#pragma push_macro("main")
#undef main
#define main two_cell_component_stationary_tiledbase_microprobe_main_unused
#include "two_cell_component_stationary_tiledbase_microprobe.cu"
#pragma pop_macro("main")

namespace {

__device__ __forceinline__ std::uint32_t add_mod_u32(
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t mod
) {
    const unsigned long long z =
        static_cast<unsigned long long>(a) + static_cast<unsigned long long>(b);
    return static_cast<std::uint32_t>(z >= mod ? z - mod : z);
}

__global__ void two_cell_stationary_lifting_kernel(
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
    unsigned long long* shear_ops,
    int* error
) {
    __shared__ PackedKey sh_src[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_value[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ Rank sh_rank[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::int8_t sh_to[WARPS_PER_BLOCK][MAX_STATES][2];
    __shared__ std::uint8_t sh_nto[WARPS_PER_BLOCK][MAX_STATES];
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
                    set_error(error, 181);
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
                const bool reuse_label = bases.retained &&
                    (lane < 3 || (lane == 3 && bases.xN_deep));
                if (!reuse_label) {
                    const int len = source.type ? W - 2 : W - 1;
                    primitive = oneesan::twocell::primitive_rank(
                        source.support, source.left, len, TC_RANK_TABLES);
                    atomicAdd(primitive_scans, 1ULL);
                }

                const Rank r = oneesan::twocell::stationary_component_source_base(
                    bases, lane) + primitive;
                sh_rank[warp][lane] = r;
                sh_nto[warp][lane] = 0;
                if (r >= state_count) {
                    sh_value[warp][lane] = 0;
                    set_error(error, 182);
                } else {
                    sh_value[warp][lane] = values[r];
                    const auto edges = oneesan::twocell::K_step(source, W, i);
                    if (edges.overflow || edges.size < 1 || edges.size > 3) {
                        set_error(error, 183);
                    } else {
                        const PackedKey diagonal = oneesan::twocell::recouple_coordinate(source, i);
                        bool found_diagonal = false;
                        int ne = 0;
                        for (int e = 0; e < edges.size; ++e) {
                            const PackedKey d = edges.value[e];
                            if (oneesan::twocell::equal(d, diagonal)) {
                                found_diagonal = true;
                                continue;
                            }
                            int target = -1;
                            for (int t = 0; t < ns; ++t) {
                                const PackedKey candidate = oneesan::twocell::recouple_coordinate(
                                    sh_src[warp][t], i);
                                if (oneesan::twocell::equal(d, candidate)) {
                                    target = t;
                                    break;
                                }
                            }
                            if (target < 0 || ne >= 2) {
                                set_error(error, 184);
                                continue;
                            }
                            sh_to[warp][lane][ne++] = static_cast<std::int8_t>(target);
                        }
                        sh_nto[warp][lane] = static_cast<std::uint8_t>(ne);
                        if (!found_diagonal) set_error(error, 185);
                    }
                }
            }
            __syncwarp();

            // The diagonal matching is implicit identity. The remaining n-1
            // edges form a directed tree on stationary coordinates. Lane zero
            // removes sinks and executes each shear exactly once. A sink's value
            // is still original because no incoming source can have been removed
            // earlier while that edge pointed to a live vertex.
            if (lane == 0 && ns > 0) {
                std::uint32_t alive = oneesan::twocell::low_mask(ns);
                int removed = 0;
                int local_shears = 0;
                while (alive) {
                    std::uint32_t sinks = 0;
                    for (int v = 0; v < ns; ++v) {
                        if (!((alive >> v) & 1u)) continue;
                        bool has_live_out = false;
                        const int ne = sh_nto[warp][v];
                        for (int e = 0; e < ne; ++e) {
                            const int t = sh_to[warp][v][e];
                            has_live_out |= ((alive >> t) & 1u) != 0;
                        }
                        if (!has_live_out) sinks |= std::uint32_t(1) << v;
                    }
                    if (!sinks) {
                        set_error(error, 186);
                        break;
                    }

                    for (int v = 0; v < ns; ++v) {
                        if (!((sinks >> v) & 1u)) continue;
                        const std::uint32_t value = sh_value[warp][v];
                        const int ne = sh_nto[warp][v];
                        for (int e = 0; e < ne; ++e) {
                            const int t = sh_to[warp][v][e];
                            sh_value[warp][t] = add_mod_u32(
                                sh_value[warp][t], value, mod);
                            ++local_shears;
                        }
                        ++removed;
                    }
                    alive &= ~sinks;
                }
                if (removed != ns || local_shears != ns - 1)
                    set_error(error, 187);
                atomicAdd(shear_ops, static_cast<unsigned long long>(local_shears));
            }
            __syncwarp();

            if (lane < ns) {
                const Rank r = sh_rank[warp][lane];
                const unsigned long long component_id =
                    (static_cast<unsigned long long>(support_rank) << 32) |
                    static_cast<unsigned long long>(pr);
                const unsigned long long empty = ~0ULL;
                const unsigned long long previous = atomicCAS(
                    owner + r, empty, component_id);
                if (previous != empty && previous != component_id)
                    set_error(error, 188);
                values[r] = sh_value[warp][lane];
            }
            __syncwarp();
        }
    }
}

void run_stationary_lifting_position(
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
            if (c != 1) std::exit(189);
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
    unsigned long long* d_shear_ops = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)), "lifting alloc values");
    ck(cudaMalloc(&d_lut, host_lut.value.size() * sizeof(std::uint32_t)), "lifting alloc lut");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "lifting alloc owner");
    ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "lifting alloc processed");
    ck(cudaMalloc(&d_primitive_scans, sizeof(unsigned long long)), "lifting alloc primitive scans");
    ck(cudaMalloc(&d_support_base_builds, sizeof(unsigned long long)), "lifting alloc support builds");
    ck(cudaMalloc(&d_shear_ops, sizeof(unsigned long long)), "lifting alloc shears");
    ck(cudaMalloc(&d_error, sizeof(int)), "lifting alloc error");
    ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "lifting copy input");
    ck(cudaMemcpy(d_lut, host_lut.value.data(),
                  host_lut.value.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "lifting copy lut");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "lifting clear owner");
    ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "lifting zero processed");
    ck(cudaMemset(d_primitive_scans, 0, sizeof(unsigned long long)), "lifting zero primitive scans");
    ck(cudaMemset(d_support_base_builds, 0, sizeof(unsigned long long)), "lifting zero support builds");
    ck(cudaMemset(d_shear_ops, 0, sizeof(unsigned long long)), "lifting zero shears");
    ck(cudaMemset(d_error, 0, sizeof(int)), "lifting zero error");

    cudaEvent_t begin{}, end{};
    ck(cudaEventCreate(&begin), "lifting event begin");
    ck(cudaEventCreate(&end), "lifting event end");
    ck(cudaEventRecord(begin), "lifting record begin");
    for (int occupied = 1; occupied <= W - 2; occupied += 2) {
        const Rank supports = rt.choose[W - 2][occupied];
        const Rank pc = rt.primitive[occupied][1];
        const Rank tiles = sector_tiles(W, occupied, rt);
        const Rank one_pass_blocks = (tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned blocks = static_cast<unsigned>(std::max<Rank>(
            1, std::min<Rank>(requested_blocks, one_pass_blocks)));
        two_cell_stationary_lifting_kernel<<<blocks, THREADS>>>(
            d_values, d_owner, d_lut + host_lut.offset[occupied],
            supports, pc, states, occupied, W, i, mod,
            d_processed, d_primitive_scans, d_support_base_builds,
            d_shear_ops, d_error);
        ck(cudaGetLastError(), "lifting launch sector");
    }
    ck(cudaEventRecord(end), "lifting record end");
    ck(cudaEventSynchronize(end), "lifting sync end");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, begin, end), "lifting elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    unsigned long long processed = 0;
    unsigned long long primitive_scans = 0;
    unsigned long long support_base_builds = 0;
    unsigned long long shear_ops = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
       "lifting copy output");
    ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost),
       "lifting copy processed");
    ck(cudaMemcpy(&primitive_scans, d_primitive_scans, sizeof(primitive_scans), cudaMemcpyDeviceToHost),
       "lifting copy primitive scans");
    ck(cudaMemcpy(&support_base_builds, d_support_base_builds, sizeof(support_base_builds), cudaMemcpyDeviceToHost),
       "lifting copy support builds");
    ck(cudaMemcpy(&shear_ops, d_shear_ops, sizeof(shear_ops), cudaMemcpyDeviceToHost),
       "lifting copy shears");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "lifting copy error");
    if (error || processed != components || shear_ops != states - components) {
        std::cerr << "FAIL lifting error=" << error
                  << " processed=" << processed << " expected=" << components
                  << " shears=" << shear_ops << " expected_shears=" << (states - components)
                  << '\n';
        std::exit(190);
    }
    for (Rank r = 0; r < states; ++r) {
        if (output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL lifting arithmetic W=" << W << " i=" << i
                      << " rank=" << r << '\n';
            std::exit(191);
        }
    }

    std::cout << "two-cell-stationary-lifting-microprobe"
              << " W=" << W
              << " i=" << i
              << " states=" << states
              << " components=" << components
              << " shear_ops=" << shear_ops
              << " sparse_nnz_baseline=" << (states + shear_ops)
              << " explicit_edge_fraction="
              << double(shear_ops) / double(states + shear_ops)
              << " primitive_scans=" << primitive_scans
              << " support_base_builds=" << support_base_builds
              << " kernel_ms=" << ms
              << " value_vectors=1 destination_rank_calls=0"
              << " destination_accumulation_scan=0 arithmetic=OK\n";

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(d_values);
    cudaFree(d_lut);
    cudaFree(d_owner);
    cudaFree(d_processed);
    cudaFree(d_primitive_scans);
    cudaFree(d_support_base_builds);
    cudaFree(d_shear_ops);
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
    const Rank shears = states - components;
    if (plan_only) {
        Rank lut_entries = 0;
        for (int occupied = 1; occupied <= W - 2; occupied += 2)
            lut_entries += rt.primitive[occupied][1];
        std::cout << "two-cell-stationary-lifting-microprobe-plan"
                  << " W=" << W
                  << " states=" << states
                  << " components=" << components
                  << " shear_ops=" << shears
                  << " sparse_nnz_baseline=" << (states + shears)
                  << " explicit_edge_fraction="
                  << double(shears) / double(states + shears)
                  << " value_vectors=1 destination_rank_calls=0"
                  << " destination_accumulation_scan=0"
                  << " primitive_lut_MiB="
                  << double(lut_entries * sizeof(std::uint32_t)) / double(1ULL << 20)
                  << "\n";
        if (W == 28) {
            std::cout << "W=28 one_vector_GiB="
                      << double(states * 4ULL) / double(1ULL << 30)
                      << " diagonal_edges_elided=" << states
                      << " explicit_shears=" << shears
                      << "\n";
        }
        return 0;
    }

    if (W > 11) {
        std::cerr << "execution mode is intentionally limited to W<=11; use --plan-only above that\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "lifting device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "lifting set device");
    install_tables(rt);
    install_stationary_tables(st);
    const PrimitiveLut lut = build_primitive_lut(W, rt);
    for (int i = 1; i <= W - 5; ++i)
        run_stationary_lifting_position(W, i, rt, st, lut, mod, blocks);
    std::cout << "ALL_OK two_cell_stationary_lifting_cuda=1 W=" << W << '\n';
    return 0;
}
