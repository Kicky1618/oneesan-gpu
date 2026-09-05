#pragma push_macro("main")
#undef main
#define main two_cell_fusion2_hybrid_microprobe_main_unused
#include "two_cell_fusion2_hybrid_microprobe.cu"
#pragma pop_macro("main")

#include "../../common/two_cell_turn_closed_device.cuh"

namespace {

constexpr int BOUNDARY_STEPS = 1; // coordinate-window growth, not operator count

__device__ __forceinline__ std::uint32_t boundary_double_mod(
    std::uint32_t x,
    std::uint32_t mod
) {
    const unsigned long long z = 2ULL * static_cast<unsigned long long>(x);
    return static_cast<std::uint32_t>(z >= mod ? z - mod : z);
}

__global__ void two_cell_boundary_fusion_kernel(
    std::uint32_t* __restrict__ values,
    int W,
    int outer_ones,
    Rank support_count,
    Rank state_count,
    std::uint32_t mod,
    unsigned long long* processed_blocks,
    unsigned long long* processed_components,
    unsigned long long* global_loads,
    unsigned long long* global_stores,
    unsigned long long* local_adds,
    int* error
) {
    extern __shared__ std::uint32_t block_values[];
    __shared__ oneesan::twocell::PackedKey sh_state[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ std::uint32_t sh_value[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ std::uint32_t sh_output[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ Rank sh_rank[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ oneesan::twocell::ComponentMatching sh_matching[FUSION_WARPS];
    __shared__ int sh_ns[FUSION_WARPS];
    __shared__ int sh_singular[FUSION_WARPS];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (warp >= FUSION_WARPS) return;

    const int start = W - 4;
    const int edge_active = W - 3;
    const int outer_bits = W - 4;
    const Rank n = oneesan::twocell::fusion_block_size(
        BOUNDARY_STEPS, outer_ones, TC_RANK_TABLES);
    const Rank nc = oneesan::twocell::fusion_component_count(
        BOUNDARY_STEPS, outer_ones, TC_RANK_TABLES);

    for (Rank support_rank = blockIdx.x; support_rank < support_count;
         support_rank += gridDim.x) {
        const std::uint32_t outer = oneesan::twocell::support_unrank(
            outer_bits, outer_ones, support_rank, TC_RANK_TABLES);

        for (Rank r = threadIdx.x; r < n; r += blockDim.x) {
            const auto d = oneesan::twocell::fusion_local_unrank_at(
                r, outer, W, start, BOUNDARY_STEPS, start, TC_RANK_TABLES);
            if (!d.valid) {
                set_error(error, 281);
                continue;
            }
            const Rank gr = oneesan::twocell::stationary_rank_with_primitive(
                d.key, W, start, d.primitive,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
            if (gr >= state_count) {
                set_error(error, 282);
                continue;
            }
            block_values[r] = values[gr];
            atomicAdd(global_loads, 1ULL);
        }
        __syncthreads();

        // Operator 1: final forward interior K_{W-4}. recouple identifies the
        // destination coordinates, while the actual unique matrix matching is
        // reconstructed locally from K edges by leaf peeling.
        for (Rank cr = warp; cr < nc; cr += FUSION_WARPS) {
            if (lane == 0) {
                sh_ns[warp] = 0;
                const auto label = oneesan::twocell::fusion_component_unrank(
                    cr, outer, W, start, BOUNDARY_STEPS, TC_RANK_TABLES);
                if (!label.valid) {
                    set_error(error, 283);
                } else {
                    const auto src = oneesan::twocell::direct_component_sources(
                        label.word, W, start);
                    if (src.overflow || src.size <= 0 || src.size > FUSION_MAX_COMPONENT) {
                        set_error(error, 284);
                    } else {
                        sh_ns[warp] = src.size;
                        for (int q = 0; q < src.size; ++q) sh_state[warp][q] = src.value[q];
                        const auto m = oneesan::twocell::build_component_matching(
                            sh_state[warp], src.size, W, start);
                        if (!m.ok) {
                            set_error(error, 286);
                        } else {
                            sh_matching[warp] = m;
                            atomicAdd(processed_components, 1ULL);
                        }
                    }
                }
            }
            __syncwarp();

            const int ns = sh_ns[warp];
            if (lane < ns) {
                const auto source = sh_state[warp][lane];
                const int len = source.type ? W - 2 : W - 1;
                const Rank primitive = oneesan::twocell::primitive_rank(
                    source.support, source.left, len, TC_RANK_TABLES);
                const Rank lr = oneesan::twocell::fusion_local_rank_at_with_primitive(
                    source, start, BOUNDARY_STEPS, start, outer_ones,
                    primitive, TC_RANK_TABLES);
                sh_rank[warp][lane] = lr;
                if (lr >= n) {
                    sh_value[warp][lane] = 0;
                    set_error(error, 285);
                } else {
                    sh_value[warp][lane] = block_values[lr];
                }
            }
            __syncwarp();

            if (lane == 0 && ns > 0) {
                const auto m = sh_matching[warp];
                if (!oneesan::twocell::apply_component_matching(
                        m, sh_value[warp], sh_output[warp], mod)) {
                    set_error(error, 287);
                } else {
                    atomicAdd(local_adds,
                              static_cast<unsigned long long>(m.residual_edges));
                }
            }
            __syncwarp();
            if (lane < ns && sh_rank[warp][lane] < n)
                block_values[sh_rank[warp][lane]] = sh_output[warp][lane];
            __syncwarp();
        }
        __syncthreads();

        // Operator 2: physical right row turn. This closed cyclic block has its
        // own exact alpha/beta/passive arithmetic and does not use the interior
        // matching assumption.
        for (Rank cr = warp; cr < nc; cr += FUSION_WARPS) {
            if (lane == 0) {
                sh_ns[warp] = 0;
                const auto label = oneesan::twocell::fusion_component_unrank(
                    cr, outer, W, start, BOUNDARY_STEPS, TC_RANK_TABLES);
                if (!label.valid) {
                    set_error(error, 291);
                } else {
                    const auto b = oneesan::twocell::right_turn_closed_block(label.word, W);
                    sh_singular[warp] = b.singular;
                    if (b.overflow || b.size < 3 || b.size > FUSION_MAX_COMPONENT) {
                        set_error(error, 292);
                    } else {
                        sh_ns[warp] = b.size;
                        for (int q = 0; q < b.size; ++q) sh_state[warp][q] = b.state[q];
                        atomicAdd(processed_components, 1ULL);
                    }
                }
            }
            __syncwarp();

            const int ns = sh_ns[warp];
            if (lane < ns) {
                const auto s = sh_state[warp][lane];
                const int len = s.type ? W - 2 : W - 1;
                const Rank primitive = oneesan::twocell::primitive_rank(
                    s.support, s.left, len, TC_RANK_TABLES);
                const Rank lr = oneesan::twocell::fusion_local_rank_at_with_primitive(
                    s, start, BOUNDARY_STEPS, edge_active, outer_ones,
                    primitive, TC_RANK_TABLES);
                sh_rank[warp][lane] = lr;
                if (lr >= n) {
                    sh_value[warp][lane] = 0;
                    set_error(error, 293);
                } else {
                    sh_value[warp][lane] = block_values[lr];
                }
            }
            __syncwarp();

            if (lane == 0 && ns > 0) {
                if (sh_singular[warp]) {
                    if (ns != 3) {
                        set_error(error, 294);
                    } else {
                        const std::uint32_t t0 = fusion_add_mod(
                            sh_value[warp][0], sh_value[warp][2], mod);
                        const std::uint32_t t1 = fusion_add_mod(
                            sh_value[warp][1], sh_value[warp][2], mod);
                        sh_value[warp][0] = boundary_double_mod(t0, mod);
                        sh_value[warp][1] = t1;
                        sh_value[warp][2] = t1;
                        atomicAdd(local_adds, 2ULL);
                    }
                } else {
                    std::uint32_t t = sh_value[warp][1];
                    for (int q = 2; q < ns; ++q)
                        t = fusion_add_mod(t, sh_value[warp][q], mod);
                    sh_value[warp][0] = fusion_add_mod(
                        boundary_double_mod(sh_value[warp][0], mod), t, mod);
                    sh_value[warp][1] = t;
                    for (int q = 2; q < ns; ++q)
                        sh_value[warp][q] = boundary_double_mod(sh_value[warp][q], mod);
                    atomicAdd(local_adds, static_cast<unsigned long long>(ns - 1));
                }
            }
            __syncwarp();
            if (lane < ns && sh_rank[warp][lane] < n)
                block_values[sh_rank[warp][lane]] = sh_value[warp][lane];
            __syncwarp();
        }
        __syncthreads();

        for (Rank r = threadIdx.x; r < n; r += blockDim.x) {
            const auto d = oneesan::twocell::fusion_local_unrank_at(
                r, outer, W, start, BOUNDARY_STEPS, edge_active, TC_RANK_TABLES);
            if (!d.valid) {
                set_error(error, 295);
                continue;
            }
            const Rank gr = oneesan::twocell::stationary_rank_with_primitive(
                d.key, W, edge_active, d.primitive,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
            if (gr >= state_count) {
                set_error(error, 296);
                continue;
            }
            values[gr] = block_values[r];
            atomicAdd(global_stores, 1ULL);
        }
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(processed_blocks, 1ULL);
        __syncthreads();
    }
}

std::vector<std::uint32_t> boundary_reference(
    const std::vector<std::uint32_t>& input,
    int W,
    const RankTables& rt,
    const StationaryRankTables& st,
    std::uint32_t mod
) {
    const int start = W - 4;
    const int edge_active = W - 3;
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(W + 1));
    for (int n = 1; n <= W; ++n) words[n] = gen_words(n);
    std::vector<std::uint32_t> mid(input.size()), out(input.size());
    for (const Key& s : q_basis(W, start, words)) {
        const Rank sr = oneesan::twocell::stationary_rank(
            device_key(s), W, start, rt, st);
        const std::uint32_t x = input[static_cast<std::size_t>(sr)];
        for (const auto& [d, c] : K_basis(s, W, start)) {
            if (c != 1) std::exit(297);
            const Rank dr = oneesan::twocell::stationary_rank(
                device_key(d), W, edge_active, rt, st);
            mid[static_cast<std::size_t>(dr)] = add_ref_mod(
                mid[static_cast<std::size_t>(dr)], x, mod);
        }
    }
    for (const Key& s : q_basis(W, edge_active, words)) {
        const Rank sr = oneesan::twocell::stationary_rank(
            device_key(s), W, edge_active, rt, st);
        const std::uint32_t x = mid[static_cast<std::size_t>(sr)];
        for (const auto& [d, c] : turn_right_basis(s, W)) {
            const Rank dr = oneesan::twocell::stationary_rank(
                device_key(d), W, edge_active, rt, st);
            const unsigned long long add = static_cast<unsigned long long>(x) *
                                           static_cast<unsigned long long>(c);
            out[static_cast<std::size_t>(dr)] = static_cast<std::uint32_t>(
                (out[static_cast<std::size_t>(dr)] + add) % mod);
        }
    }
    return out;
}

void print_boundary_singlebuffer_plan(
    int W,
    Rank shared_limit,
    Rank reserve,
    const RankTables& rt
) {
    const int outer_bits = W - 4;
    Rank fit = 0, total = 0;
    int max_o = -1;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank blocks = rt.choose[outer_bits][o];
        const Rank n = oneesan::twocell::fusion_block_size(BOUNDARY_STEPS, o, rt);
        total += blocks * n;
        if (reserve + n * sizeof(std::uint32_t) <= shared_limit) {
            fit += blocks * n;
            max_o = o;
        }
    }
    const double f = double(fit) / double(total);
    std::cout << "boundary_singlebuffer_plan"
              << " W=" << W
              << " shared_limit=" << shared_limit
              << " max_fused_outer_ones=" << max_o
              << " fused_state_fraction=" << f
              << " HBM_reduction_for_interior_plus_turn=" << 0.5 * f
              << " matching=leaf_peeling\n";
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 8;
    const Rank shared_kib = argc > 2
        ? static_cast<Rank>(std::strtoull(argv[2], nullptr, 10)) : 228ULL;
    const std::uint32_t mod = argc > 3
        ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10))
        : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth || mod < 3) return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    const Rank shared_limit = shared_kib * 1024ULL;
    if (plan_only) {
        const Rank reserve = singlebuffer_workspace_estimate();
        print_boundary_singlebuffer_plan(W, shared_limit, reserve, rt);
        if (W == 28) {
            for (Rank kib : {64ULL, 96ULL, 128ULL, 160ULL, 192ULL, 228ULL, 256ULL})
                print_boundary_singlebuffer_plan(W, kib * 1024ULL, reserve, rt);
        }
        return 0;
    }

    if (W > 10) {
        std::cerr << "execution mode intentionally limited to W<=10; use --plan-only above that\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "boundary fusion device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "boundary fusion set device");
    install_tables(rt);
    install_stationary_tables(st);

    cudaFuncAttributes attr{};
    ck(cudaFuncGetAttributes(&attr, two_cell_boundary_fusion_kernel),
       "boundary fusion attributes");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "boundary fusion props");
    const Rank hardware_limit = prop.sharedMemPerBlockOptin;
    const Rank total_limit = std::min(shared_limit, hardware_limit);
    const Rank static_shared = attr.sharedSizeBytes;
    const Rank states = st.total[W];
    const int outer_bits = W - 4;

    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + 181ULL) % (mod - 1ULL)));
    const auto reference = boundary_reference(input, W, rt, st, mod);

    std::uint32_t* d_values = nullptr;
    unsigned long long *d_blocks = nullptr, *d_components = nullptr;
    unsigned long long *d_loads = nullptr, *d_stores = nullptr, *d_adds = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)), "boundary alloc values");
    ck(cudaMalloc(&d_blocks, sizeof(unsigned long long)), "boundary alloc blocks");
    ck(cudaMalloc(&d_components, sizeof(unsigned long long)), "boundary alloc components");
    ck(cudaMalloc(&d_loads, sizeof(unsigned long long)), "boundary alloc loads");
    ck(cudaMalloc(&d_stores, sizeof(unsigned long long)), "boundary alloc stores");
    ck(cudaMalloc(&d_adds, sizeof(unsigned long long)), "boundary alloc adds");
    ck(cudaMalloc(&d_error, sizeof(int)), "boundary alloc error");
    ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice), "boundary copy values");
    ck(cudaMemset(d_blocks, 0, sizeof(unsigned long long)), "boundary zero blocks");
    ck(cudaMemset(d_components, 0, sizeof(unsigned long long)), "boundary zero components");
    ck(cudaMemset(d_loads, 0, sizeof(unsigned long long)), "boundary zero loads");
    ck(cudaMemset(d_stores, 0, sizeof(unsigned long long)), "boundary zero stores");
    ck(cudaMemset(d_adds, 0, sizeof(unsigned long long)), "boundary zero adds");
    ck(cudaMemset(d_error, 0, sizeof(int)), "boundary zero error");

    Rank fit_states = 0, fit_blocks = 0, fit_components = 0;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank n = oneesan::twocell::fusion_block_size(BOUNDARY_STEPS, o, rt);
        if (static_shared + n * sizeof(std::uint32_t) > total_limit) continue;
        const Rank supports = rt.choose[outer_bits][o];
        const Rank nc = oneesan::twocell::fusion_component_count(BOUNDARY_STEPS, o, rt);
        fit_states += supports * n;
        fit_blocks += supports;
        fit_components += 2 * supports * nc;
        const Rank dynamic_bytes = n * sizeof(std::uint32_t);
        ck(cudaFuncSetAttribute(
               two_cell_boundary_fusion_kernel,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               static_cast<int>(dynamic_bytes)),
           "boundary optin shared");
        const unsigned grid = static_cast<unsigned>(std::min<Rank>(supports, 65535));
        two_cell_boundary_fusion_kernel<<<grid, FUSION_THREADS, dynamic_bytes>>>(
            d_values, W, o, supports, states, mod,
            d_blocks, d_components, d_loads, d_stores, d_adds, d_error);
        ck(cudaGetLastError(), "boundary launch");
    }
    ck(cudaDeviceSynchronize(), "boundary sync");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    unsigned long long blocks = 0, components = 0, loads = 0, stores = 0, adds = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t),
                  cudaMemcpyDeviceToHost), "boundary copy output");
    ck(cudaMemcpy(&blocks, d_blocks, sizeof(blocks), cudaMemcpyDeviceToHost),
       "boundary copy blocks");
    ck(cudaMemcpy(&components, d_components, sizeof(components), cudaMemcpyDeviceToHost),
       "boundary copy components");
    ck(cudaMemcpy(&loads, d_loads, sizeof(loads), cudaMemcpyDeviceToHost),
       "boundary copy loads");
    ck(cudaMemcpy(&stores, d_stores, sizeof(stores), cudaMemcpyDeviceToHost),
       "boundary copy stores");
    ck(cudaMemcpy(&adds, d_adds, sizeof(adds), cudaMemcpyDeviceToHost),
       "boundary copy adds");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "boundary copy error");

    if (error || blocks != fit_blocks || components != fit_components ||
        loads != fit_states || stores != fit_states) {
        std::cerr << "FAIL boundary counters error=" << error
                  << " blocks=" << blocks << "/" << fit_blocks
                  << " components=" << components << "/" << fit_components
                  << " loads=" << loads << "/" << fit_states
                  << " stores=" << stores << "/" << fit_states << '\n';
        return 5;
    }
    if (fit_states == states && output != reference) {
        std::cerr << "FAIL boundary arithmetic W=" << W << '\n';
        return 6;
    }

    std::cout << "two-cell-boundary-fusion-singlebuffer"
              << " W=" << W
              << " states=" << states
              << " fused_states=" << fit_states
              << " global_loads=" << loads
              << " global_stores=" << stores
              << " local_adds=" << adds
              << " matching=leaf_peeling"
              << " second_global_vector_bytes=0"
              << " arithmetic=" << (fit_states == states ? "OK" : "PARTIAL")
              << "\n";

    cudaFree(d_values);
    cudaFree(d_blocks);
    cudaFree(d_components);
    cudaFree(d_loads);
    cudaFree(d_stores);
    cudaFree(d_adds);
    cudaFree(d_error);
    return 0;
}
