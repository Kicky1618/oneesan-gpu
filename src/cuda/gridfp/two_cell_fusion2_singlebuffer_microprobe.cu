#pragma push_macro("main")
#undef main
#define main two_cell_fusion2_bucketed_microprobe_main_unused
#include "two_cell_fusion2_bucketed_microprobe.cu"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_component.cuh"
#include "../../common/two_cell_component_matching.cuh"

namespace {

constexpr int FUSION_WARPS = 4;
constexpr int FUSION_THREADS = 32 * FUSION_WARPS;
constexpr int FUSION_MAX_COMPONENT = 18;

__device__ __forceinline__ std::uint32_t fusion_add_mod(
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t mod
) {
    const unsigned long long z =
        static_cast<unsigned long long>(a) + static_cast<unsigned long long>(b);
    return static_cast<std::uint32_t>(z >= mod ? z - mod : z);
}

__global__ void two_cell_fusion2_singlebuffer_kernel(
    std::uint32_t* __restrict__ values,
    int W,
    int start,
    int outer_ones,
    Rank support_count,
    Rank state_count,
    std::uint32_t mod,
    unsigned long long* processed_blocks,
    unsigned long long* processed_components,
    unsigned long long* global_loads,
    unsigned long long* global_stores,
    unsigned long long* residual_adds,
    int* error
) {
    extern __shared__ std::uint32_t block_values[];
    __shared__ oneesan::twocell::PackedKey sh_src[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ std::uint32_t sh_value[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ std::uint32_t sh_output[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ Rank sh_rank[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ oneesan::twocell::ComponentMatching sh_matching[FUSION_WARPS];
    __shared__ int sh_ns[FUSION_WARPS];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (warp >= FUSION_WARPS) return;

    const int outer_bits = W - FUSION_STEPS - 3;
    const Rank n = oneesan::twocell::fusion_block_size(
        FUSION_STEPS, outer_ones, TC_RANK_TABLES);
    const Rank nc = oneesan::twocell::fusion_component_count(
        FUSION_STEPS, outer_ones, TC_RANK_TABLES);

    for (Rank support_rank = blockIdx.x; support_rank < support_count;
         support_rank += gridDim.x) {
        const std::uint32_t outer = oneesan::twocell::support_unrank(
            outer_bits, outer_ones, support_rank, TC_RANK_TABLES);

        // Load the union block exactly once from the stationary global vector.
        for (Rank r = threadIdx.x; r < n; r += blockDim.x) {
            const auto d = oneesan::twocell::fusion_local_unrank_at(
                r, outer, W, start, FUSION_STEPS, start, TC_RANK_TABLES);
            if (!d.valid) {
                set_error(error, 251);
                continue;
            }
            const Rank gr = oneesan::twocell::stationary_rank_with_primitive(
                d.key, W, start, d.primitive,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
            if (gr >= state_count) {
                set_error(error, 252);
                continue;
            }
            block_values[r] = values[gr];
            atomicAdd(global_loads, 1ULL);
        }
        __syncthreads();

        for (int phase = 0; phase < FUSION_STEPS; ++phase) {
            const int active = start + phase;

            // K_active components partition this union block.  The stationary
            // destination coordinate indexed by t is recouple(src[t]), but the
            // matrix perfect matching is recovered separately by leaf peeling.
            for (Rank cr = warp; cr < nc; cr += FUSION_WARPS) {
                if (lane == 0) {
                    sh_ns[warp] = 0;
                    const auto label = oneesan::twocell::fusion_component_unrank(
                        cr, outer, W, start, FUSION_STEPS, TC_RANK_TABLES);
                    if (!label.valid) {
                        set_error(error, 253);
                    } else {
                        const auto src = oneesan::twocell::direct_component_sources(
                            label.word, W, active);
                        if (src.overflow || src.size <= 0 ||
                            src.size > FUSION_MAX_COMPONENT) {
                            set_error(error, 254);
                        } else {
                            sh_ns[warp] = src.size;
                            for (int s = 0; s < src.size; ++s)
                                sh_src[warp][s] = src.value[s];
                            const auto m = oneesan::twocell::build_component_matching(
                                sh_src[warp], src.size, W, active);
                            if (!m.ok) {
                                set_error(error, 256);
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
                    const auto source = sh_src[warp][lane];
                    const int len = source.type ? W - 2 : W - 1;
                    const Rank primitive = oneesan::twocell::primitive_rank(
                        source.support, source.left, len, TC_RANK_TABLES);
                    const Rank lr = oneesan::twocell::fusion_local_rank_at_with_primitive(
                        source, start, FUSION_STEPS, active, outer_ones,
                        primitive, TC_RANK_TABLES);
                    sh_rank[warp][lane] = lr;
                    if (lr >= n) {
                        sh_value[warp][lane] = 0;
                        set_error(error, 255);
                    } else {
                        sh_value[warp][lane] = block_values[lr];
                    }
                }
                __syncwarp();

                if (lane == 0 && ns > 0) {
                    const auto m = sh_matching[warp];
                    if (!oneesan::twocell::apply_component_matching(
                            m, sh_value[warp], sh_output[warp], mod)) {
                        set_error(error, 257);
                    } else {
                        atomicAdd(residual_adds,
                                  static_cast<unsigned long long>(m.residual_edges));
                    }
                }
                __syncwarp();

                // Local stationary rank of src[t] equals the destination local
                // rank of recouple(src[t]) at active+1, so index t can be stored
                // back to the same union-block slot after the matching multiply.
                if (lane < ns) {
                    const Rank lr = sh_rank[warp][lane];
                    if (lr < n) block_values[lr] = sh_output[warp][lane];
                }
                __syncwarp();
            }
            __syncthreads();
        }

        // Store the final stationary union block exactly once.
        for (Rank r = threadIdx.x; r < n; r += blockDim.x) {
            const auto d = oneesan::twocell::fusion_local_unrank_at(
                r, outer, W, start, FUSION_STEPS, start + FUSION_STEPS,
                TC_RANK_TABLES);
            if (!d.valid) {
                set_error(error, 261);
                continue;
            }
            const Rank gr = oneesan::twocell::stationary_rank_with_primitive(
                d.key, W, start + FUSION_STEPS, d.primitive,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
            if (gr >= state_count) {
                set_error(error, 262);
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

constexpr Rank singlebuffer_workspace_estimate() {
    // Includes one component input/output scratch pair and a local matching.
    // Actual launch planning uses cudaFuncGetAttributes() static shared size.
    return 4096ULL;
}

Fusion2BucketPlan make_singlebuffer_plan(
    int W,
    Rank shared_limit_bytes,
    Rank static_reserve,
    const RankTables& rt
) {
    Fusion2BucketPlan p{};
    const int outer_bits = W - 5;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank blocks = rt.choose[outer_bits][o];
        const Rank n = oneesan::twocell::fusion_block_size(2, o, rt);
        const bool fit = static_reserve + n * sizeof(std::uint32_t) <=
                         shared_limit_bytes;
        if (fit) {
            p.fitted_blocks += blocks;
            p.fitted_states += blocks * n;
            p.max_fused_outer_ones = o;
        } else {
            p.fallback_blocks += blocks;
            p.fallback_states += blocks * n;
        }
    }
    return p;
}

void print_singlebuffer_plan(
    int W,
    Rank shared_limit,
    const RankTables& rt
) {
    const Rank reserve = singlebuffer_workspace_estimate();
    const auto p = make_singlebuffer_plan(W, shared_limit, reserve, rt);
    const Rank states = p.fitted_states + p.fallback_states;
    const double f = states ? double(p.fitted_states) / double(states) : 0.0;
    std::cout << "fusion2_singlebuffer_plan"
              << " W=" << W
              << " shared_limit_bytes=" << shared_limit
              << " workspace_estimate=" << reserve
              << " max_fused_outer_ones=" << p.max_fused_outer_ones
              << " fused_state_fraction=" << f
              << " ideal_HBM_reduction_vs_two_pass=" << 0.5 * f
              << " matching=leaf_peeling"
              << " second_shared_block_buffer_bytes=0\n";
}

void run_singlebuffer_fused_only(
    int W,
    int start,
    Rank requested_shared_limit,
    const RankTables& rt,
    const StationaryRankTables& st,
    std::uint32_t mod
) {
    const Rank states = st.total[W];
    const int outer_bits = W - 5;

    cudaFuncAttributes attr{};
    ck(cudaFuncGetAttributes(&attr, two_cell_fusion2_singlebuffer_kernel),
       "singlebuffer func attributes");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "singlebuffer device props");
    const Rank hardware_limit = static_cast<Rank>(prop.sharedMemPerBlockOptin);
    const Rank total_limit = std::min(requested_shared_limit, hardware_limit);
    const Rank static_shared = static_cast<Rank>(attr.sharedSizeBytes);
    const auto plan = make_singlebuffer_plan(W, total_limit, static_shared, rt);

    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + Rank(start) * 149ULL) % (mod - 1ULL)));
    const auto reference = fusion2_reference(input, W, start, rt, st, mod);

    std::uint32_t* d_values = nullptr;
    unsigned long long* d_blocks = nullptr;
    unsigned long long* d_components = nullptr;
    unsigned long long* d_loads = nullptr;
    unsigned long long* d_stores = nullptr;
    unsigned long long* d_residual = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)), "singlebuffer alloc values");
    ck(cudaMalloc(&d_blocks, sizeof(unsigned long long)), "singlebuffer alloc blocks");
    ck(cudaMalloc(&d_components, sizeof(unsigned long long)), "singlebuffer alloc components");
    ck(cudaMalloc(&d_loads, sizeof(unsigned long long)), "singlebuffer alloc loads");
    ck(cudaMalloc(&d_stores, sizeof(unsigned long long)), "singlebuffer alloc stores");
    ck(cudaMalloc(&d_residual, sizeof(unsigned long long)), "singlebuffer alloc residual");
    ck(cudaMalloc(&d_error, sizeof(int)), "singlebuffer alloc error");
    ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice), "singlebuffer copy values");
    ck(cudaMemset(d_blocks, 0, sizeof(unsigned long long)), "singlebuffer zero blocks");
    ck(cudaMemset(d_components, 0, sizeof(unsigned long long)), "singlebuffer zero components");
    ck(cudaMemset(d_loads, 0, sizeof(unsigned long long)), "singlebuffer zero loads");
    ck(cudaMemset(d_stores, 0, sizeof(unsigned long long)), "singlebuffer zero stores");
    ck(cudaMemset(d_residual, 0, sizeof(unsigned long long)), "singlebuffer zero residual");
    ck(cudaMemset(d_error, 0, sizeof(int)), "singlebuffer zero error");

    Rank expected_components = 0;
    Rank expected_residual = 0;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank n = oneesan::twocell::fusion_block_size(2, o, rt);
        if (static_shared + n * sizeof(std::uint32_t) > total_limit) continue;
        const Rank support_count = rt.choose[outer_bits][o];
        const Rank nc = oneesan::twocell::fusion_component_count(2, o, rt);
        expected_components += 2 * support_count * nc;
        // Each phase partitions every fitted state into nc components and each
        // component contributes size-1 residual additions.
        expected_residual += 2 * support_count * (n - nc);
        const Rank dynamic_bytes = n * sizeof(std::uint32_t);
        ck(cudaFuncSetAttribute(
               two_cell_fusion2_singlebuffer_kernel,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               static_cast<int>(dynamic_bytes)),
           "singlebuffer optin shared");
        const unsigned grid = static_cast<unsigned>(std::min<Rank>(support_count, 65535));
        two_cell_fusion2_singlebuffer_kernel<<<grid, FUSION_THREADS, dynamic_bytes>>>(
            d_values, W, start, o, support_count, states, mod,
            d_blocks, d_components, d_loads, d_stores, d_residual, d_error);
        ck(cudaGetLastError(), "singlebuffer launch");
    }
    ck(cudaDeviceSynchronize(), "singlebuffer sync");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    unsigned long long processed = 0, components = 0, loads = 0, stores = 0, residual = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t),
                  cudaMemcpyDeviceToHost), "singlebuffer copy output");
    ck(cudaMemcpy(&processed, d_blocks, sizeof(processed), cudaMemcpyDeviceToHost),
       "singlebuffer copy blocks");
    ck(cudaMemcpy(&components, d_components, sizeof(components), cudaMemcpyDeviceToHost),
       "singlebuffer copy components");
    ck(cudaMemcpy(&loads, d_loads, sizeof(loads), cudaMemcpyDeviceToHost),
       "singlebuffer copy loads");
    ck(cudaMemcpy(&stores, d_stores, sizeof(stores), cudaMemcpyDeviceToHost),
       "singlebuffer copy stores");
    ck(cudaMemcpy(&residual, d_residual, sizeof(residual), cudaMemcpyDeviceToHost),
       "singlebuffer copy residual");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "singlebuffer copy error");

    if (error || processed != plan.fitted_blocks ||
        components != expected_components || residual != expected_residual ||
        loads != plan.fitted_states || stores != plan.fitted_states) {
        std::cerr << "FAIL singlebuffer counters W=" << W << " start=" << start
                  << " error=" << error
                  << " blocks=" << processed << "/" << plan.fitted_blocks
                  << " components=" << components << "/" << expected_components
                  << " residual=" << residual << "/" << expected_residual
                  << " loads=" << loads << "/" << plan.fitted_states
                  << " stores=" << stores << "/" << plan.fitted_states << '\n';
        std::exit(263);
    }

    if (plan.fallback_states == 0 && output != reference) {
        std::cerr << "FAIL singlebuffer arithmetic W=" << W
                  << " start=" << start << '\n';
        std::exit(264);
    }

    std::cout << "two-cell-fusion2-singlebuffer"
              << " W=" << W
              << " start=" << start
              << " static_shared=" << static_shared
              << " total_shared_limit=" << total_limit
              << " fused_states=" << plan.fitted_states
              << " fallback_states=" << plan.fallback_states
              << " global_loads=" << loads
              << " global_stores=" << stores
              << " residual_adds=" << residual
              << " matching=leaf_peeling"
              << " second_shared_block_buffer_bytes=0"
              << " arithmetic=" << (plan.fallback_states ? "PARTIAL" : "OK")
              << "\n";

    cudaFree(d_values);
    cudaFree(d_blocks);
    cudaFree(d_components);
    cudaFree(d_loads);
    cudaFree(d_stores);
    cudaFree(d_residual);
    cudaFree(d_error);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 8;
    const Rank requested_kib = argc > 2
        ? static_cast<Rank>(std::strtoull(argv[2], nullptr, 10)) : 228ULL;
    const std::uint32_t mod = argc > 3
        ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10))
        : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth || mod < 3) return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    const Rank requested_bytes = requested_kib * 1024ULL;
    if (plan_only) {
        print_singlebuffer_plan(W, requested_bytes, rt);
        if (W == 28) {
            for (Rank kib : {64ULL, 96ULL, 128ULL, 160ULL, 192ULL, 228ULL, 256ULL})
                print_singlebuffer_plan(W, kib * 1024ULL, rt);
        }
        return 0;
    }

    if (W > 10) {
        std::cerr << "execution mode intentionally limited to W<=10; use --plan-only above that\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "singlebuffer device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "singlebuffer set device");
    install_tables(rt);
    install_stationary_tables(st);

    for (int start = 0; start + 2 <= W - 3; ++start)
        run_singlebuffer_fused_only(W, start, requested_bytes, rt, st, mod);
    std::cout << "ALL_OK two_cell_fusion2_singlebuffer=1 W=" << W << '\n';
    return 0;
}
