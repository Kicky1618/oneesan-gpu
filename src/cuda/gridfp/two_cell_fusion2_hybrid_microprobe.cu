#pragma push_macro("main")
#undef main
#define main two_cell_fusion2_singlebuffer_microprobe_main_unused
#include "two_cell_fusion2_singlebuffer_microprobe.cu"
#pragma pop_macro("main")

namespace {

// Fallback for a union block that does not fit in the large shared buffer. It
// still uses the stationary one-vector layout. Each component captures all of
// its source values locally, recovers the unique matching, applies matching
// permutation + residual n-1 adds, then writes the same stationary addresses.
__global__ void two_cell_fusion2_global_fallback_kernel(
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
    const Rank nc = oneesan::twocell::fusion_component_count(
        FUSION_STEPS, outer_ones, TC_RANK_TABLES);

    for (Rank support_rank = blockIdx.x; support_rank < support_count;
         support_rank += gridDim.x) {
        const std::uint32_t outer = oneesan::twocell::support_unrank(
            outer_bits, outer_ones, support_rank, TC_RANK_TABLES);

        for (int phase = 0; phase < FUSION_STEPS; ++phase) {
            const int active = start + phase;
            for (Rank cr = warp; cr < nc; cr += FUSION_WARPS) {
                if (lane == 0) {
                    sh_ns[warp] = 0;
                    const auto label = oneesan::twocell::fusion_component_unrank(
                        cr, outer, W, start, FUSION_STEPS, TC_RANK_TABLES);
                    if (!label.valid) {
                        set_error(error, 271);
                    } else {
                        const auto src = oneesan::twocell::direct_component_sources(
                            label.word, W, active);
                        if (src.overflow || src.size <= 0 ||
                            src.size > FUSION_MAX_COMPONENT) {
                            set_error(error, 272);
                        } else {
                            sh_ns[warp] = src.size;
                            for (int s = 0; s < src.size; ++s)
                                sh_src[warp][s] = src.value[s];
                            const auto m = oneesan::twocell::build_component_matching(
                                sh_src[warp], src.size, W, active);
                            if (!m.ok) {
                                set_error(error, 274);
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
                    const Rank gr = oneesan::twocell::stationary_rank_with_primitive(
                        source, W, active, primitive,
                        TC_RANK_TABLES, TC_STATIONARY_TABLES);
                    sh_rank[warp][lane] = gr;
                    if (gr >= state_count) {
                        sh_value[warp][lane] = 0;
                        set_error(error, 273);
                    } else {
                        sh_value[warp][lane] = values[gr];
                        atomicAdd(global_loads, 1ULL);
                    }
                }
                __syncwarp();

                if (lane == 0 && ns > 0) {
                    const auto m = sh_matching[warp];
                    if (!oneesan::twocell::apply_component_matching(
                            m, sh_value[warp], sh_output[warp], mod)) {
                        set_error(error, 275);
                    } else {
                        atomicAdd(residual_adds,
                                  static_cast<unsigned long long>(m.residual_edges));
                    }
                }
                __syncwarp();

                if (lane < ns) {
                    const Rank gr = sh_rank[warp][lane];
                    if (gr < state_count) {
                        values[gr] = sh_output[warp][lane];
                        atomicAdd(global_stores, 1ULL);
                    }
                }
                __syncwarp();
            }
            // Every component in this union block has completed K_active before
            // the same CTA starts K_{active+1}; no grid-wide barrier is needed.
            __syncthreads();
        }
        if (threadIdx.x == 0) atomicAdd(processed_blocks, 1ULL);
        __syncthreads();
    }
}

struct HybridTrafficPlan {
    Fusion2BucketPlan split{};
    Rank value_loads = 0;
    Rank value_stores = 0;
    Rank baseline_loads = 0;
    Rank baseline_stores = 0;
};

HybridTrafficPlan make_hybrid_plan(
    int W,
    Rank shared_limit,
    Rank static_shared,
    const RankTables& rt
) {
    HybridTrafficPlan h{};
    h.split = make_singlebuffer_plan(W, shared_limit, static_shared, rt);
    const Rank states = h.split.fitted_states + h.split.fallback_states;
    h.value_loads = h.split.fitted_states + 2 * h.split.fallback_states;
    h.value_stores = h.value_loads;
    h.baseline_loads = 2 * states;
    h.baseline_stores = 2 * states;
    return h;
}

void print_hybrid_plan(
    int W,
    Rank shared_limit,
    const RankTables& rt
) {
    const Rank reserve = singlebuffer_workspace_estimate();
    const auto h = make_hybrid_plan(W, shared_limit, reserve, rt);
    const Rank traffic = h.value_loads + h.value_stores;
    const Rank baseline = h.baseline_loads + h.baseline_stores;
    const double fused_fraction =
        double(h.split.fitted_states) /
        double(h.split.fitted_states + h.split.fallback_states);
    std::cout << "fusion2_hybrid_plan"
              << " W=" << W
              << " shared_limit_bytes=" << shared_limit
              << " workspace_estimate=" << reserve
              << " max_fused_outer_ones=" << h.split.max_fused_outer_ones
              << " fused_state_fraction=" << fused_fraction
              << " value_loads=" << h.value_loads
              << " value_stores=" << h.value_stores
              << " baseline_value_transfers=" << baseline
              << " hybrid_value_transfers=" << traffic
              << " HBM_reduction=" << (1.0 - double(traffic) / double(baseline))
              << " matching=leaf_peeling second_global_vector_bytes=0\n";
}

void run_hybrid(
    int W,
    int start,
    Rank requested_shared_limit,
    const RankTables& rt,
    const StationaryRankTables& st,
    std::uint32_t mod
) {
    const Rank states = st.total[W];
    const int outer_bits = W - 5;

    cudaFuncAttributes fit_attr{}, fallback_attr{};
    ck(cudaFuncGetAttributes(&fit_attr, two_cell_fusion2_singlebuffer_kernel),
       "hybrid fused attributes");
    ck(cudaFuncGetAttributes(&fallback_attr, two_cell_fusion2_global_fallback_kernel),
       "hybrid fallback attributes");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "hybrid device props");
    const Rank total_limit = std::min<Rank>(
        requested_shared_limit, static_cast<Rank>(prop.sharedMemPerBlockOptin));
    const Rank static_shared = static_cast<Rank>(fit_attr.sharedSizeBytes);
    const auto plan = make_hybrid_plan(W, total_limit, static_shared, rt);

    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + Rank(start) * 163ULL) % (mod - 1ULL)));
    const auto reference = fusion2_reference(input, W, start, rt, st, mod);

    std::uint32_t* d_values = nullptr;
    unsigned long long* d_blocks = nullptr;
    unsigned long long* d_components = nullptr;
    unsigned long long* d_loads = nullptr;
    unsigned long long* d_stores = nullptr;
    unsigned long long* d_residual = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)), "hybrid alloc values");
    ck(cudaMalloc(&d_blocks, sizeof(unsigned long long)), "hybrid alloc blocks");
    ck(cudaMalloc(&d_components, sizeof(unsigned long long)), "hybrid alloc components");
    ck(cudaMalloc(&d_loads, sizeof(unsigned long long)), "hybrid alloc loads");
    ck(cudaMalloc(&d_stores, sizeof(unsigned long long)), "hybrid alloc stores");
    ck(cudaMalloc(&d_residual, sizeof(unsigned long long)), "hybrid alloc residual");
    ck(cudaMalloc(&d_error, sizeof(int)), "hybrid alloc error");
    ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice), "hybrid copy values");
    ck(cudaMemset(d_blocks, 0, sizeof(unsigned long long)), "hybrid zero blocks");
    ck(cudaMemset(d_components, 0, sizeof(unsigned long long)), "hybrid zero components");
    ck(cudaMemset(d_loads, 0, sizeof(unsigned long long)), "hybrid zero loads");
    ck(cudaMemset(d_stores, 0, sizeof(unsigned long long)), "hybrid zero stores");
    ck(cudaMemset(d_residual, 0, sizeof(unsigned long long)), "hybrid zero residual");
    ck(cudaMemset(d_error, 0, sizeof(int)), "hybrid zero error");

    Rank expected_components = 0;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank n = oneesan::twocell::fusion_block_size(2, o, rt);
        const Rank support_count = rt.choose[outer_bits][o];
        const Rank nc = oneesan::twocell::fusion_component_count(2, o, rt);
        expected_components += 2 * support_count * nc;
        const unsigned grid = static_cast<unsigned>(std::min<Rank>(support_count, 65535));
        if (static_shared + n * sizeof(std::uint32_t) <= total_limit) {
            const Rank dynamic_bytes = n * sizeof(std::uint32_t);
            ck(cudaFuncSetAttribute(
                   two_cell_fusion2_singlebuffer_kernel,
                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                   static_cast<int>(dynamic_bytes)),
               "hybrid fused optin shared");
            two_cell_fusion2_singlebuffer_kernel<<<grid, FUSION_THREADS, dynamic_bytes>>>(
                d_values, W, start, o, support_count, states, mod,
                d_blocks, d_components, d_loads, d_stores, d_residual, d_error);
            ck(cudaGetLastError(), "hybrid fused launch");
        } else {
            two_cell_fusion2_global_fallback_kernel<<<grid, FUSION_THREADS>>>(
                d_values, W, start, o, support_count, states, mod,
                d_blocks, d_components, d_loads, d_stores, d_residual, d_error);
            ck(cudaGetLastError(), "hybrid fallback launch");
        }
    }
    ck(cudaDeviceSynchronize(), "hybrid sync");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    unsigned long long processed = 0, components = 0, loads = 0, stores = 0, residual = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t),
                  cudaMemcpyDeviceToHost), "hybrid copy output");
    ck(cudaMemcpy(&processed, d_blocks, sizeof(processed), cudaMemcpyDeviceToHost),
       "hybrid copy blocks");
    ck(cudaMemcpy(&components, d_components, sizeof(components), cudaMemcpyDeviceToHost),
       "hybrid copy components");
    ck(cudaMemcpy(&loads, d_loads, sizeof(loads), cudaMemcpyDeviceToHost),
       "hybrid copy loads");
    ck(cudaMemcpy(&stores, d_stores, sizeof(stores), cudaMemcpyDeviceToHost),
       "hybrid copy stores");
    ck(cudaMemcpy(&residual, d_residual, sizeof(residual), cudaMemcpyDeviceToHost),
       "hybrid copy residual");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "hybrid copy error");

    const Rank expected_blocks = Rank(1) << outer_bits;
    const Rank expected_residual =
        2 * (states - oneesan::twocell::component_label_count(W, rt));
    if (error || processed != expected_blocks || components != expected_components ||
        loads != plan.value_loads || stores != plan.value_stores ||
        residual != expected_residual || output != reference) {
        std::cerr << "FAIL hybrid W=" << W << " start=" << start
                  << " error=" << error
                  << " blocks=" << processed << "/" << expected_blocks
                  << " components=" << components << "/" << expected_components
                  << " loads=" << loads << "/" << plan.value_loads
                  << " stores=" << stores << "/" << plan.value_stores
                  << " residual=" << residual << "/" << expected_residual << '\n';
        std::exit(279);
    }

    std::cout << "two-cell-fusion2-hybrid"
              << " W=" << W
              << " start=" << start
              << " static_fused_shared=" << static_shared
              << " static_fallback_shared=" << fallback_attr.sharedSizeBytes
              << " total_shared_limit=" << total_limit
              << " fused_states=" << plan.split.fitted_states
              << " fallback_states=" << plan.split.fallback_states
              << " global_loads=" << loads
              << " global_stores=" << stores
              << " residual_adds=" << residual
              << " matching=leaf_peeling"
              << " second_global_vector_bytes=0 arithmetic=OK\n";

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
        print_hybrid_plan(W, requested_bytes, rt);
        if (W == 28) {
            for (Rank kib : {64ULL, 96ULL, 128ULL, 160ULL, 192ULL, 228ULL, 256ULL})
                print_hybrid_plan(W, kib * 1024ULL, rt);
        }
        return 0;
    }

    if (W > 10) {
        std::cerr << "execution mode intentionally limited to W<=10; use --plan-only above that\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "hybrid device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "hybrid set device");
    install_tables(rt);
    install_stationary_tables(st);

    for (int start = 0; start + 2 <= W - 3; ++start)
        run_hybrid(W, start, requested_bytes, rt, st, mod);
    std::cout << "ALL_OK two_cell_fusion2_hybrid=1 W=" << W << '\n';
    return 0;
}
