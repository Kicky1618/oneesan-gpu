#pragma push_macro("main")
#undef main
#define main two_cell_fusion2_compact_face_microprobe_main_unused
#include "two_cell_fusion2_compact_face_microprobe.cu"
#pragma pop_macro("main")

#include "two_cell_component_warp_arithmetic.cuh"

namespace {

__global__ void two_cell_fusion2_register_face_kernel(
    std::uint32_t* __restrict__ values,
    int W,
    int start,
    int outer_ones,
    Rank support_count,
    Rank state_count,
    std::uint32_t mod,
    unsigned long long* processed_blocks,
    unsigned long long* processed_components,
    unsigned long long* deep_components,
    unsigned long long* global_loads,
    unsigned long long* global_stores,
    unsigned long long* residual_adds,
    int* error
) {
    extern __shared__ std::uint32_t block_values[];
    __shared__ PackedKey sh_src[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ PackedWord sh_label[FUSION_WARPS];
    __shared__ int sh_ns[FUSION_WARPS];
    __shared__ int sh_deep[FUSION_WARPS];
    __shared__ int sh_partner_rounds[FUSION_WARPS];

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

        for (Rank r = threadIdx.x; r < n; r += blockDim.x) {
            const auto d = oneesan::twocell::fusion_local_unrank_at(
                r, outer, W, start, FUSION_STEPS, start, TC_RANK_TABLES);
            if (!d.valid) {
                set_error(error, 421);
                continue;
            }
            const Rank gr = oneesan::twocell::stationary_rank_with_primitive(
                d.key, W, start, d.primitive,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
            if (gr >= state_count) {
                set_error(error, 422);
                continue;
            }
            block_values[r] = values[gr];
            atomicAdd(global_loads, 1ULL);
        }
        __syncthreads();

        for (int phase = 0; phase < FUSION_STEPS; ++phase) {
            const int active = start + phase;
            for (Rank cr = warp; cr < nc; cr += FUSION_WARPS) {
                if (lane == 0) {
                    sh_ns[warp] = 0;
                    sh_partner_rounds[warp] = 0;
                    const auto label = oneesan::twocell::fusion_component_unrank(
                        cr, outer, W, start, FUSION_STEPS, TC_RANK_TABLES);
                    if (!label.valid) {
                        sh_deep[warp] = 0;
                        set_error(error, 423);
                    } else {
                        sh_label[warp] = label.word;
                        PackedWord collapsed{};
                        sh_deep[warp] = oneesan::twocell::deep_collapse(
                            label.word, active, collapsed) ? 1 : 0;
                    }
                }
                __syncwarp();

                if (sh_deep[warp]) {
                    oneesan::twocell::cuda_face::deep_component_sources_compact(
                        sh_label[warp], W, active,
                        sh_src[warp], &sh_ns[warp],
                        &sh_partner_rounds[warp], error);
                    __syncwarp();
                    if (lane == 0) atomicAdd(deep_components, 1ULL);
                } else if (lane == 0) {
                    const auto src = oneesan::twocell::direct_component_sources(
                        sh_label[warp], W, active);
                    if (src.overflow || src.size <= 0 ||
                        src.size > FUSION_MAX_COMPONENT) {
                        set_error(error, 424);
                    } else {
                        sh_ns[warp] = src.size;
                        for (int s = 0; s < src.size; ++s)
                            sh_src[warp][s] = src.value[s];
                    }
                }
                __syncwarp();

                const int ns = sh_ns[warp];
                if (ns <= 0 || ns > FUSION_MAX_COMPONENT) {
                    if (lane == 0) set_error(error, 425);
                    __syncwarp();
                    continue;
                }
                if (lane == 0) {
                    atomicAdd(processed_components, 1ULL);
                    atomicAdd(residual_adds,
                              static_cast<unsigned long long>(ns - 1));
                }

                Rank local_rank = ~Rank(0);
                std::uint32_t x = 0;
                PackedKey source{};
                if (lane < ns) {
                    source = sh_src[warp][lane];
                    const int len = source.type ? W - 2 : W - 1;
                    const Rank primitive = oneesan::twocell::primitive_rank(
                        source.support, source.left, len, TC_RANK_TABLES);
                    local_rank = oneesan::twocell::fusion_local_rank_at_with_primitive(
                        source, start, FUSION_STEPS, active, outer_ones,
                        primitive, TC_RANK_TABLES);
                    if (local_rank >= n) {
                        set_error(error, 426);
                    } else {
                        x = block_values[local_rank];
                    }
                }
                __syncwarp();

                const std::uint32_t y =
                    oneesan::twocell::cuda_component::apply_closed_component_warp(
                        sh_label[warp], source, ns, W, active, x, mod, error);
                __syncwarp();

                if (lane < ns && local_rank < n)
                    block_values[local_rank] = y;
                __syncwarp();
            }
            __syncthreads();
        }

        for (Rank r = threadIdx.x; r < n; r += blockDim.x) {
            const auto d = oneesan::twocell::fusion_local_unrank_at(
                r, outer, W, start, FUSION_STEPS, start + FUSION_STEPS,
                TC_RANK_TABLES);
            if (!d.valid) {
                set_error(error, 427);
                continue;
            }
            const Rank gr = oneesan::twocell::stationary_rank_with_primitive(
                d.key, W, start + FUSION_STEPS, d.primitive,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
            if (gr >= state_count) {
                set_error(error, 428);
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

Rank fusion2_register_workspace_estimate() {
    return Rank(FUSION_WARPS) * (
        Rank(FUSION_MAX_COMPONENT) * sizeof(PackedKey) +
        sizeof(PackedWord) + 3 * sizeof(int)) + 256ULL;
}

void print_fusion2_register_plan(
    int W,
    Rank shared_limit,
    const RankTables& rt
) {
    const Rank reserve = fusion2_register_workspace_estimate();
    const auto p = make_singlebuffer_plan(W, shared_limit, reserve, rt);
    const Rank total = p.fitted_states + p.fallback_states;
    const double f = total ? double(p.fitted_states) / double(total) : 0.0;
    std::cout << "fusion2_register_face_plan"
              << " W=" << W
              << " shared_limit_bytes=" << shared_limit
              << " workspace_estimate=" << reserve
              << " max_fused_outer_ones=" << p.max_fused_outer_ones
              << " fused_state_fraction=" << f
              << " ideal_HBM_reduction_vs_two_pass=" << 0.5 * f
              << " component_value_shared_bytes=0"
              << " component_output_shared_bytes=0"
              << " component_rank_shared_bytes=0"
              << " candidate_shared_scratch_bytes=0"
              << " matching_descriptor_bytes=0"
              << " partner_scan_rounds=0"
              << " matching_K_step_calls=0\n";
}

void run_fusion2_register_face(
    int W,
    int start,
    Rank requested_shared_limit,
    const RankTables& rt,
    const StationaryRankTables& st,
    std::uint32_t mod
) {
    const Rank states = st.total[W];
    const int outer_bits = W - 5;

    cudaFuncAttributes attr{}, compact_attr{};
    ck(cudaFuncGetAttributes(&attr, two_cell_fusion2_register_face_kernel),
       "fusion register attributes");
    ck(cudaFuncGetAttributes(&compact_attr, two_cell_fusion2_compact_face_kernel),
       "fusion compact attributes");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "fusion register props");
    const Rank total_limit = std::min<Rank>(
        requested_shared_limit, static_cast<Rank>(prop.sharedMemPerBlockOptin));
    const Rank static_shared = static_cast<Rank>(attr.sharedSizeBytes);
    const auto plan = make_singlebuffer_plan(W, total_limit, static_shared, rt);

    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + Rank(start) * 307ULL) % (mod - 1ULL)));
    const auto reference = fusion2_reference(input, W, start, rt, st, mod);

    std::uint32_t* d_values = nullptr;
    unsigned long long *d_blocks = nullptr, *d_components = nullptr, *d_deep = nullptr;
    unsigned long long *d_loads = nullptr, *d_stores = nullptr, *d_adds = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)), "fusion register alloc values");
    ck(cudaMalloc(&d_blocks, sizeof(unsigned long long)), "fusion register alloc blocks");
    ck(cudaMalloc(&d_components, sizeof(unsigned long long)), "fusion register alloc components");
    ck(cudaMalloc(&d_deep, sizeof(unsigned long long)), "fusion register alloc deep");
    ck(cudaMalloc(&d_loads, sizeof(unsigned long long)), "fusion register alloc loads");
    ck(cudaMalloc(&d_stores, sizeof(unsigned long long)), "fusion register alloc stores");
    ck(cudaMalloc(&d_adds, sizeof(unsigned long long)), "fusion register alloc adds");
    ck(cudaMalloc(&d_error, sizeof(int)), "fusion register alloc error");
    ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "fusion register copy values");
    ck(cudaMemset(d_blocks, 0, sizeof(unsigned long long)), "fusion register zero blocks");
    ck(cudaMemset(d_components, 0, sizeof(unsigned long long)), "fusion register zero components");
    ck(cudaMemset(d_deep, 0, sizeof(unsigned long long)), "fusion register zero deep");
    ck(cudaMemset(d_loads, 0, sizeof(unsigned long long)), "fusion register zero loads");
    ck(cudaMemset(d_stores, 0, sizeof(unsigned long long)), "fusion register zero stores");
    ck(cudaMemset(d_adds, 0, sizeof(unsigned long long)), "fusion register zero adds");
    ck(cudaMemset(d_error, 0, sizeof(int)), "fusion register zero error");

    Rank expected_components = 0;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank n = oneesan::twocell::fusion_block_size(2, o, rt);
        if (static_shared + n * sizeof(std::uint32_t) > total_limit) continue;
        const Rank support_count = rt.choose[outer_bits][o];
        const Rank nc = oneesan::twocell::fusion_component_count(2, o, rt);
        expected_components += 2 * support_count * nc;
        const Rank dynamic_bytes = n * sizeof(std::uint32_t);
        ck(cudaFuncSetAttribute(
               two_cell_fusion2_register_face_kernel,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               static_cast<int>(dynamic_bytes)),
           "fusion register optin shared");
        const unsigned grid = static_cast<unsigned>(std::min<Rank>(support_count, 65535));
        two_cell_fusion2_register_face_kernel<<<grid, FUSION_THREADS, dynamic_bytes>>>(
            d_values, W, start, o, support_count, states, mod,
            d_blocks, d_components, d_deep, d_loads, d_stores, d_adds, d_error);
        ck(cudaGetLastError(), "fusion register launch");
    }
    ck(cudaDeviceSynchronize(), "fusion register sync");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    unsigned long long blocks = 0, components = 0, deep = 0;
    unsigned long long loads = 0, stores = 0, adds = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
       "fusion register copy output");
    ck(cudaMemcpy(&blocks, d_blocks, sizeof(blocks), cudaMemcpyDeviceToHost),
       "fusion register copy blocks");
    ck(cudaMemcpy(&components, d_components, sizeof(components), cudaMemcpyDeviceToHost),
       "fusion register copy components");
    ck(cudaMemcpy(&deep, d_deep, sizeof(deep), cudaMemcpyDeviceToHost),
       "fusion register copy deep");
    ck(cudaMemcpy(&loads, d_loads, sizeof(loads), cudaMemcpyDeviceToHost),
       "fusion register copy loads");
    ck(cudaMemcpy(&stores, d_stores, sizeof(stores), cudaMemcpyDeviceToHost),
       "fusion register copy stores");
    ck(cudaMemcpy(&adds, d_adds, sizeof(adds), cudaMemcpyDeviceToHost),
       "fusion register copy adds");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "fusion register copy error");

    if (error || blocks != plan.fitted_blocks || components != expected_components ||
        loads != plan.fitted_states || stores != plan.fitted_states) {
        std::cerr << "FAIL fusion register counters W=" << W
                  << " start=" << start << " error=" << error << '\n';
        std::exit(429);
    }
    if (plan.fallback_states == 0 && output != reference) {
        std::cerr << "FAIL fusion register arithmetic W=" << W
                  << " start=" << start << '\n';
        std::exit(430);
    }

    std::cout << "two-cell-fusion2-register-face"
              << " W=" << W
              << " start=" << start
              << " static_shared_bytes=" << static_shared
              << " compact_static_shared_bytes=" << compact_attr.sharedSizeBytes
              << " total_shared_limit=" << total_limit
              << " fused_states=" << plan.fitted_states
              << " fallback_states=" << plan.fallback_states
              << " deep_components=" << deep
              << " global_loads=" << loads
              << " global_stores=" << stores
              << " residual_adds=" << adds
              << " component_value_shared_bytes=0"
              << " component_output_shared_bytes=0"
              << " component_rank_shared_bytes=0"
              << " warp_register_arithmetic=1"
              << " arithmetic=" << (plan.fallback_states == 0 ? "OK" : "PARTIAL")
              << '\n';

    cudaFree(d_values);
    cudaFree(d_blocks);
    cudaFree(d_components);
    cudaFree(d_deep);
    cudaFree(d_loads);
    cudaFree(d_stores);
    cudaFree(d_adds);
    cudaFree(d_error);
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
    if (plan_only) {
        print_fusion2_register_plan(W, shared_kib * 1024ULL, rt);
        if (W == 28) {
            for (Rank kib : {64ULL, 96ULL, 128ULL, 160ULL, 192ULL, 228ULL, 256ULL})
                print_fusion2_register_plan(W, kib * 1024ULL, rt);
        }
        return 0;
    }

    if (W > 10) {
        std::cerr << "execution mode intentionally limited to W<=10; use --plan-only above that\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "fusion register device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "fusion register set device");
    install_tables(rt);
    install_stationary_tables(st);

    for (int start = 1; start + 1 <= W - 4; start += 2)
        run_fusion2_register_face(W, start, shared_kib * 1024ULL, rt, st, mod);

    std::cout << "ALL_OK two_cell_fusion2_register_face=1 W=" << W << '\n';
    return 0;
}
