#pragma push_macro("main")
#undef main
#define main two_cell_fusion2_register_reuse_microprobe_main_unused
#include "two_cell_fusion2_register_reuse_microprobe.cu"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_sectors.hpp"

namespace {

__device__ __forceinline__ void fusion_sector_load_block(
    std::uint32_t* block_values,
    const std::uint32_t* values,
    std::uint32_t outer,
    int W,
    int start,
    unsigned long long* sector_builds,
    unsigned long long* loads
) {
    const int sectors = oneesan::twocell::fusion_sector_count(FUSION_STEPS);
    for (int q = 0; q < sectors; ++q) {
        const auto sec = oneesan::twocell::fusion_sector(
            q, outer, W, start, FUSION_STEPS,
            TC_RANK_TABLES, TC_STATIONARY_TABLES);
        if (threadIdx.x == 0 && sec.valid) atomicAdd(sector_builds, 1ULL);
        if (!sec.valid) continue;
        for (Rank p = threadIdx.x; p < sec.count; p += blockDim.x) {
            block_values[sec.local_base + p] = values[sec.global_base + p];
            atomicAdd(loads, 1ULL);
        }
    }
}

__device__ __forceinline__ void fusion_sector_store_block(
    std::uint32_t* values,
    const std::uint32_t* block_values,
    std::uint32_t outer,
    int W,
    int start,
    unsigned long long* stores
) {
    const int sectors = oneesan::twocell::fusion_sector_count(FUSION_STEPS);
    for (int q = 0; q < sectors; ++q) {
        const auto sec = oneesan::twocell::fusion_sector(
            q, outer, W, start, FUSION_STEPS,
            TC_RANK_TABLES, TC_STATIONARY_TABLES);
        if (!sec.valid) continue;
        for (Rank p = threadIdx.x; p < sec.count; p += blockDim.x) {
            values[sec.global_base + p] = block_values[sec.local_base + p];
            atomicAdd(stores, 1ULL);
        }
    }
}

__global__ void two_cell_fusion2_sectorcopy_kernel(
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
    unsigned long long* primitive_scans,
    unsigned long long* primitive_reuses,
    unsigned long long* sector_builds,
    unsigned long long* global_loads,
    unsigned long long* global_stores,
    unsigned long long* residual_adds,
    int* error
) {
    extern __shared__ std::uint32_t block_values[];
    __shared__ PackedKey sh_src[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ PackedWord sh_label[FUSION_WARPS];
    __shared__ Rank sh_label_primitive[FUSION_WARPS];
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

        fusion_sector_load_block(
            block_values, values, outer, W, start, sector_builds, global_loads);
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
                        set_error(error, 441);
                    } else {
                        sh_label[warp] = label.word;
                        sh_label_primitive[warp] = label.primitive;
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
                        set_error(error, 442);
                    } else {
                        sh_ns[warp] = src.size;
                        for (int s = 0; s < src.size; ++s)
                            sh_src[warp][s] = src.value[s];
                    }
                }
                __syncwarp();

                const int ns = sh_ns[warp];
                if (ns <= 0 || ns > FUSION_MAX_COMPONENT) {
                    if (lane == 0) set_error(error, 443);
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
                    const PackedWord label = sh_label[warp];
                    const bool retained = oneesan::twocell::symbol(label, active) !=
                                          oneesan::twocell::TC_N;
                    const bool xN_deep = retained &&
                        oneesan::twocell::symbol(label, active + 1) ==
                        oneesan::twocell::TC_N;
                    const bool reuse_label = retained &&
                        (lane < 3 || (lane == 3 && xN_deep));

                    Rank primitive = sh_label_primitive[warp];
                    if (reuse_label) {
                        atomicAdd(primitive_reuses, 1ULL);
                    } else {
                        const int len = source.type ? W - 2 : W - 1;
                        primitive = oneesan::twocell::primitive_rank(
                            source.support, source.left, len, TC_RANK_TABLES);
                        atomicAdd(primitive_scans, 1ULL);
                    }
                    local_rank = oneesan::twocell::fusion_local_rank_at_with_primitive(
                        source, start, FUSION_STEPS, active, outer_ones,
                        primitive, TC_RANK_TABLES);
                    if (local_rank >= n) {
                        set_error(error, 444);
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

        fusion_sector_store_block(
            values, block_values, outer, W, start, global_stores);
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(processed_blocks, 1ULL);
        __syncthreads();
    }
}

Rank fusion2_sectorcopy_workspace_estimate() {
    return fusion2_register_reuse_workspace_estimate();
}

void print_fusion2_sectorcopy_plan(
    int W,
    Rank shared_limit,
    const RankTables& rt
) {
    const Rank reserve = fusion2_sectorcopy_workspace_estimate();
    const auto p = make_singlebuffer_plan(W, shared_limit, reserve, rt);
    const Rank total = p.fitted_states + p.fallback_states;
    const double f = total ? double(p.fitted_states) / double(total) : 0.0;
    std::cout << "fusion2_sectorcopy_plan"
              << " W=" << W
              << " shared_limit_bytes=" << shared_limit
              << " workspace_estimate=" << reserve
              << " max_fused_outer_ones=" << p.max_fused_outer_ones
              << " fused_state_fraction=" << f
              << " ideal_HBM_reduction_vs_two_pass=" << 0.5 * f
              << " max_copy_sectors=20"
              << " per_state_block_boundary_unrank=0"
              << " per_state_block_boundary_stationary_rank=0"
              << " start_end_sector_bases_identical=1"
              << " warp_register_arithmetic=1\n";
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const Rank shared_kib = argc > 2
        ? static_cast<Rank>(std::strtoull(argv[2], nullptr, 10)) : 228ULL;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth) return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    if (!plan_only) {
        std::cerr << "execution harness intentionally deferred; use --plan-only until sector-copy A/B is compiled on CUDA hardware\n";
        return 3;
    }
    print_fusion2_sectorcopy_plan(W, shared_kib * 1024ULL, rt);
    if (W == 28) {
        for (Rank kib : {64ULL, 96ULL, 128ULL, 160ULL, 192ULL, 228ULL, 256ULL})
            print_fusion2_sectorcopy_plan(W, kib * 1024ULL, rt);
    }
    return 0;
}
