#pragma push_macro("main")
#undef main
#define main two_cell_boundary_fusion_singlebuffer_microprobe_main_unused
#include "two_cell_boundary_fusion_singlebuffer_microprobe.cu"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_sectors.hpp"
#include "two_cell_parallel_face_device.cuh"
#include "two_cell_component_warp_arithmetic.cuh"
#include "two_cell_turn_warp_block.cuh"
#include "two_cell_turn_warp_arithmetic.cuh"

namespace {

constexpr int BOUNDARY_SECTORS = 10; // steps=1 => 8 A + 2 C

__global__ void two_cell_boundary_register_sector_kernel(
    std::uint32_t* __restrict__ values,
    int W,
    int outer_ones,
    Rank support_count,
    Rank state_count,
    std::uint32_t mod,
    unsigned long long* processed_blocks,
    unsigned long long* processed_components,
    unsigned long long* deep_components,
    unsigned long long* global_loads,
    unsigned long long* global_stores,
    unsigned long long* local_adds,
    int* error
) {
    extern __shared__ std::uint32_t block_values[];
    __shared__ PackedKey sh_state[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ PackedWord sh_label[FUSION_WARPS];
    __shared__ Rank sh_label_primitive[FUSION_WARPS];
    __shared__ oneesan::twocell::FusionSector sh_sector[BOUNDARY_SECTORS];
    __shared__ int sh_ns[FUSION_WARPS];
    __shared__ int sh_deep[FUSION_WARPS];
    __shared__ int sh_singular[FUSION_WARPS];
    __shared__ int sh_partner_rounds[FUSION_WARPS];

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

        if (threadIdx.x < BOUNDARY_SECTORS) {
            sh_sector[threadIdx.x] = oneesan::twocell::fusion_sector(
                threadIdx.x, outer, W, start, BOUNDARY_STEPS,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
        }
        __syncthreads();

        for (int q = 0; q < BOUNDARY_SECTORS; ++q) {
            const auto sec = sh_sector[q];
            if (!sec.valid) continue;
            for (Rank p = threadIdx.x; p < sec.count; p += blockDim.x)
                block_values[sec.local_base + p] = values[sec.global_base + p];
        }
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(global_loads, static_cast<unsigned long long>(n));

        // Final forward interior K_{W-4}.
        for (Rank cr = warp; cr < nc; cr += FUSION_WARPS) {
            if (lane == 0) {
                sh_ns[warp] = 0;
                sh_partner_rounds[warp] = 0;
                const auto label = oneesan::twocell::fusion_component_unrank(
                    cr, outer, W, start, BOUNDARY_STEPS, TC_RANK_TABLES);
                if (!label.valid) {
                    sh_deep[warp] = 0;
                    set_error(error, 511);
                } else {
                    sh_label[warp] = label.word;
                    sh_label_primitive[warp] = label.primitive;
                    PackedWord collapsed{};
                    sh_deep[warp] = oneesan::twocell::deep_collapse(
                        label.word, start, collapsed) ? 1 : 0;
                }
            }
            __syncwarp();

            if (sh_deep[warp]) {
                oneesan::twocell::cuda_face::deep_component_sources_compact(
                    sh_label[warp], W, start,
                    sh_state[warp], &sh_ns[warp],
                    &sh_partner_rounds[warp], error);
                __syncwarp();
                if (lane == 0) atomicAdd(deep_components, 1ULL);
            } else if (lane == 0) {
                const auto src = oneesan::twocell::direct_component_sources(
                    sh_label[warp], W, start);
                if (src.overflow || src.size <= 0 ||
                    src.size > FUSION_MAX_COMPONENT) {
                    set_error(error, 512);
                } else {
                    sh_ns[warp] = src.size;
                    for (int s = 0; s < src.size; ++s)
                        sh_state[warp][s] = src.value[s];
                }
            }
            __syncwarp();

            const int ns = sh_ns[warp];
            if (ns <= 0 || ns > FUSION_MAX_COMPONENT) {
                if (lane == 0) set_error(error, 513);
                __syncwarp();
                continue;
            }
            if (lane == 0) {
                atomicAdd(processed_components, 1ULL);
                atomicAdd(local_adds, static_cast<unsigned long long>(ns - 1));
            }

            Rank lr = ~Rank(0);
            std::uint32_t x = 0;
            PackedKey source{};
            if (lane < ns) {
                source = sh_state[warp][lane];
                const PackedWord label = sh_label[warp];
                const bool retained = oneesan::twocell::symbol(label, start) !=
                                      oneesan::twocell::TC_N;
                const bool xN_deep = retained &&
                    oneesan::twocell::symbol(label, start + 1) ==
                    oneesan::twocell::TC_N;
                const bool reuse = retained &&
                    (lane < 3 || (lane == 3 && xN_deep));
                Rank primitive = sh_label_primitive[warp];
                if (!reuse) {
                    const int len = source.type ? W - 2 : W - 1;
                    primitive = oneesan::twocell::primitive_rank(
                        source.support, source.left, len, TC_RANK_TABLES);
                }
                lr = oneesan::twocell::fusion_local_rank_at_with_primitive(
                    source, start, BOUNDARY_STEPS, start, outer_ones,
                    primitive, TC_RANK_TABLES);
                if (lr >= n) set_error(error, 514);
                else x = block_values[lr];
            }
            __syncwarp();

            const std::uint32_t y =
                oneesan::twocell::cuda_component::apply_closed_component_warp(
                    sh_label[warp], source, ns, W, start, x, mod, error);
            __syncwarp();
            if (lane < ns && lr < n) block_values[lr] = y;
            __syncwarp();
        }
        __syncthreads();

        // Physical right row turn on Q_{W-3}.
        for (Rank cr = warp; cr < nc; cr += FUSION_WARPS) {
            if (lane == 0) {
                const auto label = oneesan::twocell::fusion_component_unrank(
                    cr, outer, W, start, BOUNDARY_STEPS, TC_RANK_TABLES);
                if (!label.valid) {
                    sh_ns[warp] = 0;
                    set_error(error, 515);
                } else {
                    sh_label[warp] = label.word;
                    sh_label_primitive[warp] = label.primitive;
                }
            }
            __syncwarp();

            oneesan::twocell::cuda_turn::right_turn_closed_block_warp(
                sh_label[warp], W, sh_state[warp], &sh_ns[warp],
                &sh_singular[warp], error);
            __syncwarp();

            const int ns = sh_ns[warp];
            if (ns <= 0 || ns > oneesan::twocell::kMaxTurnStates) {
                if (lane == 0) set_error(error, 516);
                __syncwarp();
                continue;
            }
            if (lane == 0) {
                atomicAdd(processed_components, 1ULL);
                atomicAdd(local_adds,
                          static_cast<unsigned long long>(
                              sh_singular[warp] ? 2 : ns - 1));
            }

            Rank lr = ~Rank(0);
            std::uint32_t x = 0;
            if (lane < ns) {
                const PackedKey state = sh_state[warp][lane];
                Rank primitive = sh_label_primitive[warp];
                const bool reuse = sh_singular[warp] || lane == 1;
                if (!reuse) {
                    const int len = state.type ? W - 2 : W - 1;
                    primitive = oneesan::twocell::primitive_rank(
                        state.support, state.left, len, TC_RANK_TABLES);
                }
                lr = oneesan::twocell::fusion_local_rank_at_with_primitive(
                    state, start, BOUNDARY_STEPS, edge_active, outer_ones,
                    primitive, TC_RANK_TABLES);
                if (lr >= n) set_error(error, 517);
                else x = block_values[lr];
            }
            __syncwarp();

            const std::uint32_t y =
                oneesan::twocell::cuda_turn::apply_closed_turn_warp(
                    sh_singular[warp] != 0, ns, x, mod, error);
            __syncwarp();
            if (lane < ns && lr < n) block_values[lr] = y;
            __syncwarp();
        }
        __syncthreads();

        // Stationary sector intervals have identical global bases at start and
        // edge_active, so the same descriptors are reused for the store.
        for (int q = 0; q < BOUNDARY_SECTORS; ++q) {
            const auto sec = sh_sector[q];
            if (!sec.valid) continue;
            for (Rank p = threadIdx.x; p < sec.count; p += blockDim.x)
                values[sec.global_base + p] = block_values[sec.local_base + p];
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            atomicAdd(global_stores, static_cast<unsigned long long>(n));
            atomicAdd(processed_blocks, 1ULL);
        }
        __syncthreads();
    }
}

Rank boundary_register_workspace_estimate() {
    return Rank(FUSION_WARPS) * (
        Rank(FUSION_MAX_COMPONENT) * sizeof(PackedKey) +
        sizeof(PackedWord) + sizeof(Rank) + 4 * sizeof(int)) +
        Rank(BOUNDARY_SECTORS) * sizeof(oneesan::twocell::FusionSector) + 256ULL;
}

void print_boundary_register_plan(
    int W,
    Rank shared_limit,
    const RankTables& rt
) {
    const Rank reserve = boundary_register_workspace_estimate();
    print_boundary_singlebuffer_plan(W, shared_limit, reserve, rt);
    std::cout << "boundary_register_features"
              << " sector_descriptors=10"
              << " per_state_boundary_rank_calls=0"
              << " interior_register_arithmetic=1"
              << " turn_register_arithmetic=1"
              << " turn_serial_height_loop=0"
              << " matching_K_step_calls=0\n";
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 8;
    const Rank shared_kib = argc > 2
        ? static_cast<Rank>(std::strtoull(argv[2], nullptr, 10)) : 228ULL;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth) return 2;
    const RankTables rt = oneesan::twocell::make_rank_tables();
    if (plan_only) {
        print_boundary_register_plan(W, shared_kib * 1024ULL, rt);
        return 0;
    }
    std::cerr << "execution harness intentionally deferred; use --plan-only until the register/sector boundary kernel is compiled on CUDA hardware\n";
    return 3;
}
