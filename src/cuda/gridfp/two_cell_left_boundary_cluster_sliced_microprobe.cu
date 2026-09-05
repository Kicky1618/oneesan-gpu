#pragma push_macro("main")
#undef main
#define main two_cell_boundary_fusion_left_microprobe_main_unused
#include "two_cell_boundary_fusion_left_microprobe.cu"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_slices.hpp"
#include "../../common/two_cell_primitive_reflection.hpp"
#include "two_cell_parallel_face_device.cuh"
#include "two_cell_component_warp_arithmetic.cuh"
#include "two_cell_turn_warp_block.cuh"
#include "two_cell_turn_warp_arithmetic.cuh"

#include <cooperative_groups.h>

namespace {
namespace cg = cooperative_groups;

constexpr int LEFT_BOUNDARY_SECTORS = 10;
constexpr int LEFT_BOUNDARY_CLUSTER_MAX = 8;

struct LeftBoundaryPrimitiveLut {
    std::vector<std::uint32_t> label_left;
    std::vector<std::uint32_t> reflection_meta;
    Rank offset[oneesan::twocell::kMaxWidth + 1]{};
};

LeftBoundaryPrimitiveLut build_left_boundary_primitive_lut(
    int W,
    const RankTables& rt
) {
    LeftBoundaryPrimitiveLut lut;
    for (int occupied = 1; occupied <= W - 1; occupied += 2) {
        lut.offset[occupied] = static_cast<Rank>(lut.reflection_meta.size());
        const Rank pc = rt.primitive[occupied][1];
        const std::uint32_t support = oneesan::twocell::low_mask(occupied);
        for (Rank r = 0; r < pc; ++r) {
            const std::uint32_t left = oneesan::twocell::primitive_left_unrank(
                support, occupied, occupied, r, rt);
            if (occupied <= W - 2) lut.label_left.push_back(left);
            lut.reflection_meta.push_back(
                oneesan::twocell::make_primitive_reflection_meta(
                    left, occupied, rt));
        }
    }
    return lut;
}

__device__ __forceinline__ std::uint32_t left_boundary_deposit_left_warp(
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

__device__ __forceinline__ std::uint32_t left_boundary_dsm_load(
    cg::cluster_group cluster,
    std::uint32_t* local_values,
    const oneesan::twocell::FusionSector* sectors,
    const Rank slice_base[LEFT_BOUNDARY_CLUSTER_MAX][LEFT_BOUNDARY_SECTORS],
    int sector,
    Rank primitive
) {
    const int owners = static_cast<int>(cluster.num_blocks());
    const Rank count = sectors[sector].count;
    const int owner = oneesan::twocell::primitive_slice_owner(
        primitive, count, owners);
    if (owner < 0) return 0;
    const Rank begin = oneesan::twocell::primitive_slice_begin(
        count, owner, owners);
    const Rank offset = slice_base[owner][sector] + primitive - begin;
    std::uint32_t* remote = cluster.map_shared_rank(local_values, owner);
    return remote[offset];
}

__device__ __forceinline__ void left_boundary_dsm_store(
    cg::cluster_group cluster,
    std::uint32_t* local_values,
    const oneesan::twocell::FusionSector* sectors,
    const Rank slice_base[LEFT_BOUNDARY_CLUSTER_MAX][LEFT_BOUNDARY_SECTORS],
    int sector,
    Rank primitive,
    std::uint32_t value
) {
    const int owners = static_cast<int>(cluster.num_blocks());
    const Rank count = sectors[sector].count;
    const int owner = oneesan::twocell::primitive_slice_owner(
        primitive, count, owners);
    if (owner < 0) return;
    const Rank begin = oneesan::twocell::primitive_slice_begin(
        count, owner, owners);
    const Rank offset = slice_base[owner][sector] + primitive - begin;
    std::uint32_t* remote = cluster.map_shared_rank(local_values, owner);
    remote[offset] = value;
}

__global__ void two_cell_left_boundary_cluster_sliced_kernel(
    std::uint32_t* __restrict__ values,
    const std::uint32_t* __restrict__ label_left_lut,
    const std::uint32_t* __restrict__ reflection_lut,
    const Rank* __restrict__ primitive_offset,
    int W,
    int outer_ones,
    Rank support_count,
    Rank state_count,
    Rank local_capacity,
    std::uint32_t mod,
    int* error
) {
    extern __shared__ std::uint32_t local_values[];
    __shared__ PackedKey sh_forward[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ PackedWord sh_forward_label[FUSION_WARPS];
    __shared__ Rank sh_forward_label_primitive[FUSION_WARPS];
    __shared__ oneesan::twocell::FusionSector sh_sector[LEFT_BOUNDARY_SECTORS];
    __shared__ Rank sh_slice_base[LEFT_BOUNDARY_CLUSTER_MAX][LEFT_BOUNDARY_SECTORS];
    __shared__ Rank sh_owner_total[LEFT_BOUNDARY_CLUSTER_MAX];
    __shared__ int sh_ns[FUSION_WARPS];
    __shared__ int sh_deep[FUSION_WARPS];
    __shared__ int sh_singular[FUSION_WARPS];
    __shared__ int sh_partner_rounds[FUSION_WARPS];
    __shared__ int sh_capacity_ok;

    cg::cluster_group cluster = cg::this_cluster();
    const int cluster_size = static_cast<int>(cluster.num_blocks());
    const int block_rank = static_cast<int>(cluster.block_rank());
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (blockDim.x != FUSION_THREADS || cluster_size < 1 ||
        cluster_size > LEFT_BOUNDARY_CLUSTER_MAX) {
        if (threadIdx.x == 0) set_error(error, 591);
        return;
    }

    const int start = 0;
    const int source_active = 1;
    const int edge_active = 0;
    const int forward_i = W - 4;
    const int outer_bits = W - 4;
    const Rank cluster_id = Rank(blockIdx.x) / Rank(cluster_size);
    const Rank cluster_count = Rank(gridDim.x) / Rank(cluster_size);

    for (Rank support_rank = cluster_id; support_rank < support_count;
         support_rank += cluster_count) {
        const std::uint32_t outer = oneesan::twocell::support_unrank(
            outer_bits, outer_ones, support_rank, TC_RANK_TABLES);

        if (threadIdx.x < LEFT_BOUNDARY_SECTORS) {
            sh_sector[threadIdx.x] = oneesan::twocell::fusion_sector(
                threadIdx.x, outer, W, start, BOUNDARY_STEPS,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
        }
        __syncthreads();
        if (threadIdx.x < cluster_size) {
            const int owner = threadIdx.x;
            Rank base = 0;
            for (int q = 0; q < LEFT_BOUNDARY_SECTORS; ++q) {
                sh_slice_base[owner][q] = base;
                if (!sh_sector[q].valid) continue;
                base += oneesan::twocell::primitive_slice_count(
                    sh_sector[q].count, owner, cluster_size);
            }
            sh_owner_total[owner] = base;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            sh_capacity_ok = 1;
            for (int owner = 0; owner < cluster_size; ++owner) {
                if (sh_owner_total[owner] > local_capacity) {
                    sh_capacity_ok = 0;
                    set_error(error, 592);
                }
            }
        }
        __syncthreads();
        if (!sh_capacity_ok) {
            cluster.sync();
            continue;
        }

        for (int q = 0; q < LEFT_BOUNDARY_SECTORS; ++q) {
            const auto sec = sh_sector[q];
            if (!sec.valid) continue;
            const Rank begin = oneesan::twocell::primitive_slice_begin(
                sec.count, block_rank, cluster_size);
            const Rank end = oneesan::twocell::primitive_slice_end(
                sec.count, block_rank, cluster_size);
            const Rank base = sh_slice_base[block_rank][q];
            for (Rank p = begin + threadIdx.x; p < end; p += blockDim.x)
                local_values[base + p - begin] = values[sec.global_base + p];
        }
        cluster.sync();

        // Final reverse interior K: reflect to forward K_{W-4}, apply its
        // closed component arithmetic, then write to the same stationary rank.
        for (std::uint32_t code = 0; code < 4; ++code) {
            const int occupied = outer_ones + oneesan::twocell::popcount32(code);
            const Rank pc = oneesan::twocell::primitive_count_for_occupied(
                occupied, TC_RANK_TABLES);
            if (!pc) continue;
            const Rank pbegin = oneesan::twocell::primitive_slice_begin(
                pc, block_rank, cluster_size);
            const Rank pend = oneesan::twocell::primitive_slice_end(
                pc, block_rank, cluster_size);
            const std::uint32_t label_support = oneesan::twocell::insert_support_window(
                outer, start, BOUNDARY_STEPS + 1, code);

            for (Rank primitive = pbegin + warp;
                 primitive < pend; primitive += FUSION_WARPS) {
                std::uint32_t compact_left = 0, label_meta = 0;
                if (lane == 0) {
                    compact_left = label_left_lut[
                        primitive_offset[occupied] + primitive];
                    label_meta = reflection_lut[
                        primitive_offset[occupied] + primitive];
                }
                compact_left = __shfl_sync(0xffffffffu, compact_left, 0);
                label_meta = __shfl_sync(0xffffffffu, label_meta, 0);
                const std::uint32_t left = left_boundary_deposit_left_warp(
                    label_support, compact_left, W - 2);

                if (lane == 0) {
                    const PackedWord reverse_label{
                        label_support, left, static_cast<std::uint8_t>(W - 2)};
                    sh_forward_label[warp] =
                        oneesan::twocell::reflect_word_with_reflection_meta(
                            reverse_label, label_meta);
                    sh_forward_label_primitive[warp] =
                        oneesan::twocell::primitive_reflection_mirror_rank(label_meta);
                    sh_ns[warp] = 0;
                    sh_partner_rounds[warp] = 0;
                    PackedWord collapsed{};
                    sh_deep[warp] = oneesan::twocell::deep_collapse(
                        sh_forward_label[warp], forward_i, collapsed) ? 1 : 0;
                }
                __syncwarp();

                if (sh_deep[warp]) {
                    oneesan::twocell::cuda_face::deep_component_sources_compact(
                        sh_forward_label[warp], W, forward_i,
                        sh_forward[warp], &sh_ns[warp],
                        &sh_partner_rounds[warp], error);
                    __syncwarp();
                } else if (lane == 0) {
                    const auto src = oneesan::twocell::direct_component_sources(
                        sh_forward_label[warp], W, forward_i);
                    if (src.overflow || src.size <= 0 ||
                        src.size > FUSION_MAX_COMPONENT) {
                        set_error(error, 593);
                    } else {
                        sh_ns[warp] = src.size;
                        for (int q = 0; q < src.size; ++q)
                            sh_forward[warp][q] = src.value[q];
                    }
                }
                __syncwarp();

                const int ns = sh_ns[warp];
                if (ns <= 0 || ns > FUSION_MAX_COMPONENT) {
                    if (lane == 0) set_error(error, 594);
                    __syncwarp();
                    continue;
                }

                PackedKey forward_source{};
                Rank reverse_primitive = 0;
                int sector = -1;
                std::uint32_t x = 0;
                if (lane < ns) {
                    forward_source = sh_forward[warp][lane];
                    const PackedWord flabel = sh_forward_label[warp];
                    const bool retained = oneesan::twocell::symbol(
                        flabel, forward_i) != oneesan::twocell::TC_N;
                    const bool xN_deep = retained &&
                        oneesan::twocell::symbol(flabel, forward_i + 1) ==
                        oneesan::twocell::TC_N;
                    const bool reuse_forward = retained &&
                        (lane < 3 || (lane == 3 && xN_deep));
                    Rank forward_primitive = sh_forward_label_primitive[warp];
                    if (!reuse_forward) {
                        const int len = forward_source.type ? W - 2 : W - 1;
                        forward_primitive = oneesan::twocell::primitive_rank(
                            forward_source.support, forward_source.left,
                            len, TC_RANK_TABLES);
                    }
                    const int len = forward_source.type ? W - 2 : W - 1;
                    const int source_occupied = oneesan::twocell::popcount32(
                        forward_source.support & oneesan::twocell::low_mask(len));
                    const std::uint32_t meta = reflection_lut[
                        primitive_offset[source_occupied] + forward_primitive];
                    reverse_primitive =
                        oneesan::twocell::primitive_reflection_mirror_rank(meta);
                    const PackedKey reverse_source =
                        oneesan::twocell::reflect_key_with_reflection_meta(
                            forward_source, W, meta);
                    sector = oneesan::twocell::fusion_sector_index_at(
                        reverse_source, start, BOUNDARY_STEPS, source_active);
                    if (sector < 0 || sector >= LEFT_BOUNDARY_SECTORS ||
                        !sh_sector[sector].valid ||
                        reverse_primitive >= sh_sector[sector].count) {
                        set_error(error, 595);
                    } else {
                        x = left_boundary_dsm_load(
                            cluster, local_values, sh_sector, sh_slice_base,
                            sector, reverse_primitive);
                    }
                }
                __syncwarp();

                const std::uint32_t y =
                    oneesan::twocell::cuda_component::apply_closed_component_warp(
                        sh_forward_label[warp], forward_source,
                        ns, W, forward_i, x, mod, error);
                __syncwarp();
                if (lane < ns && sector >= 0 && sector < LEFT_BOUNDARY_SECTORS)
                    left_boundary_dsm_store(
                        cluster, local_values, sh_sector, sh_slice_base,
                        sector, reverse_primitive, y);
                __syncwarp();
            }
        }
        cluster.sync();

        // Physical left turn: construct the mirrored right-turn block, then
        // reflect each state back using primitive mirror metadata.
        for (std::uint32_t code = 0; code < 4; ++code) {
            const int occupied = outer_ones + oneesan::twocell::popcount32(code);
            const Rank pc = oneesan::twocell::primitive_count_for_occupied(
                occupied, TC_RANK_TABLES);
            if (!pc) continue;
            const Rank pbegin = oneesan::twocell::primitive_slice_begin(
                pc, block_rank, cluster_size);
            const Rank pend = oneesan::twocell::primitive_slice_end(
                pc, block_rank, cluster_size);
            const std::uint32_t label_support = oneesan::twocell::insert_support_window(
                outer, start, BOUNDARY_STEPS + 1, code);

            for (Rank primitive = pbegin + warp;
                 primitive < pend; primitive += FUSION_WARPS) {
                std::uint32_t compact_left = 0, label_meta = 0;
                if (lane == 0) {
                    compact_left = label_left_lut[
                        primitive_offset[occupied] + primitive];
                    label_meta = reflection_lut[
                        primitive_offset[occupied] + primitive];
                }
                compact_left = __shfl_sync(0xffffffffu, compact_left, 0);
                label_meta = __shfl_sync(0xffffffffu, label_meta, 0);
                const std::uint32_t left = left_boundary_deposit_left_warp(
                    label_support, compact_left, W - 2);

                if (lane == 0) {
                    const PackedWord left_label{
                        label_support, left, static_cast<std::uint8_t>(W - 2)};
                    sh_forward_label[warp] =
                        oneesan::twocell::reflect_word_with_reflection_meta(
                            left_label, label_meta);
                    sh_forward_label_primitive[warp] =
                        oneesan::twocell::primitive_reflection_mirror_rank(label_meta);
                }
                __syncwarp();

                oneesan::twocell::cuda_turn::right_turn_closed_block_warp(
                    sh_forward_label[warp], W, sh_forward[warp], &sh_ns[warp],
                    &sh_singular[warp], error);
                __syncwarp();

                const int ns = sh_ns[warp];
                if (ns <= 0 || ns > oneesan::twocell::kMaxTurnStates) {
                    if (lane == 0) set_error(error, 596);
                    __syncwarp();
                    continue;
                }

                Rank left_primitive = 0;
                int sector = -1;
                std::uint32_t x = 0;
                if (lane < ns) {
                    const PackedKey forward_state = sh_forward[warp][lane];
                    Rank forward_primitive = sh_forward_label_primitive[warp];
                    const bool reuse_forward = sh_singular[warp] || lane == 1;
                    if (!reuse_forward) {
                        const int len = forward_state.type ? W - 2 : W - 1;
                        forward_primitive = oneesan::twocell::primitive_rank(
                            forward_state.support, forward_state.left,
                            len, TC_RANK_TABLES);
                    }
                    const int len = forward_state.type ? W - 2 : W - 1;
                    const int state_occupied = oneesan::twocell::popcount32(
                        forward_state.support & oneesan::twocell::low_mask(len));
                    const std::uint32_t meta = reflection_lut[
                        primitive_offset[state_occupied] + forward_primitive];
                    left_primitive =
                        oneesan::twocell::primitive_reflection_mirror_rank(meta);
                    const PackedKey left_state =
                        oneesan::twocell::reflect_key_with_reflection_meta(
                            forward_state, W, meta);
                    sector = oneesan::twocell::fusion_sector_index_at(
                        left_state, start, BOUNDARY_STEPS, edge_active);
                    if (sector < 0 || sector >= LEFT_BOUNDARY_SECTORS ||
                        !sh_sector[sector].valid ||
                        left_primitive >= sh_sector[sector].count) {
                        set_error(error, 597);
                    } else {
                        x = left_boundary_dsm_load(
                            cluster, local_values, sh_sector, sh_slice_base,
                            sector, left_primitive);
                    }
                }
                __syncwarp();

                const std::uint32_t y = oneesan::twocell::cuda_turn::apply_closed_turn_warp(
                    sh_singular[warp] != 0, ns, x, mod, error);
                __syncwarp();
                if (lane < ns && sector >= 0 && sector < LEFT_BOUNDARY_SECTORS)
                    left_boundary_dsm_store(
                        cluster, local_values, sh_sector, sh_slice_base,
                        sector, left_primitive, y);
                __syncwarp();
            }
        }
        cluster.sync();

        for (int q = 0; q < LEFT_BOUNDARY_SECTORS; ++q) {
            const auto sec = sh_sector[q];
            if (!sec.valid) continue;
            const Rank begin = oneesan::twocell::primitive_slice_begin(
                sec.count, block_rank, cluster_size);
            const Rank end = oneesan::twocell::primitive_slice_end(
                sec.count, block_rank, cluster_size);
            const Rank base = sh_slice_base[block_rank][q];
            for (Rank p = begin + threadIdx.x; p < end; p += blockDim.x)
                values[sec.global_base + p] = local_values[base + p - begin];
        }
        cluster.sync();
    }
}

Rank left_boundary_slice_max_owner_states(
    int outer_ones,
    int owners,
    const RankTables& rt
) {
    Rank worst = 0;
    for (int owner = 0; owner < owners; ++owner) {
        Rank local = 0;
        for (std::uint32_t code = 0; code < 8; ++code) {
            const Rank count = oneesan::twocell::primitive_count_for_occupied(
                outer_ones + oneesan::twocell::popcount32(code), rt);
            local += oneesan::twocell::primitive_slice_count(count, owner, owners);
        }
        for (std::uint32_t code = 0; code < 2; ++code) {
            const Rank count = oneesan::twocell::primitive_count_for_occupied(
                outer_ones + 1 + oneesan::twocell::popcount32(code), rt);
            local += oneesan::twocell::primitive_slice_count(count, owner, owners);
        }
        worst = std::max(worst, local);
    }
    return worst;
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const Rank shared_kib = argc > 2
        ? static_cast<Rank>(std::strtoull(argv[2], nullptr, 10)) : 228ULL;
    const Rank reserve = argc > 3
        ? static_cast<Rank>(std::strtoull(argv[3], nullptr, 10)) : 8192ULL;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth) return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    if (!plan_only) {
        std::cerr << "execution harness deferred; use --plan-only until forced left-boundary DSM validation is added\n";
        return 3;
    }
    const int outer_bits = W - 4;
    for (int max_cluster : {1, 2, 4, 8}) {
        Rank total = 0, fused = 0;
        for (int o = 0; o <= outer_bits; ++o) {
            const Rank blocks = rt.choose[outer_bits][o];
            const Rank n = oneesan::twocell::fusion_block_size(
                BOUNDARY_STEPS, o, rt);
            total += blocks * n;
            for (int c : {1, 2, 4, 8}) {
                if (c > max_cluster) break;
                const Rank local = left_boundary_slice_max_owner_states(o, c, rt);
                if (local * sizeof(std::uint32_t) + reserve <= shared_kib * 1024ULL) {
                    fused += blocks * n;
                    break;
                }
            }
        }
        const double f = total ? double(fused) / double(total) : 0.0;
        std::cout << "left_boundary_cluster_plan"
                  << " W=" << W
                  << " max_cluster=" << max_cluster
                  << " fused_state_fraction=" << f
                  << " ideal_HBM_reduction_vs_two_pass=" << 0.5 * f
                  << " capacity_same_as_right_boundary=1\n";
    }
    return 0;
}
