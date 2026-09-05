#pragma push_macro("main")
#undef main
#define main two_cell_boundary_register_sector_microprobe_main_unused
#include "two_cell_boundary_register_sector_microprobe.cu"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_slices.hpp"

#include <cooperative_groups.h>

namespace {
namespace cg = cooperative_groups;

constexpr int BOUNDARY_CLUSTER_MAX = 8;

struct BoundaryPrimitiveLut {
    std::vector<std::uint32_t> value;
    Rank offset[oneesan::twocell::kMaxWidth + 1]{};
};

BoundaryPrimitiveLut build_boundary_primitive_lut(
    int W,
    const RankTables& rt
) {
    BoundaryPrimitiveLut lut;
    const int label_len = W - 2;
    for (int occupied = 1; occupied <= label_len; occupied += 2) {
        lut.offset[occupied] = static_cast<Rank>(lut.value.size());
        const std::uint32_t compact_support = oneesan::twocell::low_mask(occupied);
        const Rank pc = rt.primitive[occupied][1];
        for (Rank r = 0; r < pc; ++r) {
            lut.value.push_back(oneesan::twocell::primitive_left_unrank(
                compact_support, occupied, occupied, r, rt));
        }
    }
    return lut;
}

__device__ __forceinline__ std::uint32_t boundary_deposit_left_warp(
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

__device__ __forceinline__ std::uint32_t boundary_dsm_load(
    cg::cluster_group cluster,
    std::uint32_t* local_values,
    const oneesan::twocell::FusionSector* sectors,
    const Rank slice_base[BOUNDARY_CLUSTER_MAX][BOUNDARY_SECTORS],
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

__device__ __forceinline__ void boundary_dsm_store(
    cg::cluster_group cluster,
    std::uint32_t* local_values,
    const oneesan::twocell::FusionSector* sectors,
    const Rank slice_base[BOUNDARY_CLUSTER_MAX][BOUNDARY_SECTORS],
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

__global__ void two_cell_boundary_cluster_sliced_lut_kernel(
    std::uint32_t* __restrict__ values,
    const std::uint32_t* __restrict__ primitive_lut,
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
    __shared__ PackedKey sh_state[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ PackedWord sh_label[FUSION_WARPS];
    __shared__ Rank sh_label_primitive[FUSION_WARPS];
    __shared__ oneesan::twocell::FusionSector sh_sector[BOUNDARY_SECTORS];
    __shared__ Rank sh_slice_base[BOUNDARY_CLUSTER_MAX][BOUNDARY_SECTORS];
    __shared__ Rank sh_owner_total[BOUNDARY_CLUSTER_MAX];
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
        cluster_size > BOUNDARY_CLUSTER_MAX) {
        if (threadIdx.x == 0) set_error(error, 531);
        return;
    }

    const int start = W - 4;
    const int edge_active = W - 3;
    const int outer_bits = W - 4;
    const Rank cluster_id = Rank(blockIdx.x) / Rank(cluster_size);
    const Rank cluster_count = Rank(gridDim.x) / Rank(cluster_size);

    for (Rank support_rank = cluster_id; support_rank < support_count;
         support_rank += cluster_count) {
        const std::uint32_t outer = oneesan::twocell::support_unrank(
            outer_bits, outer_ones, support_rank, TC_RANK_TABLES);

        if (threadIdx.x < BOUNDARY_SECTORS) {
            sh_sector[threadIdx.x] = oneesan::twocell::fusion_sector(
                threadIdx.x, outer, W, start, BOUNDARY_STEPS,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
        }
        __syncthreads();

        if (threadIdx.x < cluster_size) {
            const int owner = threadIdx.x;
            Rank base = 0;
            for (int q = 0; q < BOUNDARY_SECTORS; ++q) {
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
                    set_error(error, 532);
                }
            }
        }
        __syncthreads();
        if (!sh_capacity_ok) {
            cluster.sync();
            continue;
        }

        // Load each stationary sector slice exactly once from global memory.
        for (int q = 0; q < BOUNDARY_SECTORS; ++q) {
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

        // Operator 1: final forward interior K_{W-4}.  Component labels inside
        // a steps=1 union block have only two local support bits.
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
                std::uint32_t compact_left = 0;
                if (lane == 0)
                    compact_left = primitive_lut[primitive_offset[occupied] + primitive];
                compact_left = __shfl_sync(0xffffffffu, compact_left, 0);
                const std::uint32_t left = boundary_deposit_left_warp(
                    label_support, compact_left, W - 2);

                if (lane == 0) {
                    sh_ns[warp] = 0;
                    sh_partner_rounds[warp] = 0;
                    sh_label[warp] = PackedWord{
                        label_support, left, static_cast<std::uint8_t>(W - 2)};
                    sh_label_primitive[warp] = primitive;
                    PackedWord collapsed{};
                    sh_deep[warp] = oneesan::twocell::deep_collapse(
                        sh_label[warp], start, collapsed) ? 1 : 0;
                }
                __syncwarp();

                if (sh_deep[warp]) {
                    oneesan::twocell::cuda_face::deep_component_sources_compact(
                        sh_label[warp], W, start,
                        sh_state[warp], &sh_ns[warp],
                        &sh_partner_rounds[warp], error);
                    __syncwarp();
                } else if (lane == 0) {
                    const auto src = oneesan::twocell::direct_component_sources(
                        sh_label[warp], W, start);
                    if (src.overflow || src.size <= 0 ||
                        src.size > FUSION_MAX_COMPONENT) {
                        set_error(error, 533);
                    } else {
                        sh_ns[warp] = src.size;
                        for (int s = 0; s < src.size; ++s)
                            sh_state[warp][s] = src.value[s];
                    }
                }
                __syncwarp();

                const int ns = sh_ns[warp];
                if (ns <= 0 || ns > FUSION_MAX_COMPONENT) {
                    if (lane == 0) set_error(error, 534);
                    __syncwarp();
                    continue;
                }

                std::uint32_t x = 0;
                PackedKey source{};
                Rank source_primitive = 0;
                int sector = -1;
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
                    source_primitive = sh_label_primitive[warp];
                    if (!reuse) {
                        const int len = source.type ? W - 2 : W - 1;
                        source_primitive = oneesan::twocell::primitive_rank(
                            source.support, source.left, len, TC_RANK_TABLES);
                    }
                    sector = oneesan::twocell::fusion_sector_index_at(
                        source, start, BOUNDARY_STEPS, start);
                    if (sector < 0 || sector >= BOUNDARY_SECTORS ||
                        !sh_sector[sector].valid) {
                        set_error(error, 535);
                    } else {
                        x = boundary_dsm_load(
                            cluster, local_values, sh_sector, sh_slice_base,
                            sector, source_primitive);
                    }
                }
                __syncwarp();

                const std::uint32_t y =
                    oneesan::twocell::cuda_component::apply_closed_component_warp(
                        sh_label[warp], source, ns, W, start, x, mod, error);
                __syncwarp();
                if (lane < ns && sector >= 0 && sector < BOUNDARY_SECTORS)
                    boundary_dsm_store(
                        cluster, local_values, sh_sector, sh_slice_base,
                        sector, source_primitive, y);
                __syncwarp();
            }
        }
        cluster.sync();

        // Operator 2: physical right row turn. Re-enumerate the same labels by
        // primitive owner; the turn states live in the same stationary sectors.
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
                std::uint32_t compact_left = 0;
                if (lane == 0)
                    compact_left = primitive_lut[primitive_offset[occupied] + primitive];
                compact_left = __shfl_sync(0xffffffffu, compact_left, 0);
                const std::uint32_t left = boundary_deposit_left_warp(
                    label_support, compact_left, W - 2);

                if (lane == 0) {
                    sh_label[warp] = PackedWord{
                        label_support, left, static_cast<std::uint8_t>(W - 2)};
                    sh_label_primitive[warp] = primitive;
                }
                __syncwarp();

                oneesan::twocell::cuda_turn::right_turn_closed_block_warp(
                    sh_label[warp], W, sh_state[warp], &sh_ns[warp],
                    &sh_singular[warp], error);
                __syncwarp();

                const int ns = sh_ns[warp];
                if (ns <= 0 || ns > oneesan::twocell::kMaxTurnStates) {
                    if (lane == 0) set_error(error, 536);
                    __syncwarp();
                    continue;
                }

                std::uint32_t x = 0;
                Rank state_primitive = 0;
                int sector = -1;
                if (lane < ns) {
                    const PackedKey state = sh_state[warp][lane];
                    state_primitive = sh_label_primitive[warp];
                    const bool reuse = sh_singular[warp] || lane == 1;
                    if (!reuse) {
                        const int len = state.type ? W - 2 : W - 1;
                        state_primitive = oneesan::twocell::primitive_rank(
                            state.support, state.left, len, TC_RANK_TABLES);
                    }
                    sector = oneesan::twocell::fusion_sector_index_at(
                        state, start, BOUNDARY_STEPS, edge_active);
                    if (sector < 0 || sector >= BOUNDARY_SECTORS ||
                        !sh_sector[sector].valid) {
                        set_error(error, 537);
                    } else {
                        x = boundary_dsm_load(
                            cluster, local_values, sh_sector, sh_slice_base,
                            sector, state_primitive);
                    }
                }
                __syncwarp();

                const std::uint32_t y = oneesan::twocell::cuda_turn::apply_closed_turn_warp(
                    sh_singular[warp] != 0, ns, x, mod, error);
                __syncwarp();
                if (lane < ns && sector >= 0 && sector < BOUNDARY_SECTORS)
                    boundary_dsm_store(
                        cluster, local_values, sh_sector, sh_slice_base,
                        sector, state_primitive, y);
                __syncwarp();
            }
        }
        cluster.sync();

        // Store each stationary primitive slice back once after both operators.
        for (int q = 0; q < BOUNDARY_SECTORS; ++q) {
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

Rank boundary_slice_owner_states(
    int outer_ones,
    int owner,
    int owners,
    const RankTables& rt
) {
    Rank total = 0;
    for (std::uint32_t code = 0; code < 8; ++code) {
        const Rank count = oneesan::twocell::primitive_count_for_occupied(
            outer_ones + oneesan::twocell::popcount32(code), rt);
        total += oneesan::twocell::primitive_slice_count(count, owner, owners);
    }
    for (std::uint32_t code = 0; code < 2; ++code) {
        const Rank count = oneesan::twocell::primitive_count_for_occupied(
            outer_ones + 1 + oneesan::twocell::popcount32(code), rt);
        total += oneesan::twocell::primitive_slice_count(count, owner, owners);
    }
    return total;
}

Rank boundary_slice_max_owner_states(
    int outer_ones,
    int owners,
    const RankTables& rt
) {
    Rank z = 0;
    for (int owner = 0; owner < owners; ++owner)
        z = std::max(z, boundary_slice_owner_states(
            outer_ones, owner, owners, rt));
    return z;
}

void print_boundary_cluster_plan(
    int W,
    Rank shared_limit,
    Rank reserve,
    int max_cluster,
    const RankTables& rt
) {
    const int outer_bits = W - 4;
    Rank total = 0, fused = 0;
    int max_o = -1;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank blocks = rt.choose[outer_bits][o];
        const Rank n = oneesan::twocell::fusion_block_size(
            BOUNDARY_STEPS, o, rt);
        total += blocks * n;
        int chosen = 0;
        for (int c : {1, 2, 4, 8}) {
            if (c > max_cluster) break;
            const Rank local = boundary_slice_max_owner_states(o, c, rt);
            if (local * sizeof(std::uint32_t) + reserve <= shared_limit) {
                chosen = c;
                break;
            }
        }
        if (chosen) {
            fused += blocks * n;
            max_o = o;
        }
    }
    const double f = total ? double(fused) / double(total) : 0.0;
    std::cout << "boundary_cluster_slice_plan"
              << " W=" << W
              << " max_cluster=" << max_cluster
              << " per_cta_shared_limit=" << shared_limit
              << " reserve=" << reserve
              << " max_fused_outer_ones=" << max_o
              << " fused_state_fraction=" << f
              << " ideal_HBM_reduction_vs_two_pass=" << 0.5 * f
              << " primitive_sliced_DSM=1"
              << " boundary_ops=2\n";
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const Rank shared_kib = argc > 2
        ? static_cast<Rank>(std::strtoull(argv[2], nullptr, 10)) : 228ULL;
    const Rank reserve = argc > 3
        ? static_cast<Rank>(std::strtoull(argv[3], nullptr, 10)) : 4096ULL;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth) return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    if (!plan_only) {
        std::cerr << "execution harness intentionally deferred; use --plan-only until boundary DSM is compiled on cluster-capable CUDA hardware\n";
        return 3;
    }
    for (int c : {1, 2, 4, 8})
        print_boundary_cluster_plan(
            W, shared_kib * 1024ULL, reserve, c, rt);
    return 0;
}
