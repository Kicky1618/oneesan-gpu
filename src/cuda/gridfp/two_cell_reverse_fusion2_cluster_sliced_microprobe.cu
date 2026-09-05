#pragma push_macro("main")
#undef main
#define main two_cell_reverse_fusion2_register_sector_microprobe_main_unused
#include "two_cell_reverse_fusion2_register_sector_microprobe.cu"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_slices.hpp"

#include <cooperative_groups.h>

namespace {
namespace cg = cooperative_groups;

constexpr int REVERSE_CLUSTER_MAX = 8;

__device__ __forceinline__ std::uint32_t reverse_dsm_load(
    cg::cluster_group cluster,
    std::uint32_t* local_values,
    const oneesan::twocell::FusionSector* sectors,
    const Rank slice_base[REVERSE_CLUSTER_MAX][FUSION2_SECTORS],
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

__device__ __forceinline__ void reverse_dsm_store(
    cg::cluster_group cluster,
    std::uint32_t* local_values,
    const oneesan::twocell::FusionSector* sectors,
    const Rank slice_base[REVERSE_CLUSTER_MAX][FUSION2_SECTORS],
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

__global__ void two_cell_reverse_fusion2_cluster_sliced_kernel(
    std::uint32_t* __restrict__ values,
    const std::uint32_t* __restrict__ label_left_lut,
    const std::uint32_t* __restrict__ reflection_lut,
    const Rank* __restrict__ primitive_offset,
    int W,
    int start,
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
    __shared__ oneesan::twocell::FusionSector sh_sector[FUSION2_SECTORS];
    __shared__ Rank sh_slice_base[REVERSE_CLUSTER_MAX][FUSION2_SECTORS];
    __shared__ Rank sh_owner_total[REVERSE_CLUSTER_MAX];
    __shared__ int sh_ns[FUSION_WARPS];
    __shared__ int sh_deep[FUSION_WARPS];
    __shared__ int sh_partner_rounds[FUSION_WARPS];
    __shared__ int sh_capacity_ok;

    cg::cluster_group cluster = cg::this_cluster();
    const int cluster_size = static_cast<int>(cluster.num_blocks());
    const int block_rank = static_cast<int>(cluster.block_rank());
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (blockDim.x != FUSION_THREADS || cluster_size < 1 ||
        cluster_size > REVERSE_CLUSTER_MAX) {
        if (threadIdx.x == 0) set_error(error, 581);
        return;
    }

    const int outer_bits = W - FUSION_STEPS - 3;
    const Rank cluster_id = Rank(blockIdx.x) / Rank(cluster_size);
    const Rank cluster_count = Rank(gridDim.x) / Rank(cluster_size);

    for (Rank support_rank = cluster_id; support_rank < support_count;
         support_rank += cluster_count) {
        const std::uint32_t outer = oneesan::twocell::support_unrank(
            outer_bits, outer_ones, support_rank, TC_RANK_TABLES);

        if (threadIdx.x < FUSION2_SECTORS) {
            sh_sector[threadIdx.x] = oneesan::twocell::fusion_sector(
                threadIdx.x, outer, W, start, FUSION_STEPS,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
        }
        __syncthreads();

        if (threadIdx.x < cluster_size) {
            const int owner = threadIdx.x;
            Rank base = 0;
            for (int q = 0; q < FUSION2_SECTORS; ++q) {
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
                    set_error(error, 582);
                }
            }
        }
        __syncthreads();
        if (!sh_capacity_ok) {
            cluster.sync();
            continue;
        }

        for (int q = 0; q < FUSION2_SECTORS; ++q) {
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

        for (int phase = 0; phase < FUSION_STEPS; ++phase) {
            const int reverse_active = start + FUSION_STEPS - phase;
            const int forward_i = W - start - FUSION_STEPS - 3 + phase;

            for (std::uint32_t code = 0; code < 8; ++code) {
                const int occupied = outer_ones + oneesan::twocell::popcount32(code);
                const Rank pc = oneesan::twocell::primitive_count_for_occupied(
                    occupied, TC_RANK_TABLES);
                if (!pc) continue;
                const Rank pbegin = oneesan::twocell::primitive_slice_begin(
                    pc, block_rank, cluster_size);
                const Rank pend = oneesan::twocell::primitive_slice_end(
                    pc, block_rank, cluster_size);
                const std::uint32_t label_support =
                    oneesan::twocell::insert_support_window(
                        outer, start, FUSION_STEPS + 1, code);

                for (Rank primitive = pbegin + warp;
                     primitive < pend; primitive += FUSION_WARPS) {
                    std::uint32_t compact_left = 0;
                    std::uint32_t label_ref_meta = 0;
                    if (lane == 0) {
                        compact_left = label_left_lut[
                            primitive_offset[occupied] + primitive];
                        label_ref_meta = reflection_lut[
                            primitive_offset[occupied] + primitive];
                    }
                    compact_left = __shfl_sync(0xffffffffu, compact_left, 0);
                    label_ref_meta = __shfl_sync(
                        0xffffffffu, label_ref_meta, 0);
                    const std::uint32_t left = reverse_deposit_left_warp(
                        label_support, compact_left, W - 2);

                    if (lane == 0) {
                        const PackedWord reverse_label{
                            label_support, left, static_cast<std::uint8_t>(W - 2)};
                        sh_forward_label[warp] =
                            oneesan::twocell::reflect_word_with_reflection_meta(
                                reverse_label, label_ref_meta);
                        sh_forward_label_primitive[warp] =
                            oneesan::twocell::primitive_reflection_mirror_rank(
                                label_ref_meta);
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
                            set_error(error, 583);
                        } else {
                            sh_ns[warp] = src.size;
                            for (int s = 0; s < src.size; ++s)
                                sh_forward[warp][s] = src.value[s];
                        }
                    }
                    __syncwarp();

                    const int ns = sh_ns[warp];
                    if (ns <= 0 || ns > FUSION_MAX_COMPONENT) {
                        if (lane == 0) set_error(error, 584);
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
                            reverse_source, start, FUSION_STEPS, reverse_active);
                        if (sector < 0 || sector >= FUSION2_SECTORS ||
                            !sh_sector[sector].valid ||
                            reverse_primitive >= sh_sector[sector].count) {
                            set_error(error, 585);
                        } else {
                            x = reverse_dsm_load(
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
                    if (lane < ns && sector >= 0 && sector < FUSION2_SECTORS)
                        reverse_dsm_store(
                            cluster, local_values, sh_sector, sh_slice_base,
                            sector, reverse_primitive, y);
                    __syncwarp();
                }
            }
            cluster.sync();
        }

        for (int q = 0; q < FUSION2_SECTORS; ++q) {
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

Rank reverse_slice_max_owner_states(
    int outer_ones,
    int owners,
    const RankTables& rt
) {
    Rank worst = 0;
    for (int owner = 0; owner < owners; ++owner) {
        Rank local = 0;
        for (std::uint32_t code = 0; code < 16; ++code) {
            const Rank count = oneesan::twocell::primitive_count_for_occupied(
                outer_ones + oneesan::twocell::popcount32(code), rt);
            local += oneesan::twocell::primitive_slice_count(
                count, owner, owners);
        }
        for (std::uint32_t code = 0; code < 4; ++code) {
            const Rank count = oneesan::twocell::primitive_count_for_occupied(
                outer_ones + 1 + oneesan::twocell::popcount32(code), rt);
            local += oneesan::twocell::primitive_slice_count(
                count, owner, owners);
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
        std::cerr << "execution harness deferred; use reverse single-CTA correctness first or --plan-only\n";
        return 3;
    }

    const int outer_bits = W - 5;
    for (int max_cluster : {1, 2, 4, 8}) {
        Rank total = 0, fused = 0;
        int max_o = -1;
        for (int o = 0; o <= outer_bits; ++o) {
            const Rank blocks = rt.choose[outer_bits][o];
            const Rank n = oneesan::twocell::fusion_block_size(2, o, rt);
            total += blocks * n;
            int chosen = 0;
            for (int c : {1, 2, 4, 8}) {
                if (c > max_cluster) break;
                const Rank local = reverse_slice_max_owner_states(o, c, rt);
                if (local * sizeof(std::uint32_t) + reserve <= shared_kib * 1024ULL) {
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
        std::cout << "reverse_fusion2_cluster_plan"
                  << " W=" << W
                  << " max_cluster=" << max_cluster
                  << " reserve=" << reserve
                  << " max_fused_outer_ones=" << max_o
                  << " fused_state_fraction=" << f
                  << " ideal_HBM_reduction_vs_two_pass=" << 0.5 * f
                  << " capacity_distribution_equals_forward=1\n";
    }
    return 0;
}
