#pragma push_macro("main")
#undef main
#define main two_cell_fusion2_cluster_dsm_microprobe_main_unused
#include "two_cell_fusion2_cluster_dsm_microprobe.cu"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_slices.hpp"

namespace {

constexpr int MAX_CLUSTER_BLOCKS = 8;

__device__ __forceinline__ std::uint32_t dsm_load_slice(
    cg::cluster_group cluster,
    std::uint32_t* local_values,
    const oneesan::twocell::FusionSector* sectors,
    const Rank slice_base[MAX_CLUSTER_BLOCKS][FUSION2_SECTORS],
    int sector,
    Rank primitive
) {
    const int owners = static_cast<int>(cluster.num_blocks());
    const Rank count = sectors[sector].count;
    const int owner = oneesan::twocell::primitive_slice_owner(
        primitive, count, owners);
    const Rank begin = oneesan::twocell::primitive_slice_begin(
        count, owner, owners);
    const Rank offset = slice_base[owner][sector] + primitive - begin;
    std::uint32_t* remote = cluster.map_shared_rank(local_values, owner);
    return remote[offset];
}

__device__ __forceinline__ void dsm_store_slice(
    cg::cluster_group cluster,
    std::uint32_t* local_values,
    const oneesan::twocell::FusionSector* sectors,
    const Rank slice_base[MAX_CLUSTER_BLOCKS][FUSION2_SECTORS],
    int sector,
    Rank primitive,
    std::uint32_t value
) {
    const int owners = static_cast<int>(cluster.num_blocks());
    const Rank count = sectors[sector].count;
    const int owner = oneesan::twocell::primitive_slice_owner(
        primitive, count, owners);
    const Rank begin = oneesan::twocell::primitive_slice_begin(
        count, owner, owners);
    const Rank offset = slice_base[owner][sector] + primitive - begin;
    std::uint32_t* remote = cluster.map_shared_rank(local_values, owner);
    remote[offset] = value;
}

__global__ void two_cell_fusion2_cluster_sliced_kernel(
    std::uint32_t* __restrict__ values,
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
    __shared__ PackedKey sh_src[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ PackedWord sh_label[FUSION_WARPS];
    __shared__ Rank sh_label_primitive[FUSION_WARPS];
    __shared__ oneesan::twocell::FusionSector sh_sector[FUSION2_SECTORS];
    __shared__ Rank sh_slice_base[MAX_CLUSTER_BLOCKS][FUSION2_SECTORS];
    __shared__ Rank sh_owner_total[MAX_CLUSTER_BLOCKS];
    __shared__ int sh_ns[FUSION_WARPS];
    __shared__ int sh_deep[FUSION_WARPS];
    __shared__ int sh_partner_rounds[FUSION_WARPS];

    cg::cluster_group cluster = cg::this_cluster();
    const int cluster_size = static_cast<int>(cluster.num_blocks());
    const int block_rank = static_cast<int>(cluster.block_rank());
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (warp >= FUSION_WARPS) return;

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

        // Build every owner's local sector prefix. Only <=8 threads perform
        // this work, once per outer-support block.
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
        if (sh_owner_total[block_rank] > local_capacity) {
            if (threadIdx.x == 0) set_error(error, 471);
        }

        // Each CTA owns a contiguous primitive interval inside every stationary
        // sector. Global reads stay contiguous within that interval.
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
            const int active = start + phase;

            // Enumerate component labels directly by their 3-bit local support
            // and the primitive interval owned by this CTA. This avoids decoding
            // and rejecting component ranks assigned to other CTAs.
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
                    if (lane == 0) {
                        sh_ns[warp] = 0;
                        sh_partner_rounds[warp] = 0;
                        const std::uint32_t left =
                            oneesan::twocell::primitive_left_unrank(
                                label_support, W - 2, occupied, primitive,
                                TC_RANK_TABLES);
                        sh_label[warp] = PackedWord{
                            label_support, left, static_cast<std::uint8_t>(W - 2)};
                        sh_label_primitive[warp] = primitive;
                        PackedWord collapsed{};
                        sh_deep[warp] = oneesan::twocell::deep_collapse(
                            sh_label[warp], active, collapsed) ? 1 : 0;
                    }
                    __syncwarp();

                    if (sh_deep[warp]) {
                        oneesan::twocell::cuda_face::deep_component_sources_compact(
                            sh_label[warp], W, active,
                            sh_src[warp], &sh_ns[warp],
                            &sh_partner_rounds[warp], error);
                        __syncwarp();
                    } else if (lane == 0) {
                        const auto src = oneesan::twocell::direct_component_sources(
                            sh_label[warp], W, active);
                        if (src.overflow || src.size <= 0 ||
                            src.size > FUSION_MAX_COMPONENT) {
                            set_error(error, 472);
                        } else {
                            sh_ns[warp] = src.size;
                            for (int s = 0; s < src.size; ++s)
                                sh_src[warp][s] = src.value[s];
                        }
                    }
                    __syncwarp();

                    const int ns = sh_ns[warp];
                    if (ns <= 0 || ns > FUSION_MAX_COMPONENT) {
                        if (lane == 0) set_error(error, 473);
                        __syncwarp();
                        continue;
                    }

                    std::uint32_t x = 0;
                    PackedKey source{};
                    Rank source_primitive = 0;
                    int sector = -1;
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

                        source_primitive = sh_label_primitive[warp];
                        if (!reuse_label) {
                            const int len = source.type ? W - 2 : W - 1;
                            source_primitive = oneesan::twocell::primitive_rank(
                                source.support, source.left, len, TC_RANK_TABLES);
                        }
                        sector = oneesan::twocell::fusion_sector_index_at(
                            source, start, FUSION_STEPS, active);
                        if (sector < 0 || sector >= FUSION2_SECTORS ||
                            !sh_sector[sector].valid) {
                            set_error(error, 474);
                        } else {
                            x = dsm_load_slice(
                                cluster, local_values, sh_sector, sh_slice_base,
                                sector, source_primitive);
                        }
                    }
                    __syncwarp();

                    const std::uint32_t y =
                        oneesan::twocell::cuda_component::apply_closed_component_warp(
                            sh_label[warp], source, ns, W, active, x, mod, error);
                    __syncwarp();

                    if (lane < ns && sector >= 0 && sector < FUSION2_SECTORS)
                        dsm_store_slice(
                            cluster, local_values, sh_sector, sh_slice_base,
                            sector, source_primitive, y);
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

Rank fusion2_slice_owner_states(
    int outer_ones,
    int owner,
    int owners,
    const RankTables& rt
) {
    Rank total = 0;
    for (std::uint32_t code = 0; code < 16; ++code) {
        const Rank count = oneesan::twocell::primitive_count_for_occupied(
            outer_ones + oneesan::twocell::popcount32(code), rt);
        total += oneesan::twocell::primitive_slice_count(
            count, owner, owners);
    }
    for (std::uint32_t code = 0; code < 4; ++code) {
        const Rank count = oneesan::twocell::primitive_count_for_occupied(
            outer_ones + 1 + oneesan::twocell::popcount32(code), rt);
        total += oneesan::twocell::primitive_slice_count(
            count, owner, owners);
    }
    return total;
}

Rank fusion2_slice_max_owner_states(
    int outer_ones,
    int owners,
    const RankTables& rt
) {
    Rank z = 0;
    for (int owner = 0; owner < owners; ++owner)
        z = std::max(z, fusion2_slice_owner_states(
            outer_ones, owner, owners, rt));
    return z;
}

void print_cluster_slice_plan(
    int W,
    Rank shared_limit,
    Rank reserve,
    int max_cluster,
    const RankTables& rt
) {
    const int outer_bits = W - 5;
    Rank total = 0, fused = 0;
    int max_o = -1;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank blocks = rt.choose[outer_bits][o];
        const Rank n = oneesan::twocell::fusion_block_size(2, o, rt);
        total += blocks * n;
        int chosen = 0;
        for (int c : {1, 2, 4, 8}) {
            if (c > max_cluster) break;
            const Rank local = fusion2_slice_max_owner_states(o, c, rt);
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
    std::cout << "fusion2_cluster_slice_plan"
              << " W=" << W
              << " max_cluster=" << max_cluster
              << " per_cta_shared_limit=" << shared_limit
              << " reserve=" << reserve
              << " max_fused_outer_ones=" << max_o
              << " fused_state_fraction=" << f
              << " ideal_HBM_reduction_vs_two_pass=" << 0.5 * f
              << " primitive_sliced_DSM=1"
              << " component_owner=label_primitive_slice\n";
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
        std::cerr << "execution harness intentionally deferred; use --plan-only until primitive-sliced DSM is compiled on target cluster hardware\n";
        return 3;
    }
    for (int c : {1, 2, 4, 8})
        print_cluster_slice_plan(
            W, shared_kib * 1024ULL, reserve, c, rt);
    if (W == 28) {
        for (int o : {15, 16, 17, 18}) {
            std::cout << "outer=" << o;
            for (int c : {1, 2, 4, 8})
                std::cout << " c" << c << "_max_owner_states="
                          << fusion2_slice_max_owner_states(o, c, rt);
            std::cout << '\n';
        }
    }
    return 0;
}
