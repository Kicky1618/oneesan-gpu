#pragma push_macro("main")
#undef main
#define main two_cell_fusion2_cluster_sliced_microprobe_main_unused
#include "two_cell_fusion2_cluster_sliced_microprobe.cu"
#pragma pop_macro("main")

namespace {

__global__ void two_cell_fusion2_cluster_sliced_lut_kernel(
    std::uint32_t* __restrict__ values,
    const std::uint32_t* __restrict__ primitive_lut,
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
    __shared__ PackedKey sh_src[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ PackedWord sh_label[FUSION_WARPS];
    __shared__ Rank sh_label_primitive[FUSION_WARPS];
    __shared__ oneesan::twocell::FusionSector sh_sector[FUSION2_SECTORS];
    __shared__ Rank sh_slice_base[MAX_CLUSTER_BLOCKS][FUSION2_SECTORS];
    __shared__ Rank sh_owner_total[MAX_CLUSTER_BLOCKS];
    __shared__ int sh_ns[FUSION_WARPS];
    __shared__ int sh_deep[FUSION_WARPS];
    __shared__ int sh_partner_rounds[FUSION_WARPS];
    __shared__ int sh_capacity_ok;

    cg::cluster_group cluster = cg::this_cluster();
    const int cluster_size = static_cast<int>(cluster.num_blocks());
    const int block_rank = static_cast<int>(cluster.block_rank());
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    // Collective cluster operations below require the whole CTA. Reject an
    // accidental launch configuration uniformly before entering any barrier.
    if (blockDim.x != FUSION_THREADS || cluster_size < 1 ||
        cluster_size > MAX_CLUSTER_BLOCKS) {
        if (threadIdx.x == 0) set_error(error, 481);
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
                    set_error(error, 482);
                }
            }
        }
        __syncthreads();
        if (!sh_capacity_ok) {
            // Every CTA computes the same owner totals for this outer block, so
            // every block in the cluster takes the same collective skip path.
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
            const int active = start + phase;

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
                const Rank lut_base = primitive_offset[occupied];

                for (Rank primitive = pbegin + warp;
                     primitive < pend; primitive += FUSION_WARPS) {
                    std::uint32_t compact_left = 0;
                    if (lane == 0)
                        compact_left = primitive_lut[lut_base + primitive];
                    compact_left = __shfl_sync(0xffffffffu, compact_left, 0);
                    const std::uint32_t left = deposit_left_warp(
                        label_support, compact_left, W - 2);

                    if (lane == 0) {
                        sh_ns[warp] = 0;
                        sh_partner_rounds[warp] = 0;
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
                            set_error(error, 483);
                        } else {
                            sh_ns[warp] = src.size;
                            for (int s = 0; s < src.size; ++s)
                                sh_src[warp][s] = src.value[s];
                        }
                    }
                    __syncwarp();

                    const int ns = sh_ns[warp];
                    if (ns <= 0 || ns > FUSION_MAX_COMPONENT) {
                        if (lane == 0) set_error(error, 484);
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
                            set_error(error, 485);
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

Rank primitive_lut_entries_for_width(
    int W,
    const RankTables& rt
) {
    Rank z = 0;
    for (int occupied = 1; occupied <= W - 2; occupied += 2)
        z += rt.primitive[occupied][1];
    return z;
}

void print_cluster_lut_plan(
    int W,
    const RankTables& rt
) {
    const Rank entries = primitive_lut_entries_for_width(W, rt);
    std::cout << "fusion2_cluster_sliced_lut_plan"
              << " W=" << W
              << " primitive_lut_entries=" << entries
              << " primitive_lut_MiB="
              << double(entries * sizeof(std::uint32_t)) / double(1ULL << 20)
              << " component_label_primitive_unrank_calls=0"
              << " label_left_deposit=warp_ballot"
              << " capacity_guard=cluster_collective\n";
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    if (W < 6 || W > oneesan::twocell::kMaxWidth) return 2;
    const RankTables rt = oneesan::twocell::make_rank_tables();
    print_cluster_lut_plan(W, rt);
    return 0;
}
