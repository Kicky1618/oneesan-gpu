#pragma push_macro("main")
#undef main
#define main two_cell_fusion2_sectorcache_microprobe_main_unused
#include "two_cell_fusion2_sectorcache_microprobe.cu"
#pragma pop_macro("main")

#include <cooperative_groups.h>

namespace {
namespace cg = cooperative_groups;

__device__ __forceinline__ std::uint32_t dsm_load_rank(
    cg::cluster_group cluster,
    std::uint32_t* local_values,
    Rank rank,
    Rank chunk
) {
    const int owner = static_cast<int>(rank / chunk);
    const Rank offset = rank - Rank(owner) * chunk;
    std::uint32_t* remote = cluster.map_shared_rank(local_values, owner);
    return remote[offset];
}

__device__ __forceinline__ void dsm_store_rank(
    cg::cluster_group cluster,
    std::uint32_t* local_values,
    Rank rank,
    Rank chunk,
    std::uint32_t value
) {
    const int owner = static_cast<int>(rank / chunk);
    const Rank offset = rank - Rank(owner) * chunk;
    std::uint32_t* remote = cluster.map_shared_rank(local_values, owner);
    remote[offset] = value;
}

__global__ void two_cell_fusion2_cluster_dsm_kernel(
    std::uint32_t* __restrict__ values,
    int W,
    int start,
    int outer_ones,
    Rank support_count,
    Rank state_count,
    std::uint32_t mod,
    int* error
) {
    extern __shared__ std::uint32_t local_values[];
    __shared__ PackedKey sh_src[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ PackedWord sh_label[FUSION_WARPS];
    __shared__ Rank sh_label_primitive[FUSION_WARPS];
    __shared__ oneesan::twocell::FusionSector sh_sector[FUSION2_SECTORS];
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
    const Rank n = oneesan::twocell::fusion_block_size(
        FUSION_STEPS, outer_ones, TC_RANK_TABLES);
    const Rank nc = oneesan::twocell::fusion_component_count(
        FUSION_STEPS, outer_ones, TC_RANK_TABLES);
    const Rank chunk = (n + Rank(cluster_size) - 1) / Rank(cluster_size);
    const Rank local_begin = Rank(block_rank) * chunk;
    const Rank local_end = min(n, local_begin + chunk);

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

        // Each CTA owns one contiguous local-rank chunk. Intersect the <=20
        // stationary sectors with that chunk and fill only local shared memory.
        for (int q = 0; q < FUSION2_SECTORS; ++q) {
            const auto sec = sh_sector[q];
            if (!sec.valid) continue;
            const Rank lo = max(local_begin, sec.local_base);
            const Rank hi = min(local_end, sec.local_base + sec.count);
            for (Rank lr = lo + threadIdx.x; lr < hi; lr += blockDim.x) {
                local_values[lr - local_begin] =
                    values[sec.global_base + (lr - sec.local_base)];
            }
        }

        // All block-local shared partitions must exist before any remote DSM
        // pointer is dereferenced.
        cluster.sync();

        for (int phase = 0; phase < FUSION_STEPS; ++phase) {
            const int active = start + phase;
            const int cluster_warp = block_rank * FUSION_WARPS + warp;
            const int cluster_warps = cluster_size * FUSION_WARPS;

            for (Rank cr = cluster_warp; cr < nc; cr += cluster_warps) {
                if (lane == 0) {
                    sh_ns[warp] = 0;
                    sh_partner_rounds[warp] = 0;
                    const auto label = oneesan::twocell::fusion_component_unrank(
                        cr, outer, W, start, FUSION_STEPS, TC_RANK_TABLES);
                    if (!label.valid) {
                        sh_deep[warp] = 0;
                        set_error(error, 461);
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
                } else if (lane == 0) {
                    const auto src = oneesan::twocell::direct_component_sources(
                        sh_label[warp], W, active);
                    if (src.overflow || src.size <= 0 ||
                        src.size > FUSION_MAX_COMPONENT) {
                        set_error(error, 462);
                    } else {
                        sh_ns[warp] = src.size;
                        for (int s = 0; s < src.size; ++s)
                            sh_src[warp][s] = src.value[s];
                    }
                }
                __syncwarp();

                const int ns = sh_ns[warp];
                if (ns <= 0 || ns > FUSION_MAX_COMPONENT) {
                    if (lane == 0) set_error(error, 463);
                    __syncwarp();
                    continue;
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
                    if (!reuse_label) {
                        const int len = source.type ? W - 2 : W - 1;
                        primitive = oneesan::twocell::primitive_rank(
                            source.support, source.left, len, TC_RANK_TABLES);
                    }
                    local_rank = oneesan::twocell::fusion_local_rank_at_with_primitive(
                        source, start, FUSION_STEPS, active, outer_ones,
                        primitive, TC_RANK_TABLES);
                    if (local_rank >= n) {
                        set_error(error, 464);
                    } else {
                        x = dsm_load_rank(cluster, local_values, local_rank, chunk);
                    }
                }
                __syncwarp();

                const std::uint32_t y =
                    oneesan::twocell::cuda_component::apply_closed_component_warp(
                        sh_label[warp], source, ns, W, active, x, mod, error);
                __syncwarp();

                if (lane < ns && local_rank < n)
                    dsm_store_rank(cluster, local_values, local_rank, chunk, y);
                __syncwarp();
            }

            // Every K_active component is disjoint within the phase, but phase
            // active+1 may consume values produced by any CTA in this cluster.
            cluster.sync();
        }

        // Store only the chunk owned by this CTA. Stationary sector global bases
        // are the same at the beginning and end of the fused segment.
        for (int q = 0; q < FUSION2_SECTORS; ++q) {
            const auto sec = sh_sector[q];
            if (!sec.valid) continue;
            const Rank lo = max(local_begin, sec.local_base);
            const Rank hi = min(local_end, sec.local_base + sec.count);
            for (Rank lr = lo + threadIdx.x; lr < hi; lr += blockDim.x) {
                values[sec.global_base + (lr - sec.local_base)] =
                    local_values[lr - local_begin];
            }
        }

        // No block may recycle its shared partition for the next outer support
        // until all remote DSM operations in the cluster are complete.
        cluster.sync();
    }
}

int choose_cluster_for_states(
    Rank states,
    Rank shared_limit,
    Rank static_reserve,
    int max_cluster
) {
    for (int c : {1, 2, 4, 8}) {
        if (c > max_cluster) break;
        const Rank chunk = (states + Rank(c) - 1) / Rank(c);
        if (chunk * sizeof(std::uint32_t) + static_reserve <= shared_limit)
            return c;
    }
    return 0;
}

void launch_cluster_bucket(
    std::uint32_t* d_values,
    int W,
    int start,
    int outer_ones,
    Rank support_count,
    Rank states,
    Rank shared_limit,
    int cluster_size,
    std::uint32_t mod,
    int* d_error
) {
    const Rank n = oneesan::twocell::fusion_block_size(
        FUSION_STEPS, outer_ones, oneesan::twocell::make_rank_tables());
    const Rank chunk = (n + Rank(cluster_size) - 1) / Rank(cluster_size);
    const Rank dynamic_bytes = chunk * sizeof(std::uint32_t);
    if (dynamic_bytes > shared_limit) {
        std::cerr << "cluster dynamic shared exceeds requested limit\n";
        std::exit(465);
    }

    ck(cudaFuncSetAttribute(
           two_cell_fusion2_cluster_dsm_kernel,
           cudaFuncAttributeMaxDynamicSharedMemorySize,
           static_cast<int>(dynamic_bytes)),
       "cluster DSM optin shared");

    const Rank clusters = std::max<Rank>(1, std::min<Rank>(support_count, 256));
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(static_cast<unsigned>(clusters * cluster_size), 1, 1);
    config.blockDim = dim3(FUSION_THREADS, 1, 1);
    config.dynamicSmemBytes = static_cast<std::size_t>(dynamic_bytes);

    cudaLaunchAttribute attr[1]{};
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = cluster_size;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    config.attrs = attr;
    config.numAttrs = 1;

    cudaLaunchKernelEx(
        &config,
        two_cell_fusion2_cluster_dsm_kernel,
        d_values, W, start, outer_ones, support_count, states, mod, d_error);
    ck(cudaGetLastError(), "cluster DSM launch");
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 8;
    const int cluster_size = argc > 2 ? std::atoi(argv[2]) : 2;
    const Rank shared_kib = argc > 3
        ? static_cast<Rank>(std::strtoull(argv[3], nullptr, 10)) : 228ULL;
    const std::uint32_t mod = argc > 4
        ? static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 10))
        : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth ||
        (cluster_size != 1 && cluster_size != 2 &&
         cluster_size != 4 && cluster_size != 8)) return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    const Rank shared_limit = shared_kib * 1024ULL;

    if (plan_only) {
        std::cout << "two-cell-fusion2-cluster-dsm-plan"
                  << " W=" << W
                  << " requested_cluster=" << cluster_size
                  << " per_cta_shared_limit=" << shared_limit
                  << " DSM_capacity_bytes=" << shared_limit * Rank(cluster_size)
                  << " max_copy_sectors=20"
                  << " remote_rank_address=owner+offset"
                  << " cluster_barriers_per_union_block=4"
                  << "\n";
        return 0;
    }

    if (W > 10) {
        std::cerr << "execution mode intentionally limited to W<=10; use --plan-only above that\n";
        return 3;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cluster DSM device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "cluster DSM set device");
    install_tables(rt);
    install_stationary_tables(st);

    const Rank states = st.total[W];
    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + 467ULL) % (mod - 1ULL)));

    for (int start = 1; start + 1 <= W - 4; start += 2) {
        const auto reference = fusion2_reference(input, W, start, rt, st, mod);
        std::uint32_t* d_values = nullptr;
        int* d_error = nullptr;
        ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)), "cluster DSM alloc values");
        ck(cudaMalloc(&d_error, sizeof(int)), "cluster DSM alloc error");
        ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice), "cluster DSM copy input");
        ck(cudaMemset(d_error, 0, sizeof(int)), "cluster DSM zero error");

        const int outer_bits = W - 5;
        for (int o = 0; o <= outer_bits; ++o) {
            const Rank support_count = rt.choose[outer_bits][o];
            launch_cluster_bucket(
                d_values, W, start, o, support_count, states,
                shared_limit, cluster_size, mod, d_error);
        }
        ck(cudaDeviceSynchronize(), "cluster DSM sync");

        std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
        int error = 0;
        ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost), "cluster DSM copy output");
        ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
           "cluster DSM copy error");
        if (error || output != reference) {
            std::cerr << "FAIL cluster DSM W=" << W
                      << " start=" << start
                      << " cluster=" << cluster_size
                      << " error=" << error << '\n';
            return 5;
        }
        cudaFree(d_values);
        cudaFree(d_error);
        std::cout << "two-cell-fusion2-cluster-dsm"
                  << " W=" << W
                  << " start=" << start
                  << " cluster=" << cluster_size
                  << " arithmetic=OK\n";
    }

    std::cout << "ALL_OK two_cell_fusion2_cluster_dsm=1 W=" << W
              << " cluster=" << cluster_size << '\n';
    return 0;
}
