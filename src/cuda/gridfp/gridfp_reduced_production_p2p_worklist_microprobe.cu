#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_ownerfirst_ab_microprobe_main_unused
#include "gridfp_reduced_production_p2p_ownerfirst_ab_microprobe.cu"
#pragma pop_macro("main")

namespace {

__global__ void p2p_worklist_count_kernel(
    Rank64 base_supports,
    int W,
    int K,
    bool reverse,
    int ngpu,
    unsigned long long* __restrict__ owner_counts,
    int* error
) {
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);

    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const EqualTileRunSeed run = seeds[ri];
            const bool blocked = run.blocked != 0;
            const int cycle_len = shift_cycle_leader_length_device(
                run.support, blocked, W, q, K, K, reverse);
            if (cycle_len < 0 || cycle_len > RP_MAX_W) {
                atomicCAS(error, 0, 331);
                continue;
            }
            if (cycle_len <= 1) continue;
            const int exec = p2p_modal_owner_only_device<P2P_MAX_GPU>(
                run.support, blocked, cycle_len, W, q, K, K,
                reverse, old_start, ngpu);
            if (exec < 0 || exec >= ngpu) {
                atomicCAS(error, 0, 332);
                continue;
            }
            atomicAdd(owner_counts + exec, 1ULL);
        }
    }
}

__global__ void p2p_worklist_fill_kernel(
    Rank64 base_supports,
    int W,
    int K,
    bool reverse,
    int ngpu,
    const unsigned long long* __restrict__ owner_offset,
    unsigned long long* __restrict__ owner_cursor,
    std::uint32_t* __restrict__ descriptors,
    int* error
) {
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);

    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const EqualTileRunSeed run = seeds[ri];
            const bool blocked = run.blocked != 0;
            const int cycle_len = shift_cycle_leader_length_device(
                run.support, blocked, W, q, K, K, reverse);
            if (cycle_len < 0 || cycle_len > RP_MAX_W) {
                atomicCAS(error, 0, 333);
                continue;
            }
            if (cycle_len <= 1) continue;
            const int exec = p2p_modal_owner_only_device<P2P_MAX_GPU>(
                run.support, blocked, cycle_len, W, q, K, K,
                reverse, old_start, ngpu);
            if (exec < 0 || exec >= ngpu) {
                atomicCAS(error, 0, 334);
                continue;
            }
            const unsigned long long local = atomicAdd(owner_cursor + exec, 1ULL);
            const unsigned long long slot = owner_offset[exec] + local;
            descriptors[slot] = p2p_pack_work_descriptor_device(base_rank, ri);
        }
    }
}

__global__ void p2p_worklist_execute_kernel(
    std::uint32_t** __restrict__ peer_state,
    const std::uint32_t* __restrict__ descriptors,
    Rank64 descriptor_count,
    int W,
    int K,
    bool reverse,
    int ngpu,
    int gpu_id,
    const Rank64* __restrict__ owner_begin,
    unsigned long long* __restrict__ local_cycles,
    unsigned long long* __restrict__ remote_value_ops,
    int* error
) {
    __shared__ Rank64 sh_local[WARPS_PER_BLOCK][RP_MAX_W];
    __shared__ int sh_owner[WARPS_PER_BLOCK][RP_MAX_W];
    __shared__ int sh_len[WARPS_PER_BLOCK];
    __shared__ Rank64 sh_pc[WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);

    for (Rank64 work_rank = first; work_rank < descriptor_count; work_rank += stride) {
        if (lane == 0) {
            sh_len[warp] = 0;
            sh_pc[warp] = 0;
            const std::uint32_t desc = descriptors[work_rank];
            const Rank64 base_rank = p2p_work_descriptor_base_device(desc);
            const int ri = p2p_work_descriptor_ri_device(desc);
            EqualTileRunSeed seeds[3]{};
            const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
            if (ri < 0 || ri >= nr) {
                atomicCAS(error, 0, 341);
            } else {
                const EqualTileRunSeed run = seeds[ri];
                const bool blocked = run.blocked != 0;
                const int cycle_len = shift_cycle_leader_length_device(
                    run.support, blocked, W, q, K, K, reverse);
                if (cycle_len <= 1 || cycle_len > RP_MAX_W) {
                    atomicCAS(error, 0, 342);
                } else {
                    int counts[P2P_MAX_GPU]{};
                    std::uint32_t cur = run.support;
                    for (int h = 0; h < cycle_len; ++h) {
                        const DeviceKey key = equal_run_key0_device(
                            cur, blocked, W, q, reverse);
                        const GroupedDeviceRank gr = grouped_rank_device(
                            key, W, q, reverse, old_start, K, ngpu, owner_begin);
                        if (gr.owner < 0 || gr.owner >= ngpu) {
                            atomicCAS(error, 0, 343);
                            break;
                        }
                        sh_owner[warp][h] = gr.owner;
                        sh_local[warp][h] = gr.local;
                        ++counts[gr.owner];
                        cur = shift_next_support_device(
                            cur, blocked, W, q, K, K, reverse);
                    }
                    int exec = 0;
                    for (int g = 1; g < ngpu; ++g)
                        if (counts[g] > counts[exec]) exec = g;
                    if (exec != gpu_id) {
                        atomicCAS(error, 0, 344);
                    } else {
                        sh_len[warp] = cycle_len;
                        sh_pc[warp] = RP_PRIMITIVE[__popc(run.support)][1];
                    }
                }
            }
        }
        __syncwarp();

        const int len = sh_len[warp];
        if (len > 1) {
            const Rank64 pc = sh_pc[warp];
            for (Rank64 i = Rank64(lane); i < pc; i += 32) {
                std::uint32_t temp =
                    peer_state[sh_owner[warp][0]][sh_local[warp][0] + i];
                for (int h = 1; h < len; ++h) {
                    std::uint32_t* ptr =
                        peer_state[sh_owner[warp][h]] + sh_local[warp][h] + i;
                    const std::uint32_t next = *ptr;
                    *ptr = temp;
                    temp = next;
                }
                peer_state[sh_owner[warp][0]][sh_local[warp][0] + i] = temp;
            }
            __syncwarp();
            if (lane == 0) {
                unsigned remote_runs = 0;
                for (int h = 0; h < len; ++h)
                    remote_runs += sh_owner[warp][h] != gpu_id;
                atomicAdd(local_cycles, 1ULL);
                atomicAdd(remote_value_ops,
                          2ULL * static_cast<unsigned long long>(remote_runs) *
                          static_cast<unsigned long long>(pc));
            }
        }
        __syncwarp();
    }
}

struct DeviceWorkCtx : DevicePeerCtx {
    std::uint32_t* work = nullptr;
    Rank64 work_count = 0;
};

void run_p2p_worklist_probe(
    int W, int K, bool reverse, int ngpu, unsigned blocks
) {
    ProductionFactorTables tables(W);
    const HostTilePlan plan = make_host_tile_plan(tables, K, ngpu);
    std::vector<std::uint32_t> flat_input, flat_expected;
    build_shift_boundary_vectors(
        W, K, K, reverse, ngpu, tables, plan, flat_input, flat_expected);

    enable_all_peer_access(ngpu);
    std::vector<DeviceWorkCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> peer_ptr(static_cast<std::size_t>(ngpu));
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "worklist alloc set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)), "worklist alloc state");
        peer_ptr[static_cast<std::size_t>(g)] = c.state;
        const Rank64 base = plan.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(c.state, flat_input.data() + base, n * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice), "worklist copy state");
        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)), "worklist alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                      cudaMemcpyHostToDevice), "worklist copy owner begin");
        ck(cudaMalloc(&c.cycles, sizeof(unsigned long long)), "worklist alloc cycles");
        ck(cudaMalloc(&c.cross_values, sizeof(unsigned long long)), "worklist alloc remote ops");
        ck(cudaMalloc(&c.error, sizeof(int)), "worklist alloc error");
        ck(cudaMemset(c.cycles, 0, sizeof(unsigned long long)), "worklist zero cycles");
        ck(cudaMemset(c.cross_values, 0, sizeof(unsigned long long)), "worklist zero remote ops");
        ck(cudaMemset(c.error, 0, sizeof(int)), "worklist zero error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "worklist peer table set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.peer, ngpu * sizeof(std::uint32_t*)), "worklist alloc peer table");
        ck(cudaMemcpy(c.peer, peer_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "worklist copy peer table");
    }

    // Build once on GPU 0.  The resulting owner-partitioned u32 descriptor
    // slices are tiny compared with the state stream and are reusable for every
    // subsequent row/residue with the same W/K/ngpu layout.
    ck(cudaSetDevice(0), "worklist builder set device");
    install_tables(tables);
    unsigned long long* d_counts = nullptr;
    unsigned long long* d_offsets = nullptr;
    unsigned long long* d_cursor = nullptr;
    int* d_build_error = nullptr;
    ck(cudaMalloc(&d_counts, ngpu * sizeof(unsigned long long)), "worklist alloc counts");
    ck(cudaMalloc(&d_offsets, ngpu * sizeof(unsigned long long)), "worklist alloc offsets");
    ck(cudaMalloc(&d_cursor, ngpu * sizeof(unsigned long long)), "worklist alloc cursor");
    ck(cudaMalloc(&d_build_error, sizeof(int)), "worklist alloc build error");
    ck(cudaMemset(d_counts, 0, ngpu * sizeof(unsigned long long)), "worklist zero counts");
    ck(cudaMemset(d_build_error, 0, sizeof(int)), "worklist zero build error");

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const unsigned build_threads = 256;
    const Rank64 build_one_pass = (base_supports + build_threads - 1) / build_threads;
    const unsigned build_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(blocks, build_one_pass)));
    cudaEvent_t ba{}, bb{}, bc{};
    ck(cudaEventCreate(&ba), "worklist build event a");
    ck(cudaEventCreate(&bb), "worklist build event b");
    ck(cudaEventCreate(&bc), "worklist build event c");
    ck(cudaEventRecord(ba), "worklist record build a");
    p2p_worklist_count_kernel<<<build_blocks, build_threads>>>(
        base_supports, W, K, reverse, ngpu, d_counts, d_build_error);
    ck(cudaGetLastError(), "worklist count launch");
    ck(cudaEventRecord(bb), "worklist record build b");
    ck(cudaEventSynchronize(bb), "worklist count sync");

    std::vector<unsigned long long> h_counts(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> h_offsets(static_cast<std::size_t>(ngpu));
    ck(cudaMemcpy(h_counts.data(), d_counts, ngpu * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost), "worklist copy counts");
    unsigned long long total_work = 0;
    for (int g = 0; g < ngpu; ++g) {
        h_offsets[static_cast<std::size_t>(g)] = total_work;
        total_work += h_counts[static_cast<std::size_t>(g)];
    }
    ck(cudaMemcpy(d_offsets, h_offsets.data(), ngpu * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice), "worklist copy offsets");
    ck(cudaMemset(d_cursor, 0, ngpu * sizeof(unsigned long long)), "worklist zero cursor");
    std::uint32_t* d_all_work = nullptr;
    ck(cudaMalloc(&d_all_work, total_work * sizeof(std::uint32_t)), "worklist alloc descriptors");
    p2p_worklist_fill_kernel<<<build_blocks, build_threads>>>(
        base_supports, W, K, reverse, ngpu, d_offsets, d_cursor,
        d_all_work, d_build_error);
    ck(cudaGetLastError(), "worklist fill launch");
    ck(cudaEventRecord(bc), "worklist record build c");
    ck(cudaEventSynchronize(bc), "worklist fill sync");
    float count_ms = 0.0f, fill_ms = 0.0f;
    ck(cudaEventElapsedTime(&count_ms, ba, bb), "worklist count time");
    ck(cudaEventElapsedTime(&fill_ms, bb, bc), "worklist fill time");
    int build_error = 0;
    ck(cudaMemcpy(&build_error, d_build_error, sizeof(build_error),
                  cudaMemcpyDeviceToHost), "worklist copy build error");
    if (build_error) fail("worklist builder device error=" + std::to_string(build_error));

    for (int g = 0; g < ngpu; ++g) {
        auto& c = ctx[static_cast<std::size_t>(g)];
        c.work_count = h_counts[static_cast<std::size_t>(g)];
        ck(cudaSetDevice(g), "worklist local list set device");
        ck(cudaMalloc(&c.work, c.work_count * sizeof(std::uint32_t)),
           "worklist alloc local descriptors");
        if (c.work_count) {
            ck(cudaMemcpyPeer(
                   c.work, g,
                   d_all_work + h_offsets[static_cast<std::size_t>(g)], 0,
                   c.work_count * sizeof(std::uint32_t)),
               "worklist distribute descriptors");
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "worklist execute set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 one_pass =
            (c.work_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned launch_blocks = static_cast<unsigned>(
            std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass)));
        p2p_worklist_execute_kernel<<<launch_blocks, THREADS>>>(
            c.peer, c.work, c.work_count, W, K, reverse, ngpu, g,
            c.owner_begin, c.cycles, c.cross_values, c.error);
        ck(cudaGetLastError(), "worklist execute launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "worklist execute sync set device");
        ck(cudaDeviceSynchronize(), "worklist execute sync");
    }
    const double exec_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> flat_output(flat_expected.size());
    unsigned long long cycles = 0, remote_ops = 0, counted_work = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "worklist gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0; unsigned long long gc = 0, gr = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "worklist copy error");
        ck(cudaMemcpy(&gc, c.cycles, sizeof(gc), cudaMemcpyDeviceToHost),
           "worklist copy cycles");
        ck(cudaMemcpy(&gr, c.cross_values, sizeof(gr), cudaMemcpyDeviceToHost),
           "worklist copy remote ops");
        if (error) fail("worklist execute device error=" + std::to_string(error));
        cycles += gc;
        remote_ops += gr;
        counted_work += c.work_count;
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        const Rank64 base = plan.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(flat_output.data() + base, c.state,
                      n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "worklist gather state");
    }
    if (counted_work != total_work || cycles != total_work)
        fail("worklist executed descriptor count mismatch");
    if (flat_output != flat_expected) fail("worklist redistribution mismatch");

    std::cout << "gridfp-reduced-production-p2p-worklist"
              << " W=" << W << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu << " states=" << tables.size()
              << " descriptors=" << total_work
              << " descriptor_bytes=" << total_work * sizeof(std::uint32_t)
              << " build_count_ms=" << count_ms
              << " build_fill_ms=" << fill_ms
              << " execute_wall_ms=" << exec_ms
              << " remote_u32_load_store_ops=" << remote_ops
              << " remote_GiB=" << double(remote_ops) * 4.0 / double(1ULL << 30)
              << " repeated_base_support_scan_per_row=0"
              << " modal_owner_recompute_per_row=0"
              << " reusable_worklist=1 staging_state_bytes=0 exact=OK\n";

    ck(cudaSetDevice(0), "worklist builder free set device");
    cudaFree(d_all_work); cudaFree(d_build_error); cudaFree(d_cursor);
    cudaFree(d_offsets); cudaFree(d_counts);
    cudaEventDestroy(bc); cudaEventDestroy(bb); cudaEventDestroy(ba);
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "worklist free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        cudaFree(c.work);
        cudaFree(c.error); cudaFree(c.cross_values); cudaFree(c.cycles);
        cudaFree(c.peer); cudaFree(c.owner_begin); cudaFree(c.state);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int K = argc > 2 ? std::atoi(argv[2]) : (W - 2) / 2;
    const unsigned blocks = argc > 3
        ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10)) : 256u;
    const int ngpu = argc > 4 ? std::atoi(argv[4]) : 2;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 8 || W > RP_MAX_W || (W & 1) || K != (W - 2) / 2 ||
        !blocks || ngpu < 2 || ngpu > P2P_MAX_GPU) return 2;

    if (plan_only) {
        if (W == 28 && K == 13) {
            constexpr unsigned long long descriptors = 21566612ULL;
            constexpr unsigned long long bytes = descriptors * sizeof(std::uint32_t);
            std::cout << "gridfp-reduced-production-p2p-worklist-plan"
                      << " W=28 K=13 ngpu=" << ngpu
                      << " descriptors=" << descriptors
                      << " descriptor_MiB=" << double(bytes) / double(1ULL << 20)
                      << " avg_descriptor_MiB_per_gpu="
                      << double(bytes) / double(ngpu) / double(1ULL << 20)
                      << " forward_reverse_both_MiB="
                      << 2.0 * double(bytes) / double(1ULL << 20)
                      << " descriptor_format=u32_base26_ri2"
                      << " build_once_reuse_rows_residues=1\n";
            return 0;
        }
        std::cerr << "plan-only closed-form is currently pinned for W=28 K=13\n";
        return 3;
    }
    if (W > 11) {
        std::cerr << "execution mode intentionally limited to W<=11; use --plan-only for W=28\n";
        return 4;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "worklist device count");
    if (visible < ngpu) return 5;

    run_p2p_worklist_probe(W, K, false, ngpu, blocks);
    run_p2p_worklist_probe(W, K, true, ngpu, blocks);
    std::cout << "ALL_OK production_p2p_compact_worklist_cuda=1\n";
    return 0;
}
