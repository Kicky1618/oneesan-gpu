#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_compiled_schedule_microprobe_main_unused
#include "gridfp_reduced_production_p2p_compiled_schedule_microprobe.cu"
#pragma pop_macro("main")

namespace {

__device__ __forceinline__ void p2p_pack_run39_device(
    int owner,
    Rank64 local,
    std::uint32_t& low,
    std::uint8_t& high
) {
    low = std::uint32_t(owner) |
          (std::uint32_t(local & ((Rank64(1) << 29) - 1)) << 3);
    high = static_cast<std::uint8_t>((local >> 29) & 0x7fu);
}

__device__ __forceinline__ Rank64 p2p_unpack_run39_local_device(
    std::uint32_t low,
    std::uint8_t high
) {
    return Rank64(low >> 3) | (Rank64(high & 0x7fu) << 29);
}

__global__ void p2p_packed_count_kernel(
    Rank64 base_supports,
    int W,
    int K,
    bool reverse,
    int ngpu,
    unsigned long long* __restrict__ cycle_counts,
    unsigned long long* __restrict__ run_counts,
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
            const EqualTileRunSeed root = seeds[ri];
            const bool blocked = root.blocked != 0;
            const int len = shift_cycle_leader_length_device(
                root.support, blocked, W, q, K, K, reverse);
            if (len < 0 || len > RP_MAX_W) {
                atomicCAS(error, 0, 381);
                continue;
            }
            if (len <= 1) continue;
            const int exec = p2p_modal_owner_only_device<P2P_MAX_GPU>(
                root.support, blocked, len, W, q, K, K,
                reverse, old_start, ngpu, true);
            if (exec < 0 || exec >= ngpu) {
                atomicCAS(error, 0, 382);
                continue;
            }
            atomicAdd(cycle_counts + exec, 1ULL);
            atomicAdd(run_counts + exec, static_cast<unsigned long long>(len));
        }
    }
}

__global__ void p2p_packed_fill_kernel(
    Rank64 base_supports,
    int W,
    int K,
    bool reverse,
    int ngpu,
    const unsigned long long* __restrict__ header_offset,
    const unsigned long long* __restrict__ run_offset,
    unsigned long long* __restrict__ cycle_cursor,
    unsigned long long* __restrict__ run_cursor,
    P2PCompiledHeader* __restrict__ header,
    std::uint32_t* __restrict__ run_low,
    std::uint8_t* __restrict__ run_high,
    const Rank64* __restrict__ owner_begin,
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
            const EqualTileRunSeed root = seeds[ri];
            const bool blocked = root.blocked != 0;
            const int len = shift_cycle_leader_length_device(
                root.support, blocked, W, q, K, K, reverse);
            if (len < 0 || len > RP_MAX_W) {
                atomicCAS(error, 0, 383);
                continue;
            }
            if (len <= 1) continue;
            const int exec = p2p_modal_owner_only_device<P2P_MAX_GPU>(
                root.support, blocked, len, W, q, K, K,
                reverse, old_start, ngpu, true);
            if (exec < 0 || exec >= ngpu) {
                atomicCAS(error, 0, 384);
                continue;
            }
            const unsigned long long ci = atomicAdd(cycle_cursor + exec, 1ULL);
            const unsigned long long rb = atomicAdd(
                run_cursor + exec, static_cast<unsigned long long>(len));
            if (rb + static_cast<unsigned long long>(len) > 0xffffffffULL) {
                atomicCAS(error, 0, 385);
                continue;
            }
            const Rank64 pc = RP_PRIMITIVE[__popc(root.support)][1];
            if (pc > P2P_COMPILED_PC_MASK) {
                atomicCAS(error, 0, 386);
                continue;
            }
            header[header_offset[exec] + ci] = P2PCompiledHeader{
                static_cast<std::uint32_t>(rb),
                static_cast<std::uint32_t>(pc) |
                    (static_cast<std::uint32_t>(len) << P2P_COMPILED_LEN_SHIFT)};

            std::uint32_t cur = root.support;
            for (int h = 0; h < len; ++h) {
                const DeviceKey key = equal_run_key0_device(
                    cur, blocked, W, q, reverse);
                const GroupedDeviceRank gr = grouped_rank_device(
                    key, W, q, reverse, old_start, K, ngpu, owner_begin);
                if (gr.owner < 0 || gr.owner >= ngpu || gr.local >= (Rank64(1) << 36)) {
                    atomicCAS(error, 0, 387);
                    break;
                }
                std::uint32_t low = 0;
                std::uint8_t high = 0;
                p2p_pack_run39_device(gr.owner, gr.local, low, high);
                const Rank64 slot = run_offset[exec] + rb + h;
                run_low[slot] = low;
                run_high[slot] = high;
                cur = shift_next_support_device(
                    cur, blocked, W, q, K, K, reverse);
            }
        }
    }
}

__global__ void p2p_packed_execute_kernel(
    std::uint32_t** __restrict__ peer_state,
    const P2PCompiledHeader* __restrict__ header,
    const std::uint32_t* __restrict__ run_low,
    const std::uint8_t* __restrict__ run_high,
    Rank64 cycles,
    Rank64 run_count,
    int gpu_id,
    unsigned long long* __restrict__ executed,
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
    for (Rank64 ci = first; ci < cycles; ci += stride) {
        if (lane == 0) {
            sh_len[warp] = 0;
            const P2PCompiledHeader a = header[ci];
            const int len = p2p_compiled_len(a);
            const Rank64 pc = p2p_compiled_pc(a);
            const Rank64 end = Rank64(a.run_begin) + Rank64(len);
            if (len < 2 || len > RP_MAX_W || !pc || end > run_count) {
                atomicCAS(error, 0, 391);
            } else {
                for (int h = 0; h < len; ++h) {
                    const Rank64 r = Rank64(a.run_begin) + h;
                    const std::uint32_t low = run_low[r];
                    sh_owner[warp][h] = int(low & 7u);
                    sh_local[warp][h] =
                        p2p_unpack_run39_local_device(low, run_high[r]);
                }
                sh_len[warp] = len;
                sh_pc[warp] = pc;
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
                atomicAdd(executed, 1ULL);
                atomicAdd(remote_value_ops,
                          2ULL * static_cast<unsigned long long>(remote_runs) *
                          static_cast<unsigned long long>(pc));
            }
        }
        __syncwarp();
    }
}

struct DevicePackedCtx : DevicePeerCtx {
    P2PCompiledHeader* header = nullptr;
    std::uint32_t* run_low = nullptr;
    std::uint8_t* run_high = nullptr;
    Rank64 cycle_count = 0;
    Rank64 run_count = 0;
};

void run_p2p_packed_schedule_probe(
    int W, int K, bool reverse, int ngpu, unsigned blocks
) {
    ProductionFactorTables tables(W);
    const HostTilePlan plan = make_host_tile_plan(tables, K, ngpu);
    std::vector<std::uint32_t> flat_input, flat_expected;
    build_shift_boundary_vectors(
        W, K, K, reverse, ngpu, tables, plan, flat_input, flat_expected);
    enable_all_peer_access(ngpu);

    std::vector<DevicePackedCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> peer_ptr(static_cast<std::size_t>(ngpu));
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "packed alloc set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)), "packed alloc state");
        peer_ptr[static_cast<std::size_t>(g)] = c.state;
        const Rank64 base = plan.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(c.state, flat_input.data() + base, n * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice), "packed copy state");
        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)), "packed alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                      cudaMemcpyHostToDevice), "packed copy owner begin");
        ck(cudaMalloc(&c.cycles, sizeof(unsigned long long)), "packed alloc executed");
        ck(cudaMalloc(&c.cross_values, sizeof(unsigned long long)), "packed alloc remote ops");
        ck(cudaMalloc(&c.error, sizeof(int)), "packed alloc error");
        ck(cudaMemset(c.cycles, 0, sizeof(unsigned long long)), "packed zero executed");
        ck(cudaMemset(c.cross_values, 0, sizeof(unsigned long long)), "packed zero remote ops");
        ck(cudaMemset(c.error, 0, sizeof(int)), "packed zero error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "packed peer table set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.peer, ngpu * sizeof(std::uint32_t*)), "packed alloc peer table");
        ck(cudaMemcpy(c.peer, peer_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "packed copy peer table");
    }

    ck(cudaSetDevice(0), "packed builder set device");
    install_tables(tables);
    unsigned long long *d_cycle_counts = nullptr, *d_run_counts = nullptr;
    unsigned long long *d_header_offset = nullptr, *d_run_offset = nullptr;
    unsigned long long *d_cycle_cursor = nullptr, *d_run_cursor = nullptr;
    int* d_build_error = nullptr;
    ck(cudaMalloc(&d_cycle_counts, ngpu * sizeof(unsigned long long)), "packed alloc cycle counts");
    ck(cudaMalloc(&d_run_counts, ngpu * sizeof(unsigned long long)), "packed alloc run counts");
    ck(cudaMalloc(&d_header_offset, ngpu * sizeof(unsigned long long)), "packed alloc header offsets");
    ck(cudaMalloc(&d_run_offset, ngpu * sizeof(unsigned long long)), "packed alloc run offsets");
    ck(cudaMalloc(&d_cycle_cursor, ngpu * sizeof(unsigned long long)), "packed alloc cycle cursor");
    ck(cudaMalloc(&d_run_cursor, ngpu * sizeof(unsigned long long)), "packed alloc run cursor");
    ck(cudaMalloc(&d_build_error, sizeof(int)), "packed alloc build error");
    ck(cudaMemset(d_cycle_counts, 0, ngpu * sizeof(unsigned long long)), "packed zero cycle counts");
    ck(cudaMemset(d_run_counts, 0, ngpu * sizeof(unsigned long long)), "packed zero run counts");
    ck(cudaMemset(d_build_error, 0, sizeof(int)), "packed zero build error");

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const unsigned build_threads = 256;
    const Rank64 build_one_pass = (base_supports + build_threads - 1) / build_threads;
    const unsigned build_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(blocks, build_one_pass)));
    cudaEvent_t e0{}, e1{}, e2{};
    ck(cudaEventCreate(&e0), "packed event 0");
    ck(cudaEventCreate(&e1), "packed event 1");
    ck(cudaEventCreate(&e2), "packed event 2");
    ck(cudaEventRecord(e0), "packed record 0");
    p2p_packed_count_kernel<<<build_blocks, build_threads>>>(
        base_supports, W, K, reverse, ngpu,
        d_cycle_counts, d_run_counts, d_build_error);
    ck(cudaGetLastError(), "packed count launch");
    ck(cudaEventRecord(e1), "packed record 1");
    ck(cudaEventSynchronize(e1), "packed count sync");

    std::vector<unsigned long long> h_cycle_counts(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> h_run_counts(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> h_header_offset(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> h_run_offset(static_cast<std::size_t>(ngpu));
    ck(cudaMemcpy(h_cycle_counts.data(), d_cycle_counts,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "packed copy cycle counts");
    ck(cudaMemcpy(h_run_counts.data(), d_run_counts,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "packed copy run counts");
    unsigned long long total_headers = 0, total_runs = 0;
    for (int g = 0; g < ngpu; ++g) {
        h_header_offset[static_cast<std::size_t>(g)] = total_headers;
        h_run_offset[static_cast<std::size_t>(g)] = total_runs;
        total_headers += h_cycle_counts[static_cast<std::size_t>(g)];
        total_runs += h_run_counts[static_cast<std::size_t>(g)];
    }
    ck(cudaMemcpy(d_header_offset, h_header_offset.data(),
                  ngpu * sizeof(unsigned long long), cudaMemcpyHostToDevice),
       "packed copy header offsets");
    ck(cudaMemcpy(d_run_offset, h_run_offset.data(),
                  ngpu * sizeof(unsigned long long), cudaMemcpyHostToDevice),
       "packed copy run offsets");
    ck(cudaMemset(d_cycle_cursor, 0, ngpu * sizeof(unsigned long long)),
       "packed zero cycle cursor");
    ck(cudaMemset(d_run_cursor, 0, ngpu * sizeof(unsigned long long)),
       "packed zero run cursor");

    P2PCompiledHeader* d_all_header = nullptr;
    std::uint32_t* d_all_low = nullptr;
    std::uint8_t* d_all_high = nullptr;
    ck(cudaMalloc(&d_all_header,
                  std::max<unsigned long long>(1, total_headers) * sizeof(P2PCompiledHeader)),
       "packed alloc all headers");
    ck(cudaMalloc(&d_all_low,
                  std::max<unsigned long long>(1, total_runs) * sizeof(std::uint32_t)),
       "packed alloc all low");
    ck(cudaMalloc(&d_all_high,
                  std::max<unsigned long long>(1, total_runs) * sizeof(std::uint8_t)),
       "packed alloc all high");
    p2p_packed_fill_kernel<<<build_blocks, build_threads>>>(
        base_supports, W, K, reverse, ngpu,
        d_header_offset, d_run_offset, d_cycle_cursor, d_run_cursor,
        d_all_header, d_all_low, d_all_high, ctx[0].owner_begin, d_build_error);
    ck(cudaGetLastError(), "packed fill launch");
    ck(cudaEventRecord(e2), "packed record 2");
    ck(cudaEventSynchronize(e2), "packed fill sync");
    float count_ms = 0.0f, fill_ms = 0.0f;
    ck(cudaEventElapsedTime(&count_ms, e0, e1), "packed count time");
    ck(cudaEventElapsedTime(&fill_ms, e1, e2), "packed fill time");
    int build_error = 0;
    ck(cudaMemcpy(&build_error, d_build_error, sizeof(build_error),
                  cudaMemcpyDeviceToHost), "packed copy build error");
    if (build_error) fail("packed builder device error=" + std::to_string(build_error));

    std::vector<unsigned long long> h_cycle_cursor(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> h_run_cursor(static_cast<std::size_t>(ngpu));
    ck(cudaMemcpy(h_cycle_cursor.data(), d_cycle_cursor,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "packed copy cycle cursor");
    ck(cudaMemcpy(h_run_cursor.data(), d_run_cursor,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "packed copy run cursor");
    for (int g = 0; g < ngpu; ++g) {
        if (h_cycle_cursor[static_cast<std::size_t>(g)] !=
                h_cycle_counts[static_cast<std::size_t>(g)] ||
            h_run_cursor[static_cast<std::size_t>(g)] !=
                h_run_counts[static_cast<std::size_t>(g)])
            fail("packed fill cursor/count mismatch");
    }

    for (int g = 0; g < ngpu; ++g) {
        auto& c = ctx[static_cast<std::size_t>(g)];
        c.cycle_count = h_cycle_counts[static_cast<std::size_t>(g)];
        c.run_count = h_run_counts[static_cast<std::size_t>(g)];
        ck(cudaSetDevice(g), "packed distribute set device");
        ck(cudaMalloc(&c.header,
                      std::max<Rank64>(1, c.cycle_count) * sizeof(P2PCompiledHeader)),
           "packed alloc local headers");
        ck(cudaMalloc(&c.run_low,
                      std::max<Rank64>(1, c.run_count) * sizeof(std::uint32_t)),
           "packed alloc local low");
        ck(cudaMalloc(&c.run_high,
                      std::max<Rank64>(1, c.run_count) * sizeof(std::uint8_t)),
           "packed alloc local high");
        const auto* hs = d_all_header + h_header_offset[static_cast<std::size_t>(g)];
        const auto* ls = d_all_low + h_run_offset[static_cast<std::size_t>(g)];
        const auto* xs = d_all_high + h_run_offset[static_cast<std::size_t>(g)];
        if (g == 0) {
            if (c.cycle_count)
                ck(cudaMemcpy(c.header, hs, c.cycle_count * sizeof(P2PCompiledHeader),
                              cudaMemcpyDeviceToDevice), "packed local header copy");
            if (c.run_count) {
                ck(cudaMemcpy(c.run_low, ls, c.run_count * sizeof(std::uint32_t),
                              cudaMemcpyDeviceToDevice), "packed local low copy");
                ck(cudaMemcpy(c.run_high, xs, c.run_count * sizeof(std::uint8_t),
                              cudaMemcpyDeviceToDevice), "packed local high copy");
            }
        } else {
            if (c.cycle_count)
                ck(cudaMemcpyPeer(c.header, g, hs, 0,
                                  c.cycle_count * sizeof(P2PCompiledHeader)),
                   "packed peer header copy");
            if (c.run_count) {
                ck(cudaMemcpyPeer(c.run_low, g, ls, 0,
                                  c.run_count * sizeof(std::uint32_t)),
                   "packed peer low copy");
                ck(cudaMemcpyPeer(c.run_high, g, xs, 0,
                                  c.run_count * sizeof(std::uint8_t)),
                   "packed peer high copy");
            }
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "packed execute set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 one_pass =
            (c.cycle_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned launch_blocks = static_cast<unsigned>(
            std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass)));
        p2p_packed_execute_kernel<<<launch_blocks, THREADS>>>(
            c.peer, c.header, c.run_low, c.run_high,
            c.cycle_count, c.run_count, g,
            c.cycles, c.cross_values, c.error);
        ck(cudaGetLastError(), "packed execute launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "packed execute sync set device");
        ck(cudaDeviceSynchronize(), "packed execute sync");
    }
    const double exec_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> flat_output(flat_expected.size());
    unsigned long long executed = 0, remote_ops = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "packed gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0; unsigned long long ge = 0, gr = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "packed copy error");
        ck(cudaMemcpy(&ge, c.cycles, sizeof(ge), cudaMemcpyDeviceToHost),
           "packed copy executed");
        ck(cudaMemcpy(&gr, c.cross_values, sizeof(gr), cudaMemcpyDeviceToHost),
           "packed copy remote ops");
        if (error) fail("packed execute device error=" + std::to_string(error));
        executed += ge; remote_ops += gr;
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        const Rank64 base = plan.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(flat_output.data() + base, c.state,
                      n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "packed gather state");
    }
    unsigned long long expected_cycles = 0;
    for (auto z : h_cycle_counts) expected_cycles += z;
    if (executed != expected_cycles) fail("packed executed cycle count mismatch");
    if (flat_output != flat_expected) fail("packed redistribution mismatch");

    const unsigned long long metadata_bytes =
        total_headers * sizeof(P2PCompiledHeader) + total_runs * 5ULL;
    std::cout << "gridfp-reduced-production-p2p-packed-schedule"
              << " W=" << W << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu << " states=" << tables.size()
              << " cycles=" << expected_cycles
              << " compiled_runs=" << total_runs
              << " metadata_bytes=" << metadata_bytes
              << " build_count_ms=" << count_ms
              << " build_fill_ms=" << fill_ms
              << " execute_wall_ms=" << exec_ms
              << " remote_u32_load_store_ops=" << remote_ops
              << " remote_GiB=" << double(remote_ops) * 4.0 / double(1ULL << 30)
              << " tie_hash=1 run_record_bytes=5 soa_u32_u8=1"
              << " support_ops_per_row=0 owner_ops_per_row=0 grouped_rank_ops_per_row=0"
              << " reusable_schedule=1 staging_state_bytes=0 exact=OK\n";

    ck(cudaSetDevice(0), "packed builder free set device");
    cudaFree(d_all_high); cudaFree(d_all_low); cudaFree(d_all_header);
    cudaFree(d_build_error); cudaFree(d_run_cursor); cudaFree(d_cycle_cursor);
    cudaFree(d_run_offset); cudaFree(d_header_offset);
    cudaFree(d_run_counts); cudaFree(d_cycle_counts);
    cudaEventDestroy(e2); cudaEventDestroy(e1); cudaEventDestroy(e0);
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "packed free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        cudaFree(c.run_high); cudaFree(c.run_low); cudaFree(c.header);
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
            constexpr unsigned long long runs = 167763968ULL;
            constexpr unsigned long long cycles = 21566612ULL;
            constexpr unsigned long long bytes = runs * 5ULL + cycles * 8ULL;
            std::cout << "gridfp-reduced-production-p2p-packed-schedule-plan"
                      << " W=28 K=13 ngpu=" << ngpu
                      << " cycles=" << cycles
                      << " runs=" << runs
                      << " one_direction_GiB=" << double(bytes) / double(1ULL << 30)
                      << " avg_one_direction_MiB_per_gpu="
                      << double(bytes) / double(ngpu) / double(1ULL << 20)
                      << " forward_reverse_both_GiB="
                      << 2.0 * double(bytes) / double(1ULL << 30)
                      << " run_record_bytes=5 tie_hash=1"
                      << " state_stream_GiB_per_gpu=220.442683"
                      << " support_ops_per_row=0 owner_ops_per_row=0 grouped_rank_ops_per_row=0\n";
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
    ck(cudaGetDeviceCount(&visible), "packed schedule device count");
    if (visible < ngpu) return 5;

    run_p2p_packed_schedule_probe(W, K, false, ngpu, blocks);
    run_p2p_packed_schedule_probe(W, K, true, ngpu, blocks);
    std::cout << "ALL_OK production_p2p_packed_schedule_cuda=1\n";
    return 0;
}
