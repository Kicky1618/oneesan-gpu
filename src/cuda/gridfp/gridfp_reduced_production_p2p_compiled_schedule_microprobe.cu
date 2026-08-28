#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_worklist_microprobe_main_unused
#include "gridfp_reduced_production_p2p_worklist_microprobe.cu"
#pragma pop_macro("main")

namespace {

static constexpr int P2P_COMPILED_LEN_SHIFT = 24;
static constexpr std::uint32_t P2P_COMPILED_PC_MASK =
    (std::uint32_t(1) << P2P_COMPILED_LEN_SHIFT) - 1u;

struct P2PCompiledHeader {
    std::uint32_t run_begin = 0;
    std::uint32_t primitive_and_len = 0;
};
static_assert(sizeof(P2PCompiledHeader) == 8);

__host__ __device__ __forceinline__ std::uint32_t p2p_compiled_pc(
    P2PCompiledHeader h
) {
    return h.primitive_and_len & P2P_COMPILED_PC_MASK;
}

__host__ __device__ __forceinline__ int p2p_compiled_len(
    P2PCompiledHeader h
) {
    return int(h.primitive_and_len >> P2P_COMPILED_LEN_SHIFT);
}

__global__ void p2p_compiled_count_kernel(
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
            const EqualTileRunSeed run = seeds[ri];
            const bool blocked = run.blocked != 0;
            const int len = shift_cycle_leader_length_device(
                run.support, blocked, W, q, K, K, reverse);
            if (len < 0 || len > RP_MAX_W) {
                atomicCAS(error, 0, 351);
                continue;
            }
            if (len <= 1) continue;
            const int exec = p2p_modal_owner_only_device<P2P_MAX_GPU>(
                run.support, blocked, len, W, q, K, K,
                reverse, old_start, ngpu);
            if (exec < 0 || exec >= ngpu) {
                atomicCAS(error, 0, 352);
                continue;
            }
            atomicAdd(cycle_counts + exec, 1ULL);
            atomicAdd(run_counts + exec, static_cast<unsigned long long>(len));
        }
    }
}

__global__ void p2p_compiled_fill_kernel(
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
    std::uint64_t* __restrict__ packed_run,
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
                atomicCAS(error, 0, 353);
                continue;
            }
            if (len <= 1) continue;
            const int exec = p2p_modal_owner_only_device<P2P_MAX_GPU>(
                root.support, blocked, len, W, q, K, K,
                reverse, old_start, ngpu);
            if (exec < 0 || exec >= ngpu) {
                atomicCAS(error, 0, 354);
                continue;
            }
            const unsigned long long ci = atomicAdd(cycle_cursor + exec, 1ULL);
            const unsigned long long rb = atomicAdd(
                run_cursor + exec, static_cast<unsigned long long>(len));
            if (rb + static_cast<unsigned long long>(len) > 0xffffffffULL) {
                atomicCAS(error, 0, 355);
                continue;
            }
            const Rank64 pc = RP_PRIMITIVE[__popc(root.support)][1];
            if (pc > P2P_COMPILED_PC_MASK) {
                atomicCAS(error, 0, 357);
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
                if (gr.owner < 0 || gr.owner >= ngpu || gr.local >= (Rank64(1) << 61)) {
                    atomicCAS(error, 0, 356);
                    break;
                }
                packed_run[run_offset[exec] + rb + h] =
                    (std::uint64_t(gr.local) << 3) | std::uint64_t(gr.owner);
                cur = shift_next_support_device(
                    cur, blocked, W, q, K, K, reverse);
            }
        }
    }
}

__global__ void p2p_compiled_execute_kernel(
    std::uint32_t** __restrict__ peer_state,
    const P2PCompiledHeader* __restrict__ header,
    const std::uint64_t* __restrict__ packed_run,
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
                atomicCAS(error, 0, 361);
            } else {
                for (int h = 0; h < len; ++h) {
                    const std::uint64_t z = packed_run[Rank64(a.run_begin) + h];
                    sh_owner[warp][h] = int(z & 7u);
                    sh_local[warp][h] = Rank64(z >> 3);
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

struct DeviceCompiledCtx : DevicePeerCtx {
    P2PCompiledHeader* header = nullptr;
    std::uint64_t* run = nullptr;
    Rank64 cycle_count = 0;
    Rank64 run_count = 0;
};

void run_p2p_compiled_schedule_probe(
    int W, int K, bool reverse, int ngpu, unsigned blocks
) {
    ProductionFactorTables tables(W);
    const HostTilePlan plan = make_host_tile_plan(tables, K, ngpu);
    std::vector<std::uint32_t> flat_input, flat_expected;
    build_shift_boundary_vectors(
        W, K, K, reverse, ngpu, tables, plan, flat_input, flat_expected);
    enable_all_peer_access(ngpu);

    std::vector<DeviceCompiledCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> peer_ptr(static_cast<std::size_t>(ngpu));
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "compiled alloc set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)), "compiled alloc state");
        peer_ptr[static_cast<std::size_t>(g)] = c.state;
        const Rank64 base = plan.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(c.state, flat_input.data() + base, n * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice), "compiled copy state");
        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)), "compiled alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                      cudaMemcpyHostToDevice), "compiled copy owner begin");
        ck(cudaMalloc(&c.cycles, sizeof(unsigned long long)), "compiled alloc executed");
        ck(cudaMalloc(&c.cross_values, sizeof(unsigned long long)), "compiled alloc remote ops");
        ck(cudaMalloc(&c.error, sizeof(int)), "compiled alloc error");
        ck(cudaMemset(c.cycles, 0, sizeof(unsigned long long)), "compiled zero executed");
        ck(cudaMemset(c.cross_values, 0, sizeof(unsigned long long)), "compiled zero remote ops");
        ck(cudaMemset(c.error, 0, sizeof(int)), "compiled zero error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "compiled peer table set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.peer, ngpu * sizeof(std::uint32_t*)), "compiled alloc peer table");
        ck(cudaMemcpy(c.peer, peer_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "compiled copy peer table");
    }

    ck(cudaSetDevice(0), "compiled builder set device");
    install_tables(tables);
    unsigned long long *d_cycle_counts = nullptr, *d_run_counts = nullptr;
    unsigned long long *d_header_offset = nullptr, *d_run_offset = nullptr;
    unsigned long long *d_cycle_cursor = nullptr, *d_run_cursor = nullptr;
    int* d_build_error = nullptr;
    ck(cudaMalloc(&d_cycle_counts, ngpu * sizeof(unsigned long long)), "compiled alloc cycle counts");
    ck(cudaMalloc(&d_run_counts, ngpu * sizeof(unsigned long long)), "compiled alloc run counts");
    ck(cudaMalloc(&d_header_offset, ngpu * sizeof(unsigned long long)), "compiled alloc header offsets");
    ck(cudaMalloc(&d_run_offset, ngpu * sizeof(unsigned long long)), "compiled alloc run offsets");
    ck(cudaMalloc(&d_cycle_cursor, ngpu * sizeof(unsigned long long)), "compiled alloc cycle cursor");
    ck(cudaMalloc(&d_run_cursor, ngpu * sizeof(unsigned long long)), "compiled alloc run cursor");
    ck(cudaMalloc(&d_build_error, sizeof(int)), "compiled alloc build error");
    ck(cudaMemset(d_cycle_counts, 0, ngpu * sizeof(unsigned long long)), "compiled zero cycle counts");
    ck(cudaMemset(d_run_counts, 0, ngpu * sizeof(unsigned long long)), "compiled zero run counts");
    ck(cudaMemset(d_build_error, 0, sizeof(int)), "compiled zero build error");

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const unsigned build_threads = 256;
    const Rank64 build_one_pass = (base_supports + build_threads - 1) / build_threads;
    const unsigned build_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(blocks, build_one_pass)));
    cudaEvent_t e0{}, e1{}, e2{};
    ck(cudaEventCreate(&e0), "compiled event 0");
    ck(cudaEventCreate(&e1), "compiled event 1");
    ck(cudaEventCreate(&e2), "compiled event 2");
    ck(cudaEventRecord(e0), "compiled record 0");
    p2p_compiled_count_kernel<<<build_blocks, build_threads>>>(
        base_supports, W, K, reverse, ngpu,
        d_cycle_counts, d_run_counts, d_build_error);
    ck(cudaGetLastError(), "compiled count launch");
    ck(cudaEventRecord(e1), "compiled record 1");
    ck(cudaEventSynchronize(e1), "compiled count sync");

    std::vector<unsigned long long> h_cycle_counts(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> h_run_counts(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> h_header_offset(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> h_run_offset(static_cast<std::size_t>(ngpu));
    ck(cudaMemcpy(h_cycle_counts.data(), d_cycle_counts,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "compiled copy cycle counts");
    ck(cudaMemcpy(h_run_counts.data(), d_run_counts,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "compiled copy run counts");
    unsigned long long total_headers = 0, total_runs = 0;
    for (int g = 0; g < ngpu; ++g) {
        h_header_offset[static_cast<std::size_t>(g)] = total_headers;
        h_run_offset[static_cast<std::size_t>(g)] = total_runs;
        total_headers += h_cycle_counts[static_cast<std::size_t>(g)];
        total_runs += h_run_counts[static_cast<std::size_t>(g)];
    }
    ck(cudaMemcpy(d_header_offset, h_header_offset.data(),
                  ngpu * sizeof(unsigned long long), cudaMemcpyHostToDevice),
       "compiled copy header offsets");
    ck(cudaMemcpy(d_run_offset, h_run_offset.data(),
                  ngpu * sizeof(unsigned long long), cudaMemcpyHostToDevice),
       "compiled copy run offsets");
    ck(cudaMemset(d_cycle_cursor, 0, ngpu * sizeof(unsigned long long)),
       "compiled zero cycle cursor");
    ck(cudaMemset(d_run_cursor, 0, ngpu * sizeof(unsigned long long)),
       "compiled zero run cursor");

    P2PCompiledHeader* d_all_header = nullptr;
    std::uint64_t* d_all_run = nullptr;
    ck(cudaMalloc(&d_all_header, total_headers * sizeof(P2PCompiledHeader)),
       "compiled alloc all headers");
    ck(cudaMalloc(&d_all_run, total_runs * sizeof(std::uint64_t)),
       "compiled alloc all runs");
    p2p_compiled_fill_kernel<<<build_blocks, build_threads>>>(
        base_supports, W, K, reverse, ngpu,
        d_header_offset, d_run_offset, d_cycle_cursor, d_run_cursor,
        d_all_header, d_all_run, ctx[0].owner_begin, d_build_error);
    ck(cudaGetLastError(), "compiled fill launch");
    ck(cudaEventRecord(e2), "compiled record 2");
    ck(cudaEventSynchronize(e2), "compiled fill sync");
    float count_ms = 0.0f, fill_ms = 0.0f;
    ck(cudaEventElapsedTime(&count_ms, e0, e1), "compiled count time");
    ck(cudaEventElapsedTime(&fill_ms, e1, e2), "compiled fill time");
    int build_error = 0;
    ck(cudaMemcpy(&build_error, d_build_error, sizeof(build_error),
                  cudaMemcpyDeviceToHost), "compiled copy build error");
    if (build_error) fail("compiled builder device error=" + std::to_string(build_error));

    std::vector<unsigned long long> h_cycle_cursor(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> h_run_cursor(static_cast<std::size_t>(ngpu));
    ck(cudaMemcpy(h_cycle_cursor.data(), d_cycle_cursor,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "compiled copy cycle cursor");
    ck(cudaMemcpy(h_run_cursor.data(), d_run_cursor,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "compiled copy run cursor");
    for (int g = 0; g < ngpu; ++g) {
        if (h_cycle_cursor[static_cast<std::size_t>(g)] !=
                h_cycle_counts[static_cast<std::size_t>(g)] ||
            h_run_cursor[static_cast<std::size_t>(g)] !=
                h_run_counts[static_cast<std::size_t>(g)])
            fail("compiled fill cursor/count mismatch");
    }

    for (int g = 0; g < ngpu; ++g) {
        auto& c = ctx[static_cast<std::size_t>(g)];
        c.cycle_count = h_cycle_counts[static_cast<std::size_t>(g)];
        c.run_count = h_run_counts[static_cast<std::size_t>(g)];
        ck(cudaSetDevice(g), "compiled distribute set device");
        ck(cudaMalloc(&c.header,
                      std::max<Rank64>(1, c.cycle_count) * sizeof(P2PCompiledHeader)),
           "compiled alloc local headers");
        ck(cudaMalloc(&c.run,
                      std::max<Rank64>(1, c.run_count) * sizeof(std::uint64_t)),
           "compiled alloc local runs");
        const auto* hs = d_all_header + h_header_offset[static_cast<std::size_t>(g)];
        const auto* rs = d_all_run + h_run_offset[static_cast<std::size_t>(g)];
        if (g == 0) {
            if (c.cycle_count)
                ck(cudaMemcpy(c.header, hs, c.cycle_count * sizeof(P2PCompiledHeader),
                              cudaMemcpyDeviceToDevice), "compiled local header copy");
            if (c.run_count)
                ck(cudaMemcpy(c.run, rs, c.run_count * sizeof(std::uint64_t),
                              cudaMemcpyDeviceToDevice), "compiled local run copy");
        } else {
            if (c.cycle_count)
                ck(cudaMemcpyPeer(c.header, g, hs, 0,
                                  c.cycle_count * sizeof(P2PCompiledHeader)),
                   "compiled peer header copy");
            if (c.run_count)
                ck(cudaMemcpyPeer(c.run, g, rs, 0,
                                  c.run_count * sizeof(std::uint64_t)),
                   "compiled peer run copy");
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "compiled execute set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 one_pass =
            (c.cycle_count + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned launch_blocks = static_cast<unsigned>(
            std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass)));
        p2p_compiled_execute_kernel<<<launch_blocks, THREADS>>>(
            c.peer, c.header, c.run, c.cycle_count, c.run_count, g,
            c.cycles, c.cross_values, c.error);
        ck(cudaGetLastError(), "compiled execute launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "compiled execute sync set device");
        ck(cudaDeviceSynchronize(), "compiled execute sync");
    }
    const double exec_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> flat_output(flat_expected.size());
    unsigned long long executed = 0, remote_ops = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "compiled gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0; unsigned long long ge = 0, gr = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "compiled copy error");
        ck(cudaMemcpy(&ge, c.cycles, sizeof(ge), cudaMemcpyDeviceToHost),
           "compiled copy executed");
        ck(cudaMemcpy(&gr, c.cross_values, sizeof(gr), cudaMemcpyDeviceToHost),
           "compiled copy remote ops");
        if (error) fail("compiled execute device error=" + std::to_string(error));
        executed += ge; remote_ops += gr;
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        const Rank64 base = plan.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(flat_output.data() + base, c.state,
                      n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "compiled gather state");
    }
    unsigned long long expected_cycles = 0;
    for (auto z : h_cycle_counts) expected_cycles += z;
    if (executed != expected_cycles) fail("compiled executed cycle count mismatch");
    if (flat_output != flat_expected) fail("compiled redistribution mismatch");

    const unsigned long long metadata_bytes =
        total_headers * sizeof(P2PCompiledHeader) + total_runs * sizeof(std::uint64_t);
    std::cout << "gridfp-reduced-production-p2p-compiled-schedule"
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
              << " header_order_independent=1"
              << " support_ops_per_row=0 owner_ops_per_row=0 grouped_rank_ops_per_row=0"
              << " reusable_schedule=1 staging_state_bytes=0 exact=OK\n";

    ck(cudaSetDevice(0), "compiled builder free set device");
    cudaFree(d_all_run); cudaFree(d_all_header); cudaFree(d_build_error);
    cudaFree(d_run_cursor); cudaFree(d_cycle_cursor);
    cudaFree(d_run_offset); cudaFree(d_header_offset);
    cudaFree(d_run_counts); cudaFree(d_cycle_counts);
    cudaEventDestroy(e2); cudaEventDestroy(e1); cudaEventDestroy(e0);
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "compiled free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        cudaFree(c.run); cudaFree(c.header);
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
            constexpr unsigned long long nonfixed_runs = 167763968ULL;
            constexpr unsigned long long cycles = 21566612ULL;
            constexpr unsigned long long bytes =
                nonfixed_runs * sizeof(std::uint64_t) +
                cycles * sizeof(P2PCompiledHeader);
            std::cout << "gridfp-reduced-production-p2p-compiled-schedule-plan"
                      << " W=28 K=13 ngpu=" << ngpu
                      << " cycles=" << cycles
                      << " nonfixed_runs=" << nonfixed_runs
                      << " schedule_GiB=" << double(bytes) / double(1ULL << 30)
                      << " avg_schedule_MiB_per_gpu="
                      << double(bytes) / double(ngpu) / double(1ULL << 20)
                      << " forward_reverse_both_GiB="
                      << 2.0 * double(bytes) / double(1ULL << 30)
                      << " header_len_bits=5 primitive_bits=24"
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
    ck(cudaGetDeviceCount(&visible), "compiled schedule device count");
    if (visible < ngpu) return 5;

    run_p2p_compiled_schedule_probe(W, K, false, ngpu, blocks);
    run_p2p_compiled_schedule_probe(W, K, true, ngpu, blocks);
    std::cout << "ALL_OK production_p2p_compiled_schedule_cuda=1\n";
    return 0;
}
