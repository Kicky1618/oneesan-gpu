#include <cuda_runtime.h>

#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_cycle_owner_route_lut_pipeline_microprobe_main_unused
#include "gridfp_reduced_production_p2p_cycle_owner_route_lut_pipeline_microprobe.cu"
#pragma pop_macro("main")

namespace {

static constexpr int B300_THREADS = 256;
static constexpr int B300_WARP = 32;
static constexpr int B300_WARPS_PER_BLOCK = B300_THREADS / B300_WARP;
static constexpr int B300_PIPELINE_SLOTS = 2;

struct B300SegmentTask {
    unsigned long long scratch_offset = 0;
    std::uint32_t primitive_count = 0;
    std::uint32_t route_bits = 0;
};
static_assert(sizeof(B300SegmentTask) == 16, "B300 segment task must stay compact");

__host__ __device__ __forceinline__ std::uint32_t b300_pack_route_bits(
    std::uint32_t start_support,
    std::uint32_t dst_support,
    int len,
    bool blocked
) {
    return (start_support & 0x7ffu) |
           ((dst_support & 0x7ffu) << 11) |
           ((static_cast<std::uint32_t>(len) & 0x1fu) << 22) |
           (static_cast<std::uint32_t>(blocked ? 1u : 0u) << 27);
}

__device__ __forceinline__ std::uint32_t b300_task_start(
    const B300SegmentTask& t
) {
    return t.route_bits & 0x7ffu;
}

__device__ __forceinline__ std::uint32_t b300_task_dst(
    const B300SegmentTask& t
) {
    return (t.route_bits >> 11) & 0x7ffu;
}

__device__ __forceinline__ int b300_task_len(
    const B300SegmentTask& t
) {
    return int((t.route_bits >> 22) & 0x1fu);
}

__device__ __forceinline__ bool b300_task_blocked(
    const B300SegmentTask& t
) {
    return ((t.route_bits >> 27) & 1u) != 0;
}

__global__ void b300_local_cycle_warp_kernel(
    std::uint32_t* state,
    const std::uint32_t* __restrict__ local_list,
    unsigned long long count,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int S,
    int ngpu,
    int owner,
    const Rank64* __restrict__ owner_begin,
    int* error
) {
    __shared__ Rank64 sh_local[B300_WARPS_PER_BLOCK][RP_MAX_W];
    __shared__ int sh_len[B300_WARPS_PER_BLOCK];
    __shared__ std::uint32_t sh_pc[B300_WARPS_PER_BLOCK];
    __shared__ int sh_valid[B300_WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const unsigned long long first =
        static_cast<unsigned long long>(blockIdx.x) * B300_WARPS_PER_BLOCK + warp;
    const unsigned long long stride =
        static_cast<unsigned long long>(gridDim.x) * B300_WARPS_PER_BLOCK;

    for (unsigned long long ix = first; ix < count; ix += stride) {
        if (lane == 0) {
            sh_valid[warp] = 1;
            sh_len[warp] = 0;
            const std::uint32_t packed = local_list[ix];
            const std::uint32_t start = packed & PERSISTENT_SUPPORT_MASK;
            const bool blocked = (packed & PERSISTENT_BLOCKED_BIT) != 0;
            const Rank64 pc64 = RP_PRIMITIVE[__popc(start)][1];
            if (!pc64 || pc64 > 0xffffffffULL) {
                set_error(error, 421);
                sh_valid[warp] = 0;
            } else {
                sh_pc[warp] = static_cast<std::uint32_t>(pc64);
                std::uint32_t cur = start;
                int len = 0;
                do {
                    if (len >= RP_MAX_W) {
                        set_error(error, 422);
                        sh_valid[warp] = 0;
                        break;
                    }
                    const GroupedDeviceRank gr = support_rank_lut_lookup_device(
                        cur, blocked, W, q, reverse, tile_start,
                        Kwin, ngpu, owner_begin);
                    if (gr.owner != owner) {
                        set_error(error, 423);
                        sh_valid[warp] = 0;
                        break;
                    }
                    sh_local[warp][len++] = gr.local;
                    cur = next_support_lut_lookup_device(
                        cur, blocked, W, q, Kwin, S, reverse);
                    if (cur == 0xffffffffu) {
                        set_error(error, 424);
                        sh_valid[warp] = 0;
                        break;
                    }
                } while (cur != start);
                sh_len[warp] = len;
                if (len <= 1) {
                    set_error(error, 425);
                    sh_valid[warp] = 0;
                }
            }
        }
        __syncwarp();
        if (!sh_valid[warp]) continue;

        const int len = sh_len[warp];
        const std::uint32_t pc = sh_pc[warp];
        const int tail = len - 1;
        for (std::uint32_t i = static_cast<std::uint32_t>(lane);
             i < pc; i += B300_WARP) {
            const std::uint32_t tmp =
                state[sh_local[warp][tail] + static_cast<Rank64>(i)];
            for (int h = tail; h > 0; --h) {
                state[sh_local[warp][h] + static_cast<Rank64>(i)] =
                    state[sh_local[warp][h - 1] + static_cast<Rank64>(i)];
            }
            state[sh_local[warp][0] + static_cast<Rank64>(i)] = tmp;
        }
        __syncwarp();
    }
}

__global__ void b300_phase_a_warp_kernel(
    std::uint32_t* state,
    std::uint32_t* scratch,
    const B300SegmentTask* __restrict__ tasks,
    unsigned long long count,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int S,
    int ngpu,
    int owner,
    const Rank64* __restrict__ owner_begin,
    int* error
) {
    __shared__ Rank64 sh_local[B300_WARPS_PER_BLOCK][RP_MAX_W];
    __shared__ int sh_valid[B300_WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const unsigned long long first =
        static_cast<unsigned long long>(blockIdx.x) * B300_WARPS_PER_BLOCK + warp;
    const unsigned long long stride =
        static_cast<unsigned long long>(gridDim.x) * B300_WARPS_PER_BLOCK;

    for (unsigned long long ti = first; ti < count; ti += stride) {
        const B300SegmentTask task = tasks[ti];
        const int len = b300_task_len(task);
        const bool blocked = b300_task_blocked(task);
        if (lane == 0) {
            sh_valid[warp] = 1;
            if (len < 1 || len > RP_MAX_W || !task.primitive_count) {
                set_error(error, 431);
                sh_valid[warp] = 0;
            } else {
                std::uint32_t cur = b300_task_start(task);
                for (int h = 0; h < len; ++h) {
                    const GroupedDeviceRank gr = support_rank_lut_lookup_device(
                        cur, blocked, W, q, reverse, tile_start,
                        Kwin, ngpu, owner_begin);
                    if (gr.owner != owner) {
                        set_error(error, 432);
                        sh_valid[warp] = 0;
                        break;
                    }
                    sh_local[warp][h] = gr.local;
                    cur = next_support_lut_lookup_device(
                        cur, blocked, W, q, Kwin, S, reverse);
                    if (cur == 0xffffffffu) {
                        set_error(error, 433);
                        sh_valid[warp] = 0;
                        break;
                    }
                }
            }
        }
        __syncwarp();
        if (!sh_valid[warp]) continue;

        const int tail = len - 1;
        const unsigned long long base = task.scratch_offset;
        for (std::uint32_t i = static_cast<std::uint32_t>(lane);
             i < task.primitive_count; i += B300_WARP) {
            scratch[base + static_cast<unsigned long long>(i)] =
                state[sh_local[warp][tail] + static_cast<Rank64>(i)];
            for (int h = tail; h > 0; --h) {
                state[sh_local[warp][h] + static_cast<Rank64>(i)] =
                    state[sh_local[warp][h - 1] + static_cast<Rank64>(i)];
            }
        }
        __syncwarp();
    }
}

__global__ void b300_phase_b_warp_kernel(
    std::uint32_t* const* __restrict__ peer_state,
    const std::uint32_t* __restrict__ scratch,
    const B300SegmentTask* __restrict__ tasks,
    unsigned long long count,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int ngpu,
    int owner,
    const Rank64* __restrict__ owner_begin,
    int* error
) {
    __shared__ unsigned long long sh_dst_addr[B300_WARPS_PER_BLOCK];
    __shared__ int sh_valid[B300_WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const unsigned long long first =
        static_cast<unsigned long long>(blockIdx.x) * B300_WARPS_PER_BLOCK + warp;
    const unsigned long long stride =
        static_cast<unsigned long long>(gridDim.x) * B300_WARPS_PER_BLOCK;

    for (unsigned long long ti = first; ti < count; ti += stride) {
        const B300SegmentTask task = tasks[ti];
        if (lane == 0) {
            sh_valid[warp] = 1;
            const GroupedDeviceRank dst = support_rank_lut_lookup_device(
                b300_task_dst(task), b300_task_blocked(task),
                W, q, reverse, tile_start, Kwin, ngpu, owner_begin);
            if (dst.owner < 0 || dst.owner >= ngpu || dst.owner == owner) {
                set_error(error, 441);
                sh_valid[warp] = 0;
                sh_dst_addr[warp] = 0;
            } else {
                std::uint32_t* ptr = peer_state[dst.owner] + dst.local;
                sh_dst_addr[warp] =
                    reinterpret_cast<unsigned long long>(ptr);
            }
        }
        __syncwarp();
        if (!sh_valid[warp]) continue;

        std::uint32_t* dst = reinterpret_cast<std::uint32_t*>(sh_dst_addr[warp]);
        const unsigned long long base = task.scratch_offset;
        for (std::uint32_t i = static_cast<std::uint32_t>(lane);
             i < task.primitive_count; i += B300_WARP) {
            dst[i] = scratch[base + static_cast<unsigned long long>(i)];
        }
        __syncwarp();
    }
}

struct B300Ctx {
    std::uint32_t* state = nullptr;
    std::uint32_t* scratch[B300_PIPELINE_SLOTS]{};
    B300SegmentTask* tasks = nullptr;
    std::uint32_t* local_list = nullptr;
    std::uint32_t** peer_state = nullptr;
    Rank64* owner_begin = nullptr;
    int* error = nullptr;
    cudaStream_t produce{};
    cudaStream_t consume{};
    cudaEvent_t ready[B300_PIPELINE_SLOTS]{};
    cudaEvent_t consumed[B300_PIPELINE_SLOTS]{};
    int sm_count = 1;
};

unsigned b300_grid_blocks(
    unsigned long long work_items,
    int sm_count,
    unsigned blocks_per_sm
) {
    if (!work_items) return 0;
    const unsigned long long natural =
        (work_items + B300_WARPS_PER_BLOCK - 1) / B300_WARPS_PER_BLOCK;
    const unsigned long long target =
        std::max<unsigned long long>(1,
            static_cast<unsigned long long>(std::max(1, sm_count)) *
            std::max(1u, blocks_per_sm));
    return static_cast<unsigned>(std::min(natural, target));
}

void run_b300_feed_pipeline(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    int batches,
    unsigned blocks_per_sm
) {
    ProductionFactorTables tables(W);
    const HostTilePlan plan = make_host_tile_plan(tables, Kwin, ngpu);
    const int tile_start = reverse ? 1 : W - 1;
    const int q = tile_start + (reverse ? S : -S);

    std::vector<std::uint32_t> input;
    std::vector<std::uint32_t> expected;
    build_shift_boundary_vectors(
        W, Kwin, S, reverse, ngpu, tables, plan, input, expected);
    enable_scratch_full_peer_mesh(ngpu);

    const auto setup0 = std::chrono::steady_clock::now();
    FusedHostCycleOwnerLists lists = build_fused_cycle_owner_lists(
        W, Kwin, S, reverse, ngpu, batches, tables);

    std::vector<B300Ctx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> state_ptr(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<std::vector<unsigned long long>> batch_task_offset(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches + 1)));
    std::vector<std::vector<unsigned long long>> batch_scratch_words(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches)));

    unsigned long long total_tasks = 0;
    unsigned long long total_peer_words = 0;
    unsigned long long total_cross_logical_bytes = 0;
    unsigned long long max_slot_words[B300_PIPELINE_SLOTS]{};

    for (int g = 0; g < ngpu; ++g) {
        std::vector<B300SegmentTask> tasks;
        unsigned long long slot_words[B300_PIPELINE_SLOTS] = {1, 1};

        for (int b = 0; b < batches; ++b) {
            batch_task_offset[static_cast<std::size_t>(g)]
                             [static_cast<std::size_t>(b)] = tasks.size();
            unsigned long long scratch_cursor = 0;
            const auto& part =
                lists.batch[static_cast<std::size_t>(g)]
                           [static_cast<std::size_t>(b)];
            const auto& support =
                lists.route_support[static_cast<std::size_t>(g)]
                                   [static_cast<std::size_t>(b)];
            const auto& length =
                lists.route_len[static_cast<std::size_t>(g)]
                               [static_cast<std::size_t>(b)];
            if (support.size() != length.size())
                fail("B300 route arena mismatch");

            for (const auto& entry : part) {
                const bool blocked =
                    (entry.base.packed & PERSISTENT_BLOCKED_BIT) != 0;
                const Rank64 pc64 =
                    tables.primitive[static_cast<std::size_t>(entry.base.occupied)][1];
                if (!pc64 || pc64 > 0xffffffffULL)
                    fail("B300 primitive count range");
                const std::uint32_t pc = static_cast<std::uint32_t>(pc64);
                const unsigned long long begin = entry.route_offset;
                const unsigned long long end =
                    begin + static_cast<unsigned long long>(entry.base.segments);
                if (end > support.size()) fail("B300 route offset");

                for (unsigned long long si = begin; si < end; ++si) {
                    const std::uint32_t start = support[static_cast<std::size_t>(si)];
                    const int len = int(length[static_cast<std::size_t>(si)]);
                    if (start >= (1u << W) || len < 1 || len > RP_MAX_W)
                        fail("B300 route task");
                    std::uint32_t dst = start;
                    for (int h = 0; h < len; ++h)
                        dst = hostlist_next(dst, blocked, W, q, Kwin, S, reverse);
                    if (dst >= (1u << W) ||
                        hostlist_owner(dst, W, Kwin, reverse, ngpu, tables) == g)
                        fail("B300 destination route");

                    B300SegmentTask task{};
                    task.scratch_offset = scratch_cursor;
                    task.primitive_count = pc;
                    task.route_bits = b300_pack_route_bits(start, dst, len, blocked);
                    tasks.push_back(task);
                    scratch_cursor += static_cast<unsigned long long>(pc);
                    total_cross_logical_bytes +=
                        static_cast<unsigned long long>(pc) *
                        static_cast<unsigned long long>(len + 1) * 8ULL;
                }
            }

            if (scratch_cursor !=
                lists.words[static_cast<std::size_t>(g)]
                           [static_cast<std::size_t>(b)])
                fail("B300 scratch word accounting");
            batch_scratch_words[static_cast<std::size_t>(g)]
                               [static_cast<std::size_t>(b)] = scratch_cursor;
            slot_words[b & (B300_PIPELINE_SLOTS - 1)] =
                std::max(slot_words[b & (B300_PIPELINE_SLOTS - 1)], scratch_cursor);
            total_peer_words += scratch_cursor;
        }
        batch_task_offset[static_cast<std::size_t>(g)]
                         [static_cast<std::size_t>(batches)] = tasks.size();
        total_tasks += tasks.size();

        ck(cudaSetDevice(g), "B300 set device");
        install_tables(tables);
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, g), "B300 get device properties");

        auto& c = ctx[static_cast<std::size_t>(g)];
        c.sm_count = std::max(1, prop.multiProcessorCount);

        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "B300 alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(),
                      ngpu * sizeof(Rank64), cudaMemcpyHostToDevice),
           "B300 copy owner begin");

        ck(cudaMalloc(&c.tasks,
                      std::max<std::size_t>(1, tasks.size()) * sizeof(B300SegmentTask)),
           "B300 alloc tasks");
        if (!tasks.empty())
            ck(cudaMemcpy(c.tasks, tasks.data(), tasks.size() * sizeof(B300SegmentTask),
                          cudaMemcpyHostToDevice), "B300 copy tasks");

        const auto& local = lists.local[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.local_list,
                      std::max<std::size_t>(1, local.size()) * sizeof(std::uint32_t)),
           "B300 alloc local list");
        if (!local.empty())
            ck(cudaMemcpy(c.local_list, local.data(),
                          local.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "B300 copy local list");

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)), "B300 alloc state");
        state_ptr[static_cast<std::size_t>(g)] = c.state;
        ck(cudaMemcpy(c.state,
                      input.data() + plan.shard_base[static_cast<std::size_t>(g)],
                      n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "B300 copy state");

        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "B300 alloc peer table");
        ck(cudaMalloc(&c.error, sizeof(int)), "B300 alloc error");
        ck(cudaMemset(c.error, 0, sizeof(int)), "B300 clear error");

        for (int slot = 0; slot < B300_PIPELINE_SLOTS; ++slot) {
            max_slot_words[slot] = std::max(max_slot_words[slot], slot_words[slot]);
            ck(cudaMalloc(&c.scratch[slot],
                          slot_words[slot] * sizeof(std::uint32_t)),
               "B300 alloc scratch slot");
        }

        ck(cudaStreamCreateWithFlags(&c.produce, cudaStreamNonBlocking),
           "B300 create produce stream");
        ck(cudaStreamCreateWithFlags(&c.consume, cudaStreamNonBlocking),
           "B300 create consume stream");
        for (int slot = 0; slot < B300_PIPELINE_SLOTS; ++slot) {
            ck(cudaEventCreateWithFlags(&c.ready[slot], cudaEventDisableTiming),
               "B300 create ready event");
            ck(cudaEventCreateWithFlags(&c.consumed[slot], cudaEventDisableTiming),
               "B300 create consumed event");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "B300 peer set device");
        ck(cudaMemcpy(ctx[static_cast<std::size_t>(g)].peer_state,
                      state_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "B300 copy peer table");
    }

    const double setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup0).count();

    for (int g = 0; g < ngpu; ++g) {
        const auto count = lists.local[static_cast<std::size_t>(g)].size();
        if (!count) continue;
        ck(cudaSetDevice(g), "B300 local set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const unsigned blocks = b300_grid_blocks(
            count, c.sm_count, blocks_per_sm);
        b300_local_cycle_warp_kernel<<<blocks, B300_THREADS>>>(
            c.state, c.local_list, count,
            W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "B300 local launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "B300 local sync set device");
        ck(cudaDeviceSynchronize(), "B300 local sync");
        int error = 0;
        ck(cudaMemcpy(&error, ctx[static_cast<std::size_t>(g)].error,
                      sizeof(error), cudaMemcpyDeviceToHost),
           "B300 local copy error");
        if (error) fail("B300 local device error=" + std::to_string(error));
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int batch = 0; batch < batches; ++batch) {
        const int slot = batch & (B300_PIPELINE_SLOTS - 1);

        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "B300 phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            if (batch >= B300_PIPELINE_SLOTS)
                ck(cudaStreamWaitEvent(c.produce, c.consumed[slot], 0),
                   "B300 wait slot consumed");

            const auto begin =
                batch_task_offset[static_cast<std::size_t>(g)]
                                 [static_cast<std::size_t>(batch)];
            const auto end =
                batch_task_offset[static_cast<std::size_t>(g)]
                                 [static_cast<std::size_t>(batch + 1)];
            const auto count = end - begin;
            if (count) {
                const unsigned blocks = b300_grid_blocks(
                    count, c.sm_count, blocks_per_sm);
                b300_phase_a_warp_kernel<<<blocks, B300_THREADS, 0, c.produce>>>(
                    c.state, c.scratch[slot], c.tasks + begin, count,
                    W, q, reverse, tile_start, Kwin, S,
                    ngpu, g, c.owner_begin, c.error);
                ck(cudaGetLastError(), "B300 phase A launch");
            }
            ck(cudaEventRecord(c.ready[slot], c.produce),
               "B300 record ready");
        }

        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "B300 phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            for (int src = 0; src < ngpu; ++src)
                ck(cudaStreamWaitEvent(
                       c.consume,
                       ctx[static_cast<std::size_t>(src)].ready[slot], 0),
                   "B300 wait all phase A");

            const auto begin =
                batch_task_offset[static_cast<std::size_t>(g)]
                                 [static_cast<std::size_t>(batch)];
            const auto end =
                batch_task_offset[static_cast<std::size_t>(g)]
                                 [static_cast<std::size_t>(batch + 1)];
            const auto count = end - begin;
            if (count) {
                const unsigned blocks = b300_grid_blocks(
                    count, c.sm_count, blocks_per_sm);
                b300_phase_b_warp_kernel<<<blocks, B300_THREADS, 0, c.consume>>>(
                    c.peer_state, c.scratch[slot], c.tasks + begin, count,
                    W, q, reverse, tile_start, Kwin,
                    ngpu, g, c.owner_begin, c.error);
                ck(cudaGetLastError(), "B300 phase B launch");
            }
            ck(cudaEventRecord(c.consumed[slot], c.consume),
               "B300 record consumed");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "B300 final sync set device");
        ck(cudaStreamSynchronize(ctx[static_cast<std::size_t>(g)].consume),
           "B300 final consume sync");
        ck(cudaStreamSynchronize(ctx[static_cast<std::size_t>(g)].produce),
           "B300 final produce sync");
    }
    const double runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> output(input.size());
    unsigned long long local_entries = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "B300 gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "B300 copy error");
        if (error) fail("B300 device error=" + std::to_string(error));

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(output.data() + plan.shard_base[static_cast<std::size_t>(g)],
                      c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "B300 gather state");
        local_entries += lists.local[static_cast<std::size_t>(g)].size();
    }

    if (output != expected)
        fail("B300 feed pipeline redistribution mismatch");

    const double logical_gbps = runtime_ms > 0.0
        ? (double(total_cross_logical_bytes) / 1.0e9) /
          (runtime_ms / 1000.0)
        : 0.0;

    std::cout << "gridfp-p2p-cycle-owner-b300-feed-pipeline"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " batches=" << batches
              << " blocks_per_sm=" << blocks_per_sm
              << " threads=" << B300_THREADS
              << " warps_per_block=" << B300_WARPS_PER_BLOCK
              << " segment_tasks=" << total_tasks
              << " local_cycles=" << local_entries
              << " task_bytes=" << total_tasks * sizeof(B300SegmentTask)
              << " logical_peer_KiB="
              << double(total_peer_words) * sizeof(std::uint32_t) / 1024.0
              << " scratch_even_KiB="
              << double(max_slot_words[0]) * sizeof(std::uint32_t) / 1024.0
              << " scratch_odd_KiB="
              << double(max_slot_words[1]) * sizeof(std::uint32_t) / 1024.0
              << " cross_logical_GBps=" << logical_gbps
              << " setup_ms=" << setup_ms
              << " runtime_ms=" << runtime_ms
              << " peer_word_audit_atomics=0"
              << " audit_kernel=0"
              << " entry_header_prepare=0"
              << " class_metadata=0"
              << " warp_per_segment=1"
              << " blockwide_segment_barriers=0"
              << " local_cycle_legacy_prev_length=0"
              << " runtime_support_rank_combinadics=0"
              << " cross_route_shift_next_bitops=0"
              << " destination_support_precomputed=1"
              << " host_batch_barriers=0"
              << " pipeline_slots=" << B300_PIPELINE_SLOTS
              << " cycle_closed_batches=1"
              << " double_scratch=1"
              << " remote_state_reads=0"
              << " native_peer_atomics_required=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "B300 free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        for (int slot = 0; slot < B300_PIPELINE_SLOTS; ++slot) {
            cudaEventDestroy(c.consumed[slot]);
            cudaEventDestroy(c.ready[slot]);
        }
        cudaStreamDestroy(c.consume);
        cudaStreamDestroy(c.produce);
        cudaFree(c.error);
        cudaFree(c.peer_state);
        cudaFree(c.state);
        cudaFree(c.local_list);
        cudaFree(c.tasks);
        for (int slot = 0; slot < B300_PIPELINE_SLOTS; ++slot)
            cudaFree(c.scratch[slot]);
        cudaFree(c.owner_begin);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 4;
    const int S = argc > 3 ? std::atoi(argv[3]) : 4;
    const int batches = argc > 4 ? std::atoi(argv[4]) : 4;
    const unsigned blocks_per_sm = argc > 5
        ? static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10)) : 16u;
    const int ngpu = argc > 6 ? std::atoi(argv[6]) : 8;

    if (W < 7 || W > NEXT_SUPPORT_LUT_MAX_W || Kwin < 1 || S != Kwin ||
        Kwin + S + 2 != W ||
        batches < 2 || batches > PERSISTENT_BATCH_MAX ||
        (batches & (batches - 1)) != 0 ||
        !blocks_per_sm || blocks_per_sm > 64 ||
        ngpu < 2 || ngpu > SCRATCH_FULL_MAX_GPU) {
        return 2;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "B300 feed pipeline device count");
    if (visible < ngpu) return 3;

    const double forward_rank_ms =
        install_support_rank_lut(W, Kwin, S, false, ngpu);
    const double forward_next_ms =
        install_next_support_lut(W, Kwin, S, false, ngpu);
    print_route_lut_stats(W, "forward", forward_rank_ms, forward_next_ms);
    run_b300_feed_pipeline(
        W, Kwin, S, false, ngpu, batches, blocks_per_sm);

    const double reverse_rank_ms =
        install_support_rank_lut(W, Kwin, S, true, ngpu);
    const double reverse_next_ms =
        install_next_support_lut(W, Kwin, S, true, ngpu);
    print_route_lut_stats(W, "reverse", reverse_rank_ms, reverse_next_ms);
    run_b300_feed_pipeline(
        W, Kwin, S, true, ngpu, batches, blocks_per_sm);

    std::cout
        << "ALL_OK gridfp_p2p_cycle_owner_b300_feed_pipeline=1"
        << " peer_word_audit_atomics=0"
        << " warp_per_segment=1"
        << " local_cycle_warp_kernel=1"
        << " blocks_per_sm=" << blocks_per_sm << '\n';
    return 0;
}
