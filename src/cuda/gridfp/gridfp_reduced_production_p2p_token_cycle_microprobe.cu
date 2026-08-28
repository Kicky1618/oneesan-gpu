#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_shift_cycle_microprobe_main_unused
#include "gridfp_reduced_production_shift_cycle_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_grouped_support_device.cuh"

#include <cuda/atomic>

namespace {

static constexpr int TOKEN_MAX_GPU = 8;
static constexpr int TOKEN_MAX_ROUTE = RP_MAX_W;
static constexpr int TOKEN_MAX_PRIMITIVE = 128;

struct RouteProbeResult {
    int found = 0;
    int route_len = 0;
    int blocked = 0;
    int occupied = 0;
    Rank64 primitive_count = 0;
    int owner[TOKEN_MAX_ROUTE]{};
    Rank64 local[TOKEN_MAX_ROUTE]{};
};

struct TokenPlan {
    int route_len = 0;
    int segment_count = 0;
    int primitive_count = 0;
    int segment_owner[TOKEN_MAX_ROUTE]{};
    int segment_begin[TOKEN_MAX_ROUTE]{};
    int segment_len[TOKEN_MAX_ROUTE]{};
    Rank64 route_local[TOKEN_MAX_ROUTE]{};
    int owner_occurrences[TOKEN_MAX_GPU]{};
};

struct alignas(16) TokenMailbox {
    unsigned int ready = 0;
    int segment = -1;
    unsigned int pad[2]{};
    std::uint32_t token[TOKEN_MAX_PRIMITIVE]{};
};

__global__ void find_cross_owner_shift_cycle_kernel(
    Rank64 base_supports,
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    const Rank64* __restrict__ owner_begin,
    RouteProbeResult* result,
    int* error
) {
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? S : -S);

    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        if (result->found) return;
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const EqualTileRunSeed run = seeds[ri];
            const bool blocked = run.blocked != 0;
            const int cycle_len = shift_cycle_leader_length_device(
                run.support, blocked, W, q, Kwin, S, reverse);
            if (cycle_len < 0) {
                set_error(error, 241);
                continue;
            }
            if (cycle_len <= 1 || cycle_len > TOKEN_MAX_ROUTE) continue;

            int owner[TOKEN_MAX_ROUTE]{};
            Rank64 local[TOKEN_MAX_ROUTE]{};
            std::uint32_t support = run.support;
            bool crosses = false;
            int prev_owner = -1;
            for (int h = 0; h < cycle_len; ++h) {
                const GroupedDeviceRank gr = grouped_support_slab_rank_device(
                    support, blocked, W, q, reverse, old_start,
                    Kwin, ngpu, owner_begin);
                if (gr.owner < 0 || gr.owner >= ngpu) {
                    set_error(error, 242);
                    break;
                }
                owner[h] = gr.owner;
                local[h] = gr.local;
                if (h && gr.owner != prev_owner) crosses = true;
                prev_owner = gr.owner;
                support = shift_next_support_device(
                    support, blocked, W, q, Kwin, S, reverse);
            }
            if (support != run.support) {
                set_error(error, 243);
                continue;
            }
            crosses = crosses || owner[cycle_len - 1] != owner[0];
            if (!crosses) continue;

            const int occupied = __popc(run.support);
            const Rank64 pc = RP_PRIMITIVE[occupied][1];
            if (pc == 0 || pc > TOKEN_MAX_PRIMITIVE) continue;

            if (atomicCAS(&result->found, 0, 1) == 0) {
                result->route_len = cycle_len;
                result->blocked = blocked ? 1 : 0;
                result->occupied = occupied;
                result->primitive_count = pc;
                for (int h = 0; h < cycle_len; ++h) {
                    result->owner[h] = owner[h];
                    result->local[h] = local[h];
                }
            }
            return;
        }
    }
}

TokenPlan build_token_plan(const RouteProbeResult& route, int ngpu) {
    if (!route.found || route.route_len < 2 ||
        route.primitive_count == 0 ||
        route.primitive_count > TOKEN_MAX_PRIMITIVE)
        fail("token cycle route");

    // Rotate the cyclic route to an owner boundary so every equal-owner
    // segment is contiguous in the linear representation.
    int start = 0;
    while (route.owner[start] ==
           route.owner[(start + route.route_len - 1) % route.route_len]) {
        ++start;
        if (start >= route.route_len) fail("token cycle no owner boundary");
    }

    TokenPlan plan;
    plan.route_len = route.route_len;
    plan.primitive_count = static_cast<int>(route.primitive_count);
    int rotated_owner[TOKEN_MAX_ROUTE]{};
    for (int h = 0; h < route.route_len; ++h) {
        const int src = (start + h) % route.route_len;
        rotated_owner[h] = route.owner[src];
        plan.route_local[h] = route.local[src];
    }

    int h = 0;
    while (h < route.route_len) {
        const int s = plan.segment_count++;
        const int owner = rotated_owner[h];
        plan.segment_owner[s] = owner;
        plan.segment_begin[s] = h;
        int len = 1;
        while (h + len < route.route_len &&
               rotated_owner[h + len] == owner) {
            ++len;
        }
        plan.segment_len[s] = len;
        ++plan.owner_occurrences[owner];
        h += len;
    }
    if (plan.segment_count < 2) fail("token cycle segment count");
    if (plan.segment_owner[0] == plan.segment_owner[plan.segment_count - 1])
        fail("token cycle wrapped segment");
    for (int g = ngpu; g < TOKEN_MAX_GPU; ++g)
        if (plan.owner_occurrences[g]) fail("token cycle owner range");
    return plan;
}

void enable_token_peer_mesh(int ngpu) {
    for (int src = 0; src < ngpu; ++src) {
        ck(cudaSetDevice(src), "token cycle peer source");
        for (int dst = 0; dst < ngpu; ++dst) {
            if (src == dst) continue;
            int can = 0, native = 0;
            ck(cudaDeviceCanAccessPeer(&can, src, dst),
               "token cycle can access peer");
            ck(cudaDeviceGetP2PAttribute(
                   &native, cudaDevP2PAttrNativeAtomicSupported, src, dst),
               "token cycle native atomic");
            if (!can || !native) {
                std::cerr << "token cycle requires peer+native atomic src="
                          << src << " dst=" << dst
                          << " can=" << can << " native=" << native << '\n';
                std::exit(244);
            }
            const cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                ck(e, "token cycle enable peer");
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        }
    }
}

__device__ void token_send(
    TokenMailbox* destination,
    const std::uint32_t* token,
    int pc,
    int segment
) {
    for (int i = 0; i < pc; ++i) destination->token[i] = token[i];
    destination->segment = segment;
    cuda::atomic_ref<unsigned int, cuda::thread_scope_system> ready(
        destination->ready);
    ready.store(1u, cuda::memory_order_release);
}

__global__ void token_cycle_worker_kernel(
    std::uint32_t* local_state,
    TokenMailbox* inbox,
    TokenMailbox** mailboxes,
    const TokenPlan* plan,
    int gpu,
    int* error
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const TokenPlan p = *plan;
    const int pc = p.primitive_count;
    if (pc <= 0 || pc > TOKEN_MAX_PRIMITIVE) {
        atomicCAS(error, 0, 245);
        return;
    }

    cuda::atomic_ref<unsigned int, cuda::thread_scope_system> ready(inbox->ready);
    std::uint32_t token[TOKEN_MAX_PRIMITIVE]{};
    int processed = 0;

    if (p.segment_owner[0] == gpu) {
        const int begin = p.segment_begin[0];
        const int len = p.segment_len[0];
        const int tail = begin + len - 1;
        for (int i = 0; i < pc; ++i) {
            token[i] = local_state[p.route_local[tail] + i];
            for (int h = tail; h > begin; --h) {
                local_state[p.route_local[h] + i] =
                    local_state[p.route_local[h - 1] + i];
            }
        }
        ++processed;
        const int next_owner = p.segment_owner[1];
        token_send(mailboxes[next_owner], token, pc, 1);
    }

    const int target = p.owner_occurrences[gpu];
    while (processed < target) {
        while (ready.load(cuda::memory_order_acquire) != 1u) {}
        const int s = inbox->segment;
        if (s <= 0 || s >= p.segment_count || p.segment_owner[s] != gpu) {
            atomicCAS(error, 0, 246);
            return;
        }

        const int begin = p.segment_begin[s];
        const int len = p.segment_len[s];
        const int tail = begin + len - 1;
        std::uint32_t next_token[TOKEN_MAX_PRIMITIVE]{};
        for (int i = 0; i < pc; ++i) {
            const std::uint32_t incoming = inbox->token[i];
            next_token[i] = local_state[p.route_local[tail] + i];
            for (int h = tail; h > begin; --h) {
                local_state[p.route_local[h] + i] =
                    local_state[p.route_local[h - 1] + i];
            }
            local_state[p.route_local[begin] + i] = incoming;
        }
        ready.store(0u, cuda::memory_order_release);
        ++processed;

        const int next_segment = (s + 1) % p.segment_count;
        const int next_owner = p.segment_owner[next_segment];
        token_send(mailboxes[next_owner], next_token, pc, next_segment);
    }

    if (p.segment_owner[0] != gpu) return;

    // The final segment sends its outgoing token back with segment id 0.
    while (ready.load(cuda::memory_order_acquire) != 1u) {}
    if (inbox->segment != 0) {
        atomicCAS(error, 0, 247);
        return;
    }
    const int first = p.segment_begin[0];
    for (int i = 0; i < pc; ++i)
        local_state[p.route_local[first] + i] = inbox->token[i];
    ready.store(0u, cuda::memory_order_release);
}

RouteProbeResult find_route(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    const HostTilePlan& host_plan
) {
    ck(cudaSetDevice(0), "token find set device");
    Rank64* d_owner_begin = nullptr;
    RouteProbeResult* d_result = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_owner_begin, ngpu * sizeof(Rank64)),
       "token find alloc owner begin");
    ck(cudaMemcpy(d_owner_begin, host_plan.owner_begin.data(),
                  ngpu * sizeof(Rank64), cudaMemcpyHostToDevice),
       "token find copy owner begin");
    ck(cudaMalloc(&d_result, sizeof(RouteProbeResult)), "token find alloc result");
    ck(cudaMalloc(&d_error, sizeof(int)), "token find alloc error");
    ck(cudaMemset(d_result, 0, sizeof(RouteProbeResult)), "token find zero result");
    ck(cudaMemset(d_error, 0, sizeof(int)), "token find zero error");

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const Rank64 blocks64 =
        (base_supports + Rank64(THREADS) - 1) / Rank64(THREADS);
    const unsigned blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(256, blocks64)));
    find_cross_owner_shift_cycle_kernel<<<blocks, THREADS>>>(
        base_supports, W, Kwin, S, reverse, ngpu,
        d_owner_begin, d_result, d_error);
    ck(cudaGetLastError(), "token find launch");
    ck(cudaDeviceSynchronize(), "token find sync");

    int error = 0;
    RouteProbeResult result{};
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "token find copy error");
    ck(cudaMemcpy(&result, d_result, sizeof(result), cudaMemcpyDeviceToHost),
       "token find copy result");
    if (error) fail("token find device error=" + std::to_string(error));
    if (!result.found) fail("no cross-owner shift cycle found");

    cudaFree(d_error);
    cudaFree(d_result);
    cudaFree(d_owner_begin);
    return result;
}

void run_real_token_cycle(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu
) {
    ProductionFactorTables tables(W);
    const HostTilePlan host_plan = make_host_tile_plan(tables, Kwin, ngpu);
    ck(cudaSetDevice(0), "token tables set device");
    install_tables(tables);

    const RouteProbeResult route =
        find_route(W, Kwin, S, reverse, ngpu, host_plan);
    const TokenPlan plan = build_token_plan(route, ngpu);

    std::vector<std::uint32_t> input(static_cast<std::size_t>(tables.size()));
    for (Rank64 r = 0; r < tables.size(); ++r) {
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + (r * 2654435761ULL) % 4294967290ULL);
    }
    std::vector<std::uint32_t> expected = input;
    const int pc = plan.primitive_count;
    for (int h = 0; h < plan.route_len; ++h) {
        const int next = (h + 1) % plan.route_len;
        const int src_owner = plan.segment_owner[0];
        (void)src_owner;
        int owner_h = -1, owner_next = -1;
        // Resolve owner from segment coverage. Route length is tiny, so a
        // simple scan keeps the host reference independent of device helpers.
        for (int s = 0; s < plan.segment_count; ++s) {
            const int b = plan.segment_begin[s];
            const int e = b + plan.segment_len[s];
            if (h >= b && h < e) owner_h = plan.segment_owner[s];
            if (next >= b && next < e) owner_next = plan.segment_owner[s];
        }
        if (owner_h < 0 || owner_next < 0) fail("token reference owner");
        const Rank64 src = host_plan.shard_base[owner_h] + plan.route_local[h];
        const Rank64 dst = host_plan.shard_base[owner_next] + plan.route_local[next];
        for (int i = 0; i < pc; ++i)
            expected[static_cast<std::size_t>(dst + i)] =
                input[static_cast<std::size_t>(src + i)];
    }

    enable_token_peer_mesh(ngpu);
    std::vector<std::uint32_t*> d_state(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<TokenMailbox*> d_mailbox(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<TokenMailbox**> d_mailbox_table(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<TokenPlan*> d_plan(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<int*> d_error(static_cast<std::size_t>(ngpu), nullptr);

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "token alloc set device");
        const Rank64 n = host_plan.owner_size[static_cast<std::size_t>(g)];
        const Rank64 alloc_n = std::max<Rank64>(1, n);
        ck(cudaMalloc(&d_state[static_cast<std::size_t>(g)],
                      alloc_n * sizeof(std::uint32_t)), "token alloc state");
        if (n) {
            ck(cudaMemcpy(
                   d_state[static_cast<std::size_t>(g)],
                   input.data() + host_plan.shard_base[static_cast<std::size_t>(g)],
                   n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
               "token copy state");
        }
        ck(cudaMalloc(&d_mailbox[static_cast<std::size_t>(g)], sizeof(TokenMailbox)),
           "token alloc mailbox");
        ck(cudaMemset(d_mailbox[static_cast<std::size_t>(g)], 0, sizeof(TokenMailbox)),
           "token zero mailbox");
        ck(cudaMalloc(&d_plan[static_cast<std::size_t>(g)], sizeof(TokenPlan)),
           "token alloc plan");
        ck(cudaMemcpy(d_plan[static_cast<std::size_t>(g)], &plan, sizeof(plan),
                      cudaMemcpyHostToDevice), "token copy plan");
        ck(cudaMalloc(&d_error[static_cast<std::size_t>(g)], sizeof(int)),
           "token alloc error");
        ck(cudaMemset(d_error[static_cast<std::size_t>(g)], 0, sizeof(int)),
           "token zero error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "token table set device");
        ck(cudaMalloc(&d_mailbox_table[static_cast<std::size_t>(g)],
                      ngpu * sizeof(TokenMailbox*)), "token alloc mailbox table");
        ck(cudaMemcpy(d_mailbox_table[static_cast<std::size_t>(g)],
                      d_mailbox.data(), ngpu * sizeof(TokenMailbox*),
                      cudaMemcpyHostToDevice), "token copy mailbox table");
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "token launch set device");
        token_cycle_worker_kernel<<<1, 1>>>(
            d_state[static_cast<std::size_t>(g)],
            d_mailbox[static_cast<std::size_t>(g)],
            d_mailbox_table[static_cast<std::size_t>(g)],
            d_plan[static_cast<std::size_t>(g)], g,
            d_error[static_cast<std::size_t>(g)]);
        ck(cudaGetLastError(), "token cycle launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "token sync set device");
        ck(cudaDeviceSynchronize(), "token cycle sync");
    }

    std::vector<std::uint32_t> output(input.size());
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "token gather set device");
        int error = 0;
        ck(cudaMemcpy(&error, d_error[static_cast<std::size_t>(g)], sizeof(error),
                      cudaMemcpyDeviceToHost), "token copy error");
        if (error) fail("token cycle GPU error=" + std::to_string(error));
        const Rank64 n = host_plan.owner_size[static_cast<std::size_t>(g)];
        if (n) {
            ck(cudaMemcpy(
                   output.data() + host_plan.shard_base[static_cast<std::size_t>(g)],
                   d_state[static_cast<std::size_t>(g)],
                   n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
               "token gather state");
        }
    }
    if (output != expected) fail("real token cycle redistribution mismatch");

    std::cout << "gridfp-real-p2p-token-cycle"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " route_len=" << plan.route_len
              << " segments=" << plan.segment_count
              << " primitive_count=" << plan.primitive_count
              << " token_bytes=" << plan.primitive_count * sizeof(std::uint32_t)
              << " peer_payload_writes=" << plan.segment_count
              << " remote_state_reads=0"
              << " logical_peer_lower_bound_attained=1"
              << " exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "token free set device");
        cudaFree(d_error[static_cast<std::size_t>(g)]);
        cudaFree(d_plan[static_cast<std::size_t>(g)]);
        cudaFree(d_mailbox_table[static_cast<std::size_t>(g)]);
        cudaFree(d_mailbox[static_cast<std::size_t>(g)]);
        cudaFree(d_state[static_cast<std::size_t>(g)]);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 4;
    const int S = argc > 3 ? std::atoi(argv[3]) : 3;
    const int ngpu = argc > 4 ? std::atoi(argv[4]) : 8;
    if (W < 7 || W > 11 || Kwin < 1 || S < 1 || S > Kwin ||
        Kwin + S + 2 > W || ngpu < 2 || ngpu > TOKEN_MAX_GPU) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "token cycle device count");
    if (visible < ngpu) return 3;

    run_real_token_cycle(W, Kwin, S, false, ngpu);
    run_real_token_cycle(W, Kwin, S, true, ngpu);
    std::cout << "ALL_OK gridfp_real_p2p_token_cycle=1\n";
    return 0;
}
