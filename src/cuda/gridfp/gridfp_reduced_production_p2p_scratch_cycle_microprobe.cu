#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_token_cycle_microprobe_main_unused
#include "gridfp_reduced_production_p2p_token_cycle_microprobe.cu"
#pragma pop_macro("main")

namespace {

struct ScratchCyclePlan {
    TokenPlan token{};
    Rank64 scratch_offset[TOKEN_MAX_ROUTE]{};
};

void enable_scratch_peer_mesh(int ngpu) {
    for (int src = 0; src < ngpu; ++src) {
        ck(cudaSetDevice(src), "scratch cycle set peer source");
        for (int dst = 0; dst < ngpu; ++dst) {
            if (src == dst) continue;
            int can = 0;
            ck(cudaDeviceCanAccessPeer(&can, src, dst),
               "scratch cycle can access peer");
            if (!can) {
                std::cerr << "scratch cycle peer access unavailable src="
                          << src << " dst=" << dst << '\n';
                std::exit(281);
            }
            const cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                ck(e, "scratch cycle enable peer");
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        }
    }
}

ScratchCyclePlan build_scratch_cycle_plan(
    const RouteProbeResult& route,
    int ngpu,
    std::vector<Rank64>& scratch_words
) {
    ScratchCyclePlan out;
    out.token = build_token_plan(route, ngpu);
    scratch_words.assign(static_cast<std::size_t>(ngpu), 0);
    const Rank64 pc = static_cast<Rank64>(out.token.primitive_count);
    for (int s = 0; s < out.token.segment_count; ++s) {
        const int owner = out.token.segment_owner[s];
        out.scratch_offset[s] = scratch_words[static_cast<std::size_t>(owner)];
        scratch_words[static_cast<std::size_t>(owner)] += pc;
    }
    return out;
}

__global__ void scratch_cycle_phase_a_kernel(
    std::uint32_t* local_state,
    std::uint32_t* scratch,
    const ScratchCyclePlan* plan_ptr,
    int gpu,
    int* error
) {
    if (blockIdx.x != 0) return;
    const ScratchCyclePlan plan = *plan_ptr;
    const TokenPlan p = plan.token;
    const int pc = p.primitive_count;
    if (pc <= 0 || pc > TOKEN_MAX_PRIMITIVE) {
        if (threadIdx.x == 0) atomicCAS(error, 0, 282);
        return;
    }

    for (int s = 0; s < p.segment_count; ++s) {
        if (p.segment_owner[s] != gpu) continue;
        const int begin = p.segment_begin[s];
        const int tail = begin + p.segment_len[s] - 1;
        const Rank64 scratch_base = plan.scratch_offset[s];

        for (int i = threadIdx.x; i < pc; i += blockDim.x) {
            scratch[scratch_base + static_cast<Rank64>(i)] =
                local_state[p.route_local[tail] + static_cast<Rank64>(i)];
            for (int h = tail; h > begin; --h) {
                local_state[p.route_local[h] + static_cast<Rank64>(i)] =
                    local_state[p.route_local[h - 1] + static_cast<Rank64>(i)];
            }
        }
        __syncthreads();
    }
}

__global__ void scratch_cycle_phase_b_kernel(
    std::uint32_t* const* peer_state,
    const std::uint32_t* scratch,
    const ScratchCyclePlan* plan_ptr,
    int gpu,
    unsigned long long* peer_words,
    int* error
) {
    if (blockIdx.x != 0) return;
    const ScratchCyclePlan plan = *plan_ptr;
    const TokenPlan p = plan.token;
    const int pc = p.primitive_count;
    if (pc <= 0 || pc > TOKEN_MAX_PRIMITIVE) {
        if (threadIdx.x == 0) atomicCAS(error, 0, 283);
        return;
    }

    unsigned long long local_peer = 0;
    for (int s = 0; s < p.segment_count; ++s) {
        if (p.segment_owner[s] != gpu) continue;
        const int next = (s + 1) % p.segment_count;
        const int dst_owner = p.segment_owner[next];
        if (dst_owner == gpu) {
            if (threadIdx.x == 0) atomicCAS(error, 0, 284);
            continue;
        }
        const Rank64 dst = p.route_local[p.segment_begin[next]];
        const Rank64 scratch_base = plan.scratch_offset[s];
        for (int i = threadIdx.x; i < pc; i += blockDim.x) {
            peer_state[dst_owner][dst + static_cast<Rank64>(i)] =
                scratch[scratch_base + static_cast<Rank64>(i)];
            ++local_peer;
        }
        __syncthreads();
    }
    if (local_peer) atomicAdd(peer_words, local_peer);
}

int route_owner_at(const TokenPlan& p, int h) {
    for (int s = 0; s < p.segment_count; ++s) {
        const int begin = p.segment_begin[s];
        const int end = begin + p.segment_len[s];
        if (h >= begin && h < end) return p.segment_owner[s];
    }
    fail("scratch cycle route owner");
    return -1;
}

void run_real_scratch_cycle(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu
) {
    ProductionFactorTables tables(W);
    const HostTilePlan host_plan = make_host_tile_plan(tables, Kwin, ngpu);
    ck(cudaSetDevice(0), "scratch cycle tables set device");
    install_tables(tables);

    const RouteProbeResult route =
        find_route(W, Kwin, S, reverse, ngpu, host_plan);
    std::vector<Rank64> scratch_words;
    const ScratchCyclePlan plan =
        build_scratch_cycle_plan(route, ngpu, scratch_words);

    std::vector<std::uint32_t> input;
    std::vector<std::uint32_t> unused_full_reference;
    build_shift_boundary_vectors(
        W, Kwin, S, reverse, ngpu, tables, host_plan,
        input, unused_full_reference);
    std::vector<std::uint32_t> expected = input;

    const TokenPlan& p = plan.token;
    const int pc = p.primitive_count;
    for (int h = 0; h < p.route_len; ++h) {
        const int next = (h + 1) % p.route_len;
        const int src_owner = route_owner_at(p, h);
        const int dst_owner = route_owner_at(p, next);
        const Rank64 src = host_plan.shard_base[static_cast<std::size_t>(src_owner)] +
                           p.route_local[h];
        const Rank64 dst = host_plan.shard_base[static_cast<std::size_t>(dst_owner)] +
                           p.route_local[next];
        for (int i = 0; i < pc; ++i) {
            expected[static_cast<std::size_t>(dst + static_cast<Rank64>(i))] =
                input[static_cast<std::size_t>(src + static_cast<Rank64>(i))];
        }
    }

    enable_scratch_peer_mesh(ngpu);
    std::vector<std::uint32_t*> d_state(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<std::uint32_t*> d_scratch(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<std::uint32_t**> d_peer(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<ScratchCyclePlan*> d_plan(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<unsigned long long*> d_peer_words(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<int*> d_error(static_cast<std::size_t>(ngpu), nullptr);

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch cycle alloc set device");
        const Rank64 n = host_plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&d_state[static_cast<std::size_t>(g)],
                      n * sizeof(std::uint32_t)), "scratch cycle alloc state");
        ck(cudaMemcpy(
               d_state[static_cast<std::size_t>(g)],
               input.data() + host_plan.shard_base[static_cast<std::size_t>(g)],
               n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "scratch cycle copy state");
        const Rank64 sw = std::max<Rank64>(1, scratch_words[static_cast<std::size_t>(g)]);
        ck(cudaMalloc(&d_scratch[static_cast<std::size_t>(g)],
                      sw * sizeof(std::uint32_t)), "scratch cycle alloc scratch");
        ck(cudaMalloc(&d_plan[static_cast<std::size_t>(g)], sizeof(ScratchCyclePlan)),
           "scratch cycle alloc plan");
        ck(cudaMemcpy(d_plan[static_cast<std::size_t>(g)], &plan,
                      sizeof(plan), cudaMemcpyHostToDevice),
           "scratch cycle copy plan");
        ck(cudaMalloc(&d_peer_words[static_cast<std::size_t>(g)],
                      sizeof(unsigned long long)), "scratch cycle alloc peer words");
        ck(cudaMemset(d_peer_words[static_cast<std::size_t>(g)], 0,
                      sizeof(unsigned long long)), "scratch cycle zero peer words");
        ck(cudaMalloc(&d_error[static_cast<std::size_t>(g)], sizeof(int)),
           "scratch cycle alloc error");
        ck(cudaMemset(d_error[static_cast<std::size_t>(g)], 0, sizeof(int)),
           "scratch cycle zero error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch cycle peer table set device");
        ck(cudaMalloc(&d_peer[static_cast<std::size_t>(g)],
                      ngpu * sizeof(std::uint32_t*)),
           "scratch cycle alloc peer table");
        ck(cudaMemcpy(d_peer[static_cast<std::size_t>(g)], d_state.data(),
                      ngpu * sizeof(std::uint32_t*), cudaMemcpyHostToDevice),
           "scratch cycle copy peer table");
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch cycle phase A set device");
        scratch_cycle_phase_a_kernel<<<1, THREADS>>>(
            d_state[static_cast<std::size_t>(g)],
            d_scratch[static_cast<std::size_t>(g)],
            d_plan[static_cast<std::size_t>(g)], g,
            d_error[static_cast<std::size_t>(g)]);
        ck(cudaGetLastError(), "scratch cycle phase A launch");
    }
    // This host-side all-device synchronization is the only global barrier
    // required by the two-phase algorithm. No native peer atomic is involved.
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch cycle phase A sync set device");
        ck(cudaDeviceSynchronize(), "scratch cycle phase A sync");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch cycle phase B set device");
        scratch_cycle_phase_b_kernel<<<1, THREADS>>>(
            d_peer[static_cast<std::size_t>(g)],
            d_scratch[static_cast<std::size_t>(g)],
            d_plan[static_cast<std::size_t>(g)], g,
            d_peer_words[static_cast<std::size_t>(g)],
            d_error[static_cast<std::size_t>(g)]);
        ck(cudaGetLastError(), "scratch cycle phase B launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch cycle phase B sync set device");
        ck(cudaDeviceSynchronize(), "scratch cycle phase B sync");
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> output(input.size());
    unsigned long long peer_words = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch cycle gather set device");
        int error = 0;
        unsigned long long words = 0;
        ck(cudaMemcpy(&error, d_error[static_cast<std::size_t>(g)], sizeof(error),
                      cudaMemcpyDeviceToHost), "scratch cycle copy error");
        ck(cudaMemcpy(&words, d_peer_words[static_cast<std::size_t>(g)],
                      sizeof(words), cudaMemcpyDeviceToHost),
           "scratch cycle copy peer words");
        if (error) fail("scratch cycle device error=" + std::to_string(error));
        peer_words += words;
        const Rank64 n = host_plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + host_plan.shard_base[static_cast<std::size_t>(g)],
               d_state[static_cast<std::size_t>(g)],
               n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "scratch cycle gather state");
    }
    if (output != expected) fail("scratch cycle redistribution mismatch");

    const unsigned long long expected_peer_words =
        static_cast<unsigned long long>(p.segment_count) *
        static_cast<unsigned long long>(pc);
    if (peer_words != expected_peer_words)
        fail("scratch cycle peer word count mismatch");

    Rank64 max_scratch = 0, total_scratch = 0;
    for (Rank64 x : scratch_words) {
        max_scratch = std::max(max_scratch, x);
        total_scratch += x;
    }
    std::cout << "gridfp-p2p-scratch-cycle"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " route_len=" << p.route_len
              << " owner_segments=" << p.segment_count
              << " primitive_count=" << pc
              << " peer_words=" << peer_words
              << " peer_KiB="
              << double(peer_words) * sizeof(std::uint32_t) / 1024.0
              << " max_gpu_scratch_KiB="
              << double(max_scratch) * sizeof(std::uint32_t) / 1024.0
              << " total_scratch_KiB="
              << double(total_scratch) * sizeof(std::uint32_t) / 1024.0
              << " wall_ms=" << ms
              << " phase_count=2"
              << " remote_state_reads=0"
              << " native_peer_atomics_required=0"
              << " peer_writes_equal_owner_crossings=1 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch cycle free set device");
        cudaFree(d_error[static_cast<std::size_t>(g)]);
        cudaFree(d_peer_words[static_cast<std::size_t>(g)]);
        cudaFree(d_plan[static_cast<std::size_t>(g)]);
        cudaFree(d_peer[static_cast<std::size_t>(g)]);
        cudaFree(d_scratch[static_cast<std::size_t>(g)]);
        cudaFree(d_state[static_cast<std::size_t>(g)]);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 4;
    const int S = argc > 3 ? std::atoi(argv[3]) : 3;
    const int ngpu = argc > 4 ? std::atoi(argv[4]) : 8;
    if (W < 7 || W > 11 || Kwin < 2 || S < 1 || S > Kwin ||
        Kwin + S + 2 > W || ngpu < 2 || ngpu > TOKEN_MAX_GPU) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "scratch cycle device count");
    if (visible < ngpu) return 3;

    run_real_scratch_cycle(W, Kwin, S, false, ngpu);
    run_real_scratch_cycle(W, Kwin, S, true, ngpu);
    std::cout << "ALL_OK gridfp_p2p_scratch_cycle=1\n";
    return 0;
}
