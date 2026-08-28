#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_shift_cycle_microprobe_main_unused
#include "gridfp_reduced_production_shift_cycle_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_grouped_support_device.cuh"

#include <array>
#include <vector>

namespace {

static constexpr int SCRATCH_FULL_MAX_GPU = 8;
static constexpr int SCRATCH_FULL_MAX_ROUTE = RP_MAX_W;
static constexpr int SCRATCH_FULL_THREADS = 128;

struct CycleSegments {
    int status = 0; // 0=skip/non-leader, 1=valid, -1=error
    int route_len = 0;
    int segment_count = 0;
    Rank64 primitive_count = 0;
    int owner[SCRATCH_FULL_MAX_ROUTE]{};
    Rank64 local[SCRATCH_FULL_MAX_ROUTE]{};
    int segment_owner[SCRATCH_FULL_MAX_ROUTE]{};
    int segment_begin[SCRATCH_FULL_MAX_ROUTE]{};
    int segment_len[SCRATCH_FULL_MAX_ROUTE]{};
};

struct alignas(32) ScratchDescriptor {
    Rank64 scratch_offset = 0;
    Rank64 dst_local = 0;
    std::uint32_t words = 0;
    std::uint8_t dst_owner = 0;
    std::uint8_t pad[11]{};
};

void enable_scratch_full_peer_mesh(int ngpu) {
    for (int src = 0; src < ngpu; ++src) {
        ck(cudaSetDevice(src), "scratch full set peer source");
        for (int dst = 0; dst < ngpu; ++dst) {
            if (src == dst) continue;
            int can = 0;
            ck(cudaDeviceCanAccessPeer(&can, src, dst),
               "scratch full can access peer");
            if (!can) {
                std::cerr << "scratch full peer unavailable src="
                          << src << " dst=" << dst << '\n';
                std::exit(291);
            }
            const cudaError_t e = cudaDeviceEnablePeerAccess(dst, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled)
                ck(e, "scratch full enable peer");
            if (e == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        }
    }
}

__device__ bool build_cycle_segments_device(
    EqualTileRunSeed run,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int ngpu,
    const Rank64* owner_begin,
    CycleSegments& out
) {
    out = CycleSegments{};
    const bool blocked = run.blocked != 0;
    const int cycle_len = shift_cycle_leader_length_device(
        run.support, blocked, W, q, Kwin, Kwin, reverse);
    if (cycle_len < 0 || cycle_len > SCRATCH_FULL_MAX_ROUTE) {
        out.status = -1;
        return false;
    }
    if (cycle_len <= 1) return false;

    int raw_owner[SCRATCH_FULL_MAX_ROUTE]{};
    Rank64 raw_local[SCRATCH_FULL_MAX_ROUTE]{};
    std::uint32_t support = run.support;
    for (int h = 0; h < cycle_len; ++h) {
        const GroupedDeviceRank gr = grouped_support_slab_rank_device(
            support, blocked, W, q, reverse, tile_start,
            Kwin, ngpu, owner_begin);
        if (gr.owner < 0 || gr.owner >= ngpu) {
            out.status = -1;
            return false;
        }
        raw_owner[h] = gr.owner;
        raw_local[h] = gr.local;
        support = shift_next_support_device(
            support, blocked, W, q, Kwin, Kwin, reverse);
    }
    if (support != run.support) {
        out.status = -1;
        return false;
    }

    int start = -1;
    for (int h = 0; h < cycle_len; ++h) {
        const int prev = (h + cycle_len - 1) % cycle_len;
        if (raw_owner[h] != raw_owner[prev]) {
            start = h;
            break;
        }
    }
    if (start < 0) start = 0;

    out.route_len = cycle_len;
    for (int h = 0; h < cycle_len; ++h) {
        const int src = (start + h) % cycle_len;
        out.owner[h] = raw_owner[src];
        out.local[h] = raw_local[src];
    }

    if (start == 0 && raw_owner[0] == raw_owner[cycle_len - 1]) {
        // This only happens for an all-local cycle because start remains zero
        // when no owner boundary exists.
        bool same = true;
        for (int h = 1; h < cycle_len; ++h)
            same = same && raw_owner[h] == raw_owner[0];
        if (same) {
            out.segment_count = 1;
            out.segment_owner[0] = raw_owner[0];
            out.segment_begin[0] = 0;
            out.segment_len[0] = cycle_len;
        }
    }
    if (out.segment_count == 0) {
        int h = 0;
        while (h < cycle_len) {
            const int s = out.segment_count++;
            const int owner = out.owner[h];
            out.segment_owner[s] = owner;
            out.segment_begin[s] = h;
            int len = 1;
            while (h + len < cycle_len && out.owner[h + len] == owner) ++len;
            out.segment_len[s] = len;
            h += len;
        }
    }

    const int occupied = __popc(run.support);
    out.primitive_count = RP_PRIMITIVE[occupied][1];
    if (!out.primitive_count) {
        out.status = -1;
        return false;
    }
    out.status = 1;
    return true;
}

__global__ void scratch_full_count_kernel(
    Rank64 base_supports,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int ngpu,
    int gpu,
    const Rank64* __restrict__ owner_begin,
    unsigned long long* __restrict__ scratch_words,
    unsigned long long* __restrict__ descriptors,
    int* error
) {
    unsigned long long local_words = 0;
    unsigned long long local_desc = 0;
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            CycleSegments cycle;
            if (!build_cycle_segments_device(
                    seeds[ri], W, q, reverse, tile_start, Kwin, ngpu,
                    owner_begin, cycle)) {
                if (cycle.status < 0) set_error(error, 292);
                continue;
            }
            if (cycle.segment_count <= 1) continue;
            for (int s = 0; s < cycle.segment_count; ++s) {
                if (cycle.segment_owner[s] != gpu) continue;
                local_words += cycle.primitive_count;
                ++local_desc;
            }
        }
    }
    if (local_words) atomicAdd(scratch_words, local_words);
    if (local_desc) atomicAdd(descriptors, local_desc);
}

__global__ void scratch_full_phase_a_kernel(
    std::uint32_t* local_state,
    std::uint32_t* scratch,
    ScratchDescriptor* descriptor,
    unsigned long long scratch_capacity,
    unsigned long long descriptor_capacity,
    unsigned long long* scratch_head,
    unsigned long long* descriptor_head,
    Rank64 base_supports,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int ngpu,
    int gpu,
    const Rank64* __restrict__ owner_begin,
    int* error
) {
    __shared__ EqualTileRunSeed seeds[3];
    __shared__ int nr;
    __shared__ CycleSegments cycle;
    __shared__ unsigned long long alloc_offset;
    __shared__ unsigned long long alloc_desc;

    const Rank64 first = Rank64(blockIdx.x);
    const Rank64 stride = Rank64(gridDim.x);
    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        if (threadIdx.x == 0)
            nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        __syncthreads();

        for (int ri = 0; ri < nr; ++ri) {
            if (threadIdx.x == 0) {
                build_cycle_segments_device(
                    seeds[ri], W, q, reverse, tile_start, Kwin, ngpu,
                    owner_begin, cycle);
                if (cycle.status < 0) set_error(error, 293);
            }
            __syncthreads();
            if (cycle.status <= 0) continue;

            const int pc = static_cast<int>(cycle.primitive_count);
            if (cycle.segment_count == 1) {
                if (cycle.segment_owner[0] == gpu) {
                    const int begin = cycle.segment_begin[0];
                    const int tail = begin + cycle.segment_len[0] - 1;
                    for (int i = threadIdx.x; i < pc; i += blockDim.x) {
                        const std::uint32_t temp =
                            local_state[cycle.local[tail] + static_cast<Rank64>(i)];
                        for (int h = tail; h > begin; --h) {
                            local_state[cycle.local[h] + static_cast<Rank64>(i)] =
                                local_state[cycle.local[h - 1] + static_cast<Rank64>(i)];
                        }
                        local_state[cycle.local[begin] + static_cast<Rank64>(i)] = temp;
                    }
                }
                __syncthreads();
                continue;
            }

            for (int s = 0; s < cycle.segment_count; ++s) {
                if (cycle.segment_owner[s] != gpu) continue;
                if (threadIdx.x == 0) {
                    alloc_offset = atomicAdd(
                        scratch_head,
                        static_cast<unsigned long long>(cycle.primitive_count));
                    alloc_desc = atomicAdd(descriptor_head, 1ULL);
                    if (alloc_offset + cycle.primitive_count > scratch_capacity ||
                        alloc_desc >= descriptor_capacity) {
                        set_error(error, 294);
                    }
                }
                __syncthreads();
                if (*error) return;

                const int begin = cycle.segment_begin[s];
                const int tail = begin + cycle.segment_len[s] - 1;
                for (int i = threadIdx.x; i < pc; i += blockDim.x) {
                    scratch[alloc_offset + static_cast<unsigned long long>(i)] =
                        local_state[cycle.local[tail] + static_cast<Rank64>(i)];
                    for (int h = tail; h > begin; --h) {
                        local_state[cycle.local[h] + static_cast<Rank64>(i)] =
                            local_state[cycle.local[h - 1] + static_cast<Rank64>(i)];
                    }
                }
                __syncthreads();
                if (threadIdx.x == 0) {
                    const int next = (s + 1) % cycle.segment_count;
                    ScratchDescriptor d{};
                    d.scratch_offset = alloc_offset;
                    d.dst_local = cycle.local[cycle.segment_begin[next]];
                    d.words = static_cast<std::uint32_t>(pc);
                    d.dst_owner = static_cast<std::uint8_t>(cycle.segment_owner[next]);
                    descriptor[alloc_desc] = d;
                }
                __syncthreads();
            }
        }
    }
}

__global__ void scratch_full_phase_b_kernel(
    std::uint32_t* const* peer_state,
    const std::uint32_t* scratch,
    const ScratchDescriptor* descriptor,
    unsigned long long descriptor_count,
    unsigned long long* peer_words,
    int* error
) {
    const unsigned long long first = blockIdx.x;
    const unsigned long long stride = gridDim.x;
    unsigned long long local_peer = 0;
    for (unsigned long long di = first; di < descriptor_count; di += stride) {
        const ScratchDescriptor d = descriptor[di];
        if (d.dst_owner >= SCRATCH_FULL_MAX_GPU || !d.words) {
            if (threadIdx.x == 0) set_error(error, 295);
            continue;
        }
        std::uint32_t* const dst = peer_state[d.dst_owner] + d.dst_local;
        const std::uint32_t* const src = scratch + d.scratch_offset;
        for (unsigned int i = threadIdx.x; i < d.words; i += blockDim.x) {
            dst[i] = src[i];
            ++local_peer;
        }
        __syncthreads();
    }
    if (local_peer) atomicAdd(peer_words, local_peer);
}

struct ScratchFullCtx {
    std::uint32_t* state = nullptr;
    std::uint32_t* scratch = nullptr;
    ScratchDescriptor* descriptor = nullptr;
    std::uint32_t** peer_state = nullptr;
    Rank64* owner_begin = nullptr;
    unsigned long long* count_words = nullptr;
    unsigned long long* count_desc = nullptr;
    unsigned long long* head_words = nullptr;
    unsigned long long* head_desc = nullptr;
    unsigned long long* peer_words = nullptr;
    int* error = nullptr;
};

void run_scratch_full(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    unsigned requested_blocks
) {
    ProductionFactorTables tables(W);
    const HostTilePlan plan = make_host_tile_plan(tables, Kwin, ngpu);
    const int tile_start = reverse ? 1 : W - 1;
    const int q = tile_start + (reverse ? S : -S);
    const Rank64 base_supports = Rank64(1) << (W - 2);

    std::vector<std::uint32_t> input;
    std::vector<std::uint32_t> expected;
    build_shift_boundary_vectors(
        W, Kwin, S, reverse, ngpu, tables, plan, input, expected);
    enable_scratch_full_peer_mesh(ngpu);

    std::vector<ScratchFullCtx> ctx(static_cast<std::size_t>(ngpu));
    const Rank64 count_needed =
        (base_supports + Rank64(SCRATCH_FULL_THREADS) - 1) /
        Rank64(SCRATCH_FULL_THREADS);
    const unsigned count_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(requested_blocks, count_needed)));

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch full count alloc set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "scratch full alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                      cudaMemcpyHostToDevice), "scratch full copy owner begin");
        ck(cudaMalloc(&c.count_words, sizeof(unsigned long long)),
           "scratch full alloc count words");
        ck(cudaMalloc(&c.count_desc, sizeof(unsigned long long)),
           "scratch full alloc count desc");
        ck(cudaMalloc(&c.error, sizeof(int)), "scratch full alloc error");
        ck(cudaMemset(c.count_words, 0, sizeof(unsigned long long)),
           "scratch full zero count words");
        ck(cudaMemset(c.count_desc, 0, sizeof(unsigned long long)),
           "scratch full zero count desc");
        ck(cudaMemset(c.error, 0, sizeof(int)), "scratch full zero error");
        scratch_full_count_kernel<<<count_blocks, SCRATCH_FULL_THREADS>>>(
            base_supports, W, q, reverse, tile_start, Kwin, ngpu, g,
            c.owner_begin, c.count_words, c.count_desc, c.error);
        ck(cudaGetLastError(), "scratch full count launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch full count sync set device");
        ck(cudaDeviceSynchronize(), "scratch full count sync");
    }

    std::vector<unsigned long long> scratch_words(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> desc_count(static_cast<std::size_t>(ngpu));
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch full count copy set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "scratch full copy count error");
        ck(cudaMemcpy(&scratch_words[static_cast<std::size_t>(g)], c.count_words,
                      sizeof(unsigned long long), cudaMemcpyDeviceToHost),
           "scratch full copy count words");
        ck(cudaMemcpy(&desc_count[static_cast<std::size_t>(g)], c.count_desc,
                      sizeof(unsigned long long), cudaMemcpyDeviceToHost),
           "scratch full copy count desc");
        if (error) fail("scratch full count device error=" + std::to_string(error));
    }

    std::vector<std::uint32_t*> state_ptr(static_cast<std::size_t>(ngpu), nullptr);
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch full data alloc set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)),
           "scratch full alloc state");
        state_ptr[static_cast<std::size_t>(g)] = c.state;
        ck(cudaMemcpy(c.state,
                      input.data() + plan.shard_base[static_cast<std::size_t>(g)],
                      n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "scratch full copy state");
        const auto sw = std::max<unsigned long long>(1, scratch_words[static_cast<std::size_t>(g)]);
        const auto dc = std::max<unsigned long long>(1, desc_count[static_cast<std::size_t>(g)]);
        ck(cudaMalloc(&c.scratch, sw * sizeof(std::uint32_t)),
           "scratch full alloc scratch");
        ck(cudaMalloc(&c.descriptor, dc * sizeof(ScratchDescriptor)),
           "scratch full alloc descriptors");
        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "scratch full alloc peer table");
        ck(cudaMalloc(&c.head_words, sizeof(unsigned long long)),
           "scratch full alloc head words");
        ck(cudaMalloc(&c.head_desc, sizeof(unsigned long long)),
           "scratch full alloc head desc");
        ck(cudaMalloc(&c.peer_words, sizeof(unsigned long long)),
           "scratch full alloc peer words");
        ck(cudaMemset(c.head_words, 0, sizeof(unsigned long long)),
           "scratch full zero head words");
        ck(cudaMemset(c.head_desc, 0, sizeof(unsigned long long)),
           "scratch full zero head desc");
        ck(cudaMemset(c.peer_words, 0, sizeof(unsigned long long)),
           "scratch full zero peer words");
        ck(cudaMemset(c.error, 0, sizeof(int)), "scratch full reset error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch full peer table set device");
        ck(cudaMemcpy(ctx[static_cast<std::size_t>(g)].peer_state, state_ptr.data(),
                      ngpu * sizeof(std::uint32_t*), cudaMemcpyHostToDevice),
           "scratch full copy peer table");
    }

    const unsigned phase_a_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(requested_blocks, base_supports)));
    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch full phase A set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        scratch_full_phase_a_kernel<<<phase_a_blocks, SCRATCH_FULL_THREADS>>>(
            c.state, c.scratch, c.descriptor,
            scratch_words[static_cast<std::size_t>(g)],
            desc_count[static_cast<std::size_t>(g)],
            c.head_words, c.head_desc,
            base_supports, W, q, reverse, tile_start, Kwin, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "scratch full phase A launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch full phase A sync set device");
        ck(cudaDeviceSynchronize(), "scratch full phase A sync");
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch full phase B set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        if (!desc_count[static_cast<std::size_t>(g)]) continue;
        const unsigned blocks = static_cast<unsigned>(
            std::max<unsigned long long>(1,
                std::min<unsigned long long>(requested_blocks,
                    desc_count[static_cast<std::size_t>(g)])));
        scratch_full_phase_b_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            c.peer_state, c.scratch, c.descriptor,
            desc_count[static_cast<std::size_t>(g)], c.peer_words, c.error);
        ck(cudaGetLastError(), "scratch full phase B launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch full phase B sync set device");
        ck(cudaDeviceSynchronize(), "scratch full phase B sync");
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> output(input.size());
    unsigned long long total_peer_words = 0;
    unsigned long long total_scratch_words = 0;
    unsigned long long total_desc = 0;
    unsigned long long max_scratch_words = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch full gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        unsigned long long head_words = 0, head_desc = 0, peer_words = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "scratch full copy error");
        ck(cudaMemcpy(&head_words, c.head_words, sizeof(head_words),
                      cudaMemcpyDeviceToHost), "scratch full copy head words");
        ck(cudaMemcpy(&head_desc, c.head_desc, sizeof(head_desc),
                      cudaMemcpyDeviceToHost), "scratch full copy head desc");
        ck(cudaMemcpy(&peer_words, c.peer_words, sizeof(peer_words),
                      cudaMemcpyDeviceToHost), "scratch full copy peer words");
        if (error) fail("scratch full device error=" + std::to_string(error));
        if (head_words != scratch_words[static_cast<std::size_t>(g)] ||
            head_desc != desc_count[static_cast<std::size_t>(g)])
            fail("scratch full count/execute mismatch");
        total_peer_words += peer_words;
        total_scratch_words += scratch_words[static_cast<std::size_t>(g)];
        total_desc += desc_count[static_cast<std::size_t>(g)];
        max_scratch_words = std::max(
            max_scratch_words, scratch_words[static_cast<std::size_t>(g)]);
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "scratch full gather state");
    }
    if (output != expected) fail("scratch full redistribution mismatch");
    if (total_peer_words != total_scratch_words)
        fail("scratch full peer traffic mismatch");

    std::cout << "gridfp-p2p-scratch-full"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " states=" << tables.size()
              << " base_supports=" << base_supports
              << " scratch_words=" << total_scratch_words
              << " descriptors=" << total_desc
              << " logical_peer_KiB="
              << double(total_peer_words) * sizeof(std::uint32_t) / 1024.0
              << " max_gpu_scratch_KiB="
              << double(max_scratch_words) * sizeof(std::uint32_t) / 1024.0
              << " wall_ms=" << ms
              << " support_scan_replicas=" << ngpu
              << " count_pass=1 phase_a=1 phase_b=1"
              << " remote_state_reads=0"
              << " native_peer_atomics_required=0"
              << " peer_writes_equal_logical_lower_bound=1 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch full free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        cudaFree(c.error);
        cudaFree(c.peer_words);
        cudaFree(c.head_desc);
        cudaFree(c.head_words);
        cudaFree(c.peer_state);
        cudaFree(c.descriptor);
        cudaFree(c.scratch);
        cudaFree(c.state);
        cudaFree(c.count_desc);
        cudaFree(c.count_words);
        cudaFree(c.owner_begin);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 4;
    const int S = argc > 3 ? std::atoi(argv[3]) : 3;
    const unsigned blocks = argc > 4
        ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)) : 256u;
    const int ngpu = argc > 5 ? std::atoi(argv[5]) : 8;
    if (W < 7 || W > 11 || Kwin < 1 || S < 1 || S != Kwin ||
        Kwin + S + 2 > W || !blocks || ngpu < 2 || ngpu > SCRATCH_FULL_MAX_GPU)
        return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "scratch full device count");
    if (visible < ngpu) return 3;

    run_scratch_full(W, Kwin, S, false, ngpu, blocks);
    run_scratch_full(W, Kwin, S, true, ngpu, blocks);
    std::cout << "ALL_OK gridfp_p2p_scratch_full=1\n";
    return 0;
}
