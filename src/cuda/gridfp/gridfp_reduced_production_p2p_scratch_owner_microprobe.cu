#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_scratch_full_microprobe_main_unused
#include "gridfp_reduced_production_p2p_scratch_full_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_owner_support_device.cuh"

namespace {

static constexpr int SCRATCH_OWNER_MAX_ROUTE = RP_MAX_W;

struct OwnerLocalSegment {
    int status = 0; // 0=skip, 1=all-local cycle, 2=cross-owner segment, -1=error
    int len = 0;
    int dst_owner = -1;
    Rank64 primitive_count = 0;
    Rank64 local[SCRATCH_OWNER_MAX_ROUTE]{};
    Rank64 dst_local = 0;
};

__device__ __forceinline__ std::uint32_t owner_shift_prev_support_device(
    std::uint32_t support,
    bool blocked,
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse
) {
    const int span = Kwin + S + 2;
    const int lo = reverse ? 0 : W - span;
    if (!blocked) {
        const std::uint32_t low_mask = (std::uint32_t(1) << span) - 1u;
        const std::uint32_t span_mask = low_mask << lo;
        const std::uint32_t x = (support & span_mask) >> lo;
        const int shift = reverse ? S : span - S;
        return (support & ~span_mask) |
               (shift_rotate_bits_device(x, span, shift) << lo);
    }

    const int compact_len = span - 2;
    std::uint32_t compact = 0;
    int cp = 0;
    for (int bit = lo; bit < lo + span; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        if ((support >> bit) & 1u) compact |= std::uint32_t(1) << cp;
        ++cp;
    }
    const int shift = reverse ? S : compact_len - S;
    const std::uint32_t rotated =
        shift_rotate_bits_device(compact, compact_len, shift);
    std::uint32_t out = support;
    cp = 0;
    for (int bit = lo; bit < lo + span; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        out &= ~(std::uint32_t(1) << bit);
        if ((rotated >> cp) & 1u) out |= std::uint32_t(1) << bit;
        ++cp;
    }
    return out;
}

__device__ bool owner_local_segment_device(
    OwnerSupportSlabDevice slab,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int S,
    int owner,
    int ngpu,
    const Rank64* owner_begin,
    OwnerLocalSegment& out
) {
    out = OwnerLocalSegment{};
    if (!slab.valid) {
        out.status = -1;
        return false;
    }
    const bool blocked = slab.blocked != 0;
    if (grouped_support_owner_device(
            slab.support, W, reverse, tile_start, Kwin, ngpu) != owner) {
        out.status = -1;
        return false;
    }

    const std::uint32_t prev = owner_shift_prev_support_device(
        slab.support, blocked, W, q, Kwin, S, reverse);
    const int prev_owner = grouped_support_owner_device(
        prev, W, reverse, tile_start, Kwin, ngpu);

    out.primitive_count = RP_PRIMITIVE[__popc(slab.support)][1];
    if (!out.primitive_count) {
        out.status = -1;
        return false;
    }

    if (prev_owner != owner) {
        // Exactly one such slab exists per maximal segment owned by this GPU.
        std::uint32_t cur = slab.support;
        for (int h = 0; h < SCRATCH_OWNER_MAX_ROUTE; ++h) {
            const GroupedDeviceRank gr = grouped_support_slab_rank_device(
                cur, blocked, W, q, reverse, tile_start, Kwin, ngpu,
                owner_begin);
            if (gr.owner != owner) {
                out.status = -1;
                return false;
            }
            out.local[out.len++] = gr.local;

            const std::uint32_t next = shift_next_support_device(
                cur, blocked, W, q, Kwin, S, reverse);
            const GroupedDeviceRank ngr = grouped_support_slab_rank_device(
                next, blocked, W, q, reverse, tile_start, Kwin, ngpu,
                owner_begin);
            if (ngr.owner != owner) {
                if (ngr.owner < 0 || ngr.owner >= ngpu) {
                    out.status = -1;
                    return false;
                }
                out.dst_owner = ngr.owner;
                out.dst_local = ngr.local;
                out.status = 2;
                return true;
            }
            cur = next;
        }
        out.status = -1;
        return false;
    }

    // Cross-owner cycles are already covered by their segment starts.  The
    // only owner-local slab that may still require work is the unique
    // lexicographic leader of an all-local cycle.
    const int cycle_len = shift_cycle_leader_length_device(
        slab.support, blocked, W, q, Kwin, S, reverse);
    if (cycle_len <= 0) {
        if (cycle_len < 0) out.status = -1;
        return false;
    }
    if (cycle_len > SCRATCH_OWNER_MAX_ROUTE) {
        out.status = -1;
        return false;
    }

    std::uint32_t cur = slab.support;
    for (int h = 0; h < cycle_len; ++h) {
        const GroupedDeviceRank gr = grouped_support_slab_rank_device(
            cur, blocked, W, q, reverse, tile_start, Kwin, ngpu,
            owner_begin);
        if (gr.owner != owner) {
            // Not all-local: some segment start elsewhere owns this cycle.
            out = OwnerLocalSegment{};
            return false;
        }
        out.local[out.len++] = gr.local;
        cur = shift_next_support_device(
            cur, blocked, W, q, Kwin, S, reverse);
    }
    if (cur != slab.support) {
        out.status = -1;
        return false;
    }
    out.status = 1;
    return true;
}

__global__ void scratch_owner_slab_counts_kernel(
    int W,
    int Kwin,
    int ngpu,
    unsigned long long* counts
) {
    const int owner = threadIdx.x;
    if (owner >= ngpu) return;
    counts[owner] = owner_support_slab_count_device(W, Kwin, owner, ngpu);
}

__global__ void scratch_owner_count_kernel(
    Rank64 owner_slabs,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int S,
    int ngpu,
    int owner,
    const Rank64* __restrict__ owner_begin,
    unsigned long long* __restrict__ scratch_words,
    unsigned long long* __restrict__ descriptors,
    unsigned long long* __restrict__ selected_segments,
    int* error
) {
    unsigned long long local_words = 0;
    unsigned long long local_desc = 0;
    unsigned long long local_segments = 0;
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    for (Rank64 rank = first; rank < owner_slabs; rank += stride) {
        const OwnerSupportSlabDevice slab = owner_support_slab_unrank_device(
            W, q, reverse, tile_start, Kwin, owner, ngpu, rank);
        OwnerLocalSegment segment;
        if (!owner_local_segment_device(
                slab, W, q, reverse, tile_start, Kwin, S, owner, ngpu,
                owner_begin, segment)) {
            if (segment.status < 0) set_error(error, 311);
            continue;
        }
        ++local_segments;
        if (segment.status == 2) {
            local_words += segment.primitive_count;
            ++local_desc;
        }
    }
    if (local_words) atomicAdd(scratch_words, local_words);
    if (local_desc) atomicAdd(descriptors, local_desc);
    if (local_segments) atomicAdd(selected_segments, local_segments);
}

__global__ void scratch_owner_phase_a_kernel(
    std::uint32_t* local_state,
    std::uint32_t* scratch,
    ScratchDescriptor* descriptor,
    unsigned long long scratch_capacity,
    unsigned long long descriptor_capacity,
    unsigned long long* scratch_head,
    unsigned long long* descriptor_head,
    Rank64 owner_slabs,
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
    __shared__ OwnerLocalSegment segment;
    __shared__ unsigned long long alloc_offset;
    __shared__ unsigned long long alloc_desc;

    const Rank64 first = Rank64(blockIdx.x);
    const Rank64 stride = Rank64(gridDim.x);
    for (Rank64 rank = first; rank < owner_slabs; rank += stride) {
        if (threadIdx.x == 0) {
            const OwnerSupportSlabDevice slab = owner_support_slab_unrank_device(
                W, q, reverse, tile_start, Kwin, owner, ngpu, rank);
            owner_local_segment_device(
                slab, W, q, reverse, tile_start, Kwin, S, owner, ngpu,
                owner_begin, segment);
            if (segment.status < 0) set_error(error, 312);
        }
        __syncthreads();
        if (segment.status <= 0) continue;

        const int pc = static_cast<int>(segment.primitive_count);
        if (segment.status == 1) {
            const int tail = segment.len - 1;
            for (int i = threadIdx.x; i < pc; i += blockDim.x) {
                const std::uint32_t temp =
                    local_state[segment.local[tail] + static_cast<Rank64>(i)];
                for (int h = tail; h > 0; --h) {
                    local_state[segment.local[h] + static_cast<Rank64>(i)] =
                        local_state[segment.local[h - 1] + static_cast<Rank64>(i)];
                }
                local_state[segment.local[0] + static_cast<Rank64>(i)] = temp;
            }
            __syncthreads();
            continue;
        }

        if (threadIdx.x == 0) {
            alloc_offset = atomicAdd(
                scratch_head,
                static_cast<unsigned long long>(segment.primitive_count));
            alloc_desc = atomicAdd(descriptor_head, 1ULL);
            if (alloc_offset + segment.primitive_count > scratch_capacity ||
                alloc_desc >= descriptor_capacity) {
                set_error(error, 313);
            }
        }
        __syncthreads();
        if (*error) return;

        const int tail = segment.len - 1;
        for (int i = threadIdx.x; i < pc; i += blockDim.x) {
            scratch[alloc_offset + static_cast<unsigned long long>(i)] =
                local_state[segment.local[tail] + static_cast<Rank64>(i)];
            for (int h = tail; h > 0; --h) {
                local_state[segment.local[h] + static_cast<Rank64>(i)] =
                    local_state[segment.local[h - 1] + static_cast<Rank64>(i)];
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            ScratchDescriptor d{};
            d.scratch_offset = alloc_offset;
            d.dst_local = segment.dst_local;
            d.words = static_cast<std::uint32_t>(pc);
            d.dst_owner = static_cast<std::uint8_t>(segment.dst_owner);
            descriptor[alloc_desc] = d;
        }
        __syncthreads();
    }
}

struct ScratchOwnerCtx : ScratchFullCtx {
    unsigned long long* selected_segments = nullptr;
};

void run_scratch_owner(
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
    const unsigned long long total_support_slabs =
        5ULL * (1ULL << (W - 3));

    std::vector<std::uint32_t> input;
    std::vector<std::uint32_t> expected;
    build_shift_boundary_vectors(
        W, Kwin, S, reverse, ngpu, tables, plan, input, expected);
    enable_scratch_full_peer_mesh(ngpu);

    ck(cudaSetDevice(0), "scratch owner slab counts set device");
    install_tables(tables);
    unsigned long long* d_owner_slabs = nullptr;
    ck(cudaMalloc(&d_owner_slabs, ngpu * sizeof(unsigned long long)),
       "scratch owner alloc slab counts");
    scratch_owner_slab_counts_kernel<<<1, ngpu>>>(
        W, Kwin, ngpu, d_owner_slabs);
    ck(cudaGetLastError(), "scratch owner slab counts launch");
    ck(cudaDeviceSynchronize(), "scratch owner slab counts sync");
    std::vector<unsigned long long> owner_slabs(static_cast<std::size_t>(ngpu));
    ck(cudaMemcpy(owner_slabs.data(), d_owner_slabs,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "scratch owner copy slab counts");
    cudaFree(d_owner_slabs);
    unsigned long long scanned_slabs = 0;
    for (auto x : owner_slabs) scanned_slabs += x;
    if (scanned_slabs != total_support_slabs)
        fail("scratch owner slab coverage count");

    std::vector<ScratchOwnerCtx> ctx(static_cast<std::size_t>(ngpu));
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch owner count alloc set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "scratch owner alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                      cudaMemcpyHostToDevice), "scratch owner copy owner begin");
        ck(cudaMalloc(&c.count_words, sizeof(unsigned long long)),
           "scratch owner alloc count words");
        ck(cudaMalloc(&c.count_desc, sizeof(unsigned long long)),
           "scratch owner alloc count desc");
        ck(cudaMalloc(&c.selected_segments, sizeof(unsigned long long)),
           "scratch owner alloc selected segments");
        ck(cudaMalloc(&c.error, sizeof(int)), "scratch owner alloc error");
        ck(cudaMemset(c.count_words, 0, sizeof(unsigned long long)),
           "scratch owner zero count words");
        ck(cudaMemset(c.count_desc, 0, sizeof(unsigned long long)),
           "scratch owner zero count desc");
        ck(cudaMemset(c.selected_segments, 0, sizeof(unsigned long long)),
           "scratch owner zero selected segments");
        ck(cudaMemset(c.error, 0, sizeof(int)), "scratch owner zero error");

        const Rank64 count = owner_slabs[static_cast<std::size_t>(g)];
        const Rank64 needed =
            (count + Rank64(SCRATCH_FULL_THREADS) - 1) /
            Rank64(SCRATCH_FULL_THREADS);
        const unsigned blocks = static_cast<unsigned>(
            std::max<Rank64>(1, std::min<Rank64>(requested_blocks, needed)));
        scratch_owner_count_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            count, W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.count_words, c.count_desc,
            c.selected_segments, c.error);
        ck(cudaGetLastError(), "scratch owner count launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch owner count sync set device");
        ck(cudaDeviceSynchronize(), "scratch owner count sync");
    }

    std::vector<unsigned long long> scratch_words(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> desc_count(static_cast<std::size_t>(ngpu));
    std::vector<unsigned long long> selected_segments(static_cast<std::size_t>(ngpu));
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch owner count copy set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "scratch owner copy count error");
        ck(cudaMemcpy(&scratch_words[static_cast<std::size_t>(g)], c.count_words,
                      sizeof(unsigned long long), cudaMemcpyDeviceToHost),
           "scratch owner copy count words");
        ck(cudaMemcpy(&desc_count[static_cast<std::size_t>(g)], c.count_desc,
                      sizeof(unsigned long long), cudaMemcpyDeviceToHost),
           "scratch owner copy count desc");
        ck(cudaMemcpy(&selected_segments[static_cast<std::size_t>(g)],
                      c.selected_segments, sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost),
           "scratch owner copy selected segments");
        if (error) fail("scratch owner count device error=" + std::to_string(error));
    }

    std::vector<std::uint32_t*> state_ptr(static_cast<std::size_t>(ngpu), nullptr);
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch owner data alloc set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)),
           "scratch owner alloc state");
        state_ptr[static_cast<std::size_t>(g)] = c.state;
        ck(cudaMemcpy(c.state,
                      input.data() + plan.shard_base[static_cast<std::size_t>(g)],
                      n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "scratch owner copy state");
        const auto sw = std::max<unsigned long long>(
            1, scratch_words[static_cast<std::size_t>(g)]);
        const auto dc = std::max<unsigned long long>(
            1, desc_count[static_cast<std::size_t>(g)]);
        ck(cudaMalloc(&c.scratch, sw * sizeof(std::uint32_t)),
           "scratch owner alloc scratch");
        ck(cudaMalloc(&c.descriptor, dc * sizeof(ScratchDescriptor)),
           "scratch owner alloc descriptors");
        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "scratch owner alloc peer table");
        ck(cudaMalloc(&c.head_words, sizeof(unsigned long long)),
           "scratch owner alloc head words");
        ck(cudaMalloc(&c.head_desc, sizeof(unsigned long long)),
           "scratch owner alloc head desc");
        ck(cudaMalloc(&c.peer_words, sizeof(unsigned long long)),
           "scratch owner alloc peer words");
        ck(cudaMemset(c.head_words, 0, sizeof(unsigned long long)),
           "scratch owner zero head words");
        ck(cudaMemset(c.head_desc, 0, sizeof(unsigned long long)),
           "scratch owner zero head desc");
        ck(cudaMemset(c.peer_words, 0, sizeof(unsigned long long)),
           "scratch owner zero peer words");
        ck(cudaMemset(c.error, 0, sizeof(int)), "scratch owner reset error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch owner peer table set device");
        ck(cudaMemcpy(ctx[static_cast<std::size_t>(g)].peer_state,
                      state_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "scratch owner copy peer table");
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch owner phase A set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 count = owner_slabs[static_cast<std::size_t>(g)];
        const unsigned blocks = static_cast<unsigned>(
            std::max<Rank64>(1, std::min<Rank64>(requested_blocks, count)));
        scratch_owner_phase_a_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            c.state, c.scratch, c.descriptor,
            scratch_words[static_cast<std::size_t>(g)],
            desc_count[static_cast<std::size_t>(g)],
            c.head_words, c.head_desc, count,
            W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "scratch owner phase A launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch owner phase A sync set device");
        ck(cudaDeviceSynchronize(), "scratch owner phase A sync");
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch owner phase B set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        if (!desc_count[static_cast<std::size_t>(g)]) continue;
        const unsigned blocks = static_cast<unsigned>(
            std::max<unsigned long long>(1,
                std::min<unsigned long long>(requested_blocks,
                    desc_count[static_cast<std::size_t>(g)])));
        scratch_full_phase_b_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            c.peer_state, c.scratch, c.descriptor,
            desc_count[static_cast<std::size_t>(g)], c.peer_words, c.error);
        ck(cudaGetLastError(), "scratch owner phase B launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch owner phase B sync set device");
        ck(cudaDeviceSynchronize(), "scratch owner phase B sync");
    }
    const double ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> output(input.size());
    unsigned long long total_peer_words = 0;
    unsigned long long total_scratch_words = 0;
    unsigned long long total_desc = 0;
    unsigned long long total_selected_segments = 0;
    unsigned long long max_scratch_words = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch owner gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        unsigned long long head_words = 0, head_desc = 0, peer_words = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "scratch owner copy error");
        ck(cudaMemcpy(&head_words, c.head_words, sizeof(head_words),
                      cudaMemcpyDeviceToHost), "scratch owner copy head words");
        ck(cudaMemcpy(&head_desc, c.head_desc, sizeof(head_desc),
                      cudaMemcpyDeviceToHost), "scratch owner copy head desc");
        ck(cudaMemcpy(&peer_words, c.peer_words, sizeof(peer_words),
                      cudaMemcpyDeviceToHost), "scratch owner copy peer words");
        if (error) fail("scratch owner device error=" + std::to_string(error));
        if (head_words != scratch_words[static_cast<std::size_t>(g)] ||
            head_desc != desc_count[static_cast<std::size_t>(g)])
            fail("scratch owner count/execute mismatch");
        total_peer_words += peer_words;
        total_scratch_words += scratch_words[static_cast<std::size_t>(g)];
        total_desc += desc_count[static_cast<std::size_t>(g)];
        total_selected_segments += selected_segments[static_cast<std::size_t>(g)];
        max_scratch_words = std::max(
            max_scratch_words, scratch_words[static_cast<std::size_t>(g)]);
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "scratch owner gather state");
    }
    if (output != expected) fail("scratch owner redistribution mismatch");
    if (total_peer_words != total_scratch_words)
        fail("scratch owner peer traffic mismatch");

    const unsigned long long replicated_base_scans =
        static_cast<unsigned long long>(ngpu) *
        static_cast<unsigned long long>(base_supports);
    std::cout << "gridfp-p2p-scratch-owner"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " states=" << tables.size()
              << " owner_support_slabs=" << scanned_slabs
              << " selected_segments=" << total_selected_segments
              << " scratch_words=" << total_scratch_words
              << " descriptors=" << total_desc
              << " logical_peer_KiB="
              << double(total_peer_words) * sizeof(std::uint32_t) / 1024.0
              << " max_gpu_scratch_KiB="
              << double(max_scratch_words) * sizeof(std::uint32_t) / 1024.0
              << " wall_ms=" << ms
              << " support_scan_replicas=1"
              << " old_base_scan_work=" << replicated_base_scans
              << " scan_iteration_reduction="
              << double(replicated_base_scans) / double(scanned_slabs)
              << " remote_state_reads=0"
              << " native_peer_atomics_required=0"
              << " peer_writes_equal_logical_lower_bound=1 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "scratch owner free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        cudaFree(c.selected_segments);
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
    const int S = argc > 3 ? std::atoi(argv[3]) : 4;
    const unsigned blocks = argc > 4
        ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)) : 256u;
    const int ngpu = argc > 5 ? std::atoi(argv[5]) : 8;
    if (W < 7 || W > 11 || Kwin < 1 || S < 1 || S != Kwin ||
        Kwin + S + 2 > W || !blocks || ngpu < 2 || ngpu > SCRATCH_FULL_MAX_GPU)
        return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "scratch owner device count");
    if (visible < ngpu) return 3;

    run_scratch_owner(W, Kwin, S, false, ngpu, blocks);
    run_scratch_owner(W, Kwin, S, true, ngpu, blocks);
    std::cout << "ALL_OK gridfp_p2p_scratch_owner=1\n";
    return 0;
}
