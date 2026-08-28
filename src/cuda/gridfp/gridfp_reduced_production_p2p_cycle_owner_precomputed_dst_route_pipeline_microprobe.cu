#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_cycle_owner_compact_route_pipeline_microprobe_main_unused
#include "gridfp_reduced_production_p2p_cycle_owner_compact_route_pipeline_microprobe.cu"
#pragma pop_macro("main")

namespace {

struct PrecomputedDstRouteCtx : RoutePipelineCtx {
    std::uint32_t* segment_dst_support = nullptr;
};

__device__ bool precomputed_dst_route_destination_device(
    std::uint32_t dst_support,
    bool blocked,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int src_owner,
    int ngpu,
    const Rank64* owner_begin,
    RouteDestination& out
) {
    out = RouteDestination{};
    const GroupedDeviceRank dst = grouped_support_slab_rank_device(
        dst_support, blocked, W, q, reverse, tile_start,
        Kwin, ngpu, owner_begin);
    if (dst.owner < 0 || dst.owner >= ngpu || dst.owner == src_owner) {
        out.status = -1;
        return false;
    }
    out.owner = dst.owner;
    out.local = dst.local;
    out.status = 1;
    return true;
}

__global__ void precomputed_dst_route_cycle_owner_phase_b_kernel(
    std::uint32_t* const* __restrict__ peer_state,
    const std::uint32_t* __restrict__ scratch,
    const std::uint16_t* __restrict__ entry_meta,
    const std::uint32_t* __restrict__ segment_dst_support,
    unsigned long long count,
    const unsigned long long* __restrict__ class_entry_begin,
    const unsigned long long* __restrict__ class_scratch_begin,
    const unsigned long long* __restrict__ class_segment_begin,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int ngpu,
    int owner,
    const Rank64* __restrict__ owner_begin,
    unsigned long long* peer_words,
    int* error
) {
    __shared__ CompactRouteEntryHeader header;
    __shared__ RouteDestination destination;
    __shared__ int segment_index;

    const unsigned long long first = blockIdx.x;
    const unsigned long long stride = gridDim.x;
    for (unsigned long long ix = first; ix < count; ix += stride) {
        if (threadIdx.x == 0) {
            if (!compact_route_entry_prepare(
                    entry_meta[ix], ix,
                    class_entry_begin, class_scratch_begin,
                    class_segment_begin, W, header)) {
                set_error(error, 411);
            }
        }
        __syncthreads();
        if (*error) return;

        for (int s = 0; s < header.segments; ++s) {
            if (threadIdx.x == 0) {
                segment_index = s;
                const unsigned long long si =
                    header.segment_base + static_cast<unsigned long long>(s);
                if (!precomputed_dst_route_destination_device(
                        segment_dst_support[si], header.blocked != 0,
                        W, q, reverse, tile_start, Kwin,
                        owner, ngpu, owner_begin, destination) ||
                    destination.status != 1) {
                    set_error(error, 412);
                    segment_index = -1;
                }
            }
            __syncthreads();
            if (*error) return;

            if (segment_index >= 0) {
                const unsigned long long slot =
                    header.scratch_base +
                    static_cast<unsigned long long>(segment_index) *
                    header.primitive_count;
                std::uint32_t* dst =
                    peer_state[destination.owner] + destination.local;
                const int pc = static_cast<int>(header.primitive_count);
                for (int i = threadIdx.x; i < pc; i += blockDim.x) {
                    dst[i] = scratch[
                        slot + static_cast<unsigned long long>(i)];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            atomicAdd(
                peer_words,
                static_cast<unsigned long long>(header.segments) *
                header.primitive_count);
        }
        __syncthreads();
    }
}

void run_precomputed_dst_route_cycle_owner_pipeline(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    int batches,
    unsigned requested_blocks
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

    const int class_stride =
        (W + 1) * (CYCLE_OWNER_MAX_SEGMENTS + 1);
    std::vector<PrecomputedDstRouteCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> state_ptr(
        static_cast<std::size_t>(ngpu), nullptr);
    std::vector<std::vector<unsigned long long>> batch_offset(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches + 1)));
    unsigned long long max_slot_words[ROUTE_PIPELINE_SLOTS]{};
    unsigned long long total_route_segments = 0;

    for (int g = 0; g < ngpu; ++g) {
        auto& offsets = batch_offset[static_cast<std::size_t>(g)];
        std::vector<std::uint16_t> meta;
        std::vector<std::uint32_t> route_support;
        std::vector<std::uint8_t> route_len;
        std::vector<std::uint32_t> route_dst_support;
        std::vector<unsigned long long> entry_begin(
            static_cast<std::size_t>(batches * class_stride));
        std::vector<unsigned long long> scratch_begin(
            static_cast<std::size_t>(batches * class_stride));
        std::vector<unsigned long long> segment_begin(
            static_cast<std::size_t>(batches * class_stride));
        unsigned long long slot_words[ROUTE_PIPELINE_SLOTS] = {1, 1};

        for (int b = 0; b < batches; ++b) {
            auto& part =
                lists.batch[static_cast<std::size_t>(g)]
                           [static_cast<std::size_t>(b)];
            std::sort(part.begin(), part.end(), [](const auto& a, const auto& b) {
                if (a.base.occupied != b.base.occupied)
                    return a.base.occupied < b.base.occupied;
                if (a.base.segments != b.base.segments)
                    return a.base.segments < b.base.segments;
                return a.base.packed < b.base.packed;
            });
            offsets[static_cast<std::size_t>(b)] = meta.size();

            std::vector<unsigned long long> class_count(
                static_cast<std::size_t>(class_stride));
            for (const auto& entry : part) {
                const int cls =
                    int(entry.base.occupied) *
                        (CYCLE_OWNER_MAX_SEGMENTS + 1) +
                    int(entry.base.segments);
                ++class_count[static_cast<std::size_t>(cls)];
            }

            unsigned long long entry_cursor = 0;
            unsigned long long scratch_cursor = 0;
            unsigned long long segment_cursor = route_support.size();
            for (int occ = 0; occ <= W; ++occ) {
                const Rank64 pc =
                    tables.primitive[static_cast<std::size_t>(occ)][1];
                for (int nseg = 0; nseg <= CYCLE_OWNER_MAX_SEGMENTS; ++nseg) {
                    const int cls =
                        occ * (CYCLE_OWNER_MAX_SEGMENTS + 1) + nseg;
                    const std::size_t mi =
                        static_cast<std::size_t>(b * class_stride + cls);
                    entry_begin[mi] = entry_cursor;
                    scratch_begin[mi] = scratch_cursor;
                    segment_begin[mi] = segment_cursor;
                    const auto n = class_count[static_cast<std::size_t>(cls)];
                    entry_cursor += n;
                    scratch_cursor +=
                        n * static_cast<unsigned long long>(nseg) * pc;
                    segment_cursor +=
                        n * static_cast<unsigned long long>(nseg);
                }
            }
            if (entry_cursor != part.size() ||
                scratch_cursor !=
                    lists.words[static_cast<std::size_t>(g)]
                               [static_cast<std::size_t>(b)]) {
                fail("precomputed dst route cycle-owner class plan mismatch");
            }

            const int slot = b & (ROUTE_PIPELINE_SLOTS - 1);
            slot_words[slot] = std::max(slot_words[slot], scratch_cursor);

            const auto& source_support =
                lists.route_support[static_cast<std::size_t>(g)]
                                   [static_cast<std::size_t>(b)];
            const auto& source_len =
                lists.route_len[static_cast<std::size_t>(g)]
                               [static_cast<std::size_t>(b)];
            if (source_support.size() != source_len.size())
                fail("precomputed dst route cycle-owner source route mismatch");

            for (const auto& entry : part) {
                meta.push_back(compact_route_entry_meta(entry));
                const bool blocked =
                    (entry.base.packed & PERSISTENT_BLOCKED_BIT) != 0;
                const unsigned long long begin = entry.route_offset;
                const unsigned long long end =
                    begin + static_cast<unsigned long long>(entry.base.segments);
                if (end > source_support.size())
                    fail("precomputed dst route cycle-owner route offset");
                for (unsigned long long si = begin; si < end; ++si) {
                    const std::uint32_t start =
                        source_support[static_cast<std::size_t>(si)];
                    const int len =
                        int(source_len[static_cast<std::size_t>(si)]);
                    if (len < 1 || len > RP_MAX_W)
                        fail("precomputed dst route cycle-owner segment length");
                    route_support.push_back(start);
                    route_len.push_back(static_cast<std::uint8_t>(len));
                    std::uint32_t dst = start;
                    for (int h = 0; h < len; ++h) {
                        dst = hostlist_next(
                            dst, blocked, W, q, Kwin, S, reverse);
                    }
                    if (hostlist_owner(
                            dst, W, Kwin, reverse, ngpu, tables) == g) {
                        fail("precomputed dst route cycle-owner destination owner");
                    }
                    route_dst_support.push_back(dst);
                }
            }
            if (route_support.size() != segment_cursor ||
                route_len.size() != route_support.size() ||
                route_dst_support.size() != route_support.size()) {
                fail("precomputed dst route cycle-owner packed route mismatch");
            }
        }
        offsets[static_cast<std::size_t>(batches)] = meta.size();
        total_route_segments += route_support.size();

        ck(cudaSetDevice(g), "precomputed dst route cycle-owner set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];

        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "precomputed dst route cycle-owner alloc owner begin");
        ck(cudaMemcpy(
               c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
               cudaMemcpyHostToDevice),
           "precomputed dst route cycle-owner copy owner begin");

        ck(cudaMalloc(
               &c.entry_meta,
               std::max<std::size_t>(1, meta.size()) * sizeof(std::uint16_t)),
           "precomputed dst route cycle-owner alloc entry meta");
        if (!meta.empty()) {
            ck(cudaMemcpy(
                   c.entry_meta, meta.data(),
                   meta.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice),
               "precomputed dst route cycle-owner copy entry meta");
        }

        ck(cudaMalloc(
               &c.segment_support,
               std::max<std::size_t>(1, route_support.size()) *
                   sizeof(std::uint32_t)),
           "precomputed dst route cycle-owner alloc segment support");
        if (!route_support.empty()) {
            ck(cudaMemcpy(
                   c.segment_support, route_support.data(),
                   route_support.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice),
               "precomputed dst route cycle-owner copy segment support");
        }

        ck(cudaMalloc(
               &c.segment_len,
               std::max<std::size_t>(1, route_len.size()) * sizeof(std::uint8_t)),
           "precomputed dst route cycle-owner alloc segment len");
        if (!route_len.empty()) {
            ck(cudaMemcpy(
                   c.segment_len, route_len.data(),
                   route_len.size() * sizeof(std::uint8_t),
                   cudaMemcpyHostToDevice),
               "precomputed dst route cycle-owner copy segment len");
        }

        ck(cudaMalloc(
               &c.segment_dst_support,
               std::max<std::size_t>(1, route_dst_support.size()) *
                   sizeof(std::uint32_t)),
           "precomputed dst route cycle-owner alloc destination support");
        if (!route_dst_support.empty()) {
            ck(cudaMemcpy(
                   c.segment_dst_support, route_dst_support.data(),
                   route_dst_support.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice),
               "precomputed dst route cycle-owner copy destination support");
        }

        const auto& local = lists.local[static_cast<std::size_t>(g)];
        ck(cudaMalloc(
               &c.local_list,
               std::max<std::size_t>(1, local.size()) * sizeof(std::uint32_t)),
           "precomputed dst route cycle-owner alloc local list");
        if (!local.empty()) {
            ck(cudaMemcpy(
                   c.local_list, local.data(),
                   local.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice),
               "precomputed dst route cycle-owner copy local list");
        }

        ck(cudaMalloc(
               &c.class_list_begin,
               entry_begin.size() * sizeof(unsigned long long)),
           "precomputed dst route cycle-owner alloc entry begin");
        ck(cudaMalloc(
               &c.class_scratch_begin,
               scratch_begin.size() * sizeof(unsigned long long)),
           "precomputed dst route cycle-owner alloc scratch begin");
        ck(cudaMalloc(
               &c.class_segment_begin,
               segment_begin.size() * sizeof(unsigned long long)),
           "precomputed dst route cycle-owner alloc segment begin");
        ck(cudaMemcpy(
               c.class_list_begin, entry_begin.data(),
               entry_begin.size() * sizeof(unsigned long long),
               cudaMemcpyHostToDevice),
           "precomputed dst route cycle-owner copy entry begin");
        ck(cudaMemcpy(
               c.class_scratch_begin, scratch_begin.data(),
               scratch_begin.size() * sizeof(unsigned long long),
               cudaMemcpyHostToDevice),
           "precomputed dst route cycle-owner copy scratch begin");
        ck(cudaMemcpy(
               c.class_segment_begin, segment_begin.data(),
               segment_begin.size() * sizeof(unsigned long long),
               cudaMemcpyHostToDevice),
           "precomputed dst route cycle-owner copy segment begin");

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)),
           "precomputed dst route cycle-owner alloc state");
        state_ptr[static_cast<std::size_t>(g)] = c.state;
        ck(cudaMemcpy(
               c.state,
               input.data() + plan.shard_base[static_cast<std::size_t>(g)],
               n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "precomputed dst route cycle-owner copy state");

        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "precomputed dst route cycle-owner alloc peer table");
        ck(cudaMalloc(&c.error, sizeof(int)),
           "precomputed dst route cycle-owner alloc error");
        ck(cudaMemset(c.error, 0, sizeof(int)),
           "precomputed dst route cycle-owner clear error");

        for (int s = 0; s < ROUTE_PIPELINE_SLOTS; ++s) {
            max_slot_words[s] = std::max(max_slot_words[s], slot_words[s]);
            ck(cudaMalloc(
                   &c.scratch[s], slot_words[s] * sizeof(std::uint32_t)),
               "precomputed dst route cycle-owner alloc scratch slot");
            ck(cudaMalloc(
                   &c.peer_words[s], sizeof(unsigned long long)),
               "precomputed dst route cycle-owner alloc peer words slot");
        }

        ck(cudaStreamCreateWithFlags(&c.produce, cudaStreamNonBlocking),
           "precomputed dst route cycle-owner create produce stream");
        ck(cudaStreamCreateWithFlags(&c.consume, cudaStreamNonBlocking),
           "precomputed dst route cycle-owner create consume stream");
        for (int s = 0; s < ROUTE_PIPELINE_SLOTS; ++s) {
            ck(cudaEventCreateWithFlags(&c.ready[s], cudaEventDisableTiming),
               "precomputed dst route cycle-owner create ready event");
            ck(cudaEventCreateWithFlags(&c.consumed[s], cudaEventDisableTiming),
               "precomputed dst route cycle-owner create consumed event");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "precomputed dst route cycle-owner peer set device");
        ck(cudaMemcpy(
               ctx[static_cast<std::size_t>(g)].peer_state,
               state_ptr.data(), ngpu * sizeof(std::uint32_t*),
               cudaMemcpyHostToDevice),
           "precomputed dst route cycle-owner copy peer table");
    }
    const double setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup0).count();

    for (int g = 0; g < ngpu; ++g) {
        const auto count = lists.local[static_cast<std::size_t>(g)].size();
        if (!count) continue;
        ck(cudaSetDevice(g), "precomputed dst route cycle-owner local set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const unsigned blocks = static_cast<unsigned>(
            std::max<std::size_t>(
                1, std::min<std::size_t>(requested_blocks, count)));
        persistent_local_cycle_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            c.state, c.local_list, count,
            W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "precomputed dst route cycle-owner local launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "precomputed dst route cycle-owner local sync set device");
        ck(cudaDeviceSynchronize(), "precomputed dst route cycle-owner local sync");
        int error = 0;
        ck(cudaMemcpy(
               &error, ctx[static_cast<std::size_t>(g)].error,
               sizeof(error), cudaMemcpyDeviceToHost),
           "precomputed dst route cycle-owner local copy error");
        if (error) {
            fail("precomputed dst route cycle-owner local device error=" +
                 std::to_string(error));
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int batch = 0; batch < batches; ++batch) {
        const int slot = batch & (ROUTE_PIPELINE_SLOTS - 1);

        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "precomputed dst route cycle-owner phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            if (batch >= ROUTE_PIPELINE_SLOTS) {
                ck(cudaStreamWaitEvent(c.produce, c.consumed[slot], 0),
                   "precomputed dst route cycle-owner wait slot consumed");
            }
            ck(cudaMemsetAsync(
                   c.peer_words[slot], 0,
                   sizeof(unsigned long long), c.produce),
               "precomputed dst route cycle-owner zero peer words");

            const auto count =
                lists.batch[static_cast<std::size_t>(g)]
                           [static_cast<std::size_t>(batch)].size();
            if (count) {
                const unsigned blocks = static_cast<unsigned>(
                    std::max<std::size_t>(
                        1, std::min<std::size_t>(requested_blocks, count)));
                const auto offset =
                    batch_offset[static_cast<std::size_t>(g)]
                                [static_cast<std::size_t>(batch)];
                compact_route_cycle_owner_phase_a_kernel<<<
                    blocks, SCRATCH_FULL_THREADS, 0, c.produce>>>(
                    c.state, c.scratch[slot],
                    c.entry_meta + offset,
                    c.segment_support, c.segment_len, count,
                    c.class_list_begin + batch * class_stride,
                    c.class_scratch_begin + batch * class_stride,
                    c.class_segment_begin + batch * class_stride,
                    W, q, reverse, tile_start, Kwin, S,
                    ngpu, g, c.owner_begin, c.error);
                ck(cudaGetLastError(),
                   "precomputed dst route cycle-owner phase A launch");
            }
            ck(cudaEventRecord(c.ready[slot], c.produce),
               "precomputed dst route cycle-owner record ready");
        }

        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "precomputed dst route cycle-owner phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            for (int src = 0; src < ngpu; ++src) {
                ck(cudaStreamWaitEvent(
                       c.consume,
                       ctx[static_cast<std::size_t>(src)].ready[slot], 0),
                   "precomputed dst route cycle-owner wait all phase A");
            }

            const auto count =
                lists.batch[static_cast<std::size_t>(g)]
                           [static_cast<std::size_t>(batch)].size();
            const auto expected_words =
                lists.words[static_cast<std::size_t>(g)]
                           [static_cast<std::size_t>(batch)];
            if (count) {
                const unsigned blocks = static_cast<unsigned>(
                    std::max<std::size_t>(
                        1, std::min<std::size_t>(requested_blocks, count)));
                const auto offset =
                    batch_offset[static_cast<std::size_t>(g)]
                                [static_cast<std::size_t>(batch)];
                precomputed_dst_route_cycle_owner_phase_b_kernel<<<
                    blocks, SCRATCH_FULL_THREADS, 0, c.consume>>>(
                    c.peer_state, c.scratch[slot],
                    c.entry_meta + offset,
                    c.segment_dst_support, count,
                    c.class_list_begin + batch * class_stride,
                    c.class_scratch_begin + batch * class_stride,
                    c.class_segment_begin + batch * class_stride,
                    W, q, reverse, tile_start, Kwin,
                    ngpu, g, c.owner_begin,
                    c.peer_words[slot], c.error);
                ck(cudaGetLastError(),
                   "precomputed dst route cycle-owner phase B launch");
            }
            route_pipeline_audit_kernel<<<1, 1, 0, c.consume>>>(
                c.peer_words[slot], expected_words, c.error);
            ck(cudaGetLastError(),
               "precomputed dst route cycle-owner audit launch");
            ck(cudaEventRecord(c.consumed[slot], c.consume),
               "precomputed dst route cycle-owner record consumed");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "precomputed dst route cycle-owner final sync set device");
        ck(cudaStreamSynchronize(ctx[static_cast<std::size_t>(g)].consume),
           "precomputed dst route cycle-owner final consume sync");
        ck(cudaStreamSynchronize(ctx[static_cast<std::size_t>(g)].produce),
           "precomputed dst route cycle-owner final produce sync");
    }
    const double runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> output(input.size());
    unsigned long long cross_entries = 0;
    unsigned long long local_entries = 0;
    unsigned long long total_peer_words = 0;
    unsigned long long local_list_bytes = 0;
    unsigned long long cross_entry_meta_bytes = 0;

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "precomputed dst route cycle-owner gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(
               &error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "precomputed dst route cycle-owner copy error");
        if (error) {
            fail("precomputed dst route cycle-owner device error=" +
                 std::to_string(error));
        }

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "precomputed dst route cycle-owner gather state");

        const auto local_count =
            lists.local[static_cast<std::size_t>(g)].size();
        local_entries += local_count;
        local_list_bytes +=
            static_cast<unsigned long long>(local_count) * sizeof(std::uint32_t);
        for (int b = 0; b < batches; ++b) {
            const auto nentry =
                lists.batch[static_cast<std::size_t>(g)]
                           [static_cast<std::size_t>(b)].size();
            cross_entries += nentry;
            cross_entry_meta_bytes +=
                static_cast<unsigned long long>(nentry) * sizeof(std::uint16_t);
            total_peer_words +=
                lists.words[static_cast<std::size_t>(g)]
                           [static_cast<std::size_t>(b)];
        }
    }

    if (output != expected)
        fail("precomputed dst route cycle-owner pipeline redistribution mismatch");

    const unsigned long long segment_route_bytes =
        total_route_segments *
        (2ULL * sizeof(std::uint32_t) + sizeof(std::uint8_t));
    const unsigned long long class_meta_bytes =
        static_cast<unsigned long long>(ngpu) * batches * class_stride *
        3ULL * sizeof(unsigned long long);
    const unsigned long long static_index_bytes =
        local_list_bytes + cross_entry_meta_bytes +
        segment_route_bytes + class_meta_bytes;

    std::cout << "gridfp-p2p-cycle-owner-precomputed-dst-route-pipeline"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " batches=" << batches
              << " states=" << tables.size()
              << " cross_entries=" << cross_entries
              << " local_entries=" << local_entries
              << " route_segments=" << total_route_segments
              << " local_list_KiB=" << double(local_list_bytes) / 1024.0
              << " cross_entry_meta_KiB="
              << double(cross_entry_meta_bytes) / 1024.0
              << " segment_route_KiB="
              << double(segment_route_bytes) / 1024.0
              << " class_meta_KiB=" << double(class_meta_bytes) / 1024.0
              << " static_index_KiB=" << double(static_index_bytes) / 1024.0
              << " logical_peer_KiB="
              << double(total_peer_words) * sizeof(std::uint32_t) / 1024.0
              << " scratch_even_KiB="
              << double(max_slot_words[0]) * sizeof(std::uint32_t) / 1024.0
              << " scratch_odd_KiB="
              << double(max_slot_words[1]) * sizeof(std::uint32_t) / 1024.0
              << " cross_entry_bytes=2"
              << " route_segment_bytes=9"
              << " cross_packed_list_bytes=0"
              << " gpu_cycle_len_meta_bytes=0"
              << " compact_cross_entries=1"
              << " destination_support_precomputed=1"
              << " runtime_destination_support_shift_steps=0"
              << " setup_destination_support_recompute=1"
              << " descriptor_bytes=0"
              << " scratch_allocator_atomics=0"
              << " segment_count_recompute=0"
              << " runtime_batch_hash_recompute=0"
              << " runtime_cycle_length_recompute=0"
              << " runtime_cycle_support_scan_passes=0"
              << " runtime_owner_boundary_search=0"
              << " segment_routes_precomputed=1"
              << " setup_route_recompute=0"
              << " fused_cycle_route_builder=1"
              << " phase_a_rank_scope=owner_segment"
              << " phase_b_destination_rank_only=1"
              << " startup_gpu_support_scan_passes=0"
              << " runtime_support_scan_passes=0"
              << " runtime_count_passes=0"
              << " setup_ms=" << setup_ms
              << " runtime_ms=" << runtime_ms
              << " pipeline_slots=" << ROUTE_PIPELINE_SLOTS
              << " host_batch_barriers=0"
              << " cross_gpu_phase_a_fence=1"
              << " cycle_closed_batches=1"
              << " double_scratch=1"
              << " remote_state_reads=0"
              << " native_peer_atomics_required=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "precomputed dst route cycle-owner free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        for (int s = 0; s < ROUTE_PIPELINE_SLOTS; ++s) {
            cudaEventDestroy(c.consumed[s]);
            cudaEventDestroy(c.ready[s]);
        }
        cudaStreamDestroy(c.consume);
        cudaStreamDestroy(c.produce);
        for (int s = 0; s < ROUTE_PIPELINE_SLOTS; ++s) {
            cudaFree(c.peer_words[s]);
            cudaFree(c.scratch[s]);
        }
        cudaFree(c.error);
        cudaFree(c.peer_state);
        cudaFree(c.state);
        cudaFree(c.class_segment_begin);
        cudaFree(c.class_scratch_begin);
        cudaFree(c.class_list_begin);
        cudaFree(c.local_list);
        cudaFree(c.segment_dst_support);
        cudaFree(c.segment_len);
        cudaFree(c.segment_support);
        cudaFree(c.entry_meta);
        cudaFree(c.owner_begin);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 4;
    const int S = argc > 3 ? std::atoi(argv[3]) : 4;
    const int batches = argc > 4 ? std::atoi(argv[4]) : 16;
    const unsigned blocks = argc > 5
        ? static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10)) : 256u;
    const int ngpu = argc > 6 ? std::atoi(argv[6]) : 8;

    if (W < 7 || W > 11 || Kwin < 1 || S != Kwin ||
        Kwin + S + 2 != W ||
        batches < 2 || batches > PERSISTENT_BATCH_MAX ||
        (batches & (batches - 1)) != 0 || !blocks ||
        ngpu < 2 || ngpu > SCRATCH_FULL_MAX_GPU) {
        return 2;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible),
       "precomputed dst route cycle-owner pipeline device count");
    if (visible < ngpu) return 3;

    run_precomputed_dst_route_cycle_owner_pipeline(
        W, Kwin, S, false, ngpu, batches, blocks);
    run_precomputed_dst_route_cycle_owner_pipeline(
        W, Kwin, S, true, ngpu, batches, blocks);
    std::cout
        << "ALL_OK gridfp_p2p_cycle_owner_precomputed_dst_route_pipeline=1\n";
    return 0;
}
