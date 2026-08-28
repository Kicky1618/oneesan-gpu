#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_cycle_owner_route_pipeline_microprobe_main_unused
#include "gridfp_reduced_production_p2p_cycle_owner_route_pipeline_microprobe.cu"
#pragma pop_macro("main")

namespace {

struct FusedHostCycleOwnerEntry {
    HostCycleOwnerEntry base{};
    std::uint8_t cycle_len = 0;
    unsigned long long route_offset = 0;
};

struct FusedHostCycleOwnerLists {
    std::vector<std::vector<std::vector<FusedHostCycleOwnerEntry>>> batch;
    std::vector<std::vector<std::uint32_t>> local;
    std::vector<std::vector<unsigned long long>> words;
    std::vector<std::vector<std::vector<std::uint32_t>>> route_support;
    std::vector<std::vector<std::vector<std::uint8_t>>> route_len;
};

FusedHostCycleOwnerLists build_fused_cycle_owner_lists(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    int batches,
    const ProductionFactorTables& tables
) {
    FusedHostCycleOwnerLists out;
    out.batch.assign(
        static_cast<std::size_t>(ngpu),
        std::vector<std::vector<FusedHostCycleOwnerEntry>>(
            static_cast<std::size_t>(batches)));
    out.local.assign(static_cast<std::size_t>(ngpu), {});
    out.words.assign(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches)));
    out.route_support.assign(
        static_cast<std::size_t>(ngpu),
        std::vector<std::vector<std::uint32_t>>(
            static_cast<std::size_t>(batches)));
    out.route_len.assign(
        static_cast<std::size_t>(ngpu),
        std::vector<std::vector<std::uint8_t>>(
            static_cast<std::size_t>(batches)));

    const int tile_start = reverse ? 1 : W - 1;
    const int q = tile_start + (reverse ? S : -S);
    const Rank64 base_supports = Rank64(1) << (W - 2);

    for (Rank64 compact = 0; compact < base_supports; ++compact) {
        EqualTileRunSeed seeds[3]{};
        int nr = 0;
        hostlist_expand_seeds(compact, W, q, reverse, seeds, nr);
        for (int ri = 0; ri < nr; ++ri) {
            const auto seed = seeds[ri];
            const bool blocked = seed.blocked != 0;
            const int cycle_len = hostlist_leader_length(
                seed.support, blocked, W, q, Kwin, S, reverse);
            if (cycle_len < 0 || cycle_len > RP_MAX_W)
                fail("fused route cycle-owner host cycle length");
            if (cycle_len <= 1) continue;

            std::uint32_t route_support[RP_MAX_W]{};
            int route_owner[RP_MAX_W]{};
            std::uint32_t cur = seed.support;
            for (int hop = 0; hop < cycle_len; ++hop) {
                route_support[hop] = cur;
                route_owner[hop] = hostlist_owner(
                    cur, W, Kwin, reverse, ngpu, tables);
                if (route_owner[hop] < 0 || route_owner[hop] >= ngpu)
                    fail("fused route cycle-owner host owner");
                cur = hostlist_next(
                    cur, blocked, W, q, Kwin, S, reverse);
            }
            if (cur != seed.support)
                fail("fused route cycle-owner host cycle closure");

            bool all_local = true;
            for (int hop = 1; hop < cycle_len; ++hop)
                all_local = all_local && route_owner[hop] == route_owner[0];

            const std::uint32_t packed =
                (seed.support & PERSISTENT_SUPPORT_MASK) |
                (blocked ? PERSISTENT_BLOCKED_BIT : 0u);
            if (all_local) {
                out.local[static_cast<std::size_t>(route_owner[0])]
                    .push_back(packed);
                continue;
            }

            int boundary = -1;
            for (int hop = 0; hop < cycle_len; ++hop) {
                const int prev = (hop + cycle_len - 1) % cycle_len;
                if (route_owner[hop] != route_owner[prev]) {
                    boundary = hop;
                    break;
                }
            }
            if (boundary < 0)
                fail("fused route cycle-owner missing boundary");

            int segments_per_owner[SCRATCH_FULL_MAX_GPU]{};
            std::uint32_t segment_support
                [SCRATCH_FULL_MAX_GPU][CYCLE_OWNER_MAX_SEGMENTS]{};
            std::uint8_t segment_len
                [SCRATCH_FULL_MAX_GPU][CYCLE_OWNER_MAX_SEGMENTS]{};

            int pos = 0;
            while (pos < cycle_len) {
                const int h = (boundary + pos) % cycle_len;
                const int owner = route_owner[h];
                int len = 1;
                while (pos + len < cycle_len &&
                       route_owner[(boundary + pos + len) % cycle_len] == owner) {
                    ++len;
                }

                const int si = segments_per_owner[owner]++;
                if (si < 0 || si >= CYCLE_OWNER_MAX_SEGMENTS ||
                    len < 1 || len >= cycle_len || len > RP_MAX_W) {
                    fail("fused route cycle-owner segment plan");
                }
                segment_support[owner][si] = route_support[h];
                segment_len[owner][si] = static_cast<std::uint8_t>(len);
                pos += len;
            }
            if (pos != cycle_len)
                fail("fused route cycle-owner segment coverage");

            const int occupied = __builtin_popcount(seed.support);
            const Rank64 pc =
                tables.primitive[static_cast<std::size_t>(occupied)][1];
            if (!pc) fail("fused route cycle-owner primitive count");
            const int batch = hostlist_batch_id(
                seed.support, blocked, W, q, Kwin, batches);
            if (batch < 0 || batch >= batches)
                fail("fused route cycle-owner batch");

            for (int owner = 0; owner < ngpu; ++owner) {
                const int nseg = segments_per_owner[owner];
                if (!nseg) continue;
                if (nseg < 1 || nseg > CYCLE_OWNER_MAX_SEGMENTS)
                    fail("fused route cycle-owner segment multiplicity");

                auto& supports =
                    out.route_support[static_cast<std::size_t>(owner)]
                                     [static_cast<std::size_t>(batch)];
                auto& lengths =
                    out.route_len[static_cast<std::size_t>(owner)]
                                 [static_cast<std::size_t>(batch)];
                const unsigned long long route_offset = supports.size();
                for (int s = 0; s < nseg; ++s) {
                    supports.push_back(segment_support[owner][s]);
                    lengths.push_back(segment_len[owner][s]);
                }
                if (supports.size() != lengths.size())
                    fail("fused route cycle-owner route arena mismatch");

                out.batch[static_cast<std::size_t>(owner)]
                         [static_cast<std::size_t>(batch)]
                    .push_back(FusedHostCycleOwnerEntry{
                        HostCycleOwnerEntry{
                            packed,
                            static_cast<std::uint8_t>(occupied),
                            static_cast<std::uint8_t>(nseg),
                        },
                        static_cast<std::uint8_t>(cycle_len),
                        route_offset,
                    });
                out.words[static_cast<std::size_t>(owner)]
                         [static_cast<std::size_t>(batch)] +=
                    static_cast<unsigned long long>(nseg) * pc;
            }
        }
    }

    return out;
}

void run_fused_route_cycle_owner_pipeline(
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
    std::vector<RoutePipelineCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> state_ptr(
        static_cast<std::size_t>(ngpu), nullptr);
    std::vector<std::vector<unsigned long long>> batch_offset(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches + 1)));
    unsigned long long max_slot_words[ROUTE_PIPELINE_SLOTS]{};
    unsigned long long total_route_segments = 0;

    for (int g = 0; g < ngpu; ++g) {
        auto& offsets = batch_offset[static_cast<std::size_t>(g)];
        std::vector<std::uint32_t> packed;
        std::vector<std::uint16_t> meta;
        std::vector<std::uint32_t> route_support;
        std::vector<std::uint8_t> route_len;
        std::vector<unsigned long long> list_meta(
            static_cast<std::size_t>(batches * class_stride));
        std::vector<unsigned long long> scratch_meta(
            static_cast<std::size_t>(batches * class_stride));
        std::vector<unsigned long long> segment_meta(
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
            offsets[static_cast<std::size_t>(b)] = packed.size();

            std::vector<unsigned long long> class_count(
                static_cast<std::size_t>(class_stride));
            for (const auto& entry : part) {
                const int cls =
                    int(entry.base.occupied) *
                        (CYCLE_OWNER_MAX_SEGMENTS + 1) +
                    int(entry.base.segments);
                ++class_count[static_cast<std::size_t>(cls)];
            }

            unsigned long long list_cursor = 0;
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
                    list_meta[mi] = list_cursor;
                    scratch_meta[mi] = scratch_cursor;
                    segment_meta[mi] = segment_cursor;
                    const auto n = class_count[static_cast<std::size_t>(cls)];
                    list_cursor += n;
                    scratch_cursor +=
                        n * static_cast<unsigned long long>(nseg) * pc;
                    segment_cursor +=
                        n * static_cast<unsigned long long>(nseg);
                }
            }
            if (list_cursor != part.size() ||
                scratch_cursor !=
                    lists.words[static_cast<std::size_t>(g)]
                               [static_cast<std::size_t>(b)]) {
                fail("fused route cycle-owner class plan mismatch");
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
                fail("fused route cycle-owner source route mismatch");

            for (const auto& entry : part) {
                packed.push_back(entry.base.packed);
                meta.push_back(route_entry_meta(entry.base, entry.cycle_len));
                const unsigned long long begin = entry.route_offset;
                const unsigned long long end =
                    begin + static_cast<unsigned long long>(entry.base.segments);
                if (end > source_support.size())
                    fail("fused route cycle-owner route offset");
                for (unsigned long long si = begin; si < end; ++si) {
                    route_support.push_back(
                        source_support[static_cast<std::size_t>(si)]);
                    route_len.push_back(
                        source_len[static_cast<std::size_t>(si)]);
                }
            }
            if (route_support.size() != segment_cursor ||
                route_len.size() != route_support.size()) {
                fail("fused route cycle-owner packed route mismatch");
            }
        }
        offsets[static_cast<std::size_t>(batches)] = packed.size();
        if (meta.size() != packed.size())
            fail("fused route cycle-owner entry metadata mismatch");
        total_route_segments += route_support.size();

        ck(cudaSetDevice(g), "fused route cycle-owner set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];

        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "fused route cycle-owner alloc owner begin");
        ck(cudaMemcpy(
               c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
               cudaMemcpyHostToDevice),
           "fused route cycle-owner copy owner begin");

        ck(cudaMalloc(
               &c.batch_list,
               std::max<std::size_t>(1, packed.size()) * sizeof(std::uint32_t)),
           "fused route cycle-owner alloc batch list");
        if (!packed.empty()) {
            ck(cudaMemcpy(
                   c.batch_list, packed.data(),
                   packed.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice),
               "fused route cycle-owner copy batch list");
        }

        ck(cudaMalloc(
               &c.entry_meta,
               std::max<std::size_t>(1, meta.size()) * sizeof(std::uint16_t)),
           "fused route cycle-owner alloc entry meta");
        if (!meta.empty()) {
            ck(cudaMemcpy(
                   c.entry_meta, meta.data(),
                   meta.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice),
               "fused route cycle-owner copy entry meta");
        }

        ck(cudaMalloc(
               &c.segment_support,
               std::max<std::size_t>(1, route_support.size()) *
                   sizeof(std::uint32_t)),
           "fused route cycle-owner alloc segment support");
        if (!route_support.empty()) {
            ck(cudaMemcpy(
                   c.segment_support, route_support.data(),
                   route_support.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice),
               "fused route cycle-owner copy segment support");
        }

        ck(cudaMalloc(
               &c.segment_len,
               std::max<std::size_t>(1, route_len.size()) * sizeof(std::uint8_t)),
           "fused route cycle-owner alloc segment len");
        if (!route_len.empty()) {
            ck(cudaMemcpy(
                   c.segment_len, route_len.data(),
                   route_len.size() * sizeof(std::uint8_t),
                   cudaMemcpyHostToDevice),
               "fused route cycle-owner copy segment len");
        }

        const auto& local = lists.local[static_cast<std::size_t>(g)];
        ck(cudaMalloc(
               &c.local_list,
               std::max<std::size_t>(1, local.size()) * sizeof(std::uint32_t)),
           "fused route cycle-owner alloc local list");
        if (!local.empty()) {
            ck(cudaMemcpy(
                   c.local_list, local.data(),
                   local.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice),
               "fused route cycle-owner copy local list");
        }

        ck(cudaMalloc(
               &c.class_list_begin,
               list_meta.size() * sizeof(unsigned long long)),
           "fused route cycle-owner alloc list meta");
        ck(cudaMalloc(
               &c.class_scratch_begin,
               scratch_meta.size() * sizeof(unsigned long long)),
           "fused route cycle-owner alloc scratch meta");
        ck(cudaMalloc(
               &c.class_segment_begin,
               segment_meta.size() * sizeof(unsigned long long)),
           "fused route cycle-owner alloc segment meta");
        ck(cudaMemcpy(
               c.class_list_begin, list_meta.data(),
               list_meta.size() * sizeof(unsigned long long),
               cudaMemcpyHostToDevice),
           "fused route cycle-owner copy list meta");
        ck(cudaMemcpy(
               c.class_scratch_begin, scratch_meta.data(),
               scratch_meta.size() * sizeof(unsigned long long),
               cudaMemcpyHostToDevice),
           "fused route cycle-owner copy scratch meta");
        ck(cudaMemcpy(
               c.class_segment_begin, segment_meta.data(),
               segment_meta.size() * sizeof(unsigned long long),
               cudaMemcpyHostToDevice),
           "fused route cycle-owner copy segment meta");

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)),
           "fused route cycle-owner alloc state");
        state_ptr[static_cast<std::size_t>(g)] = c.state;
        ck(cudaMemcpy(
               c.state,
               input.data() + plan.shard_base[static_cast<std::size_t>(g)],
               n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "fused route cycle-owner copy state");

        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "fused route cycle-owner alloc peer table");
        ck(cudaMalloc(&c.error, sizeof(int)),
           "fused route cycle-owner alloc error");
        ck(cudaMemset(c.error, 0, sizeof(int)),
           "fused route cycle-owner clear error");

        for (int slot = 0; slot < ROUTE_PIPELINE_SLOTS; ++slot) {
            max_slot_words[slot] =
                std::max(max_slot_words[slot], slot_words[slot]);
            ck(cudaMalloc(
                   &c.scratch[slot],
                   slot_words[slot] * sizeof(std::uint32_t)),
               "fused route cycle-owner alloc scratch slot");
            ck(cudaMalloc(
                   &c.peer_words[slot], sizeof(unsigned long long)),
               "fused route cycle-owner alloc peer words slot");
        }

        ck(cudaStreamCreateWithFlags(&c.produce, cudaStreamNonBlocking),
           "fused route cycle-owner create produce stream");
        ck(cudaStreamCreateWithFlags(&c.consume, cudaStreamNonBlocking),
           "fused route cycle-owner create consume stream");
        for (int slot = 0; slot < ROUTE_PIPELINE_SLOTS; ++slot) {
            ck(cudaEventCreateWithFlags(
                   &c.ready[slot], cudaEventDisableTiming),
               "fused route cycle-owner create ready event");
            ck(cudaEventCreateWithFlags(
                   &c.consumed[slot], cudaEventDisableTiming),
               "fused route cycle-owner create consumed event");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "fused route cycle-owner peer set device");
        ck(cudaMemcpy(
               ctx[static_cast<std::size_t>(g)].peer_state,
               state_ptr.data(), ngpu * sizeof(std::uint32_t*),
               cudaMemcpyHostToDevice),
           "fused route cycle-owner copy peer table");
    }
    const double setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup0).count();

    for (int g = 0; g < ngpu; ++g) {
        const auto count = lists.local[static_cast<std::size_t>(g)].size();
        if (!count) continue;
        ck(cudaSetDevice(g), "fused route cycle-owner local set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const unsigned blocks = static_cast<unsigned>(
            std::max<std::size_t>(
                1, std::min<std::size_t>(requested_blocks, count)));
        persistent_local_cycle_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            c.state, c.local_list, count,
            W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "fused route cycle-owner local launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "fused route cycle-owner local sync set device");
        ck(cudaDeviceSynchronize(), "fused route cycle-owner local sync");
        int error = 0;
        ck(cudaMemcpy(
               &error, ctx[static_cast<std::size_t>(g)].error,
               sizeof(error), cudaMemcpyDeviceToHost),
           "fused route cycle-owner local copy error");
        if (error) {
            fail("fused route cycle-owner local device error=" +
                 std::to_string(error));
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int batch = 0; batch < batches; ++batch) {
        const int slot = batch & (ROUTE_PIPELINE_SLOTS - 1);

        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "fused route cycle-owner phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            if (batch >= ROUTE_PIPELINE_SLOTS) {
                ck(cudaStreamWaitEvent(c.produce, c.consumed[slot], 0),
                   "fused route cycle-owner wait slot consumed");
            }
            ck(cudaMemsetAsync(
                   c.peer_words[slot], 0,
                   sizeof(unsigned long long), c.produce),
               "fused route cycle-owner zero peer words");

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
                route_cycle_owner_phase_a_kernel<<<
                    blocks, SCRATCH_FULL_THREADS, 0, c.produce>>>(
                    c.state, c.scratch[slot],
                    c.batch_list + offset, c.entry_meta + offset,
                    c.segment_support, c.segment_len, count,
                    c.class_list_begin + batch * class_stride,
                    c.class_scratch_begin + batch * class_stride,
                    c.class_segment_begin + batch * class_stride,
                    W, q, reverse, tile_start, Kwin, S,
                    ngpu, g, c.owner_begin, c.error);
                ck(cudaGetLastError(),
                   "fused route cycle-owner phase A launch");
            }
            ck(cudaEventRecord(c.ready[slot], c.produce),
               "fused route cycle-owner record ready");
        }

        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "fused route cycle-owner phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            for (int src = 0; src < ngpu; ++src) {
                ck(cudaStreamWaitEvent(
                       c.consume,
                       ctx[static_cast<std::size_t>(src)].ready[slot], 0),
                   "fused route cycle-owner wait all phase A");
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
                route_cycle_owner_phase_b_kernel<<<
                    blocks, SCRATCH_FULL_THREADS, 0, c.consume>>>(
                    c.peer_state, c.scratch[slot],
                    c.batch_list + offset, c.entry_meta + offset,
                    c.segment_support, c.segment_len, count,
                    c.class_list_begin + batch * class_stride,
                    c.class_scratch_begin + batch * class_stride,
                    c.class_segment_begin + batch * class_stride,
                    W, q, reverse, tile_start, Kwin, S,
                    ngpu, g, c.owner_begin, c.peer_words[slot], c.error);
                ck(cudaGetLastError(),
                   "fused route cycle-owner phase B launch");
            }
            route_pipeline_audit_kernel<<<1, 1, 0, c.consume>>>(
                c.peer_words[slot], expected_words, c.error);
            ck(cudaGetLastError(),
               "fused route cycle-owner audit launch");
            ck(cudaEventRecord(c.consumed[slot], c.consume),
               "fused route cycle-owner record consumed");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "fused route cycle-owner final sync set device");
        ck(cudaStreamSynchronize(
               ctx[static_cast<std::size_t>(g)].consume),
           "fused route cycle-owner final consume sync");
        ck(cudaStreamSynchronize(
               ctx[static_cast<std::size_t>(g)].produce),
           "fused route cycle-owner final produce sync");
    }
    const double runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> output(input.size());
    unsigned long long cross_entries = 0;
    unsigned long long local_entries = 0;
    unsigned long long total_peer_words = 0;
    unsigned long long list_bytes = 0;
    unsigned long long entry_meta_bytes = 0;

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "fused route cycle-owner gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(
               &error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "fused route cycle-owner copy error");
        if (error) {
            fail("fused route cycle-owner device error=" +
                 std::to_string(error));
        }

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "fused route cycle-owner gather state");

        const auto local_count =
            lists.local[static_cast<std::size_t>(g)].size();
        local_entries += local_count;
        list_bytes +=
            static_cast<unsigned long long>(local_count) * sizeof(std::uint32_t);
        for (int b = 0; b < batches; ++b) {
            const auto nentry =
                lists.batch[static_cast<std::size_t>(g)]
                           [static_cast<std::size_t>(b)].size();
            cross_entries += nentry;
            list_bytes +=
                static_cast<unsigned long long>(nentry) * sizeof(std::uint32_t);
            entry_meta_bytes +=
                static_cast<unsigned long long>(nentry) * sizeof(std::uint16_t);
            total_peer_words +=
                lists.words[static_cast<std::size_t>(g)]
                           [static_cast<std::size_t>(b)];
        }
    }

    if (output != expected)
        fail("fused route cycle-owner pipeline redistribution mismatch");

    const unsigned long long segment_route_bytes =
        total_route_segments *
        (sizeof(std::uint32_t) + sizeof(std::uint8_t));
    const unsigned long long class_meta_bytes =
        static_cast<unsigned long long>(ngpu) * batches * class_stride *
        3ULL * sizeof(unsigned long long);

    std::cout << "gridfp-p2p-cycle-owner-fused-route-pipeline"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " batches=" << batches
              << " states=" << tables.size()
              << " cross_entries=" << cross_entries
              << " local_entries=" << local_entries
              << " total_list_entries=" << (cross_entries + local_entries)
              << " route_segments=" << total_route_segments
              << " persistent_list_KiB=" << double(list_bytes) / 1024.0
              << " entry_meta_KiB=" << double(entry_meta_bytes) / 1024.0
              << " segment_route_KiB=" << double(segment_route_bytes) / 1024.0
              << " class_meta_KiB=" << double(class_meta_bytes) / 1024.0
              << " logical_peer_KiB="
              << double(total_peer_words) * sizeof(std::uint32_t) / 1024.0
              << " scratch_even_KiB="
              << double(max_slot_words[0]) * sizeof(std::uint32_t) / 1024.0
              << " scratch_odd_KiB="
              << double(max_slot_words[1]) * sizeof(std::uint32_t) / 1024.0
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
        ck(cudaSetDevice(g), "fused route cycle-owner free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        for (int slot = 0; slot < ROUTE_PIPELINE_SLOTS; ++slot) {
            cudaEventDestroy(c.consumed[slot]);
            cudaEventDestroy(c.ready[slot]);
        }
        cudaStreamDestroy(c.consume);
        cudaStreamDestroy(c.produce);
        for (int slot = 0; slot < ROUTE_PIPELINE_SLOTS; ++slot) {
            cudaFree(c.peer_words[slot]);
            cudaFree(c.scratch[slot]);
        }
        cudaFree(c.error);
        cudaFree(c.peer_state);
        cudaFree(c.state);
        cudaFree(c.class_segment_begin);
        cudaFree(c.class_scratch_begin);
        cudaFree(c.class_list_begin);
        cudaFree(c.local_list);
        cudaFree(c.segment_len);
        cudaFree(c.segment_support);
        cudaFree(c.entry_meta);
        cudaFree(c.batch_list);
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
       "fused route cycle-owner pipeline device count");
    if (visible < ngpu) return 3;

    run_fused_route_cycle_owner_pipeline(
        W, Kwin, S, false, ngpu, batches, blocks);
    run_fused_route_cycle_owner_pipeline(
        W, Kwin, S, true, ngpu, batches, blocks);
    std::cout << "ALL_OK gridfp_p2p_cycle_owner_fused_route_pipeline=1\n";
    return 0;
}
