#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_host_persistent_descriptorless_microprobe_main_unused
#include "gridfp_reduced_production_p2p_host_persistent_descriptorless_microprobe.cu"
#pragma pop_macro("main")

namespace {

static constexpr int CYCLE_OWNER_MAX_SEGMENTS = 14;

struct HostCycleOwnerEntry {
    std::uint32_t packed = 0;
    std::uint8_t occupied = 0;
    std::uint8_t segments = 0;
};

struct HostCycleOwnerLists {
    std::vector<std::vector<std::vector<HostCycleOwnerEntry>>> batch;
    std::vector<std::vector<std::uint32_t>> local;
    std::vector<std::vector<unsigned long long>> words;
};

HostCycleOwnerLists build_cycle_owner_lists(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    int batches,
    const ProductionFactorTables& tables
) {
    HostCycleOwnerLists out;
    out.batch.assign(
        static_cast<std::size_t>(ngpu),
        std::vector<std::vector<HostCycleOwnerEntry>>(static_cast<std::size_t>(batches)));
    out.local.assign(static_cast<std::size_t>(ngpu), {});
    out.words.assign(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches)));

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
            if (cycle_len < 0) fail("cycle-owner host cycle length");
            if (cycle_len <= 1) continue;

            int route_owner[RP_MAX_W]{};
            std::uint32_t cur = seed.support;
            for (int hop = 0; hop < cycle_len; ++hop) {
                route_owner[hop] = hostlist_owner(
                    cur, W, Kwin, reverse, ngpu, tables);
                cur = hostlist_next(cur, blocked, W, q, Kwin, S, reverse);
            }
            if (cur != seed.support) fail("cycle-owner host cycle closure");

            bool all_local = true;
            for (int hop = 1; hop < cycle_len; ++hop)
                all_local = all_local && route_owner[hop] == route_owner[0];

            const std::uint32_t packed =
                (seed.support & PERSISTENT_SUPPORT_MASK) |
                (blocked ? PERSISTENT_BLOCKED_BIT : 0u);
            if (all_local) {
                out.local[static_cast<std::size_t>(route_owner[0])].push_back(packed);
                continue;
            }

            int segments_per_owner[SCRATCH_FULL_MAX_GPU]{};
            int hop = 0;
            while (hop < cycle_len) {
                const int owner = route_owner[hop];
                int len = 1;
                while (hop + len < cycle_len && route_owner[hop + len] == owner) ++len;
                ++segments_per_owner[owner];
                hop += len;
            }
            if (route_owner[0] == route_owner[cycle_len - 1])
                --segments_per_owner[route_owner[0]];

            const int occupied = __builtin_popcount(seed.support);
            const Rank64 pc =
                tables.primitive[static_cast<std::size_t>(occupied)][1];
            const int batch = hostlist_batch_id(
                seed.support, blocked, W, q, Kwin, batches);
            for (int owner = 0; owner < ngpu; ++owner) {
                const int nseg = segments_per_owner[owner];
                if (!nseg) continue;
                if (nseg < 1 || nseg > CYCLE_OWNER_MAX_SEGMENTS)
                    fail("cycle-owner host segment multiplicity");
                out.batch[static_cast<std::size_t>(owner)][static_cast<std::size_t>(batch)]
                    .push_back(HostCycleOwnerEntry{
                        packed,
                        static_cast<std::uint8_t>(occupied),
                        static_cast<std::uint8_t>(nseg),
                    });
                out.words[static_cast<std::size_t>(owner)][static_cast<std::size_t>(batch)] +=
                    static_cast<unsigned long long>(nseg) * pc;
            }
        }
    }
    return out;
}

__device__ int cycle_owner_segment_count(
    const CycleSegments& cycle,
    int owner
) {
    int count = 0;
    for (int s = 0; s < cycle.segment_count; ++s)
        count += cycle.segment_owner[s] == owner;
    return count;
}

__global__ void cycle_owner_phase_a_kernel(
    std::uint32_t* local_state,
    std::uint32_t* scratch,
    const std::uint32_t* __restrict__ list,
    unsigned long long count,
    const unsigned long long* __restrict__ class_list_begin,
    const unsigned long long* __restrict__ class_scratch_begin,
    int expected_batch,
    int batches,
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
    __shared__ CycleSegments cycle;
    __shared__ unsigned long long scratch_base;
    __shared__ int owner_segments;
    const unsigned long long first = blockIdx.x;
    const unsigned long long stride = gridDim.x;

    for (unsigned long long ix = first; ix < count; ix += stride) {
        if (threadIdx.x == 0) {
            const OwnerSupportSlabDevice slab = persistent_unpack_slab(list[ix]);
            const EqualTileRunSeed run{
                slab.support,
                slab.blocked,
                slab.valid,
            };
            if (!build_cycle_segments_device(
                    run, W, q, reverse, tile_start, Kwin, ngpu,
                    owner_begin, cycle) || cycle.status != 1) {
                set_error(error, 361);
            } else {
                owner_segments = cycle_owner_segment_count(cycle, owner);
                if (owner_segments < 1 ||
                    owner_segments > CYCLE_OWNER_MAX_SEGMENTS ||
                    persistent_batch_id(
                        slab.support, slab.blocked != 0, W, q, Kwin, batches) !=
                        expected_batch) {
                    set_error(error, 362);
                } else {
                    const int occupied = __popc(slab.support);
                    const int cls =
                        occupied * (CYCLE_OWNER_MAX_SEGMENTS + 1) + owner_segments;
                    const unsigned long long class_ix =
                        ix - class_list_begin[cls];
                    scratch_base = class_scratch_begin[cls] +
                        class_ix * static_cast<unsigned long long>(owner_segments) *
                        cycle.primitive_count;
                }
            }
        }
        __syncthreads();
        if (*error) return;

        int local_segment = 0;
        for (int s = 0; s < cycle.segment_count; ++s) {
            if (cycle.segment_owner[s] != owner) continue;
            const int begin = cycle.segment_begin[s];
            const int len = cycle.segment_len[s];
            const int tail = begin + len - 1;
            const unsigned long long slot = scratch_base +
                static_cast<unsigned long long>(local_segment) * cycle.primitive_count;
            const int pc = static_cast<int>(cycle.primitive_count);
            for (int i = threadIdx.x; i < pc; i += blockDim.x) {
                scratch[slot + static_cast<unsigned long long>(i)] =
                    local_state[cycle.local[tail] + static_cast<Rank64>(i)];
                for (int h = tail; h > begin; --h) {
                    local_state[cycle.local[h] + static_cast<Rank64>(i)] =
                        local_state[cycle.local[h - 1] + static_cast<Rank64>(i)];
                }
            }
            ++local_segment;
            __syncthreads();
        }
        if (threadIdx.x == 0 && local_segment != owner_segments)
            set_error(error, 363);
        __syncthreads();
    }
}

__global__ void cycle_owner_phase_b_kernel(
    std::uint32_t* const* __restrict__ peer_state,
    const std::uint32_t* __restrict__ scratch,
    const std::uint32_t* __restrict__ list,
    unsigned long long count,
    const unsigned long long* __restrict__ class_list_begin,
    const unsigned long long* __restrict__ class_scratch_begin,
    int expected_batch,
    int batches,
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
    __shared__ CycleSegments cycle;
    __shared__ unsigned long long scratch_base;
    __shared__ int owner_segments;
    const unsigned long long first = blockIdx.x;
    const unsigned long long stride = gridDim.x;

    for (unsigned long long ix = first; ix < count; ix += stride) {
        if (threadIdx.x == 0) {
            const OwnerSupportSlabDevice slab = persistent_unpack_slab(list[ix]);
            const EqualTileRunSeed run{
                slab.support,
                slab.blocked,
                slab.valid,
            };
            if (!build_cycle_segments_device(
                    run, W, q, reverse, tile_start, Kwin, ngpu,
                    owner_begin, cycle) || cycle.status != 1) {
                set_error(error, 364);
            } else {
                owner_segments = cycle_owner_segment_count(cycle, owner);
                if (owner_segments < 1 ||
                    owner_segments > CYCLE_OWNER_MAX_SEGMENTS ||
                    persistent_batch_id(
                        slab.support, slab.blocked != 0, W, q, Kwin, batches) !=
                        expected_batch) {
                    set_error(error, 365);
                } else {
                    const int occupied = __popc(slab.support);
                    const int cls =
                        occupied * (CYCLE_OWNER_MAX_SEGMENTS + 1) + owner_segments;
                    const unsigned long long class_ix =
                        ix - class_list_begin[cls];
                    scratch_base = class_scratch_begin[cls] +
                        class_ix * static_cast<unsigned long long>(owner_segments) *
                        cycle.primitive_count;
                }
            }
        }
        __syncthreads();
        if (*error) return;

        int local_segment = 0;
        for (int s = 0; s < cycle.segment_count; ++s) {
            if (cycle.segment_owner[s] != owner) continue;
            const int begin = cycle.segment_begin[s];
            const int len = cycle.segment_len[s];
            const int dst_index = (begin + len) % cycle.route_len;
            const int dst_owner = cycle.owner[dst_index];
            if (dst_owner == owner || dst_owner < 0 || dst_owner >= ngpu) {
                if (threadIdx.x == 0) set_error(error, 366);
                __syncthreads();
                return;
            }
            const Rank64 dst_local = cycle.local[dst_index];
            const unsigned long long slot = scratch_base +
                static_cast<unsigned long long>(local_segment) * cycle.primitive_count;
            std::uint32_t* dst = peer_state[dst_owner] + dst_local;
            const int pc = static_cast<int>(cycle.primitive_count);
            for (int i = threadIdx.x; i < pc; i += blockDim.x)
                dst[i] = scratch[slot + static_cast<unsigned long long>(i)];
            ++local_segment;
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            if (local_segment != owner_segments) set_error(error, 367);
            else atomicAdd(
                peer_words,
                static_cast<unsigned long long>(owner_segments) *
                cycle.primitive_count);
        }
        __syncthreads();
    }
}

struct CycleOwnerCtx {
    std::uint32_t* state = nullptr;
    std::uint32_t* scratch = nullptr;
    std::uint32_t* batch_list = nullptr;
    std::uint32_t* local_list = nullptr;
    std::uint32_t** peer_state = nullptr;
    Rank64* owner_begin = nullptr;
    unsigned long long* class_list_begin = nullptr;
    unsigned long long* class_scratch_begin = nullptr;
    unsigned long long* peer_words = nullptr;
    int* error = nullptr;
};

void run_cycle_owner_executor(
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
    HostCycleOwnerLists lists = build_cycle_owner_lists(
        W, Kwin, S, reverse, ngpu, batches, tables);

    const int class_stride =
        (W + 1) * (CYCLE_OWNER_MAX_SEGMENTS + 1);
    std::vector<CycleOwnerCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> state_ptr(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<std::vector<unsigned long long>> batch_offset(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches + 1)));

    for (int g = 0; g < ngpu; ++g) {
        auto& offsets = batch_offset[static_cast<std::size_t>(g)];
        std::vector<std::uint32_t> packed;
        std::vector<unsigned long long> list_meta(
            static_cast<std::size_t>(batches * class_stride));
        std::vector<unsigned long long> scratch_meta(
            static_cast<std::size_t>(batches * class_stride));
        unsigned long long max_words = 1;

        for (int b = 0; b < batches; ++b) {
            auto& part = lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
            std::sort(part.begin(), part.end(), [](const auto& a, const auto& b) {
                if (a.occupied != b.occupied) return a.occupied < b.occupied;
                if (a.segments != b.segments) return a.segments < b.segments;
                return a.packed < b.packed;
            });
            offsets[static_cast<std::size_t>(b)] = packed.size();

            std::vector<unsigned long long> class_count(
                static_cast<std::size_t>(class_stride));
            for (const auto& entry : part) {
                const int cls = int(entry.occupied) *
                    (CYCLE_OWNER_MAX_SEGMENTS + 1) + int(entry.segments);
                ++class_count[static_cast<std::size_t>(cls)];
            }

            unsigned long long list_cursor = 0;
            unsigned long long scratch_cursor = 0;
            for (int occ = 0; occ <= W; ++occ) {
                const Rank64 pc = tables.primitive[static_cast<std::size_t>(occ)][1];
                for (int nseg = 0; nseg <= CYCLE_OWNER_MAX_SEGMENTS; ++nseg) {
                    const int cls = occ * (CYCLE_OWNER_MAX_SEGMENTS + 1) + nseg;
                    const std::size_t mi =
                        static_cast<std::size_t>(b * class_stride + cls);
                    list_meta[mi] = list_cursor;
                    scratch_meta[mi] = scratch_cursor;
                    const auto n = class_count[static_cast<std::size_t>(cls)];
                    list_cursor += n;
                    scratch_cursor += n * static_cast<unsigned long long>(nseg) * pc;
                }
            }
            if (list_cursor != part.size() ||
                scratch_cursor !=
                    lists.words[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)])
                fail("cycle-owner class plan mismatch");
            max_words = std::max(max_words, scratch_cursor);
            for (const auto& entry : part) packed.push_back(entry.packed);
        }
        offsets[static_cast<std::size_t>(batches)] = packed.size();

        ck(cudaSetDevice(g), "cycle-owner set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "cycle-owner alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                      cudaMemcpyHostToDevice), "cycle-owner copy owner begin");
        ck(cudaMalloc(&c.batch_list,
                      std::max<std::size_t>(1, packed.size()) * sizeof(std::uint32_t)),
           "cycle-owner alloc batch list");
        if (!packed.empty())
            ck(cudaMemcpy(c.batch_list, packed.data(), packed.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "cycle-owner copy batch list");
        const auto& local = lists.local[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.local_list,
                      std::max<std::size_t>(1, local.size()) * sizeof(std::uint32_t)),
           "cycle-owner alloc local list");
        if (!local.empty())
            ck(cudaMemcpy(c.local_list, local.data(), local.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "cycle-owner copy local list");

        ck(cudaMalloc(&c.class_list_begin,
                      list_meta.size() * sizeof(unsigned long long)),
           "cycle-owner alloc list meta");
        ck(cudaMalloc(&c.class_scratch_begin,
                      scratch_meta.size() * sizeof(unsigned long long)),
           "cycle-owner alloc scratch meta");
        ck(cudaMemcpy(c.class_list_begin, list_meta.data(),
                      list_meta.size() * sizeof(unsigned long long),
                      cudaMemcpyHostToDevice), "cycle-owner copy list meta");
        ck(cudaMemcpy(c.class_scratch_begin, scratch_meta.data(),
                      scratch_meta.size() * sizeof(unsigned long long),
                      cudaMemcpyHostToDevice), "cycle-owner copy scratch meta");

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)),
           "cycle-owner alloc state");
        state_ptr[static_cast<std::size_t>(g)] = c.state;
        ck(cudaMemcpy(c.state,
                      input.data() + plan.shard_base[static_cast<std::size_t>(g)],
                      n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "cycle-owner copy state");
        ck(cudaMalloc(&c.scratch, max_words * sizeof(std::uint32_t)),
           "cycle-owner alloc scratch");
        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "cycle-owner alloc peer table");
        ck(cudaMalloc(&c.peer_words, sizeof(unsigned long long)),
           "cycle-owner alloc peer words");
        ck(cudaMalloc(&c.error, sizeof(int)), "cycle-owner alloc error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner peer table set device");
        ck(cudaMemcpy(ctx[static_cast<std::size_t>(g)].peer_state,
                      state_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "cycle-owner copy peer table");
    }
    const double setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup0).count();

    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        const auto count = lists.local[static_cast<std::size_t>(g)].size();
        if (!count) continue;
        ck(cudaSetDevice(g), "cycle-owner local set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMemset(c.error, 0, sizeof(int)), "cycle-owner zero local error");
        const unsigned blocks = static_cast<unsigned>(
            std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
        persistent_local_cycle_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            c.state, c.local_list, count,
            W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "cycle-owner local launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner local sync set device");
        ck(cudaDeviceSynchronize(), "cycle-owner local sync");
        int error = 0;
        ck(cudaMemcpy(&error, ctx[static_cast<std::size_t>(g)].error,
                      sizeof(error), cudaMemcpyDeviceToHost),
           "cycle-owner local copy error");
        if (error) fail("cycle-owner local device error=" + std::to_string(error));
    }

    unsigned long long total_peer_words = 0;
    for (int batch = 0; batch < batches; ++batch) {
        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            if (!count) continue;
            ck(cudaSetDevice(g), "cycle-owner phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            ck(cudaMemset(c.peer_words, 0, sizeof(unsigned long long)),
               "cycle-owner zero peer words");
            ck(cudaMemset(c.error, 0, sizeof(int)), "cycle-owner zero error");
            const unsigned blocks = static_cast<unsigned>(
                std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
            const auto offset =
                batch_offset[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            cycle_owner_phase_a_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
                c.state, c.scratch, c.batch_list + offset, count,
                c.class_list_begin + batch * class_stride,
                c.class_scratch_begin + batch * class_stride,
                batch, batches, W, q, reverse, tile_start, Kwin,
                ngpu, g, c.owner_begin, c.error);
            ck(cudaGetLastError(), "cycle-owner phase A launch");
        }
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "cycle-owner phase A sync set device");
            ck(cudaDeviceSynchronize(), "cycle-owner phase A sync");
        }

        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            if (!count) continue;
            ck(cudaSetDevice(g), "cycle-owner phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            const unsigned blocks = static_cast<unsigned>(
                std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
            const auto offset =
                batch_offset[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            cycle_owner_phase_b_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
                c.peer_state, c.scratch, c.batch_list + offset, count,
                c.class_list_begin + batch * class_stride,
                c.class_scratch_begin + batch * class_stride,
                batch, batches, W, q, reverse, tile_start, Kwin,
                ngpu, g, c.owner_begin, c.peer_words, c.error);
            ck(cudaGetLastError(), "cycle-owner phase B launch");
        }
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "cycle-owner phase B sync set device");
            ck(cudaDeviceSynchronize(), "cycle-owner phase B sync");
        }

        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            if (!count) continue;
            ck(cudaSetDevice(g), "cycle-owner audit set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            int error = 0;
            unsigned long long peer_words = 0;
            ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
               "cycle-owner copy error");
            ck(cudaMemcpy(&peer_words, c.peer_words, sizeof(peer_words),
                          cudaMemcpyDeviceToHost), "cycle-owner copy peer words");
            const auto expected_words =
                lists.words[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            if (error) fail("cycle-owner device error=" + std::to_string(error));
            if (peer_words != expected_words)
                fail("cycle-owner peer word mismatch");
            total_peer_words += peer_words;
        }
    }
    const double runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> output(input.size());
    unsigned long long cross_entries = 0;
    unsigned long long local_entries = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "cycle-owner gather state");
        local_entries += lists.local[static_cast<std::size_t>(g)].size();
        for (int b = 0; b < batches; ++b)
            cross_entries +=
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)].size();
    }
    if (output != expected) fail("cycle-owner redistribution mismatch");

    std::cout << "gridfp-p2p-cycle-owner-descriptorless"
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
              << " logical_peer_KiB="
              << double(total_peer_words) * sizeof(std::uint32_t) / 1024.0
              << " max_segments_per_owner_cycle=" << CYCLE_OWNER_MAX_SEGMENTS
              << " descriptor_bytes=0"
              << " scratch_allocator_atomics=0"
              << " startup_gpu_support_scan_passes=0"
              << " runtime_support_scan_passes=0"
              << " runtime_count_passes=0"
              << " setup_ms=" << setup_ms
              << " runtime_ms=" << runtime_ms
              << " remote_state_reads=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        cudaFree(c.error);
        cudaFree(c.peer_words);
        cudaFree(c.peer_state);
        cudaFree(c.scratch);
        cudaFree(c.state);
        cudaFree(c.class_scratch_begin);
        cudaFree(c.class_list_begin);
        cudaFree(c.local_list);
        cudaFree(c.batch_list);
        cudaFree(c.owner_begin);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 4;
    const int S = argc > 3 ? std::atoi(argv[3]) : 4;
    const int batches = argc > 4 ? std::atoi(argv[4]) : 8;
    const unsigned blocks = argc > 5
        ? static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10)) : 256u;
    const int ngpu = argc > 6 ? std::atoi(argv[6]) : 8;
    if (W < 7 || W > 11 || Kwin < 1 || S != Kwin ||
        Kwin + S + 2 != W || batches < 2 || batches > PERSISTENT_BATCH_MAX ||
        (batches & (batches - 1)) != 0 || !blocks ||
        ngpu < 2 || ngpu > SCRATCH_FULL_MAX_GPU) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cycle-owner device count");
    if (visible < ngpu) return 3;

    run_cycle_owner_executor(W, Kwin, S, false, ngpu, batches, blocks);
    run_cycle_owner_executor(W, Kwin, S, true, ngpu, batches, blocks);
    std::cout << "ALL_OK gridfp_p2p_cycle_owner_descriptorless=1\n";
    return 0;
}
