#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_cycle_owner_selective_pipeline_microprobe_main_unused
#include "gridfp_reduced_production_p2p_cycle_owner_selective_pipeline_microprobe.cu"
#pragma pop_macro("main")

namespace {

static constexpr int ROUTE_PIPELINE_SLOTS = 2;

struct HostOwnerSegmentRoutes {
    int cycle_len = 0;
    std::vector<std::uint32_t> support;
    std::vector<std::uint8_t> len;
};

HostOwnerSegmentRoutes build_host_owner_segment_routes(
    const HostCycleOwnerEntry& entry,
    int owner,
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    const ProductionFactorTables& tables
) {
    HostOwnerSegmentRoutes out;
    const std::uint32_t leader = entry.packed & PERSISTENT_SUPPORT_MASK;
    const bool blocked = (entry.packed & PERSISTENT_BLOCKED_BIT) != 0;
    const int cycle_len = hostlist_leader_length(
        leader, blocked, W, q, Kwin, S, reverse);
    if (cycle_len <= 1 || cycle_len > RP_MAX_W)
        fail("route cycle-owner host cycle length");

    std::uint32_t route_support[RP_MAX_W]{};
    int route_owner[RP_MAX_W]{};
    std::uint32_t cur = leader;
    for (int h = 0; h < cycle_len; ++h) {
        route_support[h] = cur;
        route_owner[h] = hostlist_owner(
            cur, W, Kwin, reverse, ngpu, tables);
        if (route_owner[h] < 0 || route_owner[h] >= ngpu)
            fail("route cycle-owner host owner");
        cur = hostlist_next(cur, blocked, W, q, Kwin, S, reverse);
    }
    if (cur != leader) fail("route cycle-owner host closure");

    for (int h = 0; h < cycle_len; ++h) {
        const int prev = (h + cycle_len - 1) % cycle_len;
        if (route_owner[h] != owner || route_owner[prev] == owner) continue;

        int len = 1;
        while (len < cycle_len &&
               route_owner[(h + len) % cycle_len] == owner) {
            ++len;
        }
        if (len < 1 || len >= cycle_len || len > RP_MAX_W)
            fail("route cycle-owner host segment length");
        out.support.push_back(route_support[h]);
        out.len.push_back(static_cast<std::uint8_t>(len));
    }

    if (out.support.size() != entry.segments ||
        out.len.size() != entry.segments) {
        fail("route cycle-owner host segment count");
    }
    out.cycle_len = cycle_len;
    return out;
}

static std::uint16_t route_entry_meta(
    const HostCycleOwnerEntry& entry,
    int cycle_len
) {
    if (entry.occupied > RP_MAX_W ||
        entry.segments < 1 || entry.segments > CYCLE_OWNER_MAX_SEGMENTS ||
        cycle_len <= 1 || cycle_len > RP_MAX_W) {
        fail("route cycle-owner host entry metadata");
    }
    return static_cast<std::uint16_t>(entry.occupied) |
           (static_cast<std::uint16_t>(entry.segments) << 5) |
           (static_cast<std::uint16_t>(cycle_len) << 9);
}

__device__ __forceinline__ int route_meta_occupied(std::uint16_t meta) {
    return int(meta & 0x1fu);
}

__device__ __forceinline__ int route_meta_segments(std::uint16_t meta) {
    return int((meta >> 5) & 0x0fu);
}

__device__ __forceinline__ int route_meta_cycle_len(std::uint16_t meta) {
    return int((meta >> 9) & 0x1fu);
}

struct RouteEntryHeader {
    std::uint8_t blocked = 0;
    int occupied = 0;
    int segments = 0;
    int cycle_len = 0;
    Rank64 primitive_count = 0;
    unsigned long long scratch_base = 0;
    unsigned long long segment_base = 0;
};

__device__ bool route_entry_prepare(
    std::uint32_t packed,
    std::uint16_t meta,
    unsigned long long ix,
    const unsigned long long* __restrict__ class_list_begin,
    const unsigned long long* __restrict__ class_scratch_begin,
    const unsigned long long* __restrict__ class_segment_begin,
    int W,
    RouteEntryHeader& out
) {
    out = RouteEntryHeader{};
    const OwnerSupportSlabDevice slab = persistent_unpack_slab(packed);
    const int occupied = route_meta_occupied(meta);
    const int segments = route_meta_segments(meta);
    const int cycle_len = route_meta_cycle_len(meta);
    if (!slab.valid ||
        occupied < 0 || occupied > W ||
        segments < 1 || segments > CYCLE_OWNER_MAX_SEGMENTS ||
        cycle_len <= 1 || cycle_len > RP_MAX_W) {
        return false;
    }

    const Rank64 pc = RP_PRIMITIVE[occupied][1];
    if (!pc) return false;
    const int cls = occupied * (CYCLE_OWNER_MAX_SEGMENTS + 1) + segments;
    const unsigned long long class_begin = class_list_begin[cls];
    if (ix < class_begin) return false;
    const unsigned long long class_ix = ix - class_begin;

    out.blocked = slab.blocked;
    out.occupied = occupied;
    out.segments = segments;
    out.cycle_len = cycle_len;
    out.primitive_count = pc;
    out.scratch_base =
        class_scratch_begin[cls] +
        class_ix * static_cast<unsigned long long>(segments) * pc;
    out.segment_base =
        class_segment_begin[cls] +
        class_ix * static_cast<unsigned long long>(segments);
    return true;
}

struct RouteLocalSegment {
    int status = 0;
    int len = 0;
    Rank64 local[RP_MAX_W]{};
};

__device__ bool route_rank_local_segment_device(
    std::uint32_t start_support,
    std::uint8_t len8,
    bool blocked,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int S,
    int owner,
    int ngpu,
    const Rank64* owner_begin,
    RouteLocalSegment& out
) {
    out = RouteLocalSegment{};
    const int len = int(len8);
    if (len < 1 || len > RP_MAX_W) {
        out.status = -1;
        return false;
    }

    std::uint32_t cur = start_support;
    for (int h = 0; h < len; ++h) {
        const GroupedDeviceRank gr = grouped_support_slab_rank_device(
            cur, blocked, W, q, reverse, tile_start,
            Kwin, ngpu, owner_begin);
        if (gr.owner != owner) {
            out.status = -1;
            return false;
        }
        out.local[h] = gr.local;
        cur = shift_next_support_device(
            cur, blocked, W, q, Kwin, S, reverse);
    }
    out.len = len;
    out.status = 1;
    return true;
}

struct RouteDestination {
    int status = 0;
    int owner = -1;
    Rank64 local = 0;
};

__device__ bool route_destination_device(
    std::uint32_t start_support,
    std::uint8_t len8,
    bool blocked,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int S,
    int src_owner,
    int ngpu,
    const Rank64* owner_begin,
    RouteDestination& out
) {
    out = RouteDestination{};
    const int len = int(len8);
    if (len < 1 || len > RP_MAX_W) {
        out.status = -1;
        return false;
    }

    std::uint32_t dst_support = start_support;
    for (int h = 0; h < len; ++h) {
        dst_support = shift_next_support_device(
            dst_support, blocked, W, q, Kwin, S, reverse);
    }
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

__global__ void route_cycle_owner_phase_a_kernel(
    std::uint32_t* local_state,
    std::uint32_t* scratch,
    const std::uint32_t* __restrict__ list,
    const std::uint16_t* __restrict__ entry_meta,
    const std::uint32_t* __restrict__ segment_support,
    const std::uint8_t* __restrict__ segment_len,
    unsigned long long count,
    const unsigned long long* __restrict__ class_list_begin,
    const unsigned long long* __restrict__ class_scratch_begin,
    const unsigned long long* __restrict__ class_segment_begin,
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
    __shared__ RouteEntryHeader header;
    __shared__ RouteLocalSegment segment;
    __shared__ int segment_index;

    const unsigned long long first = blockIdx.x;
    const unsigned long long stride = gridDim.x;
    for (unsigned long long ix = first; ix < count; ix += stride) {
        if (threadIdx.x == 0) {
            if (!route_entry_prepare(
                    list[ix], entry_meta[ix], ix,
                    class_list_begin, class_scratch_begin, class_segment_begin,
                    W, header)) {
                set_error(error, 391);
            }
        }
        __syncthreads();
        if (*error) return;

        for (int s = 0; s < header.segments; ++s) {
            if (threadIdx.x == 0) {
                segment_index = s;
                const unsigned long long si =
                    header.segment_base + static_cast<unsigned long long>(s);
                if (!route_rank_local_segment_device(
                        segment_support[si], segment_len[si],
                        header.blocked != 0,
                        W, q, reverse, tile_start, Kwin, S,
                        owner, ngpu, owner_begin, segment) ||
                    segment.status != 1) {
                    set_error(error, 392);
                    segment_index = -1;
                }
            }
            __syncthreads();
            if (*error) return;

            if (segment_index >= 0) {
                const int tail = segment.len - 1;
                const unsigned long long slot =
                    header.scratch_base +
                    static_cast<unsigned long long>(segment_index) *
                    header.primitive_count;
                const int pc = static_cast<int>(header.primitive_count);
                for (int i = threadIdx.x; i < pc; i += blockDim.x) {
                    scratch[slot + static_cast<unsigned long long>(i)] =
                        local_state[
                            segment.local[tail] + static_cast<Rank64>(i)];
                    for (int p = tail; p > 0; --p) {
                        local_state[
                            segment.local[p] + static_cast<Rank64>(i)] =
                            local_state[
                                segment.local[p - 1] + static_cast<Rank64>(i)];
                    }
                }
            }
            __syncthreads();
        }
    }
}

__global__ void route_cycle_owner_phase_b_kernel(
    std::uint32_t* const* __restrict__ peer_state,
    const std::uint32_t* __restrict__ scratch,
    const std::uint32_t* __restrict__ list,
    const std::uint16_t* __restrict__ entry_meta,
    const std::uint32_t* __restrict__ segment_support,
    const std::uint8_t* __restrict__ segment_len,
    unsigned long long count,
    const unsigned long long* __restrict__ class_list_begin,
    const unsigned long long* __restrict__ class_scratch_begin,
    const unsigned long long* __restrict__ class_segment_begin,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int S,
    int ngpu,
    int owner,
    const Rank64* __restrict__ owner_begin,
    unsigned long long* peer_words,
    int* error
) {
    __shared__ RouteEntryHeader header;
    __shared__ RouteDestination destination;
    __shared__ int segment_index;

    const unsigned long long first = blockIdx.x;
    const unsigned long long stride = gridDim.x;
    for (unsigned long long ix = first; ix < count; ix += stride) {
        if (threadIdx.x == 0) {
            if (!route_entry_prepare(
                    list[ix], entry_meta[ix], ix,
                    class_list_begin, class_scratch_begin, class_segment_begin,
                    W, header)) {
                set_error(error, 393);
            }
        }
        __syncthreads();
        if (*error) return;

        for (int s = 0; s < header.segments; ++s) {
            if (threadIdx.x == 0) {
                segment_index = s;
                const unsigned long long si =
                    header.segment_base + static_cast<unsigned long long>(s);
                if (!route_destination_device(
                        segment_support[si], segment_len[si],
                        header.blocked != 0,
                        W, q, reverse, tile_start, Kwin, S,
                        owner, ngpu, owner_begin, destination) ||
                    destination.status != 1) {
                    set_error(error, 394);
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
                    dst[i] = scratch[slot + static_cast<unsigned long long>(i)];
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

__global__ void route_pipeline_audit_kernel(
    const unsigned long long* peer_words,
    unsigned long long expected_words,
    int* error
) {
    if (blockIdx.x || threadIdx.x) return;
    if (*peer_words != expected_words) set_error(error, 395);
}

struct RoutePipelineCtx {
    std::uint32_t* state = nullptr;
    std::uint32_t* scratch[ROUTE_PIPELINE_SLOTS]{};
    std::uint32_t* batch_list = nullptr;
    std::uint16_t* entry_meta = nullptr;
    std::uint32_t* segment_support = nullptr;
    std::uint8_t* segment_len = nullptr;
    std::uint32_t* local_list = nullptr;
    std::uint32_t** peer_state = nullptr;
    Rank64* owner_begin = nullptr;
    unsigned long long* class_list_begin = nullptr;
    unsigned long long* class_scratch_begin = nullptr;
    unsigned long long* class_segment_begin = nullptr;
    unsigned long long* peer_words[ROUTE_PIPELINE_SLOTS]{};
    int* error = nullptr;
    cudaStream_t produce{};
    cudaStream_t consume{};
    cudaEvent_t ready[ROUTE_PIPELINE_SLOTS]{};
    cudaEvent_t consumed[ROUTE_PIPELINE_SLOTS]{};
};

void run_route_cycle_owner_pipeline(
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
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
            std::sort(part.begin(), part.end(), [](const auto& a, const auto& b) {
                if (a.occupied != b.occupied) return a.occupied < b.occupied;
                if (a.segments != b.segments) return a.segments < b.segments;
                return a.packed < b.packed;
            });
            offsets[static_cast<std::size_t>(b)] = packed.size();

            std::vector<unsigned long long> class_count(
                static_cast<std::size_t>(class_stride));
            for (const auto& entry : part) {
                const int cls =
                    int(entry.occupied) * (CYCLE_OWNER_MAX_SEGMENTS + 1) +
                    int(entry.segments);
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
                    lists.words[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)]) {
                fail("route cycle-owner class plan mismatch");
            }

            const int slot = b & (ROUTE_PIPELINE_SLOTS - 1);
            slot_words[slot] = std::max(slot_words[slot], scratch_cursor);

            for (const auto& entry : part) {
                const HostOwnerSegmentRoutes routes =
                    build_host_owner_segment_routes(
                        entry, g, W, q, Kwin, S, reverse, ngpu, tables);
                packed.push_back(entry.packed);
                meta.push_back(route_entry_meta(entry, routes.cycle_len));
                route_support.insert(
                    route_support.end(), routes.support.begin(), routes.support.end());
                route_len.insert(
                    route_len.end(), routes.len.begin(), routes.len.end());
            }
            if (route_support.size() != segment_cursor ||
                route_len.size() != route_support.size()) {
                fail("route cycle-owner segment route plan mismatch");
            }
        }
        offsets[static_cast<std::size_t>(batches)] = packed.size();
        if (meta.size() != packed.size())
            fail("route cycle-owner entry metadata mismatch");
        total_route_segments += route_support.size();

        ck(cudaSetDevice(g), "route cycle-owner set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];

        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "route cycle-owner alloc owner begin");
        ck(cudaMemcpy(
               c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
               cudaMemcpyHostToDevice),
           "route cycle-owner copy owner begin");

        ck(cudaMalloc(
               &c.batch_list,
               std::max<std::size_t>(1, packed.size()) * sizeof(std::uint32_t)),
           "route cycle-owner alloc batch list");
        if (!packed.empty()) {
            ck(cudaMemcpy(
                   c.batch_list, packed.data(),
                   packed.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice),
               "route cycle-owner copy batch list");
        }

        ck(cudaMalloc(
               &c.entry_meta,
               std::max<std::size_t>(1, meta.size()) * sizeof(std::uint16_t)),
           "route cycle-owner alloc entry meta");
        if (!meta.empty()) {
            ck(cudaMemcpy(
                   c.entry_meta, meta.data(),
                   meta.size() * sizeof(std::uint16_t),
                   cudaMemcpyHostToDevice),
               "route cycle-owner copy entry meta");
        }

        ck(cudaMalloc(
               &c.segment_support,
               std::max<std::size_t>(1, route_support.size()) * sizeof(std::uint32_t)),
           "route cycle-owner alloc segment support");
        if (!route_support.empty()) {
            ck(cudaMemcpy(
                   c.segment_support, route_support.data(),
                   route_support.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice),
               "route cycle-owner copy segment support");
        }

        ck(cudaMalloc(
               &c.segment_len,
               std::max<std::size_t>(1, route_len.size()) * sizeof(std::uint8_t)),
           "route cycle-owner alloc segment len");
        if (!route_len.empty()) {
            ck(cudaMemcpy(
                   c.segment_len, route_len.data(),
                   route_len.size() * sizeof(std::uint8_t),
                   cudaMemcpyHostToDevice),
               "route cycle-owner copy segment len");
        }

        const auto& local = lists.local[static_cast<std::size_t>(g)];
        ck(cudaMalloc(
               &c.local_list,
               std::max<std::size_t>(1, local.size()) * sizeof(std::uint32_t)),
           "route cycle-owner alloc local list");
        if (!local.empty()) {
            ck(cudaMemcpy(
                   c.local_list, local.data(),
                   local.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice),
               "route cycle-owner copy local list");
        }

        ck(cudaMalloc(
               &c.class_list_begin,
               list_meta.size() * sizeof(unsigned long long)),
           "route cycle-owner alloc list meta");
        ck(cudaMalloc(
               &c.class_scratch_begin,
               scratch_meta.size() * sizeof(unsigned long long)),
           "route cycle-owner alloc scratch meta");
        ck(cudaMalloc(
               &c.class_segment_begin,
               segment_meta.size() * sizeof(unsigned long long)),
           "route cycle-owner alloc segment meta");
        ck(cudaMemcpy(
               c.class_list_begin, list_meta.data(),
               list_meta.size() * sizeof(unsigned long long),
               cudaMemcpyHostToDevice),
           "route cycle-owner copy list meta");
        ck(cudaMemcpy(
               c.class_scratch_begin, scratch_meta.data(),
               scratch_meta.size() * sizeof(unsigned long long),
               cudaMemcpyHostToDevice),
           "route cycle-owner copy scratch meta");
        ck(cudaMemcpy(
               c.class_segment_begin, segment_meta.data(),
               segment_meta.size() * sizeof(unsigned long long),
               cudaMemcpyHostToDevice),
           "route cycle-owner copy segment meta");

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)),
           "route cycle-owner alloc state");
        state_ptr[static_cast<std::size_t>(g)] = c.state;
        ck(cudaMemcpy(
               c.state,
               input.data() + plan.shard_base[static_cast<std::size_t>(g)],
               n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "route cycle-owner copy state");

        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "route cycle-owner alloc peer table");
        ck(cudaMalloc(&c.error, sizeof(int)),
           "route cycle-owner alloc error");
        ck(cudaMemset(c.error, 0, sizeof(int)),
           "route cycle-owner clear error");

        for (int slot = 0; slot < ROUTE_PIPELINE_SLOTS; ++slot) {
            max_slot_words[slot] =
                std::max(max_slot_words[slot], slot_words[slot]);
            ck(cudaMalloc(
                   &c.scratch[slot],
                   slot_words[slot] * sizeof(std::uint32_t)),
               "route cycle-owner alloc scratch slot");
            ck(cudaMalloc(
                   &c.peer_words[slot], sizeof(unsigned long long)),
               "route cycle-owner alloc peer words slot");
        }

        ck(cudaStreamCreateWithFlags(&c.produce, cudaStreamNonBlocking),
           "route cycle-owner create produce stream");
        ck(cudaStreamCreateWithFlags(&c.consume, cudaStreamNonBlocking),
           "route cycle-owner create consume stream");
        for (int slot = 0; slot < ROUTE_PIPELINE_SLOTS; ++slot) {
            ck(cudaEventCreateWithFlags(
                   &c.ready[slot], cudaEventDisableTiming),
               "route cycle-owner create ready event");
            ck(cudaEventCreateWithFlags(
                   &c.consumed[slot], cudaEventDisableTiming),
               "route cycle-owner create consumed event");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "route cycle-owner peer set device");
        ck(cudaMemcpy(
               ctx[static_cast<std::size_t>(g)].peer_state,
               state_ptr.data(), ngpu * sizeof(std::uint32_t*),
               cudaMemcpyHostToDevice),
           "route cycle-owner copy peer table");
    }
    const double setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup0).count();

    for (int g = 0; g < ngpu; ++g) {
        const auto count = lists.local[static_cast<std::size_t>(g)].size();
        if (!count) continue;
        ck(cudaSetDevice(g), "route cycle-owner local set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const unsigned blocks = static_cast<unsigned>(
            std::max<std::size_t>(
                1, std::min<std::size_t>(requested_blocks, count)));
        persistent_local_cycle_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            c.state, c.local_list, count,
            W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "route cycle-owner local launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "route cycle-owner local sync set device");
        ck(cudaDeviceSynchronize(), "route cycle-owner local sync");
        int error = 0;
        ck(cudaMemcpy(
               &error, ctx[static_cast<std::size_t>(g)].error,
               sizeof(error), cudaMemcpyDeviceToHost),
           "route cycle-owner local copy error");
        if (error) {
            fail("route cycle-owner local device error=" +
                 std::to_string(error));
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int batch = 0; batch < batches; ++batch) {
        const int slot = batch & (ROUTE_PIPELINE_SLOTS - 1);

        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "route cycle-owner phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            if (batch >= ROUTE_PIPELINE_SLOTS) {
                ck(cudaStreamWaitEvent(c.produce, c.consumed[slot], 0),
                   "route cycle-owner wait slot consumed");
            }
            ck(cudaMemsetAsync(
                   c.peer_words[slot], 0,
                   sizeof(unsigned long long), c.produce),
               "route cycle-owner zero peer words");

            const auto count =
                lists.batch[static_cast<std::size_t>(g)][
                    static_cast<std::size_t>(batch)].size();
            if (count) {
                const unsigned blocks = static_cast<unsigned>(
                    std::max<std::size_t>(
                        1, std::min<std::size_t>(requested_blocks, count)));
                const auto offset =
                    batch_offset[static_cast<std::size_t>(g)][
                        static_cast<std::size_t>(batch)];
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
                ck(cudaGetLastError(), "route cycle-owner phase A launch");
            }
            ck(cudaEventRecord(c.ready[slot], c.produce),
               "route cycle-owner record ready");
        }

        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "route cycle-owner phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            for (int src = 0; src < ngpu; ++src) {
                ck(cudaStreamWaitEvent(
                       c.consume,
                       ctx[static_cast<std::size_t>(src)].ready[slot], 0),
                   "route cycle-owner wait all phase A");
            }

            const auto count =
                lists.batch[static_cast<std::size_t>(g)][
                    static_cast<std::size_t>(batch)].size();
            const auto expected_words =
                lists.words[static_cast<std::size_t>(g)][
                    static_cast<std::size_t>(batch)];
            if (count) {
                const unsigned blocks = static_cast<unsigned>(
                    std::max<std::size_t>(
                        1, std::min<std::size_t>(requested_blocks, count)));
                const auto offset =
                    batch_offset[static_cast<std::size_t>(g)][
                        static_cast<std::size_t>(batch)];
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
                ck(cudaGetLastError(), "route cycle-owner phase B launch");
            }
            route_pipeline_audit_kernel<<<1, 1, 0, c.consume>>>(
                c.peer_words[slot], expected_words, c.error);
            ck(cudaGetLastError(), "route cycle-owner audit launch");
            ck(cudaEventRecord(c.consumed[slot], c.consume),
               "route cycle-owner record consumed");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "route cycle-owner final sync set device");
        ck(cudaStreamSynchronize(
               ctx[static_cast<std::size_t>(g)].consume),
           "route cycle-owner final consume sync");
        ck(cudaStreamSynchronize(
               ctx[static_cast<std::size_t>(g)].produce),
           "route cycle-owner final produce sync");
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
        ck(cudaSetDevice(g), "route cycle-owner gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(
               &error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "route cycle-owner copy error");
        if (error) {
            fail("route cycle-owner device error=" + std::to_string(error));
        }

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "route cycle-owner gather state");

        const auto local_count =
            lists.local[static_cast<std::size_t>(g)].size();
        local_entries += local_count;
        list_bytes +=
            static_cast<unsigned long long>(local_count) * sizeof(std::uint32_t);
        for (int b = 0; b < batches; ++b) {
            const auto nentry =
                lists.batch[static_cast<std::size_t>(g)][
                    static_cast<std::size_t>(b)].size();
            cross_entries += nentry;
            list_bytes +=
                static_cast<unsigned long long>(nentry) * sizeof(std::uint32_t);
            entry_meta_bytes +=
                static_cast<unsigned long long>(nentry) * sizeof(std::uint16_t);
            total_peer_words +=
                lists.words[static_cast<std::size_t>(g)][
                    static_cast<std::size_t>(b)];
        }
    }

    if (output != expected)
        fail("route cycle-owner pipeline redistribution mismatch");

    const unsigned long long segment_route_bytes =
        total_route_segments *
        (sizeof(std::uint32_t) + sizeof(std::uint8_t));
    const unsigned long long class_meta_bytes =
        static_cast<unsigned long long>(ngpu) * batches * class_stride *
        3ULL * sizeof(unsigned long long);

    std::cout << "gridfp-p2p-cycle-owner-route-pipeline"
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
              << " setup_route_recompute=1"
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
        ck(cudaSetDevice(g), "route cycle-owner free set device");
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
    ck(cudaGetDeviceCount(&visible), "route cycle-owner pipeline device count");
    if (visible < ngpu) return 3;

    run_route_cycle_owner_pipeline(
        W, Kwin, S, false, ngpu, batches, blocks);
    run_route_cycle_owner_pipeline(
        W, Kwin, S, true, ngpu, batches, blocks);
    std::cout << "ALL_OK gridfp_p2p_cycle_owner_route_pipeline=1\n";
    return 0;
}
