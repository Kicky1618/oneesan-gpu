#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_cycle_owner_descriptorless_microprobe_main_unused
#include "gridfp_reduced_production_p2p_cycle_owner_descriptorless_microprobe.cu"
#pragma pop_macro("main")

namespace {

static constexpr int SELECTIVE_PIPELINE_SLOTS = 2;

struct SelectiveCycleHeader {
    std::uint32_t support = 0;
    std::uint8_t blocked = 0;
    int cycle_len = 0;
    int expected_segments = 0;
    Rank64 primitive_count = 0;
    unsigned long long scratch_base = 0;
};

struct SelectiveOwnerLocalSegment {
    int status = 0;
    int len = 0;
    Rank64 primitive_count = 0;
    Rank64 local[SCRATCH_OWNER_MAX_ROUTE]{};
};

struct SelectiveOwnerDestination {
    int status = 0;
    int dst_owner = -1;
    Rank64 dst_local = 0;
    Rank64 primitive_count = 0;
};

__device__ bool selective_cycle_prepare(
    std::uint32_t packed,
    std::uint8_t expected_segments,
    unsigned long long ix,
    const unsigned long long* __restrict__ class_list_begin,
    const unsigned long long* __restrict__ class_scratch_begin,
    int expected_batch,
    int batches,
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse,
    SelectiveCycleHeader& out
) {
    out = SelectiveCycleHeader{};
    const OwnerSupportSlabDevice slab = persistent_unpack_slab(packed);
    if (!slab.valid ||
        expected_segments < 1 ||
        expected_segments > CYCLE_OWNER_MAX_SEGMENTS ||
        persistent_batch_id(
            slab.support, slab.blocked != 0, W, q, Kwin, batches) != expected_batch) {
        return false;
    }

    const int cycle_len = shift_cycle_leader_length_device(
        slab.support, slab.blocked != 0, W, q, Kwin, S, reverse);
    if (cycle_len <= 1 || cycle_len > SCRATCH_OWNER_MAX_ROUTE) return false;

    const int occupied = __popc(slab.support);
    const Rank64 pc = RP_PRIMITIVE[occupied][1];
    if (!pc) return false;

    const int cls =
        occupied * (CYCLE_OWNER_MAX_SEGMENTS + 1) + int(expected_segments);
    const unsigned long long class_begin = class_list_begin[cls];
    if (ix < class_begin) return false;

    const unsigned long long class_ix = ix - class_begin;
    out.support = slab.support;
    out.blocked = slab.blocked;
    out.cycle_len = cycle_len;
    out.expected_segments = int(expected_segments);
    out.primitive_count = pc;
    out.scratch_base =
        class_scratch_begin[cls] +
        class_ix * static_cast<unsigned long long>(expected_segments) * pc;
    return true;
}

__device__ bool selective_owner_local_segment_device(
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
    SelectiveOwnerLocalSegment& out
) {
    out = SelectiveOwnerLocalSegment{};
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
    if (grouped_support_owner_device(
            prev, W, reverse, tile_start, Kwin, ngpu) == owner) {
        out.status = -1;
        return false;
    }

    out.primitive_count = RP_PRIMITIVE[__popc(slab.support)][1];
    if (!out.primitive_count) {
        out.status = -1;
        return false;
    }

    std::uint32_t cur = slab.support;
    for (int h = 0; h < SCRATCH_OWNER_MAX_ROUTE; ++h) {
        const GroupedDeviceRank gr = grouped_support_slab_rank_device(
            cur, blocked, W, q, reverse, tile_start, Kwin, ngpu, owner_begin);
        if (gr.owner != owner) {
            out.status = -1;
            return false;
        }
        out.local[out.len++] = gr.local;

        const std::uint32_t next = shift_next_support_device(
            cur, blocked, W, q, Kwin, S, reverse);
        const int next_owner = grouped_support_owner_device(
            next, W, reverse, tile_start, Kwin, ngpu);
        if (next_owner != owner) {
            if (next_owner < 0 || next_owner >= ngpu) {
                out.status = -1;
                return false;
            }
            out.status = 2;
            return true;
        }
        cur = next;
    }

    out.status = -1;
    return false;
}

__device__ bool selective_owner_destination_device(
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
    SelectiveOwnerDestination& out
) {
    out = SelectiveOwnerDestination{};
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
    if (grouped_support_owner_device(
            prev, W, reverse, tile_start, Kwin, ngpu) == owner) {
        out.status = -1;
        return false;
    }

    out.primitive_count = RP_PRIMITIVE[__popc(slab.support)][1];
    if (!out.primitive_count) {
        out.status = -1;
        return false;
    }

    std::uint32_t cur = slab.support;
    for (int h = 0; h < SCRATCH_OWNER_MAX_ROUTE; ++h) {
        const std::uint32_t next = shift_next_support_device(
            cur, blocked, W, q, Kwin, S, reverse);
        const int next_owner = grouped_support_owner_device(
            next, W, reverse, tile_start, Kwin, ngpu);
        if (next_owner != owner) {
            if (next_owner < 0 || next_owner >= ngpu) {
                out.status = -1;
                return false;
            }
            const GroupedDeviceRank dst = grouped_support_slab_rank_device(
                next, blocked, W, q, reverse, tile_start, Kwin, ngpu, owner_begin);
            if (dst.owner != next_owner) {
                out.status = -1;
                return false;
            }
            out.dst_owner = next_owner;
            out.dst_local = dst.local;
            out.status = 2;
            return true;
        }
        cur = next;
    }

    out.status = -1;
    return false;
}

__global__ void selective_cycle_owner_phase_a_kernel(
    std::uint32_t* local_state,
    std::uint32_t* scratch,
    const std::uint32_t* __restrict__ list,
    const std::uint8_t* __restrict__ segment_count,
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
    int S,
    int ngpu,
    int owner,
    const Rank64* __restrict__ owner_begin,
    int* error
) {
    __shared__ SelectiveCycleHeader header;
    __shared__ SelectiveOwnerLocalSegment segment;
    __shared__ std::uint32_t cur_support;
    __shared__ int prev_owner;
    __shared__ int found_segments;
    __shared__ int segment_index;

    const unsigned long long first = blockIdx.x;
    const unsigned long long stride = gridDim.x;
    for (unsigned long long ix = first; ix < count; ix += stride) {
        if (threadIdx.x == 0) {
            if (!selective_cycle_prepare(
                    list[ix], segment_count[ix], ix,
                    class_list_begin, class_scratch_begin,
                    expected_batch, batches, W, q, Kwin, S, reverse, header)) {
                set_error(error, 371);
            } else {
                cur_support = header.support;
                const std::uint32_t prev = owner_shift_prev_support_device(
                    header.support, header.blocked != 0,
                    W, q, Kwin, S, reverse);
                prev_owner = grouped_support_owner_device(
                    prev, W, reverse, tile_start, Kwin, ngpu);
                found_segments = 0;
            }
        }
        __syncthreads();
        if (*error) return;

        for (int h = 0; h < header.cycle_len; ++h) {
            if (threadIdx.x == 0) {
                segment = SelectiveOwnerLocalSegment{};
                segment_index = -1;
                const int cur_owner = grouped_support_owner_device(
                    cur_support, W, reverse, tile_start, Kwin, ngpu);
                if (cur_owner < 0 || cur_owner >= ngpu) {
                    set_error(error, 372);
                } else if (cur_owner == owner && prev_owner != owner) {
                    const OwnerSupportSlabDevice start{
                        cur_support, header.blocked, 1};
                    if (!selective_owner_local_segment_device(
                            start, W, q, reverse, tile_start, Kwin, S,
                            owner, ngpu, owner_begin, segment) ||
                        segment.status != 2 ||
                        segment.primitive_count != header.primitive_count) {
                        set_error(error, 373);
                    } else {
                        segment_index = found_segments++;
                    }
                }
                prev_owner = cur_owner;
                cur_support = shift_next_support_device(
                    cur_support, header.blocked != 0,
                    W, q, Kwin, S, reverse);
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

        if (threadIdx.x == 0 &&
            (cur_support != header.support ||
             found_segments != header.expected_segments)) {
            set_error(error, 374);
        }
        __syncthreads();
        if (*error) return;
    }
}

__global__ void selective_cycle_owner_phase_b_kernel(
    std::uint32_t* const* __restrict__ peer_state,
    const std::uint32_t* __restrict__ scratch,
    const std::uint32_t* __restrict__ list,
    const std::uint8_t* __restrict__ segment_count,
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
    int S,
    int ngpu,
    int owner,
    const Rank64* __restrict__ owner_begin,
    unsigned long long* peer_words,
    int* error
) {
    __shared__ SelectiveCycleHeader header;
    __shared__ SelectiveOwnerDestination destination;
    __shared__ std::uint32_t cur_support;
    __shared__ int prev_owner;
    __shared__ int found_segments;
    __shared__ int segment_index;

    const unsigned long long first = blockIdx.x;
    const unsigned long long stride = gridDim.x;
    for (unsigned long long ix = first; ix < count; ix += stride) {
        if (threadIdx.x == 0) {
            if (!selective_cycle_prepare(
                    list[ix], segment_count[ix], ix,
                    class_list_begin, class_scratch_begin,
                    expected_batch, batches, W, q, Kwin, S, reverse, header)) {
                set_error(error, 375);
            } else {
                cur_support = header.support;
                const std::uint32_t prev = owner_shift_prev_support_device(
                    header.support, header.blocked != 0,
                    W, q, Kwin, S, reverse);
                prev_owner = grouped_support_owner_device(
                    prev, W, reverse, tile_start, Kwin, ngpu);
                found_segments = 0;
            }
        }
        __syncthreads();
        if (*error) return;

        for (int h = 0; h < header.cycle_len; ++h) {
            if (threadIdx.x == 0) {
                destination = SelectiveOwnerDestination{};
                segment_index = -1;
                const int cur_owner = grouped_support_owner_device(
                    cur_support, W, reverse, tile_start, Kwin, ngpu);
                if (cur_owner < 0 || cur_owner >= ngpu) {
                    set_error(error, 376);
                } else if (cur_owner == owner && prev_owner != owner) {
                    const OwnerSupportSlabDevice start{
                        cur_support, header.blocked, 1};
                    if (!selective_owner_destination_device(
                            start, W, q, reverse, tile_start, Kwin, S,
                            owner, ngpu, owner_begin, destination) ||
                        destination.status != 2 ||
                        destination.primitive_count != header.primitive_count) {
                        set_error(error, 377);
                    } else {
                        segment_index = found_segments++;
                    }
                }
                prev_owner = cur_owner;
                cur_support = shift_next_support_device(
                    cur_support, header.blocked != 0,
                    W, q, Kwin, S, reverse);
            }
            __syncthreads();
            if (*error) return;

            if (segment_index >= 0) {
                const unsigned long long slot =
                    header.scratch_base +
                    static_cast<unsigned long long>(segment_index) *
                    header.primitive_count;
                std::uint32_t* dst =
                    peer_state[destination.dst_owner] + destination.dst_local;
                const int pc = static_cast<int>(header.primitive_count);
                for (int i = threadIdx.x; i < pc; i += blockDim.x) {
                    dst[i] =
                        scratch[slot + static_cast<unsigned long long>(i)];
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            if (cur_support != header.support ||
                found_segments != header.expected_segments) {
                set_error(error, 378);
            } else {
                atomicAdd(
                    peer_words,
                    static_cast<unsigned long long>(header.expected_segments) *
                    header.primitive_count);
            }
        }
        __syncthreads();
        if (*error) return;
    }
}

__global__ void selective_pipeline_audit_kernel(
    const unsigned long long* peer_words,
    unsigned long long expected_words,
    int* error
) {
    if (blockIdx.x || threadIdx.x) return;
    if (*peer_words != expected_words) set_error(error, 379);
}

struct SelectivePipelineCtx {
    std::uint32_t* state = nullptr;
    std::uint32_t* scratch[SELECTIVE_PIPELINE_SLOTS]{};
    std::uint32_t* batch_list = nullptr;
    std::uint8_t* batch_segments = nullptr;
    std::uint32_t* local_list = nullptr;
    std::uint32_t** peer_state = nullptr;
    Rank64* owner_begin = nullptr;
    unsigned long long* class_list_begin = nullptr;
    unsigned long long* class_scratch_begin = nullptr;
    unsigned long long* peer_words[SELECTIVE_PIPELINE_SLOTS]{};
    int* error = nullptr;
    cudaStream_t produce{};
    cudaStream_t consume{};
    cudaEvent_t ready[SELECTIVE_PIPELINE_SLOTS]{};
    cudaEvent_t consumed[SELECTIVE_PIPELINE_SLOTS]{};
};

void run_selective_cycle_owner_pipeline(
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
    std::vector<SelectivePipelineCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> state_ptr(
        static_cast<std::size_t>(ngpu), nullptr);
    std::vector<std::vector<unsigned long long>> batch_offset(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches + 1)));
    unsigned long long max_slot_words[SELECTIVE_PIPELINE_SLOTS]{};

    for (int g = 0; g < ngpu; ++g) {
        auto& offsets = batch_offset[static_cast<std::size_t>(g)];
        std::vector<std::uint32_t> packed;
        std::vector<std::uint8_t> segment_meta;
        std::vector<unsigned long long> list_meta(
            static_cast<std::size_t>(batches * class_stride));
        std::vector<unsigned long long> scratch_meta(
            static_cast<std::size_t>(batches * class_stride));
        unsigned long long slot_words[SELECTIVE_PIPELINE_SLOTS] = {1, 1};

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
                    const auto n = class_count[static_cast<std::size_t>(cls)];
                    list_cursor += n;
                    scratch_cursor +=
                        n * static_cast<unsigned long long>(nseg) * pc;
                }
            }
            if (list_cursor != part.size() ||
                scratch_cursor !=
                    lists.words[static_cast<std::size_t>(g)][
                        static_cast<std::size_t>(b)]) {
                fail("selective cycle-owner class plan mismatch");
            }

            const int slot = b & (SELECTIVE_PIPELINE_SLOTS - 1);
            slot_words[slot] = std::max(slot_words[slot], scratch_cursor);
            for (const auto& entry : part) {
                packed.push_back(entry.packed);
                segment_meta.push_back(entry.segments);
            }
        }
        offsets[static_cast<std::size_t>(batches)] = packed.size();
        if (segment_meta.size() != packed.size())
            fail("selective cycle-owner segment metadata mismatch");

        ck(cudaSetDevice(g), "selective cycle-owner set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];

        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "selective cycle-owner alloc owner begin");
        ck(cudaMemcpy(
               c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
               cudaMemcpyHostToDevice),
           "selective cycle-owner copy owner begin");

        ck(cudaMalloc(
               &c.batch_list,
               std::max<std::size_t>(1, packed.size()) * sizeof(std::uint32_t)),
           "selective cycle-owner alloc batch list");
        if (!packed.empty()) {
            ck(cudaMemcpy(
                   c.batch_list, packed.data(),
                   packed.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice),
               "selective cycle-owner copy batch list");
        }

        ck(cudaMalloc(
               &c.batch_segments,
               std::max<std::size_t>(1, segment_meta.size()) *
                   sizeof(std::uint8_t)),
           "selective cycle-owner alloc segment metadata");
        if (!segment_meta.empty()) {
            ck(cudaMemcpy(
                   c.batch_segments, segment_meta.data(),
                   segment_meta.size() * sizeof(std::uint8_t),
                   cudaMemcpyHostToDevice),
               "selective cycle-owner copy segment metadata");
        }

        const auto& local = lists.local[static_cast<std::size_t>(g)];
        ck(cudaMalloc(
               &c.local_list,
               std::max<std::size_t>(1, local.size()) * sizeof(std::uint32_t)),
           "selective cycle-owner alloc local list");
        if (!local.empty()) {
            ck(cudaMemcpy(
                   c.local_list, local.data(),
                   local.size() * sizeof(std::uint32_t),
                   cudaMemcpyHostToDevice),
               "selective cycle-owner copy local list");
        }

        ck(cudaMalloc(
               &c.class_list_begin,
               list_meta.size() * sizeof(unsigned long long)),
           "selective cycle-owner alloc list meta");
        ck(cudaMalloc(
               &c.class_scratch_begin,
               scratch_meta.size() * sizeof(unsigned long long)),
           "selective cycle-owner alloc scratch meta");
        ck(cudaMemcpy(
               c.class_list_begin, list_meta.data(),
               list_meta.size() * sizeof(unsigned long long),
               cudaMemcpyHostToDevice),
           "selective cycle-owner copy list meta");
        ck(cudaMemcpy(
               c.class_scratch_begin, scratch_meta.data(),
               scratch_meta.size() * sizeof(unsigned long long),
               cudaMemcpyHostToDevice),
           "selective cycle-owner copy scratch meta");

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)),
           "selective cycle-owner alloc state");
        state_ptr[static_cast<std::size_t>(g)] = c.state;
        ck(cudaMemcpy(
               c.state,
               input.data() + plan.shard_base[static_cast<std::size_t>(g)],
               n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "selective cycle-owner copy state");

        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "selective cycle-owner alloc peer table");
        ck(cudaMalloc(&c.error, sizeof(int)),
           "selective cycle-owner alloc error");
        ck(cudaMemset(c.error, 0, sizeof(int)),
           "selective cycle-owner clear error");

        for (int slot = 0; slot < SELECTIVE_PIPELINE_SLOTS; ++slot) {
            max_slot_words[slot] =
                std::max(max_slot_words[slot], slot_words[slot]);
            ck(cudaMalloc(
                   &c.scratch[slot],
                   slot_words[slot] * sizeof(std::uint32_t)),
               "selective cycle-owner alloc scratch slot");
            ck(cudaMalloc(
                   &c.peer_words[slot], sizeof(unsigned long long)),
               "selective cycle-owner alloc peer words slot");
        }

        ck(cudaStreamCreateWithFlags(&c.produce, cudaStreamNonBlocking),
           "selective cycle-owner create produce stream");
        ck(cudaStreamCreateWithFlags(&c.consume, cudaStreamNonBlocking),
           "selective cycle-owner create consume stream");
        for (int slot = 0; slot < SELECTIVE_PIPELINE_SLOTS; ++slot) {
            ck(cudaEventCreateWithFlags(
                   &c.ready[slot], cudaEventDisableTiming),
               "selective cycle-owner create ready event");
            ck(cudaEventCreateWithFlags(
                   &c.consumed[slot], cudaEventDisableTiming),
               "selective cycle-owner create consumed event");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "selective cycle-owner peer set device");
        ck(cudaMemcpy(
               ctx[static_cast<std::size_t>(g)].peer_state,
               state_ptr.data(), ngpu * sizeof(std::uint32_t*),
               cudaMemcpyHostToDevice),
           "selective cycle-owner copy peer table");
    }
    const double setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup0).count();

    for (int g = 0; g < ngpu; ++g) {
        const auto count = lists.local[static_cast<std::size_t>(g)].size();
        if (!count) continue;
        ck(cudaSetDevice(g), "selective cycle-owner local set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const unsigned blocks = static_cast<unsigned>(
            std::max<std::size_t>(
                1, std::min<std::size_t>(requested_blocks, count)));
        persistent_local_cycle_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            c.state, c.local_list, count,
            W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "selective cycle-owner local launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "selective cycle-owner local sync set device");
        ck(cudaDeviceSynchronize(), "selective cycle-owner local sync");
        int error = 0;
        ck(cudaMemcpy(
               &error, ctx[static_cast<std::size_t>(g)].error,
               sizeof(error), cudaMemcpyDeviceToHost),
           "selective cycle-owner local copy error");
        if (error) {
            fail("selective cycle-owner local device error=" +
                 std::to_string(error));
        }
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int batch = 0; batch < batches; ++batch) {
        const int slot = batch & (SELECTIVE_PIPELINE_SLOTS - 1);

        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "selective cycle-owner phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            if (batch >= SELECTIVE_PIPELINE_SLOTS) {
                ck(cudaStreamWaitEvent(c.produce, c.consumed[slot], 0),
                   "selective cycle-owner wait slot consumed");
            }

            ck(cudaMemsetAsync(
                   c.peer_words[slot], 0,
                   sizeof(unsigned long long), c.produce),
               "selective cycle-owner zero peer words");

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
                selective_cycle_owner_phase_a_kernel<<<
                    blocks, SCRATCH_FULL_THREADS, 0, c.produce>>>(
                    c.state, c.scratch[slot],
                    c.batch_list + offset, c.batch_segments + offset, count,
                    c.class_list_begin + batch * class_stride,
                    c.class_scratch_begin + batch * class_stride,
                    batch, batches, W, q, reverse, tile_start, Kwin, S,
                    ngpu, g, c.owner_begin, c.error);
                ck(cudaGetLastError(),
                   "selective cycle-owner phase A launch");
            }
            ck(cudaEventRecord(c.ready[slot], c.produce),
               "selective cycle-owner record ready");
        }

        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "selective cycle-owner phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            for (int src = 0; src < ngpu; ++src) {
                ck(cudaStreamWaitEvent(
                       c.consume,
                       ctx[static_cast<std::size_t>(src)].ready[slot], 0),
                   "selective cycle-owner wait all phase A");
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
                selective_cycle_owner_phase_b_kernel<<<
                    blocks, SCRATCH_FULL_THREADS, 0, c.consume>>>(
                    c.peer_state, c.scratch[slot],
                    c.batch_list + offset, c.batch_segments + offset, count,
                    c.class_list_begin + batch * class_stride,
                    c.class_scratch_begin + batch * class_stride,
                    batch, batches, W, q, reverse, tile_start, Kwin, S,
                    ngpu, g, c.owner_begin, c.peer_words[slot], c.error);
                ck(cudaGetLastError(),
                   "selective cycle-owner phase B launch");
            }
            selective_pipeline_audit_kernel<<<1, 1, 0, c.consume>>>(
                c.peer_words[slot], expected_words, c.error);
            ck(cudaGetLastError(),
               "selective cycle-owner audit launch");
            ck(cudaEventRecord(c.consumed[slot], c.consume),
               "selective cycle-owner record consumed");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "selective cycle-owner final sync set device");
        ck(cudaStreamSynchronize(
               ctx[static_cast<std::size_t>(g)].consume),
           "selective cycle-owner final consume sync");
        ck(cudaStreamSynchronize(
               ctx[static_cast<std::size_t>(g)].produce),
           "selective cycle-owner final produce sync");
    }
    const double runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> output(input.size());
    unsigned long long cross_entries = 0;
    unsigned long long local_entries = 0;
    unsigned long long total_peer_words = 0;
    unsigned long long list_bytes = 0;
    unsigned long long segment_meta_bytes = 0;

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "selective cycle-owner gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(
               &error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "selective cycle-owner copy error");
        if (error) {
            fail("selective cycle-owner device error=" +
                 std::to_string(error));
        }

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "selective cycle-owner gather state");

        const auto local_count =
            lists.local[static_cast<std::size_t>(g)].size();
        local_entries += local_count;
        list_bytes +=
            static_cast<unsigned long long>(local_count) *
            sizeof(std::uint32_t);

        for (int b = 0; b < batches; ++b) {
            const auto nentry =
                lists.batch[static_cast<std::size_t>(g)][
                    static_cast<std::size_t>(b)].size();
            cross_entries += nentry;
            list_bytes +=
                static_cast<unsigned long long>(nentry) *
                sizeof(std::uint32_t);
            segment_meta_bytes +=
                static_cast<unsigned long long>(nentry) *
                sizeof(std::uint8_t);
            total_peer_words +=
                lists.words[static_cast<std::size_t>(g)][
                    static_cast<std::size_t>(b)];
        }
    }

    if (output != expected) {
        fail("selective cycle-owner pipeline redistribution mismatch");
    }

    const unsigned long long class_meta_bytes =
        static_cast<unsigned long long>(ngpu) * batches * class_stride *
        2ULL * sizeof(unsigned long long);

    std::cout << "gridfp-p2p-cycle-owner-selective-pipeline"
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
              << " persistent_list_KiB="
              << double(list_bytes) / 1024.0
              << " segment_meta_KiB="
              << double(segment_meta_bytes) / 1024.0
              << " class_meta_KiB="
              << double(class_meta_bytes) / 1024.0
              << " logical_peer_KiB="
              << double(total_peer_words) * sizeof(std::uint32_t) / 1024.0
              << " scratch_even_KiB="
              << double(max_slot_words[0]) * sizeof(std::uint32_t) / 1024.0
              << " scratch_odd_KiB="
              << double(max_slot_words[1]) * sizeof(std::uint32_t) / 1024.0
              << " max_segments_per_owner_cycle="
              << CYCLE_OWNER_MAX_SEGMENTS
              << " descriptor_bytes=0"
              << " scratch_allocator_atomics=0"
              << " segment_count_recompute=0"
              << " full_cycle_rank_passes=0"
              << " phase_a_rank_scope=owner_segment"
              << " phase_b_destination_rank_only=1"
              << " startup_gpu_support_scan_passes=0"
              << " runtime_support_scan_passes=0"
              << " runtime_count_passes=0"
              << " setup_ms=" << setup_ms
              << " runtime_ms=" << runtime_ms
              << " pipeline_slots=" << SELECTIVE_PIPELINE_SLOTS
              << " host_batch_barriers=0"
              << " cross_gpu_phase_a_fence=1"
              << " cycle_closed_batches=1"
              << " double_scratch=1"
              << " remote_state_reads=0"
              << " native_peer_atomics_required=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "selective cycle-owner free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        for (int slot = 0; slot < SELECTIVE_PIPELINE_SLOTS; ++slot) {
            cudaEventDestroy(c.consumed[slot]);
            cudaEventDestroy(c.ready[slot]);
        }
        cudaStreamDestroy(c.consume);
        cudaStreamDestroy(c.produce);
        for (int slot = 0; slot < SELECTIVE_PIPELINE_SLOTS; ++slot) {
            cudaFree(c.peer_words[slot]);
            cudaFree(c.scratch[slot]);
        }
        cudaFree(c.error);
        cudaFree(c.peer_state);
        cudaFree(c.state);
        cudaFree(c.class_scratch_begin);
        cudaFree(c.class_list_begin);
        cudaFree(c.local_list);
        cudaFree(c.batch_segments);
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
       "selective cycle-owner pipeline device count");
    if (visible < ngpu) return 3;

    run_selective_cycle_owner_pipeline(
        W, Kwin, S, false, ngpu, batches, blocks);
    run_selective_cycle_owner_pipeline(
        W, Kwin, S, true, ngpu, batches, blocks);
    std::cout << "ALL_OK gridfp_p2p_cycle_owner_selective_pipeline=1\n";
    return 0;
}
