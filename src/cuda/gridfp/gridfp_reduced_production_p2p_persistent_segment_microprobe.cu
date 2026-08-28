#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_scratch_owner_microprobe_main_unused
#include "gridfp_reduced_production_p2p_scratch_owner_microprobe.cu"
#pragma pop_macro("main")

namespace {

static constexpr int PERSISTENT_BATCH_MAX = 32;
static constexpr std::uint32_t PERSISTENT_BLOCKED_BIT = 0x80000000u;
static constexpr std::uint32_t PERSISTENT_SUPPORT_MASK = 0x0fffffffu;

__device__ __forceinline__ std::uint32_t persistent_mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ std::uint32_t persistent_blocked_compact(
    std::uint32_t support,
    int W,
    int q
) {
    std::uint32_t compact = 0;
    int cp = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        if ((support >> bit) & 1u) compact |= std::uint32_t(1) << cp;
        ++cp;
    }
    return compact;
}

__device__ __forceinline__ int persistent_batch_id(
    std::uint32_t support,
    bool blocked,
    int W,
    int q,
    int Kwin,
    int batches
) {
    std::uint32_t h = 0;
    if (!blocked) {
        h = __popc(support) * 0x9e3779b1u;
        h ^= __popc(support & shift_rotate_bits_device(support, W, 1)) * 0x85ebca6bu;
        h ^= __popc(support & shift_rotate_bits_device(support, W, 3)) * 0xc2b2ae35u;
        h ^= __popc(support & shift_rotate_bits_device(support, W, 5)) * 0x27d4eb2fu;
        h ^= __popc(support & shift_rotate_bits_device(support, W, 7)) * 0x165667b1u;
    } else {
        const std::uint32_t compact = persistent_blocked_compact(support, W, q);
        const std::uint32_t half_mask = (std::uint32_t(1) << Kwin) - 1u;
        const std::uint32_t a = compact & half_mask;
        const std::uint32_t b = (compact >> Kwin) & half_mask;
        const std::uint32_t lo = a < b ? a : b;
        const std::uint32_t hi = a < b ? b : a;
        h = lo * 0x9e3779b1u;
        h ^= hi * 0x85ebca6bu;
        h ^= __popc(support) * 0xc2b2ae35u;
    }
    return int(persistent_mix32(h) & static_cast<std::uint32_t>(batches - 1));
}

__device__ __forceinline__ std::uint32_t persistent_pack_slab(
    OwnerSupportSlabDevice slab
) {
    return (slab.support & PERSISTENT_SUPPORT_MASK) |
           (slab.blocked ? PERSISTENT_BLOCKED_BIT : 0u);
}

__device__ __forceinline__ OwnerSupportSlabDevice persistent_unpack_slab(
    std::uint32_t packed
) {
    return OwnerSupportSlabDevice{
        packed & PERSISTENT_SUPPORT_MASK,
        static_cast<std::uint8_t>((packed & PERSISTENT_BLOCKED_BIT) != 0),
        1};
}

__global__ void persistent_list_count_kernel(
    Rank64 owner_slabs,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int S,
    int ngpu,
    int owner,
    int batches,
    const Rank64* __restrict__ owner_begin,
    unsigned long long* __restrict__ batch_entries,
    unsigned long long* __restrict__ batch_words,
    unsigned long long* __restrict__ local_entries,
    int* error
) {
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    for (Rank64 rank = first; rank < owner_slabs; rank += stride) {
        const OwnerSupportSlabDevice slab = owner_support_slab_unrank_device(
            W, q, reverse, tile_start, Kwin, owner, ngpu, rank);
        OwnerLocalSegment segment;
        if (!owner_local_segment_device(
                slab, W, q, reverse, tile_start, Kwin, S, owner, ngpu,
                owner_begin, segment)) {
            if (segment.status < 0) set_error(error, 331);
            continue;
        }
        if (segment.status == 2) {
            const int batch = persistent_batch_id(
                slab.support, slab.blocked != 0, W, q, Kwin, batches);
            if (batch < 0 || batch >= batches) {
                set_error(error, 332);
                continue;
            }
            atomicAdd(batch_entries + batch, 1ULL);
            atomicAdd(batch_words + batch,
                      static_cast<unsigned long long>(segment.primitive_count));
        } else if (segment.status == 1 && segment.len > 1) {
            atomicAdd(local_entries, 1ULL);
        }
    }
}

__global__ void persistent_list_fill_kernel(
    Rank64 owner_slabs,
    int W,
    int q,
    bool reverse,
    int tile_start,
    int Kwin,
    int S,
    int ngpu,
    int owner,
    int batches,
    const Rank64* __restrict__ owner_begin,
    const unsigned long long* __restrict__ batch_offset,
    unsigned long long* __restrict__ batch_head,
    std::uint32_t* __restrict__ batch_list,
    unsigned long long* __restrict__ local_head,
    std::uint32_t* __restrict__ local_list,
    int* error
) {
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    for (Rank64 rank = first; rank < owner_slabs; rank += stride) {
        const OwnerSupportSlabDevice slab = owner_support_slab_unrank_device(
            W, q, reverse, tile_start, Kwin, owner, ngpu, rank);
        OwnerLocalSegment segment;
        if (!owner_local_segment_device(
                slab, W, q, reverse, tile_start, Kwin, S, owner, ngpu,
                owner_begin, segment)) {
            if (segment.status < 0) set_error(error, 333);
            continue;
        }
        const std::uint32_t packed = persistent_pack_slab(slab);
        if (segment.status == 2) {
            const int batch = persistent_batch_id(
                slab.support, slab.blocked != 0, W, q, Kwin, batches);
            const unsigned long long slot = atomicAdd(batch_head + batch, 1ULL);
            batch_list[batch_offset[batch] + slot] = packed;
        } else if (segment.status == 1 && segment.len > 1) {
            const unsigned long long slot = atomicAdd(local_head, 1ULL);
            local_list[slot] = packed;
        }
    }
}

__global__ void persistent_local_cycle_kernel(
    std::uint32_t* local_state,
    const std::uint32_t* __restrict__ list,
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
    __shared__ OwnerLocalSegment segment;
    const unsigned long long first = blockIdx.x;
    const unsigned long long stride = gridDim.x;
    for (unsigned long long ix = first; ix < count; ix += stride) {
        if (threadIdx.x == 0) {
            const OwnerSupportSlabDevice slab = persistent_unpack_slab(list[ix]);
            owner_local_segment_device(
                slab, W, q, reverse, tile_start, Kwin, S, owner, ngpu,
                owner_begin, segment);
            if (segment.status != 1 || segment.len <= 1) set_error(error, 334);
        }
        __syncthreads();
        if (*error) return;
        const int pc = static_cast<int>(segment.primitive_count);
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
    }
}

__global__ void persistent_phase_a_kernel(
    std::uint32_t* local_state,
    std::uint32_t* scratch,
    ScratchDescriptor* descriptor,
    unsigned long long scratch_capacity,
    unsigned long long descriptor_capacity,
    unsigned long long* scratch_head,
    unsigned long long* descriptor_head,
    const std::uint32_t* __restrict__ list,
    unsigned long long count,
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
    __shared__ OwnerLocalSegment segment;
    __shared__ unsigned long long alloc_offset;
    __shared__ unsigned long long alloc_desc;
    const unsigned long long first = blockIdx.x;
    const unsigned long long stride = gridDim.x;
    for (unsigned long long ix = first; ix < count; ix += stride) {
        if (threadIdx.x == 0) {
            const OwnerSupportSlabDevice slab = persistent_unpack_slab(list[ix]);
            owner_local_segment_device(
                slab, W, q, reverse, tile_start, Kwin, S, owner, ngpu,
                owner_begin, segment);
            if (segment.status != 2 ||
                persistent_batch_id(
                    slab.support, slab.blocked != 0, W, q, Kwin, batches) !=
                    expected_batch) {
                set_error(error, 335);
            }
            if (segment.status == 2) {
                alloc_offset = atomicAdd(
                    scratch_head,
                    static_cast<unsigned long long>(segment.primitive_count));
                alloc_desc = atomicAdd(descriptor_head, 1ULL);
                if (alloc_offset + segment.primitive_count > scratch_capacity ||
                    alloc_desc >= descriptor_capacity) set_error(error, 336);
            }
        }
        __syncthreads();
        if (*error) return;

        const int pc = static_cast<int>(segment.primitive_count);
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

struct PersistentCtx : ScratchFullCtx {
    std::uint32_t* batch_list = nullptr;
    std::uint32_t* local_list = nullptr;
    unsigned long long* batch_offset = nullptr;
    unsigned long long* batch_head = nullptr;
    unsigned long long* local_head = nullptr;
};

void run_persistent_segments(
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

    ck(cudaSetDevice(0), "persistent slab counts set device");
    install_tables(tables);
    unsigned long long* d_owner_slabs = nullptr;
    ck(cudaMalloc(&d_owner_slabs, ngpu * sizeof(unsigned long long)),
       "persistent alloc owner slabs");
    scratch_owner_slab_counts_kernel<<<1, ngpu>>>(
        W, Kwin, ngpu, d_owner_slabs);
    ck(cudaGetLastError(), "persistent owner slab launch");
    ck(cudaDeviceSynchronize(), "persistent owner slab sync");
    std::vector<unsigned long long> owner_slabs(static_cast<std::size_t>(ngpu));
    ck(cudaMemcpy(owner_slabs.data(), d_owner_slabs,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "persistent copy owner slabs");
    cudaFree(d_owner_slabs);

    std::vector<PersistentCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::vector<unsigned long long>> batch_entries(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches)));
    std::vector<std::vector<unsigned long long>> batch_words = batch_entries;
    std::vector<unsigned long long> local_entries(static_cast<std::size_t>(ngpu));

    const auto setup0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "persistent count set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "persistent alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                      cudaMemcpyHostToDevice), "persistent copy owner begin");
        ck(cudaMalloc(&c.count_desc, batches * sizeof(unsigned long long)),
           "persistent alloc batch entries");
        ck(cudaMalloc(&c.count_words, batches * sizeof(unsigned long long)),
           "persistent alloc batch words");
        ck(cudaMalloc(&c.local_head, sizeof(unsigned long long)),
           "persistent alloc local count");
        ck(cudaMalloc(&c.error, sizeof(int)), "persistent alloc error");
        ck(cudaMemset(c.count_desc, 0, batches * sizeof(unsigned long long)),
           "persistent zero batch entries");
        ck(cudaMemset(c.count_words, 0, batches * sizeof(unsigned long long)),
           "persistent zero batch words");
        ck(cudaMemset(c.local_head, 0, sizeof(unsigned long long)),
           "persistent zero local count");
        ck(cudaMemset(c.error, 0, sizeof(int)), "persistent zero error");

        const Rank64 count = owner_slabs[static_cast<std::size_t>(g)];
        const Rank64 needed =
            (count + Rank64(SCRATCH_FULL_THREADS) - 1) /
            Rank64(SCRATCH_FULL_THREADS);
        const unsigned blocks = static_cast<unsigned>(
            std::max<Rank64>(1, std::min<Rank64>(requested_blocks, needed)));
        persistent_list_count_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            count, W, q, reverse, tile_start, Kwin, S, ngpu, g, batches,
            c.owner_begin, c.count_desc, c.count_words, c.local_head, c.error);
        ck(cudaGetLastError(), "persistent list count launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "persistent count sync set device");
        ck(cudaDeviceSynchronize(), "persistent count sync");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "persistent copy count error");
        if (error) fail("persistent count device error=" + std::to_string(error));
        ck(cudaMemcpy(batch_entries[static_cast<std::size_t>(g)].data(),
                      c.count_desc, batches * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost), "persistent copy batch entries");
        ck(cudaMemcpy(batch_words[static_cast<std::size_t>(g)].data(),
                      c.count_words, batches * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost), "persistent copy batch words");
        ck(cudaMemcpy(&local_entries[static_cast<std::size_t>(g)], c.local_head,
                      sizeof(unsigned long long), cudaMemcpyDeviceToHost),
           "persistent copy local count");
    }

    std::vector<std::uint32_t*> state_ptr(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<std::vector<unsigned long long>> batch_offset(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches + 1)));
    for (int g = 0; g < ngpu; ++g) {
        auto& offsets = batch_offset[static_cast<std::size_t>(g)];
        for (int b = 0; b < batches; ++b)
            offsets[static_cast<std::size_t>(b + 1)] =
                offsets[static_cast<std::size_t>(b)] +
                batch_entries[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];

        ck(cudaSetDevice(g), "persistent list alloc set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const unsigned long long total_cross = offsets.back();
        ck(cudaMalloc(&c.batch_list,
                      std::max<unsigned long long>(1, total_cross) * sizeof(std::uint32_t)),
           "persistent alloc batch list");
        ck(cudaMalloc(&c.local_list,
                      std::max<unsigned long long>(1, local_entries[static_cast<std::size_t>(g)]) *
                      sizeof(std::uint32_t)), "persistent alloc local list");
        ck(cudaMalloc(&c.batch_offset,
                      (batches + 1) * sizeof(unsigned long long)),
           "persistent alloc batch offsets");
        ck(cudaMalloc(&c.batch_head, batches * sizeof(unsigned long long)),
           "persistent alloc batch heads");
        ck(cudaMemcpy(c.batch_offset, offsets.data(),
                      (batches + 1) * sizeof(unsigned long long),
                      cudaMemcpyHostToDevice), "persistent copy batch offsets");
        ck(cudaMemset(c.batch_head, 0, batches * sizeof(unsigned long long)),
           "persistent zero batch heads");
        ck(cudaMemset(c.local_head, 0, sizeof(unsigned long long)),
           "persistent reset local head");
        ck(cudaMemset(c.error, 0, sizeof(int)), "persistent reset fill error");

        const Rank64 count = owner_slabs[static_cast<std::size_t>(g)];
        const Rank64 needed =
            (count + Rank64(SCRATCH_FULL_THREADS) - 1) /
            Rank64(SCRATCH_FULL_THREADS);
        const unsigned blocks = static_cast<unsigned>(
            std::max<Rank64>(1, std::min<Rank64>(requested_blocks, needed)));
        persistent_list_fill_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            count, W, q, reverse, tile_start, Kwin, S, ngpu, g, batches,
            c.owner_begin, c.batch_offset, c.batch_head, c.batch_list,
            c.local_head, c.local_list, c.error);
        ck(cudaGetLastError(), "persistent list fill launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "persistent fill sync set device");
        ck(cudaDeviceSynchronize(), "persistent fill sync");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        unsigned long long local_head = 0;
        std::vector<unsigned long long> heads(static_cast<std::size_t>(batches));
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "persistent copy fill error");
        ck(cudaMemcpy(heads.data(), c.batch_head,
                      batches * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
           "persistent copy fill heads");
        ck(cudaMemcpy(&local_head, c.local_head, sizeof(local_head),
                      cudaMemcpyDeviceToHost), "persistent copy local head");
        if (error) fail("persistent fill device error=" + std::to_string(error));
        if (heads != batch_entries[static_cast<std::size_t>(g)] ||
            local_head != local_entries[static_cast<std::size_t>(g)])
            fail("persistent list count/fill mismatch");
    }
    const double setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup0).count();

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "persistent runtime alloc set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)),
           "persistent alloc state");
        state_ptr[static_cast<std::size_t>(g)] = c.state;
        ck(cudaMemcpy(c.state,
                      input.data() + plan.shard_base[static_cast<std::size_t>(g)],
                      n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "persistent copy state");

        unsigned long long max_words = 1, max_desc = 1;
        for (int b = 0; b < batches; ++b) {
            max_words = std::max(
                max_words,
                batch_words[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)]);
            max_desc = std::max(
                max_desc,
                batch_entries[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)]);
        }
        ck(cudaMalloc(&c.scratch, max_words * sizeof(std::uint32_t)),
           "persistent alloc scratch");
        ck(cudaMalloc(&c.descriptor, max_desc * sizeof(ScratchDescriptor)),
           "persistent alloc descriptors");
        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "persistent alloc peer table");
        ck(cudaMalloc(&c.head_words, sizeof(unsigned long long)),
           "persistent alloc scratch head");
        ck(cudaMalloc(&c.head_desc, sizeof(unsigned long long)),
           "persistent alloc descriptor head");
        ck(cudaMalloc(&c.peer_words, sizeof(unsigned long long)),
           "persistent alloc peer words");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "persistent peer table set device");
        ck(cudaMemcpy(ctx[static_cast<std::size_t>(g)].peer_state,
                      state_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "persistent copy peer table");
    }

    const auto t0 = std::chrono::steady_clock::now();
    // All-local cycles are disjoint from cross-owner cycles and need no batch
    // barrier or scratch, so execute them once before the cross batches.
    for (int g = 0; g < ngpu; ++g) {
        const auto count = local_entries[static_cast<std::size_t>(g)];
        if (!count) continue;
        ck(cudaSetDevice(g), "persistent local launch set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMemset(c.error, 0, sizeof(int)), "persistent local reset error");
        const unsigned blocks = static_cast<unsigned>(
            std::max<unsigned long long>(1,
                std::min<unsigned long long>(requested_blocks, count)));
        persistent_local_cycle_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            c.state, c.local_list, count,
            W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "persistent local launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "persistent local sync set device");
        ck(cudaDeviceSynchronize(), "persistent local sync");
    }

    unsigned long long total_peer_words = 0;
    for (int batch = 0; batch < batches; ++batch) {
        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                batch_entries[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            if (!count) continue;
            ck(cudaSetDevice(g), "persistent phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            ck(cudaMemset(c.head_words, 0, sizeof(unsigned long long)),
               "persistent zero scratch head");
            ck(cudaMemset(c.head_desc, 0, sizeof(unsigned long long)),
               "persistent zero descriptor head");
            ck(cudaMemset(c.peer_words, 0, sizeof(unsigned long long)),
               "persistent zero peer words");
            ck(cudaMemset(c.error, 0, sizeof(int)), "persistent reset batch error");
            const unsigned blocks = static_cast<unsigned>(
                std::max<unsigned long long>(1,
                    std::min<unsigned long long>(requested_blocks, count)));
            const unsigned long long offset =
                batch_offset[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            persistent_phase_a_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
                c.state, c.scratch, c.descriptor,
                batch_words[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)],
                count, c.head_words, c.head_desc,
                c.batch_list + offset, count, batch, batches,
                W, q, reverse, tile_start, Kwin, S, ngpu, g,
                c.owner_begin, c.error);
            ck(cudaGetLastError(), "persistent phase A launch");
        }
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "persistent phase A sync set device");
            ck(cudaDeviceSynchronize(), "persistent phase A sync");
        }

        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                batch_entries[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            if (!count) continue;
            ck(cudaSetDevice(g), "persistent phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            const unsigned blocks = static_cast<unsigned>(
                std::max<unsigned long long>(1,
                    std::min<unsigned long long>(requested_blocks, count)));
            scratch_full_phase_b_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
                c.peer_state, c.scratch, c.descriptor,
                count, c.peer_words, c.error);
            ck(cudaGetLastError(), "persistent phase B launch");
        }
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "persistent phase B sync set device");
            ck(cudaDeviceSynchronize(), "persistent phase B sync");
        }

        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                batch_entries[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            if (!count) continue;
            ck(cudaSetDevice(g), "persistent batch audit set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            int error = 0;
            unsigned long long head_words = 0, head_desc = 0, peer_words = 0;
            ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
               "persistent copy batch error");
            ck(cudaMemcpy(&head_words, c.head_words, sizeof(head_words),
                          cudaMemcpyDeviceToHost), "persistent copy scratch head");
            ck(cudaMemcpy(&head_desc, c.head_desc, sizeof(head_desc),
                          cudaMemcpyDeviceToHost), "persistent copy descriptor head");
            ck(cudaMemcpy(&peer_words, c.peer_words, sizeof(peer_words),
                          cudaMemcpyDeviceToHost), "persistent copy peer words");
            if (error) fail("persistent batch device error=" + std::to_string(error));
            if (head_words !=
                    batch_words[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)] ||
                head_desc != count || peer_words != head_words)
                fail("persistent batch count/execute mismatch");
            total_peer_words += peer_words;
        }
    }
    const double runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> output(input.size());
    unsigned long long total_cross_entries = 0;
    unsigned long long total_local_entries = 0;
    unsigned long long total_list_bytes = 0;
    unsigned long long max_batch_words = 0;
    unsigned long long max_batch_entries = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "persistent gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "persistent gather state");
        total_local_entries += local_entries[static_cast<std::size_t>(g)];
        for (int b = 0; b < batches; ++b) {
            const auto e =
                batch_entries[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
            const auto w =
                batch_words[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
            total_cross_entries += e;
            max_batch_entries = std::max(max_batch_entries, e);
            max_batch_words = std::max(max_batch_words, w);
        }
        total_list_bytes +=
            (batch_offset[static_cast<std::size_t>(g)].back() +
             local_entries[static_cast<std::size_t>(g)]) * sizeof(std::uint32_t);
    }
    if (output != expected) fail("persistent segment redistribution mismatch");

    std::cout << "gridfp-p2p-persistent-segments"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " batches=" << batches
              << " states=" << tables.size()
              << " cross_segment_entries=" << total_cross_entries
              << " local_cycle_entries=" << total_local_entries
              << " persistent_list_KiB=" << double(total_list_bytes) / 1024.0
              << " logical_peer_KiB="
              << double(total_peer_words) * sizeof(std::uint32_t) / 1024.0
              << " max_batch_scratch_KiB="
              << double(max_batch_words) * sizeof(std::uint32_t) / 1024.0
              << " max_batch_descriptors=" << max_batch_entries
              << " startup_support_scan_passes=2"
              << " runtime_support_scan_passes=0"
              << " runtime_count_passes=0"
              << " setup_ms=" << setup_ms
              << " runtime_ms=" << runtime_ms
              << " cycle_closed_batches=1"
              << " remote_state_reads=0"
              << " native_peer_atomics_required=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "persistent free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        cudaFree(c.peer_words);
        cudaFree(c.head_desc);
        cudaFree(c.head_words);
        cudaFree(c.peer_state);
        cudaFree(c.descriptor);
        cudaFree(c.scratch);
        cudaFree(c.state);
        cudaFree(c.local_head);
        cudaFree(c.batch_head);
        cudaFree(c.batch_offset);
        cudaFree(c.local_list);
        cudaFree(c.batch_list);
        cudaFree(c.error);
        cudaFree(c.count_words);
        cudaFree(c.count_desc);
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
    ck(cudaGetDeviceCount(&visible), "persistent segment device count");
    if (visible < ngpu) return 3;

    run_persistent_segments(W, Kwin, S, false, ngpu, batches, blocks);
    run_persistent_segments(W, Kwin, S, true, ngpu, batches, blocks);
    std::cout << "ALL_OK gridfp_p2p_persistent_segments=1\n";
    return 0;
}
