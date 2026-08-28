#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_host_persistent_list_microprobe_main_unused
#include "gridfp_reduced_production_p2p_host_persistent_list_microprobe.cu"
#pragma pop_macro("main")

namespace {

struct NoDescriptorCtx {
    std::uint32_t* state = nullptr;
    std::uint32_t* scratch = nullptr;
    std::uint32_t* batch_list = nullptr;
    std::uint32_t* local_list = nullptr;
    std::uint32_t** peer_state = nullptr;
    Rank64* owner_begin = nullptr;
    unsigned long long* occ_list_begin = nullptr;
    unsigned long long* occ_scratch_begin = nullptr;
    unsigned long long* peer_words = nullptr;
    int* error = nullptr;
};

__global__ void noddesc_phase_a_kernel(
    std::uint32_t* local_state,
    std::uint32_t* scratch,
    const std::uint32_t* __restrict__ list,
    unsigned long long count,
    const unsigned long long* __restrict__ occ_list_begin,
    const unsigned long long* __restrict__ occ_scratch_begin,
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
    __shared__ unsigned long long scratch_offset;
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
                set_error(error, 351);
            }
            if (segment.status == 2) {
                const int occupied = __popc(slab.support);
                const unsigned long long sector_ix =
                    ix - occ_list_begin[occupied];
                scratch_offset = occ_scratch_begin[occupied] +
                    sector_ix * static_cast<unsigned long long>(segment.primitive_count);
            }
        }
        __syncthreads();
        if (*error) return;

        const int pc = static_cast<int>(segment.primitive_count);
        const int tail = segment.len - 1;
        for (int i = threadIdx.x; i < pc; i += blockDim.x) {
            scratch[scratch_offset + static_cast<unsigned long long>(i)] =
                local_state[segment.local[tail] + static_cast<Rank64>(i)];
            for (int h = tail; h > 0; --h) {
                local_state[segment.local[h] + static_cast<Rank64>(i)] =
                    local_state[segment.local[h - 1] + static_cast<Rank64>(i)];
            }
        }
        __syncthreads();
    }
}

__global__ void noddesc_phase_b_kernel(
    std::uint32_t* const* __restrict__ peer_state,
    const std::uint32_t* __restrict__ scratch,
    const std::uint32_t* __restrict__ list,
    unsigned long long count,
    const unsigned long long* __restrict__ occ_list_begin,
    const unsigned long long* __restrict__ occ_scratch_begin,
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
    __shared__ OwnerLocalSegment segment;
    __shared__ unsigned long long scratch_offset;
    const unsigned long long first = blockIdx.x;
    const unsigned long long stride = gridDim.x;
    for (unsigned long long ix = first; ix < count; ix += stride) {
        if (threadIdx.x == 0) {
            const OwnerSupportSlabDevice slab = persistent_unpack_slab(list[ix]);
            owner_local_segment_device(
                slab, W, q, reverse, tile_start, Kwin, S, owner, ngpu,
                owner_begin, segment);
            if (segment.status != 2 || segment.dst_owner < 0 ||
                segment.dst_owner >= ngpu ||
                persistent_batch_id(
                    slab.support, slab.blocked != 0, W, q, Kwin, batches) !=
                    expected_batch) {
                set_error(error, 352);
            }
            if (segment.status == 2) {
                const int occupied = __popc(slab.support);
                const unsigned long long sector_ix =
                    ix - occ_list_begin[occupied];
                scratch_offset = occ_scratch_begin[occupied] +
                    sector_ix * static_cast<unsigned long long>(segment.primitive_count);
            }
        }
        __syncthreads();
        if (*error) return;

        std::uint32_t* dst = peer_state[segment.dst_owner] + segment.dst_local;
        const int pc = static_cast<int>(segment.primitive_count);
        for (int i = threadIdx.x; i < pc; i += blockDim.x)
            dst[i] = scratch[scratch_offset + static_cast<unsigned long long>(i)];
        __syncthreads();
        if (threadIdx.x == 0)
            atomicAdd(peer_words, static_cast<unsigned long long>(pc));
        __syncthreads();
    }
}

void run_noddesc_executor(
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
    HostPersistentLists lists = build_host_persistent_lists(
        W, Kwin, S, reverse, ngpu, batches, tables);

    const int sectors = W + 1;
    std::vector<NoDescriptorCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> state_ptr(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<std::vector<unsigned long long>> batch_offset(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches + 1)));

    for (int g = 0; g < ngpu; ++g) {
        auto& offsets = batch_offset[static_cast<std::size_t>(g)];
        std::vector<std::uint32_t> packed;
        std::vector<unsigned long long> list_meta(
            static_cast<std::size_t>(batches * sectors));
        std::vector<unsigned long long> scratch_meta(
            static_cast<std::size_t>(batches * sectors));
        unsigned long long max_words = 1;

        for (int b = 0; b < batches; ++b) {
            auto& part = lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
            std::sort(part.begin(), part.end(), [](std::uint32_t a, std::uint32_t b) {
                const int pa = __builtin_popcount(a & PERSISTENT_SUPPORT_MASK);
                const int pb = __builtin_popcount(b & PERSISTENT_SUPPORT_MASK);
                return pa != pb ? pa < pb : a < b;
            });
            offsets[static_cast<std::size_t>(b)] = packed.size();

            std::array<unsigned long long, RP_MAX_W + 1> count_by_occ{};
            for (std::uint32_t x : part)
                ++count_by_occ[static_cast<std::size_t>(
                    __builtin_popcount(x & PERSISTENT_SUPPORT_MASK))];

            unsigned long long list_cursor = 0;
            unsigned long long scratch_cursor = 0;
            for (int occ = 0; occ <= W; ++occ) {
                const std::size_t mi = static_cast<std::size_t>(b * sectors + occ);
                list_meta[mi] = list_cursor;
                scratch_meta[mi] = scratch_cursor;
                const auto n = count_by_occ[static_cast<std::size_t>(occ)];
                list_cursor += n;
                scratch_cursor += n *
                    tables.primitive[static_cast<std::size_t>(occ)][1];
            }
            if (list_cursor != part.size() ||
                scratch_cursor !=
                    lists.words[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)])
                fail("descriptorless host sector plan mismatch");
            max_words = std::max(max_words, scratch_cursor);
            packed.insert(packed.end(), part.begin(), part.end());
        }
        offsets[static_cast<std::size_t>(batches)] = packed.size();

        ck(cudaSetDevice(g), "descriptorless set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "descriptorless alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                      cudaMemcpyHostToDevice), "descriptorless copy owner begin");
        ck(cudaMalloc(&c.batch_list,
                      std::max<std::size_t>(1, packed.size()) * sizeof(std::uint32_t)),
           "descriptorless alloc batch list");
        if (!packed.empty())
            ck(cudaMemcpy(c.batch_list, packed.data(), packed.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "descriptorless copy batch list");
        const auto& local = lists.local[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.local_list,
                      std::max<std::size_t>(1, local.size()) * sizeof(std::uint32_t)),
           "descriptorless alloc local list");
        if (!local.empty())
            ck(cudaMemcpy(c.local_list, local.data(), local.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "descriptorless copy local list");

        ck(cudaMalloc(&c.occ_list_begin,
                      list_meta.size() * sizeof(unsigned long long)),
           "descriptorless alloc list meta");
        ck(cudaMalloc(&c.occ_scratch_begin,
                      scratch_meta.size() * sizeof(unsigned long long)),
           "descriptorless alloc scratch meta");
        ck(cudaMemcpy(c.occ_list_begin, list_meta.data(),
                      list_meta.size() * sizeof(unsigned long long),
                      cudaMemcpyHostToDevice), "descriptorless copy list meta");
        ck(cudaMemcpy(c.occ_scratch_begin, scratch_meta.data(),
                      scratch_meta.size() * sizeof(unsigned long long),
                      cudaMemcpyHostToDevice), "descriptorless copy scratch meta");

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)),
           "descriptorless alloc state");
        state_ptr[static_cast<std::size_t>(g)] = c.state;
        ck(cudaMemcpy(c.state,
                      input.data() + plan.shard_base[static_cast<std::size_t>(g)],
                      n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "descriptorless copy state");
        ck(cudaMalloc(&c.scratch, max_words * sizeof(std::uint32_t)),
           "descriptorless alloc scratch");
        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "descriptorless alloc peer table");
        ck(cudaMalloc(&c.peer_words, sizeof(unsigned long long)),
           "descriptorless alloc peer words");
        ck(cudaMalloc(&c.error, sizeof(int)), "descriptorless alloc error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "descriptorless peer table set device");
        ck(cudaMemcpy(ctx[static_cast<std::size_t>(g)].peer_state,
                      state_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "descriptorless copy peer table");
    }
    const double setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup0).count();

    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        const auto count = lists.local[static_cast<std::size_t>(g)].size();
        if (!count) continue;
        ck(cudaSetDevice(g), "descriptorless local set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMemset(c.error, 0, sizeof(int)), "descriptorless zero local error");
        const unsigned blocks = static_cast<unsigned>(
            std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
        persistent_local_cycle_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            c.state, c.local_list, count,
            W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "descriptorless local launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "descriptorless local sync set device");
        ck(cudaDeviceSynchronize(), "descriptorless local sync");
        int error = 0;
        ck(cudaMemcpy(&error, ctx[static_cast<std::size_t>(g)].error,
                      sizeof(error), cudaMemcpyDeviceToHost),
           "descriptorless local copy error");
        if (error) fail("descriptorless local device error=" + std::to_string(error));
    }

    unsigned long long total_peer_words = 0;
    for (int batch = 0; batch < batches; ++batch) {
        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            if (!count) continue;
            ck(cudaSetDevice(g), "descriptorless phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            ck(cudaMemset(c.peer_words, 0, sizeof(unsigned long long)),
               "descriptorless zero peer words");
            ck(cudaMemset(c.error, 0, sizeof(int)), "descriptorless zero error");
            const unsigned blocks = static_cast<unsigned>(
                std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
            const auto offset =
                batch_offset[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            noddesc_phase_a_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
                c.state, c.scratch, c.batch_list + offset, count,
                c.occ_list_begin + batch * sectors,
                c.occ_scratch_begin + batch * sectors,
                batch, batches, W, q, reverse, tile_start, Kwin, S,
                ngpu, g, c.owner_begin, c.error);
            ck(cudaGetLastError(), "descriptorless phase A launch");
        }
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "descriptorless phase A sync set device");
            ck(cudaDeviceSynchronize(), "descriptorless phase A sync");
        }

        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            if (!count) continue;
            ck(cudaSetDevice(g), "descriptorless phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            const unsigned blocks = static_cast<unsigned>(
                std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
            const auto offset =
                batch_offset[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            noddesc_phase_b_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
                c.peer_state, c.scratch, c.batch_list + offset, count,
                c.occ_list_begin + batch * sectors,
                c.occ_scratch_begin + batch * sectors,
                batch, batches, W, q, reverse, tile_start, Kwin, S,
                ngpu, g, c.owner_begin, c.peer_words, c.error);
            ck(cudaGetLastError(), "descriptorless phase B launch");
        }
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "descriptorless phase B sync set device");
            ck(cudaDeviceSynchronize(), "descriptorless phase B sync");
        }

        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            if (!count) continue;
            ck(cudaSetDevice(g), "descriptorless audit set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            int error = 0;
            unsigned long long peer_words = 0;
            ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
               "descriptorless copy error");
            ck(cudaMemcpy(&peer_words, c.peer_words, sizeof(peer_words),
                          cudaMemcpyDeviceToHost), "descriptorless copy peer words");
            const auto expected_words =
                lists.words[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            if (error) fail("descriptorless device error=" + std::to_string(error));
            if (peer_words != expected_words)
                fail("descriptorless peer word mismatch");
            total_peer_words += peer_words;
        }
    }
    const double runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> output(input.size());
    unsigned long long list_entries = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "descriptorless gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "descriptorless gather state");
        list_entries += lists.local[static_cast<std::size_t>(g)].size();
        for (int b = 0; b < batches; ++b)
            list_entries +=
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)].size();
    }
    if (output != expected) fail("descriptorless redistribution mismatch");

    std::cout << "gridfp-p2p-host-persistent-descriptorless"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " batches=" << batches
              << " states=" << tables.size()
              << " list_entries=" << list_entries
              << " logical_peer_KiB="
              << double(total_peer_words) * sizeof(std::uint32_t) / 1024.0
              << " descriptor_bytes=0"
              << " scratch_allocator_atomics=0"
              << " startup_gpu_support_scan_passes=0"
              << " runtime_support_scan_passes=0"
              << " runtime_count_passes=0"
              << " setup_ms=" << setup_ms
              << " runtime_ms=" << runtime_ms
              << " remote_state_reads=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "descriptorless free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        cudaFree(c.error);
        cudaFree(c.peer_words);
        cudaFree(c.peer_state);
        cudaFree(c.scratch);
        cudaFree(c.state);
        cudaFree(c.occ_scratch_begin);
        cudaFree(c.occ_list_begin);
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
    ck(cudaGetDeviceCount(&visible), "descriptorless device count");
    if (visible < ngpu) return 3;

    run_noddesc_executor(W, Kwin, S, false, ngpu, batches, blocks);
    run_noddesc_executor(W, Kwin, S, true, ngpu, batches, blocks);
    std::cout << "ALL_OK gridfp_p2p_host_persistent_descriptorless=1\n";
    return 0;
}
