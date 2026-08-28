#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_host_persistent_descriptorless_microprobe_main_unused
#include "gridfp_reduced_production_p2p_host_persistent_descriptorless_microprobe.cu"
#pragma pop_macro("main")

namespace {

struct PipelineCtx {
    std::uint32_t* state = nullptr;
    std::uint32_t* scratch[2]{nullptr, nullptr};
    std::uint32_t* batch_list = nullptr;
    std::uint32_t* local_list = nullptr;
    std::uint32_t** peer_state = nullptr;
    Rank64* owner_begin = nullptr;
    unsigned long long* occ_list_begin = nullptr;
    unsigned long long* occ_scratch_begin = nullptr;
    unsigned long long* peer_words = nullptr; // one counter per batch
    int* error = nullptr;
    cudaStream_t stream[2]{nullptr, nullptr};
};

void run_pipeline_executor(
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
    std::vector<PipelineCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> state_ptr(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<std::vector<unsigned long long>> batch_offset(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches + 1)));
    std::vector<std::array<unsigned long long, 2>> plane_words(
        static_cast<std::size_t>(ngpu));

    for (int g = 0; g < ngpu; ++g) {
        auto& offsets = batch_offset[static_cast<std::size_t>(g)];
        std::vector<std::uint32_t> packed;
        std::vector<unsigned long long> list_meta(
            static_cast<std::size_t>(batches * sectors));
        std::vector<unsigned long long> scratch_meta(
            static_cast<std::size_t>(batches * sectors));
        std::array<unsigned long long, 2> max_plane_words{1, 1};

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
                fail("pipeline host sector plan mismatch");
            max_plane_words[static_cast<std::size_t>(b & 1)] = std::max(
                max_plane_words[static_cast<std::size_t>(b & 1)], scratch_cursor);
            packed.insert(packed.end(), part.begin(), part.end());
        }
        offsets[static_cast<std::size_t>(batches)] = packed.size();
        plane_words[static_cast<std::size_t>(g)] = max_plane_words;

        ck(cudaSetDevice(g), "pipeline set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "pipeline alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                      cudaMemcpyHostToDevice), "pipeline copy owner begin");
        ck(cudaMalloc(&c.batch_list,
                      std::max<std::size_t>(1, packed.size()) * sizeof(std::uint32_t)),
           "pipeline alloc batch list");
        if (!packed.empty())
            ck(cudaMemcpy(c.batch_list, packed.data(), packed.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "pipeline copy batch list");

        const auto& local = lists.local[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.local_list,
                      std::max<std::size_t>(1, local.size()) * sizeof(std::uint32_t)),
           "pipeline alloc local list");
        if (!local.empty())
            ck(cudaMemcpy(c.local_list, local.data(), local.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "pipeline copy local list");

        ck(cudaMalloc(&c.occ_list_begin,
                      list_meta.size() * sizeof(unsigned long long)),
           "pipeline alloc list meta");
        ck(cudaMalloc(&c.occ_scratch_begin,
                      scratch_meta.size() * sizeof(unsigned long long)),
           "pipeline alloc scratch meta");
        ck(cudaMemcpy(c.occ_list_begin, list_meta.data(),
                      list_meta.size() * sizeof(unsigned long long),
                      cudaMemcpyHostToDevice), "pipeline copy list meta");
        ck(cudaMemcpy(c.occ_scratch_begin, scratch_meta.data(),
                      scratch_meta.size() * sizeof(unsigned long long),
                      cudaMemcpyHostToDevice), "pipeline copy scratch meta");

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)),
           "pipeline alloc state");
        state_ptr[static_cast<std::size_t>(g)] = c.state;
        ck(cudaMemcpy(c.state,
                      input.data() + plan.shard_base[static_cast<std::size_t>(g)],
                      n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "pipeline copy state");

        for (int p = 0; p < 2; ++p) {
            ck(cudaMalloc(&c.scratch[p],
                          max_plane_words[static_cast<std::size_t>(p)] *
                              sizeof(std::uint32_t)),
               "pipeline alloc scratch plane");
            ck(cudaStreamCreateWithFlags(&c.stream[p], cudaStreamNonBlocking),
               "pipeline create stream");
        }
        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "pipeline alloc peer table");
        ck(cudaMalloc(&c.peer_words,
                      static_cast<std::size_t>(batches) * sizeof(unsigned long long)),
           "pipeline alloc peer counters");
        ck(cudaMalloc(&c.error, sizeof(int)), "pipeline alloc error");
        ck(cudaMemset(c.peer_words, 0,
                      static_cast<std::size_t>(batches) * sizeof(unsigned long long)),
           "pipeline zero peer counters");
        ck(cudaMemset(c.error, 0, sizeof(int)), "pipeline zero error");
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline peer table set device");
        ck(cudaMemcpy(ctx[static_cast<std::size_t>(g)].peer_state,
                      state_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "pipeline copy peer table");
    }
    const double setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup0).count();

    const auto t0 = std::chrono::steady_clock::now();

    // All-local cycles are disjoint from every cross-owner batch.  Execute
    // them once and drain before entering the double-scratch pipeline.
    for (int g = 0; g < ngpu; ++g) {
        const auto count = lists.local[static_cast<std::size_t>(g)].size();
        if (!count) continue;
        ck(cudaSetDevice(g), "pipeline local set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const unsigned blocks = static_cast<unsigned>(
            std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
        persistent_local_cycle_kernel<<<blocks, SCRATCH_FULL_THREADS, 0, c.stream[0]>>>(
            c.state, c.local_list, count,
            W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "pipeline local launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline local sync set device");
        ck(cudaStreamSynchronize(ctx[static_cast<std::size_t>(g)].stream[0]),
           "pipeline local sync");
    }

    // Batch b uses plane b&1.  Synchronizing that plane before B_b gives a
    // global A_b barrier and also waits for B_{b-2} before the scratch plane is
    // reused.  B_{b-1} lives on the other stream and can overlap A_b because
    // cycle-closed batches are disjoint state sets.
    for (int batch = 0; batch < batches; ++batch) {
        const int plane = batch & 1;
        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            if (!count) continue;
            ck(cudaSetDevice(g), "pipeline phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            const unsigned blocks = static_cast<unsigned>(
                std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
            const auto offset =
                batch_offset[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            noddesc_phase_a_kernel<<<blocks, SCRATCH_FULL_THREADS, 0, c.stream[plane]>>>(
                c.state, c.scratch[plane], c.batch_list + offset, count,
                c.occ_list_begin + batch * sectors,
                c.occ_scratch_begin + batch * sectors,
                batch, batches, W, q, reverse, tile_start, Kwin, S,
                ngpu, g, c.owner_begin, c.error);
            ck(cudaGetLastError(), "pipeline phase A launch");
        }

        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "pipeline phase A barrier set device");
            ck(cudaStreamSynchronize(ctx[static_cast<std::size_t>(g)].stream[plane]),
               "pipeline phase A barrier");
        }

        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            if (!count) continue;
            ck(cudaSetDevice(g), "pipeline phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            const unsigned blocks = static_cast<unsigned>(
                std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
            const auto offset =
                batch_offset[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            noddesc_phase_b_kernel<<<blocks, SCRATCH_FULL_THREADS, 0, c.stream[plane]>>>(
                c.peer_state, c.scratch[plane], c.batch_list + offset, count,
                c.occ_list_begin + batch * sectors,
                c.occ_scratch_begin + batch * sectors,
                batch, batches, W, q, reverse, tile_start, Kwin, S,
                ngpu, g, c.owner_begin, c.peer_words + batch, c.error);
            ck(cudaGetLastError(), "pipeline phase B launch");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline final sync set device");
        for (int p = 0; p < 2; ++p)
            ck(cudaStreamSynchronize(ctx[static_cast<std::size_t>(g)].stream[p]),
               "pipeline final sync");
    }
    const double runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    unsigned long long total_peer_words = 0;
    std::vector<std::uint32_t> output(input.size());
    unsigned long long list_entries = 0;
    unsigned long long max_plane0_words = 0;
    unsigned long long max_plane1_words = 0;

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline audit set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        std::vector<unsigned long long> got_words(static_cast<std::size_t>(batches));
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "pipeline copy error");
        ck(cudaMemcpy(got_words.data(), c.peer_words,
                      static_cast<std::size_t>(batches) * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost), "pipeline copy peer counters");
        if (error) fail("pipeline device error=" + std::to_string(error));
        for (int b = 0; b < batches; ++b) {
            const auto want =
                lists.words[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
            if (got_words[static_cast<std::size_t>(b)] != want)
                fail("pipeline peer word mismatch");
            total_peer_words += got_words[static_cast<std::size_t>(b)];
        }

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "pipeline gather state");
        list_entries += lists.local[static_cast<std::size_t>(g)].size();
        for (int b = 0; b < batches; ++b)
            list_entries +=
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)].size();
        max_plane0_words = std::max(
            max_plane0_words, plane_words[static_cast<std::size_t>(g)][0]);
        max_plane1_words = std::max(
            max_plane1_words, plane_words[static_cast<std::size_t>(g)][1]);
    }
    if (output != expected) fail("pipeline redistribution mismatch");

    std::cout << "gridfp-p2p-host-persistent-pipeline"
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
              << " scratch_plane0_max_KiB="
              << double(max_plane0_words) * sizeof(std::uint32_t) / 1024.0
              << " scratch_plane1_max_KiB="
              << double(max_plane1_words) * sizeof(std::uint32_t) / 1024.0
              << " descriptor_bytes=0"
              << " scratch_allocator_atomics=0"
              << " startup_gpu_support_scan_passes=0"
              << " runtime_support_scan_passes=0"
              << " runtime_count_passes=0"
              << " scratch_planes=2"
              << " streams_per_gpu=2"
              << " phase_b_next_a_overlap=1"
              << " cycle_closed_batches=1"
              << " setup_ms=" << setup_ms
              << " runtime_ms=" << runtime_ms
              << " remote_state_reads=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        for (int p = 0; p < 2; ++p) {
            if (c.stream[p]) cudaStreamDestroy(c.stream[p]);
            cudaFree(c.scratch[p]);
        }
        cudaFree(c.error);
        cudaFree(c.peer_words);
        cudaFree(c.peer_state);
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
    const int batches = argc > 4 ? std::atoi(argv[4]) : 16;
    const unsigned blocks = argc > 5
        ? static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10)) : 256u;
    const int ngpu = argc > 6 ? std::atoi(argv[6]) : 8;
    if (W < 7 || W > 11 || Kwin < 1 || S != Kwin ||
        Kwin + S + 2 != W || batches < 4 || batches > PERSISTENT_BATCH_MAX ||
        (batches & (batches - 1)) != 0 || !blocks ||
        ngpu < 2 || ngpu > SCRATCH_FULL_MAX_GPU) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "pipeline device count");
    if (visible < ngpu) return 3;

    run_pipeline_executor(W, Kwin, S, false, ngpu, batches, blocks);
    run_pipeline_executor(W, Kwin, S, true, ngpu, batches, blocks);
    std::cout << "ALL_OK gridfp_p2p_host_persistent_pipeline=1\n";
    return 0;
}
