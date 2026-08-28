#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_cycle_owner_descriptorless_microprobe_main_unused
#include "gridfp_reduced_production_p2p_cycle_owner_descriptorless_microprobe.cu"
#pragma pop_macro("main")

namespace {

struct CycleOwnerPipelineCtx {
    std::uint32_t* state = nullptr;
    std::uint32_t* scratch[2]{nullptr, nullptr};
    std::uint32_t* batch_list = nullptr;
    std::uint32_t* local_list = nullptr;
    std::uint32_t** peer_state = nullptr;
    Rank64* owner_begin = nullptr;
    unsigned long long* class_list_begin = nullptr;
    unsigned long long* class_scratch_begin = nullptr;
    unsigned long long* peer_words = nullptr; // one counter per batch
    int* error = nullptr;
    cudaStream_t stream[2]{nullptr, nullptr};
};

void run_cycle_owner_pipeline(
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
    std::vector<CycleOwnerPipelineCtx> ctx(static_cast<std::size_t>(ngpu));
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
            static_cast<std::size_t>(batches * class_stride));
        std::vector<unsigned long long> scratch_meta(
            static_cast<std::size_t>(batches * class_stride));
        std::array<unsigned long long, 2> max_plane_words{1, 1};

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
                    scratch_cursor +=
                        n * static_cast<unsigned long long>(nseg) * pc;
                }
            }
            if (list_cursor != part.size() ||
                scratch_cursor !=
                    lists.words[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)])
                fail("cycle-owner pipeline class plan mismatch");
            max_plane_words[static_cast<std::size_t>(b & 1)] = std::max(
                max_plane_words[static_cast<std::size_t>(b & 1)], scratch_cursor);
            for (const auto& entry : part) packed.push_back(entry.packed);
        }
        offsets[static_cast<std::size_t>(batches)] = packed.size();
        plane_words[static_cast<std::size_t>(g)] = max_plane_words;

        ck(cudaSetDevice(g), "cycle-owner pipeline set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "cycle-owner pipeline alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                      cudaMemcpyHostToDevice), "cycle-owner pipeline copy owner begin");

        ck(cudaMalloc(&c.batch_list,
                      std::max<std::size_t>(1, packed.size()) * sizeof(std::uint32_t)),
           "cycle-owner pipeline alloc batch list");
        if (!packed.empty())
            ck(cudaMemcpy(c.batch_list, packed.data(), packed.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice),
               "cycle-owner pipeline copy batch list");
        const auto& local = lists.local[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.local_list,
                      std::max<std::size_t>(1, local.size()) * sizeof(std::uint32_t)),
           "cycle-owner pipeline alloc local list");
        if (!local.empty())
            ck(cudaMemcpy(c.local_list, local.data(), local.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice),
               "cycle-owner pipeline copy local list");

        ck(cudaMalloc(&c.class_list_begin,
                      list_meta.size() * sizeof(unsigned long long)),
           "cycle-owner pipeline alloc list meta");
        ck(cudaMalloc(&c.class_scratch_begin,
                      scratch_meta.size() * sizeof(unsigned long long)),
           "cycle-owner pipeline alloc scratch meta");
        ck(cudaMemcpy(c.class_list_begin, list_meta.data(),
                      list_meta.size() * sizeof(unsigned long long),
                      cudaMemcpyHostToDevice),
           "cycle-owner pipeline copy list meta");
        ck(cudaMemcpy(c.class_scratch_begin, scratch_meta.data(),
                      scratch_meta.size() * sizeof(unsigned long long),
                      cudaMemcpyHostToDevice),
           "cycle-owner pipeline copy scratch meta");

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)),
           "cycle-owner pipeline alloc state");
        state_ptr[static_cast<std::size_t>(g)] = c.state;
        ck(cudaMemcpy(c.state,
                      input.data() + plan.shard_base[static_cast<std::size_t>(g)],
                      n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "cycle-owner pipeline copy state");

        for (int p = 0; p < 2; ++p) {
            ck(cudaMalloc(&c.scratch[p],
                          max_plane_words[static_cast<std::size_t>(p)] *
                              sizeof(std::uint32_t)),
               "cycle-owner pipeline alloc scratch plane");
            ck(cudaStreamCreateWithFlags(&c.stream[p], cudaStreamNonBlocking),
               "cycle-owner pipeline create stream");
        }
        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "cycle-owner pipeline alloc peer table");
        ck(cudaMalloc(&c.peer_words,
                      static_cast<std::size_t>(batches) * sizeof(unsigned long long)),
           "cycle-owner pipeline alloc peer counters");
        ck(cudaMalloc(&c.error, sizeof(int)),
           "cycle-owner pipeline alloc error");
        ck(cudaMemset(c.peer_words, 0,
                      static_cast<std::size_t>(batches) * sizeof(unsigned long long)),
           "cycle-owner pipeline zero peer counters");
        ck(cudaMemset(c.error, 0, sizeof(int)),
           "cycle-owner pipeline zero error");
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner pipeline peer table set device");
        ck(cudaMemcpy(ctx[static_cast<std::size_t>(g)].peer_state,
                      state_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice),
           "cycle-owner pipeline copy peer table");
    }
    const double setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup0).count();

    const auto t0 = std::chrono::steady_clock::now();

    for (int g = 0; g < ngpu; ++g) {
        const auto count = lists.local[static_cast<std::size_t>(g)].size();
        if (!count) continue;
        ck(cudaSetDevice(g), "cycle-owner pipeline local set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const unsigned blocks = static_cast<unsigned>(
            std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
        persistent_local_cycle_kernel<<<blocks, SCRATCH_FULL_THREADS, 0, c.stream[0]>>>(
            c.state, c.local_list, count,
            W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "cycle-owner pipeline local launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner pipeline local sync set device");
        ck(cudaStreamSynchronize(ctx[static_cast<std::size_t>(g)].stream[0]),
           "cycle-owner pipeline local sync");
    }

    for (int batch = 0; batch < batches; ++batch) {
        const int plane = batch & 1;
        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            if (!count) continue;
            ck(cudaSetDevice(g), "cycle-owner pipeline phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            const unsigned blocks = static_cast<unsigned>(
                std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
            const auto offset =
                batch_offset[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            cycle_owner_phase_a_kernel<<<blocks, SCRATCH_FULL_THREADS, 0, c.stream[plane]>>>(
                c.state, c.scratch[plane], c.batch_list + offset, count,
                c.class_list_begin + batch * class_stride,
                c.class_scratch_begin + batch * class_stride,
                batch, batches, W, q, reverse, tile_start, Kwin,
                ngpu, g, c.owner_begin, c.error);
            ck(cudaGetLastError(), "cycle-owner pipeline phase A launch");
        }

        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "cycle-owner pipeline A barrier set device");
            ck(cudaStreamSynchronize(ctx[static_cast<std::size_t>(g)].stream[plane]),
               "cycle-owner pipeline A barrier");
        }

        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            if (!count) continue;
            ck(cudaSetDevice(g), "cycle-owner pipeline phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            const unsigned blocks = static_cast<unsigned>(
                std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
            const auto offset =
                batch_offset[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            cycle_owner_phase_b_kernel<<<blocks, SCRATCH_FULL_THREADS, 0, c.stream[plane]>>>(
                c.peer_state, c.scratch[plane], c.batch_list + offset, count,
                c.class_list_begin + batch * class_stride,
                c.class_scratch_begin + batch * class_stride,
                batch, batches, W, q, reverse, tile_start, Kwin,
                ngpu, g, c.owner_begin, c.peer_words + batch, c.error);
            ck(cudaGetLastError(), "cycle-owner pipeline phase B launch");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner pipeline final sync set device");
        for (int p = 0; p < 2; ++p)
            ck(cudaStreamSynchronize(ctx[static_cast<std::size_t>(g)].stream[p]),
               "cycle-owner pipeline final sync");
    }
    const double runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    unsigned long long total_peer_words = 0;
    unsigned long long cross_entries = 0;
    unsigned long long local_entries = 0;
    unsigned long long max_plane0_words = 0;
    unsigned long long max_plane1_words = 0;
    std::vector<std::uint32_t> output(input.size());

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner pipeline audit set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        std::vector<unsigned long long> got_words(static_cast<std::size_t>(batches));
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "cycle-owner pipeline copy error");
        ck(cudaMemcpy(got_words.data(), c.peer_words,
                      static_cast<std::size_t>(batches) * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost),
           "cycle-owner pipeline copy peer counters");
        if (error) fail("cycle-owner pipeline device error=" + std::to_string(error));
        for (int b = 0; b < batches; ++b) {
            const auto want =
                lists.words[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
            if (got_words[static_cast<std::size_t>(b)] != want)
                fail("cycle-owner pipeline peer word mismatch");
            total_peer_words += got_words[static_cast<std::size_t>(b)];
            cross_entries +=
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)].size();
        }
        local_entries += lists.local[static_cast<std::size_t>(g)].size();
        max_plane0_words = std::max(
            max_plane0_words, plane_words[static_cast<std::size_t>(g)][0]);
        max_plane1_words = std::max(
            max_plane1_words, plane_words[static_cast<std::size_t>(g)][1]);

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "cycle-owner pipeline gather state");
    }
    if (output != expected) fail("cycle-owner pipeline redistribution mismatch");

    std::cout << "gridfp-p2p-cycle-owner-pipeline"
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
              << " scratch_plane0_max_KiB="
              << double(max_plane0_words) * sizeof(std::uint32_t) / 1024.0
              << " scratch_plane1_max_KiB="
              << double(max_plane1_words) * sizeof(std::uint32_t) / 1024.0
              << " max_segments_per_owner_cycle=" << CYCLE_OWNER_MAX_SEGMENTS
              << " descriptor_bytes=0"
              << " scratch_allocator_atomics=0"
              << " startup_gpu_support_scan_passes=0"
              << " runtime_support_scan_passes=0"
              << " runtime_count_passes=0"
              << " scratch_planes=2"
              << " streams_per_gpu=2"
              << " phase_b_next_a_overlap=1"
              << " compressed_cycle_owner_list=1"
              << " setup_ms=" << setup_ms
              << " runtime_ms=" << runtime_ms
              << " remote_state_reads=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner pipeline free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        for (int p = 0; p < 2; ++p) {
            if (c.stream[p]) cudaStreamDestroy(c.stream[p]);
            cudaFree(c.scratch[p]);
        }
        cudaFree(c.error);
        cudaFree(c.peer_words);
        cudaFree(c.peer_state);
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
    const int batches = argc > 4 ? std::atoi(argv[4]) : 16;
    const unsigned blocks = argc > 5
        ? static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10)) : 256u;
    const int ngpu = argc > 6 ? std::atoi(argv[6]) : 8;
    if (W < 7 || W > 11 || Kwin < 1 || S != Kwin ||
        Kwin + S + 2 != W || batches < 4 || batches > PERSISTENT_BATCH_MAX ||
        (batches & (batches - 1)) != 0 || !blocks ||
        ngpu < 2 || ngpu > SCRATCH_FULL_MAX_GPU) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cycle-owner pipeline device count");
    if (visible < ngpu) return 3;

    run_cycle_owner_pipeline(W, Kwin, S, false, ngpu, batches, blocks);
    run_cycle_owner_pipeline(W, Kwin, S, true, ngpu, batches, blocks);
    std::cout << "ALL_OK gridfp_p2p_cycle_owner_pipeline=1\n";
    return 0;
}
