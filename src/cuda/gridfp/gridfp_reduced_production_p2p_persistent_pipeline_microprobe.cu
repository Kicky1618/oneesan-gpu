#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_persistent_segment_microprobe_main_unused
#include "gridfp_reduced_production_p2p_persistent_segment_microprobe.cu"
#pragma pop_macro("main")

namespace {

static constexpr int PIPELINE_SLOTS = 2;

__global__ void persistent_pipeline_check_phase_a_kernel(
    const unsigned long long* scratch_head,
    const unsigned long long* descriptor_head,
    unsigned long long expected_words,
    unsigned long long expected_descriptors,
    int* error
) {
    if (blockIdx.x || threadIdx.x) return;
    if (*scratch_head != expected_words || *descriptor_head != expected_descriptors)
        set_error(error, 337);
}

__global__ void persistent_pipeline_check_phase_b_kernel(
    const unsigned long long* peer_words,
    unsigned long long expected_words,
    int* error
) {
    if (blockIdx.x || threadIdx.x) return;
    if (*peer_words != expected_words) set_error(error, 338);
}

struct PersistentPipelineCtx : PersistentCtx {
    std::uint32_t* slot_scratch[PIPELINE_SLOTS]{};
    ScratchDescriptor* slot_descriptor[PIPELINE_SLOTS]{};
    unsigned long long* slot_head_words[PIPELINE_SLOTS]{};
    unsigned long long* slot_head_desc[PIPELINE_SLOTS]{};
    unsigned long long* slot_peer_words[PIPELINE_SLOTS]{};
    cudaStream_t produce{};
    cudaStream_t consume{};
    cudaEvent_t ready[PIPELINE_SLOTS]{};
    cudaEvent_t consumed[PIPELINE_SLOTS]{};
};

void run_persistent_pipeline(
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

    ck(cudaSetDevice(0), "pipeline slab counts set device");
    install_tables(tables);
    unsigned long long* d_owner_slabs = nullptr;
    ck(cudaMalloc(&d_owner_slabs, ngpu * sizeof(unsigned long long)),
       "pipeline alloc owner slabs");
    scratch_owner_slab_counts_kernel<<<1, ngpu>>>(
        W, Kwin, ngpu, d_owner_slabs);
    ck(cudaGetLastError(), "pipeline owner slab launch");
    ck(cudaDeviceSynchronize(), "pipeline owner slab sync");
    std::vector<unsigned long long> owner_slabs(static_cast<std::size_t>(ngpu));
    ck(cudaMemcpy(owner_slabs.data(), d_owner_slabs,
                  ngpu * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "pipeline copy owner slabs");
    cudaFree(d_owner_slabs);

    std::vector<PersistentPipelineCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::vector<unsigned long long>> batch_entries(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches)));
    std::vector<std::vector<unsigned long long>> batch_words = batch_entries;
    std::vector<unsigned long long> local_entries(static_cast<std::size_t>(ngpu));

    const auto setup0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline count set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "pipeline alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                      cudaMemcpyHostToDevice), "pipeline copy owner begin");
        ck(cudaMalloc(&c.count_desc, batches * sizeof(unsigned long long)),
           "pipeline alloc batch entries");
        ck(cudaMalloc(&c.count_words, batches * sizeof(unsigned long long)),
           "pipeline alloc batch words");
        ck(cudaMalloc(&c.local_head, sizeof(unsigned long long)),
           "pipeline alloc local count");
        ck(cudaMalloc(&c.error, sizeof(int)), "pipeline alloc error");
        ck(cudaMemset(c.count_desc, 0, batches * sizeof(unsigned long long)),
           "pipeline zero batch entries");
        ck(cudaMemset(c.count_words, 0, batches * sizeof(unsigned long long)),
           "pipeline zero batch words");
        ck(cudaMemset(c.local_head, 0, sizeof(unsigned long long)),
           "pipeline zero local count");
        ck(cudaMemset(c.error, 0, sizeof(int)), "pipeline zero count error");

        const Rank64 count = owner_slabs[static_cast<std::size_t>(g)];
        const Rank64 needed =
            (count + Rank64(SCRATCH_FULL_THREADS) - 1) /
            Rank64(SCRATCH_FULL_THREADS);
        const unsigned blocks = static_cast<unsigned>(
            std::max<Rank64>(1, std::min<Rank64>(requested_blocks, needed)));
        persistent_list_count_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            count, W, q, reverse, tile_start, Kwin, S, ngpu, g, batches,
            c.owner_begin, c.count_desc, c.count_words, c.local_head, c.error);
        ck(cudaGetLastError(), "pipeline list count launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline count sync set device");
        ck(cudaDeviceSynchronize(), "pipeline count sync");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "pipeline copy count error");
        if (error) fail("pipeline count device error=" + std::to_string(error));
        ck(cudaMemcpy(batch_entries[static_cast<std::size_t>(g)].data(),
                      c.count_desc, batches * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost), "pipeline copy batch entries");
        ck(cudaMemcpy(batch_words[static_cast<std::size_t>(g)].data(),
                      c.count_words, batches * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost), "pipeline copy batch words");
        ck(cudaMemcpy(&local_entries[static_cast<std::size_t>(g)], c.local_head,
                      sizeof(unsigned long long), cudaMemcpyDeviceToHost),
           "pipeline copy local count");
    }

    std::vector<std::vector<unsigned long long>> batch_offset(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches + 1)));
    for (int g = 0; g < ngpu; ++g) {
        auto& offsets = batch_offset[static_cast<std::size_t>(g)];
        for (int b = 0; b < batches; ++b)
            offsets[static_cast<std::size_t>(b + 1)] =
                offsets[static_cast<std::size_t>(b)] +
                batch_entries[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];

        ck(cudaSetDevice(g), "pipeline list alloc set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const unsigned long long total_cross = offsets.back();
        ck(cudaMalloc(&c.batch_list,
                      std::max<unsigned long long>(1, total_cross) * sizeof(std::uint32_t)),
           "pipeline alloc batch list");
        ck(cudaMalloc(&c.local_list,
                      std::max<unsigned long long>(1, local_entries[static_cast<std::size_t>(g)]) *
                          sizeof(std::uint32_t)),
           "pipeline alloc local list");
        ck(cudaMalloc(&c.batch_offset,
                      (batches + 1) * sizeof(unsigned long long)),
           "pipeline alloc batch offsets");
        ck(cudaMalloc(&c.batch_head, batches * sizeof(unsigned long long)),
           "pipeline alloc batch heads");
        ck(cudaMemcpy(c.batch_offset, offsets.data(),
                      (batches + 1) * sizeof(unsigned long long),
                      cudaMemcpyHostToDevice), "pipeline copy batch offsets");
        ck(cudaMemset(c.batch_head, 0, batches * sizeof(unsigned long long)),
           "pipeline zero batch heads");
        ck(cudaMemset(c.local_head, 0, sizeof(unsigned long long)),
           "pipeline reset local head");
        ck(cudaMemset(c.error, 0, sizeof(int)), "pipeline reset fill error");

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
        ck(cudaGetLastError(), "pipeline list fill launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline fill sync set device");
        ck(cudaDeviceSynchronize(), "pipeline fill sync");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        unsigned long long local_head = 0;
        std::vector<unsigned long long> heads(static_cast<std::size_t>(batches));
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "pipeline copy fill error");
        ck(cudaMemcpy(heads.data(), c.batch_head,
                      batches * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
           "pipeline copy fill heads");
        ck(cudaMemcpy(&local_head, c.local_head, sizeof(local_head),
                      cudaMemcpyDeviceToHost), "pipeline copy local head");
        if (error) fail("pipeline fill device error=" + std::to_string(error));
        if (heads != batch_entries[static_cast<std::size_t>(g)] ||
            local_head != local_entries[static_cast<std::size_t>(g)])
            fail("pipeline list count/fill mismatch");
    }
    const double setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup0).count();

    std::vector<std::uint32_t*> state_ptr(static_cast<std::size_t>(ngpu), nullptr);
    unsigned long long max_slot_words[PIPELINE_SLOTS]{};
    unsigned long long max_slot_entries[PIPELINE_SLOTS]{};
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline runtime alloc set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)),
           "pipeline alloc state");
        state_ptr[static_cast<std::size_t>(g)] = c.state;
        ck(cudaMemcpy(c.state,
                      input.data() + plan.shard_base[static_cast<std::size_t>(g)],
                      n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "pipeline copy state");
        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "pipeline alloc peer table");

        for (int slot = 0; slot < PIPELINE_SLOTS; ++slot) {
            unsigned long long slot_words = 1;
            unsigned long long slot_entries = 1;
            for (int b = slot; b < batches; b += PIPELINE_SLOTS) {
                slot_words = std::max(
                    slot_words,
                    batch_words[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)]);
                slot_entries = std::max(
                    slot_entries,
                    batch_entries[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)]);
            }
            max_slot_words[slot] = std::max(max_slot_words[slot], slot_words);
            max_slot_entries[slot] = std::max(max_slot_entries[slot], slot_entries);
            ck(cudaMalloc(&c.slot_scratch[slot], slot_words * sizeof(std::uint32_t)),
               "pipeline alloc slot scratch");
            ck(cudaMalloc(&c.slot_descriptor[slot],
                          slot_entries * sizeof(ScratchDescriptor)),
               "pipeline alloc slot descriptors");
            ck(cudaMalloc(&c.slot_head_words[slot], sizeof(unsigned long long)),
               "pipeline alloc slot scratch head");
            ck(cudaMalloc(&c.slot_head_desc[slot], sizeof(unsigned long long)),
               "pipeline alloc slot descriptor head");
            ck(cudaMalloc(&c.slot_peer_words[slot], sizeof(unsigned long long)),
               "pipeline alloc slot peer words");
        }

        ck(cudaStreamCreateWithFlags(&c.produce, cudaStreamNonBlocking),
           "pipeline create produce stream");
        ck(cudaStreamCreateWithFlags(&c.consume, cudaStreamNonBlocking),
           "pipeline create consume stream");
        for (int slot = 0; slot < PIPELINE_SLOTS; ++slot) {
            ck(cudaEventCreateWithFlags(&c.ready[slot], cudaEventDisableTiming),
               "pipeline create ready event");
            ck(cudaEventCreateWithFlags(&c.consumed[slot], cudaEventDisableTiming),
               "pipeline create consumed event");
        }
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline peer table set device");
        ck(cudaMemcpy(ctx[static_cast<std::size_t>(g)].peer_state,
                      state_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "pipeline copy peer table");
    }

    // Local cycles never enter a cross-owner batch, so they can be completed
    // before the pipelined section without reducing A/B overlap.
    for (int g = 0; g < ngpu; ++g) {
        const auto count = local_entries[static_cast<std::size_t>(g)];
        ck(cudaSetDevice(g), "pipeline local set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMemset(c.error, 0, sizeof(int)), "pipeline clear runtime error");
        if (!count) continue;
        const unsigned blocks = static_cast<unsigned>(
            std::max<unsigned long long>(1,
                std::min<unsigned long long>(requested_blocks, count)));
        persistent_local_cycle_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            c.state, c.local_list, count,
            W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "pipeline local launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline local sync set device");
        ck(cudaDeviceSynchronize(), "pipeline local sync");
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int batch = 0; batch < batches; ++batch) {
        const int slot = batch & (PIPELINE_SLOTS - 1);

        // Phase A for b+1 is intentionally allowed to overlap phase B for b.
        // A slot is reused only after B[b-2] finished consuming it.
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "pipeline phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            if (batch >= PIPELINE_SLOTS)
                ck(cudaStreamWaitEvent(c.produce, c.consumed[slot], 0),
                   "pipeline wait slot consumed");

            ck(cudaMemsetAsync(c.slot_head_words[slot], 0,
                               sizeof(unsigned long long), c.produce),
               "pipeline zero scratch head");
            ck(cudaMemsetAsync(c.slot_head_desc[slot], 0,
                               sizeof(unsigned long long), c.produce),
               "pipeline zero descriptor head");
            ck(cudaMemsetAsync(c.slot_peer_words[slot], 0,
                               sizeof(unsigned long long), c.produce),
               "pipeline zero peer words");

            const auto count =
                batch_entries[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            const auto words =
                batch_words[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            if (count) {
                const unsigned blocks = static_cast<unsigned>(
                    std::max<unsigned long long>(1,
                        std::min<unsigned long long>(requested_blocks, count)));
                const unsigned long long offset =
                    batch_offset[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
                persistent_phase_a_kernel<<<blocks, SCRATCH_FULL_THREADS, 0, c.produce>>>(
                    c.state, c.slot_scratch[slot], c.slot_descriptor[slot],
                    words, count,
                    c.slot_head_words[slot], c.slot_head_desc[slot],
                    c.batch_list + offset, count, batch, batches,
                    W, q, reverse, tile_start, Kwin, S, ngpu, g,
                    c.owner_begin, c.error);
                ck(cudaGetLastError(), "pipeline phase A launch");
            }
            persistent_pipeline_check_phase_a_kernel<<<1, 1, 0, c.produce>>>(
                c.slot_head_words[slot], c.slot_head_desc[slot],
                words, count, c.error);
            ck(cudaGetLastError(), "pipeline phase A audit launch");
            ck(cudaEventRecord(c.ready[slot], c.produce),
               "pipeline record ready");
        }

        // Phase B may write another GPU's state. Waiting for every A[b]
        // preserves the old global A->B dependency without synchronizing the
        // host or blocking A[b+1].
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "pipeline phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            for (int src = 0; src < ngpu; ++src)
                ck(cudaStreamWaitEvent(
                       c.consume,
                       ctx[static_cast<std::size_t>(src)].ready[slot], 0),
                   "pipeline wait all phase A");

            const auto count =
                batch_entries[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            const auto words =
                batch_words[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            if (count) {
                const unsigned blocks = static_cast<unsigned>(
                    std::max<unsigned long long>(1,
                        std::min<unsigned long long>(requested_blocks, count)));
                scratch_full_phase_b_kernel<<<blocks, SCRATCH_FULL_THREADS, 0, c.consume>>>(
                    c.peer_state, c.slot_scratch[slot], c.slot_descriptor[slot],
                    count, c.slot_peer_words[slot], c.error);
                ck(cudaGetLastError(), "pipeline phase B launch");
            }
            persistent_pipeline_check_phase_b_kernel<<<1, 1, 0, c.consume>>>(
                c.slot_peer_words[slot], words, c.error);
            ck(cudaGetLastError(), "pipeline phase B audit launch");
            ck(cudaEventRecord(c.consumed[slot], c.consume),
               "pipeline record consumed");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline final sync set device");
        ck(cudaStreamSynchronize(ctx[static_cast<std::size_t>(g)].consume),
           "pipeline final consume sync");
        ck(cudaStreamSynchronize(ctx[static_cast<std::size_t>(g)].produce),
           "pipeline final produce sync");
    }
    const double runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> output(input.size());
    unsigned long long total_cross_entries = 0;
    unsigned long long total_local_entries = 0;
    unsigned long long total_list_bytes = 0;
    unsigned long long total_peer_words = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "pipeline copy runtime error");
        if (error) fail("pipeline runtime device error=" + std::to_string(error));

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "pipeline gather state");
        total_local_entries += local_entries[static_cast<std::size_t>(g)];
        for (int b = 0; b < batches; ++b) {
            total_cross_entries +=
                batch_entries[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
            total_peer_words +=
                batch_words[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
        }
        total_list_bytes +=
            (batch_offset[static_cast<std::size_t>(g)].back() +
             local_entries[static_cast<std::size_t>(g)]) * sizeof(std::uint32_t);
    }
    if (output != expected) fail("persistent pipeline redistribution mismatch");

    std::cout << "gridfp-p2p-persistent-pipeline"
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
              << " scratch_even_KiB="
              << double(max_slot_words[0]) * sizeof(std::uint32_t) / 1024.0
              << " scratch_odd_KiB="
              << double(max_slot_words[1]) * sizeof(std::uint32_t) / 1024.0
              << " descriptor_even_max=" << max_slot_entries[0]
              << " descriptor_odd_max=" << max_slot_entries[1]
              << " startup_support_scan_passes=2"
              << " runtime_support_scan_passes=0"
              << " runtime_count_passes=0"
              << " setup_ms=" << setup_ms
              << " runtime_ms=" << runtime_ms
              << " pipeline_slots=" << PIPELINE_SLOTS
              << " host_batch_barriers=0"
              << " cross_gpu_phase_a_fence=1"
              << " cycle_closed_batches=1"
              << " double_scratch=1"
              << " remote_state_reads=0"
              << " native_peer_atomics_required=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        for (int slot = 0; slot < PIPELINE_SLOTS; ++slot) {
            cudaEventDestroy(c.consumed[slot]);
            cudaEventDestroy(c.ready[slot]);
        }
        cudaStreamDestroy(c.consume);
        cudaStreamDestroy(c.produce);
        for (int slot = 0; slot < PIPELINE_SLOTS; ++slot) {
            cudaFree(c.slot_peer_words[slot]);
            cudaFree(c.slot_head_desc[slot]);
            cudaFree(c.slot_head_words[slot]);
            cudaFree(c.slot_descriptor[slot]);
            cudaFree(c.slot_scratch[slot]);
        }
        cudaFree(c.peer_state);
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
    const int batches = argc > 4 ? std::atoi(argv[4]) : 16;
    const unsigned blocks = argc > 5
        ? static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10)) : 256u;
    const int ngpu = argc > 6 ? std::atoi(argv[6]) : 8;
    if (W < 7 || W > 11 || Kwin < 1 || S != Kwin ||
        Kwin + S + 2 != W || batches < 2 || batches > PERSISTENT_BATCH_MAX ||
        (batches & (batches - 1)) != 0 || !blocks ||
        ngpu < 2 || ngpu > SCRATCH_FULL_MAX_GPU) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "persistent pipeline device count");
    if (visible < ngpu) return 3;

    run_persistent_pipeline(W, Kwin, S, false, ngpu, batches, blocks);
    run_persistent_pipeline(W, Kwin, S, true, ngpu, batches, blocks);
    std::cout << "ALL_OK gridfp_p2p_persistent_pipeline=1\n";
    return 0;
}
