#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_cycle_owner_descriptorless_microprobe_main_unused
#include "gridfp_reduced_production_p2p_cycle_owner_descriptorless_microprobe.cu"
#pragma pop_macro("main")

namespace {

static constexpr int CYCLE_OWNER_PIPELINE_SLOTS = 2;

__global__ void cycle_owner_pipeline_audit_kernel(
    const unsigned long long* peer_words,
    unsigned long long expected_words,
    int* error
) {
    if (blockIdx.x || threadIdx.x) return;
    if (*peer_words != expected_words) set_error(error, 368);
}

struct CycleOwnerPipelineCtx {
    std::uint32_t* state = nullptr;
    std::uint32_t* scratch[CYCLE_OWNER_PIPELINE_SLOTS]{};
    std::uint32_t* batch_list = nullptr;
    std::uint32_t* local_list = nullptr;
    std::uint32_t** peer_state = nullptr;
    Rank64* owner_begin = nullptr;
    unsigned long long* class_list_begin = nullptr;
    unsigned long long* class_scratch_begin = nullptr;
    unsigned long long* peer_words[CYCLE_OWNER_PIPELINE_SLOTS]{};
    int* error = nullptr;
    cudaStream_t produce{};
    cudaStream_t consume{};
    cudaEvent_t ready[CYCLE_OWNER_PIPELINE_SLOTS]{};
    cudaEvent_t consumed[CYCLE_OWNER_PIPELINE_SLOTS]{};
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
    unsigned long long max_slot_words[CYCLE_OWNER_PIPELINE_SLOTS]{};

    for (int g = 0; g < ngpu; ++g) {
        auto& offsets = batch_offset[static_cast<std::size_t>(g)];
        std::vector<std::uint32_t> packed;
        std::vector<unsigned long long> list_meta(
            static_cast<std::size_t>(batches * class_stride));
        std::vector<unsigned long long> scratch_meta(
            static_cast<std::size_t>(batches * class_stride));
        unsigned long long slot_words[CYCLE_OWNER_PIPELINE_SLOTS] = {1, 1};

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

            const int slot = b & (CYCLE_OWNER_PIPELINE_SLOTS - 1);
            slot_words[slot] = std::max(slot_words[slot], scratch_cursor);
            for (const auto& entry : part) packed.push_back(entry.packed);
        }
        offsets[static_cast<std::size_t>(batches)] = packed.size();

        ck(cudaSetDevice(g), "cycle-owner pipeline set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "cycle-owner pipeline alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                      cudaMemcpyHostToDevice),
           "cycle-owner pipeline copy owner begin");
        ck(cudaMalloc(&c.batch_list,
                      std::max<std::size_t>(1, packed.size()) * sizeof(std::uint32_t)),
           "cycle-owner pipeline alloc batch list");
        if (!packed.empty())
            ck(cudaMemcpy(c.batch_list, packed.data(),
                          packed.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice),
               "cycle-owner pipeline copy batch list");

        const auto& local = lists.local[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.local_list,
                      std::max<std::size_t>(1, local.size()) * sizeof(std::uint32_t)),
           "cycle-owner pipeline alloc local list");
        if (!local.empty())
            ck(cudaMemcpy(c.local_list, local.data(),
                          local.size() * sizeof(std::uint32_t),
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
        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "cycle-owner pipeline alloc peer table");
        ck(cudaMalloc(&c.error, sizeof(int)),
           "cycle-owner pipeline alloc error");
        ck(cudaMemset(c.error, 0, sizeof(int)),
           "cycle-owner pipeline clear error");

        for (int slot = 0; slot < CYCLE_OWNER_PIPELINE_SLOTS; ++slot) {
            max_slot_words[slot] = std::max(max_slot_words[slot], slot_words[slot]);
            ck(cudaMalloc(&c.scratch[slot],
                          slot_words[slot] * sizeof(std::uint32_t)),
               "cycle-owner pipeline alloc scratch slot");
            ck(cudaMalloc(&c.peer_words[slot], sizeof(unsigned long long)),
               "cycle-owner pipeline alloc peer words slot");
        }

        ck(cudaStreamCreateWithFlags(&c.produce, cudaStreamNonBlocking),
           "cycle-owner pipeline create produce stream");
        ck(cudaStreamCreateWithFlags(&c.consume, cudaStreamNonBlocking),
           "cycle-owner pipeline create consume stream");
        for (int slot = 0; slot < CYCLE_OWNER_PIPELINE_SLOTS; ++slot) {
            ck(cudaEventCreateWithFlags(&c.ready[slot], cudaEventDisableTiming),
               "cycle-owner pipeline create ready event");
            ck(cudaEventCreateWithFlags(&c.consumed[slot], cudaEventDisableTiming),
               "cycle-owner pipeline create consumed event");
        }
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

    // All-local cycles are outside the cross-owner cycle batches. Finish them
    // once before starting the A/B overlap pipeline.
    for (int g = 0; g < ngpu; ++g) {
        const auto count = lists.local[static_cast<std::size_t>(g)].size();
        if (!count) continue;
        ck(cudaSetDevice(g), "cycle-owner pipeline local set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const unsigned blocks = static_cast<unsigned>(
            std::max<std::size_t>(1,
                std::min<std::size_t>(requested_blocks, count)));
        persistent_local_cycle_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            c.state, c.local_list, count,
            W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "cycle-owner pipeline local launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner pipeline local sync set device");
        ck(cudaDeviceSynchronize(), "cycle-owner pipeline local sync");
        int error = 0;
        ck(cudaMemcpy(&error, ctx[static_cast<std::size_t>(g)].error,
                      sizeof(error), cudaMemcpyDeviceToHost),
           "cycle-owner pipeline local copy error");
        if (error)
            fail("cycle-owner pipeline local device error=" +
                 std::to_string(error));
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int batch = 0; batch < batches; ++batch) {
        const int slot = batch & (CYCLE_OWNER_PIPELINE_SLOTS - 1);

        // Reusing a parity slot waits only for B[b-2] on that GPU. Otherwise
        // A[b+1] remains free to overlap B[b].
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "cycle-owner pipeline phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            if (batch >= CYCLE_OWNER_PIPELINE_SLOTS)
                ck(cudaStreamWaitEvent(c.produce, c.consumed[slot], 0),
                   "cycle-owner pipeline wait slot consumed");

            ck(cudaMemsetAsync(c.peer_words[slot], 0,
                               sizeof(unsigned long long), c.produce),
               "cycle-owner pipeline zero peer words");

            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            if (count) {
                const unsigned blocks = static_cast<unsigned>(
                    std::max<std::size_t>(1,
                        std::min<std::size_t>(requested_blocks, count)));
                const auto offset =
                    batch_offset[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
                cycle_owner_phase_a_kernel<<<
                    blocks, SCRATCH_FULL_THREADS, 0, c.produce>>>(
                    c.state, c.scratch[slot], c.batch_list + offset, count,
                    c.class_list_begin + batch * class_stride,
                    c.class_scratch_begin + batch * class_stride,
                    batch, batches, W, q, reverse, tile_start, Kwin,
                    ngpu, g, c.owner_begin, c.error);
                ck(cudaGetLastError(), "cycle-owner pipeline phase A launch");
            }
            ck(cudaEventRecord(c.ready[slot], c.produce),
               "cycle-owner pipeline record ready");
        }

        // B[b] keeps the old all-GPU A->B fence, but as stream dependencies
        // instead of host-side device synchronizations.
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "cycle-owner pipeline phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            for (int src = 0; src < ngpu; ++src)
                ck(cudaStreamWaitEvent(
                       c.consume,
                       ctx[static_cast<std::size_t>(src)].ready[slot], 0),
                   "cycle-owner pipeline wait all phase A");

            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            const auto expected_words =
                lists.words[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            if (count) {
                const unsigned blocks = static_cast<unsigned>(
                    std::max<std::size_t>(1,
                        std::min<std::size_t>(requested_blocks, count)));
                const auto offset =
                    batch_offset[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
                cycle_owner_phase_b_kernel<<<
                    blocks, SCRATCH_FULL_THREADS, 0, c.consume>>>(
                    c.peer_state, c.scratch[slot], c.batch_list + offset, count,
                    c.class_list_begin + batch * class_stride,
                    c.class_scratch_begin + batch * class_stride,
                    batch, batches, W, q, reverse, tile_start, Kwin,
                    ngpu, g, c.owner_begin, c.peer_words[slot], c.error);
                ck(cudaGetLastError(), "cycle-owner pipeline phase B launch");
            }
            cycle_owner_pipeline_audit_kernel<<<1, 1, 0, c.consume>>>(
                c.peer_words[slot], expected_words, c.error);
            ck(cudaGetLastError(), "cycle-owner pipeline audit launch");
            ck(cudaEventRecord(c.consumed[slot], c.consume),
               "cycle-owner pipeline record consumed");
        }
    }

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner pipeline final sync set device");
        ck(cudaStreamSynchronize(ctx[static_cast<std::size_t>(g)].consume),
           "cycle-owner pipeline final consume sync");
        ck(cudaStreamSynchronize(ctx[static_cast<std::size_t>(g)].produce),
           "cycle-owner pipeline final produce sync");
    }
    const double runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> output(input.size());
    unsigned long long cross_entries = 0;
    unsigned long long local_entries = 0;
    unsigned long long total_peer_words = 0;
    unsigned long long list_bytes = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner pipeline gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "cycle-owner pipeline copy error");
        if (error)
            fail("cycle-owner pipeline device error=" + std::to_string(error));

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "cycle-owner pipeline gather state");

        local_entries += lists.local[static_cast<std::size_t>(g)].size();
        for (int b = 0; b < batches; ++b) {
            cross_entries +=
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)].size();
            total_peer_words +=
                lists.words[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
        }
        list_bytes +=
            (batch_offset[static_cast<std::size_t>(g)].back() +
             lists.local[static_cast<std::size_t>(g)].size()) *
            sizeof(std::uint32_t);
    }
    if (output != expected)
        fail("cycle-owner descriptorless pipeline redistribution mismatch");

    const unsigned long long class_meta_bytes =
        static_cast<unsigned long long>(ngpu) * batches * class_stride *
        2ULL * sizeof(unsigned long long);
    std::cout << "gridfp-p2p-cycle-owner-descriptorless-pipeline"
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
              << " persistent_list_KiB=" << double(list_bytes) / 1024.0
              << " class_meta_KiB=" << double(class_meta_bytes) / 1024.0
              << " logical_peer_KiB="
              << double(total_peer_words) * sizeof(std::uint32_t) / 1024.0
              << " scratch_even_KiB="
              << double(max_slot_words[0]) * sizeof(std::uint32_t) / 1024.0
              << " scratch_odd_KiB="
              << double(max_slot_words[1]) * sizeof(std::uint32_t) / 1024.0
              << " max_segments_per_owner_cycle=" << CYCLE_OWNER_MAX_SEGMENTS
              << " descriptor_bytes=0"
              << " scratch_allocator_atomics=0"
              << " startup_gpu_support_scan_passes=0"
              << " runtime_support_scan_passes=0"
              << " runtime_count_passes=0"
              << " setup_ms=" << setup_ms
              << " runtime_ms=" << runtime_ms
              << " pipeline_slots=" << CYCLE_OWNER_PIPELINE_SLOTS
              << " host_batch_barriers=0"
              << " cross_gpu_phase_a_fence=1"
              << " cycle_closed_batches=1"
              << " double_scratch=1"
              << " remote_state_reads=0"
              << " native_peer_atomics_required=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner pipeline free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        for (int slot = 0; slot < CYCLE_OWNER_PIPELINE_SLOTS; ++slot) {
            cudaEventDestroy(c.consumed[slot]);
            cudaEventDestroy(c.ready[slot]);
        }
        cudaStreamDestroy(c.consume);
        cudaStreamDestroy(c.produce);
        for (int slot = 0; slot < CYCLE_OWNER_PIPELINE_SLOTS; ++slot) {
            cudaFree(c.peer_words[slot]);
            cudaFree(c.scratch[slot]);
        }
        cudaFree(c.error);
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
        Kwin + S + 2 != W || batches < 2 || batches > PERSISTENT_BATCH_MAX ||
        (batches & (batches - 1)) != 0 || !blocks ||
        ngpu < 2 || ngpu > SCRATCH_FULL_MAX_GPU) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible),
       "cycle-owner descriptorless pipeline device count");
    if (visible < ngpu) return 3;

    run_cycle_owner_pipeline(W, Kwin, S, false, ngpu, batches, blocks);
    run_cycle_owner_pipeline(W, Kwin, S, true, ngpu, batches, blocks);
    std::cout << "ALL_OK gridfp_p2p_cycle_owner_descriptorless_pipeline=1\n";
    return 0;
}
