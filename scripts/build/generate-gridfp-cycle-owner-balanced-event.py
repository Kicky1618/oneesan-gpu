#!/usr/bin/env python3
from pathlib import Path
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: generate-gridfp-cycle-owner-balanced-event.py BALANCED_PIPELINE OUT", file=sys.stderr)
        return 2
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    s = src.read_text()

    setup_marker = "    const double setup_ms = std::chrono::duration<double, std::milli>(\n"
    if s.count(setup_marker) != 1:
        raise SystemExit("balanced event generator: setup marker mismatch")
    setup = r'''    std::vector<std::vector<cudaEvent_t>> a_done(
        static_cast<std::size_t>(ngpu),
        std::vector<cudaEvent_t>(static_cast<std::size_t>(batches), nullptr));
    std::vector<cudaEvent_t> barrier_done(
        static_cast<std::size_t>(batches), nullptr);
    cudaStream_t barrier_stream = nullptr;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "balanced event create A event device");
        for (int b = 0; b < batches; ++b)
            ck(cudaEventCreateWithFlags(
                   &a_done[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)],
                   cudaEventDisableTiming),
               "balanced event create A event");
    }
    ck(cudaSetDevice(0), "balanced event barrier setup device");
    ck(cudaStreamCreateWithFlags(&barrier_stream, cudaStreamNonBlocking),
       "balanced event create barrier stream");
    for (int b = 0; b < batches; ++b)
        ck(cudaEventCreateWithFlags(
               &barrier_done[static_cast<std::size_t>(b)], cudaEventDisableTiming),
           "balanced event create barrier event");

'''
    s = s.replace(setup_marker, setup + setup_marker)

    local_tail = '''            "cycle-owner pipeline local sync");
    }

'''
    start = s.find(local_tail)
    if start < 0:
        raise SystemExit("balanced event generator: local sync marker missing")
    start += len(local_tail)
    start_marker = "    for (int batch = 0; batch < batches; ++batch) {"
    if not s.startswith(start_marker, start):
        raise SystemExit("balanced event generator: runtime batch marker mismatch")
    end_marker = '''    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner pipeline final sync set device");'''
    end = s.find(end_marker, start)
    if end < 0:
        raise SystemExit("balanced event generator: final sync marker missing")

    scheduler = r'''    // Balanced canonical-leader batches are complete disjoint support
    // cycles.  The staged schedule waits only for the previous global A
    // barrier before A_b, so B_{b-1} can overlap A_b without allowing adjacent
    // A batches to contend for HBM.  No host synchronization occurs per batch.
    for (int batch = 0; batch < batches; ++batch) {
        const int plane = batch & 1;
        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            ck(cudaSetDevice(g), "balanced event phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            if (batch > 0)
                ck(cudaStreamWaitEvent(
                       c.stream[plane],
                       barrier_done[static_cast<std::size_t>(batch - 1)], 0),
                   "balanced event staged A wait");
            if (count) {
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
                ck(cudaGetLastError(), "balanced event phase A launch");
            }
            ck(cudaEventRecord(
                   a_done[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)],
                   c.stream[plane]),
               "balanced event record A done");
        }

        ck(cudaSetDevice(0), "balanced event fanin device");
        for (int g = 0; g < ngpu; ++g)
            ck(cudaStreamWaitEvent(
                   barrier_stream,
                   a_done[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)], 0),
               "balanced event fanin wait");
        ck(cudaEventRecord(
               barrier_done[static_cast<std::size_t>(batch)], barrier_stream),
           "balanced event record A barrier");

        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            ck(cudaSetDevice(g), "balanced event phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            ck(cudaStreamWaitEvent(
                   c.stream[plane],
                   barrier_done[static_cast<std::size_t>(batch)], 0),
               "balanced event fanout wait");
            if (!count) continue;
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
            ck(cudaGetLastError(), "balanced event phase B launch");
        }
    }

'''
    s = s[:start] + scheduler + s[end:]

    cleanup_marker = '''    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "cycle-owner pipeline free set device");'''
    if s.count(cleanup_marker) != 1:
        raise SystemExit("balanced event generator: cleanup marker mismatch")
    cleanup = r'''    ck(cudaSetDevice(0), "balanced event cleanup barrier device");
    for (cudaEvent_t ev : barrier_done)
        if (ev) cudaEventDestroy(ev);
    if (barrier_stream) cudaStreamDestroy(barrier_stream);
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "balanced event cleanup A event device");
        for (cudaEvent_t ev : a_done[static_cast<std::size_t>(g)])
            if (ev) cudaEventDestroy(ev);
    }

'''
    s = s.replace(cleanup_marker, cleanup + cleanup_marker)

    old_tag = 'std::cout << "gridfp-p2p-cycle-owner-balanced-pipeline"'
    new_tag = 'std::cout << "gridfp-p2p-cycle-owner-balanced-event-pipeline"'
    if s.count(old_tag) != 1:
        raise SystemExit("balanced event generator: output tag mismatch")
    s = s.replace(old_tag, new_tag)

    flag = '              << " batch_hash_shift=12"\n'
    if s.count(flag) != 1:
        raise SystemExit("balanced event generator: flag marker mismatch")
    s = s.replace(
        flag,
        flag +
        '              << " host_batch_barriers=0"\n'
        '              << " cross_device_events=1"\n'
        '              << " event_barrier_fanin=1"\n'
        '              << " event_schedule=staged"\n'
        '              << " adjacent_A_overlap=0"\n')

    old_ok = 'std::cout << "ALL_OK gridfp_p2p_cycle_owner_balanced_pipeline=1\\n";'
    new_ok = 'std::cout << "ALL_OK gridfp_p2p_cycle_owner_balanced_event_pipeline=1\\n";'
    if s.count(old_ok) != 1:
        raise SystemExit("balanced event generator: ALL_OK marker mismatch")
    s = s.replace(old_ok, new_ok)

    body = s[start:s.find(end_marker, start)]
    if 'cudaStreamSynchronize' in body:
        raise SystemExit("balanced event generator: host batch sync survived")
    if 'balanced event staged A wait' not in body:
        raise SystemExit("balanced event generator: staged dependency missing")
    if 'cudaStreamWaitEvent' not in body:
        raise SystemExit("balanced event generator: event waits missing")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
