#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
BASE_SRC="$(repo_path src/cuda/gridfp/gridfp_reduced_production_p2p_host_persistent_pipeline_microprobe.cu)"
SRC_DIR="$(dirname "$BASE_SRC")"
GEN_SRC="$(build_path gridfp_reduced_production_p2p_host_persistent_event_pipeline_generated.cu)"
OUT="$(build_path "${OUT:-gridfp_reduced_component_p2p-host-persistent-event-pipeline}")"

python3 - "$BASE_SRC" "$GEN_SRC" <<'PY'
from pathlib import Path
import sys

src_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
s = src_path.read_text()

setup_marker = "    const double setup_ms = std::chrono::duration<double, std::milli>(\n"
if s.count(setup_marker) != 1:
    raise SystemExit("event pipeline generator: setup marker mismatch")
event_setup = r'''    // Persistent cross-device events are setup metadata, not per-row work.
    std::vector<std::vector<cudaEvent_t>> a_done(
        static_cast<std::size_t>(ngpu),
        std::vector<cudaEvent_t>(static_cast<std::size_t>(batches), nullptr));
    std::vector<cudaEvent_t> barrier_done(
        static_cast<std::size_t>(batches), nullptr);
    cudaStream_t barrier_stream = nullptr;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "event pipeline event set device");
        for (int b = 0; b < batches; ++b)
            ck(cudaEventCreateWithFlags(
                   &a_done[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)],
                   cudaEventDisableTiming),
               "event pipeline create A event");
    }
    ck(cudaSetDevice(0), "event pipeline barrier device");
    ck(cudaStreamCreateWithFlags(&barrier_stream, cudaStreamNonBlocking),
       "event pipeline create barrier stream");
    for (int b = 0; b < batches; ++b)
        ck(cudaEventCreateWithFlags(
               &barrier_done[static_cast<std::size_t>(b)], cudaEventDisableTiming),
           "event pipeline create barrier event");

'''
s = s.replace(setup_marker, event_setup + setup_marker)

start_marker = "    // Batch b uses plane b&1."
end_marker = '''    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline final sync set device");'''
start = s.find(start_marker)
end = s.find(end_marker, start)
if start < 0 or end < 0:
    raise SystemExit("event pipeline generator: scheduler markers missing")

scheduler = r'''    // Event-driven global A barrier.  Each GPU records A_done after its
    // local Phase A.  GPU0's dedicated barrier stream fans those events in,
    // records one barrier_done event, and every Phase B stream waits on that
    // single event.  The host never synchronizes between batches.
    for (int batch = 0; batch < batches; ++batch) {
        const int plane = batch & 1;
        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            ck(cudaSetDevice(g), "event pipeline phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            if (count) {
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
                ck(cudaGetLastError(), "event pipeline phase A launch");
            }
            ck(cudaEventRecord(
                   a_done[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)],
                   c.stream[plane]),
               "event pipeline record A done");
        }

        ck(cudaSetDevice(0), "event pipeline fanin device");
        for (int g = 0; g < ngpu; ++g)
            ck(cudaStreamWaitEvent(
                   barrier_stream,
                   a_done[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)],
                   0),
               "event pipeline fanin wait");
        ck(cudaEventRecord(
               barrier_done[static_cast<std::size_t>(batch)], barrier_stream),
           "event pipeline record global A barrier");

        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            ck(cudaSetDevice(g), "event pipeline phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            ck(cudaStreamWaitEvent(
                   c.stream[plane],
                   barrier_done[static_cast<std::size_t>(batch)], 0),
               "event pipeline fanout wait");
            if (!count) continue;
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
            ck(cudaGetLastError(), "event pipeline phase B launch");
        }
    }

'''
s = s[:start] + scheduler + s[end:]

free_marker = '''    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "pipeline free set device");'''
if s.count(free_marker) != 1:
    raise SystemExit("event pipeline generator: cleanup marker mismatch")
event_cleanup = r'''    ck(cudaSetDevice(0), "event pipeline cleanup barrier device");
    for (cudaEvent_t ev : barrier_done)
        if (ev) cudaEventDestroy(ev);
    if (barrier_stream) cudaStreamDestroy(barrier_stream);
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "event pipeline cleanup A event device");
        for (cudaEvent_t ev : a_done[static_cast<std::size_t>(g)])
            if (ev) cudaEventDestroy(ev);
    }

'''
s = s.replace(free_marker, event_cleanup + free_marker)

old_tag = 'std::cout << "gridfp-p2p-host-persistent-pipeline"'
new_tag = 'std::cout << "gridfp-p2p-host-persistent-event-pipeline"'
if s.count(old_tag) != 1:
    raise SystemExit("event pipeline generator: output tag mismatch")
s = s.replace(old_tag, new_tag)

flag = '              << " cycle_closed_batches=1"\n'
# The Python string above uses \n as an actual newline.  Keep this assertion so
# upstream formatting changes fail loudly instead of silently generating the
# host-barrier executor.
if s.count(flag) != 1:
    raise SystemExit("event pipeline generator: output flag mismatch")
s = s.replace(
    flag,
    flag +
    '              << " host_batch_barriers=0"\n'
    '              << " cross_device_events=1"\n'
    '              << " event_barrier_fanin=1"\n')

old_ok = 'std::cout << "ALL_OK gridfp_p2p_host_persistent_pipeline=1\\n";'
new_ok = 'std::cout << "ALL_OK gridfp_p2p_host_persistent_event_pipeline=1\\n";'
# old_ok/new_ok contain one literal C++ backslash before n.
if s.count(old_ok) != 1:
    raise SystemExit("event pipeline generator: ALL_OK marker mismatch")
s = s.replace(old_ok, new_ok)

# Generation-time audit: batch-local host synchronization must be gone, while
# the one-time local-cycle drain and final two-stream drain remain in place.
body = s[start:s.find(end_marker, start)]
if 'cudaStreamSynchronize' in body:
    raise SystemExit("event pipeline generator: host batch synchronize survived")
if 'cudaStreamWaitEvent' not in body or 'barrier_done' not in body:
    raise SystemExit("event pipeline generator: event barrier missing")

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(s)
PY

PTXAS_FLAGS=()
if [[ "$PTXAS_VERBOSE" == 1 ]]; then PTXAS_FLAGS+=("-Xptxas=-v"); fi
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -I"$SRC_DIR" "${PTXAS_FLAGS[@]}" "$GEN_SRC" -o "$OUT"
echo "built $OUT (persistent_event_pipeline=1 arch=$ARCH generated=$GEN_SRC)"
