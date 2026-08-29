#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
MAX_W="${MAX_W:-10}"
NGPU_MODEL="${NGPU_MODEL:-8}"
BATCHES="${BATCHES:-8}"

command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
command -v bash >/dev/null || exit 2

sh=(
  scripts/build/gridfp-reduced-p2p-schedule-probe.sh
  scripts/run/gridfp-reduced-p2p-schedule-probes.sh
  scripts/run/gridfp-p2p-select-segment-major-batches.sh
)
for f in "${sh[@]}"; do
  [[ -f "$ONEESAN_ROOT/$f" ]] || { echo "missing $f" >&2; exit 3; }
  bash -n "$ONEESAN_ROOT/$f"
done

build_and_run() {
  local name="$1" src="$2"; shift 2
  local out="$(build_path "$name")"
  "$CXX" -O2 -std=c++17 "$(repo_path "$src")" -o "$out"
  "$out" "$@"
}

build_and_run gridfp_p2p_modal_probe \
  src/cpp/probes/gridfp_reduced_production_p2p_modal_probe.cpp \
  "$MAX_W" "$NGPU_MODEL"
build_and_run gridfp_p2p_worklist_probe \
  src/cpp/probes/gridfp_reduced_production_p2p_worklist_probe.cpp \
  "$MAX_W" "$NGPU_MODEL"
build_and_run gridfp_p2p_compiled_probe \
  src/cpp/probes/gridfp_reduced_production_p2p_compiled_schedule_probe.cpp \
  "$MAX_W" "$NGPU_MODEL"
build_and_run gridfp_p2p_packed_probe \
  src/cpp/probes/gridfp_reduced_production_p2p_packed_schedule_probe.cpp \
  "$MAX_W" "$NGPU_MODEL"
build_and_run gridfp_p2p_compact_segment_probe \
  src/cpp/probes/gridfp_reduced_production_p2p_compact_segment_probe.cpp \
  "$MAX_W" "$NGPU_MODEL" "$BATCHES"
build_and_run gridfp_p2p_segment_major_probe \
  src/cpp/probes/gridfp_reduced_production_p2p_segment_major_probe.cpp \
  "$MAX_W" "$NGPU_MODEL" "$BATCHES"

echo "gridfp-p2p-segment-major-source-preflight OK cpu_proofs=6 shell_syntax=3 nvcc=not_required actions=not_used"
