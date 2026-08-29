#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
MAX_W="${MAX_W:-12}"
NGPU_MODEL="${NGPU_MODEL:-8}"
BATCHES="${BATCHES:-8}"
RUN_CUDA="${RUN_CUDA:-0}"
CUDA_W="${CUDA_W:-10}"
CUDA_NGPU="${CUDA_NGPU:-2}"
CUDA_BLOCKS="${CUDA_BLOCKS:-256}"
CUDA_BATCHES="${CUDA_BATCHES:-4}"
CUDA_SALT="${CUDA_SALT:-0}"
ARCH="${ARCH:-native}"

build_cpu_probe() {
  local name="$1" src_rel="$2"
  local out
  out="$(build_path "$name")"
  "$CXX" -O2 -std=c++17 "$(repo_path "$src_rel")" -o "$out"
  printf '%s\n' "$out"
}

tie_bin="$(build_cpu_probe \
  gridfp_reduced_p2p_tie_balance_probe \
  src/cpp/probes/gridfp_reduced_production_p2p_tie_balance_probe.cpp)"
worklist_bin="$(build_cpu_probe \
  gridfp_reduced_p2p_worklist_probe \
  src/cpp/probes/gridfp_reduced_production_p2p_worklist_probe.cpp)"
compiled_bin="$(build_cpu_probe \
  gridfp_reduced_p2p_compiled_schedule_probe \
  src/cpp/probes/gridfp_reduced_production_p2p_compiled_schedule_probe.cpp)"
packed_bin="$(build_cpu_probe \
  gridfp_reduced_p2p_packed_schedule_probe \
  src/cpp/probes/gridfp_reduced_production_p2p_packed_schedule_probe.cpp)"
compact_segment_bin="$(build_cpu_probe \
  gridfp_reduced_p2p_compact_segment_probe \
  src/cpp/probes/gridfp_reduced_production_p2p_compact_segment_probe.cpp)"
segment_major_bin="$(build_cpu_probe \
  gridfp_reduced_p2p_segment_major_probe \
  src/cpp/probes/gridfp_reduced_production_p2p_segment_major_probe.cpp)"

"$tie_bin" "$MAX_W" "$NGPU_MODEL"
"$worklist_bin" "$MAX_W" "$NGPU_MODEL"
"$compiled_bin" "$MAX_W" "$NGPU_MODEL"
"$packed_bin" "$MAX_W" "$NGPU_MODEL"
"$compact_segment_bin" "$MAX_W" "$NGPU_MODEL" "$BATCHES"
"$segment_major_bin" "$MAX_W" "$NGPU_MODEL" "$BATCHES"

if [[ "$RUN_CUDA" == 1 ]]; then
  K=$(((CUDA_W - 2) / 2))

  MODE=tie ARCH="$ARCH" \
    "$(repo_path scripts/build/gridfp-reduced-p2p-schedule-probe.sh)"
  "$(build_path gridfp_reduced_p2p_tie)" "$CUDA_W" "$NGPU_MODEL" "$CUDA_BLOCKS"

  MODE=ownerfirst ARCH="$ARCH" \
    "$(repo_path scripts/build/gridfp-reduced-p2p-schedule-probe.sh)"
  "$(build_path gridfp_reduced_p2p_ownerfirst)" "$CUDA_W" "$K" "$K" "$CUDA_BLOCKS" "$CUDA_NGPU"

  MODE=worklist ARCH="$ARCH" \
    "$(repo_path scripts/build/gridfp-reduced-p2p-schedule-probe.sh)"
  "$(build_path gridfp_reduced_p2p_worklist)" "$CUDA_W" "$K" "$CUDA_BLOCKS" "$CUDA_NGPU"

  MODE=compiled ARCH="$ARCH" \
    "$(repo_path scripts/build/gridfp-reduced-p2p-schedule-probe.sh)"
  "$(build_path gridfp_reduced_p2p_compiled)" "$CUDA_W" "$K" "$CUDA_BLOCKS" "$CUDA_NGPU"

  MODE=packed ARCH="$ARCH" \
    "$(repo_path scripts/build/gridfp-reduced-p2p-schedule-probe.sh)"
  "$(build_path gridfp_reduced_p2p_packed)" "$CUDA_W" "$K" "$CUDA_BLOCKS" "$CUDA_NGPU"

  MODE=segmented ARCH="$ARCH" \
    "$(repo_path scripts/build/gridfp-reduced-p2p-schedule-probe.sh)"
  "$(build_path gridfp_reduced_p2p_segmented)" "$CUDA_W" "$K" "$CUDA_BATCHES" "$CUDA_BLOCKS" "$CUDA_NGPU"

  MODE=segment-major ARCH="$ARCH" \
    "$(repo_path scripts/build/gridfp-reduced-p2p-schedule-probe.sh)"
  "$(build_path gridfp_reduced_p2p_segment-major)" "$CUDA_W" "$K" "$CUDA_BATCHES" "$CUDA_BLOCKS" "$CUDA_NGPU"

  MODE=segment-major-count ARCH="$ARCH" \
    "$(repo_path scripts/build/gridfp-reduced-p2p-schedule-probe.sh)"
  "$(build_path gridfp_reduced_p2p_segment-major-count)" "$CUDA_W" "$K" "$CUDA_BATCHES" "$CUDA_BLOCKS" "$CUDA_NGPU"

  MODE=segment-major-fill ARCH="$ARCH" \
    "$(repo_path scripts/build/gridfp-reduced-p2p-schedule-probe.sh)"
  "$(build_path gridfp_reduced_p2p_segment-major-fill)" "$CUDA_W" "$K" "$CUDA_BATCHES" "$CUDA_BLOCKS" "$CUDA_NGPU"

  MODE=two-row-segment-major ARCH="$ARCH" \
    "$(repo_path scripts/build/gridfp-reduced-p2p-schedule-probe.sh)"
  "$(build_path gridfp_reduced_p2p_two-row-segment-major)" "$CUDA_W" "$CUDA_NGPU" "$CUDA_BATCHES" "$CUDA_BLOCKS"

  MODE=two-row-segment-major-built ARCH="$ARCH" \
    "$(repo_path scripts/build/gridfp-reduced-p2p-schedule-probe.sh)"
  "$(build_path gridfp_reduced_p2p_two-row-segment-major-built)" \
    "$CUDA_W" "$CUDA_NGPU" "$CUDA_BATCHES" "$CUDA_BLOCKS" 4294967291 "$CUDA_SALT"
fi

echo "ALL_OK local_p2p_schedule_regressions=1"
