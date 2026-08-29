#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'B300 grand first-pass currently targets n=27' >&2; exit 2; }

PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
ARCH="${ARCH:-native}"
MAX_WINDOW="${MAX_WINDOW:-14}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
RUN_NEXTSELF_STAGE="${RUN_NEXTSELF_STAGE:-1}"
RUN_HYBRID_STAGE="${RUN_HYBRID_STAGE:-1}"
NEXTSELF_THREADS="${NEXTSELF_THREADS:-256}"
NEXTSELF_SEARCH_ROWS="${NEXTSELF_SEARCH_ROWS:-1}"
NEXTSELF_VALIDATE_ROWS="${NEXTSELF_VALIDATE_ROWS:-4 8}"
NEXTSELF_SEARCH_REPEATS="${NEXTSELF_SEARCH_REPEATS:-1}"
NEXTSELF_VALIDATE_REPEATS="${NEXTSELF_VALIDATE_REPEATS:-1}"
NEXTSELF_MIN_SPEEDUP="${NEXTSELF_MIN_SPEEDUP:-1.01}"
HYBRID_MIN_SPEEDUP="${HYBRID_MIN_SPEEDUP:-1.01}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"
LOG="${LOG:-${PREFIX}.log}"
META="${META:-${PREFIX}.meta}"

for x in REBUILD_BUCKETS RUN_NEXTSELF_STAGE RUN_HYBRID_STAGE; do
  v="${!x}"
  [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
[[ "$MAX_WINDOW" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_WINDOW must be positive integer' >&2; exit 2; }
[[ "$NEXTSELF_THREADS" =~ ^[0-9]+$ ]] && ((NEXTSELF_THREADS>=32 && NEXTSELF_THREADS<=768 && NEXTSELF_THREADS%32==0)) || {
  echo 'NEXTSELF_THREADS must be warp multiple 32..768' >&2; exit 2;
}
for x in NEXTSELF_SEARCH_REPEATS NEXTSELF_VALIDATE_REPEATS; do
  v="${!x}"; [[ "$v" =~ ^[1-9][0-9]*$ ]] || { echo "$x must be >=1" >&2; exit 2; }
done
[[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v git >/dev/null || { echo 'git required' >&2; exit 2; }
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
(( GPU_COUNT >= 8 )) || { echo "need 8 visible GPUs; got $GPU_COUNT" >&2; exit 2; }

mkdir -p "$(dirname "$LOG")" "$(dirname "$META")"
HEAD_SHA="$(git -C "$ONEESAN_ROOT" rev-parse HEAD)"
HEAD_DIRTY=0
[[ -z "$(git -C "$ONEESAN_ROOT" status --porcelain=v1 --untracked-files=normal)" ]] || HEAD_DIRTY=1
PROFILE_SHA="$(sha256sum "$PROFILE_FILE" | awk '{print $1}')"

{
  printf 'schema=1\n'
  printf 'n=%s\n' "$N"
  printf 'head_sha=%s\n' "$HEAD_SHA"
  printf 'head_dirty=%s\n' "$HEAD_DIRTY"
  printf 'profile_file=%s\n' "$PROFILE_FILE"
  printf 'profile_sha256=%s\n' "$PROFILE_SHA"
  printf 'gpu_count=%s\n' "$GPU_COUNT"
  printf 'arch=%s\n' "$ARCH"
  printf 'max_window=%s\n' "$MAX_WINDOW"
  printf 'select_only=1\n'
  printf 'rebuild_buckets=%s\n' "$REBUILD_BUCKETS"
  printf 'run_nextself_stage=%s\n' "$RUN_NEXTSELF_STAGE"
  printf 'run_hybrid_stage=%s\n' "$RUN_HYBRID_STAGE"
  printf 'nextself_threads=%s\n' "$NEXTSELF_THREADS"
  printf 'nextself_search_rows=%s\n' "$NEXTSELF_SEARCH_ROWS"
  printf 'nextself_validate_rows=%s\n' "$NEXTSELF_VALIDATE_ROWS"
  printf 'nextself_search_repeats=%s\n' "$NEXTSELF_SEARCH_REPEATS"
  printf 'nextself_validate_repeats=%s\n' "$NEXTSELF_VALIDATE_REPEATS"
  printf 'nextself_min_speedup=%s\n' "$NEXTSELF_MIN_SPEEDUP"
  printf 'hybrid_min_speedup=%s\n' "$HYBRID_MIN_SPEEDUP"
  printf 'hybrid8_nextself_transform_preflight=1\n'
  printf 'nvcc_version_begin=1\n'
  nvcc --version | sed 's/^/nvcc: /'
  printf 'nvcc_version_end=1\n'
  printf 'gpu_inventory_begin=1\n'
  nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv,noheader | sed 's/^/gpu: /'
  printf 'gpu_inventory_end=1\n'
} >"$META"

if (( HEAD_DIRTY )); then
  echo 'WARNING: repository has uncommitted or untracked changes; provenance records head_dirty=1' >&2
fi

echo '=== B300 grand first-pass: GPU-free preflight ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid8-nextself-transform-preflight.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-joint-nextgen-hybrid8-preflight.sh"
if [[ -f "$ONEESAN_ROOT/scripts/bench/b300-nextgen-grand-selector-preflight.sh" ]]; then
  bash "$ONEESAN_ROOT/scripts/bench/b300-nextgen-grand-selector-preflight.sh"
fi

echo "=== B300 grand first-pass: n=27 head=${HEAD_SHA:0:12} GPUs=$GPU_COUNT SELECT_ONLY=1 ===" >&2
set +e
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" MAX_WINDOW="$MAX_WINDOW" \
  SELECT_ONLY=1 REBUILD_BUCKETS="$REBUILD_BUCKETS" \
  RUN_NEXTSELF_STAGE="$RUN_NEXTSELF_STAGE" RUN_HYBRID_STAGE="$RUN_HYBRID_STAGE" \
  NEXTSELF_THREADS="$NEXTSELF_THREADS" NEXTSELF_SEARCH_ROWS="$NEXTSELF_SEARCH_ROWS" \
  NEXTSELF_VALIDATE_ROWS="$NEXTSELF_VALIDATE_ROWS" NEXTSELF_SEARCH_REPEATS="$NEXTSELF_SEARCH_REPEATS" \
  NEXTSELF_VALIDATE_REPEATS="$NEXTSELF_VALIDATE_REPEATS" NEXTSELF_MIN_SPEEDUP="$NEXTSELF_MIN_SPEEDUP" \
  HYBRID_MIN_SPEEDUP="$HYBRID_MIN_SPEEDUP" PREFIX="$PREFIX" \
  bash "$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh" 27 "$@" \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

printf 'exit_code=%s\n' "$rc" >>"$META"
if (( rc != 0 )); then
  echo "b300x8-grand-firstpass FAILED rc=$rc log=$LOG meta=$META" >&2
  exit "$rc"
fi

grep -Fq 'SINGLE PASS SELECTED' "$LOG" || {
  echo "grand first-pass completed without SINGLE PASS SELECTED marker: $LOG" >&2
  exit 4
}
grep -Fq 'SELECT_ONLY=1: selected' "$LOG" || {
  echo "grand first-pass did not stop at SELECT_ONLY boundary: $LOG" >&2
  exit 4
}

SELECTED_LINE="$(grep -F 'SINGLE PASS SELECTED' "$LOG" | tail -n1)"
printf 'selected_line=%s\n' "$SELECTED_LINE" >>"$META"
echo "b300x8-grand-firstpass OK log=$LOG meta=$META $SELECTED_LINE" >&2
