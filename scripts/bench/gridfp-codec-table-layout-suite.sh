#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BLOCKS="${BLOCKS:-4096}"; THREADS="${THREADS:-256}"; ITERS="${ITERS:-512}"
REPEATS="${REPEATS:-7}"; WARMUP="${WARMUP:-2}"; ARCH="${ARCH:-native}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
W28_BLOCKS="${W28_BLOCKS:-256}"; W28_THREADS="${W28_THREADS:-256}"; W28_ITERS="${W28_ITERS:-16}"
W28_WARMUP="${W28_WARMUP:-1}"; W28_OWNERS="${W28_OWNERS:-0}"; W28_DIRECTIONS="${W28_DIRECTIONS:-0 1}"
RUN_PRIMITIVE1="${RUN_PRIMITIVE1:-1}"; RUN_PRIMITIVE="${RUN_PRIMITIVE:-1}"; RUN_CHOOSE="${RUN_CHOOSE:-1}"; RUN_MOTZKIN="${RUN_MOTZKIN:-1}"; RUN_W28_RANK="${RUN_W28_RANK:-1}"
for name in RUN_PRIMITIVE1 RUN_PRIMITIVE RUN_CHOOSE RUN_MOTZKIN RUN_W28_RANK; do v="${!name}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$name must be 0 or 1" >&2; exit 2; }; done
if [[ "$RUN_PRIMITIVE1" == 0 && "$RUN_PRIMITIVE" == 0 && "$RUN_CHOOSE" == 0 && "$RUN_MOTZKIN" == 0 && "$RUN_W28_RANK" == 0 ]]; then echo "nothing selected" >&2; exit 2; fi
if ! command -v nvcc >/dev/null || ! command -v nvidia-smi >/dev/null; then echo "nvcc and nvidia-smi are required" >&2; exit 2; fi

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_codec_table_layout_suite}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.txt}"
mkdir -p "$LOGDIR" "$(dirname "$SUMMARY")"; : >"$SUMMARY"

bash "$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-budget-proof.sh" >"$LOGDIR/budget-proof.out" 2>"$LOGDIR/budget-proof.err"
cat "$LOGDIR/budget-proof.out" | tee -a "$SUMMARY"

run_case() {
  local name="$1" script="$2" out="$LOGDIR/${1}.out" err="$LOGDIR/${1}.err"
  echo "=== $name ===" >&2
  env BLOCKS="$BLOCKS" THREADS="$THREADS" ITERS="$ITERS" REPEATS="$REPEATS" WARMUP="$WARMUP" ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
    bash "$ONEESAN_ROOT/scripts/bench/$script" >"$out" 2>"$err"
  {
    echo "[$name]"
    grep -E '(speedup|delta_pct|old_bytes|new_bytes|saved_bytes|exact=)' "$out" || true
    grep -E ' OK ' "$err" | tail -n1 || true
    echo
  } >>"$SUMMARY"
  cat "$out"
}

[[ "$RUN_PRIMITIVE1" == 1 ]] && run_case primitive1 gridfp-primitive1-u32-table-microprobe.sh
[[ "$RUN_PRIMITIVE" == 1 ]] && run_case primitive gridfp-primitive-sym-u32-table-microprobe.sh
[[ "$RUN_CHOOSE" == 1 ]] && run_case choose gridfp-choose-sym-u32-table-microprobe.sh
[[ "$RUN_MOTZKIN" == 1 ]] && run_case motzkin gridfp-motzkin-tri-u64-table-microprobe.sh
if [[ "$RUN_W28_RANK" == 1 ]]; then
  out="$LOGDIR/w28_rank.out"; err="$LOGDIR/w28_rank.err"
  echo "=== w28_rank ===" >&2
  env BLOCKS="$W28_BLOCKS" THREADS="$W28_THREADS" ITERS="$W28_ITERS" \
      REPEATS="$REPEATS" WARMUP="$W28_WARMUP" OWNERS="$W28_OWNERS" DIRECTIONS="$W28_DIRECTIONS" \
      ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
      bash "$ONEESAN_ROOT/scripts/bench/gridfp-codec-table-w28-rank-microprobe.sh" >"$out" 2>"$err"
  {
    echo "[w28_rank]"
    grep -E '(paired_speedup|winner_|checksum_exact|summary=)' "$out" || true
    grep -E ' OK ' "$err" | tail -n1 || true
    echo
  } >>"$SUMMARY"
  cat "$out"
fi

echo "--- codec table layout suite summary ---"
cat "$SUMMARY"
echo "gridfp-codec-table-layout-suite OK summary=$SUMMARY" >&2
