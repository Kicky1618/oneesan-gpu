#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BLOCKS="${BLOCKS:-4096}"; THREADS="${THREADS:-256}"; ITERS="${ITERS:-512}"
REPEATS="${REPEATS:-7}"; WARMUP="${WARMUP:-2}"; ARCH="${ARCH:-native}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
RUN_PRIMITIVE1="${RUN_PRIMITIVE1:-1}"; RUN_PRIMITIVE="${RUN_PRIMITIVE:-1}"; RUN_CHOOSE="${RUN_CHOOSE:-1}"; RUN_MOTZKIN="${RUN_MOTZKIN:-1}"
for name in RUN_PRIMITIVE1 RUN_PRIMITIVE RUN_CHOOSE RUN_MOTZKIN; do v="${!name}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$name must be 0 or 1" >&2; exit 2; }; done
if [[ "$RUN_PRIMITIVE1" == 0 && "$RUN_PRIMITIVE" == 0 && "$RUN_CHOOSE" == 0 && "$RUN_MOTZKIN" == 0 ]]; then echo "nothing selected" >&2; exit 2; fi
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

echo "--- codec table layout suite summary ---"
cat "$SUMMARY"
echo "gridfp-codec-table-layout-suite OK summary=$SUMMARY" >&2
