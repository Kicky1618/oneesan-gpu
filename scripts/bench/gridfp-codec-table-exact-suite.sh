#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; NGPU="${NGPU:-2}"; BLOCKS="${BLOCKS:-256}"
REPEATS="${REPEATS:-7}"; WARMUP="${WARMUP:-1}"; ARCH="${ARCH:-native}"; PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
RUN_CHOOSE="${RUN_CHOOSE:-1}"; RUN_PRIMITIVE="${RUN_PRIMITIVE:-1}"; RUN_COMBINED="${RUN_COMBINED:-1}"
for name in RUN_CHOOSE RUN_PRIMITIVE RUN_COMBINED; do v="${!name}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$name must be 0 or 1" >&2; exit 2; }; done
if [[ "$RUN_CHOOSE" == 0 && "$RUN_PRIMITIVE" == 0 && "$RUN_COMBINED" == 0 ]]; then echo "nothing selected" >&2; exit 2; fi

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/gridfp_codec_table_exact_suite_w${W}_g${NGPU}_b${BLOCKS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.txt}"
mkdir -p "$LOGDIR" "$(dirname "$SUMMARY")"; : >"$SUMMARY"

bash "$ONEESAN_ROOT/scripts/bench/gridfp-runtime-ab-env-proof.sh" >"$LOGDIR/runtime-ab-env-proof.out" 2>"$LOGDIR/runtime-ab-env-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-choose-sym-u32-table-proof.sh" >"$LOGDIR/choose-proof.out" 2>"$LOGDIR/choose-proof.err"
bash "$ONEESAN_ROOT/scripts/bench/gridfp-primitive-sym-u32-table-proof.sh" >"$LOGDIR/primitive-proof.out" 2>"$LOGDIR/primitive-proof.err"

run_case() {
  local name="$1" script="$2"
  local out="$LOGDIR/${name}.out" err="$LOGDIR/${name}.err"
  echo "=== codec exact $name ===" >&2
  env W="$W" NGPU="$NGPU" BLOCKS="$BLOCKS" REPEATS="$REPEATS" WARMUP="$WARMUP" ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
    bash "$ONEESAN_ROOT/scripts/bench/$script" >"$out" 2>"$err"
  {
    echo "[$name]"
    grep -E '(wall_speedup|wall_delta_pct|candidate_bytes|_bytes=|exact=)' "$out" || true
    grep -E ' OK ' "$err" | tail -n1 || true
    echo
  } >>"$SUMMARY"
  cat "$out"
}

[[ "$RUN_CHOOSE" == 1 ]] && run_case choose gridfp-reduced-runtime-choose-sym-u32-ab.sh
[[ "$RUN_PRIMITIVE" == 1 ]] && run_case primitive gridfp-reduced-runtime-primitive-sym-u32-ab.sh
[[ "$RUN_COMBINED" == 1 ]] && run_case combined gridfp-reduced-runtime-codec-tables-sym-u32-ab.sh

echo "--- codec table exact suite summary ---"
cat "$SUMMARY"
echo "gridfp-codec-table-exact-suite OK W=$W ngpu=$NGPU blocks=$BLOCKS summary=$SUMMARY" >&2
