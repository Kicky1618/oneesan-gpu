#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; ARCH="${ARCH:-native}"; TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"; PM_ACCUM="${PM_ACCUM:-0}"; TERNARY_KEY4="${TERNARY_KEY4:-1}"
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then echo "TERNARY_KEY4 must be 0 or 1" >&2; exit 2; fi
OUT="${OUT:-$ONEESAN_ROOT/work/b300_closure_ptxas_n${N}_${TRANSPOSE_MODE}_pm${PM_ACCUM}_key4${TERNARY_KEY4}.tsv}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300_closure_ptxas_n${N}_${TRANSPOSE_MODE}_pm${PM_ACCUM}_key4${TERNARY_KEY4}_logs}"
mkdir -p "$(dirname "$OUT")" "$LOGDIR"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

build_one(){
  local label="$1" script="$2" extra_name="$3" extra_value="$4"
  local log="$LOGDIR/${label}.ptxas.log"
  echo "=== ptxas build $label ===" >&2
  if [[ -n "$extra_name" ]]; then
    env N="$N" ARCH="$ARCH" TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" PTXAS_VERBOSE=1 \
      "$extra_name=$extra_value" bash "$ONEESAN_ROOT/$script" >"$LOGDIR/${label}.out" 2>"$log"
  else
    env N="$N" ARCH="$ARCH" TRANSPOSE_MODE="$TRANSPOSE_MODE" PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" PTXAS_VERBOSE=1 \
      bash "$ONEESAN_ROOT/$script" >"$LOGDIR/${label}.out" 2>"$log"
  fi
  python3 "$PARSER" "$log" --label "$label"
}

printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$OUT"
{
  build_one zero_thread scripts/build/b300-bucket-snake-zero-graph-batch.sh ZERO_HIGH_PLAN thread
  build_one zero_shared scripts/build/b300-bucket-snake-zero-graph-batch.sh ZERO_HIGH_PLAN shared
  build_one pattern10 scripts/build/b300-bucket-snake-pattern10-graph-batch.sh '' ''
  build_one depth8_thread scripts/build/b300-bucket-snake-pattern10-depth8-graph-batch.sh HIGH_CTX thread
  build_one depth8_shared scripts/build/b300-bucket-snake-pattern10-depth8-graph-batch.sh HIGH_CTX shared
} >>"$OUT"

cat "$OUT"
echo "closure-ptxas OK n=$N arch=$ARCH transpose=$TRANSPOSE_MODE pm_accum=$PM_ACCUM ternary_key4=$TERNARY_KEY4 result=$OUT logs=$LOGDIR" >&2
