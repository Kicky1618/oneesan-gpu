#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

if ! command -v nvcc >/dev/null 2>&1 || ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo 'row6 occvmm regression: SKIP (CUDA compiler/GPU unavailable)'
  exit 0
fi

SRC="src/cuda/b300/occmajor_authvmm_rank16_batch.cu"
ARCH="${ARCH:-native}"

run_case() {
  local n="$1" low="$2" high="$3" mod="$4" expected="$5"
  local bin="$ONEESAN_BUILD_DIR/row6_occvmm_regression_n${n}"
  N="$n" SRC="$SRC" LOW_LUT_K="$low" HIGH_LUT_K="$high" ARCH="$ARCH" OUT="$bin" \
    "$ONEESAN_ROOT/scripts/build/b300-hbm32-batch.sh" >/dev/null
  python3 "$ONEESAN_ROOT/scripts/solve/build_provenance.py" verify \
    "${bin}.provenance.json" --binary "$bin" --root "$ONEESAN_ROOT" \
    --expect-compile-arg="-DTARGET_W=$((n + 1))" >/dev/null
  local log="$ONEESAN_BUILD_DIR/row6_occvmm_regression_n${n}_${mod}.log"
  GRIDFP_AUTH_VMM=1 GRIDFP_DIRECT_AUTH=3 GRIDFP_BOUNDED_PREFIX_K=6 \
    "$bin" "$n" 4096 14 1 "$mod" >"$log" 2>&1
  grep -q 'AUTHVMM main virtual_gib=' "$log"
  grep -q 'factor LOW16 VMM16 ' "$log"
  grep -q 'factor HIGH16 VMM16 ' "$log"
  local got
  got="$(sed -n 's/.* residue=\([0-9][0-9]*\) modulus=.*/\1/p' "$log" | tail -n 1)"
  if [[ "$got" != "$expected" ]]; then
    echo "row6 occvmm n=$n mod=$mod residue mismatch: got=$got expected=$expected" >&2
    tail -n 40 "$log" >&2
    exit 1
  fi
  echo "row6 occvmm n=$n mod=$mod: PASS residue=$got"
}

run_case 20 10 10 4294966997 3704549185
run_case 21 11 10 4294966997 2124618149
run_case 20 10 10 4294967291 2308006916

echo 'row6 occvmm regression: PASS'
