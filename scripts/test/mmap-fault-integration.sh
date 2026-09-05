#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-6}"
MOD="${MOD:-2305843009213693951}"
TARGET_MIB="${TARGET_MIB:-64}"
MAX_WINDOW="${MAX_WINDOW:-14}"
NGPU="${NGPU:-1}"
BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_multigpu_mmap_fault_test}"

for spec in "N:$N" "MOD:$MOD" "TARGET_MIB:$TARGET_MIB" "MAX_WINDOW:$MAX_WINDOW" "NGPU:$NGPU"; do
  require_uint "${spec%%:*}" "${spec#*:}" || exit 2
done

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo 'SKIP: a working NVIDIA driver/GPU is required for mmap fault integration' >&2
  exit 77
fi

visible="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if (( visible < NGPU )); then
  echo "requested $NGPU GPUs, but only $visible are visible" >&2
  exit 2
fi

if [[ ! -x "$BIN" ]]; then
  ARCH="${ARCH:-native}" OUT="$BIN" "$ONEESAN_ROOT/scripts/build/gridfp-multigpu-mmap.sh"
fi

ROOT="${FAULT_TEST_ROOT:-$ONEESAN_ROOT/work/mmap_fault_integration_$$}"
mkdir -p "$ROOT"
trap 'rm -rf "$ROOT"' EXIT

run_solver() {
  local store="$1"
  shift
  env "$@" "$BIN" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$store"
}

parse_residue() {
  sed -n 's/.* residue=\([0-9][0-9]*\).*/\1/p' | tail -n 1
}

echo '== baseline =='
baseline_out="$(run_solver "$ROOT/baseline" GRIDFP_RESUME=1 GRIDFP_FRESH=1)"
baseline="$(printf '%s\n' "$baseline_out" | parse_residue)"
if [[ -z "$baseline" ]]; then
  echo 'could not parse baseline residue' >&2
  printf '%s\n' "$baseline_out" >&2
  exit 3
fi
echo "baseline residue=$baseline"

for phase in journal scatter commit; do
  store="$ROOT/$phase"
  echo "== injected crash: $phase =="
  set +e
  run_solver "$store" \
    GRIDFP_RESUME=1 GRIDFP_FRESH=1 \
    GRIDFP_FAULT_GROUP=0 GRIDFP_FAULT_PHASE="$phase" \
    >"$ROOT/$phase.crash.out" 2>"$ROOT/$phase.crash.err"
  rc=$?
  set -e
  if [[ "$rc" -ne 86 ]]; then
    echo "fault phase $phase exited $rc instead of 86" >&2
    cat "$ROOT/$phase.crash.err" >&2
    cat "$ROOT/$phase.crash.out" >&2
    exit 4
  fi
  if ! grep -q "FAULT_INJECT group=0 phase=$phase exit=86" "$ROOT/$phase.crash.err"; then
    echo "fault phase $phase did not reach the requested durable point" >&2
    cat "$ROOT/$phase.crash.err" >&2
    exit 5
  fi

  echo "== resume: $phase =="
  resumed_out="$(run_solver "$store" GRIDFP_RESUME=1 GRIDFP_FRESH=0)"
  resumed="$(printf '%s\n' "$resumed_out" | parse_residue)"
  if [[ "$resumed" != "$baseline" ]]; then
    echo "resume mismatch after $phase crash: got=$resumed expected=$baseline" >&2
    printf '%s\n' "$resumed_out" >&2
    exit 6
  fi
  echo "phase=$phase residue=$resumed MATCH"
done

echo 'mmap GPU fault integration: PASS'
