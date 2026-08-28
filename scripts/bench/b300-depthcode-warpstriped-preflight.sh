#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
ARCH="${ARCH:-native}"
RUN_PTXAS="${RUN_PTXAS:-1}"
RUN_B300_AB="${RUN_B300_AB:-0}"
RUN_DELTA_AB="${RUN_DELTA_AB:-0}"
AB_N="${AB_N:-21}"
REPEATS="${REPEATS:-3}"
DEPTHCODE_DECODE_LOAD="${DEPTHCODE_DECODE_LOAD:-ldg}"

if [[ "$RUN_PTXAS" != 0 && "$RUN_PTXAS" != 1 ]]; then echo "RUN_PTXAS must be 0 or 1" >&2; exit 2; fi
if [[ "$RUN_B300_AB" != 0 && "$RUN_B300_AB" != 1 ]]; then echo "RUN_B300_AB must be 0 or 1" >&2; exit 2; fi
if [[ "$RUN_DELTA_AB" != 0 && "$RUN_DELTA_AB" != 1 ]]; then echo "RUN_DELTA_AB must be 0 or 1" >&2; exit 2; fi
case "$DEPTHCODE_DECODE_LOAD" in global|ldg) ;; *) echo "DEPTHCODE_DECODE_LOAD must be global or ldg" >&2; exit 2;; esac

echo '=== host schedule proof ===' >&2
bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-warpstriped-schedule-selftest.sh"

echo '=== W28/all-legal 10-bit bound ===' >&2
N="$N" bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-bound.sh"

echo '=== all-legal closure ternary-delta proof ===' >&2
N="$N" bash "$ONEESAN_ROOT/scripts/bench/closure-ternary-delta-proof.sh"

echo '=== direct depthcode build-plan invariants ===' >&2
N="$N" ARCH="$ARCH" bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-build-plan.sh"

echo '=== CUDA reference matrix: PM 0/1 x global/ldg x all stable HIGH contexts ===' >&2
ARCH="$ARCH" W=10 bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-selftest-matrix.sh"

if [[ "$RUN_PTXAS" == 1 ]]; then
  echo '=== ptxas resource comparison, including delta experiments ===' >&2
  N="$N" ARCH="$ARCH" bash "$ONEESAN_ROOT/scripts/bench/b300-depthcode-highctx-ptxas.sh"
fi

if [[ "$RUN_DELTA_AB" == 1 ]]; then
  echo '=== B300 resolved vs resolved-delta plan A/B ===' >&2
  N="$AB_N" REPEATS="$REPEATS" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-depthcode-resolved-delta-ab.sh"
fi

if [[ "$RUN_B300_AB" == 1 ]]; then
  echo '=== B300 warp vs warpstriped residue/timing A/B ===' >&2
  N="$AB_N" REPEATS="$REPEATS" DEPTHCODE_DECODE_LOAD="$DEPTHCODE_DECODE_LOAD" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-depthcode-warpstriped-ab.sh"
fi

echo "b300-depthcode-warpstriped-preflight OK n=$N arch=$ARCH run_ptxas=$RUN_PTXAS run_delta_ab=$RUN_DELTA_AB run_b300_ab=$RUN_B300_AB decode_load=$DEPTHCODE_DECODE_LOAD ternary_delta_proved=1" >&2
