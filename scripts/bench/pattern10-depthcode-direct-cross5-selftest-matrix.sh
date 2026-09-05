#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; ARCH="${ARCH:-sm_80}"
for pm in 0 1; do
  for load in global ldg; do
    echo "=== direct CROSS5 selftest pm=$pm decode=$load ===" >&2
    W="$W" ARCH="$ARCH" PM_ACCUM="$pm" DECODE_LOAD="$load" \
      bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-direct-cross5-selftest.sh"
  done
done
echo "pattern10-depthcode-direct-cross5-selftest-matrix OK W=$W pm=0,1 decode=global,ldg" >&2
