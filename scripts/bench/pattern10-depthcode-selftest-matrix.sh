#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"
ARCH="${ARCH:-sm_80}"

for pm in 0 1; do
  for load in global ldg; do
    echo "=== depthcode selftest pm_accum=$pm decode_load=$load ===" >&2
    W="$W" ARCH="$ARCH" PM_ACCUM="$pm" DECODE_LOAD="$load" \
      bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-selftest.sh"
  done
done

echo "pattern10-depthcode-selftest-matrix OK W=$W pm_accum=0,1 decode_load=global,ldg high_ctx=thread,resolved,warp,warpstriped" >&2
