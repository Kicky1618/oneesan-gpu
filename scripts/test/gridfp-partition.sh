#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BIN="${BIN:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_multigpu_mmap_partition_test}"
if [[ ! -x "$BIN" ]]; then
  ARCH="${ARCH:-native}" OUT="$BIN" "$ONEESAN_ROOT/scripts/build/gridfp-multigpu-mmap.sh"
fi

GRIDFP_PARTITION_SELFTEST=1 exec "$BIN"
