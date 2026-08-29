#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-24}"
MOD="${MOD:-4294967291}"
SCRATCH_MIB="${SCRATCH_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-27}"
GPUS=8
ARCH="${ARCH:-native}"
OUT="${OUT:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_hbm32_n${N}_x8_throughput}"

# B300 x8 profile:
# - K=13 rank/unrank LUTs for production widths
# - materialize main MateID once per group/window and compile the hot kernel
#   without the unrank fallback
# - p>1 main output is destination-pull: no main identity memcpy, no main
#   modular CAS scatter, and no blocked->main scatter kernel
# - division-free 8-shard address selection
# - request a wide scratch window; the executable itself caps this to
#   cudaMemGetInfo()-reserve, so 64 GiB is safe even when authoritative state is
#   already large.
N="$N" ARCH="$ARCH" OUT="$OUT" \
FAST_SHARD_ADDRESS8=1 MAIN_MATE_CACHE=1 MAIN_PULL=1 PTXAS_VERBOSE=1 \
bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh"

echo "=== B300 x8 throughput run ===" >&2
echo "n=$N mod=$MOD scratch_mib=$SCRATCH_MIB max_window=$MAX_WINDOW gpus=$GPUS" >&2
exec "$OUT" "$N" "$MOD" "$SCRATCH_MIB" "$MAX_WINDOW" "$GPUS"
