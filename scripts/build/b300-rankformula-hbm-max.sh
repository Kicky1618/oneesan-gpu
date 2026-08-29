#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Aggressive B300 bandwidth-for-latency preset.  Compared with hbm-fast this
# intentionally issues all seven CROSS source reads whenever a CROSS descriptor
# is active, even when fewer are mathematically selected.  Unused descriptor
# slots are zero-filled and masked after the reads, so semantics are unchanged.
# This is intended for the observed low-MC-utilization regime, not as a universal
# default.
N="${N:-21}"
OUT="${OUT:-$ONEESAN_BUILD_DIR/oneesan_cuda_gridfp_b300_rankformula_hbm_max_n${N}}"
N="$N" OUT="$OUT" \
RANKFORMULA_DIRECTGATHER_FORCE7="${RANKFORMULA_DIRECTGATHER_FORCE7:-1}" \
MAXRREGCOUNT="${MAXRREGCOUNT:-0}" \
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}" \
bash "$ONEESAN_ROOT/scripts/build/b300-rankformula-hbm-fast.sh"

echo "b300-rankformula-hbm-max OK out=$OUT n=$N force7=${RANKFORMULA_DIRECTGATHER_FORCE7:-1} maxrregcount=${MAXRREGCOUNT:-0}" >&2
