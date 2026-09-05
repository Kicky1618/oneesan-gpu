#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-28}"
NGPU="${NGPU:-8}"
BLOCKS="${BLOCKS:-4096}"
SCRATCH_CAP_GIB="${SCRATCH_CAP_GIB:-32}"
BATCH_CANDIDATES="${BATCH_CANDIDATES:-6 7 8 10 12 16}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-0}"

MODE=segment-major-plan ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" \
  "$(repo_path scripts/build/gridfp-reduced-p2p-schedule-probe.sh)"
BIN="$(build_path gridfp_reduced_p2p_segment-major-plan)"

selected=""
for batches in $BATCH_CANDIDATES; do
  echo "=== segment-major plan W=$W ngpu=$NGPU batches=$batches scratch_cap_GiB=$SCRATCH_CAP_GIB ==="
  out="$($BIN "$W" "$NGPU" "$batches" "$BLOCKS" "$SCRATCH_CAP_GIB")"
  printf '%s\n' "$out"
  if grep -q 'scratch_cap_ok=1' <<<"$out"; then
    selected="$batches"
    break
  fi
done

if [[ -z "$selected" ]]; then
  echo "no batch candidate satisfies scratch cap; tried: $BATCH_CANDIDATES" >&2
  exit 4
fi

echo "SELECTED_SEGMENT_MAJOR_BATCHES=$selected W=$W NGPU=$NGPU SCRATCH_CAP_GIB=$SCRATCH_CAP_GIB"
