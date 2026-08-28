#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
W="${W:-10}"
K="${K:-4}"
S="${S:-4}"
BATCHES="${BATCHES:-8}"
BLOCKS="${BLOCKS:-256}"
NGPU="${NGPU:-8}"

BUILD_SCRIPT="$(repo_path scripts/build/gridfp-reduced-component-probe.sh)"
BIN="$(build_path gridfp_reduced_component_p2p-host-persistent-list)"

MODE=p2p-host-persistent-list ARCH="$ARCH" OUT="$(basename "$BIN")" \
  bash "$BUILD_SCRIPT"

OUTPUT="$($BIN "$W" "$K" "$S" "$BATCHES" "$BLOCKS" "$NGPU")"
printf '%s\n' "$OUTPUT"

EXACT="$(printf '%s\n' "$OUTPUT" | awk '
  /gridfp-p2p-host-persistent-list/ &&
  /startup_gpu_support_scan_passes=0/ &&
  /startup_gpu_count_passes=0/ &&
  /runtime_support_scan_passes=0/ &&
  /runtime_count_passes=0/ &&
  /exact=OK/ { ++n }
  END { print n + 0 }
')"

if [[ "$EXACT" != 2 ]]; then
  echo "host persistent-list executor did not prove both directions exact" >&2
  exit 4
fi

if ! printf '%s\n' "$OUTPUT" | grep -q 'ALL_OK gridfp_p2p_host_persistent_list=1'; then
  echo "host persistent-list executor missing ALL_OK" >&2
  exit 5
fi

echo "host-persistent-list-runner exact=OK W=$W K=$K shift=$S batches=$BATCHES ngpu=$NGPU startup_gpu_support_scan_passes=0"
