#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
PROOF_W="${PROOF_W:-11}"
PROOF_K="${PROOF_K:-4}"
PROOF_BLOCKS="${PROOF_BLOCKS:-256}"
TRAFFIC_W="${TRAFFIC_W:-28}"
TRAFFIC_K="${TRAFFIC_K:-13}"
TRAFFIC_S="${TRAFFIC_S:-13}"
TRAFFIC_BLOCKS="${TRAFFIC_BLOCKS:-4096}"
NGPU="${NGPU:-8}"
MAX_DIRECT_OVER_LOGICAL="${MAX_DIRECT_OVER_LOGICAL:-0}"

BUILD_SCRIPT="$(repo_path scripts/build/gridfp-reduced-component-probe.sh)"
PROOF_BIN="$(build_path gridfp_reduced_component_support-rank)"
TRAFFIC_BIN="$(build_path gridfp_reduced_component_p2p-traffic)"

MODE=support-rank ARCH="$ARCH" OUT="$(basename "$PROOF_BIN")" \
  bash "$BUILD_SCRIPT"
MODE=p2p-traffic ARCH="$ARCH" OUT="$(basename "$TRAFFIC_BIN")" \
  bash "$BUILD_SCRIPT"

echo "== support-only slab-rank equivalence =="
"$PROOF_BIN" "$PROOF_W" "$PROOF_K" "$PROOF_BLOCKS" "$NGPU"

echo "== production-width P2P traffic =="
OUTPUT="$($TRAFFIC_BIN "$TRAFFIC_W" "$TRAFFIC_K" "$TRAFFIC_S" \
  "$TRAFFIC_BLOCKS" "$NGPU")"
printf '%s\n' "$OUTPUT"

MAX_OVERHEAD="$(printf '%s\n' "$OUTPUT" | awk '
  /gridfp-reduced-production-p2p-traffic/ {
    for (i = 1; i <= NF; ++i) {
      if ($i ~ /^direct_over_logical=/) {
        split($i, a, "=");
        if (a[2] + 0 > max) max = a[2] + 0;
      }
    }
  }
  END { printf "%.9f", max + 0 }
')"

printf 'p2p-traffic-summary W=%s Kwin=%s shift=%s ngpu=%s max_direct_over_logical=%s\n' \
  "$TRAFFIC_W" "$TRAFFIC_K" "$TRAFFIC_S" "$NGPU" "$MAX_OVERHEAD"

if awk -v limit="$MAX_DIRECT_OVER_LOGICAL" -v got="$MAX_OVERHEAD" \
  'BEGIN { exit !(limit > 0 && got > limit) }'; then
  echo "direct P2P traffic overhead ${MAX_OVERHEAD} exceeds limit ${MAX_DIRECT_OVER_LOGICAL}" >&2
  exit 5
fi
