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
MATRIX_BLOCKS="${MATRIX_BLOCKS:-4096}"
RUN_MATRIX="${RUN_MATRIX:-1}"
NGPU="${NGPU:-8}"
MAX_DIRECT_OVER_LOGICAL="${MAX_DIRECT_OVER_LOGICAL:-0}"

BUILD_SCRIPT="$(repo_path scripts/build/gridfp-reduced-component-probe.sh)"
CAP_BIN="$(build_path gridfp_reduced_component_p2p-capability)"
PROOF_BIN="$(build_path gridfp_reduced_component_support-rank)"
LUT_BIN="$(build_path gridfp_reduced_component_p2p-owner-lut)"
TRAFFIC_BIN="$(build_path gridfp_reduced_component_p2p-traffic)"
MATRIX_BIN="$(build_path gridfp_reduced_component_p2p-traffic-matrix)"

for spec in \
  "p2p-capability:$CAP_BIN" \
  "support-rank:$PROOF_BIN" \
  "p2p-owner-lut:$LUT_BIN" \
  "p2p-traffic:$TRAFFIC_BIN"; do
  mode="${spec%%:*}"
  bin="${spec#*:}"
  MODE="$mode" ARCH="$ARCH" OUT="$(basename "$bin")" bash "$BUILD_SCRIPT"
done
if [[ "$RUN_MATRIX" == 1 ]]; then
  MODE=p2p-traffic-matrix ARCH="$ARCH" OUT="$(basename "$MATRIX_BIN")" \
    bash "$BUILD_SCRIPT"
fi

echo "== P2P capability / native-atomic preflight =="
CAP_OUTPUT="$($CAP_BIN "$NGPU")"
printf '%s\n' "$CAP_OUTPUT"
NATIVE_FASTPATH="$(printf '%s\n' "$CAP_OUTPUT" | awk '
  /p2p-capability-summary/ {
    for (i = 1; i <= NF; ++i) {
      if ($i ~ /^full_native_atomic_mesh=/) {
        split($i, a, "="); print a[2]; exit
      }
    }
  }
')"

echo "== support-only slab-rank equivalence =="
"$PROOF_BIN" "$PROOF_W" "$PROOF_K" "$PROOF_BLOCKS" "$NGPU"

echo "== production owner-LUT equivalence =="
"$LUT_BIN" "$TRAFFIC_W" "$TRAFFIC_K" "$PROOF_BLOCKS" "$NGPU"

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

printf 'p2p-traffic-summary W=%s Kwin=%s shift=%s ngpu=%s max_direct_over_logical=%s native_atomic_fastpath=%s\n' \
  "$TRAFFIC_W" "$TRAFFIC_K" "$TRAFFIC_S" "$NGPU" "$MAX_OVERHEAD" \
  "${NATIVE_FASTPATH:-unknown}"

if [[ "$RUN_MATRIX" == 1 ]]; then
  echo "== production-width owner-pair traffic matrix =="
  "$MATRIX_BIN" "$TRAFFIC_W" "$TRAFFIC_K" "$TRAFFIC_S" \
    "$MATRIX_BLOCKS" "$NGPU"
fi

if [[ "${NATIVE_FASTPATH:-0}" == 1 ]]; then
  echo "p2p-recommendation token-mailbox-native-atomic candidate=1"
else
  echo "p2p-recommendation token-mailbox-native-atomic candidate=0 reason=no-full-native-atomic-mesh"
fi

if awk -v limit="$MAX_DIRECT_OVER_LOGICAL" -v got="$MAX_OVERHEAD" \
  'BEGIN { exit !(limit > 0 && got > limit) }'; then
  echo "direct P2P traffic overhead ${MAX_OVERHEAD} exceeds limit ${MAX_DIRECT_OVER_LOGICAL}" >&2
  exit 5
fi
