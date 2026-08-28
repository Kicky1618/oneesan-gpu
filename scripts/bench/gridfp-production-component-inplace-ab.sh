#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/scripts/lib/common.sh"

W="${W:-10}"
BLOCKS="${BLOCKS:-4096}"
MOD="${MOD:-4294967291}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
BIN="$(build_path gridfp_reduced_component_inplace)"

MODE=inplace \
ARCH="$ARCH" \
PTXAS_VERBOSE="$PTXAS_VERBOSE" \
OUT="$BIN" \
  bash "$ROOT/scripts/build/gridfp-reduced-component-probe.sh"

echo "=== W=28 memory plan ===" >&2
"$BIN" 28 "$BLOCKS" "$MOD" --plan-only

echo "=== W=$W correctness/timing A/B ===" >&2
"$BIN" "$W" "$BLOCKS" "$MOD"
