#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PLAN_ONLY="${PLAN_ONLY:-0}"
ARCH="${ARCH:-native}"
PTXAS_VERBOSE="${PTXAS_VERBOSE:-1}"
BLOCKS="${BLOCKS:-256}"

if [[ "$PLAN_ONLY" == 1 ]]; then
  W="${W:-28}"
  KWIN="${KWIN:-13}"
  SHIFT="${SHIFT:-13}"
  NGPU="${NGPU:-8}"
else
  W="${W:-10}"
  KWIN="${KWIN:-4}"
  SHIFT="${SHIFT:-3}"
  NGPU="${NGPU:-2}"
fi

OUT="gridfp_reduced_component_peer-shift-cycle"
MODE=peer-shift-cycle ARCH="$ARCH" PTXAS_VERBOSE="$PTXAS_VERBOSE" OUT="$OUT" \
  "$(repo_path scripts/build/gridfp-reduced-component-probe.sh)"

BIN="$(build_path "$OUT")"
ARGS=("$W" "$KWIN" "$SHIFT" "$BLOCKS" "$NGPU")
if [[ "$PLAN_ONLY" == 1 ]]; then ARGS+=(--plan-only); fi

printf 'running %q' "$BIN"
printf ' %q' "${ARGS[@]}"
printf '\n'
"$BIN" "${ARGS[@]}"
