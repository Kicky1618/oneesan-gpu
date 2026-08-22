#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-8}"
if (( $# > 0 )); then shift; fi
OUTFILE="${1:-$ONEESAN_ROOT/work/zdd/replay_n${N}.zdd}"
if (( $# > 0 )); then shift; fi
MAX_NODES="${MAX_NODES:-33554432}"
BIN="$(build_path "${BIN:-oneesan_zdd_replay}")"

if [[ ! -x "$BIN" ]]; then
  "$ONEESAN_ROOT/scripts/build/zdd-replay.sh"
fi
mkdir -p "$(dirname -- "$OUTFILE")"
exec "$BIN" "$N" "$OUTFILE" "$MAX_NODES" "$@"
