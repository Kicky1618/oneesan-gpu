#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
FIRST="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-latest.sh"; EXACT="$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-latest.sh"
for f in "$FIRST" "$EXACT"; do [[ -f "$f" ]] || exit 2; bash -n "$f"; done
grep -Fq 'b300x8-grand-firstpass-stagep.sh' "$FIRST" || { echo 'latest firstpass is not Stage P' >&2; exit 3; }
grep -Fq 'b300x8-grand-promote-exact-stagep.sh' "$EXACT" || { echo 'latest exact promoter is not Stage P' >&2; exit 3; }
[[ "$(grep -c '^exec bash ' "$FIRST")" == 1 && "$(grep -c '^exec bash ' "$EXACT")" == 1 ]] || { echo 'latest entrypoints must be transparent exec wrappers' >&2; exit 3; }
echo 'b300-grand-latest-entrypoint-preflight OK latest_stage=P firstpass=1 exact_promotion=1 transparent_exec=1 gpu_work=0'
