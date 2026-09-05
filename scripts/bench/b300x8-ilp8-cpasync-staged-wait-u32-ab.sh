#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BASE="$ONEESAN_ROOT/scripts/bench/b300x8-ilp8-cpasync-staged-wait-ab.sh"
[[ -f "$BASE" ]] || { echo "missing base A/B script: $BASE" >&2; exit 2; }
TMP="$(mktemp "$ONEESAN_ROOT/scripts/bench/.b300_staged_u32.XXXXXX.sh")"
cleanup(){ rm -f "$TMP"; }
trap cleanup EXIT INT TERM
sed \
  -e 's/gen-b300-main-rankstate-ilp8-cpasync-staged-wait\.py/gen-b300-main-rankstate-ilp8-cpasync-staged-wait-u32.py/g' \
  -e 's/b300x8_ilp8_cpasync_staged_wait_ab_/b300x8_ilp8_cpasync_staged_wait_u32_ab_/g' \
  "$BASE" >"$TMP"
chmod +x "$TMP"
grep -Fq 'gen-b300-main-rankstate-ilp8-cpasync-staged-wait-u32.py' "$TMP"
echo 'staged_wait_variant=u32 wait_pair_group=1 wait_block_group=0 live_partial_bits=32' >&2
exec bash "$TMP" "$@"
