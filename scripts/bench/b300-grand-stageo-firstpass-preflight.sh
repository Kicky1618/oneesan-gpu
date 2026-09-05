#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GEN="$ONEESAN_ROOT/scripts/build/gen-b300-grand-firstpass-stageo.py"
WRAP="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-stageo.sh"
SEL_WRAP="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stageo.sh"
for f in "$GEN" "$WRAP" "$SEL_WRAP"; do [[ -f "$f" ]] || { echo "missing Stage-O firstpass dependency=$f" >&2; exit 2; }; case "$f" in *.py) python3 -m py_compile "$f";; *) bash -n "$f";; esac; done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stageo-firstpass.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
STAGEN_FIRSTPASS_BUILD_DIR="$tmp/nfirst" STAGEO_FIRSTPASS_BUILD_DIR="$tmp/ofirst" PATCH_ONLY=1 bash "$WRAP" 27 >"$tmp/first.env"
# shellcheck disable=SC1090
source "$tmp/first.env"
[[ "${B300_STAGEO_FIRSTPASS_PATCHED:-0}" == 1 && -s "${B300_STAGEO_FIRSTPASS_GENERATED:-}" ]] || { echo 'Stage-O firstpass overlay not generated' >&2; exit 3; }
FIRST="$B300_STAGEO_FIRSTPASS_GENERATED"; bash -n "$FIRST"
need(){ local s="$1"; grep -Fq "$s" "$FIRST" || { echo "Stage-O firstpass marker missing: $s" >&2; exit 3; }; }
for s in \
  'RUN_STAGEO=' 'STAGEO_MIN_SPEEDUP=' 'STAGEO_PAIR_L2_LIST=' 'STAGEO_BLOCK_L2_LIST=' \
  'b300-stageo-preflight.sh' 'b300-grand-stageo-contract-preflight.sh' \
  'b300x8-joint-nextself-hybrid8-select-stageo.sh' \
  'B300_GRAND_SELECTED_STAGEO_ENABLED' 'B300_GRAND_SELECTED_STAGEO_ACCEPTED' \
  'B300_GRAND_SELECTED_STAGEO_PAIR_L2_BYTES' 'B300_GRAND_SELECTED_STAGEO_BLOCK_L2_BYTES' \
  'B300_GRAND_SELECTED_STAGEO_BASE_PAIR_L2_BYTES' 'B300_GRAND_SELECTED_STAGEO_BASE_BLOCK_L2_BYTES' \
  'B300_GRAND_SELECTED_STAGEO_SEARCH_PAIR_L2' 'B300_GRAND_SELECTED_STAGEO_SEARCH_BLOCK_L2' \
  'B300_GRAND_STAGEO_INTEGRATED'; do need "$s"; done
python3 - "$FIRST" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
if 'b300x8-joint-nextself-hybrid8-select-stageo.sh" 27' not in s:
    raise SystemExit('Stage-O firstpass does not call Stage-O-aware selector')
if 'RUN_STAGEO="$RUN_STAGEO"' not in s or 'STAGEO_MIN_SPEEDUP="$STAGEO_MIN_SPEEDUP"' not in s:
    raise SystemExit('Stage-O run/speed knobs are not forwarded')
m=re.search(r'B300_GRAND_SELECTED_SCHEMA=([0-9]+)',s)
if not m or int(m.group(1)) < 3: raise SystemExit('Stage-O firstpass must preserve selected schema >=3')
if 'B300_GRAND_COMPLETE_PRIME_RACES:-0}' not in s:
    raise SystemExit('Stage-O firstpass lost single complete-prime assertion')
print('stageo_firstpass_contract=OK schema='+m.group(1))
PY
echo 'b300-grand-stageo-firstpass-preflight OK overlay_chain=stagen_then_stageo selector=stageo selected_l2_provenance=1 summary_stageo=1 single_complete_prime=1 gpu_work=0'
