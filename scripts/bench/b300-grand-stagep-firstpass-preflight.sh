#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-grand-firstpass-stagep.py"; WRAP="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-stagep.sh"; SEL="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stagep.sh"
for f in "$GEN" "$WRAP" "$SEL"; do [[ -f "$f" ]] || exit 2; case "$f" in *.py) python3 -m py_compile "$f";; *) bash -n "$f";; esac; done
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stagep-firstpass.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
STAGEN_FIRSTPASS_BUILD_DIR="$tmp/n" STAGEO_FIRSTPASS_BUILD_DIR="$tmp/o" STAGEP_FIRSTPASS_BUILD_DIR="$tmp/p" PATCH_ONLY=1 bash "$WRAP" 27 >"$tmp/p.env"; source "$tmp/p.env"; [[ "${B300_STAGEP_FIRSTPASS_PATCHED:-0}" == 1 && -s "${B300_STAGEP_FIRSTPASS_GENERATED:-}" ]] || exit 3; FIRST="$B300_STAGEP_FIRSTPASS_GENERATED"; bash -n "$FIRST"
need(){ grep -Fq "$1" "$FIRST" || { echo "Stage-P firstpass marker missing: $1" >&2; exit 3; }; }
for s in 'RUN_STAGEP=' 'STAGEP_MIN_SPEEDUP=' 'STAGEP_MATE_L2_LIST=' 'b300-stagep-preflight.sh' 'b300-grand-stagep-contract-preflight.sh' 'b300x8-joint-nextself-hybrid8-select-stagep.sh' 'B300_GRAND_SELECTED_STAGEP_ENABLED' 'B300_GRAND_SELECTED_STAGEP_ACCEPTED' 'B300_GRAND_SELECTED_STAGEP_COUNT_UPSTREAM' 'B300_GRAND_SELECTED_STAGEP_BASE_MATE_L2_BYTES' 'B300_GRAND_SELECTED_STAGEP_MATE_L2_BYTES' 'B300_GRAND_SELECTED_STAGEP_SEARCH_MATE_L2' 'B300_GRAND_STAGEP_INTEGRATED'; do need "$s"; done
python3 - "$FIRST" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
if 'b300x8-joint-nextself-hybrid8-select-stagep.sh" 27' not in s: raise SystemExit('Stage-P firstpass does not call Stage-P selector')
if 'RUN_STAGEP="$RUN_STAGEP"' not in s or 'STAGEP_MIN_SPEEDUP="$STAGEP_MIN_SPEEDUP"' not in s: raise SystemExit('Stage-P knobs not forwarded')
m=re.search(r'B300_GRAND_SELECTED_SCHEMA=([0-9]+)',s)
if not m or int(m.group(1))<3: raise SystemExit('selected schema <3')
print('stagep_firstpass_contract=OK schema='+m.group(1))
PY
echo 'b300-grand-stagep-firstpass-preflight OK overlay_chain=stageo_then_stagep selector=stagep selected_mate_l2_provenance=1 single_complete_prime=1 gpu_work=0'
