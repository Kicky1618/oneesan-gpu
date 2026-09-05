#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
BASE="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh"
WRAP="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-stagen.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-grand-firstpass-stagen.py"
SEL_WRAP="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stagen.sh"
for f in "$BASE" "$WRAP" "$GEN" "$SEL_WRAP"; do [[ -f "$f" ]] || exit 2; case "$f" in *.py) python3 -m py_compile "$f";; *) bash -n "$f";; esac; done
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stagen-firstpass.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
STAGEN_FIRSTPASS_BUILD_DIR="$tmp" PATCH_ONLY=1 bash "$WRAP" 27 >"$tmp/first.env"
# shellcheck disable=SC1090
source "$tmp/first.env"
[[ "${B300_STAGEN_FIRSTPASS_PATCHED:-0}" == 1 && -s "$B300_STAGEN_FIRSTPASS_GENERATED" ]] || exit 3
FIRST="$B300_STAGEN_FIRSTPASS_GENERATED"; bash -n "$FIRST"
need(){ grep -Fq "$1" "$FIRST" || { echo "Stage-N firstpass marker missing: $1" >&2; exit 3; }; }
for s in 'RUN_STAGEN=' 'STAGEN_MIN_SPEEDUP=' 'STAGEN_PAIR_POLICY_LIST=' 'STAGEN_BLOCK_POLICY_LIST=' \
  'b300-stagen-preflight.sh' 'b300-grand-stagen-contract-preflight.sh' 'b300x8-joint-nextself-hybrid8-select-stagen.sh' \
  'B300_GRAND_STAGEN_INTEGRATED' 'B300_GRAND_SELECTED_STAGEN_ACCEPTED' 'B300_GRAND_SELECTED_STAGEN_UPSTREAM_KIND' \
  'B300_GRAND_SELECTED_STAGEN_PAIR_POLICY' 'B300_GRAND_SELECTED_STAGEN_BLOCK_POLICY' 'B300_GRAND_SELECTED_STAGEN_BASE_COUNT_POLICY' \
  'B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1'; do need "$s"; done
python3 - "$FIRST" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
if 'RUN_STAGEN="$RUN_STAGEN"' not in s: raise SystemExit('RUN_STAGEN not passed to selector')
if 'STAGEN_PAIR_POLICY_LIST="$STAGEN_PAIR_POLICY_LIST"' not in s or 'STAGEN_BLOCK_POLICY_LIST="$STAGEN_BLOCK_POLICY_LIST"' not in s:
    raise SystemExit('Stage-N policy lists not passed to selector')
if 'b300x8-joint-nextself-hybrid8-select-stagen.sh" 27' not in s: raise SystemExit('Stage-N selector wrapper missing')
m=re.search(r"printf 'B300_GRAND_SELECTED_SCHEMA=([0-9]+)",s)
if not m or int(m.group(1)) < 3: raise SystemExit('selected schema regressed')
if s.count('complete_prime_races_expected=1') != 1 or s.count("B300_GRAND_SELECTED_COMPLETE_PRIME_RACES=1") != 1:
    raise SystemExit('single-prime provenance markers drifted')
print('stagen_firstpass_structure=OK schema='+m.group(1))
PY
echo "b300-grand-stagen-firstpass-preflight OK native=${B300_STAGEN_FIRSTPASS_NATIVE:-0} selector_overlay=1 selected_provenance=1 single_complete_prime=1 gpu_work=0"
