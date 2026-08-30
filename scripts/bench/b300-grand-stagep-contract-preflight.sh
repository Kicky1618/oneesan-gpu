#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-grand-selector-stagep.py"; WRAP="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stagep.sh"; OWRAP="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stageo.sh"; PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-stagep-mate-cg-l2-staged-fullprime-race.sh"
for f in "$GEN" "$WRAP" "$OWRAP" "$PROMOTE"; do [[ -f "$f" ]] || exit 2; case "$f" in *.py) python3 -m py_compile "$f";; *) bash -n "$f";; esac; done
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stagep-grand.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
STAGEN_SELECTOR_BUILD_DIR="$tmp/n" STAGEO_SELECTOR_BUILD_DIR="$tmp/o" STAGEP_SELECTOR_BUILD_DIR="$tmp/p" PATCH_ONLY=1 bash "$WRAP" 27 >"$tmp/p.env"; source "$tmp/p.env"; [[ "${B300_STAGEP_SELECTOR_PATCHED:-0}" == 1 && -s "${B300_STAGEP_SELECTOR_GENERATED:-}" ]] || exit 3; SEL="$B300_STAGEP_SELECTOR_GENERATED"; bash -n "$SEL"
need(){ grep -Fq "$1" "$SEL" || { echo "Stage-P grand marker missing: $1" >&2; exit 3; }; }
for s in 'RUN_STAGEP=' 'STAGEP_MIN_SPEEDUP=' 'STAGEP_MATE_L2_LIST=' 'b300x8-nextgen-hybrid8-stagep-mate-cg-l2-staged-fullprime-race.sh' 'Stage-P control is not exact Stage-O winner' 'Stage-P control is not exact Stage-N winner' 'Stage-P mate L2 contract drift' 'MODE=stagep_matel2_grand' 'MODE=stagep_matel2_joint' 'B300_GRAND_STAGEP_OK' 'B300_GRAND_STAGEP_COUNT_UPSTREAM' 'B300_GRAND_STAGEP_MATE_L2_BYTES' 'B300_GRAND_STAGEP_INTEGRATED=1' 'B300_GRAND_STAGEO_INTEGRATED=1'; do need "$s"; done
python3 - "$SEL" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
if s.count('b300x8-race-external-forced-profiled-once.sh')!=1: raise SystemExit('Stage-P grand selector must retain one complete-prime race')
p=s.find('b300x8-nextgen-hybrid8-stagep-mate-cg-l2-staged-fullprime-race.sh'); r=s.find('b300x8-race-external-forced-profiled-once.sh')
if p<0 or r<0 or p>=r: raise SystemExit('Stage P must prepare before final race')
pb=s.find('if ((STAGEP_OK && NEXTSELF_OK)); then'); ob=s.find('elif ((STAGEO_OK && NEXTSELF_OK)); then'); nb=s.find('elif ((STAGEN_OK && NEXTSELF_OK)); then')
if min(pb,ob,nb)<0 or not (pb<ob<nb): raise SystemExit('Stage-P/O/N priority order broken')
for m in ('P_BIN="$B300_STAGEP_PREPARED_BIN"','B_BIN="$B300_STAGEP_PREPARED_CONTROL_BIN"'):
    if m not in s: raise SystemExit('Stage-P P/B mapping missing '+m)
print('stagep_grand_single_prime_contract=OK')
PY
echo 'b300-grand-stagep-contract-preflight OK overlay_chain=stagen_stageo_stagep priority=P>O>N exact_upstream_control=1 fallback_on=1 seven_candidates_preserved=1 single_complete_prime=1 gpu_work=0'
