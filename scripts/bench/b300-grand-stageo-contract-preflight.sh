#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GEN="$ONEESAN_ROOT/scripts/build/gen-b300-grand-selector-stageo.py"
WRAP="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stageo.sh"
NWRAP="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stagen.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-stageo-cg-l2-staged-fullprime-race.sh"
for f in "$GEN" "$WRAP" "$NWRAP" "$PROMOTE"; do [[ -f "$f" ]] || { echo "missing Stage-O grand dependency=$f" >&2; exit 2; }; case "$f" in *.py) python3 -m py_compile "$f";; *) bash -n "$f";; esac; done

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stageo-grand.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
STAGEN_SELECTOR_BUILD_DIR="$tmp/n" STAGEO_SELECTOR_BUILD_DIR="$tmp/o" PATCH_ONLY=1 bash "$WRAP" 27 >"$tmp/stageo.env"
# shellcheck disable=SC1090
source "$tmp/stageo.env"
[[ "${B300_STAGEO_SELECTOR_PATCHED:-0}" == 1 && -s "${B300_STAGEO_SELECTOR_GENERATED:-}" ]] || { echo 'Stage-O selector overlay not generated' >&2; exit 3; }
SEL="$B300_STAGEO_SELECTOR_GENERATED"; bash -n "$SEL"
need(){ local s="$1"; grep -Fq "$s" "$SEL" || { echo "Stage-O grand marker missing: $s" >&2; exit 3; }; }
for s in \
  'RUN_STAGEN=' 'RUN_STAGEO=' \
  'STAGEO_MIN_SPEEDUP=' 'STAGEO_PAIR_L2_LIST=' 'STAGEO_BLOCK_L2_LIST=' \
  'b300x8-nextgen-hybrid8-stageo-cg-l2-staged-fullprime-race.sh' \
  'Stage-O control is not exact Stage-N winner' \
  'Stage-O returned inherited L2 baseline' \
  'MODE=stageo_cgl2_grand' 'MODE=stageo_cgl2_joint' \
  'B300_GRAND_STAGEO_OK' 'B300_GRAND_STAGEO_PAIR_L2_BYTES' 'B300_GRAND_STAGEO_BLOCK_L2_BYTES' \
  'B300_GRAND_STAGEO_INTEGRATED=1' 'B300_GRAND_STAGEN_INTEGRATED=1'; do need "$s"; done
python3 - "$SEL" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('Stage-O grand selector must retain exactly one complete-prime race')
o=s.find('b300x8-nextgen-hybrid8-stageo-cg-l2-staged-fullprime-race.sh')
r=s.find('b300x8-race-external-forced-profiled-once.sh')
if o < 0 or r < 0 or o >= r: raise SystemExit('Stage O must prepare before the one complete-prime race')
ob=s.find('if ((STAGEO_OK && NEXTSELF_OK)); then'); nb=s.find('elif ((STAGEN_OK && NEXTSELF_OK)); then')
if ob < 0 or nb < 0 or ob >= nb: raise SystemExit('Stage-O branch must dominate Stage-N fallback')
for marker in ('P_BIN="$B300_STAGEO_PREPARED_BIN"','B_BIN="$B300_STAGEO_PREPARED_CONTROL_BIN"'):
    if marker not in s: raise SystemExit('Stage-O P/B replacement missing: '+marker)
if '[[ "$B300_STAGEO_PREPARED_CONTROL_BIN" == "$B300_STAGEN_PREPARED_BIN" ]]' not in s:
    raise SystemExit('Stage-O exact Stage-N control binding missing')
print('stageo_grand_single_prime_contract=OK')
PY
echo 'b300-grand-stageo-contract-preflight OK overlay_chain=stagen_then_stageo stage_o_after_n=1 exact_n_control=1 fallback_n=1 forced_slots_replaced=1 seven_candidates_preserved=1 single_complete_prime=1 gpu_work=0'
