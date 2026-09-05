#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
WRAP="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-staget.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-grand-selector-staget.py"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-staget-ilp2-mate-load-policy-staged-fullprime-race.sh"
BASE_WRAP="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stages.sh"
for f in "$WRAP" "$GEN" "$PROMOTE" "$BASE_WRAP"; do [[ -f "$f" ]] || exit 2; done
bash -n "$WRAP"; python3 -m py_compile "$GEN"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-staget-grand.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
STAGET_SELECTOR_BUILD_DIR="$tmp" PATCH_ONLY=1 bash "$WRAP" 27 >"$tmp/env"
# shellcheck disable=SC1090
source "$tmp/env"
[[ "${B300_STAGET_SELECTOR_PATCHED:-0}" == 1 && -s "${B300_STAGET_SELECTOR_GENERATED:-}" ]] || exit 3
G="$B300_STAGET_SELECTOR_GENERATED"; bash -n "$G"
need(){ grep -Fq "$2" "$1" || { echo "Stage-T grand marker missing: $2" >&2; exit 3; }; }
for s in \
  'RUN_STAGET=' 'STAGET_MIN_SPEEDUP=' 'STAGET_POLICY_LIST=' \
  'b300x8-nextgen-hybrid8-staget-ilp2-mate-load-policy-staged-fullprime-race.sh' 'PREPARE_ONLY=1' \
  'B300_STAGET_PREPARED_CONTROL_BIN" == "$B300_STAGES_PREPARED_BIN' \
  'B300_STAGET_PREPARED_CONTROL_BIN" == "$B300_STAGER_PREPARED_BIN' \
  'MODE=staget_ilp2_mate_grand' 'MODE=staget_ilp2_mate_joint' \
  'grand selector: Stage-T ILP2 mate policy rejected/not-applicable; retaining Stage S/R' \
  'B300_GRAND_STAGET_OK' 'B300_GRAND_STAGET_POLICY' 'B300_GRAND_STAGET_INTEGRATED=1' 'B300_GRAND_COMPLETE_PRIME_RACES=1'; do need "$G" "$s"; done
python3 - "$G" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('Stage-T grand must retain exactly one complete-prime race')
ti=s.find('b300x8-nextgen-hybrid8-staget-ilp2-mate-load-policy-staged-fullprime-race.sh'); pi=s.find('b300x8-race-external-forced-profiled-once.sh')
if ti<0 or pi<0 or ti>=pi: raise SystemExit('Stage T must prepare before the one complete-prime race')
if 'if ((STAGER_OK)); then' not in s: raise SystemExit('Stage T must require accepted Stage R lineage')
if 'if ((STAGES_OK)); then STAGET_UPSTREAM_KIND=stages; else STAGET_UPSTREAM_KIND=stager; fi' not in s:
    raise SystemExit('Stage-T maximal immediate-upstream selection missing')
for bind in (
    '[[ "$B300_STAGET_PREPARED_CONTROL_BIN" == "$B300_STAGES_PREPARED_BIN" ]]',
    '[[ "$B300_STAGET_PREPARED_CONTROL_BIN" == "$B300_STAGER_PREPARED_BIN" ]]',
):
    if bind not in s: raise SystemExit('Stage-T exact immediate-upstream control binding missing: '+bind)
m=re.search(r'if \(\(STAGET_OK && NEXTSELF_OK\)\); then(.*?)elif \(\(STAGET_OK\)\); then',s,re.S)
if not m: raise SystemExit('Stage-T grand branch missing')
b=m.group(1)
for slot,candidate in {'P_BIN':'B300_STAGET_PREPARED_BIN','B_BIN':'B300_STAGET_PREPARED_CONTROL_BIN','E1_BIN':'B300_HYBRID8_PREPARED_BASE_BIN','E2_BIN':'B300_NEXTSELF_PREPARED_BIN','E3_BIN':'JOINT_PRIMARY_BIN'}.items():
    if f'{slot}="${candidate}"' not in b: raise SystemExit(f'Stage-T candidate mapping drift {slot}->{candidate}')
pt=s.find('if ((STAGET_OK && NEXTSELF_OK)); then')
ps=s.find('elif ((STAGES_OK && NEXTSELF_OK)); then')
pr=s.find('elif ((STAGER_OK && NEXTSELF_OK)); then')
if min(pt,ps,pr)<0 or not (pt<ps<pr): raise SystemExit('grand priority must be T > S > R')
print('staget_grand_contract_structure=OK')
PY
echo 'b300-grand-staget-contract-preflight OK priority=T,S,R exact_immediate_control=1 prepare_only=1 fallback=S_or_R forced_slots=5 total_candidates=7 complete_prime_races=1 gpu_work=0'
