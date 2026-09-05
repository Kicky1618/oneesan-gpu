#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
WRAP="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stages.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-grand-selector-stages.py"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-stages-ilp2-cg-l2-staged-fullprime-race.sh"
PRE="$ONEESAN_ROOT/scripts/bench/b300-stages-preflight.sh"
for f in "$WRAP" "$GEN" "$PROMOTE" "$PRE"; do [[ -f "$f" ]] || exit 2; done
bash -n "$WRAP"; python3 -m py_compile "$GEN"; bash "$PRE" >/dev/null
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stages-grand.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
STAGES_SELECTOR_BUILD_DIR="$tmp" PATCH_ONLY=1 bash "$WRAP" 27 >"$tmp/env"
# shellcheck disable=SC1090
source "$tmp/env"
[[ "${B300_STAGES_SELECTOR_PATCHED:-0}" == 1 && -s "${B300_STAGES_SELECTOR_GENERATED:-}" ]] || exit 3
G="$B300_STAGES_SELECTOR_GENERATED"; bash -n "$G"
need(){ grep -Fq "$2" "$1" || { echo "Stage-S grand marker missing: $2" >&2; exit 3; }; }
for s in \
  'RUN_STAGES=' 'STAGES_MIN_SPEEDUP=' 'STAGES_PAIR_L2_LIST=' 'STAGES_BLOCK_L2_LIST=' \
  'b300x8-nextgen-hybrid8-stages-ilp2-cg-l2-staged-fullprime-race.sh' 'PREPARE_ONLY=1' \
  'B300_STAGES_PREPARED_CONTROL_BIN" == "$B300_STAGER_PREPARED_BIN' \
  'MODE=stages_ilp2_l2_grand' 'MODE=stages_ilp2_l2_joint' \
  'grand selector: Stage-S ILP2 CG L2 hints rejected/not-applicable; retaining Stage R' \
  'B300_GRAND_STAGES_OK' 'B300_GRAND_STAGES_INTEGRATED=1' 'B300_GRAND_COMPLETE_PRIME_RACES=1'; do need "$G" "$s"; done
python3 - "$G" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('Stage-S grand must retain exactly one complete-prime race')
si=s.find('b300x8-nextgen-hybrid8-stages-ilp2-cg-l2-staged-fullprime-race.sh'); pi=s.find('b300x8-race-external-forced-profiled-once.sh')
if si<0 or pi<0 or si>=pi: raise SystemExit('Stage S must prepare before the one complete-prime race')
if 'if ((STAGER_OK)); then' not in s: raise SystemExit('Stage S must only screen after accepted Stage R')
if '[[ "$B300_STAGES_PREPARED_CONTROL_BIN" == "$B300_STAGER_PREPARED_BIN" ]]' not in s:
    raise SystemExit('Stage-S exact Stage-R control binding missing')
m=re.search(r'if \(\(STAGES_OK && NEXTSELF_OK\)\); then(.*?)elif \(\(STAGES_OK\)\); then',s,re.S)
if not m: raise SystemExit('Stage-S grand branch missing')
b=m.group(1)
for slot,candidate in {'P_BIN':'B300_STAGES_PREPARED_BIN','B_BIN':'B300_STAGES_PREPARED_CONTROL_BIN','E1_BIN':'B300_HYBRID8_PREPARED_BASE_BIN','E2_BIN':'B300_NEXTSELF_PREPARED_BIN','E3_BIN':'JOINT_PRIMARY_BIN'}.items():
    if f'{slot}="${candidate}"' not in b: raise SystemExit(f'Stage-S candidate mapping drift {slot}->{candidate}')
# S must precede R fallback and leave the fixed five forced slots unchanged.
if s.find('if ((STAGES_OK && NEXTSELF_OK)); then') > s.find('elif ((STAGER_OK && NEXTSELF_OK)); then'):
    raise SystemExit('Stage-S priority is not above Stage R')
print('stages_grand_contract_structure=OK')
PY
echo 'b300-grand-stages-contract-preflight OK priority=S,R exact_r_control=1 prepare_only=1 fallback=R forced_slots=5 total_candidates=7 complete_prime_races=1 gpu_work=0'
