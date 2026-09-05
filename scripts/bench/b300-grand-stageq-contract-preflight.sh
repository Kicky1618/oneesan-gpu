#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
WRAP="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stageq.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-grand-selector-stageq.py"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-stageq-ilp8-count-cg-l2-staged-fullprime-race.sh"
PRE="$ONEESAN_ROOT/scripts/bench/b300-stageq-preflight.sh"
for f in "$WRAP" "$GEN" "$PROMOTE" "$PRE"; do [[ -f "$f" ]] || { echo "missing Stage-Q grand dependency=$f" >&2; exit 2; }; done
bash -n "$WRAP"; bash -n "$PROMOTE"; python3 -m py_compile "$GEN"
bash "$PRE" >/dev/null

tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stageq-grand.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
STAGEQ_SELECTOR_BUILD_DIR="$tmp" PATCH_ONLY=1 bash "$WRAP" 27 >"$tmp/env"
# shellcheck disable=SC1090
source "$tmp/env"
[[ "${B300_STAGEQ_SELECTOR_PATCHED:-0}" == 1 && -s "${B300_STAGEQ_SELECTOR_GENERATED:-}" ]] || exit 3
G="$B300_STAGEQ_SELECTOR_GENERATED"
bash -n "$G"
need(){ grep -Fq "$2" "$1" || { echo "Stage-Q grand marker missing: $2" >&2; exit 3; }; }
for s in \
  'RUN_STAGEQ=' 'STAGEQ_MIN_SPEEDUP=' 'STAGEQ_PAIR_L2_LIST=' 'STAGEQ_BLOCK_L2_LIST=' \
  'b300x8-nextgen-hybrid8-stageq-ilp8-count-cg-l2-staged-fullprime-race.sh' 'PREPARE_ONLY=1' \
  'STAGEQ_UPSTREAM_KIND=stagep' 'STAGEQ_UPSTREAM_KIND=stageo' 'STAGEQ_UPSTREAM_KIND=stagen' \
  'MODE=stageq_countl2_grand' 'MODE=stageq_countl2_joint' \
  'B300_STAGEQ_PREPARED_CONTROL_BIN' 'B300_GRAND_STAGEQ_OK' 'B300_GRAND_STAGEQ_INTEGRATED=1' 'B300_GRAND_COMPLETE_PRIME_RACES=1'; do need "$G" "$s"; done
python3 - "$G" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('Stage-Q grand must retain exactly one complete-prime race')
qi=s.find('b300x8-nextgen-hybrid8-stageq-ilp8-count-cg-l2-staged-fullprime-race.sh')
ri=s.find('b300x8-race-external-forced-profiled-once.sh')
if qi < 0 or ri < 0 or qi >= ri: raise SystemExit('Stage-Q preparation must precede complete-prime race')
if not re.search(r'if \(\(STAGEP_OK\)\); then STAGEQ_UPSTREAM_KIND=stagep\s*\nelif \(\(STAGEO_OK\)\); then STAGEQ_UPSTREAM_KIND=stageo\s*\nelif \(\(STAGEN_OK\)\); then STAGEQ_UPSTREAM_KIND=stagen',s):
    raise SystemExit('Stage-Q upstream priority must be P > O > N')
m=re.search(r'if \(\(STAGEQ_OK && NEXTSELF_OK\)\); then(.*?)elif \(\(STAGEQ_OK\)\); then',s,re.S)
if not m: raise SystemExit('Stage-Q grand branch missing')
b=m.group(1)
for slot,candidate in {'P_BIN':'B300_STAGEQ_PREPARED_BIN','B_BIN':'B300_STAGEQ_PREPARED_CONTROL_BIN','E1_BIN':'B300_HYBRID8_PREPARED_BASE_BIN','E2_BIN':'B300_NEXTSELF_PREPARED_BIN','E3_BIN':'JOINT_PRIMARY_BIN'}.items():
    if f'{slot}="${candidate}"' not in b: raise SystemExit(f'Stage-Q candidate mapping drift {slot}->{candidate}')
for kind,var in [('stagep','B300_STAGEP_PREPARED_BIN'),('stageo','B300_STAGEO_PREPARED_BIN'),('stagen','B300_STAGEN_PREPARED_BIN')]:
    if f'[[ "$B300_STAGEQ_PREPARED_CONTROL_BIN" == "${var}" ]]' not in s: raise SystemExit(f'Stage-Q exact {kind} control binding missing')
print('stageq_grand_contract_structure=OK')
PY
echo 'b300-grand-stageq-contract-preflight OK upstream_priority=P,O,N exact_control=1 prepare_only=1 fallback=1 forced_slots=5 complete_prime_races=1 gpu_work=0'
