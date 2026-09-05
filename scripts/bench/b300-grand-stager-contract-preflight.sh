#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
WRAP="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stager.sh"; GEN="$ONEESAN_ROOT/scripts/build/gen-b300-grand-selector-stager.py"; PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-stager-ilp2-load-policy-staged-fullprime-race.sh"; PRE="$ONEESAN_ROOT/scripts/bench/b300-stager-preflight.sh"
for f in "$WRAP" "$GEN" "$PROMOTE" "$PRE"; do [[ -f "$f" ]] || exit 2; done
bash -n "$WRAP"; python3 -m py_compile "$GEN"; bash "$PRE" >/dev/null
tmp="$(mktemp -d "${TMPDIR:-/tmp}/oneesan-stager-grand.XXXXXX")"; trap 'rm -rf "$tmp"' EXIT
STAGER_SELECTOR_BUILD_DIR="$tmp" PATCH_ONLY=1 bash "$WRAP" 27 >"$tmp/env"; source "$tmp/env"
[[ "${B300_STAGER_SELECTOR_PATCHED:-0}" == 1 && -s "${B300_STAGER_SELECTOR_GENERATED:-}" ]] || exit 3
G="$B300_STAGER_SELECTOR_GENERATED"; bash -n "$G"
need(){ grep -Fq "$2" "$1" || { echo "Stage-R grand marker missing: $2" >&2; exit 3; }; }
for s in 'RUN_STAGER=' 'STAGER_MIN_SPEEDUP=' 'STAGER_PAIR_POLICY_LIST=' 'STAGER_BLOCK_POLICY_LIST=' 'b300x8-nextgen-hybrid8-stager-ilp2-load-policy-staged-fullprime-race.sh' 'PREPARE_ONLY=1' 'STAGER_UPSTREAM_KIND=stageq' 'STAGER_UPSTREAM_KIND=stagep' 'STAGER_UPSTREAM_KIND=stageo' 'STAGER_UPSTREAM_KIND=stagen' 'MODE=stager_ilp2_grand' 'MODE=stager_ilp2_joint' 'B300_GRAND_STAGER_OK' 'B300_GRAND_STAGER_INTEGRATED=1' 'B300_GRAND_COMPLETE_PRIME_RACES=1'; do need "$G" "$s"; done
python3 - "$G" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1: raise SystemExit('Stage-R grand must retain exactly one complete-prime race')
ri=s.find('b300x8-nextgen-hybrid8-stager-ilp2-load-policy-staged-fullprime-race.sh'); pi=s.find('b300x8-race-external-forced-profiled-once.sh')
if ri<0 or pi<0 or ri>=pi: raise SystemExit('Stage R must prepare before the one complete-prime race')
if not re.search(r'if \(\(STAGEQ_OK\)\); then STAGER_UPSTREAM_KIND=stageq\s*\nelif \(\(STAGEP_OK\)\); then STAGER_UPSTREAM_KIND=stagep\s*\nelif \(\(STAGEO_OK\)\); then STAGER_UPSTREAM_KIND=stageo\s*\nelif \(\(STAGEN_OK\)\); then STAGER_UPSTREAM_KIND=stagen',s): raise SystemExit('Stage-R upstream priority must be Q > P > O > N')
for kind,var in [('stageq','B300_STAGEQ_PREPARED_BIN'),('stagep','B300_STAGEP_PREPARED_BIN'),('stageo','B300_STAGEO_PREPARED_BIN'),('stagen','B300_STAGEN_PREPARED_BIN')]:
    if f'[[ "$B300_STAGER_PREPARED_CONTROL_BIN" == "${var}" ]]' not in s: raise SystemExit(f'Stage-R exact {kind} control binding missing')
m=re.search(r'if \(\(STAGER_OK && NEXTSELF_OK\)\); then(.*?)elif \(\(STAGER_OK\)\); then',s,re.S)
if not m: raise SystemExit('Stage-R grand branch missing')
b=m.group(1)
for slot,candidate in {'P_BIN':'B300_STAGER_PREPARED_BIN','B_BIN':'B300_STAGER_PREPARED_CONTROL_BIN','E1_BIN':'B300_HYBRID8_PREPARED_BASE_BIN','E2_BIN':'B300_NEXTSELF_PREPARED_BIN','E3_BIN':'JOINT_PRIMARY_BIN'}.items():
    if f'{slot}="${candidate}"' not in b: raise SystemExit(f'Stage-R candidate mapping drift {slot}->{candidate}')
print('stager_grand_contract_structure=OK')
PY
echo 'b300-grand-stager-contract-preflight OK upstream_priority=Q,P,O,N exact_control=1 prepare_only=1 fallback=1 forced_slots=5 complete_prime_races=1 gpu_work=0'
