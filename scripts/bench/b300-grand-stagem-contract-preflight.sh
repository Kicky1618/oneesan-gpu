#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

GRAND="$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh"
PROMOTE_M="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-mate-load-stagem-staged-fullprime-race.sh"
PRE_M="$ONEESAN_ROOT/scripts/bench/b300-stagem-preflight.sh"
FIRST="$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh"
for f in "$GRAND" "$PROMOTE_M" "$PRE_M" "$FIRST"; do [[ -f "$f" ]] || exit 2; bash -n "$f"; done
bash "$PRE_M"

need(){ local f="$1" s="$2"; grep -Fq "$s" "$f" || { echo "Stage-M grand marker missing in $f: $s" >&2; exit 3; }; }
for s in \
  'STAGEM_PREFIX=' \
  'STAGEM_WINNER_ENV=' \
  'STAGEM_PREPARE_ENV=' \
  'RUN_STAGEM=' \
  'STAGEM_MIN_SPEEDUP=' \
  'STAGEM_POLICY_LIST=' \
  'b300x8-nextgen-hybrid8-mate-load-stagem-staged-fullprime-race.sh' \
  'PREPARE_ONLY=1' \
  'B300_STAGEM_PREPARED' \
  'B300_STAGEM_PREPARED_BIN' \
  'B300_STAGEM_PREPARED_CONTROL_BIN' \
  'Stage-M mate-load policy rejected; retaining Stage L' \
  'MODE=stagem_mateload_grand' \
  'MODE=stagem_mateload_joint' \
  'B300_GRAND_STAGEM_OK' \
  'B300_GRAND_STAGEM_POLICY'; do need "$GRAND" "$s"; done

python3 - "$GRAND" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('grand selector must still own exactly one complete-prime race')
mi=s.find('b300x8-nextgen-hybrid8-mate-load-stagem-staged-fullprime-race.sh')
ri=s.find('b300x8-race-external-forced-profiled-once.sh')
if mi < 0 or ri < 0 or mi >= ri:
    raise SystemExit('Stage M must be prepared before the one complete-prime race')
# Stage M is a refinement of Stage L, so its control is L. Keep M + L/control
# as the two principal forced candidates instead of consuming another slot.
for marker in ('P_BIN="$B300_STAGEM_PREPARED_BIN"','B_BIN="$B300_STAGEM_PREPARED_CONTROL_BIN"'):
    if marker not in s: raise SystemExit('Stage-M candidate replacement missing: '+marker)
if 'STAGEM_OK && NEXTSELF_OK' not in s or 'elif ((STAGEM_OK))' not in s:
    raise SystemExit('Stage-M grand/joint branch pair missing')
print('stagem_single_prime_contract=OK')
PY

for s in 'RUN_STAGEM=' 'stagem_min_speedup=' 'stagem_policy_list=' 'stagem_accepted='; do need "$FIRST" "$s"; done

echo 'b300-grand-stagem-contract-preflight OK stage_m_after_l=1 prepare_only=1 fallback_l=1 forced_slots_replaced=1 single_complete_prime=1 canonical_firstpass=1 gpu_work=0'
