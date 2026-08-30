#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
RUNNER="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-staget-ilp2-mate-load-policy-staged-fullprime-race.sh"
[[ -s "$RUNNER" ]] || { echo 'missing Stage-T promotion runner' >&2; exit 2; }
bash -n "$RUNNER"
need(){ grep -Fq "$2" "$1" || { echo "Stage-T promotion marker missing: $2" >&2; exit 3; }; }
for s in \
  'UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"' \
  'RUN_STAGED="${RUN_STAGED:-1}"' \
  'PREPARE_ONLY="${PREPARE_ONLY:-0}"' \
  'POLICY_LIST="${POLICY_LIST:-default cg cs}"' \
  'b300-nextgen-hybrid8-staget-ilp2-mate-load-policy-staged-calibrate.sh' \
  'case "$B300_STAGET_POLICY" in cg|cs)' \
  'case "$B300_STAGET_UPSTREAM_KIND" in' \
  'B300_STAGES_PREPARED_BIN' \
  'B300_STAGER_PREPARED_BIN' \
  '[[ "$B300_STAGET_CONTROL_BIN" == "$UP_BIN" ]]' \
  'sha256sum -c "$UP_MAN"' \
  'Stage-T control is not exact prepared immediate upstream binary' \
  'Stage-T changed high Count provenance' \
  'B300_STAGET_PROMOTION_VALIDATED=1' \
  'B300_STAGET_PREPARED=1' \
  'B300_STAGET_PREPARED_CONTROL_BIN' \
  'B300_STAGET_PREPARED_UPSTREAM_MANIFEST' \
  'B300_STAGET_PREPARED_MANIFEST' \
  'Stage-T complete-prime promotion requires NGPU=8' \
  'b300x8-race-external-forced-profiled-once.sh'; do need "$RUNNER" "$s"; done
python3 - "$RUNNER" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
# Exact immediate upstream must be selected before candidate/control binding.
case=s.find('case "$B300_STAGET_UPSTREAM_KIND" in')
ctrl=s.find('[[ "$B300_STAGET_CONTROL_BIN" == "$UP_BIN" ]]')
prep=s.find('if [[ "$PREPARE_ONLY" == 1 ]]')
ngpu=s.find("Stage-T complete-prime promotion requires NGPU=8")
race=s.find('b300x8-race-external-forced-profiled-once.sh')
if min(case,ctrl,prep,ngpu,race) < 0: raise SystemExit('Stage-T promotion ordering anchor missing')
if not (case < ctrl < prep < ngpu < race):
    raise SystemExit('Stage-T promotion ordering drift: upstream/control -> PREPARE_ONLY -> NGPU gate -> race required')
if s.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('Stage-T promotion must contain exactly one external complete-prime race call')
if 'FORCED_OVERRIDE_BIN="$B300_STAGET_FINAL_BIN"' not in s:
    raise SystemExit('Stage-T candidate is not the forced override')
if 'FORCED_BASE_BIN="$B300_STAGET_CONTROL_BIN"' not in s:
    raise SystemExit('Stage-T exact upstream is not the forced base')
# Both possible immediate-upstream manifests must be verified before use.
if s.count('sha256sum -c "$UP_MAN"') != 1:
    raise SystemExit('Stage-T immediate-upstream manifest verification count drift')
print('staget_promotion_contract_structure=OK')
PY
echo 'b300-staget-promotion-preflight OK stage=T exact_upstream=S_or_R default_control=1 candidate_policy=cg_or_cs manifest=1 prepare_before_fullprime=1 ngpu8_gate=1 complete_prime_races=1 gpu_work=0'
