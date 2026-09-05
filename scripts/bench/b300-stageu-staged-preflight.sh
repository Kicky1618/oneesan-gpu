#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stageu-ilp2-mate-cg-l2-joint-staged-calibrate.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stageu-ilp2-mate-cg-l2-joint-sweep.sh"
for f in "$STAGED" "$SWEEP"; do [[ -s "$f" ]] || exit 2; bash -n "$f"; done
need(){ grep -Fq "$2" "$1" || { echo "Stage-U staged marker missing: $2" >&2; exit 3; }; }
for s in \
  'SEARCH_ROWS="${SEARCH_ROWS:-1}"' \
  'VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"' \
  'MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"' \
  'L2_LIST="${L2_LIST:-0 64 128 256}"' \
  'SEARCH_ENV="$(run_stage "$SEARCH_ROWS" "$L2_LIST"' \
  'RES_UP="$B300_STAGEU_UPSTREAM_KIND"' \
  'CURRENT_ENV="$(run_stage "$rows" "0 $SELECTED_L2"' \
  '"$RES_UP")"' \
  'B300_STAGEU_JOINT_BEST_LABEL" != "$SELECTED_LABEL"' \
  'B300_STAGEU_JOINT_BEST_POLICY" != cg' \
  'B300_STAGEU_JOINT_BEST_L2_BYTES" != "$SELECTED_L2"' \
  'B300_STAGEU_BEST_ENABLED" != 1' \
  'B300_STAGEU_CONTROL_SPILL_FREE" != 1' \
  'B300_STAGEU_SPILL_FREE" != 1' \
  'FATAL Stage-U/upstream residue mismatch' \
  'B300_STAGEU_UPSTREAM_MANIFEST" == "$UP_MANIFEST"' \
  'B300_STAGEU_STAGED_VALIDATED' \
  'B300_STAGEU_FINAL_ENABLED' \
  'B300_STAGEU_SELECTED_L2_BYTES' \
  'B300_STAGEU_MIN_SPEEDUP'; do need "$STAGED" "$s"; done
python3 - "$STAGED" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
search=s.find('SEARCH_ENV="$(run_stage')
lock=s.find('RES_UP="$B300_STAGEU_UPSTREAM_KIND"')
validate=s.find('CURRENT_ENV="$(run_stage "$rows" "0 $SELECTED_L2"')
if min(search,lock,validate)<0 or not (search<lock<validate):
    raise SystemExit('Stage-U upstream must resolve once at search before validation')
# Validation must keep the T comparators because sweep always adds cg/cs; only the
# U L2 list is narrowed to baseline 0 plus the selected nonzero hint.
if '"0 $SELECTED_L2"' not in s: raise SystemExit('Stage-U selected-L2 validation narrowing missing')
for token in ('B300_STAGEU_JOINT_BEST_LABEL','B300_STAGEU_JOINT_BEST_POLICY','B300_STAGEU_JOINT_BEST_L2_BYTES','B300_STAGEU_BEST_ENABLED'):
    if token not in s: raise SystemExit('Stage-U global-winner validation missing '+token)
if 'for rows in $VALIDATE_ROWS "$UP_ROWS"' not in s:
    raise SystemExit('Stage-U upstream final row validation missing')
if 'check_residue "$rows" "$B300_STAGEU_RESIDUE"' not in s:
    raise SystemExit('Stage-U validation residue check missing')
if '[[ "$B300_STAGEU_UPSTREAM_KIND" == "$RES_UP"' not in s:
    raise SystemExit('Stage-U immediate-upstream lock missing')
if 'B300_STAGEU_UPSTREAM_MANIFEST" == "$UP_MANIFEST"' not in s:
    raise SystemExit('Stage-U upstream manifest lock missing')
print('stageu_staged_contract_structure=OK')
PY
echo 'b300-stageu-staged-preflight OK stage=U search_rows=1 validate_rows=4,8,upstream upstream_locked=1 exact_control=1 t_cg_cs_recompete=1 selected_nonzero_l2_locked=1 global_best_required=1 spill_gate=1 residue_gate=1 manifest_lock=1 gpu_work=0'
