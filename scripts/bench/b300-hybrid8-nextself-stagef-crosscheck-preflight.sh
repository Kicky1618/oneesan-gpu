#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

stage="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextself-staged-calibrate.sh"
[[ -f "$stage" ]] || { echo "missing $stage" >&2; exit 2; }
bash -n "$stage"

for s in \
  'B300_HYBRID8_CORE_ROWS' \
  'B300_HYBRID8_RESIDUE' \
  'B300_HYBRID8_FINAL_STAGE_ROWS' \
  'B300_HYBRID8_FINAL_STAGE_RESIDUE' \
  'stage_validate_rows=()' \
  'for rows in $VALIDATE_ROWS "$B300_HYBRID8_FINAL_STAGE_ROWS"' \
  'check_stage_e_residue(){' \
  'FATAL Stage-F/core Stage-E residue mismatch' \
  'FATAL Stage-F/final Stage-E residue mismatch' \
  'check_stage_e_residue "$SEARCH_ROWS" "$B300_HYBRID8_NEXTSELF_RESIDUE"' \
  'check_stage_e_residue "$rows" "$B300_HYBRID8_NEXTSELF_RESIDUE"' \
  'B300_HYBRID8_NEXTSELF_STAGE_E_CORE_ROWS=' \
  'B300_HYBRID8_NEXTSELF_STAGE_E_CORE_RESIDUE=' \
  'B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_ROWS=' \
  'B300_HYBRID8_NEXTSELF_STAGE_E_FINAL_RESIDUE=' \
  'stage_e_crosscheck=1'; do
  grep -Fq "$s" "$stage" || { echo "Stage F cross-check marker missing: $s" >&2; exit 3; }
done

python3 - "$stage" <<'PY'
from pathlib import Path
import re,sys
s=Path(sys.argv[1]).read_text()
# Cross-stage exactness must be tested immediately after sourcing each measured
# Stage-F env, before the speedup gate can promote the candidate.
search=s.find('source "$SEARCH_ENV"')
search_gate=s.find('check_stage_e_residue "$SEARCH_ROWS"', search)
search_perf=s.find('VALIDATED=0', search)
if not (search >= 0 and search < search_gate < search_perf):
    raise SystemExit('search-row Stage-E residue gate is not before promotion logic')
loop=re.search(r'for rows in "\$\{stage_validate_rows\[@\]\}"; do(.*?)done\nfi',s,re.S)
if not loop:
    raise SystemExit('Stage-F validation loop missing')
b=loop.group(1)
source_pos=b.find('source "$CURRENT_ENV"')
gate_pos=b.find('check_stage_e_residue "$rows"')
perf_pos=b.find('B300_HYBRID8_NEXTSELF_BEST_ENABLED')
if not (source_pos >= 0 and source_pos < gate_pos < perf_pos):
    raise SystemExit('validation-row Stage-E residue gate is not before speedup promotion gate')
print('stagef_cross_stage_order=OK search_gate_before_promotion=1 validation_gate_before_promotion=1')
PY

echo 'b300_hybrid8_nextself_stagef_crosscheck_preflight=OK bash_syntax=OK core_residue_gate=OK final_residue_gate=OK final_row_forced=OK gate_order=OK gpu_work=0 actions_triggered=0'
