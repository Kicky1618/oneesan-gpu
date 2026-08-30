#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-pair-block-cg-l2-policy.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stages-ilp2-cg-l2-policy.sh"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilp2-pair-block-cg-l2-policy-preflight.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stages-ilp2-cg-l2-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stages-ilp2-cg-l2-staged-calibrate.sh"
[[ -f "$GEN" && -f "$BUILDER" && -f "$PROOF" && -f "$SWEEP" && -f "$STAGED" ]] || exit 2
python3 -m py_compile "$GEN"; for f in "$BUILDER" "$PROOF" "$SWEEP" "$STAGED"; do bash -n "$f"; done; bash "$PROOF"
for s in \
  'Stage S requires Stage-R ILP2 pair/block policy marker' \
  'PAIR_L2_BYTES must be 0 when Stage-R pair policy is not cg' \
  'BLOCK_L2_BYTES must be 0 when Stage-R block policy is not cg' \
  'Stage S is not applicable when neither Stage-R ILP2 axis uses cg' \
  'Stage S changed ILP8 high-state kernel' \
  'b300_mainrec_stages_ilp2_pair_load_cg' \
  'b300_mainrec_stages_ilp2_block_load_cg' \
  'b300_mainrec_stages_ilp2_pair_block_cg_l2=1'; do
  grep -Fq "$s" "$GEN" || { echo "Stage-S generator marker missing: $s" >&2; exit 3; }
done
for s in \
  'b300-forced-nextgen-hybrid8-stager-ilp2-load-policy.sh' \
  'ILP2_PAIR_CG_L2_BYTES="${ILP2_PAIR_CG_L2_BYTES:-0}"' \
  'ILP2_BLOCK_CG_L2_BYTES="${ILP2_BLOCK_CG_L2_BYTES:-0}"' \
  'Stage S requires at least one ILP2 cg axis' \
  'EFFECTIVE_HIGH_PAIR_L2=0; EFFECTIVE_HIGH_BLOCK_L2=0' \
  'stageq) EFFECTIVE_HIGH_PAIR_L2="$ILP8_PAIR_CG_L2_BYTES"' \
  'stageo) EFFECTIVE_HIGH_PAIR_L2="$PAIR_CG_L2_BYTES"' \
  'STAGEP_COUNT_UPSTREAM" == stageo' \
  'stagen) EFFECTIVE_HIGH_PAIR_L2="$BASE_CG_L2_BYTES"' \
  'gen-b300-mainrec-ilp2-pair-block-cg-l2-policy.py' \
  'high_pair_l2=$EFFECTIVE_HIGH_PAIR_L2 high_block_l2=$EFFECTIVE_HIGH_BLOCK_L2' \
  'stage_s_scope=ilp2_cg_l2_hint_only ilp8_exact_upstream=1 stage_r_policy_preserved=1'; do
  grep -Fq "$s" "$BUILDER" || { echo "Stage-S builder marker missing: $s" >&2; exit 3; }
done
for s in \
  'CONTROL_BIN="$B300_STAGER_PREPARED_BIN"' \
  "printf 'control\\t0\\t0\\t%s\\t-\\n' \"\$CONTROL_BIN\"" \
  'Stage S not applicable: Stage-R selected no ILP2 cg axis' \
  'PAIR_L2_LIST must include exact Stage-R baseline 0' \
  'BLOCK_L2_LIST must include exact Stage-R baseline 0' \
  'b300-forced-nextgen-hybrid8-stages-ilp2-cg-l2-policy.sh' \
  'main_pull_kernel_ilp2' \
  'main_pull_kernel_ilp8_hybrid' \
  'clean=len(rv)>=2 and ss==0 and sl==0' \
  'FATAL Stage-S/Stage-R residue mismatch' \
  'b300_stages_exact_match=1'; do
  grep -Fq "$s" "$SWEEP" || { echo "Stage-S sweep marker missing: $s" >&2; exit 3; }
done
for s in \
  'SEARCH_ROWS="${SEARCH_ROWS:-1}"' \
  'VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"' \
  'run_stage "$SEARCH_ROWS"' \
  'validation_pair=0' \
  'validation_block=0' \
  'B300_STAGES_STAGED_VALIDATED=' \
  'B300_STAGES_FINAL_ENABLED=' \
  'B300_STAGES_FINAL_SPILL_FREE=1' \
  'FATAL Stage-S/Stage-R residue mismatch'; do
  grep -Fq "$s" "$STAGED" || { echo "Stage-S staged marker missing: $s" >&2; exit 3; }
done
python3 - "$GEN" "$BUILDER" "$SWEEP" "$STAGED" <<'PY'
from pathlib import Path
import sys
g,b,w,t=(Path(p).read_text() for p in sys.argv[1:])
if "if s[new8_start:new8_end] != ilp8_before" not in g: raise SystemExit('Stage-S ILP8 byte lock missing')
if "high_pair_l2={high_pair_l2} high_block_l2={high_block_l2}" not in g: raise SystemExit('Stage-S high-state L2 provenance missing')
if "stageq_preserved={stageq}" not in g: raise SystemExit('Stage-S Stage-Q provenance missing')
if "ld.global.cg.L2::{v}B.u32" not in g: raise SystemExit('Stage-S explicit L2 qualifier missing')
r=b.find('b300-forced-nextgen-hybrid8-stager-ilp2-load-policy.sh'); s=b.find('gen-b300-mainrec-ilp2-pair-block-cg-l2-policy.py')
if r<0 or s<0 or r>=s: raise SystemExit('Stage S must compose after exact Stage R reconstruction')
if b.count('gen-b300-mainrec-ilp2-pair-block-cg-l2-policy.py') != 1: raise SystemExit('Stage-S transform count drift')
# The sweep must benchmark the exact R prepared binary, not a rebuilt S 0/0 candidate.
if '[[ "$pl2" == 0 && "$bl2" == 0 ]] && continue' not in w: raise SystemExit('Stage-S exact control tuple is rebuilt')
if 'CONTROL_BIN="$B300_STAGER_PREPARED_BIN"' not in w: raise SystemExit('Stage-S exact prepared R control missing')
# Later validation narrows the 4x4 search to control plus the row-1 winning coordinates.
for x in ('validation_pair=0','validation_block=0','[[ "$SELECTED_PAIR" == 0 ]] || validation_pair+=','[[ "$SELECTED_BLOCK" == 0 ]] || validation_block+='):
    if x not in t: raise SystemExit('Stage-S staged tuple lock missing '+x)
print('stages_staged_contract_structure=OK')
PY
echo 'b300_stages_preflight=OK stage_s=ilp2_pair_block_cg_l2_split upstream=R exact_prepared_control=1 low_l2=0,64,128,256 noncg_zero_locked=1 high_provenance=Q,O,P,N stage_r_policy_preserved=1 high_state_locked=1 ilp8_byte_locked=1 residue_gate=1 spill_gate=1 staged_rows=1,4,8 selected_tuple_lock=1 gpu_work=0'
