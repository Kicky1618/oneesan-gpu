#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-pair-block-cg-l2-policy.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stages-ilp2-cg-l2-policy.sh"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilp2-pair-block-cg-l2-policy-preflight.sh"
[[ -f "$GEN" && -f "$BUILDER" && -f "$PROOF" ]] || exit 2
python3 -m py_compile "$GEN"; bash -n "$BUILDER"; bash -n "$PROOF"; bash "$PROOF"
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
  'ILP2_PAIR_CG_L2_BYTES must be 0 when ILP2 pair policy is not cg' \
  'ILP2_BLOCK_CG_L2_BYTES must be 0 when ILP2 block policy is not cg' \
  'ilp2_pair_load_policy=$ILP2_PAIR_LOAD_POLICY ilp2_block_load_policy=$ILP2_BLOCK_LOAD_POLICY' \
  'ilp8_pair_cg_l2_bytes=$ILP8_PAIR_CG_L2_BYTES ilp8_block_cg_l2_bytes=$ILP8_BLOCK_CG_L2_BYTES' \
  'gen-b300-mainrec-ilp2-pair-block-cg-l2-policy.py' \
  'stage_s_scope=ilp2_cg_l2_hint_only ilp8_exact_upstream=1 stage_r_policy_preserved=1'; do
  grep -Fq "$s" "$BUILDER" || { echo "Stage-S builder marker missing: $s" >&2; exit 3; }
done
python3 - "$GEN" "$BUILDER" <<'PY'
from pathlib import Path
import sys
g=Path(sys.argv[1]).read_text(); b=Path(sys.argv[2]).read_text()
if "if s[new8_start:new8_end] != ilp8_before" not in g: raise SystemExit('Stage-S ILP8 byte lock missing')
if "high_pair_l2={high_pair_l2} high_block_l2={high_block_l2}" not in g: raise SystemExit('Stage-S high-state L2 provenance missing')
if "stageq_preserved={stageq}" not in g: raise SystemExit('Stage-S Stage-Q provenance missing')
if "ld.global.cg.L2::{v}B.u32" not in g: raise SystemExit('Stage-S explicit L2 qualifier missing')
r=b.find('b300-forced-nextgen-hybrid8-stager-ilp2-load-policy.sh'); s=b.find('gen-b300-mainrec-ilp2-pair-block-cg-l2-policy.py')
if r<0 or s<0 or r>=s: raise SystemExit('Stage S must compose after exact Stage R reconstruction')
if b.count('gen-b300-mainrec-ilp2-pair-block-cg-l2-policy.py') != 1: raise SystemExit('Stage-S transform count drift')
print('stages_contract_structure=OK')
PY
echo 'b300_stages_preflight=OK stage_s=ilp2_pair_block_cg_l2_split upstream=R low_l2=0,64,128,256 noncg_zero_locked=1 stage_r_policy_preserved=1 high_state_locked=1 stageq_provenance=1 ilp8_byte_locked=1 gpu_work=0'
