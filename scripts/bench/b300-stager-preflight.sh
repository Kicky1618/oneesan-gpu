#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-pair-block-load-policy.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stager-ilp2-load-policy.sh"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilp2-pair-block-load-policy-preflight.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stager-ilp2-load-policy-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stager-ilp2-load-policy-staged-calibrate.sh"
python3 -m py_compile "$GEN"
for f in "$BUILDER" "$PROOF" "$SWEEP" "$STAGED"; do [[ -f "$f" ]] || exit 2; bash -n "$f"; done
bash "$PROOF"
need(){ local f="$1" s="$2"; grep -Fq "$s" "$f" || { echo "Stage-R marker missing in $f: $s" >&2; exit 3; }; }
for s in \
  'UPSTREAM_KIND="${UPSTREAM_KIND:-stageq}"' \
  'UPSTREAM_KIND must be stagen, stageo, stagep, or stageq' \
  'b300-forced-nextgen-hybrid8-pair-block-load-policy.sh' \
  'b300-forced-nextgen-hybrid8-stageo-cg-l2-policy.sh' \
  'b300-forced-nextgen-hybrid8-stagep-mate-cg-l2-policy.sh' \
  'b300-forced-nextgen-hybrid8-stageq-ilp8-count-cg-l2-policy.sh' \
  'gen-b300-mainrec-ilp2-pair-block-load-policy.py' \
  'stage_r_scope=ilp2_pair_block_load_policy_only' \
  'ilp8_exact_upstream=1' \
  'stageq_preserved=1'; do need "$BUILDER" "$s"; done
for s in \
  'UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"' \
  'PAIR_POLICY_LIST="${PAIR_POLICY_LIST:-default cg cs}"' \
  'BLOCK_POLICY_LIST="${BLOCK_POLICY_LIST:-default cg cs}"' \
  'RESOLVED=stageq' \
  'CONTROL_BIN="$B300_STAGEQ_PREPARED_BIN"' \
  'CONTROL_BIN="$B300_STAGEP_PREPARED_BIN"' \
  'CONTROL_BIN="$B300_STAGEO_PREPARED_BIN"' \
  'CONTROL_BIN="$B300_STAGEN_PREPARED_BIN"' \
  'ILP2_PAIR_LOAD_POLICY="$lp" ILP2_BLOCK_LOAD_POLICY="$lb"' \
  'main_pull_kernel_ilp2' \
  'main_pull_kernel_ilp8_hybrid' \
  'clean=len(rv)>=2 and ss==0 and sl==0' \
  'b300_stager_exact_match=1'; do need "$SWEEP" "$s"; done
for s in \
  'SEARCH_ROWS="${SEARCH_ROWS:-1}"' \
  'VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"' \
  'run_stage "$SEARCH_ROWS"' \
  'validation_pair="$BASE_PAIR"' \
  'validation_block="$BASE_BLOCK"' \
  'B300_STAGER_STAGED_VALIDATED=' \
  'B300_STAGER_FINAL_ENABLED=' \
  'B300_STAGER_FINAL_SPILL_FREE=1' \
  'FATAL Stage-R/upstream residue mismatch'; do need "$STAGED" "$s"; done
python3 - "$BUILDER" "$GEN" "$SWEEP" "$STAGED" <<'PY'
from pathlib import Path
import sys
b,g,w,t=map(lambda p:Path(p).read_text(),sys.argv[1:])
q=b.find('b300-forced-nextgen-hybrid8-stageq-ilp8-count-cg-l2-policy.sh'); r=b.find('gen-b300-mainrec-ilp2-pair-block-load-policy.py')
if q<0 or r<0 or q>=r: raise SystemExit('Stage R must be applied after Stage Q reconstruction')
for kind in ('stagen)','stageo)','stagep)','stageq)'):
    if kind not in b: raise SystemExit('missing Stage-R upstream branch '+kind)
if "s[new8_start:new8_end] != ilp8_before" not in g: raise SystemExit('Stage-R generator does not byte-lock ILP8')
if 'b300_mainrec_stageq_ilp8_pair_block_cg_l2=1' not in g: raise SystemExit('Stage-R generator does not read Stage-Q provenance')
# Screening must benchmark exact prepared upstreams, never a rebuilt default/default binary.
if "printf 'control\\t%s\\t%s\\t%s\\t-\\n' \"$BASE_PAIR\" \"$BASE_BLOCK\" \"$CONTROL_BIN\"" not in w:
    raise SystemExit('Stage-R exact prepared control missing')
if '[[ "$lp" == "$BASE_PAIR" && "$lb" == "$BASE_BLOCK" ]] && continue' not in w:
    raise SystemExit('Stage-R rebuilds exact control tuple')
# Validation may test only the exact control and the row-1 winner.
for x in ('validation_pair="$BASE_PAIR"','validation_block="$BASE_BLOCK"','[[ "$SELECTED_PAIR" == "$BASE_PAIR" ]] || validation_pair+=','[[ "$SELECTED_BLOCK" == "$BASE_BLOCK" ]] || validation_block+='):
    if x not in t: raise SystemExit('Stage-R staged lock missing '+x)
print('stager_staged_contract_structure=OK')
PY
echo 'b300_stager_preflight=OK stage_r=ilp2_pair_block_load_split upstream_priority=Q,P,O,N exact_prepared_control=1 stageq_provenance=1 ilp8_byte_locked=1 pair_policies=default,cg,cs block_policies=default,cg,cs residue_gate=1 spill_gate=1 staged_rows=1,4,8 selected_tuple_lock=1 gpu_work=0'
