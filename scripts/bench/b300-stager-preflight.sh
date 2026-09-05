#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-pair-block-load-policy.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stager-ilp2-load-policy.sh"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilp2-pair-block-load-policy-preflight.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stager-ilp2-load-policy-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stager-ilp2-load-policy-staged-calibrate.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-stager-ilp2-load-policy-staged-fullprime-race.sh"
python3 -m py_compile "$GEN"
for f in "$BUILDER" "$PROOF" "$SWEEP" "$STAGED" "$PROMOTE"; do [[ -f "$f" ]] || exit 2; bash -n "$f"; done
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
  'EFF_PAIR_L2=0' \
  'EFF_PAIR_L2="$B300_STAGEO_PREPARED_PAIR_L2_BYTES"' \
  'EFF_PAIR_L2="$Q_PAIR_L2"' \
  'ILP2_PAIR_LOAD_POLICY="$lp" ILP2_BLOCK_LOAD_POLICY="$lb"' \
  'main_pull_kernel_ilp2' \
  'main_pull_kernel_ilp8_hybrid' \
  'clean=len(rv)>=2 and ss==0 and sl==0' \
  'FATAL Stage-R/upstream residue mismatch rows=' \
  'B300_STAGER_HIGH_PAIR_L2_BYTES' \
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
for s in \
  'PREPARE_ONLY="${PREPARE_ONLY:-0}"' \
  'b300-nextgen-hybrid8-stager-ilp2-load-policy-staged-calibrate.sh' \
  'Stage R did not survive staged validation' \
  'Stage R retained exact upstream ILP2 tuple' \
  'Stage-R control is not exact prepared upstream binary' \
  'sha256sum -c "$UP_MAN"' \
  'sha256sum -c "$MANIFEST"' \
  'B300_STAGER_PREPARED=1' \
  'B300_STAGER_PREPARED_CONTROL_BIN=' \
  'B300_STAGER_PREPARED_HIGH_PAIR_L2_BYTES=' \
  'b300x8-race-external-forced-profiled-once.sh'; do need "$PROMOTE" "$s"; done
python3 - "$BUILDER" "$GEN" "$SWEEP" "$STAGED" "$PROMOTE" <<'PY'
from pathlib import Path
import sys
b,g,w,t,p=map(lambda x:Path(x).read_text(),sys.argv[1:])
q=b.find('b300-forced-nextgen-hybrid8-stageq-ilp8-count-cg-l2-policy.sh'); r=b.find('gen-b300-mainrec-ilp2-pair-block-load-policy.py')
if q<0 or r<0 or q>=r: raise SystemExit('Stage R must be applied after Stage Q reconstruction')
for kind in ('stagen)','stageo)','stagep)','stageq)'):
    if kind not in b: raise SystemExit('missing Stage-R upstream branch '+kind)
if "s[new8_start:new8_end] != ilp8_before" not in g: raise SystemExit('Stage-R generator does not byte-lock ILP8')
if 'b300_mainrec_stageq_ilp8_pair_block_cg_l2=1' not in g: raise SystemExit('Stage-R generator does not read Stage-Q provenance')
if "printf 'control\\t%s\\t%s\\t%s\\t-\\n' \"$BASE_PAIR\" \"$BASE_BLOCK\" \"$CONTROL_BIN\"" not in w: raise SystemExit('Stage-R exact prepared control missing')
if '[[ "$lp" == "$BASE_PAIR" && "$lb" == "$BASE_BLOCK" ]] && continue' not in w: raise SystemExit('Stage-R rebuilds exact control tuple')
if '"$EFF_PAIR_L2" "$EFF_BLOCK_L2"' not in w: raise SystemExit('Stage-R winner env is not bound to effective high-state L2')
if "if rows_arg==up_rows and only_res!=up_res:" not in w: raise SystemExit('Stage-R standalone sweep lacks upstream residue gate')
for x in ('validation_pair="$BASE_PAIR"','validation_block="$BASE_BLOCK"','[[ "$SELECTED_PAIR" == "$BASE_PAIR" ]] || validation_pair+=','[[ "$SELECTED_BLOCK" == "$BASE_BLOCK" ]] || validation_block+='):
    if x not in t: raise SystemExit('Stage-R staged lock missing '+x)
# Promotion may expose a standalone race but grand integration must consume PREPARE_ONLY.
if p.count('b300x8-race-external-forced-profiled-once.sh') != 1: raise SystemExit('Stage-R promotion race count drift')
if 'if [[ "$PREPARE_ONLY" == 1 ]]; then' not in p: raise SystemExit('Stage-R PREPARE_ONLY boundary missing')
if p.find('if [[ "$PREPARE_ONLY" == 1 ]]; then') > p.find('b300x8-race-external-forced-profiled-once.sh'): raise SystemExit('Stage-R PREPARE_ONLY must precede complete-prime race')
print('stager_promotion_contract_structure=OK')
PY
echo 'b300_stager_preflight=OK stage_r=ilp2_pair_block_load_split upstream_priority=Q,P,O,N exact_prepared_control=1 effective_high_l2_provenance=1 stageq_provenance=1 ilp8_byte_locked=1 pair_policies=default,cg,cs block_policies=default,cg,cs residue_gate=standalone_and_staged spill_gate=1 staged_rows=1,4,8 selected_tuple_lock=1 promotion_manifest=1 prepare_only=1 standalone_complete_prime=1 gpu_work=0'
