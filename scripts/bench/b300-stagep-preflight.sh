#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-mate-cg-l2-policy.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stagep-mate-cg-l2-policy.sh"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid8-mate-cg-l2-policy-preflight.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stagep-mate-cg-l2-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stagep-mate-cg-l2-staged-calibrate.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-stagep-mate-cg-l2-staged-fullprime-race.sh"
python3 -m py_compile "$GEN"
for f in "$BUILDER" "$PROOF" "$SWEEP" "$STAGED" "$PROMOTE"; do [[ -f "$f" ]] || { echo "missing Stage-P dependency=$f" >&2; exit 2; }; bash -n "$f"; done
bash "$PROOF"
need(){ local f="$1" s="$2"; grep -Fq "$s" "$f" || { echo "Stage-P marker missing in $f: $s" >&2; exit 3; }; }
for s in 'COUNT_UPSTREAM="${COUNT_UPSTREAM:-stagen}"' 'MATE_LOAD_POLICY="${MATE_LOAD_POLICY:-cg}"' 'MATE_CG_L2_BYTES="${MATE_CG_L2_BYTES:-0}"' 'COUNT_UPSTREAM must be stagen or stageo' 'Stage P requires MATE_LOAD_POLICY=cg' 'b300-forced-nextgen-hybrid8-pair-block-load-policy.sh' 'gen-b300-mainrec-pair-block-cg-l2-policy.py' 'gen-b300-mainrec-hybrid8-mate-cg-l2-policy.py' 'stage_p_scope=ilp8_mate_cg_l2_only' 'count_policy_preserved=1' 'mate_writes_unchanged=1'; do need "$BUILDER" "$s"; done
for s in 'UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"' 'MATE_L2_LIST="${MATE_L2_LIST:-0 64 128 256}"' 'MATE_L2_LIST must include inherited 0B baseline' 'Stage P not applicable: selected Stage N does not inherit Stage-M mate cg' 'Stage-N manifest mismatch before Stage P' 'Stage-O manifest mismatch before Stage P' 'B300_STAGEP_CONTROL_BIN' 'B300_STAGEP_MATE_L2_BYTES' 'B300_STAGEP_COUNT_UPSTREAM' 'FATAL Stage-P residue mismatch' 'FATAL Stage-P/upstream residue mismatch' 'b300_stagep_exact_match=1'; do need "$SWEEP" "$s"; done
for s in 'SEARCH_ROWS="${SEARCH_ROWS:-1}"' 'VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"' 'RESOLVED_UPSTREAM="$B300_STAGEP_COUNT_UPSTREAM"' 'SELECTED_L2="$B300_STAGEP_MATE_L2_BYTES"' 'validation_l2="0"' 'FATAL Stage-P/upstream residue mismatch' 'B300_STAGEP_STAGED_VALIDATED=' 'B300_STAGEP_FINAL_ENABLED=' 'B300_STAGEP_FINAL_STAGE_RESIDUE='; do need "$STAGED" "$s"; done
for s in 'sha256sum -c "$B300_STAGEN_PREPARED_MANIFEST"' 'sha256sum -c "$B300_STAGEO_PREPARED_MANIFEST"' 'Stage-P exact control/baseline contract failed' 'B300_STAGEP_PREPARED_UPSTREAM_MANIFEST' 'B300_STAGEP_PREPARED_MATE_L2_BYTES' 'B300_STAGEP_PREPARED_MANIFEST' 'PREPARE_ONLY' 'b300x8-race-external-forced-profiled-once.sh'; do need "$PROMOTE" "$s"; done
python3 - "$GEN" "$BUILDER" "$SWEEP" "$STAGED" "$PROMOTE" <<'PY'
from pathlib import Path
import sys
g,b,s,w,p=map(lambda x:Path(x).read_text(),sys.argv[1:])
for q in ('ld.global.cg.u64','ld.global.cg.L2::{l2_bytes}B.u64'):
    if q not in g: raise SystemExit('Stage-P generator missing qualifier path '+q)
if "s.count(f'{helper}(mates+i{k})')!=1" not in g: raise SystemExit('Stage-P generator does not lock all eight mate lanes')
if 'if [[ "$COUNT_UPSTREAM" == stageo ]]' not in b or 'PAIR_CG_L2_BYTES=0; BLOCK_CG_L2_BYTES=0' not in b: raise SystemExit('Stage-P count-upstream composition incomplete')
if b.find('gen-b300-mainrec-pair-block-cg-l2-policy.py') > b.find('gen-b300-mainrec-hybrid8-mate-cg-l2-policy.py'): raise SystemExit('Stage-P must apply Count Stage O before mate Stage P')
if "printf 'control\\t0\\t%s\\t-\\n' \"$CONTROL_BIN\"" not in s: raise SystemExit('Stage-P sweep must use exact upstream binary as 0B control')
if 'run_stage "$rows" "$RESOLVED_UPSTREAM" "$validation_l2"' not in w: raise SystemExit('Stage-P validation must lock upstream and selected mate L2')
if p.count('b300x8-race-external-forced-profiled-once.sh') != 1: raise SystemExit('Stage-P promotion must expose exactly one optional full-prime race')
if '[[ "$B300_STAGEP_CONTROL_BIN" == "$UP_BIN"' not in p: raise SystemExit('Stage-P promotion must bind control to exact upstream')
print('stagep_contract_structure=OK')
PY
echo 'b300_stagep_preflight=OK stage_p=mate_cg_l2 sizes=0,64,128,256 count_upstream=stagen_or_stageo exact_upstream_control=1 inherited_0b_baseline=1 stageo_preserved=1 stagen_preserved=1 ilp8_mate_only=1 row_scoped_validation=1 upstream_locked=1 residue_gate=1 ptxas_spill=1 promotion_manifest=1 single_optional_fullprime=1 gpu_work=0'
