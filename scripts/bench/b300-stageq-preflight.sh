#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp8-pair-block-cg-l2-policy.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stageq-ilp8-count-cg-l2-policy.sh"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilp8-pair-block-cg-l2-policy-preflight.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stageq-ilp8-count-cg-l2-sweep.sh"
python3 -m py_compile "$GEN"
for f in "$BUILDER" "$PROOF" "$SWEEP"; do [[ -f "$f" ]] || { echo "missing Stage-Q dependency=$f" >&2; exit 2; }; bash -n "$f"; done
bash "$PROOF"
need(){ local x="$1"; grep -Fq "$x" "$BUILDER" || { echo "Stage-Q builder marker missing: $x" >&2; exit 3; }; }
for x in \
  'UPSTREAM_KIND="${UPSTREAM_KIND:-stagen}"' \
  'STAGEP_COUNT_UPSTREAM="${STAGEP_COUNT_UPSTREAM:-stagen}"' \
  'UPSTREAM_KIND must be stagen, stageo, or stagep' \
  'b300-forced-nextgen-hybrid8-pair-block-load-policy.sh' \
  'b300-forced-nextgen-hybrid8-stageo-cg-l2-policy.sh' \
  'b300-forced-nextgen-hybrid8-stagep-mate-cg-l2-policy.sh' \
  'gen-b300-mainrec-ilp8-pair-block-cg-l2-policy.py' \
  'stage_q_scope=ilp8_count_cg_l2_only' \
  'ilp2_exact_upstream=1' \
  'mate_policy_preserved=1'; do need "$x"; done
python3 - "$GEN" "$BUILDER" "$SWEEP" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text(); b=Path(sys.argv[2]).read_text(); w=Path(sys.argv[3]).read_text()
for q in ('main_pull_kernel_ilp2','main_pull_kernel_ilp8_hybrid','b300_mainrec_stageq_ilp8_pair_load_cg','b300_mainrec_stageq_ilp8_block_load_cg'):
    if q not in s: raise SystemExit('Stage-Q generator missing '+q)
if "if s[ilp2_start2:ilp2_end2] != ilp2_before" not in s:
    raise SystemExit('Stage-Q generator does not byte-lock ILP2')
for up in ('stagen)','stageo)','stagep)'):
    if up not in b: raise SystemExit('Stage-Q builder missing upstream branch '+up)
if b.find('b300-forced-nextgen-hybrid8-stagep-mate-cg-l2-policy.sh') > b.find('gen-b300-mainrec-ilp8-pair-block-cg-l2-policy.py'):
    raise SystemExit('Stage Q must apply after optional Stage P')
if 'PAIR_CG_L2_BYTES=0; BLOCK_CG_L2_BYTES=0' not in b:
    raise SystemExit('Stage-Q direct Stage-N path must neutralize Stage-O overrides')
for q in (
    'UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"',
    'B300_STAGEP_STAGED_VALIDATED=1',
    'B300_STAGEO_STAGED_VALIDATED=1',
    'printf \'control\\t%s\\t%s\\t%s\\t-\\n\'',
    '[[ "$pl2" == "$EFF_PAIR_L2" && "$bl2" == "$EFF_BLOCK_L2" ]] && continue',
    'PAIR_L2_LIST omits exact upstream',
    'BLOCK_L2_LIST omits exact upstream',
    'sha256sum -c "$B300_STAGEN_PREPARED_MANIFEST"',
    'sha256sum -c "$B300_STAGEO_PREPARED_MANIFEST"',
    'sha256sum -c "$B300_STAGEP_PREPARED_MANIFEST"',
    "if len(res)!=1: raise SystemExit('FATAL Stage-Q residue mismatch",
    "clean=len(rv)>=2 and ss==0 and sl==0",
    "'B300_STAGEQ_UPSTREAM_MANIFEST':up_manifest",
):
    if q not in w: raise SystemExit('Stage-Q sweep missing contract '+q)
pos_o=w.find('O_VALID=0'); pos_p=w.find('P_VALID=0')
if pos_o<0 or pos_p<0 or pos_o>=pos_p:
    raise SystemExit('Stage-Q sweep must resolve O before allowing P provenance checks')
if w.find('if [[ "$UPSTREAM_KIND" == stagep || ( "$UPSTREAM_KIND" == auto && "$P_VALID" == 1 ) ]]')<0:
    raise SystemExit('Stage-Q auto path does not prefer valid Stage P')
print('stageq_contract_structure=OK sweep_exact_control=1 auto_p_o_n=1 residue_gate=1 spill_gate=1')
PY
echo 'b300_stageq_preflight=OK stage_q=ilp8_count_cg_l2 upstream=stagen_or_stageo_or_stagep auto_priority=P_O_N exact_upstream_control=1 ilp2_exact_upstream=1 stagep_preserved=1 pair_block_independent=1 residue_gate=1 ptxas_spill=1 sizes=0,64,128,256 gpu_work=0'
