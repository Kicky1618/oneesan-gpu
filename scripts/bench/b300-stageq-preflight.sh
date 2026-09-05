#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp8-pair-block-cg-l2-policy.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stageq-ilp8-count-cg-l2-policy.sh"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilp8-pair-block-cg-l2-policy-preflight.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stageq-ilp8-count-cg-l2-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-stageq-ilp8-count-cg-l2-staged-calibrate.sh"
PROMOTE="$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-stageq-ilp8-count-cg-l2-staged-fullprime-race.sh"
python3 -m py_compile "$GEN"
for f in "$BUILDER" "$PROOF" "$SWEEP" "$STAGED" "$PROMOTE"; do [[ -f "$f" ]] || { echo "missing Stage-Q dependency=$f" >&2; exit 2; }; bash -n "$f"; done
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
python3 - "$GEN" "$BUILDER" "$SWEEP" "$STAGED" "$PROMOTE" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text(); b=Path(sys.argv[2]).read_text(); w=Path(sys.argv[3]).read_text(); t=Path(sys.argv[4]).read_text(); p=Path(sys.argv[5]).read_text()
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
    'UPSTREAM_KIND="${UPSTREAM_KIND:-auto}"', 'B300_STAGEP_STAGED_VALIDATED=1', 'B300_STAGEO_STAGED_VALIDATED=1',
    'printf \'control\\t%s\\t%s\\t%s\\t-\\n\'', '[[ "$pl2" == "$EFF_PAIR_L2" && "$bl2" == "$EFF_BLOCK_L2" ]] && continue',
    'PAIR_L2_LIST omits exact upstream', 'BLOCK_L2_LIST omits exact upstream',
    'sha256sum -c "$B300_STAGEN_PREPARED_MANIFEST"', 'sha256sum -c "$B300_STAGEO_PREPARED_MANIFEST"', 'sha256sum -c "$B300_STAGEP_PREPARED_MANIFEST"',
    "if len(res)!=1: raise SystemExit('FATAL Stage-Q residue mismatch", "clean=len(rv)>=2 and ss==0 and sl==0", "'B300_STAGEQ_UPSTREAM_MANIFEST':up_manifest",
):
    if q not in w: raise SystemExit('Stage-Q sweep missing contract '+q)
pos_o=w.find('O_VALID=0'); pos_p=w.find('P_VALID=0')
if pos_o<0 or pos_p<0 or pos_o>=pos_p: raise SystemExit('Stage-Q sweep must resolve O before P provenance checks')
if w.find('if [[ "$UPSTREAM_KIND" == stagep || ( "$UPSTREAM_KIND" == auto && "$P_VALID" == 1 ) ]]')<0: raise SystemExit('Stage-Q auto path does not prefer valid Stage P')
for q in (
    'SEARCH_ROWS="${SEARCH_ROWS:-1}"', 'VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"', 'RESOLVED_UPSTREAM="$B300_STAGEQ_UPSTREAM_KIND"',
    'UP_PAIR="$B300_STAGEQ_UPSTREAM_PAIR_L2_BYTES"', 'UP_BLOCK="$B300_STAGEQ_UPSTREAM_BLOCK_L2_BYTES"', 'UP_MANIFEST="$B300_STAGEQ_UPSTREAM_MANIFEST"',
    '[[ "$B300_STAGEQ_UPSTREAM_MANIFEST" == "$UP_MANIFEST" ]]', 'FATAL Stage-Q/upstream residue mismatch',
    'validation_pair="$UP_PAIR"', 'validation_block="$UP_BLOCK"',
    '[[ "$B300_STAGEQ_PAIR_L2_BYTES" != "$SELECTED_PAIR" || "$B300_STAGEQ_BLOCK_L2_BYTES" != "$SELECTED_BLOCK"',
    'FINAL_PAIR="$UP_PAIR"; FINAL_BLOCK="$UP_BLOCK"; FINAL_BIN="$B300_STAGEQ_CONTROL_BIN"',
    'B300_STAGEQ_STAGED_VALIDATED=', 'B300_STAGEQ_FINAL_ENABLED=', 'B300_STAGEQ_FINAL_SPILL_FREE=1',
):
    if q not in t: raise SystemExit('Stage-Q staged calibration missing contract '+q)
if t.find('for rows in $VALIDATE_ROWS "$UP_ROWS"')<0: raise SystemExit('Stage-Q staged validation does not include upstream row')
for q in (
    'RUN_STAGED="${RUN_STAGED:-1}"', 'PREPARE_ONLY="${PREPARE_ONLY:-0}"',
    'b300-nextgen-hybrid8-stageq-ilp8-count-cg-l2-staged-calibrate.sh',
    'Stage Q did not survive staged validation', 'Stage-Q winner equals exact upstream L2 tuple', 'Stage-Q speedup below threshold',
    'sha256sum -c "$B300_STAGEQ_UPSTREAM_MANIFEST"', 'case "$RESOLVED" in', 'stagen)', 'stageo)', 'stagep)',
    'Stage-Q control is not exact Stage N', 'Stage-Q control is not exact Stage O', 'Stage-Q control is not exact Stage P',
    'if [[ "$RESOLVED" == stageo || ( "$RESOLVED" == stagep && "$B300_STAGEQ_STAGEP_COUNT_UPSTREAM" == stageo ) ]]',
    'sha256sum -c "$MANIFEST"', 'B300_STAGEQ_PREPARED=1', 'B300_STAGEQ_PREPARED_UPSTREAM_MANIFEST=',
    'B300_STAGEQ_PREPARED_PAIR_L2_BYTES=', 'B300_STAGEQ_PREPARED_BLOCK_L2_BYTES=', 'B300_STAGEQ_PREPARED_MANIFEST=',
    'Stage-Q complete-prime promotion requires NGPU=8', 'b300x8-race-external-forced-profiled-once.sh',
):
    if q not in p: raise SystemExit('Stage-Q promotion missing contract '+q)
if p.count('b300x8-race-external-forced-profiled-once.sh') != 1:
    raise SystemExit('Stage-Q standalone promotion must contain one complete-prime race call')
if p.find('if [[ "$PREPARE_ONLY" == 1 ]]') > p.find('b300x8-race-external-forced-profiled-once.sh'):
    raise SystemExit('Stage-Q PREPARE_ONLY gate must precede complete-prime race')
print('stageq_contract_structure=OK sweep_exact_control=1 auto_p_o_n=1 residue_gate=1 spill_gate=1 staged_1_4_8=1 selected_tuple_survival=1 fallback_upstream=1 promotion_manifest=1 prepare_only=1 standalone_prime_races=1')
PY
echo 'b300_stageq_preflight=OK stage_q=ilp8_count_cg_l2 upstream=stagen_or_stageo_or_stagep auto_priority=P_O_N exact_upstream_control=1 ilp2_exact_upstream=1 stagep_preserved=1 pair_block_independent=1 residue_gate=1 ptxas_spill=1 staged_rows=1_4_8 selected_tuple_survival=1 fallback_upstream=1 promotion_manifest=1 prepare_only=1 standalone_prime_races=1 sizes=0,64,128,256 gpu_work=0'
