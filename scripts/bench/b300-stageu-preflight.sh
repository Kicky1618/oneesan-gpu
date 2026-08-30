#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-mate-cg-l2-policy.py"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilp2-mate-cg-l2-stageu-preflight.sh"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stageu-ilp2-mate-cg-l2-policy.sh"
for f in "$GEN" "$PROOF" "$BUILDER"; do [[ -s "$f" ]] || exit 2; done
python3 -m py_compile "$GEN"; bash -n "$PROOF"; bash -n "$BUILDER"; bash "$PROOF"
need(){ grep -Fq "$2" "$1" || { echo "Stage-U contract missing in $1: $2" >&2; exit 3; }; }
for s in \
  'UPSTREAM_KIND="${UPSTREAM_KIND:-stages}"' \
  'STAGER_UPSTREAM_KIND="${STAGER_UPSTREAM_KIND:-stageq}"' \
  'ILP2_MATE_CG_L2_BYTES="${ILP2_MATE_CG_L2_BYTES:-128}"' \
  'ILP2_MATE_CG_L2_BYTES must be 0,64,128,256' \
  'b300-forced-nextgen-hybrid8-staget-ilp2-mate-load-policy.sh' \
  'ILP2_MATE_LOAD_POLICY=cg' \
  'accepted_staget_winner_required=0' \
  'immediate_control_lineage=R_or_S' \
  'Stage-U high mate lineage drift in synthetic T-cg' \
  'Stage-U Count policy lineage drift in synthetic T-cg' \
  'Stage-U Count L2 lineage drift in synthetic T-cg' \
  'gen-b300-mainrec-ilp2-mate-cg-l2-policy.py' \
  'stage_u_scope=ilp2_mate_reads_only ilp8_exact_upstream=1 count_policies_preserved=1 count_l2_preserved=1 geometry_preserved=1'; do need "$BUILDER" "$s"; done
python3 - "$BUILDER" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
# U synthesizes T-cg from R/S instead of consuming a T winner. This recovers
# cg@L2 candidates even when T's cg@0 loses, but every other lineage coordinate
# must be forwarded exactly so U changes only low mate policy+L2 jointly.
for forbidden in ('STAGET_WINNER_ENV','B300_STAGET_STAGED_VALIDATED','B300_STAGET_FINAL_ENABLED','B300_STAGET_PREPARED_BIN'):
    if forbidden in s: raise SystemExit('Stage U incorrectly depends on accepted Stage T: '+forbidden)
t=s.find('b300-forced-nextgen-hybrid8-staget-ilp2-mate-load-policy.sh')
u=s.find('gen-b300-mainrec-ilp2-mate-cg-l2-policy.py')
if min(t,u)<0 or t>=u: raise SystemExit('Stage U must compose T-cg before applying the L2 transform')
if s.count('ILP2_MATE_LOAD_POLICY=cg') != 1: raise SystemExit('Stage U synthetic cg policy count drift')
a=s.find('env N=27 ARCH="$ARCH" OUT="$T_BIN"')
b=s.find('bash "$TBUILDER"',a)
if a<0 or b<0: raise SystemExit('Stage U T-builder invocation not found')
call=s[a:b]
required=(
 'UPSTREAM_KIND','STAGER_UPSTREAM_KIND','STAGEQ_UPSTREAM_KIND','STAGEP_COUNT_UPSTREAM',
 'MATE_LOAD_POLICY','PAIR_LOAD_POLICY','BLOCK_LOAD_POLICY','ILP2_PAIR_LOAD_POLICY','ILP2_BLOCK_LOAD_POLICY',
 'ILP2_PAIR_CG_L2_BYTES','ILP2_BLOCK_CG_L2_BYTES','BASE_CG_L2_BYTES','PAIR_CG_L2_BYTES','BLOCK_CG_L2_BYTES',
 'MATE_CG_L2_BYTES','ILP8_PAIR_CG_L2_BYTES','ILP8_BLOCK_CG_L2_BYTES','HIGH_DROP_CHUNK','HYBRID_THRESHOLD',
 'SELF_WIDTH','SELF_DISTANCE','MATE_WIDTH','MATE_DISTANCE','SELF_EVICT','MATE_EVICT','SELF_GUARD','MATE_GUARD',
 'RANDOM_CG','RANDOM_CG_L2_FETCH_BYTES','PREFETCH_L2','DUALMASK','CLOSURE_BATCH','MAXRREGCOUNT')
missing=[x for x in required if f'{x}="${x}"' not in call]
if missing: raise SystemExit('Stage U failed to forward exact R/S lineage knobs: '+','.join(missing))
if 'ILP2_MATE_LOAD_POLICY=cg' not in call: raise SystemExit('Stage U did not force synthetic T-cg')
print('stageu_direct_lineage_contract=OK full_lineage_forwarded=1')
PY
echo 'b300-stageu-preflight OK stage=U search_semantics=joint_cg_l2_from_exact_S_or_R synthetic_t_cg=1 accepted_t_not_required=1 blindspot_cg0_loss_recovered=1 full_lineage_forwarded=1 low_l2=0,64,128,256 ilp8_locked=1 count_locked=1 geometry_locked=1 gpu_work=0'
