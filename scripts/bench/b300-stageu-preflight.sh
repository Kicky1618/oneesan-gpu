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
  'UPSTREAM_KIND must be stager or stages' \
  'ILP2_MATE_CG_L2_BYTES="${ILP2_MATE_CG_L2_BYTES:-128}"' \
  'ILP2_MATE_CG_L2_BYTES must be 0,64,128,256' \
  'b300-forced-nextgen-hybrid8-staget-ilp2-mate-load-policy.sh' \
  'ILP2_MATE_LOAD_POLICY=cg' \
  'accepted_staget_winner_required=0' \
  'immediate_control_lineage=R_or_S' \
  'gen-b300-mainrec-ilp2-mate-cg-l2-policy.py' \
  'stage_u_scope=ilp2_mate_reads_only ilp8_exact_upstream=1 count_policies_preserved=1 count_l2_preserved=1'; do need "$BUILDER" "$s"; done
python3 - "$BUILDER" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
# U must synthesize T-cg from R/S instead of consuming a T winner. This is what
# allows cg@64/128/256 to win even when T's cg@0 loses to default/cs.
for forbidden in ('STAGET_WINNER_ENV','B300_STAGET_STAGED_VALIDATED','B300_STAGET_FINAL_ENABLED','B300_STAGET_PREPARED_BIN'):
    if forbidden in s: raise SystemExit('Stage U incorrectly depends on accepted Stage T: '+forbidden)
t=s.find('b300-forced-nextgen-hybrid8-staget-ilp2-mate-load-policy.sh')
u=s.find('gen-b300-mainrec-ilp2-mate-cg-l2-policy.py')
if min(t,u)<0 or t>=u: raise SystemExit('Stage U must compose T-cg before applying the L2 transform')
if s.count('ILP2_MATE_LOAD_POLICY=cg') != 1: raise SystemExit('Stage U synthetic cg policy count drift')
if 'UPSTREAM_KIND="$UPSTREAM_KIND"' not in s: raise SystemExit('Stage U did not forward exact R/S lineage to synthetic T-cg')
print('stageu_direct_lineage_contract=OK')
PY
echo 'b300-stageu-preflight OK stage=U search_semantics=joint_cg_l2_from_exact_S_or_R synthetic_t_cg=1 accepted_t_not_required=1 blindspot_cg0_loss_recovered=1 low_l2=0,64,128,256 ilp8_locked=1 count_locked=1 gpu_work=0'
