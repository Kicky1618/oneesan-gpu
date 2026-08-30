#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-mate-load-policy.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-staget-ilp2-mate-load-policy.sh"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilp2-mate-load-policy-staget-preflight.sh"
python3 -m py_compile "$GEN"; for f in "$BUILDER" "$PROOF"; do [[ -f "$f" ]] || exit 2; bash -n "$f"; done; bash "$PROOF"
need(){ grep -Fq "$2" "$1" || { echo "Stage-T marker missing in $1: $2" >&2; exit 3; }; }
for s in \
  'Stage T requires Stage-R ILP2 pair/block policy marker' \
  'Stage T changed ILP8 high-state kernel' \
  'b300_mainrec_staget_ilp2_mate_load_policy_cg' \
  'b300_mainrec_staget_ilp2_mate_load_policy_cs' \
  'high_policy=' \
  'high_l2_bytes=' \
  'stages_preserved='; do need "$GEN" "$s"; done
for s in \
  'UPSTREAM_KIND="${UPSTREAM_KIND:-stages}"' \
  'UPSTREAM_KIND must be stager or stages' \
  'b300-forced-nextgen-hybrid8-stager-ilp2-load-policy.sh' \
  'b300-forced-nextgen-hybrid8-stages-ilp2-cg-l2-policy.sh' \
  'ILP2_MATE_LOAD_POLICY="${ILP2_MATE_LOAD_POLICY:-cg}"' \
  'ILP2_MATE_LOAD_POLICY must be cg or cs' \
  'gen-b300-mainrec-ilp2-mate-load-policy.py' \
  'high_policy=$MATE_LOAD_POLICY high_l2_bytes=$EXPECTED_HIGH_L2' \
  'stage_t_scope=ilp2_mate_reads_only ilp8_exact_upstream=1 count_policies_preserved=1 count_l2_preserved=1'; do need "$BUILDER" "$s"; done
python3 - "$GEN" "$BUILDER" <<'PY'
from pathlib import Path
import sys
g,b=(Path(p).read_text() for p in sys.argv[1:])
if "if s[new8_start:new8_end]!=ilp8_before" not in g: raise SystemExit('Stage-T ILP8 byte lock missing')
if "found[k]" not in g or "needle=f'mates[i{k}]'" not in g: raise SystemExit('Stage-T lane-local rewrite proof missing')
r=b.find('b300-forced-nextgen-hybrid8-stager-ilp2-load-policy.sh'); s=b.find('b300-forced-nextgen-hybrid8-stages-ilp2-cg-l2-policy.sh'); t=b.find('gen-b300-mainrec-ilp2-mate-load-policy.py')
if min(r,s,t)<0 or r>=t or s>=t: raise SystemExit('Stage T must compose strictly after R/S reconstruction')
if b.count('gen-b300-mainrec-ilp2-mate-load-policy.py') != 1: raise SystemExit('Stage-T transform count drift')
if '[[ "$UPSTREAM_KIND" == stages ]]' not in b: raise SystemExit('Stage-T optional Stage-S branch missing')
print('staget_composition_contract_structure=OK')
PY
echo 'b300_staget_preflight=OK stage_t=ilp2_mate_load_split upstream_priority=S,R low_mate_policies=cg,cs high_mate_policy=M high_mate_l2=P stage_s_optional=1 stage_r_required=1 ilp8_byte_locked=1 count_policy_locked=1 count_l2_locked=1 gpu_work=0'
