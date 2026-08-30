#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-mate-cg-l2-policy.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stagep-mate-cg-l2-policy.sh"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid8-mate-cg-l2-policy-preflight.sh"
python3 -m py_compile "$GEN"
for f in "$BUILDER" "$PROOF"; do [[ -f "$f" ]] || { echo "missing Stage-P dependency=$f" >&2; exit 2; }; bash -n "$f"; done
bash "$PROOF"
need(){ local s="$1"; grep -Fq "$s" "$BUILDER" || { echo "Stage-P builder marker missing: $s" >&2; exit 3; }; }
for s in \
  'COUNT_UPSTREAM="${COUNT_UPSTREAM:-stagen}"' \
  'MATE_LOAD_POLICY="${MATE_LOAD_POLICY:-cg}"' \
  'MATE_CG_L2_BYTES="${MATE_CG_L2_BYTES:-0}"' \
  'COUNT_UPSTREAM must be stagen or stageo' \
  'Stage P requires MATE_LOAD_POLICY=cg' \
  'b300-forced-nextgen-hybrid8-pair-block-load-policy.sh' \
  'gen-b300-mainrec-pair-block-cg-l2-policy.py' \
  'gen-b300-mainrec-hybrid8-mate-cg-l2-policy.py' \
  'stage_p_scope=ilp8_mate_cg_l2_only' \
  'count_policy_preserved=1' \
  'mate_writes_unchanged=1'; do need "$s"; done
python3 - "$GEN" "$BUILDER" <<'PY'
from pathlib import Path
import sys
src=Path(sys.argv[1]).read_text(); b=Path(sys.argv[2]).read_text()
for q in ('ld.global.cg.u64','ld.global.cg.L2::{l2_bytes}B.u64'):
    if q not in src: raise SystemExit('Stage-P generator missing qualifier path '+q)
if "s.count(f'{helper}(mates+i{k})')!=1" not in src:
    raise SystemExit('Stage-P generator does not lock all eight mate lanes')
if 'if [[ "$COUNT_UPSTREAM" == stageo ]]' not in b:
    raise SystemExit('Stage-P builder has no Stage-O composition path')
if 'PAIR_CG_L2_BYTES=0; BLOCK_CG_L2_BYTES=0' not in b:
    raise SystemExit('Stage-P Stage-N path must neutralize Stage-O L2 overrides')
if b.find('gen-b300-mainrec-pair-block-cg-l2-policy.py') > b.find('gen-b300-mainrec-hybrid8-mate-cg-l2-policy.py'):
    raise SystemExit('Stage-P must apply Count Stage O before mate Stage P')
print('stagep_contract_structure=OK')
PY
echo 'b300_stagep_preflight=OK stage_p=mate_cg_l2 sizes=0,64,128,256 count_upstream=stagen_or_stageo stageo_preserved=1 stagen_preserved=1 ilp8_mate_only=1 ilp2_unchanged=1 count_loads_unchanged=1 mate_writes_unchanged=1 gpu_work=0'
