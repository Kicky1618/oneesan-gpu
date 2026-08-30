#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-ilp2-pair-block-load-policy.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-stager-ilp2-load-policy.sh"
PROOF="$ONEESAN_ROOT/scripts/bench/b300-mainrec-ilp2-pair-block-load-policy-preflight.sh"
python3 -m py_compile "$GEN"
for f in "$BUILDER" "$PROOF"; do [[ -f "$f" ]] || exit 2; bash -n "$f"; done
bash "$PROOF"
need(){ grep -Fq "$1" "$BUILDER" || { echo "Stage-R builder marker missing: $1" >&2; exit 3; }; }
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
  'stageq_preserved=1'; do need "$s"; done
python3 - "$BUILDER" "$GEN" <<'PY'
from pathlib import Path
import sys
b=Path(sys.argv[1]).read_text(); g=Path(sys.argv[2]).read_text()
# All upstream reconstructions must occur before Stage R is applied.
q=b.find('b300-forced-nextgen-hybrid8-stageq-ilp8-count-cg-l2-policy.sh')
r=b.find('gen-b300-mainrec-ilp2-pair-block-load-policy.py')
if q<0 or r<0 or q>=r: raise SystemExit('Stage R must be applied after Stage Q reconstruction')
for kind in ('stagen)','stageo)','stagep)','stageq)'):
    if kind not in b: raise SystemExit('missing Stage-R upstream branch '+kind)
if "s[new8_start:new8_end] != ilp8_before" not in g:
    raise SystemExit('Stage-R generator does not byte-lock ILP8')
if 'b300_mainrec_stageq_ilp8_pair_block_cg_l2=1' not in g:
    raise SystemExit('Stage-R generator does not read Stage-Q provenance')
print('stager_contract_structure=OK')
PY
echo 'b300_stager_preflight=OK stage_r=ilp2_pair_block_load_split upstreams=N,O,P,Q stageq_provenance=1 ilp8_byte_locked=1 pair_policies=default,cg,cs block_policies=default,cg,cs gpu_work=0'
