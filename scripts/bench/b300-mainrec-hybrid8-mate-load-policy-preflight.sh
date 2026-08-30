#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

HYBRID="$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence-hybrid-ilp8.py"
SELF="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-self-prefetch.py"
MATE="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-mate-prefetch.py"
LOAD="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-mate-load-policy.py"
for f in "$HYBRID" "$SELF" "$MATE" "$LOAD"; do python3 -m py_compile "$f"; done
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
BASE="$TMP/base.cu"
cat >"$BASE" <<'CU'
using Code = unsigned long long;
using Count = unsigned int;
using MateID = unsigned long long;
// b300_main_pull_prepare
// b300_low_window_cache_active
// b300_high_main_state_active
// b300_low_state_advance
// b300_high_state_advance
// high_rec_groups=
static int b300_main_pull_ilp2_blocks(Code,int){return 1;}
__global__ void main_pull_kernel_ilp2(const Count*,MateID*,Code,const Count*,Code,Count*,int){}
void launch(){if(useMate)main_pull_kernel_ilp2<<<b300_main_pull_ilp2_blocks(ms.size,threads),threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p);}

static Code rank_full(MateID m,int width){return Code(m)+Code(width);}
CU
HYB="$TMP/hybrid.cu"; SELF_SRC="$TMP/self.cu"; MATE_SRC="$TMP/mate.cu"
python3 "$HYBRID" "$BASE" "$HYB" 1048576 >/dev/null
python3 "$SELF" "$HYB" "$SELF_SRC" 4 2 last >/dev/null
python3 "$MATE" "$SELF_SRC" "$MATE_SRC" 2 1 normal >/dev/null

grep -Fq 'b300_mainrec_hybrid8_prefetch_next_self_l2' "$MATE_SRC"
grep -Fq 'b300_mainrec_hybrid8_prefetch_next_mate_l2' "$MATE_SRC"
for policy in cg cs; do
  OUT="$TMP/${policy}.cu"; LOG="$TMP/${policy}.log"
  python3 "$LOAD" "$MATE_SRC" "$OUT" "$policy" >"$LOG"
  intrinsic=__ldcg; [[ "$policy" == cs ]] && intrinsic=__ldcs
  grep -Fq "b300_mainrec_hybrid8_mate_load_policy=1 policy=$policy intrinsic=$intrinsic" "$LOG"
  grep -Fq "return $intrinsic(p);" "$OUT"
  [[ "$(grep -Fc "b300_mainrec_hybrid8_mate_load_policy_${policy}(mates+i" "$OUT")" == 8 ]]
  ! grep -Fq 'const MateID m0=mates[i0];' "$OUT"
  grep -Fq 'main_pull_kernel_ilp2' "$OUT"
  grep -Fq 'mates[i7]=b300_high_state_advance' "$OUT"
  grep -Fq 'b300_mainrec_hybrid8_prefetch_next_self_l2' "$OUT"
  grep -Fq 'b300_mainrec_hybrid8_prefetch_next_mate_l2' "$OUT"
  python3 - "$OUT" "$policy" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text(); p=sys.argv[2]
loads=[s.find(f'b300_mainrec_hybrid8_mate_load_policy_{p}(mates+i{k})') for k in range(8)]
pair=s.find('const Count pair0=')
write=s.find('mates[i0]=b300_low_state_advance')
if min(loads)<0 or not max(loads)<pair<write:
    raise SystemExit('mate policy load ordering proof failed')
PY
  set +e
  python3 "$LOAD" "$OUT" "$TMP/double_${policy}.cu" "$policy" >/dev/null 2>"$TMP/double_${policy}.err"; rc=$?
  set -e
  ((rc!=0)); grep -Fq 'source already contains hybrid8 mate-load policy' "$TMP/double_${policy}.err"
done
set +e
python3 "$LOAD" "$MATE_SRC" "$TMP/bad.cu" ca >/dev/null 2>"$TMP/bad.err"; rc=$?
set -e
((rc!=0)); grep -Fq 'POLICY must be cg or cs' "$TMP/bad.err"

echo 'b300-mainrec-hybrid8-mate-load-policy-preflight OK policies=cg,cs lanes=8 scope=ilp8_mate_reads_only self_prefetch_preserved=1 mate_prefetch_preserved=1 mate_writes_unchanged=1 double_transform_rejected=1 invalid_policy_rejected=1 gpu_work=0'
