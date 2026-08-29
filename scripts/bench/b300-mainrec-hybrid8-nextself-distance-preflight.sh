#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

HYBRID="$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence-hybrid-ilp8.py"
NEXTSELF="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-self-prefetch.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-nextself-distance.sh"
python3 -m py_compile "$HYBRID" "$NEXTSELF"
bash -n "$BUILDER"
for s in NEXTSELF_WIDTH NEXTSELF_DISTANCE prefetch_distance_iterations canonical_nextgen_proof_gates_reused; do
  grep -Fq "$s" "$BUILDER" || { echo "distance builder marker missing: $s" >&2; exit 2; }
done

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
HYB="$TMP/hybrid.cu"
python3 "$HYBRID" "$BASE" "$HYB" 1048576 >"$TMP/hybrid.log"

tested=0
for d in 1 2 4; do
  advance=$((8*d))
  for w in 1 2 4 8; do
    out="$TMP/w${w}d${d}.cu"; log="$TMP/w${w}d${d}.log"
    python3 "$NEXTSELF" "$HYB" "$out" "$w" "$d" >"$log"
    grep -Fq "prefetch_width=$w" "$log"
    grep -Fq "prefetch_distance_iterations=$d" "$log"
    grep -Fq "const Code next_base=base+Code(${advance})*grid" "$out"
    last=$((w-1))
    grep -Fq "b300_mainrec_hybrid8_prefetch_next_self_l2(ni${last}<n?in+ni${last}:in,ni${last}<n)" "$out"
    if ((w<8)); then ! grep -Fq "const Code ni${w}=" "$out"; fi
    python3 - "$out" "$w" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text(); w=int(sys.argv[2])
a=s.find(f'b300_mainrec_hybrid8_prefetch_next_self_l2(ni{w-1}<n?in+ni{w-1}:in,ni{w-1}<n);')
b=s.find('const uint64_t mod=D_MOD;',a)
if not (a>=0 and b>a): raise SystemExit('prefetch ordering proof failed')
PY
    ((tested+=1))
  done
done
[[ "$tested" == 12 ]]

for bad in 0 3 8; do
  set +e
  python3 "$NEXTSELF" "$HYB" "$TMP/bad${bad}.cu" 4 "$bad" >"$TMP/bad${bad}.out" 2>"$TMP/bad${bad}.err"
  rc=$?
  set -e
  ((rc!=0)) || { echo "bad distance=$bad unexpectedly accepted" >&2; exit 3; }
  grep -Fq 'DISTANCE must be one of 1,2,4' "$TMP/bad${bad}.err"
done

echo 'b300_mainrec_hybrid8_nextself_distance_preflight=OK widths=1,2,4,8 distances=1,2,4 combinations=12 ordering=OK invalid_distance_rejected=OK gpu_work=0 actions_triggered=0'
