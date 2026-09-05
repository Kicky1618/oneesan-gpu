#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

HYBRID="$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence-hybrid-ilp8.py"
CG="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-random-cg.py"
PREFETCH="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-prefetch-l2.py"
NEXTSELF="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-self-prefetch.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh"
for f in "$HYBRID" "$CG" "$PREFETCH" "$NEXTSELF"; do python3 -m py_compile "$f"; done

for s in \
  'RECURRENCE_HYBRID_ILP8_NEXTSELF' \
  'RECURRENCE_HYBRID_ILP8_NEXTSELF_WIDTH' \
  'gen-b300-mainrec-hybrid8-next-self-prefetch.py' \
  'recurrence_hybrid_ilp8_nextself=' \
  'recurrence_hybrid_ilp8_nextself_width=' \
  'prefetch_width=' \
  'hybrid8_nextself'; do
  grep -Fq "$s" "$BUILDER" || { echo "builder hybrid8-nextself marker missing: $s" >&2; exit 2; }
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
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
__global__ void main_pull_kernel_ilp2(
    const Count* in,MateID*,Code n,const Count* in_block,Code,Count*,int){
    const Code i0=0,i1=n>1?1:0;
    Code pj0=0,pj1=0,bj0=0,bj1=0;
    bool hp0=false,hp1=false,hb0=false,hb1=false;
    const Count pair0=hp0?in[pj0]:Count(0);
    const Count pair1=hp1?in[pj1]:Count(0);
    const Count block0=hb0?in_block[bj0]:Count(0);
    const Count block1=hb1?in_block[bj1]:Count(0);
    const Count self0=in[i0];
    const Count self1=in[i1];
    (void)pair0;(void)pair1;(void)block0;(void)block1;(void)self0;(void)self1;
}
void production_launch(){
    if(useMate)main_pull_kernel_ilp2<<<b300_main_pull_ilp2_blocks(ms.size,threads),threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p);
}

static Code rank_full(MateID m,int width){return Code(m)+Code(width);}
CU

HYB="$TMP/hybrid.cu"
python3 "$HYBRID" "$BASE" "$HYB" 1048576 >"$TMP/hybrid.log"
grep -Fq 'b300_main_recurrence_hybrid_ilp8=1' "$TMP/hybrid.log"

test_chain(){
  local name="$1" use_cg="$2" l2="$3" use_prefetch="$4"
  local cur="$HYB" next
  if [[ "$use_cg" == 1 ]]; then
    next="$TMP/${name}.cg.cu"
    python3 "$CG" "$cur" "$next" "$l2" >"$TMP/${name}.cg.log"
    grep -Fq 'hybrid_policy_consistent=1' "$TMP/${name}.cg.log"
    cur="$next"
  fi
  if [[ "$use_prefetch" == 1 ]]; then
    next="$TMP/${name}.pre.cu"
    python3 "$PREFETCH" "$cur" "$next" >"$TMP/${name}.pre.log"
    grep -Fq 'hybrid_policy_consistent=1' "$TMP/${name}.pre.log"
    cur="$next"
  fi
  local out="$TMP/${name}.nextself.cu" log="$TMP/${name}.nextself.log"
  python3 "$NEXTSELF" "$cur" "$out" >"$log"
  grep -Fq 'b300_mainrec_hybrid8_next_self_prefetch=1' "$log"
  grep -Fq 'prefetch_before_current_reduction=1' "$log"
  grep -Fq 'next_iteration_self_prefetches_per_thread=8' "$log"
  grep -Fq 'prefetch_width=8' "$log"
  grep -Fq 'b300_mainrec_hybrid8_prefetch_next_self_l2' "$out"
  grep -Fq 'const Code next_base=base+Code(8)*grid' "$out"
  grep -Fq 'b300_mainrec_hybrid8_prefetch_next_self_l2(ni7<n?in+ni7:in,ni7<n)' "$out"
  grep -Fq 'const uint64_t mod=D_MOD;' "$out"
  [[ "$(grep -Fc 'b300_mainrec_hybrid8_prefetch_next_self_l2(ni7<n?in+ni7:in,ni7<n)' "$out")" -eq 1 ]]
  if [[ "$use_cg" == 1 ]]; then
    grep -Fq 'b300_mainrec_random_load_cg(in+pj7)' "$out"
    if [[ "$l2" == 0 ]]; then grep -Fq 'ld.global.cg.u32' "$out"; else grep -Fq "ld.global.cg.L2::${l2}B.u32" "$out"; fi
  fi
  if [[ "$use_prefetch" == 1 ]]; then
    grep -Fq 'b300_mainrec_prefetch_l2(in+pj7,hp7)' "$out"
  fi
  python3 - "$out" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
a=s.find('b300_mainrec_hybrid8_prefetch_next_self_l2(ni7<n?in+ni7:in,ni7<n);')
b=s.find('const uint64_t mod=D_MOD;',a)
if not (a>=0 and b>a): raise SystemExit('next-self prefetch must remain before current reduction')
PY
}

test_width(){
  local width="$1" out="$TMP/width${width}.cu" log="$TMP/width${width}.log" last=$((width-1))
  python3 "$NEXTSELF" "$HYB" "$out" "$width" >"$log"
  grep -Fq "next_iteration_self_prefetches_per_thread=$width" "$log"
  grep -Fq "prefetch_width=$width" "$log"
  grep -Fq "b300_mainrec_hybrid8_prefetch_next_self_l2(ni${last}<n?in+ni${last}:in,ni${last}<n)" "$out"
  if (( width < 8 )); then
    ! grep -Fq "const Code ni${width}=" "$out"
    ! grep -Fq "b300_mainrec_hybrid8_prefetch_next_self_l2(ni${width}<" "$out"
  fi
  count="$(grep -Ec '^[[:space:]]*b300_mainrec_hybrid8_prefetch_next_self_l2\(ni[0-7]<' "$out")"
  [[ "$count" == "$width" ]] || { echo "width=$width emitted prefetch count=$count" >&2; exit 3; }
}

test_chain plain 0 0 0
test_chain cg 1 0 0
test_chain cgl2 1 128 0
test_chain generic_prefetch 0 0 1
test_chain cgl2_generic_prefetch 1 128 1
for width in 1 2 4 8; do test_width "$width"; done

python3 "$NEXTSELF" "$HYB" "$TMP/once.cu" >/dev/null
set +e
python3 "$NEXTSELF" "$TMP/once.cu" "$TMP/twice.cu" >"$TMP/twice.out" 2>"$TMP/twice.err"
rc=$?
set -e
((rc!=0)) || { echo 'double hybrid8 next-self transform unexpectedly accepted' >&2; exit 3; }
grep -Fq 'source already contains hybrid8 next-self prefetch' "$TMP/twice.err"

set +e
python3 "$NEXTSELF" "$BASE" "$TMP/nohybrid.cu" >"$TMP/nohybrid.out" 2>"$TMP/nohybrid.err"
rc=$?
set -e
((rc!=0)) || { echo 'next-self transform unexpectedly accepted non-hybrid source' >&2; exit 3; }
grep -Fq 'hybrid8 next-self prefetch requires artifact' "$TMP/nohybrid.err"

for bad in 0 3 5 16; do
  set +e
  python3 "$NEXTSELF" "$HYB" "$TMP/bad${bad}.cu" "$bad" >"$TMP/bad${bad}.out" 2>"$TMP/bad${bad}.err"
  rc=$?
  set -e
  ((rc!=0)) || { echo "invalid width=$bad unexpectedly accepted" >&2; exit 3; }
  grep -Fq 'WIDTH must be one of 1,2,4,8' "$TMP/bad${bad}.err"
done

echo 'b300-mainrec-hybrid8-nextself-transform-preflight OK python_ast=1 chains=5 widths=1,2,4,8 exact_prefetch_count=1 invalid_width_rejected=1 ilp2_fixture=production_load_shape cg_compatible=1 cgl2_compatible=1 generic_prefetch_compatible=1 ordering=1 double_transform_rejected=1 nonhybrid_rejected=1 gpu_work=0'
