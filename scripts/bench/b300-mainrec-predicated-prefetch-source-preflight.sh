#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
HYBRID="$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence-hybrid-ilp8.py"
SELF="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-self-prefetch.py"
MATE="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-mate-prefetch.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-self-mate-geometry.sh"
for f in "$HYBRID" "$SELF" "$MATE"; do python3 -m py_compile "$f"; done
bash -n "$BUILDER"
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
HYB="$TMP/hybrid.cu"; python3 "$HYBRID" "$BASE" "$HYB" 1048576 >/dev/null
for ev in default normal last; do
  SB="$TMP/self_b_${ev}.cu"; SP="$TMP/self_p_${ev}.cu"
  python3 "$SELF" "$HYB" "$SB" 4 2 "$ev" branch >"$TMP/self_b_${ev}.log"
  python3 "$SELF" "$HYB" "$SP" 4 2 "$ev" predicated >"$TMP/self_p_${ev}.log"
  grep -Fq 'guard_mode=branch ptx_predicated_guard=0' "$TMP/self_b_${ev}.log"
  grep -Fq 'guard_mode=predicated ptx_predicated_guard=1' "$TMP/self_p_${ev}.log"
  grep -Fq 'b300_mainrec_hybrid8_prefetch_next_self_l2(ni0<n?in+ni0:in,ni0<n)' "$SB"
  ! grep -Fq '@p prefetch.global.L2' "$SB"
  grep -Fq 'b300_mainrec_hybrid8_prefetch_next_self_l2(in,ni0,ni0<n)' "$SP"
  ! grep -Fq '?in+ni' "$SP"
  grep -Fq 'setp.ne.u32 p, %2, 0' "$SP"
  grep -Fq 'add.u64 a, %0, %1' "$SP"
  grep -Fq '@p prefetch.global.L2' "$SP"

  MB="$TMP/mate_b_${ev}.cu"; MP="$TMP/mate_p_${ev}.cu"
  python3 "$MATE" "$SB" "$MB" 2 1 "$ev" branch >"$TMP/mate_b_${ev}.log"
  python3 "$MATE" "$SP" "$MP" 2 1 "$ev" predicated >"$TMP/mate_p_${ev}.log"
  grep -Fq 'guard_mode=branch ptx_predicated_guard=0' "$TMP/mate_b_${ev}.log"
  grep -Fq 'guard_mode=predicated ptx_predicated_guard=1' "$TMP/mate_p_${ev}.log"
  grep -Fq 'b300_mainrec_hybrid8_prefetch_next_mate_l2(nmi0<n?mates+nmi0:mates,nmi0<n)' "$MB"
  ! grep -Fq '@p prefetch.global.L2' "$MB"
  grep -Fq 'b300_mainrec_hybrid8_prefetch_next_mate_l2(mates,nmi0,nmi0<n)' "$MP"
  ! grep -Fq '?mates+nmi' "$MP"
  grep -Fq 'setp.ne.u32 p, %2, 0' "$MP"
  grep -Fq 'add.u64 a, %0, %1' "$MP"
  grep -Fq '@p prefetch.global.L2' "$MP"
done
for gen in "$SELF" "$MATE"; do
  set +e
  python3 "$gen" "$HYB" "$TMP/bad.cu" 4 2 default nonsense >"$TMP/bad.out" 2>"$TMP/bad.err"
  rc=$?
  set -e
  ((rc!=0)) || { echo "invalid guard accepted by $gen" >&2; exit 3; }
  grep -Fq 'GUARD must be one of branch,predicated' "$TMP/bad.err"
done
for s in 'SELF_GUARD="${SELF_GUARD:-branch}"' 'MATE_GUARD="${MATE_GUARD:-branch}"' 'guard=$SELF_GUARD' 'guard=$MATE_GUARD'; do
  grep -Fq "$s" "$BUILDER" || { echo "builder guard marker missing: $s" >&2; exit 3; }
done
echo 'b300_mainrec_predicated_prefetch_source_preflight=OK self=branch,predicated mate=branch,predicated evict=default,normal,last no_oob_cpp_pointer=OK ptx_setp=OK ptx_atp=OK defaults_unchanged=branch gpu_work=0 actions_triggered=0'
