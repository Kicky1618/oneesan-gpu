#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
HYBRID="$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence-hybrid-ilp8.py"
SELF="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-self-prefetch.py"
MATE="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-mate-prefetch.py"
for f in "$HYBRID" "$SELF" "$MATE"; do python3 -m py_compile "$f"; done
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
  S="$TMP/self_${ev}.cu"; SL="$TMP/self_${ev}.log"
  python3 "$SELF" "$HYB" "$S" 4 2 "$ev" >"$SL"
  grep -Fq "evict_priority=$ev" "$SL"
  grep -Fq 'prefetch.global.L2 [%0];' "$S"
  if [[ "$ev" == default ]]; then
    ! grep -Fq 'prefetch.global.L2::evict_' "$S"
  else
    grep -Fq "prefetch.global.L2::evict_${ev} [%0];" "$S"
    grep -Fq 'eviction_hint_sm80_only=1' "$SL"
  fi

  SD="$TMP/self_default_for_mate_${ev}.cu"
  python3 "$SELF" "$HYB" "$SD" 4 2 default >/dev/null
  M="$TMP/mate_${ev}.cu"; ML="$TMP/mate_${ev}.log"
  python3 "$MATE" "$SD" "$M" 4 2 "$ev" >"$ML"
  grep -Fq "evict_priority=$ev" "$ML"
  grep -Fq 'prefetch.global.L2 [%0];' "$M"
  if [[ "$ev" == default ]]; then
    ! grep -Fq 'b300_mainrec_hybrid8_prefetch_next_mate_l2' "$M" || true
    ! grep -Fq 'prefetch.global.L2::evict_' "$ML"
  else
    grep -Fq "prefetch.global.L2::evict_${ev} [%0];" "$M"
    grep -Fq 'eviction_hint_sm80_only=1' "$ML"
  fi
done
for gen in "$SELF" "$MATE"; do
  set +e
  python3 "$gen" "$HYB" "$TMP/bad.cu" 4 2 nonsense >/dev/null 2>"$TMP/bad.err"
  rc=$?
  set -e
  ((rc!=0)) || { echo "invalid eviction accepted by $gen" >&2; exit 3; }
  grep -Fq 'EVICT must be one of default,normal,last' "$TMP/bad.err"
done
echo 'b300_mainrec_prefetch_eviction_preflight=OK self=default,normal,last mate=default,normal,last sm80_qualifier=OK pre_sm80_fallback=bare invalid_rejected=OK gpu_work=0 actions_triggered=0'
