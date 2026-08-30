#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

HYBRID="$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence-hybrid-ilp8.py"
SELF="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-self-prefetch.py"
MATE="$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-hybrid8-next-mate-prefetch.py"
BUILDER="$ONEESAN_ROOT/scripts/build/b300-forced-nextgen-hybrid8-self-mate-geometry.sh"
SWEEP="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextmate-geometry-sweep.sh"
STAGED="$ONEESAN_ROOT/scripts/bench/b300-nextgen-hybrid8-nextmate-geometry-staged-calibrate.sh"
for f in "$HYBRID" "$SELF" "$MATE"; do python3 -m py_compile "$f"; done
for f in "$BUILDER" "$SWEEP" "$STAGED"; do bash -n "$f"; done

for s in \
  'SELF_WIDTH=' 'SELF_DISTANCE=' 'MATE_WIDTH=' 'MATE_DISTANCE=' \
  'NEXTSELF_WIDTH="$SELF_WIDTH" NEXTSELF_DISTANCE="$SELF_DISTANCE"' \
  '"$SRC" "$NEXT" "$MATE_WIDTH" "$MATE_DISTANCE" "$MATE_EVICT"' \
  'geometry_decoupled=1'; do
  grep -Fq "$s" "$BUILDER" || { echo "independent geometry builder marker missing: $s" >&2; exit 3; }
done
for s in \
  'MATE_WIDTH_LIST="${MATE_WIDTH_LIST:-1 2 4 8}"' \
  'MATE_DISTANCE_LIST="${MATE_DISTANCE_LIST:-1 2 4}"' \
  'SELF_W="$B300_HYBRID8_NEXTSELF_FINAL_WIDTH"' \
  'SELF_D="$B300_HYBRID8_NEXTSELF_FINAL_DISTANCE"' \
  'FATAL Stage-I residue mismatch' \
  'clean=len(rv)>=2 and ss==0 and sl==0' \
  'B300_STAGEI_MATE_WIDTH' \
  'B300_STAGEI_MATE_DISTANCE' \
  'b300_stagei_exact_match=1'; do
  grep -Fq "$s" "$SWEEP" || { echo "Stage-I geometry sweep marker missing: $s" >&2; exit 3; }
done
for s in \
  'SEARCH_ROWS="${SEARCH_ROWS:-1}"' \
  'VALIDATE_ROWS="${VALIDATE_ROWS:-4 8}"' \
  'MATE_WIDTH_LIST="${MATE_WIDTH_LIST:-1 2 4 8}"' \
  'MATE_DISTANCE_LIST="${MATE_DISTANCE_LIST:-1 2 4}"' \
  'FATAL Stage-I/core residue mismatch' \
  'FATAL Stage-I/Stage-E-final residue mismatch' \
  'FATAL Stage-I/Stage-F-final residue mismatch' \
  'FATAL Stage-I geometry changed during validation' \
  'B300_STAGEI_STAGED_VALIDATED' \
  'B300_STAGEI_FINAL_MATE_WIDTH' \
  'B300_STAGEI_FINAL_MATE_DISTANCE' \
  'B300_STAGEI_FINAL_STAGE_RESIDUE' \
  'B300_STAGEI_SEARCH_MATE_WIDTHS' \
  'B300_STAGEI_SEARCH_MATE_DISTANCES'; do
  grep -Fq "$s" "$STAGED" || { echo "Stage-I staged marker missing: $s" >&2; exit 3; }
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
python3 "$HYBRID" "$BASE" "$HYB" 1048576 >/dev/null

# Freeze self at w4,d2 and vary mate through all 12 geometries. The generated
# self advance must stay 16*grid while mate advance varies independently.
SELF_SRC="$TMP/self_w4_d2.cu"
python3 "$SELF" "$HYB" "$SELF_SRC" 4 2 default >"$TMP/self.log"
grep -Fq 'prefetch_width=4' "$TMP/self.log"
grep -Fq 'prefetch_distance_iterations=2' "$TMP/self.log"
grep -Fq 'const Code next_base=base+Code(16)*grid' "$SELF_SRC"
count=0
for md in 1 2 4; do
  advance=$((8*md))
  for mw in 1 2 4 8; do
    OUT="$TMP/self4d2_mate${mw}d${md}.cu"
    LOG="$TMP/mate${mw}d${md}.log"
    python3 "$MATE" "$SELF_SRC" "$OUT" "$mw" "$md" default >"$LOG"
    grep -Fq "prefetch_width=$mw" "$LOG"
    grep -Fq "prefetch_distance_iterations=$md" "$LOG"
    grep -Fq 'const Code next_base=base+Code(16)*grid' "$OUT"
    grep -Fq "const Code next_mate_base=base+Code(${advance})*grid" "$OUT"
    self_count="$(grep -Fc 'b300_mainrec_hybrid8_prefetch_next_self_l2(ni' "$OUT")"
    mate_count="$(grep -Fc 'b300_mainrec_hybrid8_prefetch_next_mate_l2(nmi' "$OUT")"
    [[ "$self_count" == 4 ]] || { echo "self prefetch count drifted mw=$mw md=$md count=$self_count" >&2; exit 4; }
    [[ "$mate_count" == "$mw" ]] || { echo "mate prefetch count mismatch mw=$mw md=$md count=$mate_count" >&2; exit 4; }
    if ((mw<8)); then ! grep -Fq "const Code nmi${mw}=" "$OUT"; fi
    ((count+=1))
  done
done
[[ "$count" == 12 ]]

echo 'b300-mainrec-self-mate-independent-geometry-preflight OK self_geometry=w4d2 mate_geometries=12 self_advance_locked=16grid mate_advance_independent=1 exact_prefetch_counts=1 builder_decoupled=1 sweep_exact_gate=1 spill_gate=1 staged_rows=1,4,8 staged_geometry_lock=1 gpu_work=0'
