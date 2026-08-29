#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ROWS="${ROWS:-1}"; THREADS="${GRIDFP_THREADS:-256}"; REPEATS="${REPEATS:-2}"
TARGET_MIB="${TARGET_MIB:-65536}"; PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
MOD="${MOD:-4294967291}"; RANDOM_CG="${RANDOM_CG:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_closure_warpscan_row${ROWS}_t${THREADS}_cg${RANDOM_CG}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || exit 2
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32&&THREADS<=1024&&THREADS%32==0)) || exit 2
[[ "$REPEATS" =~ ^[0-9]+$ ]] && ((REPEATS>=1)) || exit 2
[[ "$RANDOM_CG" == 0 || "$RANDOM_CG" == 1 ]] || exit 2
command -v nvidia-smi >/dev/null || exit 2
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || exit 2

bash "$ONEESAN_ROOT/scripts/bench/b300-block-closure-warpscan-proof.sh"

# Build/warm each isolated suffix once. These executions are intentionally not
# included in telemetry so nvcc and first-use effects cannot bias the A/B.
for ws in 0 1; do
  echo "=== prebuild warpscan=$ws ===" >&2
  ROWS=1 GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
    RANDOM_CG="$RANDOM_CG" WARP_SCAN="$ws" REBUILD=1 \
    "$ONEESAN_ROOT/scripts/run/b300x8-saturate-dualmask.sh" 27 "$MOD" >"$LOGDIR/prebuild_ws${ws}.out" 2>"$LOGDIR/prebuild_ws${ws}.err"
done

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l"|tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;p=$3+0;sg+=g;sm+=m;sp+=p;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %.6f %.6f\n",sg/n,sm/n,mm,sp/n}' >>"$out" || true;sleep 0.2;done; }
sumtele(){ awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;sp+=$4;n++}END{if(n)printf "%.6f %.6f %.6f %.6f\n",sg/n,sm/n,mm,sp/n;else print "NA NA NA NA"}' "$1"; }

printf 'mode\trepeat\tresidue\twall_s\tactive_max_s\tgpu_avg_pct\tmem_avg_pct\tmem_max_pct\tpower_avg_w\n' >"$RESULT"
run_one(){
  local ws="$1" rep="$2" mode; [[ "$ws" == 1 ]] && mode=warpscan || mode=serial
  local out="$LOGDIR/${mode}_r${rep}.out" err="$LOGDIR/${mode}_r${rep}.err" tele="$LOGDIR/${mode}_r${rep}.gpu"
  ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
    RANDOM_CG="$RANDOM_CG" WARP_SCAN="$ws" REBUILD=0 \
    "$ONEESAN_ROOT/scripts/run/b300x8-saturate-dualmask.sh" 27 "$MOD" >"$out" 2>"$err" &
  local pid=$!; sample "$pid" "$tele" & local sp=$!;set +e;wait "$pid";local rc=$?;set -e;wait "$sp"||true
  ((rc==0))||{ tail -n 160 "$err" >&2;return "$rc"; }
  local line="$(grep '^backend=gridfp-b300-hbm32' "$out"|tail -n1)";[[ -n "$line" ]]||return 4
  local gpu mem mm power;read -r gpu mem mm power < <(sumtele "$tele")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$rep" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$gpu" "$mem" "$mm" "$power" >>"$RESULT"
}
for ((r=1;r<=REPEATS;++r));do if ((r&1));then run_one 0 "$r";run_one 1 "$r";else run_one 1 "$r";run_one 0 "$r";fi;done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));res={x['residue'] for x in r}
if len(res)!=1:raise SystemExit('FATAL closure-warpscan residue mismatch '+repr(sorted(res)))
def med(m,k):return statistics.median(float(x[k]) for x in r if x['mode']==m and x[k]!='NA')
s=med('serial','wall_s');w=med('warpscan','wall_s');sm=med('serial','mem_avg_pct');wm=med('warpscan','mem_avg_pct')
print('b300_closure_warpscan_residue_match=1')
print(f'b300_closure_warpscan_speedup={s/w:.6f}x')
print(f'b300_closure_warpscan_mem_delta_pp={wm-sm:.6f}')
print(f'b300_closure_warpscan_serial_wall_s={s:.9f} mem_avg_pct={sm:.3f}')
print(f'b300_closure_warpscan_candidate_wall_s={w:.9f} mem_avg_pct={wm:.3f}')
PY
cat "$RESULT"
echo "b300-closure-warpscan-ab OK result=$RESULT logs=$LOGDIR" >&2
