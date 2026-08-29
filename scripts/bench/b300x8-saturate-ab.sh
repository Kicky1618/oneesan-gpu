#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
THREADS="${GRIDFP_THREADS:-256}"
TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"
MAX_WINDOW="${MAX_WINDOW:-14}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.20}"
REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300x8_saturate_ab_n${N}_r${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

[[ "$N" == 27 ]] || { echo 'saturate A/B currently targets n=27' >&2; exit 2; }
[[ "$ROWS" =~ ^[0-9]+$ ]] && (( ROWS >= 1 && ROWS <= 28 )) || exit 2
[[ "$THREADS" =~ ^[0-9]+$ ]] && (( THREADS >= 32 && THREADS <= 1024 && THREADS % 32 == 0 )) || exit 2
[[ "$REPEATS" =~ ^[0-9]+$ ]] && (( REPEATS >= 1 )) || exit 2
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){
  local pid="$1" out="$2"; : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits 2>/dev/null |
      awk -F',' '{g=$1+0;m=$2+0;p=$3+0;sg+=g;sm+=m;sp+=p;if(m>mm)mm=m;n++} END{if(n)printf "%.6f %.6f %.6f %.6f\n",sg/n,sm/n,mm,sp/n;}' >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}
summarize(){
  awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;sp+=$4;n++} END{if(n)printf "%.6f %.6f %.6f %.6f\n",sg/n,sm/n,mm,sp/n;else print "NA NA NA NA"}' "$1"
}

build_variant(){
  local mode="$1" bin="$ONEESAN_BUILD_DIR/b300x8_${mode}_n${N}"
  local i2=0 i4=0 warp=0
  case "$mode" in
    ilp2) i2=1 ;;
    ilp4warp) i4=1; warp=1 ;;
    *) return 2 ;;
  esac
  echo "=== build $mode ===" >&2
  N="$N" OUT="$bin" FAST_SHARD_ADDRESS8=1 \
    MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
    MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 \
    RANK_STATE_ILP2="$i2" RANK_STATE_ILP3=0 RANK_STATE_ILP4="$i4" \
    BLOCK_CLOSURE_QUAD=0 BLOCK_CLOSURE_WARP="$warp" HOT_DELTA_TABLE=0 \
    CONCURRENT_GROUP_IO=1 MAXRREGCOUNT=0 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
  [[ -x "$bin" ]] || { echo "missing $bin" >&2; return 3; }
  printf '%s' "$bin"
}

: >"$LOGDIR/binaries.tsv"
for mode in ilp2 ilp4warp; do
  bin="$(build_variant "$mode")"
  printf '%s\t%s\n' "$mode" "$bin" >>"$LOGDIR/binaries.tsv"
done

printf 'mode\trepeat\tresidue\twall_s\tactive_max_s\tgpu_avg_pct\tmem_avg_pct\tmem_max_pct\tpower_avg_w\n' >"$RESULT"
for ((r=1;r<=REPEATS;++r)); do
  for mode in ilp2 ilp4warp; do
    bin="$(awk -F '\t' -v m="$mode" '$1==m{print $2}' "$LOGDIR/binaries.tsv")"
    out="$LOGDIR/${mode}_r${r}.out"; err="$LOGDIR/${mode}_r${r}.err"; tele="$LOGDIR/${mode}_r${r}.gpu"
    echo "=== run $mode repeat=$r rows=$ROWS threads=$THREADS ===" >&2
    B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" \
      "$bin" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err" &
    pid=$!; sample "$pid" "$tele" & sp=$!
    set +e; wait "$pid"; rc=$?; set -e
    wait "$sp" || true
    (( rc == 0 )) || { echo "$mode failed rc=$rc" >&2; tail -n 100 "$err" >&2 || true; exit "$rc"; }
    line="$(grep '^backend=gridfp-b300-hbm32' "$out" | tail -n1 || true)"
    [[ -n "$line" ]] || { echo "$mode missing backend line" >&2; exit 4; }
    residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; active="$(field active_max_s "$line")"
    read -r gpu mem memmax power < <(summarize "$tele")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$r" "$residue" "$wall" "$active" "$gpu" "$mem" "$memmax" "$power" >>"$RESULT"
  done
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL residue mismatch '+repr(sorted(res)))
def med(mode,key): return statistics.median(float(r[key]) for r in rows if r['mode']==mode and r[key] != 'NA')
a=med('ilp2','wall_s'); b=med('ilp4warp','wall_s')
ma=med('ilp2','mem_avg_pct'); mb=med('ilp4warp','mem_avg_pct')
print(f'b300_saturate_residue_match=1 residue={next(iter(res))}')
print(f'b300_saturate_wall_speedup={a/b:.6f}x')
print(f'b300_saturate_mem_delta_pp={mb-ma:.6f}')
print(f'b300_saturate_ilp2_wall_s={a:.9f} mem_avg_pct={ma:.3f}')
print(f'b300_saturate_ilp4warp_wall_s={b:.9f} mem_avg_pct={mb:.3f}')
print('b300_saturate_winner='+('ilp4warp' if b<a else 'ilp2'))
PY
cat "$RESULT"
echo "b300x8-saturate-ab OK result=$RESULT logs=$LOGDIR" >&2
