#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
BASE_WEIGHT="${PRODUCER_WORKER_WEIGHT:-0}"; THRESHOLDS="${PRODUCER_ADAPTIVE_THRESHOLDS:-0 256 512 1024 2048 4096 8192 16384}"
BATCH="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"; ADAPTIVE_WAVES="${ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES:-0}"
SPARSE64="${DIRECTGATHER_SPARSE64:-1}"; COL_ILP="${ORBITCTA_COL_ILP:-4}"; CPASYNC_PAIR="${CPASYNC_PAIR:-1}"; PM_ACCUM="${PM_ACCUM:-1}"
QUAD_MLP="${QUAD_MLP:-1}"; PRODUCER_PRECTX_WARPCOOP="${PRODUCER_PRECTX_WARPCOOP:-0}"
PRECTX_FORWARD="${PRECTX_FORWARD:-1}"; PRECTX_REVERSE="${PRECTX_REVERSE:-1}"; PRECTX_COMPACT="${PRECTX_COMPACT:-1}"
PRECTX_FLAT_BID="${PRECTX_FLAT_BID:-0}"; PRECTX_FLAT_BID_FUSED="${PRECTX_FLAT_BID_FUSED:-0}"
FLAT_PER_SM="${BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM:-}"; FLAT_BLOCKS="${BUCKET_ORBITCTA_FLAT_BLOCKS:-}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_pipe2_producer_adaptive_n${N}_w${BASE_WEIGHT}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

[[ "$N" == 21 && "$MOD" == 4294967291 && "$EXPECT" == 998035516 ]] || { echo 'fixed exact gate is n21/mod4294967291/residue998035516' >&2; exit 2; }
case "$BASE_WEIGHT" in 0|2|3|4) ;; 1) echo 'base weight=1 already gives maximum producer column share; adaptive threshold is a no-op' >&2; exit 8;; *) echo 'PRODUCER_WORKER_WEIGHT must be 0..4' >&2; exit 2;; esac
case "$BATCH" in 1|2|4|8|16) ;; *) exit 2;; esac
case "$ADAPTIVE_WAVES" in 0|1|2|4) ;; *) exit 2;; esac
case "$COL_ILP" in 1|2|4) ;; *) exit 2;; esac
for x in SPARSE64 CPASYNC_PAIR PM_ACCUM QUAD_MLP PRODUCER_PRECTX_WARPCOOP PRECTX_FORWARD PRECTX_REVERSE PRECTX_COMPACT PRECTX_FLAT_BID PRECTX_FLAT_BID_FUSED; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
[[ "$QUAD_MLP" == 0 || "$COL_ILP" == 4 ]] || exit 2
[[ -z "$FLAT_PER_SM" || -z "$FLAT_BLOCKS" ]] || { echo 'set at most one flat pool override' >&2; exit 2; }
for t in $THRESHOLDS; do [[ "$t" =~ ^[0-9]+$ ]] || { echo "bad adaptive threshold=$t" >&2; exit 2; }; done
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }; command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/b300-pipe2-producer-warp-coverage-proof.sh" >"$LOGDIR/coverage.out"
grep -q 'producer_adaptive_cols=.*wide_weight=1 exact_once=1' "$LOGDIR/coverage.out" || { echo 'adaptive producer coverage gate failed' >&2; exit 5; }

COMMON=(N="$N" ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" DIRECTGATHER_SORT_RANKS=0 RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR="$CPASYNC_PAIR" ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_BATCH="$BATCH" ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="$ADAPTIVE_WAVES" ORBITCTA_COL_ILP="$COL_ILP" QUAD_MLP="$QUAD_MLP" PRECTX_FORWARD="$PRECTX_FORWARD" PRECTX_REVERSE="$PRECTX_REVERSE" PRECTX_COMPACT="$PRECTX_COMPACT" PRECTX_FLAT_BID="$PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED="$PRECTX_FLAT_BID_FUSED" PRODUCER_PRECTX_WARPCOOP="$PRODUCER_PRECTX_WARPCOOP" PRODUCER_WORKER_WEIGHT="$BASE_WEIGHT" PTXAS_VERBOSE=1)
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
declare -A BIN
for t in $THRESHOLDS; do
  tag="t${t}"; BIN[$tag]="$ONEESAN_BUILD_DIR/b300_pipe2_producer_adaptive_w${BASE_WEIGHT}_t${t}_n${N}"
  env "${COMMON[@]}" PRODUCER_ADAPTIVE_COLS="$t" OUT="${BIN[$tag]}" bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2-producer-warp.sh" >"$LOGDIR/${tag}.build.out" 2>"$LOGDIR/${tag}.build.err"
  grep -q "pipe2_producer_worker_weight=$BASE_WEIGHT" "$LOGDIR/${tag}.build.err" || { echo "$tag worker-weight marker mismatch" >&2; exit 6; }
  grep -q "pipe2_producer_adaptive_cols=$t" "$LOGDIR/${tag}.build.err" || { echo "$tag adaptive marker mismatch" >&2; exit 6; }
  python3 "$PARSER" "$LOGDIR/${tag}.build.err" --label "$tag" >>"$RESOURCE" || true
done

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }
printf 'threshold\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_pct\tavg_memctrl_pct\tpeak_gpu_pct\tpeak_memctrl_pct\n' >"$RESULT"
run_one(){
  local t="$1" rep="$2" tag="t${1}" so="$LOGDIR/t${1}_r${2}.out" se="$LOGDIR/t${1}_r${2}.err" util="$LOGDIR/t${1}_r${2}.util"
  runenv=(BUCKET_THREADS="$THREADS" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY")
  [[ -z "$FLAT_PER_SM" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$FLAT_PER_SM")
  [[ -z "$FLAT_BLOCKS" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS="$FLAT_BLOCKS")
  env "${runenv[@]}" "${BIN[$tag]}" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  local pid=$!; sample "$pid" "$util" & local sp=$!; set +e; wait "$pid"; local rc=$?; set -e; wait "$sp" || true; ((rc==0)) || exit "$rc"
  local line="$(grep '^residue=' "$so"|tail -n1||true)"; [[ -n "$line" ]] || exit 3
  local residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "t=$t residue=$residue expected=$EXPECT" >&2; exit 4; }
  local d="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1||true)" fh="$(field forward_high_s "$d")" rh="$(field reverse_high_s "$d")"; [[ -n "$fh" && -n "$rh" ]] || exit 5
  local high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)" ag am pg pm
  read -r ag am pg pm < <(awk '{sg+=$1;sm+=$2;if($3>pg)pg=$3;if($4>pm)pm=$4;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,pg,pm;else print "NA NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$pg" "$pm" >>"$RESULT"
}
for ((r=1;r<=REPEATS;++r)); do for t in $THRESHOLDS; do run_one "$t" "$r"; done; done
cat "$RESULT"
python3 - "$RESULT" "$WINNER_ENV" "$BASE_WEIGHT" <<'PY'
import csv,statistics,sys
src,out,base=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); ts=[]
for t in dict.fromkeys(r['threshold'] for r in rows):
 g=[r for r in rows if r['threshold']==t]
 wall=statistics.median(float(r['wall_s']) for r in g); high=statistics.median(float(r['high_s']) for r in g)
 mc=statistics.median(float(r['avg_memctrl_pct']) for r in g if r['avg_memctrl_pct']!='NA')
 ts.append((wall,high,int(t),mc))
for w,h,t,mc in sorted(ts): print('PRODUCER_ADAPTIVE',f'threshold={t}',f'wall_s={w:.6f}',f'high_s={h:.6f}',f'mc_avg_pct={mc:.3f}')
b=min(ts)
with open(out,'w') as f:
 f.write('ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP=1\n')
 f.write(f'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT={base}\n')
 f.write(f'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS={b[2]}\n')
 f.write(f'DYNAMIC_PRODUCER_ADAPTIVE_WALL_S={b[0]:.9f}\nDYNAMIC_PRODUCER_ADAPTIVE_HIGH_S={b[1]:.9f}\n')
print('BEST_PRODUCER_ADAPTIVE',f'threshold={b[2]}',f'wall_s={b[0]:.6f}',f'high_s={b[1]:.6f}',f'winner_env={out}')
PY
