#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27; MOD="${MOD:-4294967291}"; ARCH="${ARCH:-native}"; NGPU=8
TARGET_MIB="${TARGET_MIB:-65536}"; PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}"; THREADS="${GRIDFP_THREADS:-256}"; REPEATS="${REPEATS:-2}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_closure_contrib_table_row${ROWS}_t${THREADS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
ISO="${ISO:-$ONEESAN_BUILD_DIR/closure_contrib_table_$$}"
mkdir -p "$LOGDIR" "$ISO" "$ISO/tmp" "$(dirname "$RESULT")"
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 GPUs' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32&&THREADS<=1024&&THREADS%32==0)) || exit 2

bash "$ONEESAN_ROOT/scripts/bench/b300-block-closure-warp-batch-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-dualmask-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-closure-contrib-table-proof.sh"

GEN_BIN="$ISO/generated_base"
BUILD_OUT="$LOGDIR/generated.build.out"; BUILD_ERR="$LOGDIR/generated.build.err"
ONEESAN_BUILD_DIR="$ISO" ONEESAN_TMP_DIR="$ISO/tmp" \
N=27 ARCH="$ARCH" OUT="$GEN_BIN" FAST_SHARD_ADDRESS8=1 \
MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 \
MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 \
RANK_STATE_ILP2=0 RANK_STATE_ILP3=0 RANK_STATE_ILP4=1 \
BLOCK_CLOSURE_QUAD=0 BLOCK_CLOSURE_WARP=1 HOT_DELTA_TABLE=0 CONCURRENT_GROUP_IO=1 MAXRREGCOUNT=0 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$BUILD_OUT" 2>"$BUILD_ERR"
SRC="$(sed -nE 's/^  build_source=(.*)$/\1/p' "$BUILD_OUT" | tail -n1)"
[[ -n "$SRC" && -f "$SRC" ]] || { echo 'generated source missing' >&2; exit 3; }

BASE_SRC="$ISO/base_dualmask.cu"; CAND_SRC="$ISO/cand_dualmask_table.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-warp-dualmask.py" "$SRC" "$BASE_SRC"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-closure-contrib-table.py" "$BASE_SRC" "$CAND_SRC"
BASE_BIN="$ISO/base"; CAND_BIN="$ISO/table"
for mode in base table; do
  if [[ "$mode" == base ]]; then src="$BASE_SRC"; bin="$BASE_BIN"; else src="$CAND_SRC"; bin="$CAND_BIN"; fi
  TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v \
    -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=1 -DB300_BLOCK_CLOSURE_QUAD=0 \
    "$src" -o "$bin" >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
done

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;p=$3+0;sg+=g;sm+=m;sp+=p;if(m>mm)mm=m;n++}END{if(n)printf "%.5f %.5f %.5f %.5f\n",sg/n,sm/n,mm,sp/n}' >>"$out" || true; sleep 0.2; done; }
sumtele(){ awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;sp+=$4;n++}END{if(n)printf "%.5f %.5f %.5f %.5f\n",sg/n,sm/n,mm,sp/n;else print "NA NA NA NA"}' "$1"; }

printf 'mode\trepeat\tresidue\twall_s\tactive_max_s\tgpu_avg_pct\tmem_avg_pct\tmem_max_pct\tpower_avg_w\n' >"$RESULT"
run_one(){
  local mode="$1" bin="$2" rep="$3" out="$LOGDIR/${mode}_r${rep}.out" err="$LOGDIR/${mode}_r${rep}.err" tele="$LOGDIR/${mode}_r${rep}.gpu"
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err" &
  local pid=$!; sample "$pid" "$tele" & local sp=$!; set +e; wait "$pid"; local rc=$?; set -e; wait "$sp" || true
  ((rc==0)) || { tail -n 120 "$err" >&2; return "$rc"; }
  local line="$(grep '^backend=gridfp-b300-hbm32' "$out"|tail -n1)"; [[ -n "$line" ]] || return 4
  local gpu mem mm power; read -r gpu mem mm power < <(sumtele "$tele")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$rep" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$gpu" "$mem" "$mm" "$power" >>"$RESULT"
}
for ((r=1;r<=REPEATS;++r)); do
  if ((r&1)); then run_one base "$BASE_BIN" "$r"; run_one table "$CAND_BIN" "$r"; else run_one table "$CAND_BIN" "$r"; run_one base "$BASE_BIN" "$r"; fi
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
res={x['residue'] for x in r}
if len(res)!=1:raise SystemExit('FATAL residue mismatch '+repr(sorted(res)))
def med(m,k):return statistics.median(float(x[k]) for x in r if x['mode']==m and x[k]!='NA')
b=med('base','wall_s');c=med('table','wall_s');bm=med('base','mem_avg_pct');cm=med('table','mem_avg_pct')
print('b300_closure_contrib_table_residue_match=1')
print(f'b300_closure_contrib_table_speedup={b/c:.6f}x')
print(f'b300_closure_contrib_table_mem_delta_pp={cm-bm:.6f}')
print(f'b300_closure_contrib_table_base_wall_s={b:.9f}')
print(f'b300_closure_contrib_table_candidate_wall_s={c:.9f}')
PY
cat "$RESULT"
echo "b300-closure-contrib-table-ab OK result=$RESULT logs=$LOGDIR" >&2
