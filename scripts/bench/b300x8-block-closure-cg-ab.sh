#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27; MOD="${MOD:-4294967291}"; ARCH="${ARCH:-native}"; NGPU=8
TARGET_MIB="${TARGET_MIB:-65536}"; PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}"; THREADS="${GRIDFP_THREADS:-256}"; REPEATS="${REPEATS:-1}"; SAMPLE_MS="${SAMPLE_MS:-200}"
HOT_DELTA_TABLE="${HOT_DELTA_TABLE:-1}"; CONCURRENT_GROUP_IO="${CONCURRENT_GROUP_IO:-1}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_block_closure_cg_ab_row${ROWS}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR"

[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$HOT_DELTA_TABLE" == 0 || "$HOT_DELTA_TABLE" == 1 ]] || { echo 'HOT_DELTA_TABLE must be 0/1' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

BASE_BIN="$ONEESAN_BUILD_DIR/b300_closure_quad_l1_n27"
CG_BIN="$ONEESAN_BUILD_DIR/b300_closure_quad_cg_n27"
BUILD_OUT="$LOGDIR/base.build.out"; BUILD_ERR="$LOGDIR/base.build.err"
N=27 ARCH="$ARCH" OUT="$BASE_BIN" FAST_SHARD_ADDRESS8=1 \
  MAIN_MATE_CACHE=1 MAIN_PULL=1 BLOCK_PULL=1 BLOCK_MATE_CACHE=1 MAIN_PULL_ILP2=0 HEIGHT_CACHE=0 \
  RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1 RANK_STATE_ILP2=0 RANK_STATE_ILP4=1 BLOCK_CLOSURE_QUAD=1 \
  HOT_DELTA_TABLE="$HOT_DELTA_TABLE" CONCURRENT_GROUP_IO="$CONCURRENT_GROUP_IO" MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-hbm32.sh" >"$BUILD_OUT" 2>"$BUILD_ERR"

grep -Fq 'rank_state_ilp4=1 block_closure_quad=1' "$BUILD_OUT" || { echo 'baseline metadata mismatch' >&2; exit 3; }
FINAL_SRC="$(sed -nE 's/^  build_source=(.*)$/\1/p' "$BUILD_OUT" | tail -n1)"
[[ -n "$FINAL_SRC" && -f "$FINAL_SRC" ]] || { echo "could not resolve final generated source from $BUILD_OUT" >&2; exit 3; }
grep -Fq 'B300_BLOCK_CLOSURE_CG' "$FINAL_SRC" || { echo 'generated source lacks closure CG switch' >&2; exit 3; }

PTXAS_FLAGS=(-Xptxas=-v); REG_FLAGS=(); ((MAXRREGCOUNT>0)) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" "${PTXAS_FLAGS[@]}" "${REG_FLAGS[@]}" \
  -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=1 \
  -DB300_BLOCK_CLOSURE_QUAD=1 -DB300_BLOCK_CLOSURE_CG=1 \
  "$FINAL_SRC" -o "$CG_BIN" >"$LOGDIR/cg.build.out" 2>"$LOGDIR/cg.build.err"

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'mode\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
python3 "$PARSER" "$BUILD_ERR" --label l1 >>"$RESOURCE" || true
python3 "$PARSER" "$LOGDIR/cg.build.err" --label cg >>"$RESOURCE" || true

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
summarize(){ python3 - "$1" <<'PY'
import csv,sys
mem=[];busy=[];sm=[];power=[]
for r in csv.reader(open(sys.argv[1])):
    if len(r)<5:continue
    try:s=float(r[2]);m=float(r[3]);p=float(r[4])
    except ValueError:continue
    sm.append(s);mem.append(m);power.append(p)
    if s>=50:busy.append(m)
def avg(a):return sum(a)/len(a) if a else float('nan')
print(f'{avg(mem):.6f} {max(mem) if mem else float("nan"):.6f} {avg(busy):.6f} {avg(sm):.6f} {avg(power):.6f} {len(mem)}')
PY
}
printf 'mode\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tmem_avg_pct\tmem_max_pct\tmem_busy_avg_pct\tsm_avg_pct\tpower_avg_w\tsamples\n' >"$RESULT"
run_one(){
  local mode="$1" bin="$2" rep="$3" tag="${mode}_r${rep}" out="$LOGDIR/$tag.out" err="$LOGDIR/$tag.err" tele="$LOGDIR/$tag.gpu.csv"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits -lms "$SAMPLE_MS" >"$tele" 2>/dev/null & mon=$!
  sleep 1
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" GRIDFP_PLAN_TARGET_DIVISOR=1 GRIDFP_VRAM_RESERVE_MIB=8192 \
    "$bin" 27 "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$out" 2>"$err"
  rc=$?; set -e; kill "$mon" 2>/dev/null || true; wait "$mon" 2>/dev/null || true
  ((rc==0)) || { echo "$tag failed rc=$rc" >&2; exit "$rc"; }
  line="$(grep '^backend=gridfp-b300-hbm32' "$out" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$tag missing backend line" >&2; exit 4; }
  stats="$(summarize "$tele")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$rep" "$(field residue "$line")" "$(field wall_s "$line")" "$(field active_max_s "$line")" "$(field active_sum_s "$line")" "$stats" >>"$RESULT"
}
for ((r=1;r<=REPEATS;++r));do run_one l1 "$BASE_BIN" "$r";run_one cg "$CG_BIN" "$r";done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
res={x['residue'] for x in r}
if len(res)!=1:raise SystemExit('FATAL closure CG residue mismatch '+repr(sorted(res)))
q={}
for mode in ('l1','cg'):
 g=[x for x in r if x['mode']==mode]
 q[mode]={k:statistics.median(float(x[k]) for x in g) for k in ('wall_s','active_max_s','mem_avg_pct','mem_busy_avg_pct','sm_avg_pct')}
print('closure_cg_residue_match=1')
print(f'closure_cg_wall_speedup={q["l1"]["wall_s"]/q["cg"]["wall_s"]:.6f}x')
print(f'closure_cg_active_speedup={q["l1"]["active_max_s"]/q["cg"]["active_max_s"]:.6f}x')
print(f'closure_cg_mem_busy_delta={q["cg"]["mem_busy_avg_pct"]-q["l1"]["mem_busy_avg_pct"]:.6f}pp')
print(f'closure_cg_sm_delta={q["cg"]["sm_avg_pct"]-q["l1"]["sm_avg_pct"]:.6f}pp')
PY
cat "$RESULT";cat "$RESOURCE"
echo "b300 closure CG A/B OK result=$RESULT resources=$RESOURCE" >&2
