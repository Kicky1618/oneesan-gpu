#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27; GPUS=8
MOD="${MOD:-4294967291}"
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
BATCH_LIST="${BATCH_LIST:-2 4}"
REPEATS="${REPEATS:-1}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
DUALMASK="${DUALMASK:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_closure_batch_row${ROWS}_hd${HIGH_DROP_CHUNK}_dual${DUALMASK}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
ISO="${ISO:-$ONEESAN_BUILD_DIR/mainrec_closure_batch_$$}"
BASE_BIN="$ONEESAN_BUILD_DIR/b300_mainrec_closure_batch_base_hd${HIGH_DROP_CHUNK}_dual${DUALMASK}_n27"
mkdir -p "$LOGDIR" "$ISO" "$ISO/tmp" "$(dirname "$RESULT")"
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$REPEATS" =~ ^[0-9]+$ ]] && ((REPEATS>=1)) || { echo 'REPEATS must be >=1' >&2; exit 2; }
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || { echo 'HIGH_DROP_CHUNK must be 0 or 1' >&2; exit 2; }
[[ "$DUALMASK" == 0 || "$DUALMASK" == 1 ]] || { echo 'DUALMASK must be 0 or 1' >&2; exit 2; }
for b in $BATCH_LIST; do [[ "$b" == 2 || "$b" == 4 ]] || { echo 'BATCH_LIST supports 2 4' >&2; exit 2; }; done
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= GPUS )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

BASE_BUILD_OUT="$LOGDIR/base.build.out";BASE_BUILD_ERR="$LOGDIR/base.build.err"
echo "=== build calibrated source highdrop=$HIGH_DROP_CHUNK dualmask=$DUALMASK ===" >&2
N=27 ARCH="$ARCH" OUT="$BASE_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" DUALMASK="$DUALMASK" BUILD_ERR="$BASE_BUILD_ERR" \
CALIBRATED_BUILD_DIR="$ISO/basechain" PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-calibrated.sh" >"$BASE_BUILD_OUT" 2>"$LOGDIR/base.build.driver.err"
[[ -x "$BASE_BIN" ]] || { echo 'baseline binary missing' >&2;exit 3; }
if [[ "$DUALMASK" == 1 ]]; then BUILD_SRC="$(sed -nE 's/^  source_after_dualmask=(.*)$/\1/p' "$BASE_BUILD_OUT"|tail -n1)";else BUILD_SRC="$(sed -nE 's/^  source_after_proof_gates=(.*)$/\1/p' "$BASE_BUILD_OUT"|tail -n1)";fi
[[ -n "$BUILD_SRC" && -f "$BUILD_SRC" ]] || { echo 'could not resolve calibrated source' >&2;exit 3; }

BATCH_SRC="$ISO/final_closure_batch.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-quad.py" "$BUILD_SRC" "$BATCH_SRC" >"$LOGDIR/batch.transform.out"
grep -Fq 'batch_macro=B300_BLOCK_CLOSURE_BATCH' "$LOGDIR/batch.transform.out"
: >"$LOGDIR/binaries.tsv"
printf 'base\t0\t%s\n' "$BASE_BIN" >>"$LOGDIR/binaries.tsv"
for b in $BATCH_LIST;do
  bin="$ONEESAN_BUILD_DIR/b300_mainrec_closure_batch${b}_hd${HIGH_DROP_CHUNK}_dual${DUALMASK}_n27";err="$LOGDIR/batch${b}.build.err"
  echo "=== compile closure batch=$b ===" >&2
  TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v -DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13 \
    -DB300_BLOCK_CLOSURE_BATCH="$b" "$BATCH_SRC" -o "$bin" >"$LOGDIR/batch${b}.build.out" 2>"$err"
  [[ -x "$bin" ]] || { echo "missing batch=$b binary" >&2;exit 3; }
  printf 'batch%s\t%s\t%s\n' "$b" "$b" "$bin" >>"$LOGDIR/binaries.tsv"
done

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'mode\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
python3 "$PARSER" "$BASE_BUILD_ERR" --label base --contains block_pull_kernel >>"$RESOURCE" || true
for b in $BATCH_LIST;do python3 "$PARSER" "$LOGDIR/batch${b}.build.err" --label "batch${b}" --contains block_pull_kernel >>"$RESOURCE" || true;done

field(){ local k="$1" l="$2";sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p"<<<"$l"|tail -n1; }
summarize(){ python3 - "$1" <<'PY'
import csv,sys
mem=[];busy=[];sm=[];power=[]
for r in csv.reader(open(sys.argv[1])):
 if len(r)<5:continue
 try:s=float(r[2]);m=float(r[3]);p=float(r[4])
 except ValueError:continue
 sm.append(s);mem.append(m);power.append(p)
 if s>=50:busy.append(m)
def a(x):return sum(x)/len(x) if x else float('nan')
print(f'{a(mem):.6f} {max(mem) if mem else float("nan"):.6f} {a(busy):.6f} {a(sm):.6f} {a(power):.6f} {len(mem)}')
PY
}
printf 'mode\tbatch\tthreads\trepeat\tresidue\twall_s\tmem_avg_pct\tmem_max_pct\tmem_busy_avg_pct\tsm_avg_pct\tpower_avg_w\tsamples\n' >"$RESULT"
run_one(){
 local mode="$1" batch="$2" bin="$3" threads="$4" rep="$5"
 local tag="${mode}_t${threads}_r${rep}"
 local out="$LOGDIR/$tag.out" err="$LOGDIR/$tag.err" tele="$LOGDIR/$tag.gpu.csv"
 nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits -lms 200 >"$tele" 2>/dev/null & local mon=$!;sleep 1;set +e
 B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$threads" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$out" 2>"$err";local rc=$?;set -e
 kill "$mon" 2>/dev/null||true;wait "$mon" 2>/dev/null||true
 ((rc==0))||{ echo "$tag failed rc=$rc" >&2;tail -n 160 "$err" >&2||true;return "$rc"; }
 local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$out"|tail -n1||true)";[[ -n "$line" ]]||{ echo "$tag missing result" >&2;return 4; }
 grep -Fq " rows=$ROWS calibration=$((ROWS<28?1:0)) "<<<"$line"||{ echo "$tag row metadata mismatch" >&2;return 5; }
 local stats="$(summarize "$tele")";printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$batch" "$threads" "$rep" "$(field residue "$line")" "$(field wall_s "$line")" "$stats" >>"$RESULT"
}

for t in $THREADS_LIST;do
 [[ "$t" =~ ^[0-9]+$ ]]&&((t>=32&&t<=1024&&t%32==0))||{ echo "bad threads=$t" >&2;exit 2; }
 for((r=1;r<=REPEATS;++r));do
   mapfile -t lines <"$LOGDIR/binaries.tsv"
   if ((!(r&1)));then mapfile -t lines < <(tac "$LOGDIR/binaries.tsv");fi
   for x in "${lines[@]}";do IFS=$'\t' read -r m b bin<<<"$x";run_one "$m" "$b" "$bin" "$t" "$r";done
 done
done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
if not rows:raise SystemExit('no results')
res={r['residue'] for r in rows}
if len(res)!=1:raise SystemExit('FATAL closure batch residue mismatch '+repr(sorted(res)))
by={}
for r in rows:by.setdefault((r['mode'],int(r['threads'])),[]).append(float(r['wall_s']))
med={k:statistics.median(v) for k,v in by.items()}
base=min((v,t) for (m,t),v in med.items() if m=='base')
print('b300_mainrec_closure_batch_exact_intermediate_match=1')
print(f'b300_mainrec_closure_batch_residue={next(iter(res))}')
print(f'b300_mainrec_closure_batch_base_best_wall_s={base[0]:.9f}')
print(f'b300_mainrec_closure_batch_base_best_threads={base[1]}')
best=None
for (m,t),w in sorted(med.items()):
 if m=='base':continue
 b=med.get(('base',t));sp=b/w if b else float('nan');batch=int(m.removeprefix('batch'))
 print(f'b300_mainrec_closure_batch_{batch}_threads_{t}_wall_s={w:.9f}')
 print(f'b300_mainrec_closure_batch_{batch}_threads_{t}_speedup_same_thread={sp:.9f}x')
 if best is None or w<best[0]:best=(w,batch,t,sp)
if best:
 print(f'b300_mainrec_closure_batch_best_batch={best[1]}')
 print(f'b300_mainrec_closure_batch_best_threads={best[2]}')
 print(f'b300_mainrec_closure_batch_best_wall_s={best[0]:.9f}')
 print(f'b300_mainrec_closure_batch_best_same_thread_speedup={best[3]:.9f}x')
 print(f'b300_mainrec_closure_batch_speedup_vs_global_base_best={base[0]/best[0]:.9f}x')
PY
printf 'b300_mainrec_closure_batch_high_drop_chunk=%s\n' "$HIGH_DROP_CHUNK"
printf 'b300_mainrec_closure_batch_dualmask=%s\n' "$DUALMASK"
printf 'b300_mainrec_closure_batch_rows=%s\n' "$ROWS"
cat "$RESOURCE"
