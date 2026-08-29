#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N=27;GPUS=8
MOD="${MOD:-4294967291}";ARCH="${ARCH:-native}";TARGET_MIB="${TARGET_MIB:-65536}";MAX_WINDOW="${MAX_WINDOW:-14}"
ROWS="${ROWS:-1}";THREADS_LIST="${THREADS_LIST:-128 256 512}";BATCH_LIST="${BATCH_LIST:-2 4}";REPEATS="${REPEATS:-1}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_block_combo_row${ROWS}_hd${HIGH_DROP_CHUNK}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}";RESULT="${RESULT:-${PREFIX}.tsv}";RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
ISO="${ISO:-$ONEESAN_BUILD_DIR/mainrec_block_combo_$$}"
mkdir -p "$LOGDIR" "$ISO" "$ISO/tmp" "$(dirname "$RESULT")"
[[ "$ROWS" =~ ^[0-9]+$ ]]&&((ROWS>=1&&ROWS<=28))||{ echo 'ROWS must be 1..28' >&2;exit 2; }
[[ "$REPEATS" =~ ^[0-9]+$ ]]&&((REPEATS>=1))||{ echo 'REPEATS must be >=1' >&2;exit 2; }
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]]||{ echo 'HIGH_DROP_CHUNK must be 0 or 1' >&2;exit 2; }
for b in $BATCH_LIST;do [[ "$b" == 2 || "$b" == 4 ]]||{ echo 'BATCH_LIST supports 2 4' >&2;exit 2; };done
command -v nvcc >/dev/null||{ echo 'nvcc required' >&2;exit 2; };command -v nvidia-smi >/dev/null||{ echo 'nvidia-smi required' >&2;exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)>=GPUS ))||{ echo 'need 8 visible GPUs' >&2;exit 2; }

# Build the proof-gated production source only once, with no post-transform.
BASE_BIN="$ONEESAN_BUILD_DIR/b300_mainrec_combo_base_hd${HIGH_DROP_CHUNK}_n27";BASE_OUT="$LOGDIR/base.build.out";BASE_ERR="$LOGDIR/base.build.err"
echo "=== combo source build MAIN_RECURRENCE highdrop=$HIGH_DROP_CHUNK ===" >&2
N=27 ARCH="$ARCH" OUT="$BASE_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" DUALMASK=0 BUILD_ERR="$BASE_ERR" CALIBRATED_BUILD_DIR="$ISO/basechain" PTXAS_VERBOSE=1 \
 bash "$ONEESAN_ROOT/scripts/build/b300-forced-calibrated.sh" >"$BASE_OUT" 2>"$LOGDIR/base.build.driver.err"
[[ -x "$BASE_BIN" ]]||{ echo 'base binary missing' >&2;exit 3; }
BASE_SRC="$(sed -nE 's/^  source_after_proof_gates=(.*)$/\1/p' "$BASE_OUT"|tail -n1)";[[ -n "$BASE_SRC" && -f "$BASE_SRC" ]]||{ echo 'base source missing' >&2;exit 3; }

bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-dualmask-proof.sh"
DUAL_SRC="$ISO/dual.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-pull-dualmask.py" "$BASE_SRC" "$DUAL_SRC" >"$LOGDIR/dual.transform.out"
BATCH_SRC="$ISO/batch.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-quad.py" "$BASE_SRC" "$BATCH_SRC" >"$LOGDIR/batch.transform.out"
DUAL_BATCH_SRC="$ISO/dual_batch.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-quad.py" "$DUAL_SRC" "$DUAL_BATCH_SRC" >"$LOGDIR/dual_batch.transform.out"
grep -Fq 'b300_block_endpoint_masks(d)' "$DUAL_SRC";grep -Fq 'B300_BLOCK_CLOSURE_BATCH' "$BATCH_SRC";grep -Fq 'b300_block_endpoint_masks(d)' "$DUAL_BATCH_SRC";grep -Fq 'B300_BLOCK_CLOSURE_BATCH' "$DUAL_BATCH_SRC"

: >"$LOGDIR/variants.tsv";printf 'base\t0\t0\t%s\t%s\n' "$BASE_BIN" "$BASE_ERR" >>"$LOGDIR/variants.tsv"
compile_one(){ local mode="$1" dual="$2" batch="$3" src="$4" bin="$ONEESAN_BUILD_DIR/b300_mainrec_combo_${mode}_hd${HIGH_DROP_CHUNK}_n27" err="$LOGDIR/$mode.build.err";echo "=== compile $mode dual=$dual batch=$batch ===" >&2;local defs=();((batch))&&defs+=("-DB300_BLOCK_CLOSURE_BATCH=$batch");TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v -DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13 "${defs[@]}" "$src" -o "$bin" >"$LOGDIR/$mode.build.out" 2>"$err";[[ -x "$bin" ]]||{ echo "missing $mode binary" >&2;exit 3; };printf '%s\t%s\t%s\t%s\t%s\n' "$mode" "$dual" "$batch" "$bin" "$err" >>"$LOGDIR/variants.tsv"; }
compile_one dual 1 0 "$DUAL_SRC"
for b in $BATCH_LIST;do compile_one "batch${b}" 0 "$b" "$BATCH_SRC";compile_one "dual_batch${b}" 1 "$b" "$DUAL_BATCH_SRC";done

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py";printf 'mode\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
while IFS=$'\t' read -r mode _ _ _ err;do python3 "$PARSER" "$err" --label "$mode" --contains block_pull_kernel >>"$RESOURCE"||true;done <"$LOGDIR/variants.tsv"
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
printf 'mode\tdualmask\tclosure_batch\tthreads\trepeat\tresidue\twall_s\tmem_avg_pct\tmem_max_pct\tmem_busy_avg_pct\tsm_avg_pct\tpower_avg_w\tsamples\n' >"$RESULT"
run_one(){ local mode="$1" dual="$2" batch="$3" bin="$4" t="$5" rep="$6" tag="${mode}_t${t}_r${rep}" out="$LOGDIR/$tag.out" err="$LOGDIR/$tag.err" tele="$LOGDIR/$tag.gpu.csv";nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits -lms 200 >"$tele" 2>/dev/null&local mon=$!;sleep 1;set +e;B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$t" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$out" 2>"$err";local rc=$?;set -e;kill "$mon" 2>/dev/null||true;wait "$mon" 2>/dev/null||true;((rc==0))||{ echo "$tag failed rc=$rc" >&2;tail -n 140 "$err" >&2||true;return "$rc"; };local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$out"|tail -n1||true)";[[ -n "$line" ]]||return 4;grep -Fq " rows=$ROWS calibration=$((ROWS<28?1:0)) "<<<"$line"||return 5;local stats="$(summarize "$tele")";printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$dual" "$batch" "$t" "$rep" "$(field residue "$line")" "$(field wall_s "$line")" "$stats" >>"$RESULT"; }
for t in $THREADS_LIST;do [[ "$t" =~ ^[0-9]+$ ]]&&((t>=32&&t<=1024&&t%32==0))||{ echo "bad threads=$t" >&2;exit 2; };for((r=1;r<=REPEATS;++r));do mapfile -t vv <"$LOGDIR/variants.tsv";if((!(r&1)));then mapfile -t vv < <(tac "$LOGDIR/variants.tsv");fi;for x in "${vv[@]}";do IFS=$'\t' read -r m d b bin _<<<"$x";run_one "$m" "$d" "$b" "$bin" "$t" "$r";done;done;done

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
if not r:raise SystemExit('no combo rows')
res={x['residue'] for x in r}
if len(res)!=1:raise SystemExit('FATAL block combo residue mismatch '+repr(sorted(res)))
by={}
for x in r:by.setdefault((x['mode'],int(x['dualmask']),int(x['closure_batch']),int(x['threads'])),[]).append(float(x['wall_s']))
med={k:statistics.median(v) for k,v in by.items()};base=min((v,k[3]) for k,v in med.items() if k[0]=='base')
best=min((v,k) for k,v in med.items());w,k=best;mode,dual,batch,t=k
print('b300_mainrec_block_combo_exact_intermediate_match=1')
print(f'b300_mainrec_block_combo_residue={next(iter(res))}')
print(f'b300_mainrec_block_combo_base_best_wall_s={base[0]:.9f}')
print(f'b300_mainrec_block_combo_base_best_threads={base[1]}')
for k,v in sorted(med.items(),key=lambda z:z[1]):print(f'  mode={k[0]} dual={k[1]} batch={k[2]} threads={k[3]} wall_s={v:.9f}')
print(f'b300_mainrec_block_combo_best_mode={mode}')
print(f'b300_mainrec_block_combo_best_dualmask={dual}')
print(f'b300_mainrec_block_combo_best_closure_batch={batch}')
print(f'b300_mainrec_block_combo_best_threads={t}')
print(f'b300_mainrec_block_combo_best_wall_s={w:.9f}')
print(f'b300_mainrec_block_combo_speedup_vs_base_best={base[0]/w:.9f}x')
PY
printf 'b300_mainrec_block_combo_high_drop_chunk=%s\n' "$HIGH_DROP_CHUNK";printf 'b300_mainrec_block_combo_rows=%s\n' "$ROWS";cat "$RESOURCE"
