#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}";MOD="${MOD:-4294967291}";ROWS="${ROWS:-1}";TARGET_MIB="${TARGET_MIB:-65536}";MAX_WINDOW="${MAX_WINDOW:-14}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}";RECURRENCE_ILP="${RECURRENCE_ILP:-2}";RANDOM_CG="${RANDOM_CG:-0}";THREADS_LIST="${THREADS_LIST:-128 256 512}";REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_block_hd${HIGH_DROP_CHUNK}_ilp${RECURRENCE_ILP}_cg${RANDOM_CG}_row${ROWS}}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";RESULT="${RESULT:-${PREFIX}.tsv}";RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")"
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]]||exit 2;case "$RECURRENCE_ILP" in 2|4|8);;*)exit 2;;esac;[[ "$RANDOM_CG" == 0 || "$RANDOM_CG" == 1 ]]||exit 2
[[ "$ROWS" =~ ^[0-9]+$ ]]&&((ROWS>=1&&ROWS<=28))||exit 2;[[ "$REPEATS" =~ ^[0-9]+$ ]]&&((REPEATS>=1))||exit 2
command -v nvcc >/dev/null||{ echo 'nvcc required' >&2;exit 2; };command -v nvidia-smi >/dev/null||{ echo 'nvidia-smi required' >&2;exit 2; };(( $(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)>=8 ))||{ echo 'need 8 visible GPUs' >&2;exit 2; }

BASE_BIN="$ONEESAN_BUILD_DIR/b300_nextgen_block_base_hd${HIGH_DROP_CHUNK}_ilp${RECURRENCE_ILP}_cg${RANDOM_CG}_n27";BASE_OUT="$LOGDIR/base.build.out";BASE_ERR="$LOGDIR/base.build.err"
N=27 ARCH="$ARCH" OUT="$BASE_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" RECURRENCE_ILP="$RECURRENCE_ILP" RANDOM_CG="$RANDOM_CG" DUALMASK=0 CLOSURE_BATCH=0 BUILD_ERR="$BASE_ERR" PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh" >"$BASE_OUT" 2>"$LOGDIR/base.build.driver.err"
SRC="$(sed -nE 's/^  source_after_all=(.*)$/\1/p' "$BASE_OUT"|tail -n1)";[[ -x "$BASE_BIN" && -f "$SRC" ]]||{ echo 'nextgen block base source/bin missing' >&2;exit 3; }
printf 'base\t0\t0\t%s\n' "$BASE_BIN" >"$LOGDIR/binaries.tsv"

bash "$ONEESAN_ROOT/scripts/bench/b300-block-pull-dualmask-proof.sh" >"$LOGDIR/dualmask-proof.out" 2>"$LOGDIR/dualmask-proof.err"
DUAL_SRC="$LOGDIR/dual.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-pull-dualmask.py" "$SRC" "$DUAL_SRC" >"$LOGDIR/dual.transform.out"

compile(){ local mode="$1" dual="$2" batch="$3" src="$4" bin="$ONEESAN_BUILD_DIR/b300_nextgen_block_${mode}_hd${HIGH_DROP_CHUNK}_ilp${RECURRENCE_ILP}_cg${RANDOM_CG}_n27";local defs=(-DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13);[[ "$batch" == 0 ]]||defs+=("-DB300_BLOCK_CLOSURE_BATCH=$batch");TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v "${defs[@]}" "$src" -o "$bin" >"$LOGDIR/$mode.compile.out" 2>"$LOGDIR/$mode.build.err";[[ -x "$bin" ]]||exit 3;printf '%s\t%s\t%s\t%s\n' "$mode" "$dual" "$batch" "$bin" >>"$LOGDIR/binaries.tsv"; }
compile dual 1 0 "$DUAL_SRC"
for batch in 2 4;do q="$LOGDIR/batch${batch}.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-quad.py" "$SRC" "$q" >"$LOGDIR/batch${batch}.transform.out";compile "batch${batch}" 0 "$batch" "$q";dq="$LOGDIR/dualbatch${batch}.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-closure-quad.py" "$DUAL_SRC" "$dq" >"$LOGDIR/dualbatch${batch}.transform.out";compile "dualbatch${batch}" 1 "$batch" "$dq";done

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py";printf 'mode\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
python3 "$PARSER" "$BASE_ERR" --label base --contains block_pull_kernel >>"$RESOURCE"||true
for mode in dual batch2 batch4 dualbatch2 dualbatch4;do python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" --contains block_pull_kernel >>"$RESOURCE"||true;done
field(){ local k="$1" l="$2";sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p"<<<"$l"|tail -n1; }
printf 'mode\tdualmask\tclosure_batch\tthreads\trepeat\tresidue\twall_s\thigh_rec_groups\thigh_rec_fallback_groups\tmc_avg_pct\tmc_max_pct\n' >"$RESULT"
run_one(){ local mode="$1" dual="$2" batch="$3" bin="$4" t="$5" r="$6" so="$LOGDIR/${mode}_t${t}_r${r}.out" se="$LOGDIR/${mode}_t${t}_r${r}.err" dmon="$LOGDIR/${mode}_t${t}_r${r}.dmon";:>"$dmon";nvidia-smi dmon -s u -d 1 >"$dmon" 2>&1&mpid=$!;set +e;B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$t" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se";rc=$?;set -e;kill "$mpid" 2>/dev/null||true;wait "$mpid" 2>/dev/null||true;((rc==0))||{ echo "$mode t=$t failed rc=$rc" >&2;tail -n120 "$se" >&2||true;return "$rc";};line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so"|tail -n1||true)";[[ -n "$line" ]]||return 4;hg="$(field high_rec_groups "$line")";hf="$(field high_rec_fallback_groups "$line")";[[ "$hg" =~ ^[0-9]+$ ]]&&((hg>0))||return 5;read -r mcavg mcmax < <(awk '$1~/^[0-9]+$/&&$3~/^[0-9]+$/{s+=$3;n++;if($3>m)m=$3}END{if(n)printf "%.3f %d\n",s/n,m;else print "nan nan"}' "$dmon");printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$dual" "$batch" "$t" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$hg" "$hf" "$mcavg" "$mcmax" >>"$RESULT"; }
for t in $THREADS_LIST;do [[ "$t" =~ ^[0-9]+$ ]]&&((t>=32&&t<=1024&&t%32==0))||exit 2;while IFS=$'\t' read -r mode dual batch bin;do for((r=1;r<=REPEATS;++r));do echo "=== $mode threads=$t repeat=$r ===" >&2;run_one "$mode" "$dual" "$batch" "$bin" "$t" "$r";done;done<"$LOGDIR/binaries.tsv";done

python3 - "$RESULT" "$HIGH_DROP_CHUNK" "$RECURRENCE_ILP" "$RANDOM_CG" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));hd,ilp,cg=sys.argv[2:5]
if not rows:raise SystemExit('no block combo results')
res={r['residue'] for r in rows}
if len(res)!=1:raise SystemExit('FATAL nextgen block combo residue mismatch '+repr({(r['mode'],r['threads']):r['residue'] for r in rows}))
by={}
for r in rows:by.setdefault((r['mode'],r['dualmask'],r['closure_batch'],int(r['threads'])),[]).append(r)
med=[]
for (m,d,b,t),rs in by.items():
 w=statistics.median(float(r['wall_s']) for r in rs);mv=[float(r['mc_avg_pct']) for r in rs if r['mc_avg_pct']!='nan'];mc=statistics.median(mv) if mv else float('nan');med.append((w,m,d,b,t,mc))
for w,m,d,b,t,mc in sorted(med):print(f'{m} dual={d} batch={b} threads={t} median_wall_s={w:.9f} median_mc_avg_pct={mc:.3f}',file=sys.stderr)
base=min(x for x in med if x[1]=='base');best=min(med,key=lambda x:(x[0],-x[5] if x[5]==x[5] else 1e100))
print('b300_nextgen_block_combo_exact_intermediate_match=1')
print(f'b300_nextgen_block_combo_high_drop_chunk={hd}')
print(f'b300_nextgen_block_combo_recurrence_ilp={ilp}')
print(f'b300_nextgen_block_combo_random_cg={cg}')
print(f'b300_nextgen_block_combo_residue={next(iter(res))}')
print(f'b300_nextgen_block_combo_base_best_threads={base[4]}')
print(f'b300_nextgen_block_combo_base_best_wall_s={base[0]:.9f}')
print(f'b300_nextgen_block_combo_best_mode={best[1]}')
print(f'b300_nextgen_block_combo_best_dualmask={best[2]}')
print(f'b300_nextgen_block_combo_best_closure_batch={best[3]}')
print(f'b300_nextgen_block_combo_best_threads={best[4]}')
print(f'b300_nextgen_block_combo_best_wall_s={best[0]:.9f}')
print(f'b300_nextgen_block_combo_best_mc_avg_pct={best[5]:.3f}')
print(f'b300_nextgen_block_combo_speedup_vs_base={base[0]/best[0]:.9f}x')
PY
cat "$RESOURCE"
echo "b300-nextgen-block-combo-sweep OK result=$RESULT resources=$RESOURCE" >&2
