#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}";MOD="${MOD:-4294967291}";ROWS="${ROWS:-1}";TARGET_MIB="${TARGET_MIB:-65536}";MAX_WINDOW="${MAX_WINDOW:-14}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}";RECURRENCE_ILP="${RECURRENCE_ILP:-2}";RANDOM_CG="${RANDOM_CG:-0}";DUALMASK="${DUALMASK:-0}";CLOSURE_BATCH="${CLOSURE_BATCH:-0}"
REGCAP_LIST="${REGCAP_LIST:-0 96 128 160 192 224}";THREADS_LIST="${THREADS_LIST:-128 256 512}";REPEATS="${REPEATS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_nextgen_latency_h${HIGH_DROP_CHUNK}_i${RECURRENCE_ILP}_c${RANDOM_CG}_d${DUALMASK}_b${CLOSURE_BATCH}_row${ROWS}}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";RESULT="${RESULT:-${PREFIX}.tsv}";RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")"
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]]||exit 2;case "$RECURRENCE_ILP" in 2|4|8);;*)exit 2;;esac;[[ "$RANDOM_CG" == 0 || "$RANDOM_CG" == 1 ]]||exit 2;[[ "$DUALMASK" == 0 || "$DUALMASK" == 1 ]]||exit 2;case "$CLOSURE_BATCH" in 0|2|4);;*)exit 2;;esac
command -v nvcc >/dev/null||{ echo 'nvcc required' >&2;exit 2; };command -v nvidia-smi >/dev/null||{ echo 'nvidia-smi required' >&2;exit 2; };(( $(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)>=8 ))||{ echo 'need 8 visible GPUs' >&2;exit 2; }

# Resolve one fully composed, uncapped source. Recompile it below so every
# resource log contains only the exact tested nvcc invocation.
GEN_BIN="$ONEESAN_BUILD_DIR/b300_nextgen_latency_source_n27";GEN_OUT="$LOGDIR/source.build.out";GEN_ERR="$LOGDIR/source.build.err"
N=27 ARCH="$ARCH" OUT="$GEN_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" RECURRENCE_ILP="$RECURRENCE_ILP" RANDOM_CG="$RANDOM_CG" PREFETCH_L2=0 DUALMASK="$DUALMASK" CLOSURE_BATCH="$CLOSURE_BATCH" MAXRREGCOUNT=0 BUILD_ERR="$GEN_ERR" PTXAS_VERBOSE=1 \
 bash "$ONEESAN_ROOT/scripts/build/b300-forced-nextgen.sh">"$GEN_OUT" 2>"$LOGDIR/source.build.driver.err"
SYNC_SRC="$(sed -nE 's/^  source_after_all=(.*)$/\1/p' "$GEN_OUT"|tail -n1)";[[ -f "$SYNC_SRC" ]]||{ echo 'latency sweep source missing' >&2;exit 3; }
PREF_SRC="$LOGDIR/prefetch.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-prefetch-l2.py" "$SYNC_SRC" "$PREF_SRC">"$LOGDIR/prefetch.transform.out";grep -Fq 'mainrec_prefetch_l2=1' "$LOGDIR/prefetch.transform.out"

DEFS=(-DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13);[[ "$CLOSURE_BATCH" == 0 ]]||DEFS+=("-DB300_BLOCK_CLOSURE_BATCH=$CLOSURE_BATCH")
printf 'profile\tprefetch_l2\tcap\tbinary\tbuild_err\n'>"$LOGDIR/binaries.tsv"
declare -A seen=()
for cap in $REGCAP_LIST;do
 [[ "$cap" =~ ^[0-9]+$ ]]&&((cap==0||(cap>=32&&cap<=255)))||{ echo "bad cap=$cap" >&2;exit 2; };[[ -z "${seen[$cap]+x}" ]]||continue;seen[$cap]=1
 for pre in 0 1;do profile="$([[ "$pre" == 1 ]]&&echo prefetch||echo sync)_r${cap}";src="$SYNC_SRC";[[ "$pre" == 0 ]]||src="$PREF_SRC";bin="$ONEESAN_BUILD_DIR/b300_nextgen_${profile}_h${HIGH_DROP_CHUNK}_i${RECURRENCE_ILP}_c${RANDOM_CG}_d${DUALMASK}_b${CLOSURE_BATCH}_n27";err="$LOGDIR/${profile}.build.err";reg=();((cap>0))&&reg+=("-maxrregcount=$cap");echo "=== compile $profile ===" >&2;set +e;TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v "${reg[@]}" "${DEFS[@]}" "$src" -o "$bin">"$LOGDIR/${profile}.compile.out" 2>"$err";rc=$?;set -e;if((rc));then echo "LATENCY_COMPILE_SKIP profile=$profile rc=$rc" >&2;continue;fi;[[ -x "$bin" ]]||continue;printf '%s\t%s\t%s\t%s\t%s\n' "$profile" "$pre" "$cap" "$bin" "$err">>"$LOGDIR/binaries.tsv";done
done
(( $(wc -l<"$LOGDIR/binaries.tsv")>1 ))||{ echo 'no latency candidate compiled' >&2;exit 3; }

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py";printf 'profile\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n'>"$RESOURCE"
while IFS=$'\t' read -r profile pre cap bin err;do [[ "$profile" == profile ]]&&continue;python3 "$PARSER" "$err" --label "$profile" --contains main_pull_kernel_ilp2 >>"$RESOURCE"||true;done<"$LOGDIR/binaries.tsv"
field(){ local k="$1" l="$2";sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p"<<<"$l"|tail -n1; }
printf 'profile\tprefetch_l2\tcap\tthreads\trepeat\tresidue\twall_s\thigh_rec_groups\thigh_rec_fallback_groups\tmc_avg_pct\tmc_max_pct\n'>"$RESULT"
run_one(){ local profile="$1" pre="$2" cap="$3" bin="$4" t="$5" r="$6" so="$LOGDIR/${profile}_t${t}_rep${r}.out" se="$LOGDIR/${profile}_t${t}_rep${r}.err" dm="$LOGDIR/${profile}_t${t}_rep${r}.dmon";:>"$dm";nvidia-smi dmon -s u -d 1>"$dm" 2>&1&pid=$!;set +e;B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$t" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD">"$so" 2>"$se";rc=$?;set -e;kill "$pid" 2>/dev/null||true;wait "$pid" 2>/dev/null||true;((rc==0))||{ echo "$profile t=$t failed rc=$rc" >&2;return "$rc";};line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so"|tail -n1||true)";[[ -n "$line" ]]||return 4;hg="$(field high_rec_groups "$line")";hf="$(field high_rec_fallback_groups "$line")";[[ "$hg" =~ ^[0-9]+$ ]]&&((hg>0))||return 5;read -r mcavg mcmax < <(awk '$1~/^[0-9]+$/&&$3~/^[0-9]+$/{s+=$3;n++;if($3>m)m=$3}END{if(n)printf "%.3f %d\n",s/n,m;else print "nan nan"}' "$dm");printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$profile" "$pre" "$cap" "$t" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$hg" "$hf" "$mcavg" "$mcmax">>"$RESULT"; }
for t in $THREADS_LIST;do [[ "$t" =~ ^[0-9]+$ ]]&&((t>=32&&t<=1024&&t%32==0))||exit 2;while IFS=$'\t' read -r profile pre cap bin err;do [[ "$profile" == profile ]]&&continue;for((r=1;r<=REPEATS;++r));do echo "=== $profile threads=$t repeat=$r ===" >&2;run_one "$profile" "$pre" "$cap" "$bin" "$t" "$r";done;done<"$LOGDIR/binaries.tsv";done

python3 - "$RESULT" "$RESOURCE" <<'PY'
import csv,math,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));rr=list(csv.DictReader(open(sys.argv[2]),delimiter='\t'))
if not rows:raise SystemExit('no latency results')
res={r['residue'] for r in rows}
if len(res)!=1:raise SystemExit('FATAL latency partial residue mismatch '+repr({(r['profile'],r['threads']):r['residue'] for r in rows}))
resource={}
for r in rr:
 try:resource[r['profile']]=(int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes']))
 except:pass
by={}
for r in rows:by.setdefault((r['profile'],int(r['prefetch_l2']),int(r['cap']),int(r['threads'])),[]).append(r)
med=[]
for (p,pre,c,t),rs in by.items():
 w=statistics.median(float(r['wall_s']) for r in rs);mv=[float(r['mc_avg_pct']) for r in rs if r['mc_avg_pct']!='nan'];mc=statistics.median(mv) if mv else math.nan;regs,ss,sl=resource.get(p,(-1,-1,-1));med.append((w,p,pre,c,t,mc,regs,ss,sl))
for x in sorted(med):print(f'profile={x[1]} prefetch={x[2]} cap={x[3]} threads={x[4]} wall={x[0]:.9f} mc={x[5]:.3f} regs={x[6]} spill_store={x[7]} spill_load={x[8]}',file=sys.stderr)
base=min(x for x in med if x[2]==0 and x[3]==0)
clean=[x for x in med if x[6]>=0 and x[7]==0 and x[8]==0]
if not clean:raise SystemExit('no spill-free latency candidate with known ptxas data')
best=min(clean,key=lambda x:(x[0],-x[5] if x[5]==x[5] else math.inf))
print('b300_nextgen_latency_exact_intermediate_match=1')
print(f'b300_nextgen_latency_residue={next(iter(res))}')
print(f'b300_nextgen_latency_base_threads={base[4]}')
print(f'b300_nextgen_latency_base_wall_s={base[0]:.9f}')
print(f'b300_nextgen_latency_best_profile={best[1]}')
print(f'b300_nextgen_latency_best_prefetch_l2={best[2]}')
print(f'b300_nextgen_latency_best_cap={best[3]}')
print(f'b300_nextgen_latency_best_threads={best[4]}')
print(f'b300_nextgen_latency_best_wall_s={best[0]:.9f}')
print(f'b300_nextgen_latency_best_mc_avg_pct={best[5]:.3f}')
print(f'b300_nextgen_latency_best_registers={best[6]}')
print(f'b300_nextgen_latency_best_spill_store_bytes={best[7]}')
print(f'b300_nextgen_latency_best_spill_load_bytes={best[8]}')
print(f'b300_nextgen_latency_speedup_vs_uncapped_sync={base[0]/best[0]:.9f}x')
PY
cat "$RESOURCE"
echo "b300-nextgen-latency-regcap-sweep OK result=$RESULT resources=$RESOURCE" >&2
