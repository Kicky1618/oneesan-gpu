#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}";MOD="${MOD:-4294967291}";ROWS="${ROWS:-1}";TARGET_MIB="${TARGET_MIB:-65536}";MAX_WINDOW="${MAX_WINDOW:-14}";HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
THREADS_LIST="${THREADS_LIST:-128 256 512}";REPEATS="${REPEATS:-1}";PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_ilpcg_hd${HIGH_DROP_CHUNK}_row${ROWS}}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";RESULT="${RESULT:-${PREFIX}.tsv}";RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")"
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]]||exit 2
command -v nvcc >/dev/null||{ echo 'nvcc required' >&2;exit 2; };command -v nvidia-smi >/dev/null||{ echo 'nvidia-smi required' >&2;exit 2; };(( $(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)>=8 ))||{ echo 'need 8 visible GPUs' >&2;exit 2; }
python3 "$ONEESAN_ROOT/scripts/bench/b300-main-recurrence-ilp-partition-proof.py"

BASE_BIN="$ONEESAN_BUILD_DIR/b300_mainrec_ilpcg_ilp2_hd${HIGH_DROP_CHUNK}_n27";BASE_OUT="$LOGDIR/ilp2.build.out";BASE_ERR="$LOGDIR/ilp2.build.err"
N=27 ARCH="$ARCH" OUT="$BASE_BIN" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" DUALMASK=0 BUILD_ERR="$BASE_ERR" PTXAS_VERBOSE=1 bash "$ONEESAN_ROOT/scripts/build/b300-forced-calibrated.sh" >"$BASE_OUT" 2>"$LOGDIR/ilp2.build.driver.err"
SRC="$(sed -nE 's/^  source_after_proof_gates=(.*)$/\1/p' "$BASE_OUT"|tail -n1)";[[ -x "$BASE_BIN" && -f "$SRC" ]]||{ echo 'missing base source/bin' >&2;exit 3; }
printf 'ilp2\t%s\n' "$BASE_BIN" >"$LOGDIR/binaries.tsv"

compile_mode(){ local mode="$1" src="$2" bin="$ONEESAN_BUILD_DIR/b300_mainrec_ilpcg_${mode}_hd${HIGH_DROP_CHUNK}_n27";TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v -DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13 "$src" -o "$bin" >"$LOGDIR/$mode.compile.out" 2>"$LOGDIR/$mode.build.err";[[ -x "$bin" ]]||exit 3;printf '%s\t%s\n' "$mode" "$bin" >>"$LOGDIR/binaries.tsv"; }
CG2="$LOGDIR/ilp2cg.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-random-cg.py" "$SRC" "$CG2" >"$LOGDIR/ilp2cg.transform.out";compile_mode ilp2cg "$CG2"
for lanes in 4 8;do raw="$LOGDIR/ilp${lanes}.cu";cg="$LOGDIR/ilp${lanes}cg.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence-ilp.py" "$SRC" "$raw" "$lanes" >"$LOGDIR/ilp${lanes}.transform.out";compile_mode "ilp${lanes}" "$raw";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-random-cg.py" "$raw" "$cg" >"$LOGDIR/ilp${lanes}cg.transform.out";compile_mode "ilp${lanes}cg" "$cg";done

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py";printf 'mode\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for mode in ilp2 ilp2cg ilp4 ilp4cg ilp8 ilp8cg;do python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" --contains main_pull_kernel >>"$RESOURCE"||true;done
field(){ local k="$1" l="$2";sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p"<<<"$l"|tail -n1; }
printf 'mode\tthreads\trepeat\tresidue\twall_s\thigh_rec_groups\thigh_rec_fallback_groups\tmc_avg_pct\tmc_max_pct\n' >"$RESULT"
run_one(){
  local mode="$1" bin="$2" t="$3" r="$4" so="$LOGDIR/${mode}_t${t}_r${r}.out" se="$LOGDIR/${mode}_t${t}_r${r}.err" dmon="$LOGDIR/${mode}_t${t}_r${r}.dmon"
  : >"$dmon"; nvidia-smi dmon -s u -d 1 >"$dmon" 2>&1 & local mpid=$!
  set +e; B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$t" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se"; local rc=$?; set -e
  kill "$mpid" 2>/dev/null||true;wait "$mpid" 2>/dev/null||true
  ((rc==0))||{ echo "$mode t=$t failed rc=$rc" >&2;tail -n 120 "$se" >&2||true;return "$rc";}
  local line hg hf mc_avg mc_max
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so"|tail -n1||true)";[[ -n "$line" ]]||return 4
  hg="$(field high_rec_groups "$line")";hf="$(field high_rec_fallback_groups "$line")";[[ "$hg" =~ ^[0-9]+$ ]]&&((hg>0))||return 5
  read -r mc_avg mc_max < <(awk '$1~/^[0-9]+$/&&$3~/^[0-9]+$/{s+=$3;n++;if($3>m)m=$3}END{if(n)printf "%.3f %d\n",s/n,m;else print "nan nan"}' "$dmon")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$t" "$r" "$(field residue "$line")" "$(field wall_s "$line")" "$hg" "$hf" "$mc_avg" "$mc_max" >>"$RESULT"
}
for t in $THREADS_LIST;do [[ "$t" =~ ^[0-9]+$ ]]&&((t>=32&&t<=1024&&t%32==0))||exit 2;while IFS=$'\t' read -r mode bin;do for((r=1;r<=REPEATS;++r));do echo "=== $mode threads=$t repeat=$r ===" >&2;run_one "$mode" "$bin" "$t" "$r";done;done<"$LOGDIR/binaries.tsv";done
python3 - "$RESULT" "$HIGH_DROP_CHUNK" "$RESOURCE" <<'PY'
import csv,math,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));hd=sys.argv[2]
resources=list(csv.DictReader(open(sys.argv[3]),delimiter='\t'))
if not rows:raise SystemExit('no mainrec ILP/CG results')
res={r['residue'] for r in rows}
if len(res)!=1:raise SystemExit('FATAL mainrec ILP/CG residue mismatch '+repr({(r['mode'],r['threads']):r['residue'] for r in rows}))
rb={}
for r in resources:
    try: regs=int(r['registers']); ss=int(r['spill_store_bytes']); sl=int(r['spill_load_bytes'])
    except (ValueError,TypeError,KeyError): continue
    old=rb.get(r['mode'],(0,0,0)); rb[r['mode']]=(max(old[0],regs),max(old[1],ss),max(old[2],sl))
by={}
for r in rows:by.setdefault((r['mode'],int(r['threads'])),[]).append(r)
med=[]
for (m,t),rs in by.items():
    wall=statistics.median(float(r['wall_s']) for r in rs)
    mc=[float(r['mc_avg_pct']) for r in rs if r['mc_avg_pct']!='nan']
    mc_med=statistics.median(mc) if mc else float('nan')
    regs,ss,sl=rb.get(m,(-1,-1,-1))
    med.append((wall,m,t,mc_med,regs,ss,sl))
for w,m,t,mc,regs,ss,sl in sorted(med):print(f'{m} threads={t} median_wall_s={w:.9f} median_mc_avg_pct={mc:.3f} regs={regs} spill_store={ss} spill_load={sl}',file=sys.stderr)
def spill_free(x): return x[4]>=0 and x[5]==0 and x[6]==0
clean=[x for x in med if spill_free(x)]
pool=clean or med
key=lambda x:(x[0],-x[3] if not math.isnan(x[3]) else math.inf)
base=min((x for x in med if x[1]=='ilp2'),key=key);best=min(pool,key=key)
print('b300_mainrec_ilpcg_exact_intermediate_match=1')
print(f'b300_mainrec_ilpcg_high_drop_chunk={hd}')
print(f'b300_mainrec_ilpcg_residue={next(iter(res))}')
print(f'b300_mainrec_ilpcg_base_best_threads={base[2]}')
print(f'b300_mainrec_ilpcg_base_best_wall_s={base[0]:.9f}')
print(f'b300_mainrec_ilpcg_base_best_mc_avg_pct={base[3]:.3f}')
print(f'b300_mainrec_ilpcg_best_mode={best[1]}')
print(f'b300_mainrec_ilpcg_best_threads={best[2]}')
print(f'b300_mainrec_ilpcg_best_wall_s={best[0]:.9f}')
print(f'b300_mainrec_ilpcg_best_mc_avg_pct={best[3]:.3f}')
print(f'b300_mainrec_ilpcg_best_registers={best[4]}')
print(f'b300_mainrec_ilpcg_best_spill_store_bytes={best[5]}')
print(f'b300_mainrec_ilpcg_best_spill_load_bytes={best[6]}')
print(f'b300_mainrec_ilpcg_spill_free_pool={int(bool(clean))}')
print(f'b300_mainrec_ilpcg_speedup_vs_ilp2={base[0]/best[0]:.9f}x')
print(f'b300_mainrec_ilpcg_mc_delta_vs_ilp2={best[3]-base[3]:.3f}pp')
PY
cat "$RESOURCE"
echo "b300-mainrec-ilp-cg-sweep OK result=$RESULT resources=$RESOURCE" >&2
