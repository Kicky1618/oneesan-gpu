#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"; THREADS="${GRIDFP_THREADS:-256}"
TARGET_MIB="${TARGET_MIB:-65536}"; PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; RANDOM_CG="${RANDOM_CG:-1}"; WARP_SCAN="${WARP_SCAN:-1}"; MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"; LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_ilp8_cpasync_stage_race_n${N}}"
SUMMARY="$LOGDIR/summary.tsv"; WINNER_ENV="$LOGDIR/winner.env"; PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
ISO="$LOGDIR/base_gen"; BASE_BIN="$LOGDIR/base"; mkdir -p "$LOGDIR" "$ISO" "$ISO/tmp"
[[ "$N" == 27 ]] || { echo 'cp.async stage race targets n=27' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32&&THREADS<=768&&THREADS%32==0)) || { echo 'GRIDFP_THREADS must be warp multiple 32..768' >&2; exit 2; }
command -v nvcc >/dev/null; command -v nvidia-smi >/dev/null
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

sample_mem(){ local pid="$1" out="$2";:>"$out";while kill -0 "$pid" 2>/dev/null;do nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null|awk '{s+=$1;n++}END{if(n)printf "%.6f\n",s/n}'>>"$out"||true;sleep "$SAMPLE_INTERVAL";done; }
field(){ local k="$1" l="$2";sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p"<<<"$l"|tail -n1; }

# One stock cp.async build+run also materializes the exact source from which all
# staged candidates are derived.
BASE_LOG="$LOGDIR/base.run.log";BASE_MEM="$LOGDIR/base.mem"
set +e
env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" ARCH="$ARCH" RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" PREFETCH_L2=0 CPASYNC=1 MAXRREGCOUNT="$MAXRREGCOUNT" REBUILD=1 ISO="$ISO" BIN="$BASE_BIN" "$ONEESAN_ROOT/scripts/run/b300x8-saturate-ilp8.sh" "$N" "$MOD" >"$BASE_LOG" 2>&1 &
pid=$!;sample_mem "$pid" "$BASE_MEM"&mpid=$!;wait "$pid";rc=$?;set -e;wait "$mpid" 2>/dev/null||true
((rc==0))||{ cat "$BASE_LOG";exit "$rc"; }
BASE_SRC="$ISO/final_main_ilp8_cpasync.cu";BASE_BUILD="$ISO/final.build.err"
[[ -f "$BASE_SRC" && -f "$BASE_BUILD" && -x "$BASE_BIN" ]]||{ echo 'stock cp.async artifacts missing' >&2;exit 3; }

build_variant(){ local name="$1" gen="$2" input="$BASE_SRC" src="$LOGDIR/$name.cu" bin="$LOGDIR/$name" blog="$LOGDIR/$name.build.log";
  if [[ "$name" == pair_nextself ]];then local mid="$LOGDIR/pair_nextself.stage1.cu";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-rankstate-ilp8-cpasync-staged-wait-u32.py" "$BASE_SRC" "$mid">"$LOGDIR/$name.transform.log";python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-rankstate-ilp8-next-self-prefetch.py" "$mid" "$src">>"$LOGDIR/$name.transform.log";
  else python3 "$ONEESAN_ROOT/scripts/build/$gen" "$input" "$src">"$LOGDIR/$name.transform.log";fi
  REG=();((MAXRREGCOUNT>0))&&REG+=("-maxrregcount=$MAXRREGCOUNT");TMPDIR="$ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v "${REG[@]}" -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=1 -DB300_BLOCK_CLOSURE_QUAD=0 "$src" -o "$bin" 2>"$blog";[[ -x "$bin" ]]||exit 3; }
build_variant pair_u32 gen-b300-main-rankstate-ilp8-cpasync-staged-wait-u32.py
build_variant block_u32 gen-b300-main-rankstate-ilp8-cpasync-staged-wait-u32-blockfirst.py
build_variant pair_nextself ignored

run_direct(){ local name="$1" bin="$2" log="$LOGDIR/$name.run.log" mem="$LOGDIR/$name.mem";set +e;B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" "$bin" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8>"$log" 2>&1&local p=$!;sample_mem "$p" "$mem"&local m=$!;wait "$p";local r=$?;set -e;wait "$m" 2>/dev/null||true;((r==0))||{ cat "$log";exit "$r"; }; }
run_direct pair_u32 "$LOGDIR/pair_u32";run_direct block_u32 "$LOGDIR/block_u32";run_direct pair_nextself "$LOGDIR/pair_nextself"

printf 'profile\tresidue\twall_s\tactive_max_s\tactive_sum_s\tmc_avg_pct\tmc_max_pct\tmc_samples\tregs_main\tspill_store_main_bytes\tspill_load_main_bytes\tbinary\n'>"$SUMMARY"
parse(){ local name="$1" log="$2" mem="$3" build="$4" bin="$5" line res wall am as ma mm mn tmp regs ss sl;line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log"|tail -n1||true)";[[ -n "$line" ]]||line="$(grep '^backend=gridfp-b300-hbm32' "$log"|tail -n1||true)";[[ -n "$line" ]]||exit 4;res="$(field residue "$line")";wall="$(field wall_s "$line")";am="$(field active_max_s "$line")";as="$(field active_sum_s "$line")";read -r ma mm mn < <(awk '{s+=$1;n++;if($1>m)m=$1}END{if(n)printf "%.3f %.3f %d\n",s/n,m,n;else print "nan nan 0"}' "$mem");tmp="$LOGDIR/$name.ptxas.tsv";if python3 "$PARSER" "$build" --label "$name" --contains b300_main_pull_rankstate_ilp8_kernel>"$tmp" 2>"$tmp.err";then read -r regs ss sl < <(python3 - "$tmp" <<'PY'
import csv,sys
v=[]
for r in csv.reader(open(sys.argv[1]),delimiter='\t'):
 try:v.append((int(r[2]),int(r[4]),int(r[5])))
 except (ValueError,IndexError):pass
print('nan nan nan' if not v else f'{max(x[0] for x in v)} {max(x[1] for x in v)} {max(x[2] for x in v)}')
PY
);else regs=nan;ss=nan;sl=nan;fi;printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$res" "$wall" "${am:-nan}" "${as:-nan}" "$ma" "$mm" "$mn" "$regs" "$ss" "$sl" "$bin">>"$SUMMARY"; }
parse base "$BASE_LOG" "$BASE_MEM" "$BASE_BUILD" "$BASE_BIN"
for x in pair_u32 block_u32 pair_nextself;do parse "$x" "$LOGDIR/$x.run.log" "$LOGDIR/$x.mem" "$LOGDIR/$x.build.log" "$LOGDIR/$x";done
cat "$SUMMARY"
python3 - "$SUMMARY" "$WINNER_ENV" <<'PY'
import csv,sys,math,shlex
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));res={r['residue'] for r in rows}
if len(res)!=1:raise SystemExit('FATAL staged cp.async residue mismatch '+repr({r['profile']:r['residue'] for r in rows}))
def f(r,k,d=math.inf):
 try:return float(r[k])
 except:return d
def clean(r):return r['spill_store_main_bytes']=='0' and r['spill_load_main_bytes']=='0'
c=[r for r in rows if clean(r)]
if not c:raise SystemExit('no spill-free staged cp.async candidate')
best=min(c,key=lambda r:(f(r,'wall_s'),-f(r,'mc_avg_pct',-math.inf)))
base=next(r for r in rows if r['profile']=='base')
print(f"STAGE_RACE_WINNER profile={best['profile']} wall_s={best['wall_s']} speedup_vs_base={f(base,'wall_s')/f(best,'wall_s'):.6f}x mc_avg_pct={best['mc_avg_pct']} mc_delta={f(best,'mc_avg_pct',0)-f(base,'mc_avg_pct',0):.3f}pp regs={best['regs_main']} spill_free=1 exact_gate=1",file=sys.stderr)
with open(sys.argv[2],'w') as o:
 for k,v in [('B300_STAGE_WINNER_PROFILE',best['profile']),('B300_STAGE_WINNER_BIN',best['binary']),('B300_STAGE_WINNER_WALL_S',best['wall_s']),('B300_STAGE_WINNER_MC_AVG_PCT',best['mc_avg_pct']),('B300_STAGE_WINNER_REGS_MAIN',best['regs_main'])]:o.write(k+'='+shlex.quote(v)+'\n')
PY
echo "b300x8-ilp8-cpasync-stage-race OK summary=$SUMMARY winner_env=$WINNER_ENV" >&2
