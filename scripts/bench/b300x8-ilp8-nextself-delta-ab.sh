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
ARCH="${ARCH:-native}"
RANDOM_CG="${RANDOM_CG:-1}"
WARP_SCAN="${WARP_SCAN:-1}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
WAIT_VARIANT="${WAIT_VARIANT:-pairfirst}"
REPEATS="${REPEATS:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_ilp8_nextself_delta_${WAIT_VARIANT}_n${N}_rows${ROWS}}"
SUMMARY="${SUMMARY:-$LOGDIR/summary.tsv}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR"

[[ "$N" == 27 ]] || { echo 'next-self delta A/B targets n=27' >&2; exit 2; }
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1 && ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32 && THREADS<=768 && THREADS%32==0)) || { echo 'GRIDFP_THREADS must be warp multiple 32..768' >&2; exit 2; }
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0 || (MAXRREGCOUNT>=32 && MAXRREGCOUNT<=255))) || { echo 'MAXRREGCOUNT must be 0 or 32..255' >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo 'REPEATS must be >=1' >&2; exit 2; }
for x in RANDOM_CG WARP_SCAN; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
case "$WAIT_VARIANT" in
  pairfirst) WAIT_GEN="$ONEESAN_ROOT/scripts/build/gen-b300-main-rankstate-ilp8-cpasync-staged-wait-u32.py" ;;
  blockfirst) WAIT_GEN="$ONEESAN_ROOT/scripts/build/gen-b300-main-rankstate-ilp8-cpasync-staged-wait-u32-blockfirst.py" ;;
  *) echo 'WAIT_VARIANT must be pairfirst or blockfirst' >&2; exit 2;;
esac
NEXTSELF_GEN="$ONEESAN_ROOT/scripts/build/gen-b300-main-rankstate-ilp8-next-self-prefetch.py"
[[ -f "$WAIT_GEN" && -f "$NEXTSELF_GEN" ]] || { echo 'required transform generator missing' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }
python3 - "$SAMPLE_INTERVAL" <<'PY'
import sys
if float(sys.argv[1]) <= 0: raise SystemExit('SAMPLE_INTERVAL must be >0')
PY

field(){
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}

sample_mem(){
  local pid="$1" out="$2"; : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null |
      awk '{s+=$1;n++}END{if(n)printf "%.6f\n",s/n}' >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}

# Generate the canonical ordinary cp.async source once. Its timing is not part
# of this A/B; both measured candidates below derive from this exact source.
BASE_ISO="$LOGDIR/base_gen"
BASE_BIN="$LOGDIR/base_cpasync"
BASE_LOG="$LOGDIR/base_generation.log"
mkdir -p "$BASE_ISO" "$BASE_ISO/tmp"
env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
  ARCH="$ARCH" RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" PREFETCH_L2=0 CPASYNC=1 MAXRREGCOUNT="$MAXRREGCOUNT" \
  REBUILD=1 ISO="$BASE_ISO" BIN="$BASE_BIN" \
  "$ONEESAN_ROOT/scripts/run/b300x8-saturate-ilp8.sh" "$N" "$MOD" >"$BASE_LOG" 2>&1
BASE_SRC="$BASE_ISO/final_main_ilp8_cpasync.cu"
[[ -f "$BASE_SRC" ]] || { echo 'ordinary cp.async generated source missing' >&2; exit 3; }

STAGED_SRC="$LOGDIR/staged_${WAIT_VARIANT}.cu"
PREFETCH_SRC="$LOGDIR/staged_${WAIT_VARIANT}_nextself.cu"
python3 "$WAIT_GEN" "$BASE_SRC" "$STAGED_SRC" >"$LOGDIR/staged.transform.out"
python3 "$NEXTSELF_GEN" "$STAGED_SRC" "$PREFETCH_SRC" >"$LOGDIR/nextself.transform.out"
grep -Fq 'cp.async.wait_group 1' "$STAGED_SRC"
grep -Fq 'cp.async.wait_group 1' "$PREFETCH_SRC"
! grep -Fq 'b300_prefetch_next_self_l2' "$STAGED_SRC" || { echo 'control source unexpectedly has next-self prefetch' >&2; exit 3; }
grep -Fq 'b300_prefetch_next_self_l2' "$PREFETCH_SRC"
grep -Fq 'prefetch.global.L2' "$PREFETCH_SRC"

REG_FLAGS=(); ((MAXRREGCOUNT>0)) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
compile_one(){
  local name="$1" src="$2" bin="$LOGDIR/$name.bin" build="$LOGDIR/$name.build.err"
  TMPDIR="$BASE_ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v "${REG_FLAGS[@]}" \
    -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=1 -DB300_BLOCK_CLOSURE_QUAD=0 \
    "$src" -o "$bin" 2>"$build"
  [[ -x "$bin" ]] || { echo "$name binary missing" >&2; exit 3; }
  printf '%s\n' "$bin"
}
CONTROL_BIN="$(compile_one control "$STAGED_SRC")"
PREFETCH_BIN="$(compile_one nextself "$PREFETCH_SRC")"

printf 'profile\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tmc_avg_pct\tmc_max_pct\tmc_samples\tregs_main\tspill_store_main_bytes\tspill_load_main_bytes\n' >"$SUMMARY"

resource_triplet(){
  local build="$1" label="$2" out="$LOGDIR/$label.ptxas.tsv"
  if ! python3 "$PARSER" "$build" --label "$label" --contains b300_main_pull_rankstate_ilp8_kernel >"$out" 2>"$out.err"; then
    echo 'nan nan nan'; return
  fi
  python3 - "$out" <<'PY'
import csv,sys
v=[]
for r in csv.reader(open(sys.argv[1]),delimiter='\t'):
    if len(r)<8: continue
    try:v.append((int(r[2]),int(r[4]),int(r[5])))
    except ValueError: pass
print('nan nan nan' if not v else f'{max(x[0] for x in v)} {max(x[1] for x in v)} {max(x[2] for x in v)}')
PY
}

run_one(){
  local name="$1" bin="$2" build="$3" rep="$4"
  local log="$LOGDIR/${name}_r${rep}.log" mem="$LOGDIR/${name}_r${rep}.mem"
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" \
    "$bin" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$log" 2>&1 &
  local pid=$!; sample_mem "$pid" "$mem" & local mpid=$!
  wait "$pid"; local rc=$?
  set -e
  wait "$mpid" 2>/dev/null || true
  ((rc==0)) || { echo "$name repeat=$rep failed rc=$rc" >&2; tail -n 100 "$log" >&2 || true; return "$rc"; }
  local line residue wall active_max active_sum mc_avg mc_max mc_n regs spill_store spill_load
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || line="$(grep '^backend=gridfp-b300-hbm32' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$name missing backend result" >&2; return 4; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  active_max="$(field active_max_s "$line")"; active_sum="$(field active_sum_s "$line")"
  read -r mc_avg mc_max mc_n < <(awk '{s+=$1;n++;if($1>m)m=$1}END{if(n)printf "%.3f %.3f %d\n",s/n,m,n;else print "nan nan 0"}' "$mem")
  read -r regs spill_store spill_load < <(resource_triplet "$build" "$name")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$rep" "$residue" "$wall" "${active_max:-nan}" "${active_sum:-nan}" "$mc_avg" "$mc_max" "$mc_n" "$regs" "$spill_store" "$spill_load" >>"$SUMMARY"
}

for ((r=1; r<=REPEATS; ++r)); do run_one control "$CONTROL_BIN" "$LOGDIR/control.build.err" "$r"; done
for ((r=1; r<=REPEATS; ++r)); do run_one nextself "$PREFETCH_BIN" "$LOGDIR/nextself.build.err" "$r"; done

cat "$SUMMARY"
python3 - "$SUMMARY" "$WAIT_VARIANT" <<'PY'
import csv,math,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); variant=sys.argv[2]
if not rows: raise SystemExit('no next-self delta rows')
res={r['residue'] for r in rows}
if len(res)!=1: raise SystemExit('FATAL next-self residue mismatch '+repr({(r['profile'],r['repeat']):r['residue'] for r in rows}))
def group(name): return [r for r in rows if r['profile']==name]
def med(name,key):
    vals=[float(r[key]) for r in group(name) if r[key] not in ('','nan')]
    return statistics.median(vals) if vals else math.nan
def resource(name,key):
    vals=[]
    for r in group(name):
        try: vals.append(int(r[key]))
        except ValueError: pass
    return max(vals) if vals else -1
cw=med('control','wall_s'); pw=med('nextself','wall_s')
cm=med('control','mc_avg_pct'); pm=med('nextself','mc_avg_pct')
css=resource('control','spill_store_main_bytes'); csl=resource('control','spill_load_main_bytes')
pss=resource('nextself','spill_store_main_bytes'); psl=resource('nextself','spill_load_main_bytes')
print(f'nextself_wait_variant={variant}')
print(f'nextself_exact_residue={next(iter(res))}')
print('nextself_exact_residue_match=1')
print(f'nextself_control_wall_s={cw:.9f}')
print(f'nextself_prefetch_wall_s={pw:.9f}')
print(f'nextself_wall_speedup={cw/pw:.6f}x')
print(f'nextself_control_mc_avg_pct={cm:.3f}')
print(f'nextself_prefetch_mc_avg_pct={pm:.3f}')
print(f'nextself_mc_delta={pm-cm:.3f}pp')
print(f'nextself_control_spill_free={int(css==0 and csl==0)}')
print(f'nextself_prefetch_spill_free={int(pss==0 and psl==0)}')
print(f'nextself_promotable={int(pss==0 and psl==0 and pw<cw)}')
PY

echo "b300x8-ilp8-nextself-delta-ab OK wait_variant=$WAIT_VARIANT summary=$SUMMARY sample_interval=$SAMPLE_INTERVAL" >&2
