#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"; MOD="${MOD:-4294967291}"; ROWS="${ROWS:-1}"
THREADS="${GRIDFP_THREADS:-256}"; TARGET_MIB="${TARGET_MIB:-65536}"
PLAN_MIB="${GRIDFP_PLAN_TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; RANDOM_CG="${RANDOM_CG:-1}"; WARP_SCAN="${WARP_SCAN:-1}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300x8_ilp8_cpasync_staged_wait_ab_n${N}}"
BASE_ISO="$LOGDIR/base_gen"; BASE_BIN="$LOGDIR/base_cpasync"; STAGED_SRC="$LOGDIR/staged_wait.cu"; STAGED_BIN="$LOGDIR/staged_wait"
SUMMARY="$LOGDIR/summary.tsv"; PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$BASE_ISO" "$BASE_ISO/tmp"

[[ "$N" == 27 ]] || { echo 'staged ILP8 cp.async A/B currently targets n=27' >&2; exit 2; }
for x in RANDOM_CG WARP_SCAN; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1&&ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32&&THREADS<=768&&THREADS%32==0)) || { echo 'GRIDFP_THREADS must be warp multiple 32..768' >&2; exit 2; }
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] && ((MAXRREGCOUNT==0||(MAXRREGCOUNT>=32&&MAXRREGCOUNT<=255))) || { echo 'MAXRREGCOUNT must be 0 or 32..255' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

sample_mem(){
  local pid="$1" out="$2"; : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null |
      awk '{s+=$1;n++}END{if(n)printf "%.6f\n",s/n}' >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}
field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
mem_stats(){ awk '{s+=$1;n++;if($1>m)m=$1}END{if(n)printf "%.3f %.3f %d\n",s/n,m,n;else print "nan nan 0"}' "$1"; }

# Build and execute the ordinary cp.async candidate once.  The existing runner
# leaves its generated final CUDA source in BASE_ISO, which becomes the exact
# input to the staged-wait transform.
BASE_LOG="$LOGDIR/base.log"; BASE_MEM="$LOGDIR/base.mem"
set +e
env ROWS="$ROWS" GRIDFP_THREADS="$THREADS" TARGET_MIB="$TARGET_MIB" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" MAX_WINDOW="$MAX_WINDOW" \
    ARCH="$ARCH" RANDOM_CG="$RANDOM_CG" WARP_SCAN="$WARP_SCAN" PREFETCH_L2=0 CPASYNC=1 MAXRREGCOUNT="$MAXRREGCOUNT" \
    REBUILD=1 ISO="$BASE_ISO" BIN="$BASE_BIN" \
    "$ONEESAN_ROOT/scripts/run/b300x8-saturate-ilp8.sh" "$N" "$MOD" >"$BASE_LOG" 2>&1 &
bpid=$!; sample_mem "$bpid" "$BASE_MEM" & bmpid=$!; wait "$bpid"; brc=$?; set -e; wait "$bmpid" 2>/dev/null || true
cat "$BASE_LOG"
((brc==0)) || { echo "baseline cp.async failed rc=$brc" >&2; exit "$brc"; }
BASE_SRC="$BASE_ISO/final_main_ilp8_cpasync.cu"
[[ -f "$BASE_SRC" && -x "$BASE_BIN" ]] || { echo 'baseline generated source/binary missing' >&2; exit 3; }

python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-rankstate-ilp8-cpasync-staged-wait.py" "$BASE_SRC" "$STAGED_SRC" | tee "$LOGDIR/staged.transform.log"
grep -Fq 'b300_cpasync_wait_pair();' "$STAGED_SRC"
grep -Fq 'cp.async.wait_group 1' "$STAGED_SRC"
[[ "$(grep -Fc 'b300_cpasync_commit();' "$STAGED_SRC")" -eq 2 ]] || { echo 'staged source lost two-group contract' >&2; exit 3; }

REG_FLAGS=(); ((MAXRREGCOUNT>0)) && REG_FLAGS+=("-maxrregcount=$MAXRREGCOUNT")
STAGED_BUILD="$LOGDIR/staged.build.log"
TMPDIR="$BASE_ISO/tmp" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v "${REG_FLAGS[@]}" \
  -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=1 -DB300_BLOCK_CLOSURE_QUAD=0 \
  "$STAGED_SRC" -o "$STAGED_BIN" 2>"$STAGED_BUILD"
[[ -x "$STAGED_BIN" ]] || { echo 'staged binary missing' >&2; exit 3; }
cat "$STAGED_BUILD"

STAGED_LOG="$LOGDIR/staged.log"; STAGED_MEM="$LOGDIR/staged.mem"
set +e
B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$THREADS" GRIDFP_PLAN_TARGET_MIB="$PLAN_MIB" \
  "$STAGED_BIN" "$N" "$MOD" "$TARGET_MIB" "$MAX_WINDOW" 8 >"$STAGED_LOG" 2>&1 &
spid=$!; sample_mem "$spid" "$STAGED_MEM" & smpid=$!; wait "$spid"; src=$?; set -e; wait "$smpid" 2>/dev/null || true
cat "$STAGED_LOG"
((src==0)) || { echo "staged candidate failed rc=$src" >&2; exit "$src"; }

printf 'profile\tresidue\twall_s\tactive_max_s\tactive_sum_s\tmc_avg_pct\tmc_max_pct\tmc_samples\tregs_main\tspill_store_main_bytes\tspill_load_main_bytes\n' >"$SUMMARY"
parse_one(){
  local name="$1" log="$2" mem="$3" buildlog="$4" line residue wall amax asum mavg mmax mn regs spills spilll resfile
  line="$(grep '^backend=gridfp-b300-hbm32-fullmate-dropN ' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || line="$(grep '^backend=gridfp-b300-hbm32' "$log" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$name missing backend result" >&2; exit 4; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"; amax="$(field active_max_s "$line")"; asum="$(field active_sum_s "$line")"
  read -r mavg mmax mn < <(mem_stats "$mem")
  resfile="$LOGDIR/$name.ptxas.tsv"; : >"$resfile"
  # Baseline ptxas text is printed into BASE_LOG by its runner; staged uses its
  # raw nvcc stderr.  Fail closed on unknown resource data later rather than
  # accidentally promoting a spilling candidate.
  if python3 "$PARSER" "$buildlog" --label "$name" --contains b300_main_pull_rankstate_ilp8_kernel >"$resfile" 2>"$resfile.err"; then
    read -r regs spills spilll < <(python3 - "$resfile" <<'PY'
import csv,sys
v=[]
for r in csv.reader(open(sys.argv[1]),delimiter='\t'):
    if len(r)<8: continue
    try:v.append((int(r[2]),int(r[4]),int(r[5])))
    except ValueError:pass
print('nan nan nan' if not v else f'{max(x[0] for x in v)} {max(x[1] for x in v)} {max(x[2] for x in v)}')
PY
)
  else regs=nan;spills=nan;spilll=nan;fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$residue" "$wall" "${amax:-nan}" "${asum:-nan}" "$mavg" "$mmax" "$mn" "$regs" "$spills" "$spilll" >>"$SUMMARY"
}
# BASE_LOG contains the ptxas summary emitted by the stock runner. The staged
# raw build log is authoritative for staged resource usage.
parse_one base "$BASE_LOG" "$BASE_MEM" "$BASE_LOG"
parse_one staged "$STAGED_LOG" "$STAGED_MEM" "$STAGED_BUILD"
cat "$SUMMARY"
python3 - "$SUMMARY" <<'PY'
import csv,sys,math
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
q={x['profile']:x for x in r}
if q['base']['residue']!=q['staged']['residue']: raise SystemExit('FATAL staged wait residue mismatch')
def f(p,k): return float(q[p][k])
print(f"staged_wait_wall_speedup={f('base','wall_s')/f('staged','wall_s'):.6f}x")
print(f"staged_wait_active_speedup={f('base','active_max_s')/f('staged','active_max_s'):.6f}x")
print(f"staged_wait_mc_delta={f('staged','mc_avg_pct')-f('base','mc_avg_pct'):.3f}pp")
for p in ('base','staged'):
    ss=q[p]['spill_store_main_bytes'];sl=q[p]['spill_load_main_bytes']
    print(f"{p}_spill_known_free={int(ss=='0' and sl=='0')}")
print('wait_pair_group=1 wait_block_group=0 exact_residue_match=1')
PY

echo "b300x8-ilp8-cpasync-staged-wait-ab OK summary=$SUMMARY logs=$LOGDIR" >&2
