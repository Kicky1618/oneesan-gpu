#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
MOD="${MOD:-4294967291}"
ROWS="${ROWS:-1}"
ARCH="${ARCH:-native}"
TARGET_MIB="${TARGET_MIB:-65536}"
MAX_WINDOW="${MAX_WINDOW:-14}"
THRESHOLDS="${ILP8_THRESHOLDS:-0 262144 524288 1048576 2097152 4194304 8388608 16777216}"
THREADS_LIST="${THREADS_LIST:-128 256 512}"
REPEATS="${REPEATS:-1}"
HIGH_DROP_CHUNK="${HIGH_DROP_CHUNK:-0}"
RANDOM_CG="${RANDOM_CG:-0}"
RANDOM_CG_L2_FETCH_BYTES="${RANDOM_CG_L2_FETCH_BYTES:-0}"
DUALMASK="${DUALMASK:-0}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_mainrec_hybrid8_row${ROWS}_hd${HIGH_DROP_CHUNK}_cg${RANDOM_CG}_l2${RANDOM_CG_L2_FETCH_BYTES}_dual${DUALMASK}_r${MAXRREGCOUNT}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
mkdir -p "$LOGDIR" "$ONEESAN_BUILD_DIR" "$(dirname "$RESULT")" "$(dirname "$WINNER_ENV")"

[[ "$N" == 27 ]] || { echo 'mainrec hybrid threshold A/B targets n=27' >&2; exit 2; }
[[ "$ROWS" =~ ^[0-9]+$ ]] && ((ROWS>=1 && ROWS<=28)) || { echo 'ROWS must be 1..28' >&2; exit 2; }
[[ "$REPEATS" =~ ^[0-9]+$ ]] && ((REPEATS>=1)) || { echo 'REPEATS must be >=1' >&2; exit 2; }
[[ "$HIGH_DROP_CHUNK" == 0 || "$HIGH_DROP_CHUNK" == 1 ]] || exit 2
[[ "$RANDOM_CG" == 0 || "$RANDOM_CG" == 1 ]] || exit 2
case "$RANDOM_CG_L2_FETCH_BYTES" in 0|64|128|256) ;; *) exit 2;; esac
[[ "$DUALMASK" == 0 || "$DUALMASK" == 1 ]] || exit 2
[[ "$MAXRREGCOUNT" =~ ^[0-9]+$ ]] || exit 2
if [[ "$RANDOM_CG" == 0 && "$RANDOM_CG_L2_FETCH_BYTES" != 0 ]]; then echo 'L2 fetch size requires RANDOM_CG=1' >&2; exit 2; fi
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

read -r -a threshold_list <<<"$THRESHOLDS"
read -r -a threads_list <<<"$THREADS_LIST"
(( ${#threshold_list[@]} > 0 && ${#threads_list[@]} > 0 )) || exit 2
declare -A seen_threshold=()
for t in "${threshold_list[@]}"; do
  [[ "$t" =~ ^[0-9]+$ ]] || { echo "invalid threshold=$t" >&2; exit 2; }
  [[ -z "${seen_threshold[$t]+x}" ]] || { echo "duplicate threshold=$t" >&2; exit 2; }
  seen_threshold[$t]=1
done
[[ -n "${seen_threshold[0]+x}" ]] || { echo 'ILP8_THRESHOLDS must include 0 always-ILP8 baseline' >&2; exit 2; }
for t in "${threads_list[@]}"; do
  [[ "$t" =~ ^[0-9]+$ ]] && ((t>=32 && t<=1024 && t%32==0)) || { echo "invalid threads=$t" >&2; exit 2; }
done

bash "$ONEESAN_ROOT/scripts/bench/b300-mainrec-hybrid-ilp8-transform-preflight.sh" >"$LOGDIR/hybrid.preflight.out"
cat "$LOGDIR/hybrid.preflight.out" >&2

# Build the production recurrence source once. Every candidate below is a
# post-transform of these exact bytes, eliminating build-chain drift between
# thresholds.
BASE_GEN="$LOGDIR/base_gen"
BASE_BIN_RAW="$LOGDIR/base_production.bin"
BASE_OUT="$LOGDIR/base_production.build.out"
BASE_ERR_RAW="$LOGDIR/base_production.build.err"
ONEESAN_BUILD_DIR="$BASE_GEN" ONEESAN_TMP_DIR="$BASE_GEN/tmp" \
N=27 ARCH="$ARCH" OUT="$BASE_BIN_RAW" HIGH_DROP_CHUNK="$HIGH_DROP_CHUNK" DUALMASK=0 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-calibrated.sh" >"$BASE_OUT" 2>"$BASE_ERR_RAW"
BASE_SRC="$(sed -nE 's/^  source_after_proof_gates=(.*)$/\1/p' "$BASE_OUT" | tail -n1)"
[[ -n "$BASE_SRC" && -f "$BASE_SRC" ]] || { echo 'could not resolve production mainrec source' >&2; exit 3; }
grep -Fq 'main_recurrence=1' "$BASE_OUT"
grep -Fq 'main_pull_ilp=2' "$BASE_OUT"
grep -Fq 'b300_main_pull_ilp2_blocks(ms.size,threads)' "$BASE_SRC"

PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
printf 'mode\tthreshold\tbinary\tbuild_err\tbuild_source\n' >"$LOGDIR/binaries.tsv"
printf 'mode\tthreshold\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

compile_candidate(){
  local mode="$1" threshold="$2"
  local tag="${mode}_t${threshold}"
  local src="$BASE_SRC"
  local hybrid_src="$LOGDIR/${tag}.hybrid.cu"
  local cg_src="$LOGDIR/${tag}.cg.cu"
  local dual_src="$LOGDIR/${tag}.dual.cu"
  local bin="$LOGDIR/${tag}.bin"
  local berr="$LOGDIR/${tag}.build.err"

  if [[ "$mode" == hybrid ]]; then
    python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence-hybrid-ilp8.py" "$src" "$hybrid_src" "$threshold" >"$LOGDIR/${tag}.hybrid.out"
    grep -Fq "ilp8_min_states=$threshold" "$LOGDIR/${tag}.hybrid.out"
    grep -Fq 'batch_abi_preserved=1' "$LOGDIR/${tag}.hybrid.out"
    src="$hybrid_src"
  fi

  if [[ "$RANDOM_CG" == 1 ]]; then
    python3 "$ONEESAN_ROOT/scripts/build/gen-b300-mainrec-random-cg.py" "$src" "$cg_src" "$RANDOM_CG_L2_FETCH_BYTES" >"$LOGDIR/${tag}.cg.out"
    grep -Fq 'hybrid_policy_consistent=1' "$LOGDIR/${tag}.cg.out"
    src="$cg_src"
  fi

  if [[ "$DUALMASK" == 1 ]]; then
    python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-pull-dualmask.py" "$src" "$dual_src" >"$LOGDIR/${tag}.dual.out"
    grep -Fq 'b300_block_endpoint_masks(d)' "$dual_src"
    src="$dual_src"
  fi

  if [[ "$mode" == hybrid ]]; then
    grep -Fq 'main_pull_kernel_ilp8_hybrid' "$src"
    grep -Fq "if(ms.size>=Code($threshold))" "$src"
    grep -Fq 'base+=Code(8)*grid' "$src"
  fi
  grep -Fq 'main_pull_kernel_ilp2' "$src"

  local -a reg_flags=()
  (( MAXRREGCOUNT > 0 )) && reg_flags+=("-maxrregcount=$MAXRREGCOUNT")
  TMPDIR="$LOGDIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" -Xptxas=-v "${reg_flags[@]}" \
    -DTARGET_W=28 -DLOW_LUT_K=14 -DHIGH_LUT_K=13 "$src" -o "$bin" \
    >"$LOGDIR/${tag}.compile.out" 2>"$berr"
  [[ -x "$bin" ]] || { echo "candidate compile failed mode=$mode threshold=$threshold" >&2; exit 3; }
  printf '%s\t%s\t%s\t%s\t%s\n' "$mode" "$threshold" "$bin" "$berr" "$src" >>"$LOGDIR/binaries.tsv"

  local parsed="$LOGDIR/${tag}.ptxas.tsv"
  : >"$parsed"
  if [[ "$mode" == hybrid ]]; then
    python3 "$PARSER" "$berr" --label "$tag" --contains main_pull_kernel_ilp2 --contains main_pull_kernel_ilp8_hybrid >"$parsed"
  else
    python3 "$PARSER" "$berr" --label "$tag" --contains main_pull_kernel_ilp2 >"$parsed"
  fi
  awk -F '\t' -v OFS='\t' -v mode="$mode" -v threshold="$threshold" '{print mode,threshold,$2,$3,$4,$5,$6,$7,$8}' "$parsed" >>"$RESOURCE"
}

# Recompile the pure ILP2 baseline from the same source and flags instead of
# reusing BASE_BIN_RAW, so optional CG/dualmask/reg-cap settings are identical.
compile_candidate ilp2 NA
for threshold in "${threshold_list[@]}"; do
  compile_candidate hybrid "$threshold"
done

field(){
  local key="$1" line="$2"
  sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1
}

printf 'mode\tthreshold\tthreads\trepeat\tresidue\twall_s\tactive_max_s\tactive_sum_s\tmc_avg_pct\tmc_max_pct\n' >"$RESULT"
run_one(){
  local mode="$1" threshold="$2" bin="$3" threads="$4" rep="$5"
  local tag="${mode}_t${threshold}_th${threads}_r${rep}"
  local so="$LOGDIR/${tag}.out" se="$LOGDIR/${tag}.err" dm="$LOGDIR/${tag}.dmon"
  echo "=== mainrec hybrid A/B mode=$mode threshold=$threshold threads=$threads repeat=$rep rows=$ROWS ===" >&2
  : >"$dm"
  nvidia-smi dmon -s u -d 1 >"$dm" 2>&1 & local mpid=$!
  set +e
  B300_ROW_LIMIT="$ROWS" GRIDFP_THREADS="$threads" "$bin" 27 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se"
  local rc=$?
  set -e
  kill "$mpid" 2>/dev/null || true
  wait "$mpid" 2>/dev/null || true
  if (( rc != 0 )); then
    echo "$tag failed rc=$rc" >&2
    tail -n 120 "$se" >&2 || true
    return "$rc"
  fi
  local line residue wall active_max active_sum mc_avg mc_max
  line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"
  [[ -n "$line" ]] || { echo "$tag missing backend result line" >&2; return 4; }
  residue="$(field residue "$line")"; wall="$(field wall_s "$line")"
  active_max="$(field active_max_s "$line")"; active_sum="$(field active_sum_s "$line")"
  [[ -n "$residue" && -n "$wall" ]] || { echo "$tag missing residue/wall_s" >&2; return 4; }
  read -r mc_avg mc_max < <(awk '$1~/^[0-9]+$/&&$3~/^[0-9]+$/{s+=$3;n++;if($3>m)m=$3}END{if(n)printf "%.3f %d\n",s/n,m;else print "nan nan"}' "$dm")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$mode" "$threshold" "$threads" "$rep" "$residue" "$wall" "${active_max:-nan}" "${active_sum:-nan}" "$mc_avg" "$mc_max" >>"$RESULT"
}

mapfile -t candidate_rows < <(tail -n +2 "$LOGDIR/binaries.tsv")
for ((rep=1; rep<=REPEATS; ++rep)); do
  if (( rep % 2 == 1 )); then
    thread_order=("${threads_list[@]}")
    candidate_order=("${candidate_rows[@]}")
  else
    thread_order=(); for ((i=${#threads_list[@]}-1;i>=0;--i)); do thread_order+=("${threads_list[$i]}"); done
    candidate_order=(); for ((i=${#candidate_rows[@]}-1;i>=0;--i)); do candidate_order+=("${candidate_rows[$i]}"); done
  fi
  for threads in "${thread_order[@]}"; do
    for row in "${candidate_order[@]}"; do
      IFS=$'\t' read -r mode threshold bin berr src <<<"$row"
      run_one "$mode" "$threshold" "$bin" "$threads" "$rep"
    done
  done
done

python3 - "$RESULT" "$RESOURCE" "$LOGDIR/binaries.tsv" "$WINNER_ENV" <<'PY'
import csv,math,statistics,sys,shlex
result,resource,binaries,winner=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t'))
rr=list(csv.DictReader(open(resource),delimiter='\t'))
bins=list(csv.DictReader(open(binaries),delimiter='\t'))
if not rows: raise SystemExit('no mainrec hybrid results')
res={r['residue'] for r in rows}
if len(res)!=1:
    raise SystemExit('FATAL mainrec hybrid residue mismatch '+repr({(r['mode'],r['threshold'],r['threads'],r['repeat']):r['residue'] for r in rows}))

resource_by={}
for r in rr:
    key=(r['mode'],r['threshold'])
    try:
        vals=(int(r['registers']),int(r['spill_store_bytes']),int(r['spill_load_bytes']))
    except (ValueError,TypeError,KeyError):
        continue
    old=resource_by.get(key,[]); old.append(vals); resource_by[key]=old

def resource_summary(key):
    xs=resource_by.get(key,[])
    if not xs: return (-1,-1,-1,0)
    return (max(x[0] for x in xs),max(x[1] for x in xs),max(x[2] for x in xs),len(xs))

by={}
for r in rows:
    key=(r['mode'],r['threshold'],int(r['threads']))
    by.setdefault(key,[]).append(r)
med=[]
for (mode,t,threads),rs in by.items():
    wall=statistics.median(float(r['wall_s']) for r in rs)
    mc=[float(r['mc_avg_pct']) for r in rs if r['mc_avg_pct']!='nan']
    mc_med=statistics.median(mc) if mc else float('nan')
    regs,ss,sl,nkern=resource_summary((mode,t))
    expected=2 if mode=='hybrid' else 1
    resources_known=(nkern>=expected and regs>=0 and ss>=0 and sl>=0)
    spill_free=resources_known and ss==0 and sl==0
    med.append((wall,mode,t,threads,mc_med,regs,ss,sl,nkern,spill_free))

def order(x):
    mc=x[4]
    return (x[0], -mc if not math.isnan(mc) else math.inf)
base=min((x for x in med if x[1]=='ilp2'),key=order)
clean=[x for x in med if x[9]]
if not clean:
    raise SystemExit('no candidate has known spill-free main recurrence ptxas')
best=min(clean,key=order)
for x in sorted(med,key=order):
    print('MAINREC_HYBRID_CANDIDATE',
          f'mode={x[1]}',f'threshold={x[2]}',f'threads={x[3]}',
          f'median_wall_s={x[0]:.9f}',f'speedup_vs_ilp2={base[0]/x[0]:.6f}x',
          f'mc_avg_pct={x[4]:.3f}',f'regs_max={x[5]}',f'spill_store_max={x[6]}',
          f'spill_load_max={x[7]}',f'kernel_records={x[8]}',f'spill_free={int(x[9])}',file=sys.stderr)
print('MAINREC_HYBRID_SELECTED',
      f'mode={best[1]}',f'threshold={best[2]}',f'threads={best[3]}',
      f'median_wall_s={best[0]:.9f}',f'speedup_vs_ilp2={base[0]/best[0]:.6f}x',
      f'residue={next(iter(res))}',f'exact_gate=1',f'spill_free=1',file=sys.stderr)

bmap={(r['mode'],r['threshold']):r for r in bins}
bkey=(best[1],best[2]); basekey=('ilp2','NA')
if bkey not in bmap or basekey not in bmap: raise SystemExit('binary lookup failed')
def q(v): return shlex.quote(str(v))
with open(winner,'w') as f:
    f.write('B300_MAINREC_HYBRID_WINNER_MODE='+q(best[1])+'\n')
    f.write('B300_MAINREC_HYBRID_WINNER_THRESHOLD='+q(best[2])+'\n')
    f.write('B300_MAINREC_HYBRID_WINNER_THREADS='+q(best[3])+'\n')
    f.write('B300_MAINREC_HYBRID_WINNER_BIN='+q(bmap[bkey]['binary'])+'\n')
    f.write('B300_MAINREC_HYBRID_BASE_THREADS='+q(base[3])+'\n')
    f.write('B300_MAINREC_HYBRID_BASE_BIN='+q(bmap[basekey]['binary'])+'\n')
    f.write('B300_MAINREC_HYBRID_RESIDUE='+q(next(iter(res)))+'\n')
    f.write('B300_MAINREC_HYBRID_WINNER_WALL_S='+q(f'{best[0]:.9f}')+'\n')
    f.write('B300_MAINREC_HYBRID_BASE_WALL_S='+q(f'{base[0]:.9f}')+'\n')
    f.write('B300_MAINREC_HYBRID_SPEEDUP_VS_ILP2='+q(f'{base[0]/best[0]:.9f}')+'\n')
    f.write('B300_MAINREC_HYBRID_WINNER_REGISTERS='+q(best[5])+'\n')
    f.write('B300_MAINREC_HYBRID_WINNER_SPILL_STORE_BYTES='+q(best[6])+'\n')
    f.write('B300_MAINREC_HYBRID_WINNER_SPILL_LOAD_BYTES='+q(best[7])+'\n')
    f.write('B300_MAINREC_HYBRID_WINNER_SPILL_FREE=1\n')
    f.write('B300_MAINREC_HYBRID_EXACT_INTERMEDIATE_MATCH=1\n')
print('b300_mainrec_hybrid_winner_env='+winner)
PY

cat "$RESOURCE"
cat "$RESULT"
cat "$WINNER_ENV"
echo "b300-mainrec-hybrid-ilp8-threshold-ab OK rows=$ROWS repeats=$REPEATS result=$RESULT resource=$RESOURCE winner_env=$WINNER_ENV" >&2
