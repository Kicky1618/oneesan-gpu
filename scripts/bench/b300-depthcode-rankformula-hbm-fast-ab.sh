#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"; REPEATS="${REPEATS:-2}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
BASE_GRID_X="${BASE_GRID_X:-16}"; FAST_GRID_X="${FAST_GRID_X:-32}"
BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"; BUCKET_THREADS="${BUCKET_THREADS:-256}"
RUN_SELFTEST="${RUN_SELFTEST:-1}"; RUN_PTXAS="${RUN_PTXAS:-1}"; SAMPLE_UTIL="${SAMPLE_UTIL:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.5}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required for N=$N MOD=$MOD" >&2; exit 2; }
fi
for x in RUN_SELFTEST RUN_PTXAS SAMPLE_UTIL; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
for x in "$BASE_GRID_X" "$FAST_GRID_X" "$BUCKET_GRID_Y" "$BUCKET_THREADS" "$REPEATS"; do (( x > 0 )) || exit 2; done
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU GPUs" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankformula_hbm_fast_ab_n${N}_${TRANSPOSE_MODE}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

build_one(){
  local label="$1" mlp="$2" pm="$3" bin="$4"
  N="$N" ARCH="$ARCH" OUT="$bin" \
    RANKFORMULA_NOMETA_BLOCK=16 RANKFORMULA_NOMETA_WARPSHARE=1 \
    RANKFORMULA_NOMETA_COOPGROUP=1 RANKFORMULA_NOMETA_COOP_UNROLL=0 \
    RANKFORMULA_NOMETA_GROUP56=0 RANKFORMULA_NOMETA_GROUP61=1 \
    RANKFORMULA_ABSTRACT_SELECT8=1 RANKFORMULA_ABSTRACT_DEPTH4=1 \
    RANKFORMULA_ABSTRACT_SRCPACK10=1 RANKFORMULA_GATHER_MLP="$mlp" \
    PM_ACCUM="$pm" TERNARY_KEY4=1 DEPTHCODE_DECODE_LOAD=ldg RANKSTREAM_LUT_LOAD=ldg \
    TRANSPOSE_MODE="$TRANSPOSE_MODE" PTXAS_VERBOSE="$RUN_PTXAS" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-rankformula-nometa4-abstract.sh" \
    >"$LOGDIR/$label.build.out" 2>"$LOGDIR/$label.build.err"
}

sample_process(){
  local pid="$1" out="$2"
  : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' '{g=$1+0;m=$2+0; sg+=g;sm+=m; if(g>mg)mg=g;if(m>mm)mm=m;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' \
      >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}

run_one(){
  local label="$1" bin="$2" rep="$3" gx="$4"
  local so="$LOGDIR/${label}_r${rep}.out" se="$LOGDIR/${label}_r${rep}.err" util="$LOGDIR/${label}_r${rep}.util"
  BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$gx" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  local pid=$! sampler='' rc=0
  if [[ "$SAMPLE_UTIL" == 1 ]]; then sample_process "$pid" "$util" & sampler=$!; else : >"$util"; fi
  if wait "$pid"; then rc=0; else rc=$?; fi
  [[ -z "$sampler" ]] || wait "$sampler" || true
  (( rc == 0 )) || return "$rc"
  local line detail residue wall fh rh avg_gpu avg_mem max_gpu max_mem
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || return 3
  residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$label residue mismatch got=$residue expected=$EXPECT" >&2; return 4; }
  wall="$(field wall_s "$line")"
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
  read -r avg_gpu avg_mem max_gpu max_mem < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" "$gx" "$avg_gpu" "$avg_mem" "$max_gpu" "$max_mem" >>"$RESULT"
}

printf 'mode\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tgrid_x\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_gpu_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
[[ "$RUN_PTXAS" == 1 ]] && printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

for spec in 'baseline 0 0' 'fast 1 1'; do
  read -r label mlp pm <<<"$spec"
  if [[ "$RUN_SELFTEST" == 1 ]]; then
    RANKFORMULA_NOMETA_BLOCK=16 RANKFORMULA_NOMETA_WARPSHARE=1 RANKFORMULA_NOMETA_COOPGROUP=1 \
      RANKFORMULA_NOMETA_COOP_UNROLL=0 RANKFORMULA_NOMETA_GROUP56=0 RANKFORMULA_NOMETA_GROUP61=1 \
      RANKFORMULA_ABSTRACT_SELECT8=1 RANKFORMULA_ABSTRACT_DEPTH4=1 RANKFORMULA_ABSTRACT_SRCPACK10=1 \
      RANKFORMULA_GATHER_MLP="$mlp" PM_ACCUM="$pm" ARCH="$ARCH" \
      bash "$ONEESAN_ROOT/scripts/bench/pattern10-depthcode-rankformula-nometa4-abstract-block-selftest.sh" \
      >"$LOGDIR/$label.selftest.out" 2>"$LOGDIR/$label.selftest.err"
  fi
  bin="$ONEESAN_BUILD_DIR/ab_rankformula_hbm_${label}_n${N}"
  build_one "$label" "$mlp" "$pm" "$bin"
  [[ "$RUN_PTXAS" == 1 ]] && python3 "$PARSER" "$LOGDIR/$label.build.err" --label "$label" >>"$RESOURCE"
  gx="$BASE_GRID_X"; [[ "$label" == fast ]] && gx="$FAST_GRID_X"
  for ((r=1;r<=REPEATS;++r)); do run_one "$label" "$bin" "$r" "$gx"; done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$RESOURCE" "$RUN_PTXAS" <<'PY'
import csv, statistics, sys
src,dst,resource,run_ptxas=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]
for mode in ('baseline','fast'):
    g=[r for r in rows if r['mode']==mode]; z={'mode':mode,'repeats':len(g)}
    for k in ('wall_s','forward_high_s','reverse_high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct'):
        xs=[float(r[k]) for r in g if r[k] != 'NA']; z[k]=statistics.median(xs) if xs else None
    z['total_high_s']=(z['forward_high_s']+z['reverse_high_s']) if z['forward_high_s'] is not None and z['reverse_high_s'] is not None else None
    out.append(z)
with open(dst,'w') as f:
    keys=('mode','repeats','wall_s','forward_high_s','reverse_high_s','total_high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct')
    f.write('\t'.join(keys)+'\n')
    for z in out:f.write('\t'.join(str(z.get(k)) for k in keys)+'\n')
q={z['mode']:z for z in out}
for k in ('wall_s','forward_high_s','reverse_high_s','total_high_s'):
    if q['baseline'][k] and q['fast'][k]: print(f'fast_{k}_speedup={q["baseline"][k]/q["fast"][k]:.6f}x')
if q['baseline']['avg_memctrl_util_pct'] is not None and q['fast']['avg_memctrl_util_pct'] is not None:
    print(f'memctrl_avg_baseline_pct={q["baseline"]["avg_memctrl_util_pct"]:.3f}')
    print(f'memctrl_avg_fast_pct={q["fast"]["avg_memctrl_util_pct"]:.3f}')
if run_ptxas=='1':
    rr=list(csv.DictReader(open(resource),delimiter='\t'))
    for mode in ('baseline','fast'):
        g=[r for r in rr if r['backend']==mode and 'high' in r['kernel'].lower()]
        regs=[int(r['registers']) for r in g if r['registers']!='NA']
        ss=[int(r['spill_store_bytes']) for r in g if r['spill_store_bytes']!='NA']
        sl=[int(r['spill_load_bytes']) for r in g if r['spill_load_bytes']!='NA']
        print(f'{mode}_high_max_registers={max(regs) if regs else "NA"}')
        print(f'{mode}_high_spill_store_bytes={sum(ss) if ss else "NA"}')
        print(f'{mode}_high_spill_load_bytes={sum(sl) if sl else "NA"}')
print('baseline=gather_mlp0_pm0_gridx16')
print('fast=gather_mlp1_pm1_gridx32')
print('fixed=block16_coop_group61_depth4_srcpack10_gridy8')
PY

echo "b300-depthcode-rankformula-hbm-fast-ab OK n=$N repeats=$REPEATS result=$RESULT" >&2
