#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"; REPEATS="${REPEATS:-2}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-native}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-32}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
COL_ILP="${COL_ILP:-2}"; MLP_WINDOW4="${MLP_WINDOW4:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.25}"; RUN_PTXAS="${RUN_PTXAS:-1}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required" >&2; exit 2; }
fi
case "$COL_ILP" in 1|2|4) ;; *) echo "COL_ILP must be 1, 2, or 4" >&2; exit 2;; esac
[[ "$MLP_WINDOW4" == 0 || "$MLP_WINDOW4" == 1 ]] || exit 2
[[ "$RUN_PTXAS" == 0 || "$RUN_PTXAS" == 1 ]] || exit 2
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi required" >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU GPUs" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather64_ab_n${N}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

bash "$ONEESAN_ROOT/scripts/bench/rankformula-directgather64-proof.sh"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

sample_process(){
  local pid="$1" out="$2"; : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' \
      >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}

build_one(){
  local label="$1" dg64="$2" bin="$3"
  N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP="$COL_ILP" DEPTHMAJOR=1 \
    FORCE7=0 MLP_WINDOW4="$MLP_WINDOW4" PAIR_MLP=0 PREFETCH_NEXT=0 \
    DIRECTGATHER64="$dg64" PM_ACCUM=1 MAXRREGCOUNT=0 PTXAS_VERBOSE="$RUN_PTXAS" \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" \
    >"$LOGDIR/$label.build.out" 2>"$LOGDIR/$label.build.err"
}

sum_runtime_bytes(){
  local label="$1" se="$2"
  if [[ "$label" == dg64 ]]; then
    awk '/p10dc_low_rankformula_directgather64 /{for(i=1;i<=NF;i++)if($i~/^resident_bytes=/){split($i,a,"=");s+=a[2]}} END{print s+0}' "$se"
  else
    awk '/p10dc_low_rankformula_directgather / && $0 !~ /directgather64/{for(i=1;i<=NF;i++)if($i~/^bytes=/){split($i,a,"=");s+=a[2]}} END{print s+0}' "$se"
  fi
}

run_one(){
  local label="$1" bin="$2" rep="$3"
  local so="$LOGDIR/${label}_r${rep}.out" se="$LOGDIR/${label}_r${rep}.err" util="$LOGDIR/${label}_r${rep}.util"
  BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  local pid=$!; sample_process "$pid" "$util" & local sampler=$!
  set +e; wait "$pid"; local rc=$?; set -e
  wait "$sampler" || true
  (( rc == 0 )) || return "$rc"
  local line detail residue wall fh rh avg_gpu avg_mem max_gpu max_mem desc_bytes
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || return 3
  residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$label residue mismatch got=$residue expected=$EXPECT" >&2; return 4; }
  wall="$(field wall_s "$line")"
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
  read -r avg_gpu avg_mem max_gpu max_mem < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
  desc_bytes="$(sum_runtime_bytes "$label" "$se")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" \
    "$avg_gpu" "$avg_mem" "$max_mem" "$desc_bytes" "$COL_ILP" >>"$RESULT"
}

printf 'mode\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\tdescriptor_bytes_all_gpus\tcol_ilp\n' >"$RESULT"
[[ "$RUN_PTXAS" == 1 ]] && printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

for spec in 'uint4 0' 'dg64 1'; do
  read -r label dg64 <<<"$spec"
  bin="$ONEESAN_BUILD_DIR/ab_directgather64_${label}_n${N}"
  build_one "$label" "$dg64" "$bin"
  [[ "$RUN_PTXAS" == 1 ]] && python3 "$PARSER" "$LOGDIR/$label.build.err" --label "$label" >>"$RESOURCE"
  for ((r=1;r<=REPEATS;++r)); do run_one "$label" "$bin" "$r"; done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$RESOURCE" "$RUN_PTXAS" <<'PY'
import csv,statistics,sys
src,dst,resource,run_ptxas=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t')); out=[]
for mode in ('uint4','dg64'):
    g=[r for r in rows if r['mode']==mode]; z={'mode':mode,'repeats':len(g)}
    for k in ('wall_s','forward_high_s','reverse_high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct','descriptor_bytes_all_gpus'):
        xs=[float(r[k]) for r in g if r[k] != 'NA']; z[k]=statistics.median(xs) if xs else None
    z['total_high_s']=z['forward_high_s']+z['reverse_high_s'] if z['forward_high_s'] is not None and z['reverse_high_s'] is not None else None
    out.append(z)
keys=('mode','repeats','wall_s','forward_high_s','reverse_high_s','total_high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct','descriptor_bytes_all_gpus')
with open(dst,'w') as f:
    f.write('\t'.join(keys)+'\n')
    for z in out:f.write('\t'.join(str(z.get(k)) for k in keys)+'\n')
q={z['mode']:z for z in out}
for k in ('wall_s','forward_high_s','reverse_high_s','total_high_s'):
    if q['uint4'][k] and q['dg64'][k]: print(f'dg64_{k}_speedup={q["uint4"][k]/q["dg64"][k]:.6f}x')
if q['uint4']['descriptor_bytes_all_gpus'] and q['dg64']['descriptor_bytes_all_gpus']:
    print(f'descriptor_runtime_byte_reduction_pct={100*(1-q["dg64"]["descriptor_bytes_all_gpus"]/q["uint4"]["descriptor_bytes_all_gpus"]):.6f}')
for mode in ('uint4','dg64'):
    print(f'{mode}_avg_memctrl_pct={q[mode]["avg_memctrl_util_pct"]}')
    print(f'{mode}_max_memctrl_pct={q[mode]["max_memctrl_util_pct"]}')
if run_ptxas=='1':
    rr=list(csv.DictReader(open(resource),delimiter='\t'))
    for mode in ('uint4','dg64'):
        g=[r for r in rr if r['backend']==mode and 'high' in r['kernel'].lower()]
        regs=[int(r['registers']) for r in g if r['registers']!='NA']
        ss=[int(r['spill_store_bytes']) for r in g if r['spill_store_bytes']!='NA']
        sl=[int(r['spill_load_bytes']) for r in g if r['spill_load_bytes']!='NA']
        print(f'{mode}_high_max_registers={max(regs) if regs else "NA"}')
        print(f'{mode}_high_spill_store_bytes={sum(ss) if ss else "NA"}')
        print(f'{mode}_high_spill_load_bytes={sum(sl) if sl else "NA"}')
print('semantic_delta=descriptor_encoding_only')
print('fixed=depthmajor1_pair0_prefetch0_force70_pm1_same_colilp_geometry')
PY

echo "b300-directgather64-ab OK n=$N repeats=$REPEATS result=$RESULT" >&2
