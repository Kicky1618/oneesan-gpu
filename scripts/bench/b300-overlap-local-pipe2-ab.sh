#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"; REPEATS="${REPEATS:-1}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-native}"
THREADS="${BUCKET_THREADS:-256}"; GX="${BUCKET_GRID_X:-32}"; GY="${BUCKET_GRID_Y:-8}"
SORTED="${SORTED:-0}"; SPARSE64="${DIRECTGATHER_SPARSE64:-0}"
PRECTX_FORWARD="${PRECTX_FORWARD:-0}"; PRECTX_REVERSE="${PRECTX_REVERSE:-0}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.20}"; RUN_PTXAS="${RUN_PTXAS:-1}"
EXPECT="${EXPECT:-998035516}"
[[ "$N" == 21 && "$MOD" == 4294967291 ]] || [[ -n "${EXPECT:-}" ]] || { echo 'EXPECT required outside N=21 default modulus' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }

# Four of the five candidates use peer-global -> shared cp.async. Prove that the
# exact 8-GPU host/driver topology supports it before spending solver time.
GPUS="$NGPU" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh"

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_overlap_local_pipe2_n${N}}"
RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

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
  local label="$1" cp="$2" localcp="$3" overlap="$4" pipe2="$5" bin="$6"
  N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP=2 DEPTHMAJOR=1 PAIR_MLP=1 MLP_WINDOW4=1 \
    CPASYNC_PAIR="$cp" CPASYNC_LOCAL_PAIR="$localcp" CPASYNC_OVERLAP_LOCAL_PAIR="$overlap" \
    CPASYNC_OVERLAP_LOCAL_PIPE2="$pipe2" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" \
    SORTED="$SORTED" PRECTX_FORWARD="$PRECTX_FORWARD" PRECTX_REVERSE="$PRECTX_REVERSE" \
    FORCE7=0 PREFETCH_NEXT=0 PM_ACCUM=1 PTXAS_VERBOSE="$RUN_PTXAS" \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" \
    >"$LOGDIR/$label.build.out" 2>"$LOGDIR/$label.build.err"
}

run_one(){
  local label="$1" bin="$2" rep="$3"
  local so="$LOGDIR/${label}_r${rep}.out" se="$LOGDIR/${label}_r${rep}.err" util="$LOGDIR/${label}_r${rep}.util"
  BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$GX" BUCKET_GRID_Y="$GY" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  local pid=$!; sample_process "$pid" "$util" & local sampler=$!
  set +e; wait "$pid"; local rc=$?; set -e
  wait "$sampler" || true
  (( rc == 0 )) || { echo "$label run failed rc=$rc" >&2; return "$rc"; }

  local line detail residue wall fh rh avg_gpu avg_mem max_gpu max_mem
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$label missing residue" >&2; return 3; }
  residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$label residue mismatch got=$residue expected=$EXPECT" >&2; return 4; }
  wall="$(field wall_s "$line")"
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
  read -r avg_gpu avg_mem max_gpu max_mem < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" "$avg_gpu" "$avg_mem" "$max_gpu" "$max_mem" >>"$RESULT"
}

printf 'mode\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_gpu_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
[[ "$RUN_PTXAS" == 1 ]] && printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

# label cpasync cross->local staging overlap30 pipe2
for spec in \
  'register 0 0 0 0' \
  'cross 1 0 0 0' \
  'localstage 1 1 0 0' \
  'overlap30 1 0 1 0' \
  'pipe2 1 0 1 1'; do
  read -r label cp localcp overlap pipe2 <<<"$spec"
  bin="$ONEESAN_BUILD_DIR/ab_overlap_local_${label}_n${N}"
  build_one "$label" "$cp" "$localcp" "$overlap" "$pipe2" "$bin"
  [[ "$RUN_PTXAS" == 1 ]] && python3 "$PARSER" "$LOGDIR/$label.build.err" --label "$label" >>"$RESOURCE"
  for ((r=1;r<=REPEATS;++r)); do run_one "$label" "$bin" "$r"; done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$RESOURCE" "$RUN_PTXAS" "$WINNER_ENV" <<'PY'
import csv,statistics,sys
src,dst,resource,run_ptxas,winner_env=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
modes=('register','cross','localstage','overlap30','pipe2'); out=[]
for mode in modes:
    g=[r for r in rows if r['mode']==mode]; z={'mode':mode,'repeats':len(g)}
    for k in ('wall_s','forward_high_s','reverse_high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct'):
        xs=[float(r[k]) for r in g if r[k] != 'NA']; z[k]=statistics.median(xs) if xs else None
    z['total_high_s']=z['forward_high_s']+z['reverse_high_s'] if z['forward_high_s'] is not None and z['reverse_high_s'] is not None else None
    out.append(z)
keys=('mode','repeats','wall_s','forward_high_s','reverse_high_s','total_high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct')
with open(dst,'w') as f:
    f.write('\t'.join(keys)+'\n')
    for z in out:f.write('\t'.join(str(z.get(k)) for k in keys)+'\n')
q={z['mode']:z for z in out}
base=q['register']['total_high_s']
for mode in modes:
    z=q[mode]
    if base and z['total_high_s']:
        print(f'{mode}_high_speedup_vs_register={base/z["total_high_s"]:.6f}x')
    print(f'{mode}_avg_memctrl_pct={z["avg_memctrl_util_pct"]}')
    print(f'{mode}_max_memctrl_pct={z["max_memctrl_util_pct"]}')
valid=[z for z in out if z['total_high_s']]
if not valid: raise SystemExit('no valid candidate')
best=min(valid,key=lambda z:z['total_high_s'])
flags={
 'register':(0,0,0,0),
 'cross':(1,0,0,0),
 'localstage':(1,1,0,0),
 'overlap30':(1,0,1,0),
 'pipe2':(1,0,1,1),
}[best['mode']]
print(f'WINNER={best["mode"]}')
with open(winner_env,'w') as f:
    f.write(f'CPASYNC_PAIR={flags[0]}\n')
    f.write(f'CPASYNC_LOCAL_PAIR={flags[1]}\n')
    f.write(f'CPASYNC_OVERLAP_LOCAL_PAIR={flags[2]}\n')
    f.write(f'CPASYNC_OVERLAP_LOCAL_PIPE2={flags[3]}\n')
    f.write(f'CPASYNC_PAIR_MODE={best["mode"]}\n')
if run_ptxas=='1':
    rr=list(csv.DictReader(open(resource),delimiter='\t'))
    for mode in modes:
        g=[r for r in rr if r['backend']==mode and 'high' in r['kernel'].lower()]
        regs=[int(r['registers']) for r in g if r['registers']!='NA']
        ss=[int(r['spill_store_bytes']) for r in g if r['spill_store_bytes']!='NA']
        sl=[int(r['spill_load_bytes']) for r in g if r['spill_load_bytes']!='NA']
        print(f'{mode}_high_max_registers={max(regs) if regs else "NA"}')
        print(f'{mode}_high_spill_store_bytes={sum(ss) if ss else "NA"}')
        print(f'{mode}_high_spill_load_bytes={sum(sl) if sl else "NA"}')
PY

echo "b300-overlap-local-pipe2-ab OK n=$N repeats=$REPEATS result=$RESULT winner_env=$WINNER_ENV" >&2
