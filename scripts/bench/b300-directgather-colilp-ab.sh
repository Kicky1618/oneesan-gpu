#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-native}"; TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
REPEATS="${REPEATS:-1}"; BUCKET_THREADS="${BUCKET_THREADS:-256}"
BUCKET_GRID_X="${BUCKET_GRID_X:-32}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
PM_ACCUM="${PM_ACCUM:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.25}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"; ILPS="${ILPS:-1 2 4}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required for N=$N MOD=$MOD" >&2; exit 2; }
fi
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU GPUs" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather_colilp_n${N}_${TRANSPOSE_MODE}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

bash "$ONEESAN_ROOT/scripts/bench/warpstriped-col-ilp-proof.sh" >"$LOGDIR/coverage.out" 2>"$LOGDIR/coverage.err"

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }

sample_process(){
  local pid="$1" out="$2"
  : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' \
      >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}

build_one(){
  local ilp="$1" label="ilp$1" bin="$2"
  N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP="$ilp" PM_ACCUM="$PM_ACCUM" \
    MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE=1 TRANSPOSE_MODE="$TRANSPOSE_MODE" \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" \
    >"$LOGDIR/$label.build.out" 2>"$LOGDIR/$label.build.err"
  python3 "$PARSER" "$LOGDIR/$label.build.err" --label "$label" >>"$RESOURCE"
}

run_one(){
  local ilp="$1" bin="$2" rep="$3" label="ilp$1"
  local so="$LOGDIR/${label}_r${rep}.out" se="$LOGDIR/${label}_r${rep}.err" util="$LOGDIR/${label}_r${rep}.util"
  BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  local pid=$! sampler=''
  sample_process "$pid" "$util" & sampler=$!
  set +e
  wait "$pid"; local rc=$?
  set -e
  wait "$sampler" || true
  (( rc == 0 )) || { echo "$label failed rc=$rc" >&2; return "$rc"; }
  local line detail residue wall fh rh avg_gpu avg_mem max_gpu max_mem
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || return 3
  residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$label residue mismatch got=$residue expected=$EXPECT" >&2; return 4; }
  wall="$(field wall_s "$line")"
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
  read -r avg_gpu avg_mem max_gpu max_mem < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$ilp" "$rep" "$residue" "$wall" "${fh:-NA}" "${rh:-NA}" \
    "$avg_gpu" "$avg_mem" "$max_gpu" "$max_mem" "$bin" >>"$RESULT"
}

printf 'ilp\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_gpu_util_pct\tmax_memctrl_util_pct\tbin\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

for ilp in $ILPS; do
  case "$ilp" in 1|2|4) ;; *) echo "bad ILP in ILPS: $ilp" >&2; exit 2;; esac
  bin="$ONEESAN_BUILD_DIR/b300_directgather_colilp${ilp}_ab_n${N}"
  build_one "$ilp" "$bin"
  for ((r=1;r<=REPEATS;++r)); do run_one "$ilp" "$bin" "$r"; done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" "$RESOURCE" <<'PY'
import csv,statistics,sys
src,dst,res=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
rr=list(csv.DictReader(open(res),delimiter='\t'))
out=[]
for ilp in sorted({int(r['ilp']) for r in rows}):
    g=[r for r in rows if int(r['ilp'])==ilp]
    z={'ilp':ilp,'repeats':len(g)}
    for k in ('wall_s','forward_high_s','reverse_high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct'):
        xs=[float(r[k]) for r in g if r[k]!='NA']; z[k]=statistics.median(xs) if xs else None
    z['total_high_s']=(z['forward_high_s']+z['reverse_high_s']) if z['forward_high_s'] is not None and z['reverse_high_s'] is not None else None
    kr=[r for r in rr if r['backend']==f'ilp{ilp}' and 'high' in r['kernel'].lower()]
    regs=[int(r['registers']) for r in kr if r['registers']!='NA']
    ss=[int(r['spill_store_bytes']) for r in kr if r['spill_store_bytes']!='NA']
    sl=[int(r['spill_load_bytes']) for r in kr if r['spill_load_bytes']!='NA']
    z['max_regs']=max(regs) if regs else None; z['spill_store']=sum(ss) if ss else None; z['spill_load']=sum(sl) if sl else None
    out.append(z)
keys=('ilp','repeats','wall_s','forward_high_s','reverse_high_s','total_high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct','max_regs','spill_store','spill_load')
with open(dst,'w') as f:
    f.write('\t'.join(keys)+'\n')
    for z in out:f.write('\t'.join(str(z.get(k)) for k in keys)+'\n')
for z in out:
    print('ILP',z['ilp'],
          f"wall={z['wall_s']}",f"high={z['total_high_s']}",
          f"memavg={z['avg_memctrl_util_pct']}",f"memmax={z['max_memctrl_util_pct']}",
          f"regs={z['max_regs']}",f"spill_store={z['spill_store']}",f"spill_load={z['spill_load']}")
valid=[z for z in out if z['wall_s'] is not None]
if valid:
    best=min(valid,key=lambda z:z['wall_s'])
    print('BEST_ILP',best['ilp'],f"wall={best['wall_s']}",f"memavg={best['avg_memctrl_util_pct']}")
PY

echo "b300-directgather-colilp-ab OK n=$N gx=$BUCKET_GRID_X gy=$BUCKET_GRID_Y repeats=$REPEATS result=$RESULT" >&2
