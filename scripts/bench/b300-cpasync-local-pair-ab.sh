#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
COL_ILP="${COL_ILP:-2}"; PM_ACCUM="${PM_ACCUM:-1}"; REPEATS="${REPEATS:-2}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-32}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.10}"; RUN_PEER_PROBE="${RUN_PEER_PROBE:-1}"
SORTED="${SORTED:-1}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required" >&2; exit 2; }
fi
for x in RUN_PEER_PROBE SORTED; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_cpasync_local_pair_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

if [[ "$RUN_PEER_PROBE" == 1 ]]; then
  ARCH="$ARCH" NGPU="$NGPU" THREADS="$BUCKET_THREADS" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" \
    >"$LOGDIR/peer_probe.out" 2>"$LOGDIR/peer_probe.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/peer_probe.out" || { echo 'cp.async peer probe failed' >&2; exit 5; }
fi

field(){ local key="$1" line="$2"; sed -nE "s/(^|.*[[:space:]])${key}=([^[:space:]]+).*/\\2/p" <<<"$line" | tail -n1; }
sample_process(){
  local pid="$1" out="$2"; : >"$out"
  while kill -0 "$pid" 2>/dev/null; do
    nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' >>"$out" || true
    sleep "$SAMPLE_INTERVAL"
  done
}

printf 'local_cpasync\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

for local_cpa in 0 1; do
  mode="local${local_cpa}"
  bin="$ONEESAN_BUILD_DIR/b300_cpasync_local_pair_${mode}_n${N}"
  N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP="$COL_ILP" PM_ACCUM="$PM_ACCUM" \
    DEPTHMAJOR=1 PAIR_MLP=1 MLP_WINDOW4=1 CPASYNC_PAIR=1 CPASYNC_LOCAL_PAIR="$local_cpa" \
    DIRECTGATHER64=1 DIRECTGATHER_SPARSE64=1 SORTED="$SORTED" \
    FORCE7=0 PREFETCH_NEXT=0 PRECTX_FORWARD=0 PRECTX_REVERSE=0 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" \
    >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
  python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" >>"$RESOURCE" || true

  for ((r=1;r<=REPEATS;++r)); do
    so="$LOGDIR/${mode}_r${r}.out"; se="$LOGDIR/${mode}_r${r}.err"; util="$LOGDIR/${mode}_r${r}.util"
    BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
    pid=$!; sample_process "$pid" "$util" & sampler=$!
    set +e; wait "$pid"; rc=$?; set -e
    wait "$sampler" || true
    (( rc == 0 )) || { echo "$mode failed rc=$rc" >&2; exit "$rc"; }
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 3; }
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue mismatch got=$residue expected=$EXPECT" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    wall="$(field wall_s "$line")"; fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
    high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
    read -r ag am mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$local_cpa" "$r" "$residue" "$wall" "$fh" "$rh" "$high" "$ag" "$am" "$mm" >>"$RESULT"
  done
done

cat "$RESULT"
python3 - "$RESULT" "$SUMMARY" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); out=[]
for mode in ('0','1'):
    g=[r for r in rows if r['local_cpasync']==mode]; z={'local_cpasync':mode,'repeats':len(g)}
    for k in ('wall_s','forward_high_s','reverse_high_s','high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct'):
        xs=[float(r[k]) for r in g if r[k]!='NA']; z[k]=statistics.median(xs) if xs else None
    out.append(z)
keys=('local_cpasync','repeats','wall_s','forward_high_s','reverse_high_s','high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct')
with open(sys.argv[2],'w') as f:
    f.write('\t'.join(keys)+'\n')
    for z in out:f.write('\t'.join(str(z[k]) for k in keys)+'\n')
q={z['local_cpasync']:z for z in out}
print(f'local_cpasync_high_speedup={q["0"]["high_s"]/q["1"]["high_s"]:.6f}x')
print(f'local_cpasync_wall_speedup={q["0"]["wall_s"]/q["1"]["wall_s"]:.6f}x')
if q['0']['avg_memctrl_util_pct'] is not None and q['1']['avg_memctrl_util_pct'] is not None:
    print(f'local_cpasync_memctrl_delta={q["1"]["avg_memctrl_util_pct"]-q["0"]["avg_memctrl_util_pct"]:.6f}pp')
print('descriptor=sparse64 source_order=sorted_if_enabled cross_staging=cpasync14 local_staging=cpasync14_plus_ldg2_if_enabled')
PY

echo "b300-cpasync-local-pair-ab OK result=$RESULT summary=$SUMMARY resource=$RESOURCE" >&2
