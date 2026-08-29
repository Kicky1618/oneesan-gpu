#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-sm_103}"; REPEATS="${REPEATS:-1}"
COL_ILP="${COL_ILP:-2}"; PM_ACCUM="${PM_ACCUM:-1}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"
BUCKET_GRID_X="${BUCKET_GRID_X:-32}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.25}"; RUN_PEER_PROBE="${RUN_PEER_PROBE:-1}"
MAXRREGCOUNT="${MAXRREGCOUNT:-0}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || {
    echo "EXPECT required for N=$N MOD=$MOD" >&2; exit 2;
  }
fi
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || {
  echo "need at least $NGPU visible GPUs" >&2; exit 2;
}

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather_cpasync_pair_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

if [[ "$RUN_PEER_PROBE" == 1 ]]; then
  ARCH="$ARCH" NGPU="$NGPU" THREADS="$BUCKET_THREADS" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" \
    >"$LOGDIR/peer_probe.out" 2>"$LOGDIR/peer_probe.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/peer_probe.out" || {
    echo 'remote-peer cp.async preflight did not report exact=OK' >&2; exit 5;
  }
fi

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

printf 'mode\tdirectgather64\tcpasync\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\tforward_warp_occupancy_pct\treverse_warp_occupancy_pct\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

for dg64 in 0 1; do
  for cp in 0 1; do
    if [[ "$dg64" == 1 && "$cp" == 1 ]]; then mode=cpasync64
    elif [[ "$dg64" == 1 ]]; then mode=register64
    elif [[ "$cp" == 1 ]]; then mode=cpasync16
    else mode=register16
    fi
    bin="$ONEESAN_BUILD_DIR/b300_directgather_pair_${mode}_ab_n${N}"
    N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP="$COL_ILP" PM_ACCUM="$PM_ACCUM" \
      DEPTHMAJOR=1 PAIR_MLP=1 MLP_WINDOW4=1 CPASYNC_PAIR="$cp" \
      FORCE7=0 PREFETCH_NEXT=0 DIRECTGATHER64="$dg64" SORTED=0 \
      MAXRREGCOUNT="$MAXRREGCOUNT" PTXAS_VERBOSE=1 \
      bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" \
      >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
    python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" >>"$RESOURCE"

    for ((r=1;r<=REPEATS;++r)); do
      so="$LOGDIR/${mode}_r${r}.out"; se="$LOGDIR/${mode}_r${r}.err"; util="$LOGDIR/${mode}_r${r}.util"
      BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$BUCKET_GRID_X" BUCKET_GRID_Y="$BUCKET_GRID_Y" \
        "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
      pid=$!; sample_process "$pid" "$util" & sampler=$!
      set +e; wait "$pid"; rc=$?; set -e
      wait "$sampler" || true
      (( rc == 0 )) || { echo "$mode failed rc=$rc" >&2; exit "$rc"; }

      line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 3; }
      residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || {
        echo "$mode residue mismatch got=$residue expected=$EXPECT" >&2; exit 4;
      }
      detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
      wall="$(field wall_s "$line")"; fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
      high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f"{float(sys.argv[1])+float(sys.argv[2]):.9f}")
PY
)"
      read -r ag am mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
      occ="$(grep 'rankformula_high_occupancy' "$se" | head -n1 || true)"
      focc="$(field forward_warp_occupancy_pct "$occ")"; rocc="$(field reverse_warp_occupancy_pct "$occ")"
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$mode" "$dg64" "$cp" "$r" "$residue" "$wall" "$fh" "$rh" "$high" \
        "$ag" "$am" "$mm" "${focc:-NA}" "${rocc:-NA}" >>"$RESULT"
    done
  done
done

cat "$RESULT"
python3 - "$RESULT" "$RESOURCE" "$SUMMARY" <<'PY'
import csv,statistics,sys
src,res,dst=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t'))
resources=list(csv.DictReader(open(res),delimiter='\t'))
out=[]
modes=('register16','cpasync16','register64','cpasync64')
for mode in modes:
    g=[r for r in rows if r['mode']==mode]
    z={'mode':mode,'repeats':len(g)}
    for k in ('wall_s','high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct','forward_warp_occupancy_pct','reverse_warp_occupancy_pct'):
        xs=[float(r[k]) for r in g if r[k]!='NA']; z[k]=statistics.median(xs) if xs else None
    kr=[r for r in resources if r['backend']==mode and 'high' in r['kernel'].lower()]
    regs=[int(r['registers']) for r in kr if r['registers']!='NA']
    ss=[int(r['spill_store_bytes']) for r in kr if r['spill_store_bytes']!='NA']
    sl=[int(r['spill_load_bytes']) for r in kr if r['spill_load_bytes']!='NA']
    sm=[int(r['smem_bytes']) for r in kr if r['smem_bytes']!='NA']
    z.update(max_regs=max(regs) if regs else None,
             spill_store=sum(ss) if ss else None,
             spill_load=sum(sl) if sl else None,
             static_smem=max(sm) if sm else None)
    out.append(z)
keys=('mode','repeats','wall_s','high_s','avg_gpu_util_pct','avg_memctrl_util_pct','max_memctrl_util_pct','forward_warp_occupancy_pct','reverse_warp_occupancy_pct','max_regs','spill_store','spill_load','static_smem')
with open(dst,'w') as f:
    f.write('\t'.join(keys)+'\n')
    for z in out:f.write('\t'.join(str(z.get(k)) for k in keys)+'\n')
q={z['mode']:z for z in out}
for z in out: print(z)
for base,test,name in (
    ('register16','cpasync16','cpasync16'),
    ('register16','register64','pair64'),
    ('register64','cpasync64','cpasync64'),
    ('cpasync16','cpasync64','cpasync_descriptor64')):
    if q[base]['high_s'] and q[test]['high_s']:
        print(f"{name}_high_speedup={q[base]['high_s']/q[test]['high_s']:.6f}x")
valid=[z for z in out if z['high_s'] is not None]
if valid:
    best=min(valid,key=lambda z:z['high_s'])
    print(f"BEST_HIGH={best['mode']} high_s={best['high_s']:.6f} wall_s={best['wall_s']:.6f} memctrl={best['avg_memctrl_util_pct']}")
print('selection_metric=high_s correctness_gate=residue remote_peer_gate=cpasync_microprobe')
PY

echo "b300-directgather-cpasync-pair-ab OK result=$RESULT summary=$SUMMARY resources=$RESOURCE" >&2
