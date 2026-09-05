#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-sm_103}"; REPEATS="${REPEATS:-1}"; PM_ACCUM="${PM_ACCUM:-1}"
SPARSE64="${SPARSE64:-1}"; SORTED="${SORTED:-0}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"; BUCKET_GRID_X="${BUCKET_GRID_X:-32}"; BUCKET_GRID_Y="${BUCKET_GRID_Y:-8}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.25}"; RUN_PEER_PROBE="${RUN_PEER_PROBE:-1}"
for x in SPARSE64 SORTED RUN_PEER_PROBE; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo "EXPECT required for N=$N MOD=$MOD" >&2; exit 2; }
fi
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need at least $NGPU visible GPUs" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_cpasync_quad_vs_pair_n${N}_sp${SPARSE64}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

if [[ "$RUN_PEER_PROBE" == 1 ]]; then
  ARCH="$ARCH" NGPU="$NGPU" THREADS="$BUCKET_THREADS" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" \
    >"$LOGDIR/peer_probe.out" 2>"$LOGDIR/peer_probe.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/peer_probe.out" || { echo 'remote peer cp.async preflight failed' >&2; exit 5; }
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

printf 'mode\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\tforward_warp_occupancy_pct\treverse_warp_occupancy_pct\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

modes=(pair_local pair_overlap pair_pipe2 quad28)
for mode in "${modes[@]}"; do
  local_pair=0; overlap=0; pipe2=0; quad=0
  case "$mode" in
    pair_local) local_pair=1 ;;
    pair_overlap) overlap=1 ;;
    pair_pipe2) overlap=1; pipe2=1 ;;
    quad28) quad=1 ;;
  esac
  bin="$ONEESAN_BUILD_DIR/b300_${mode}_n${N}_sp${SPARSE64}"
  N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP=4 PM_ACCUM="$PM_ACCUM" \
    DEPTHMAJOR=1 PAIR_MLP=1 QUAD_MLP="$quad" MLP_WINDOW4=1 \
    DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" SORTED="$SORTED" \
    CPASYNC_PAIR=1 CPASYNC_LOCAL_PAIR="$local_pair" \
    CPASYNC_OVERLAP_LOCAL_PAIR="$overlap" CPASYNC_OVERLAP_LOCAL_PIPE2="$pipe2" \
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
print(f"{float(sys.argv[1])+float(sys.argv[2]):.9f}")
PY
)"
    read -r ag am mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
    occ="$(grep 'rankformula_high_occupancy' "$se" | head -n1 || true)"
    focc="$(field forward_warp_occupancy_pct "$occ")"; rocc="$(field reverse_warp_occupancy_pct "$occ")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$mode" "$r" "$residue" "$wall" "$fh" "$rh" "$high" "$ag" "$am" "$mm" "${focc:-NA}" "${rocc:-NA}" >>"$RESULT"
  done
done

cat "$RESULT"
python3 - "$RESULT" "$RESOURCE" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); by={}
for r in rows: by.setdefault(r['mode'],[]).append(r)
res={m:{x['residue'] for x in g} for m,g in by.items()}
if any(len(v)!=1 for v in res.values()) or len({next(iter(v)) for v in res.values()})!=1: raise SystemExit(f'RESIDUE MISMATCH {res}')
print('residue_match=OK',next(iter(next(iter(res.values())))))
q={}
for m,g in by.items():
    high=statistics.median(float(x['high_s']) for x in g); wall=statistics.median(float(x['wall_s']) for x in g)
    mc=statistics.median(float(x['avg_memctrl_util_pct']) for x in g if x['avg_memctrl_util_pct']!='NA')
    occ=[float(x['forward_warp_occupancy_pct']) for x in g if x['forward_warp_occupancy_pct']!='NA']
    q[m]=(high,wall,mc,statistics.median(occ) if occ else float('nan'))
    print(m,f'high_s={high:.6f}',f'wall_s={wall:.6f}',f'mc_avg_pct={mc:.3f}',f'fwd_occ_pct={q[m][3]:.3f}')
base=min((m for m in q if m!='quad28'),key=lambda m:q[m][0])
print('BEST_PAIR',base,f'high_s={q[base][0]:.6f}')
print('quad_vs_best_pair_high_speedup',f'{q[base][0]/q["quad28"][0]:.6f}x')
print('quad_vs_best_pair_wall_speedup',f'{q[base][1]/q["quad28"][1]:.6f}x')
print('quad_vs_best_pair_mc_delta_pct',f'{q["quad28"][2]-q[base][2]:.3f}')
print('BEST_HIGH',min(q,key=lambda m:q[m][0]))
PY

echo "b300-cpasync-quad-vs-pair-ab OK result=$RESULT resources=$RESOURCE" >&2
