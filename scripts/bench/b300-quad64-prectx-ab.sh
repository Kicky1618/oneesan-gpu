#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-sm_103}"; REPEATS="${REPEATS:-1}"
THREADS="${BUCKET_THREADS:-256}"; GX="${BUCKET_GRID_X:-32}"; GY="${BUCKET_GRID_Y:-8}"
PM_ACCUM="${PM_ACCUM:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.25}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || {
    echo "EXPECT required for N=$N MOD=$MOD" >&2; exit 2;
  }
fi
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU GPUs" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_quad64_prectx_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

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

printf 'mode\tquad\tcol_ilp\tpre_fwd\tpre_rev\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\tforward_occupancy_pct\treverse_occupancy_pct\tprectx_entries_per_gpu\tprectx_mib_per_gpu\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

for spec in \
  'pair64 0 2 0 0' \
  'quad64 1 4 0 0' \
  'pair64_both 0 2 1 1' \
  'quad64_both 1 4 1 1'; do
  read -r mode quad col pf pr <<<"$spec"
  bin="$ONEESAN_BUILD_DIR/b300_${mode}_ab_n${N}"
  N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP="$col" PM_ACCUM="$PM_ACCUM" \
    DEPTHMAJOR=1 PAIR_MLP=1 QUAD_MLP="$quad" MLP_WINDOW4=1 DIRECTGATHER64=1 \
    CPASYNC_PAIR=0 CPASYNC_LOCAL_PAIR=0 CPASYNC_OVERLAP_LOCAL_PAIR=0 \
    PRECTX_FORWARD="$pf" PRECTX_REVERSE="$pr" FORCE7=0 PREFETCH_NEXT=0 SORTED=0 \
    DIRECTGATHER_SPARSE64=0 PTXAS_VERBOSE=1 TRANSPOSE_MODE=pipeline \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" \
    >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
  python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" >>"$RESOURCE" || true

  for ((r=1;r<=REPEATS;++r)); do
    so="$LOGDIR/${mode}_r${r}.out"; se="$LOGDIR/${mode}_r${r}.err"; util="$LOGDIR/${mode}_r${r}.util"
    BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$GX" BUCKET_GRID_Y="$GY" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
    pid=$!; sample_process "$pid" "$util" & sampler=$!
    set +e; wait "$pid"; rc=$?; set -e
    wait "$sampler" || true
    (( rc == 0 )) || { echo "$mode failed rc=$rc" >&2; exit "$rc"; }

    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 3; }
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue=$residue expected=$EXPECT" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; wall="$(field wall_s "$line")"
    [[ -n "$fh" && -n "$rh" && -n "$wall" ]] || { echo "$mode missing timing fields" >&2; exit 3; }
    high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
    read -r ag am mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
    occ="$(grep 'rankformula_high_occupancy ' "$se" | head -n1 || true)"
    fo="$(field forward_warp_occupancy_pct "$occ")"; ro="$(field reverse_warp_occupancy_pct "$occ")"
    preline="$(grep 'p10dc_prectx_high fixed_owner=' "$se" | head -n1 || true)"
    if [[ -n "$preline" ]]; then
      cbytes="$(field closure_context_bytes "$preline")"
      f0="$(field fwd_nn "$preline")"; f1="$(field fwd_nrnl "$preline")"
      r0="$(field rev_nn "$preline")"; r1="$(field rev_nr "$preline")"; r2="$(field rev_nl "$preline")"
      entries=$(( ${f0:-0} + ${f1:-0} + ${r0:-0} + ${r1:-0} + ${r2:-0} ))
      pmib="$(python3 - "${cbytes:-0}" "$entries" <<'PY'
import sys
print(f'{int(sys.argv[1])*int(sys.argv[2])/(1<<20):.6f}')
PY
)"
    else
      entries=0; pmib=0
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$mode" "$quad" "$col" "$pf" "$pr" "$r" "$residue" "$wall" "$fh" "$rh" "$high" \
      "$ag" "$am" "$mm" "${fo:-NA}" "${ro:-NA}" "$entries" "$pmib" >>"$RESULT"
  done
done

cat "$RESULT"
echo '--- ptxas resources ---'; cat "$RESOURCE"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
def med(mode,key):
    xs=[float(r[key]) for r in rows if r['mode']==mode and r[key]!='NA']
    return statistics.median(xs) if xs else None
modes=('pair64','quad64','pair64_both','quad64_both')
for m in modes:
    print(m, 'wall=',med(m,'wall_s'),'forward=',med(m,'forward_high_s'),'reverse=',med(m,'reverse_high_s'),
          'high=',med(m,'high_s'),'memctrl=',med(m,'avg_memctrl_util_pct'),'prectx_mib=',med(m,'prectx_mib_per_gpu'))

def ratio(a,b,key):
    x,y=med(a,key),med(b,key)
    return None if not x or not y else x/y
for key in ('forward_high_s','reverse_high_s','high_s','wall_s'):
    v=ratio('pair64','quad64',key)
    if v is not None: print(f'quad64_vs_pair64_{key}_speedup={v:.6f}x')
    v=ratio('pair64_both','quad64_both',key)
    if v is not None: print(f'quad64_both_vs_pair64_both_{key}_speedup={v:.6f}x')
    v=ratio('pair64','pair64_both',key)
    if v is not None: print(f'pair64_prectx_{key}_speedup={v:.6f}x')
    v=ratio('quad64','quad64_both',key)
    if v is not None: print(f'quad64_prectx_{key}_speedup={v:.6f}x')
best=min(modes,key=lambda m:med(m,'high_s'))
print(f'BEST_HIGH={best} high_s={med(best,"high_s"):.6f} wall_s={med(best,"wall_s"):.6f}')
print('pair_col_ilp=2 quad_col_ilp=4 directgather64=1 cpasync=0 selection_metric=high_s wall_s_sanity=1 correctness_gate=residue metadata_cost_reported=1')
PY

echo "b300-quad64-prectx-ab OK result=$RESULT ptxas=$RESOURCE" >&2
