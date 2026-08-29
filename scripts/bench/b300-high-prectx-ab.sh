#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-sm_103}"; REPEATS="${REPEATS:-1}"
THREADS="${BUCKET_THREADS:-256}"; GX="${BUCKET_GRID_X:-32}"; GY="${BUCKET_GRID_Y:-8}"
COL_ILP="${COL_ILP:-2}"; PAIR_MLP="${PAIR_MLP:-1}"; WINDOW4="${MLP_WINDOW4:-1}"
SORTED="${SORTED:-0}"; CPASYNC_PAIR="${CPASYNC_PAIR:-0}"; PM_ACCUM="${PM_ACCUM:-1}"
DIRECTGATHER64="${DIRECTGATHER64:-1}"; DIRECTGATHER_SPARSE64="${DIRECTGATHER_SPARSE64:-0}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.25}"
if [[ -z "${EXPECT+x}" ]]; then [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo 'EXPECT required' >&2; exit 2; }; fi
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
for x in PAIR_MLP WINDOW4 SORTED CPASYNC_PAIR PM_ACCUM DIRECTGATHER64 DIRECTGATHER_SPARSE64; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
[[ "$DIRECTGATHER_SPARSE64" == 0 || "$DIRECTGATHER64" == 1 ]] || { echo 'DIRECTGATHER_SPARSE64 requires DIRECTGATHER64=1' >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_high_prectx_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"; mkdir -p "$LOGDIR"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{sg+=$1+0;sm+=$2+0;if(($2+0)>mm)mm=$2+0;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }

printf 'mode\tpre_fwd\tpre_rev\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tprectx_entries_per_gpu\tprectx_mib_per_gpu\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for spec in 'runtime 0 0' 'forward 1 0' 'both 1 1'; do
  read -r mode pf pr <<<"$spec"; bin="$ONEESAN_BUILD_DIR/b300_high_prectx_${mode}_dg64${DIRECTGATHER64}_sp64${DIRECTGATHER_SPARSE64}_ab_n${N}"
  N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP="$COL_ILP" PM_ACCUM="$PM_ACCUM" DEPTHMAJOR=1 \
    PAIR_MLP="$PAIR_MLP" MLP_WINDOW4="$WINDOW4" SORTED="$SORTED" CPASYNC_PAIR="$CPASYNC_PAIR" \
    PRECTX_FORWARD="$pf" PRECTX_REVERSE="$pr" FORCE7=0 PREFETCH_NEXT=0 \
    DIRECTGATHER64="$DIRECTGATHER64" DIRECTGATHER_SPARSE64="$DIRECTGATHER_SPARSE64" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
  python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" >>"$RESOURCE" || true
  for ((r=1;r<=REPEATS;++r)); do
    so="$LOGDIR/${mode}_r${r}.out"; se="$LOGDIR/${mode}_r${r}.err"; u="$LOGDIR/${mode}_r${r}.util"
    BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$GX" BUCKET_GRID_Y="$GY" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" & pid=$!
    sample "$pid" "$u" & sp=$!; set +e; wait "$pid"; rc=$?; set -e; wait "$sp" || true; ((rc==0)) || exit "$rc"
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || exit 3
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue=$residue expected=$EXPECT" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
    fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
    high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f"{float(sys.argv[1])+float(sys.argv[2]):.9f}")
PY
)"
    preline="$(grep 'p10dc_prectx_high fixed_owner=' "$se" | head -n1 || true)"
    if [[ -n "$preline" ]]; then
      cbytes="$(field closure_context_bytes "$preline")"
      f0="$(field fwd_nn "$preline")"; f1="$(field fwd_nrnl "$preline")"
      r0="$(field rev_nn "$preline")"; r1="$(field rev_nr "$preline")"; r2="$(field rev_nl "$preline")"
      entries=$(( ${f0:-0} + ${f1:-0} + ${r0:-0} + ${r1:-0} + ${r2:-0} ))
      premib="$(python3 - "${cbytes:-0}" "$entries" <<'PY'
import sys
print(f'{int(sys.argv[1])*int(sys.argv[2])/(1<<20):.6f}')
PY
)"
    else
      entries=0; premib=0
    fi
    read -r ag am mm < <(awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' "$u")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$pf" "$pr" "$r" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$entries" "$premib" "$ag" "$am" "$mm" >>"$RESULT"
  done
done
cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); q={}
for m in ('runtime','forward','both'):
 g=[x for x in r if x['mode']==m]
 q[m]={k:statistics.median(float(x[k]) for x in g) for k in ('wall_s','forward_high_s','reverse_high_s','high_s','prectx_mib_per_gpu','avg_memctrl_util_pct')}
 print(m,q[m])
base=q['runtime']
for m in ('forward','both'):
 print(f"{m}_forward_speedup={base['forward_high_s']/q[m]['forward_high_s']:.6f}x")
 print(f"{m}_reverse_speedup={base['reverse_high_s']/q[m]['reverse_high_s']:.6f}x")
 print(f"{m}_high_speedup={base['high_s']/q[m]['high_s']:.6f}x")
 print(f"{m}_wall_speedup={base['wall_s']/q[m]['wall_s']:.6f}x")
 print(f"{m}_prectx_mib_per_gpu={q[m]['prectx_mib_per_gpu']:.3f}")
print('selection_metric=wall_s_and_high_s correctness_gate=residue metadata_cost_reported=1')
PY

echo "directgather64=$DIRECTGATHER64 sparse64=$DIRECTGATHER_SPARSE64 col_ilp=$COL_ILP pair_mlp=$PAIR_MLP result=$RESULT ptxas=$RESOURCE" >&2
