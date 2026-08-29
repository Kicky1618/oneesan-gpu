#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"; ARCH="${ARCH:-sm_103}"
THREADS="${BUCKET_THREADS:-256}"; ORBIT_GY="${BUCKET_ORBITCTA_GRID_Y:-128}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
COL_ILP="${COL_ILP:-2}"; PAIR_MLP="${PAIR_MLP:-1}"; WINDOW4="${WINDOW4:-1}"; PM_ACCUM="${PM_ACCUM:-1}"
DIRECTGATHER_SPARSE64="${DIRECTGATHER_SPARSE64:-0}"; CPASYNC_PAIR="${CPASYNC_PAIR:-0}"; REPEATS="${REPEATS:-1}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.20}"
if [[ -z "${EXPECT+x}" ]]; then [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || { echo 'EXPECT required' >&2; exit 2; }; fi
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }
for x in PAIR_MLP WINDOW4 PM_ACCUM DIRECTGATHER_SPARSE64 CPASYNC_PAIR; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done
case "$COL_ILP" in 1|2|4) ;; *) echo 'COL_ILP must be 1,2,4' >&2; exit 2;; esac
if [[ "$PAIR_MLP" == 1 ]]; then [[ "$WINDOW4" == 1 ]] || { echo 'PAIR_MLP requires WINDOW4=1' >&2; exit 2; }; [[ "$COL_ILP" == 2 || "$COL_ILP" == 4 ]] || { echo 'PAIR_MLP requires ILP2/4' >&2; exit 2; }; fi
if [[ "$CPASYNC_PAIR" == 1 ]]; then [[ "$PAIR_MLP" == 1 ]] || { echo 'CPASYNC_PAIR requires PAIR_MLP=1' >&2; exit 2; }; fi

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbit_high_prectx_n${N}}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"
RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }

if [[ "$CPASYNC_PAIR" == 1 ]]; then
  ARCH="$ARCH" NGPU="$NGPU" THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/cpasync-peer.out" 2>"$LOGDIR/cpasync-peer.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync-peer.out" || { echo 'cp.async peer probe failed' >&2; exit 5; }
fi

printf 'mode\tpre_fwd\tpre_rev\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tprectx_entries_per_gpu\tprectx_mib_per_gpu\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for spec in 'runtime 0 0' 'forward 1 0' 'both 1 1'; do
  read -r mode pf pr <<<"$spec"; bin="$ONEESAN_BUILD_DIR/b300_orbit_prectx_${mode}_sp${DIRECTGATHER_SPARSE64}_n${N}"
  N="$N" ARCH="$ARCH" OUT="$bin" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$DIRECTGATHER_SPARSE64" \
    ORBITCTA_COL_ILP="$COL_ILP" PAIR_MLP="$PAIR_MLP" CPASYNC_PAIR="$CPASYNC_PAIR" \
    RANKFORMULA_MLP_WINDOW4="$WINDOW4" PM_ACCUM="$PM_ACCUM" PRECTX_FORWARD="$pf" PRECTX_REVERSE="$pr" PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
  python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" >>"$RESOURCE" || true
  for ((r=1;r<=REPEATS;++r)); do
    so="$LOGDIR/${mode}_r${r}.out"; se="$LOGDIR/${mode}_r${r}.err"; util="$LOGDIR/${mode}_r${r}.util"
    BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" & pid=$!
    sample "$pid" "$util" & sp=$!; set +e; wait "$pid"; rc=$?; set -e; wait "$sp" || true; ((rc==0)) || { echo "$mode failed rc=$rc" >&2; exit "$rc"; }
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 3; }
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue=$residue expected=$EXPECT" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"; fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
    high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
    preline="$(grep 'p10dc_prectx_high fixed_owner=' "$se" | head -n1 || true)"
    if [[ -n "$preline" ]]; then
      cbytes="$(field closure_context_bytes "$preline")"; f0="$(field fwd_nn "$preline")"; f1="$(field fwd_nrnl "$preline")"; r0="$(field rev_nn "$preline")"; r1="$(field rev_nr "$preline")"; r2="$(field rev_nl "$preline")"
      entries=$(( ${f0:-0}+${f1:-0}+${r0:-0}+${r1:-0}+${r2:-0} ))
      premib="$(python3 - "${cbytes:-0}" "$entries" <<'PY'
import sys
print(f'{int(sys.argv[1])*int(sys.argv[2])/(1<<20):.6f}')
PY
)"
    else entries=0; premib=0; fi
    read -r ag am mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$pf" "$pr" "$r" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$entries" "$premib" "$ag" "$am" "$mm" >>"$RESULT"
  done
done
cat "$RESULT"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); q={}
for m in ('runtime','forward','both'):
 g=[r for r in rows if r['mode']==m]
 q[m]={k:statistics.median(float(r[k]) for r in g) for k in ('wall_s','forward_high_s','reverse_high_s','high_s','prectx_mib_per_gpu','avg_memctrl_util_pct')}
 print(m,q[m])
base=q['runtime']
for m in ('forward','both'):
 print(f'{m}_forward_speedup={base["forward_high_s"]/q[m]["forward_high_s"]:.6f}x')
 print(f'{m}_reverse_speedup={base["reverse_high_s"]/q[m]["reverse_high_s"]:.6f}x')
 print(f'{m}_high_speedup={base["high_s"]/q[m]["high_s"]:.6f}x')
 print(f'{m}_wall_speedup={base["wall_s"]/q[m]["wall_s"]:.6f}x')
 print(f'{m}_prectx_mib_per_gpu={q[m]["prectx_mib_per_gpu"]:.3f}')
print('selection_metric=wall_s_and_high_s correctness_gate=residue orbit_prectx=1')
PY

echo "result=$RESULT ptxas=$RESOURCE sparse64=$DIRECTGATHER_SPARSE64 col_ilp=$COL_ILP pair_mlp=$PAIR_MLP cpasync_pair=$CPASYNC_PAIR" >&2
