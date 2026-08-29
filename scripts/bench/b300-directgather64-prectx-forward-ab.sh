#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; NGPU="${NGPU:-8}"
TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
ARCH="${ARCH:-sm_103}"; REPEATS="${REPEATS:-1}"
COL_ILP="${COL_ILP:-2}"; PM_ACCUM="${PM_ACCUM:-1}"
THREADS="${BUCKET_THREADS:-256}"; GX="${BUCKET_GRID_X:-32}"; GY="${BUCKET_GRID_Y:-8}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.25}"; RUN_PEER_PROBE="${RUN_PEER_PROBE:-1}"
if [[ -z "${EXPECT+x}" ]]; then
  [[ "$N" == 21 && "$MOD" == 4294967291 ]] && EXPECT=998035516 || {
    echo "EXPECT required for N=$N MOD=$MOD" >&2; exit 2;
  }
fi
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU GPUs" >&2; exit 2; }

PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_directgather64_prectx_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

if [[ "$RUN_PEER_PROBE" == 1 ]]; then
  ARCH="$ARCH" NGPU="$NGPU" THREADS="$THREADS" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" \
    >"$LOGDIR/peer_probe.out" 2>"$LOGDIR/peer_probe.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/peer_probe.out" || {
    echo 'remote-peer cp.async preflight failed' >&2; exit 5;
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

printf 'mode\tcpasync\tprectx\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_memctrl_util_pct\tmax_memctrl_util_pct\tforward_occupancy_pct\treverse_occupancy_pct\tprectx_mib\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

for cp in 0 1; do
  for pre in 0 1; do
    mode="reg64"; [[ "$cp" == 1 ]] && mode="cpa64"; [[ "$pre" == 1 ]] && mode="${mode}_prectx"
    bin="$ONEESAN_BUILD_DIR/b300_${mode}_ab_n${N}"
    N="$N" ARCH="$ARCH" OUT="$bin" COL_ILP="$COL_ILP" PM_ACCUM="$PM_ACCUM" \
      DEPTHMAJOR=1 PAIR_MLP=1 MLP_WINDOW4=1 DIRECTGATHER64=1 CPASYNC_PAIR="$cp" \
      PRECTX_FORWARD="$pre" FORCE7=0 PREFETCH_NEXT=0 SORTED=0 \
      PTXAS_VERBOSE=1 TRANSPOSE_MODE=pipeline \
      bash "$ONEESAN_ROOT/scripts/build/b300-directgather-colilp-fast.sh" \
      >"$LOGDIR/$mode.build.out" 2>"$LOGDIR/$mode.build.err"
    python3 "$PARSER" "$LOGDIR/$mode.build.err" --label "$mode" >>"$RESOURCE"

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
      high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
      read -r ag am mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++} END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
      occ="$(grep 'rankformula_high_occupancy ' "$se" | head -n1 || true)"
      fo="$(field forward_warp_occupancy_pct "$occ")"; ro="$(field reverse_warp_occupancy_pct "$occ")"
      preline="$(grep 'p10dc_prectx_forward ' "$se" | head -n1 || true)"
      pmib="$(field total_mib "$preline")"; [[ -n "$pmib" ]] || pmib=0
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$mode" "$cp" "$pre" "$r" "$residue" "$wall" "$fh" "$rh" "$high" "$am" "$mm" "${fo:-NA}" "${ro:-NA}" "$pmib" >>"$RESULT"
    done
  done
done

cat "$RESULT"
echo '--- ptxas resources ---'; cat "$RESOURCE"
python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
def stat(mode,key):
    x=[float(r[key]) for r in rows if r['mode']==mode and r[key]!='NA']
    return statistics.median(x) if x else None
for base in ('reg64','cpa64'):
    pre=base+'_prectx'
    bf,bp=stat(base,'forward_high_s'),stat(pre,'forward_high_s')
    br,rp=stat(base,'reverse_high_s'),stat(pre,'reverse_high_s')
    bh,ph=stat(base,'high_s'),stat(pre,'high_s')
    if bf and bp: print(f'{pre}_forward_speedup={bf/bp:.6f}x')
    if br and rp: print(f'{pre}_reverse_speedup={br/rp:.6f}x')
    if bh and ph: print(f'{pre}_high_speedup={bh/ph:.6f}x')
for mode in ('reg64','reg64_prectx','cpa64','cpa64_prectx'):
    print(mode, 'forward=',stat(mode,'forward_high_s'),'reverse=',stat(mode,'reverse_high_s'),'high=',stat(mode,'high_s'),'memctrl=',stat(mode,'avg_memctrl_util_pct'))
best=min({r['mode'] for r in rows},key=lambda m:stat(m,'high_s'))
print(f'BEST_HIGH={best} high_s={stat(best,"high_s"):.6f}')
print('prectx_expected_scope=forward_only selection_metric=high_s')
PY

echo "b300-directgather64-prectx-forward-ab OK result=$RESULT" >&2
