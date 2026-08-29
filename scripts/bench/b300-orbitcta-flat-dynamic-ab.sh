#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
SPARSE64="${DIRECTGATHER_SPARSE64:-1}"; SORT_RANKS="${DIRECTGATHER_SORT_RANKS:-0}"
COL_ILP="${ORBITCTA_COL_ILP:-2}"; PAIR_MLP="${PAIR_MLP:-1}"; CPASYNC_PAIR="${CPASYNC_PAIR:-0}"
WINDOW4="${RANKFORMULA_MLP_WINDOW4:-1}"; PM_ACCUM="${PM_ACCUM:-1}"
PRECTX_FLAT_BID="${PRECTX_FLAT_BID:-0}"; PRECTX_FLAT_BID_FUSED="${PRECTX_FLAT_BID_FUSED:-0}"
PRECTX_FORWARD="${PRECTX_FORWARD:-$PRECTX_FLAT_BID}"; PRECTX_REVERSE="${PRECTX_REVERSE:-$PRECTX_FLAT_BID}"; PRECTX_COMPACT="${PRECTX_COMPACT:-$PRECTX_FLAT_BID}"
FLAT_PER_SM="${BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM:-}"; FLAT_BLOCKS="${BUCKET_ORBITCTA_FLAT_BLOCKS:-}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbitcta_flat_dynamic_ab_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

[[ "$N" == 21 && "$MOD" == 4294967291 && "$EXPECT" == 998035516 ]] || { echo 'fixed exact gate is n=21/mod4294967291/residue998035516' >&2; exit 2; }
for x in SPARSE64 SORT_RANKS PAIR_MLP CPASYNC_PAIR WINDOW4 PM_ACCUM PRECTX_FLAT_BID PRECTX_FLAT_BID_FUSED PRECTX_FORWARD PRECTX_REVERSE PRECTX_COMPACT; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
case "$COL_ILP" in 1|2|4) ;; *) echo 'ORBITCTA_COL_ILP must be 1,2,4' >&2; exit 2;; esac
[[ "$SORT_RANKS" == 0 || "$SPARSE64" == 0 ]] || { echo 'sorted ranks currently require dense64' >&2; exit 2; }
[[ -z "$FLAT_PER_SM" || -z "$FLAT_BLOCKS" ]] || { echo 'set at most one flat pool override' >&2; exit 2; }
if [[ "$PAIR_MLP" == 1 ]]; then [[ "$WINDOW4" == 1 ]] || exit 2; [[ "$COL_ILP" == 2 || "$COL_ILP" == 4 ]] || exit 2; fi
[[ "$CPASYNC_PAIR" == 0 || "$PAIR_MLP" == 1 ]] || exit 2
if [[ "$PRECTX_FLAT_BID" == 1 ]]; then
  [[ "$PRECTX_FORWARD" == 1 && "$PRECTX_REVERSE" == 1 && "$PRECTX_COMPACT" == 1 ]] || { echo 'PRECTX_FLAT_BID requires full compact prectx' >&2; exit 2; }
fi
[[ "$PRECTX_FLAT_BID_FUSED" == 0 || "$PRECTX_FLAT_BID" == 1 ]] || exit 2
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

if [[ "$PRECTX_COMPACT" == 1 ]]; then
  ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" >"$LOGDIR/compact-prectx.out" 2>"$LOGDIR/compact-prectx.err"
fi
if [[ "$PRECTX_FLAT_BID" == 1 ]]; then
  ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-flat-bid-selftest.sh" >"$LOGDIR/compact-flat-bid.out" 2>"$LOGDIR/compact-flat-bid.err"
  grep -q 'bucket-compact-flat-bid-selftest OK' "$LOGDIR/compact-flat-bid.out" || { echo 'flat-bid metadata gate failed' >&2; exit 5; }
fi
if [[ "$CPASYNC_PAIR" == 1 ]]; then
  ARCH="$ARCH" NGPU="$NGPU" THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/cpasync-peer.out" 2>"$LOGDIR/cpasync-peer.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync-peer.out" || { echo 'cp.async peer gate failed' >&2; exit 5; }
fi

COMMON=(N="$N" ARCH="$ARCH" ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" DIRECTGATHER_SORT_RANKS="$SORT_RANKS" ORBITCTA_COL_ILP="$COL_ILP" PAIR_MLP="$PAIR_MLP" CPASYNC_PAIR="$CPASYNC_PAIR" QUAD_MLP=0 QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP=0 QUAD_OVERLAP_BYPASS_LOCAL0=0 QUAD_CPASYNC_PREFETCH_BYTES=0 QUAD_CPASYNC_GROUP_COLS=1 RANKFORMULA_MLP_WINDOW4="$WINDOW4" PM_ACCUM="$PM_ACCUM" PRECTX_FORWARD="$PRECTX_FORWARD" PRECTX_REVERSE="$PRECTX_REVERSE" PRECTX_COMPACT="$PRECTX_COMPACT" PRECTX_FLAT_BID="$PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED="$PRECTX_FLAT_BID_FUSED" PRECTX_WARPCOOP=0 PTXAS_VERBOSE=1)
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for dynamic in 0 1; do
  bin="$ONEESAN_BUILD_DIR/b300_flat_dynamic${dynamic}_bid${PRECTX_FLAT_BID}_n${N}"
  env "${COMMON[@]}" ORBITCTA_FLAT_DYNAMIC="$dynamic" OUT="$bin" bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/dynamic${dynamic}.build.out" 2>"$LOGDIR/dynamic${dynamic}.build.err"
  python3 "$PARSER" "$LOGDIR/dynamic${dynamic}.build.err" --label "dynamic${dynamic}" >>"$RESOURCE" || true
done

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }
printf 'dynamic\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_gpu_util_pct\tmax_memctrl_util_pct\tforward_regs\treverse_regs\tforward_blocks_per_sm\treverse_blocks_per_sm\tforward_flat_blocks\treverse_flat_blocks\tscheduler_mode\tflat_bid_mode\n' >"$RESULT"
for dynamic in 0 1; do
  bin="$ONEESAN_BUILD_DIR/b300_flat_dynamic${dynamic}_bid${PRECTX_FLAT_BID}_n${N}"
  for ((rep=1;rep<=REPEATS;++rep)); do
    so="$LOGDIR/dynamic${dynamic}_r${rep}.out"; se="$LOGDIR/dynamic${dynamic}_r${rep}.err"; util="$LOGDIR/dynamic${dynamic}_r${rep}.util"
    runenv=(BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY")
    [[ -z "$FLAT_PER_SM" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$FLAT_PER_SM")
    [[ -z "$FLAT_BLOCKS" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS="$FLAT_BLOCKS")
    env "${runenv[@]}" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
    pid=$!; sample "$pid" "$util" & sp=$!; set +e; wait "$pid"; rc=$?; set -e; wait "$sp" || true
    (( rc == 0 )) || { echo "dynamic=$dynamic failed rc=$rc" >&2; exit "$rc"; }
    line="$(grep '^residue=' "$so"|tail -n1||true)"; [[ -n "$line" ]] || exit 3
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "dynamic=$dynamic residue=$residue expected=$EXPECT" >&2; exit 4; }
    d="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1||true)"; g="$(grep 'rankformula_orbitcta_flat_grid device=0 ' "$se"|head -n1||true)"; o="$(grep 'rankformula_orbitcta_flat_occupancy device=0 ' "$se"|head -n1||true)"
    sched="$(field scheduler_mode "$g")"; bidmode="$(field flat_bid_mode "$g")"
    if [[ "$dynamic" == 0 ]]; then [[ "$sched" == static_cyclic ]] || { echo "static scheduler_mode=$sched" >&2; exit 6; }; else [[ "$sched" == dynamic_atomic_queue ]] || { echo "dynamic scheduler_mode=$sched" >&2; exit 6; }; fi
    expected_bid=binary_search; [[ "$PRECTX_FLAT_BID" == 1 ]] && expected_bid=compact_prectx
    [[ "$bidmode" == "$expected_bid" ]] || { echo "dynamic=$dynamic flat_bid_mode=$bidmode expected=$expected_bid" >&2; exit 6; }
    fh="$(field forward_high_s "$d")"; rh="$(field reverse_high_s "$d")"; [[ -n "$fh" && -n "$rh" ]] || exit 6
    high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
    read -r ag am mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$dynamic" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$mg" "$mm" "$(field forward_regs "$o")" "$(field reverse_regs "$o")" "$(field forward_blocks_per_sm "$o")" "$(field reverse_blocks_per_sm "$o")" "$(field forward_flat_blocks "$g")" "$(field reverse_flat_blocks "$g")" "$sched" "$bidmode" >>"$RESULT"
  done
done

python3 - "$RESULT" "$SUMMARY" "$WINNER_ENV" "$PRECTX_FLAT_BID" "$PRECTX_FLAT_BID_FUSED" <<'PY'
import csv,statistics,sys
src,summary,winner,bid,fused=sys.argv[1:]
rows=list(csv.DictReader(open(src),delimiter='\t')); z={}
for mode in ('0','1'):
 g=[r for r in rows if r['dynamic']==mode]
 z[mode]={k:statistics.median(float(r[k]) for r in g if r[k]!='NA') for k in ('wall_s','forward_high_s','reverse_high_s','high_s','avg_gpu_util_pct','avg_memctrl_util_pct')}
 z[mode]['fr']=int(g[0]['forward_regs'] or 0);z[mode]['rr']=int(g[0]['reverse_regs'] or 0);z[mode]['fb']=int(g[0]['forward_blocks_per_sm'] or 0);z[mode]['rb']=int(g[0]['reverse_blocks_per_sm'] or 0)
with open(summary,'w') as f:
 f.write('metric\tstatic\tdynamic\tspeedup_static_over_dynamic\n')
 for k in ('wall_s','forward_high_s','reverse_high_s','high_s'): f.write(f'{k}\t{z["0"][k]}\t{z["1"][k]}\t{z["0"][k]/z["1"][k]}\n')
w='1' if z['1']['wall_s'] < z['0']['wall_s'] else '0'
print(f"FLAT_DYNAMIC wall_speedup={z['0']['wall_s']/z['1']['wall_s']:.6f} high_speedup={z['0']['high_s']/z['1']['high_s']:.6f} gpu_util={z['0']['avg_gpu_util_pct']:.3f}->{z['1']['avg_gpu_util_pct']:.3f} memctl={z['0']['avg_memctrl_util_pct']:.3f}->{z['1']['avg_memctrl_util_pct']:.3f} regs={z['0']['fr']}/{z['0']['rr']}->{z['1']['fr']}/{z['1']['rr']} active={z['0']['fb']}/{z['0']['rb']}->{z['1']['fb']}/{z['1']['rb']} winner={w}")
with open(winner,'w') as f:
 f.write('ORBITCTA_FLAT=1\nORBITCTA_FLAT_CHUNK=1\n')
 f.write('ORBITCTA_FLAT_DYNAMIC='+w+'\n')
 f.write('PRECTX_FLAT_BID='+bid+'\nPRECTX_FLAT_BID_FUSED='+fused+'\n')
 f.write('QUAD_MLP=0\nQUAD_OVERLAP_LOCAL=0\nQUAD_LOCAL_DIRECT_MAX=0\nPRECTX_WARPCOOP=0\n')
PY
cat "$RESULT"
echo "flat dynamic A/B OK result=$RESULT summary=$SUMMARY ptxas=$RESOURCE winner_env=$WINNER_ENV" >&2
