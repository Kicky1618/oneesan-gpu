#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
COL_ILP="${ORBITCTA_COL_ILP:-2}"; CPASYNC_PAIR="${CPASYNC_PAIR:-0}"; PM_ACCUM="${PM_ACCUM:-1}"
DYNAMIC="${ORBITCTA_FLAT_DYNAMIC:-1}"; DYNAMIC_BATCH="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"
DYNAMIC_FUSE_PREP="${ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP:-1}"; DYNAMIC_ADAPTIVE_WAVES="${ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES:-0}"
PRECTX_FLAT_BID="${PRECTX_FLAT_BID:-1}"; PRECTX_FLAT_BID_FUSED="${PRECTX_FLAT_BID_FUSED:-0}"
FLAT_PER_SM="${BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM:-}"; FLAT_BLOCKS="${BUCKET_ORBITCTA_FLAT_BLOCKS:-}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_sparse64_warp_index_ab_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

[[ "$N" == 21 && "$MOD" == 4294967291 && "$EXPECT" == 998035516 ]] || { echo 'fixed exact gate is n21/mod4294967291/residue998035516' >&2; exit 2; }
for x in CPASYNC_PAIR PM_ACCUM DYNAMIC DYNAMIC_FUSE_PREP PRECTX_FLAT_BID PRECTX_FLAT_BID_FUSED; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
case "$COL_ILP" in 2|4) ;; *) echo 'ORBITCTA_COL_ILP must be 2 or 4' >&2; exit 2;; esac
case "$DYNAMIC_BATCH" in 1|2|4|8|16) ;; *) echo 'dynamic batch must be 1,2,4,8,16' >&2; exit 2;; esac
case "$DYNAMIC_ADAPTIVE_WAVES" in 0|1|2|4) ;; *) echo 'adaptive waves must be 0,1,2,4' >&2; exit 2;; esac
if [[ "$DYNAMIC" == 0 ]]; then
  [[ "$DYNAMIC_BATCH" == 1 && "$DYNAMIC_FUSE_PREP" == 0 && "$DYNAMIC_ADAPTIVE_WAVES" == 0 ]] || { echo 'static scheduler requires batch=1 fuse_prep=0 adaptive_waves=0' >&2; exit 2; }
fi
if (( DYNAMIC_ADAPTIVE_WAVES != 0 )); then (( DYNAMIC_BATCH > 1 )) || { echo 'adaptive waves require batch>1' >&2; exit 2; }; fi
[[ "$PRECTX_FLAT_BID_FUSED" == 0 || "$PRECTX_FLAT_BID" == 1 ]] || { echo 'fused prectx load requires flat-bid' >&2; exit 2; }
[[ -z "$FLAT_PER_SM" || -z "$FLAT_BLOCKS" ]] || { echo 'set at most one flat pool override' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

if [[ "$PRECTX_FLAT_BID" == 1 ]]; then
  ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" >"$LOGDIR/compact-prectx.out" 2>"$LOGDIR/compact-prectx.err"
  ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-flat-bid-selftest.sh" >"$LOGDIR/compact-flat-bid.out" 2>"$LOGDIR/compact-flat-bid.err"
  grep -q 'bucket-compact-flat-bid-selftest OK' "$LOGDIR/compact-flat-bid.out" || { echo 'compact flat-bid gate failed' >&2; exit 5; }
fi
if [[ "$CPASYNC_PAIR" == 1 ]]; then
  ARCH="$ARCH" NGPU="$NGPU" THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/cpasync.out" 2>"$LOGDIR/cpasync.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync.out" || { echo 'cp.async peer gate failed' >&2; exit 5; }
fi

PRE_F=0; PRE_R=0; PRE_C=0
if [[ "$PRECTX_FLAT_BID" == 1 ]]; then PRE_F=1; PRE_R=1; PRE_C=1; fi
COMMON=(N="$N" ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64=1 DIRECTGATHER_SORT_RANKS=0 RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR="$CPASYNC_PAIR" ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC="$DYNAMIC" ORBITCTA_FLAT_DYNAMIC_BATCH="$DYNAMIC_BATCH" ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="$DYNAMIC_FUSE_PREP" ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="$DYNAMIC_ADAPTIVE_WAVES" ORBITCTA_COL_ILP="$COL_ILP" QUAD_MLP=0 QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP=0 QUAD_OVERLAP_BYPASS_LOCAL0=0 QUAD_CPASYNC_PREFETCH_BYTES=0 QUAD_CPASYNC_GROUP_COLS=1 PRECTX_FORWARD="$PRE_F" PRECTX_REVERSE="$PRE_R" PRECTX_COMPACT="$PRE_C" PRECTX_FLAT_BID="$PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED="$PRECTX_FLAT_BID_FUSED" PRECTX_WARPCOOP=0 PTXAS_VERBOSE=1)
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for wi in 0 1; do
  bin="$ONEESAN_BUILD_DIR/b300_sparse64_warpindex${wi}_n${N}"
  env "${COMMON[@]}" SPARSE64_WARP_INDEX="$wi" OUT="$bin" bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/wi${wi}.build.out" 2>"$LOGDIR/wi${wi}.build.err"
  grep -q "sparse64=1 sparse64_warp_index=$wi" "$LOGDIR/wi${wi}.build.err" || { echo "build marker mismatch wi=$wi" >&2; exit 6; }
  python3 "$PARSER" "$LOGDIR/wi${wi}.build.err" --label "warpindex${wi}" >>"$RESOURCE" || true
done

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }
printf 'warp_index\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\tforward_regs\treverse_regs\tforward_blocks_per_sm\treverse_blocks_per_sm\tforward_flat_blocks\treverse_flat_blocks\tscheduler_mode\n' >"$RESULT"
for wi in 0 1; do
  bin="$ONEESAN_BUILD_DIR/b300_sparse64_warpindex${wi}_n${N}"
  for ((rep=1;rep<=REPEATS;++rep)); do
    so="$LOGDIR/wi${wi}_r${rep}.out"; se="$LOGDIR/wi${wi}_r${rep}.err"; util="$LOGDIR/wi${wi}_r${rep}.util"
    runenv=(BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY")
    [[ -z "$FLAT_PER_SM" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$FLAT_PER_SM")
    [[ -z "$FLAT_BLOCKS" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS="$FLAT_BLOCKS")
    env "${runenv[@]}" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
    pid=$!; sample "$pid" "$util" & sp=$!; set +e; wait "$pid"; rc=$?; set -e; wait "$sp" || true
    (( rc == 0 )) || { echo "wi=$wi failed rc=$rc" >&2; exit "$rc"; }
    line="$(grep '^residue=' "$so"|tail -n1||true)"; [[ -n "$line" ]] || { echo "wi=$wi missing residue" >&2; exit 3; }
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "wi=$wi residue=$residue expected=$EXPECT" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1||true)"; grid="$(grep 'rankformula_orbitcta_flat_grid device=0 ' "$se"|head -n1||true)"; occ="$(grep 'rankformula_orbitcta_flat_occupancy device=0 ' "$se"|head -n1||true)"
    [[ -n "$grid" && -n "$occ" ]] || { echo "wi=$wi missing runtime metadata" >&2; exit 5; }
    expected_sched=static_cyclic; [[ "$DYNAMIC" == 1 ]] && expected_sched=dynamic_atomic_queue
    [[ "$(field scheduler_mode "$grid")" == "$expected_sched" ]] || { echo "wi=$wi scheduler mismatch" >&2; exit 6; }
    fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; [[ -n "$fh" && -n "$rh" ]] || { echo "wi=$wi missing HIGH timing" >&2; exit 5; }
    high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
    read -r ag am mm < <(awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' "$util")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$wi" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$mm" "$(field forward_regs "$occ")" "$(field reverse_regs "$occ")" "$(field forward_blocks_per_sm "$occ")" "$(field reverse_blocks_per_sm "$occ")" "$(field forward_flat_blocks "$grid")" "$(field reverse_flat_blocks "$grid")" "$(field scheduler_mode "$grid")" >>"$RESULT"
  done
done

cat "$RESULT"
python3 - "$RESULT" "$WINNER_ENV" <<'PY'
import csv,statistics,sys
src,winner=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); z={}
for wi in ('0','1'):
 g=[r for r in rows if r['warp_index']==wi]
 z[wi]={k:statistics.median(float(r[k]) for r in g) for k in ('wall_s','forward_high_s','reverse_high_s','high_s')}
 z[wi]['fr']=int(g[0]['forward_regs'] or 0); z[wi]['rr']=int(g[0]['reverse_regs'] or 0); z[wi]['fb']=int(g[0]['forward_blocks_per_sm'] or 0); z[wi]['rb']=int(g[0]['reverse_blocks_per_sm'] or 0)
w='1' if z['1']['wall_s'] < z['0']['wall_s'] else '0'
print(f"SPARSE64_WARP_INDEX wall_speedup={z['0']['wall_s']/z['1']['wall_s']:.6f} high_speedup={z['0']['high_s']/z['1']['high_s']:.6f} regs0={z['0']['fr']}/{z['0']['rr']} regs1={z['1']['fr']}/{z['1']['rr']} active0={z['0']['fb']}/{z['0']['rb']} active1={z['1']['fb']}/{z['1']['rb']} winner={w}")
with open(winner,'w') as o:
 o.write(f'SPARSE64_WARP_INDEX={w}\n')
 o.write(f'SPARSE64_WARP_INDEX_WALL_S={z[w]["wall_s"]:.9f}\nSPARSE64_WARP_INDEX_HIGH_S={z[w]["high_s"]:.9f}\n')
PY

echo "sparse64 warp-index A/B OK result=$RESULT resources=$RESOURCE winner_env=$WINNER_ENV" >&2
