#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"
SPARSE64="${DIRECTGATHER_SPARSE64:-1}"; COL_ILP="${ORBITCTA_COL_ILP:-2}"
CPASYNC_PAIR="${CPASYNC_PAIR:-0}"; DYNAMIC_BATCH="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"; PM_ACCUM="${PM_ACCUM:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_orbitcta_flat_dynamic_bid_n${N}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"; WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

[[ "$N" == 21 && "$MOD" == 4294967291 && "$EXPECT" == 998035516 ]] || { echo 'fixed exact gate is n21/mod4294967291/residue998035516' >&2; exit 2; }
for x in SPARSE64 CPASYNC_PAIR PM_ACCUM; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done
case "$COL_ILP" in 2|4) ;; *) echo 'dynamic bid A/B expects ORBITCTA_COL_ILP=2 or 4' >&2; exit 2;; esac
case "$DYNAMIC_BATCH" in 1|2|4|8|16) ;; *) echo 'ORBITCTA_FLAT_DYNAMIC_BATCH must be 1,2,4,8,16' >&2; exit 2;; esac
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }

ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" >"$LOGDIR/compact-prectx.out" 2>"$LOGDIR/compact-prectx.err"
ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-flat-bid-selftest.sh" >"$LOGDIR/compact-flat-bid.out" 2>"$LOGDIR/compact-flat-bid.err"
grep -q 'bucket-compact-flat-bid-selftest OK' "$LOGDIR/compact-flat-bid.out" || { echo 'compact flat-bid metadata gate failed' >&2; exit 5; }
if [[ "$CPASYNC_PAIR" == 1 ]]; then
  ARCH="$ARCH" NGPU="$NGPU" THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/cpasync.out" 2>"$LOGDIR/cpasync.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync.out" || { echo 'cp.async peer gate failed' >&2; exit 5; }
fi

COMMON=(N="$N" ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" DIRECTGATHER_SORT_RANKS=0 RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR="$CPASYNC_PAIR" ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_BATCH="$DYNAMIC_BATCH" ORBITCTA_COL_ILP="$COL_ILP" QUAD_MLP=0 QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP=0 QUAD_OVERLAP_BYPASS_LOCAL0=0 QUAD_CPASYNC_PREFETCH_BYTES=0 QUAD_CPASYNC_GROUP_COLS=1 PRECTX_FORWARD=1 PRECTX_REVERSE=1 PRECTX_COMPACT=1 PRECTX_WARPCOOP=0 PTXAS_VERBOSE=1)
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for spec in 'base|0|0' 'bid|1|0' 'fused|1|1'; do
  IFS='|' read -r mode bid fused <<<"$spec"
  bin="$ONEESAN_BUILD_DIR/b300_dynamic_bid_${mode}_b${DYNAMIC_BATCH}_n${N}"
  env "${COMMON[@]}" PRECTX_FLAT_BID="$bid" PRECTX_FLAT_BID_FUSED="$fused" OUT="$bin" \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/${mode}.build.out" 2>"$LOGDIR/${mode}.build.err"
  grep -q "flat_dynamic=1 flat_dynamic_batch=$DYNAMIC_BATCH" "$LOGDIR/${mode}.build.err" || { echo "$mode dynamic build marker mismatch" >&2; exit 6; }
  grep -q "prectx_flat_bid=$bid prectx_flat_bid_fused=$fused" "$LOGDIR/${mode}.build.err" || { echo "$mode bid build marker mismatch" >&2; exit 6; }
  python3 "$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py" "$LOGDIR/${mode}.build.err" --label "$mode" >>"$RESOURCE" || true
done

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }
printf 'mode\tprectx_flat_bid\tprectx_flat_bid_fused\tdynamic_batch\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\tflat_bid_mode\tscheduler_mode\n' >"$RESULT"
run_one(){
  local mode="$1" bid="$2" fused="$3" rep="$4"
  local bin="$ONEESAN_BUILD_DIR/b300_dynamic_bid_${mode}_b${DYNAMIC_BATCH}_n${N}"
  local so="$LOGDIR/${mode}_r${rep}.out" se="$LOGDIR/${mode}_r${rep}.err" util="$LOGDIR/${mode}_r${rep}.util"
  env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM \
    BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  local pid=$!; sample "$pid" "$util" & local sp=$!; set +e; wait "$pid"; local rc=$?; set -e; wait "$sp" || true
  (( rc == 0 )) || { echo "$mode failed rc=$rc" >&2; exit "$rc"; }
  local line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 3; }
  local residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue=$residue expected=$EXPECT" >&2; exit 4; }
  local detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"
  local grid="$(grep 'rankformula_orbitcta_flat_grid device=0 ' "$se" | head -n1 || true)"
  local fh="$(field forward_high_s "$detail")" rh="$(field reverse_high_s "$detail")"
  [[ -n "$fh" && -n "$rh" ]] || { echo "$mode missing HIGH timing" >&2; exit 5; }
  local high; high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  local sched="$(field scheduler_mode "$grid")" bidmode="$(field flat_bid_mode "$grid")"
  [[ "$sched" == dynamic_atomic_queue ]] || { echo "$mode scheduler_mode=$sched expected=dynamic_atomic_queue" >&2; exit 6; }
  local expected=binary_search; [[ "$bid" == 1 ]] && expected=compact_prectx
  [[ "$bidmode" == "$expected" ]] || { echo "$mode flat_bid_mode=$bidmode expected=$expected" >&2; exit 6; }
  local ag am mm; read -r ag am mm < <(awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$mode" "$bid" "$fused" "$DYNAMIC_BATCH" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$mm" "$bidmode" "$sched" >>"$RESULT"
}
for ((r=1;r<=REPEATS;++r)); do
  run_one base 0 0 "$r"
  run_one bid 1 0 "$r"
  run_one fused 1 1 "$r"
done

cat "$RESULT"
python3 - "$RESULT" "$WINNER_ENV" <<'PY'
import csv,statistics,sys
src,winner=sys.argv[1:]; rows=list(csv.DictReader(open(src),delimiter='\t')); z={}
for mode in ('base','bid','fused'):
 g=[r for r in rows if r['mode']==mode]
 z[mode]={
  'wall':statistics.median(float(r['wall_s']) for r in g),
  'high':statistics.median(float(r['high_s']) for r in g),
  'fh':statistics.median(float(r['forward_high_s']) for r in g),
  'rh':statistics.median(float(r['reverse_high_s']) for r in g),
  'mc':statistics.median(float(r['avg_memctrl_util_pct']) for r in g if r['avg_memctrl_util_pct']!='NA'),
  'bid':int(g[0]['prectx_flat_bid']),'fused':int(g[0]['prectx_flat_bid_fused']),'batch':int(g[0]['dynamic_batch'])}
best=min(z,key=lambda m:z[m]['wall'])
for m in ('base','bid','fused'):
 print('DYNAMIC_BID',m,f"wall_s={z[m]['wall']:.6f}",f"high_s={z[m]['high']:.6f}",f"mc_avg_pct={z[m]['mc']:.3f}")
with open(winner,'w') as f:
 f.write('ORBITCTA_FLAT=1\nORBITCTA_FLAT_CHUNK=1\nORBITCTA_FLAT_DYNAMIC=1\n')
 f.write(f'ORBITCTA_FLAT_DYNAMIC_BATCH={z[best]["batch"]}\n')
 f.write('ORBIT_PRECTX_FORWARD=1\nORBIT_PRECTX_REVERSE=1\nORBIT_PRECTX_COMPACT=1\n')
 f.write(f'ORBIT_PRECTX_FLAT_BID={z[best]["bid"]}\nORBIT_PRECTX_FLAT_BID_FUSED={z[best]["fused"]}\n')
 f.write('ORBIT_PRECTX_WARPCOOP=0\nORBIT_QUAD_MLP=0\nORBIT_QUAD_OVERLAP_LOCAL=0\nORBIT_QUAD_LOCAL_DIRECT_MAX=0\n')
 f.write(f'ORBIT_DYNAMIC_BID_PROFILE={best}\nORBIT_DYNAMIC_BID_WALL_S={z[best]["wall"]:.9f}\nORBIT_DYNAMIC_BID_HIGH_S={z[best]["high"]:.9f}\n')
print('BEST_DYNAMIC_BID='+best,f"wall_s={z[best]['wall']:.6f}",f"high_s={z[best]['high']:.6f}",f'profile_file={winner}')
PY

echo "dynamic flat-bid A/B OK batch=$DYNAMIC_BATCH result=$RESULT resources=$RESOURCE winner_env=$WINNER_ENV" >&2
