#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-21}"; MOD="${MOD:-4294967291}"; EXPECT="${EXPECT:-998035516}"; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"; BATCH="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"
SPARSE64="${DIRECTGATHER_SPARSE64:-1}"; COL_ILP="${ORBITCTA_COL_ILP:-2}"; CPASYNC_PAIR="${CPASYNC_PAIR:-0}"
PM_ACCUM="${PM_ACCUM:-1}"; PRECTX_FLAT_BID_FUSED="${PRECTX_FLAT_BID_FUSED:-0}"
FLAT_PER_SM="${BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM:-}"; FLAT_BLOCKS="${BUCKET_ORBITCTA_FLAT_BLOCKS:-}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_dynamic_pipe2_ab_n${N}_b${BATCH}}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

[[ "$N" == 21 && "$MOD" == 4294967291 && "$EXPECT" == 998035516 ]] || {
  echo 'fixed exact gate is n21/mod4294967291/residue998035516' >&2; exit 2;
}
for x in SPARSE64 CPASYNC_PAIR PM_ACCUM PRECTX_FLAT_BID_FUSED; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
case "$BATCH" in 1|2|4|8|16) ;; *) echo 'ORBITCTA_FLAT_DYNAMIC_BATCH must be 1,2,4,8,16' >&2; exit 2;; esac
case "$COL_ILP" in 2|4) ;; *) echo 'ORBITCTA_COL_ILP must be 2 or 4' >&2; exit 2;; esac
[[ -z "$FLAT_PER_SM" || -z "$FLAT_BLOCKS" ]] || { echo 'set at most one flat pool override' >&2; exit 2; }
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

COMMON=(N="$N" ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$SPARSE64" DIRECTGATHER_SORT_RANKS=0
  RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR="$CPASYNC_PAIR"
  ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_BATCH="$BATCH" ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES=0
  ORBITCTA_COL_ILP="$COL_ILP" QUAD_MLP=0 QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP=0
  QUAD_OVERLAP_BYPASS_LOCAL0=0 QUAD_CPASYNC_PREFETCH_BYTES=0 QUAD_CPASYNC_GROUP_COLS=1
  PRECTX_FORWARD=1 PRECTX_REVERSE=1 PRECTX_COMPACT=1 PRECTX_FLAT_BID=1 PRECTX_FLAT_BID_FUSED="$PRECTX_FLAT_BID_FUSED"
  PRECTX_WARPCOOP=0 PTXAS_VERBOSE=1)

printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"

fused_bin="$ONEESAN_BUILD_DIR/b300_dynamic_fused_b${BATCH}_n${N}"
env "${COMMON[@]}" ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=1 OUT="$fused_bin" \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/fused.build.out" 2>"$LOGDIR/fused.build.err"
grep -q "flat_dynamic=1 flat_dynamic_batch=$BATCH flat_dynamic_fuse_lease_prep=1" "$LOGDIR/fused.build.err" || { echo 'fused build marker mismatch' >&2; exit 6; }
python3 "$PARSER" "$LOGDIR/fused.build.err" --label fused >>"$RESOURCE" || true

pipe_bin="$ONEESAN_BUILD_DIR/b300_dynamic_pipe2_b${BATCH}_n${N}"
env "${COMMON[@]}" ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 OUT="$pipe_bin" \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2.sh" >"$LOGDIR/pipe2.build.out" 2>"$LOGDIR/pipe2.build.err"
grep -q 'dynamic_pipe2=1' "$LOGDIR/pipe2.build.err" || { echo 'pipe2 build marker missing' >&2; exit 6; }
python3 "$PARSER" "$LOGDIR/pipe2.build.err" --label pipe2 >>"$RESOURCE" || true

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do
  nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null |
    awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(g>mg)mg=g;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' >>"$out" || true
  sleep "$SAMPLE_INTERVAL"
done; }

printf 'variant\tbatch\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_gpu_util_pct\tmax_memctrl_util_pct\tforward_regs\treverse_regs\tforward_blocks_per_sm\treverse_blocks_per_sm\tlaunch_smem_bytes\tscheduler_mode\tsteady_state_orbit_barriers\n' >"$RESULT"

run_one(){
  local name="$1" bin="$2" rep="$3" so="$LOGDIR/${name}_r${rep}.out" se="$LOGDIR/${name}_r${rep}.err" util="$LOGDIR/${name}_r${rep}.util"
  runenv=(BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY")
  [[ -z "$FLAT_PER_SM" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$FLAT_PER_SM")
  [[ -z "$FLAT_BLOCKS" ]] || runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS="$FLAT_BLOCKS")
  env "${runenv[@]}" "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  pid=$!; sample "$pid" "$util" & sp=$!; set +e; wait "$pid"; rc=$?; set -e; wait "$sp" || true
  (( rc == 0 )) || { echo "$name failed rc=$rc" >&2; exit "$rc"; }
  line="$(grep '^residue=' "$so"|tail -n1||true)"; [[ -n "$line" ]] || { echo "$name missing residue" >&2; exit 3; }
  residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$name residue=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1||true)"
  grid="$(grep 'rankformula_orbitcta_flat_grid device=0 ' "$se"|head -n1||true)"
  occ="$(grep 'rankformula_orbitcta_flat_occupancy device=0 ' "$se"|head -n1||true)"
  [[ -n "$grid" && -n "$occ" ]] || { echo "$name missing runtime metadata" >&2; exit 5; }
  if [[ "$name" == pipe2 ]]; then
    [[ "$(field scheduler_mode "$grid")" == dynamic_atomic_queue_pipe2 ]] || { echo 'pipe2 scheduler mismatch' >&2; exit 6; }
    [[ "$(field flat_dynamic_pipe2 "$grid")" == 1 ]] || { echo 'pipe2 runtime marker mismatch' >&2; exit 6; }
  else
    [[ "$(field scheduler_mode "$grid")" == dynamic_atomic_queue ]] || { echo 'fused scheduler mismatch' >&2; exit 6; }
  fi
  fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"; [[ -n "$fh" && -n "$rh" ]] || { echo "$name missing HIGH timing" >&2; exit 5; }
  high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  read -r ag am mg mm < <(awk '{sg+=$1;sm+=$2;if($3>mg)mg=$3;if($4>mm)mm=$4;n++}END{if(n)printf "%.6f %.6f %d %d\n",sg/n,sm/n,mg,mm;else print "NA NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$BATCH" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$mg" "$mm" \
    "$(field forward_regs "$occ")" "$(field reverse_regs "$occ")" "$(field forward_blocks_per_sm "$occ")" "$(field reverse_blocks_per_sm "$occ")" \
    "$(field launch_smem_bytes "$grid")" "$(field scheduler_mode "$grid")" "$(field steady_state_orbit_barriers "$grid")" >>"$RESULT"
}

for ((r=1;r<=REPEATS;++r)); do run_one fused "$fused_bin" "$r"; done
for ((r=1;r<=REPEATS;++r)); do run_one pipe2 "$pipe_bin" "$r"; done
cat "$RESULT"

python3 - "$RESULT" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
def med(v,k): return statistics.median(float(r[k]) for r in rows if r['variant']==v and r[k]!='NA')
for v in ('fused','pipe2'):
    print(f'{v}_wall_median={med(v,"wall_s"):.9f} {v}_high_median={med(v,"high_s"):.9f} {v}_memctrl_median={med(v,"avg_memctrl_util_pct"):.6f}')
print(f'pipe2_wall_speedup={med("fused","wall_s")/med("pipe2","wall_s"):.6f}x')
print(f'pipe2_high_speedup={med("fused","high_s")/med("pipe2","high_s"):.6f}x')
print(f'pipe2_memctrl_delta={med("pipe2","avg_memctrl_util_pct")-med("fused","avg_memctrl_util_pct"):.6f}pp')
PY

echo "dynamic pipe2 A/B OK result=$RESULT resources=$RESOURCE batch=$BATCH" >&2
