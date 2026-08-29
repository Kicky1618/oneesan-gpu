#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROFILE_IN="${PROFILE_IN:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_adaptive21.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_pipe21.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_dynamic_pipe21}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
REPEATS="${REPEATS:-1}"; N=21; MOD=4294967291; EXPECT=998035516; NGPU="${NGPU:-8}"
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.15}"; PM_ACCUM="${PM_ACCUM:-1}"
[[ -f "$PROFILE_IN" ]] || { echo "missing PROFILE_IN=$PROFILE_IN" >&2; exit 2; }
# shellcheck disable=SC1090
source "$PROFILE_IN"
: "${ORBIT_COL_ILP:?profile missing ORBIT_COL_ILP}"
: "${ORBIT_SPARSE64:?profile missing ORBIT_SPARSE64}"
: "${ORBIT_CPASYNC_PAIR:?profile missing ORBIT_CPASYNC_PAIR}"
ORBIT_SPARSE64_WARP_INDEX="${ORBIT_SPARSE64_WARP_INDEX:-0}"
ORBITCTA_FLAT_DYNAMIC="${ORBITCTA_FLAT_DYNAMIC:-0}"
ORBITCTA_FLAT_DYNAMIC_BATCH="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"
ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="${ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP:-0}"
ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="${ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES:-0}"
ORBITCTA_FLAT_DYNAMIC_PIPE2="${ORBITCTA_FLAT_DYNAMIC_PIPE2:-0}"
ORBIT_PRECTX_FORWARD="${ORBIT_PRECTX_FORWARD:-0}"; ORBIT_PRECTX_REVERSE="${ORBIT_PRECTX_REVERSE:-0}"; ORBIT_PRECTX_COMPACT="${ORBIT_PRECTX_COMPACT:-0}"
ORBIT_PRECTX_FLAT_BID="${ORBIT_PRECTX_FLAT_BID:-0}"; ORBIT_PRECTX_FLAT_BID_FUSED="${ORBIT_PRECTX_FLAT_BID_FUSED:-0}"
[[ "$ORBITCTA_FLAT_DYNAMIC" == 1 ]] || { echo 'dynamic pipe2 refinement requires selected dynamic queue' >&2; exit 2; }
[[ "$ORBITCTA_FLAT_DYNAMIC_PIPE2" == 0 ]] || { echo 'dynamic pipe2 input must be the pre-pipe winner' >&2; exit 2; }
case "$ORBITCTA_FLAT_DYNAMIC_BATCH" in 1|2|4|8|16) ;; *) echo 'bad dynamic batch' >&2; exit 2;; esac
case "$ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES" in 0|1|2|4) ;; *) echo 'bad adaptive waves' >&2; exit 2;; esac
case "$ORBIT_COL_ILP" in 2|4) ;; *) echo 'dynamic pipe2 expects ORBIT_COL_ILP=2 or 4' >&2; exit 2;; esac
for x in ORBIT_SPARSE64 ORBIT_SPARSE64_WARP_INDEX ORBIT_CPASYNC_PAIR ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE ORBIT_PRECTX_COMPACT ORBIT_PRECTX_FLAT_BID ORBIT_PRECTX_FLAT_BID_FUSED PM_ACCUM; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
[[ "$ORBIT_SPARSE64_WARP_INDEX" == 0 || "$ORBIT_SPARSE64" == 1 ]] || { echo 'sparse64 warp-index requires sparse64' >&2; exit 2; }
[[ "$ORBIT_PRECTX_FLAT_BID_FUSED" == 0 || "$ORBIT_PRECTX_FLAT_BID" == 1 ]] || { echo 'fused flat-bid requires flat-bid' >&2; exit 2; }
if [[ "$ORBIT_PRECTX_FLAT_BID" == 1 ]]; then
  [[ "$ORBIT_PRECTX_FORWARD" == 1 && "$ORBIT_PRECTX_REVERSE" == 1 && "$ORBIT_PRECTX_COMPACT" == 1 ]] || { echo 'flat-bid requires compact forward+reverse prectx' >&2; exit 2; }
fi
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

if [[ "$ORBIT_PRECTX_COMPACT" == 1 ]]; then ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" >"$LOGDIR/prectx.out" 2>"$LOGDIR/prectx.err"; fi
if [[ "$ORBIT_PRECTX_FLAT_BID" == 1 ]]; then
  ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-flat-bid-selftest.sh" >"$LOGDIR/flat-bid.out" 2>"$LOGDIR/flat-bid.err"
  grep -q 'bucket-compact-flat-bid-selftest OK' "$LOGDIR/flat-bid.out" || { echo 'compact flat-bid metadata gate failed' >&2; exit 5; }
fi
if [[ "$ORBIT_CPASYNC_PAIR" == 1 ]]; then
  ARCH="$ARCH" NGPU="$NGPU" THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/cpasync.out" 2>"$LOGDIR/cpasync.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync.out" || { echo 'cp.async peer gate failed' >&2; exit 5; }
fi

COMMON=(N="$N" ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$ORBIT_SPARSE64" SPARSE64_WARP_INDEX="$ORBIT_SPARSE64_WARP_INDEX" DIRECTGATHER_SORT_RANKS=0 RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR="$ORBIT_CPASYNC_PAIR" ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_BATCH="$ORBITCTA_FLAT_DYNAMIC_BATCH" ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="$ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES" ORBITCTA_COL_ILP="$ORBIT_COL_ILP" QUAD_MLP=0 QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP=0 QUAD_OVERLAP_BYPASS_LOCAL0=0 QUAD_CPASYNC_PREFETCH_BYTES=0 QUAD_CPASYNC_GROUP_COLS=1 PRECTX_FORWARD="$ORBIT_PRECTX_FORWARD" PRECTX_REVERSE="$ORBIT_PRECTX_REVERSE" PRECTX_COMPACT="$ORBIT_PRECTX_COMPACT" PRECTX_FLAT_BID="$ORBIT_PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED="$ORBIT_PRECTX_FLAT_BID_FUSED" PRECTX_WARPCOOP=0 PTXAS_VERBOSE=1)
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
BASE_BIN="$ONEESAN_BUILD_DIR/b300_dynamic_pipe_base_b${ORBITCTA_FLAT_DYNAMIC_BATCH}_n21"
PIPE_BIN="$ONEESAN_BUILD_DIR/b300_dynamic_pipe2_b${ORBITCTA_FLAT_DYNAMIC_BATCH}_n21"
env "${COMMON[@]}" ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="$ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP" ORBITCTA_FLAT_DYNAMIC_PIPE2=0 OUT="$BASE_BIN" \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/base.build.out" 2>"$LOGDIR/base.build.err"
env "${COMMON[@]}" ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP=0 ORBITCTA_FLAT_DYNAMIC_PIPE2=1 OUT="$PIPE_BIN" \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/pipe2.build.out" 2>"$LOGDIR/pipe2.build.err"
grep -q "flat_dynamic=1 flat_dynamic_batch=$ORBITCTA_FLAT_DYNAMIC_BATCH flat_dynamic_fuse_lease_prep=$ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP flat_dynamic_adaptive_waves=$ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES flat_dynamic_pipe2=0" "$LOGDIR/base.build.err" || { echo 'base build marker mismatch' >&2; exit 6; }
grep -q "flat_dynamic=1 flat_dynamic_batch=$ORBITCTA_FLAT_DYNAMIC_BATCH flat_dynamic_fuse_lease_prep=0 flat_dynamic_adaptive_waves=$ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES flat_dynamic_pipe2=1" "$LOGDIR/pipe2.build.err" || { echo 'pipe2 build marker mismatch' >&2; exit 6; }
python3 "$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py" "$LOGDIR/base.build.err" --label base >>"$RESOURCE" || true
python3 "$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py" "$LOGDIR/pipe2.build.err" --label pipe2 >>"$RESOURCE" || true

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }
printf 'mode\tpipe2\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\tforward_regs\treverse_regs\tforward_blocks_per_sm\treverse_blocks_per_sm\tdynamic_smem_bytes\tscheduler_mode\tflat_dynamic_pipe2\n' >"$RESULT"
run_one(){
  local mode="$1" pipe="$2" bin="$3" rep="$4" expected_sched=dynamic_atomic_queue
  [[ "$pipe" == 0 ]] || expected_sched=dynamic_atomic_queue_pipe2
  local so="$LOGDIR/${mode}_r${rep}.out" se="$LOGDIR/${mode}_r${rep}.err" util="$LOGDIR/${mode}_r${rep}.util"
  env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  local pid=$!; sample "$pid" "$util" & local sp=$!; set +e; wait "$pid"; local rc=$?; set -e; wait "$sp" || true
  ((rc==0)) || { echo "$mode failed rc=$rc" >&2; exit "$rc"; }
  local line="$(grep '^residue=' "$so"|tail -n1||true)"; [[ -n "$line" ]] || { echo "$mode missing residue" >&2; exit 3; }
  local residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "$mode residue=$residue expected=$EXPECT" >&2; exit 4; }
  local d="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1||true)"
  local grid="$(grep 'rankformula_orbitcta_flat_grid device=0 ' "$se"|head -n1||true)"
  local occ="$(grep 'rankformula_orbitcta_flat_occupancy device=0 ' "$se"|head -n1||true)"
  [[ -n "$grid" && -n "$occ" ]] || { echo "$mode missing flat runtime metadata" >&2; exit 5; }
  local sched="$(field scheduler_mode "$grid")" seen_pipe="$(field flat_dynamic_pipe2 "$grid")"
  [[ "$sched" == "$expected_sched" && "$seen_pipe" == "$pipe" ]] || { echo "$mode scheduler=$sched pipe=$seen_pipe expected=$expected_sched/$pipe" >&2; exit 6; }
  local fh="$(field forward_high_s "$d")" rh="$(field reverse_high_s "$d")"; [[ -n "$fh" && -n "$rh" ]] || { echo "$mode missing HIGH timing" >&2; exit 5; }
  local high; high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  local ag am mm; read -r ag am mm < <(awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$mode" "$pipe" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$mm" \
    "$(field forward_regs "$occ")" "$(field reverse_regs "$occ")" "$(field forward_blocks_per_sm "$occ")" "$(field reverse_blocks_per_sm "$occ")" "$(field dynamic_smem_bytes "$occ")" "$sched" "$seen_pipe" >>"$RESULT"
}
for ((r=1;r<=REPEATS;++r)); do run_one base 0 "$BASE_BIN" "$r"; run_one pipe2 1 "$PIPE_BIN" "$r"; done
cat "$RESULT"

python3 - "$PROFILE_IN" "$RESULT" "$PROFILE_OUT" "$ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP" <<'PY'
import csv,statistics,sys,re
pin,result,pout,base_fuse=sys.argv[1:]; rows=list(csv.DictReader(open(result),delimiter='\t')); z={}
for mode in ('base','pipe2'):
 g=[r for r in rows if r['mode']==mode]; mc=[float(r['avg_memctrl_util_pct']) for r in g if r['avg_memctrl_util_pct']!='NA']
 z[mode]={
  'wall':statistics.median(float(r['wall_s']) for r in g),'high':statistics.median(float(r['high_s']) for r in g),
  'mc':statistics.median(mc) if mc else float('nan'),'fr':int(g[0]['forward_regs'] or 0),'rr':int(g[0]['reverse_regs'] or 0),
  'fb':int(g[0]['forward_blocks_per_sm'] or 0),'rb':int(g[0]['reverse_blocks_per_sm'] or 0),'smem':int(g[0]['dynamic_smem_bytes'] or 0)}
best='pipe2' if z['pipe2']['wall'] < z['base']['wall'] else 'base'; pipe='1' if best=='pipe2' else '0'; fuse='0' if best=='pipe2' else base_fuse
for m in ('base','pipe2'):
 print('DYNAMIC_PIPE2',m,f"wall_s={z[m]['wall']:.6f}",f"high_s={z[m]['high']:.6f}",f"mc_avg_pct={z[m]['mc']:.3f}",f"regs={z[m]['fr']}/{z[m]['rr']}",f"active_blocks_sm={z[m]['fb']}/{z[m]['rb']}",f"smem={z[m]['smem']}")
updates={
 'ORBITCTA_FLAT_DYNAMIC_PIPE2':pipe,'ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP':fuse,
 'ORBIT_DYNAMIC_PIPE2_PROFILE':best,'ORBIT_DYNAMIC_PIPE2_WALL_S':f'{z[best]["wall"]:.9f}','ORBIT_DYNAMIC_PIPE2_HIGH_S':f'{z[best]["high"]:.9f}' }
out=[]; seen=set()
for line in open(pin):
 s=line.rstrip('\n')
 if '=' not in s or s.lstrip().startswith('#'): out.append(s); continue
 k,v=s.split('=',1)
 if k=='ORBIT_PROFILE':
  v=re.sub(r'_dp2$','',v); out.append('ORBIT_PROFILE='+v+('_dp2' if pipe=='1' else '')); continue
 if k in updates:
  if k not in seen: out.append(k+'='+updates[k]); seen.add(k)
  continue
 out.append(s)
for k in ('ORBITCTA_FLAT_DYNAMIC_PIPE2','ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP','ORBIT_DYNAMIC_PIPE2_PROFILE','ORBIT_DYNAMIC_PIPE2_WALL_S','ORBIT_DYNAMIC_PIPE2_HIGH_S'):
 if k not in seen: out.append(k+'='+updates[k])
open(pout,'w').write('\n'.join(out)+'\n')
print('BEST_DYNAMIC_PIPE2='+best,f"speedup={z['base']['wall']/z['pipe2']['wall']:.6f}",f'profile_file={pout}')
PY
cat "$PROFILE_OUT"
echo "dynamic pipe2 refinement OK input=$PROFILE_IN output=$PROFILE_OUT resources=$RESOURCE" >&2
