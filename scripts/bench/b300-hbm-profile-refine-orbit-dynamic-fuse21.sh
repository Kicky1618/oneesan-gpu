#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROFILE_IN="${PROFILE_IN:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_bid21.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_fuse21.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_dynamic_fuse21}"
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
ORBITCTA_FLAT_DYNAMIC="${ORBITCTA_FLAT_DYNAMIC:-0}"
ORBITCTA_FLAT_DYNAMIC_BATCH="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"
ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="${ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP:-0}"
ORBIT_PRECTX_FORWARD="${ORBIT_PRECTX_FORWARD:-0}"; ORBIT_PRECTX_REVERSE="${ORBIT_PRECTX_REVERSE:-0}"; ORBIT_PRECTX_COMPACT="${ORBIT_PRECTX_COMPACT:-0}"
ORBIT_PRECTX_FLAT_BID="${ORBIT_PRECTX_FLAT_BID:-0}"; ORBIT_PRECTX_FLAT_BID_FUSED="${ORBIT_PRECTX_FLAT_BID_FUSED:-0}"
[[ "$ORBITCTA_FLAT_DYNAMIC" == 1 ]] || { echo 'dynamic fuse refinement requires selected dynamic queue' >&2; exit 2; }
case "$ORBITCTA_FLAT_DYNAMIC_BATCH" in 1|2|4|8|16) ;; *) echo 'bad dynamic batch' >&2; exit 2;; esac
case "$ORBIT_COL_ILP" in 2|4) ;; *) echo 'dynamic fuse expects ORBIT_COL_ILP=2 or 4' >&2; exit 2;; esac
for x in ORBIT_SPARSE64 ORBIT_CPASYNC_PAIR ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE ORBIT_PRECTX_COMPACT ORBIT_PRECTX_FLAT_BID ORBIT_PRECTX_FLAT_BID_FUSED PM_ACCUM; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
[[ "$ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP" == 0 ]] || { echo 'dynamic fuse input must be the unfused dynamic winner' >&2; exit 2; }
[[ "$ORBIT_PRECTX_FLAT_BID_FUSED" == 0 || "$ORBIT_PRECTX_FLAT_BID" == 1 ]] || { echo 'fused flat-bid requires flat-bid' >&2; exit 2; }
[[ "$ORBIT_PRECTX_COMPACT" == 0 || "$ORBIT_PRECTX_FORWARD" == 1 || "$ORBIT_PRECTX_REVERSE" == 1 ]] || { echo 'compact prectx requires forward and/or reverse prectx' >&2; exit 2; }
if [[ "$ORBIT_PRECTX_FLAT_BID" == 1 ]]; then
  [[ "$ORBIT_PRECTX_FORWARD" == 1 && "$ORBIT_PRECTX_REVERSE" == 1 && "$ORBIT_PRECTX_COMPACT" == 1 ]] || { echo 'flat-bid requires compact forward+reverse prectx' >&2; exit 2; }
fi
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

if [[ "$ORBIT_PRECTX_COMPACT" == 1 ]]; then
  ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" >"$LOGDIR/prectx.out" 2>"$LOGDIR/prectx.err"
fi
if [[ "$ORBIT_PRECTX_FLAT_BID" == 1 ]]; then
  ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-flat-bid-selftest.sh" >"$LOGDIR/flat-bid.out" 2>"$LOGDIR/flat-bid.err"
  grep -q 'bucket-compact-flat-bid-selftest OK' "$LOGDIR/flat-bid.out" || { echo 'compact flat-bid metadata gate failed' >&2; exit 5; }
fi
if [[ "$ORBIT_CPASYNC_PAIR" == 1 ]]; then
  ARCH="$ARCH" NGPU="$NGPU" THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/cpasync.out" 2>"$LOGDIR/cpasync.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync.out" || { echo 'cp.async peer gate failed' >&2; exit 5; }
fi

COMMON=(N="$N" ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$ORBIT_SPARSE64" DIRECTGATHER_SORT_RANKS=0 RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR="$ORBIT_CPASYNC_PAIR" ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_BATCH="$ORBITCTA_FLAT_DYNAMIC_BATCH" ORBITCTA_COL_ILP="$ORBIT_COL_ILP" QUAD_MLP=0 QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP=0 QUAD_OVERLAP_BYPASS_LOCAL0=0 QUAD_CPASYNC_PREFETCH_BYTES=0 QUAD_CPASYNC_GROUP_COLS=1 PRECTX_FORWARD="$ORBIT_PRECTX_FORWARD" PRECTX_REVERSE="$ORBIT_PRECTX_REVERSE" PRECTX_COMPACT="$ORBIT_PRECTX_COMPACT" PRECTX_FLAT_BID="$ORBIT_PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED="$ORBIT_PRECTX_FLAT_BID_FUSED" PRECTX_WARPCOOP=0 PTXAS_VERBOSE=1)
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for fuse in 0 1; do
  bin="$ONEESAN_BUILD_DIR/b300_dynamic_fuse${fuse}_b${ORBITCTA_FLAT_DYNAMIC_BATCH}_n21"
  env "${COMMON[@]}" ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="$fuse" OUT="$bin" \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/fuse${fuse}.build.out" 2>"$LOGDIR/fuse${fuse}.build.err"
  grep -q "flat_dynamic=1 flat_dynamic_batch=$ORBITCTA_FLAT_DYNAMIC_BATCH flat_dynamic_fuse_lease_prep=$fuse" "$LOGDIR/fuse${fuse}.build.err" || { echo "fuse=$fuse build marker mismatch" >&2; exit 6; }
  python3 "$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py" "$LOGDIR/fuse${fuse}.build.err" --label "fuse${fuse}" >>"$RESOURCE" || true
done

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }
printf 'fuse\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\tforward_regs\treverse_regs\tforward_blocks_per_sm\treverse_blocks_per_sm\tscheduler_mode\tqueue_lease_batch\tflat_bid_mode\tflat_bid_fused\n' >"$RESULT"
run_one(){
  local fuse="$1" rep="$2" bin="$ONEESAN_BUILD_DIR/b300_dynamic_fuse${1}_b${ORBITCTA_FLAT_DYNAMIC_BATCH}_n21"
  local so="$LOGDIR/fuse${fuse}_r${rep}.out" se="$LOGDIR/fuse${fuse}_r${rep}.err" util="$LOGDIR/fuse${fuse}_r${rep}.util"
  env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  local pid=$!; sample "$pid" "$util" & local sp=$!; set +e; wait "$pid"; local rc=$?; set -e; wait "$sp" || true
  ((rc==0)) || { echo "fuse=$fuse failed rc=$rc" >&2; exit "$rc"; }
  local line="$(grep '^residue=' "$so"|tail -n1||true)"; [[ -n "$line" ]] || { echo "fuse=$fuse missing residue" >&2; exit 3; }
  local residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "fuse=$fuse residue=$residue expected=$EXPECT" >&2; exit 4; }
  local d="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1||true)"
  local grid="$(grep 'rankformula_orbitcta_flat_grid device=0 ' "$se"|head -n1||true)"
  local occ="$(grep 'rankformula_orbitcta_flat_occupancy device=0 ' "$se"|head -n1||true)"
  [[ -n "$grid" && -n "$occ" ]] || { echo "fuse=$fuse missing flat runtime metadata" >&2; exit 5; }
  local sched="$(field scheduler_mode "$grid")" runtime_batch="$(field flat_dynamic_batch "$grid")" lease="$(field queue_lease_batch "$grid")"
  [[ "$sched" == dynamic_atomic_queue ]] || { echo "fuse=$fuse scheduler=$sched expected=dynamic_atomic_queue" >&2; exit 6; }
  [[ "$runtime_batch" == "$ORBITCTA_FLAT_DYNAMIC_BATCH" && "$lease" == "$ORBITCTA_FLAT_DYNAMIC_BATCH" ]] || { echo "fuse=$fuse runtime_batch=$runtime_batch lease=$lease expected=$ORBITCTA_FLAT_DYNAMIC_BATCH" >&2; exit 6; }
  local bidmode="$(field flat_bid_mode "$grid")" bidfused="$(field flat_bid_fused "$grid")" expected_bid=binary_search
  [[ "$ORBIT_PRECTX_FLAT_BID" == 0 ]] || expected_bid=compact_prectx
  [[ "$bidmode" == "$expected_bid" && "$bidfused" == "$ORBIT_PRECTX_FLAT_BID_FUSED" ]] || { echo "fuse=$fuse bid_mode=$bidmode/$bidfused expected=$expected_bid/$ORBIT_PRECTX_FLAT_BID_FUSED" >&2; exit 6; }
  local fh rh high ag am mm
  fh="$(field forward_high_s "$d")"; rh="$(field reverse_high_s "$d")"; [[ -n "$fh" && -n "$rh" ]] || { echo "fuse=$fuse missing HIGH timing" >&2; exit 5; }
  high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  read -r ag am mm < <(awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$fuse" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$mm" \
    "$(field forward_regs "$occ")" "$(field reverse_regs "$occ")" "$(field forward_blocks_per_sm "$occ")" "$(field reverse_blocks_per_sm "$occ")" \
    "$sched" "$lease" "$bidmode" "$bidfused" >>"$RESULT"
}
for ((r=1;r<=REPEATS;++r)); do run_one 0 "$r"; run_one 1 "$r"; done
cat "$RESULT"
python3 - "$PROFILE_IN" "$RESULT" "$PROFILE_OUT" <<'PY'
import csv,statistics,sys,re
pin,result,pout=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); z={}
for f in ('0','1'):
 g=[r for r in rows if r['fuse']==f]
 mc=[float(r['avg_memctrl_util_pct']) for r in g if r['avg_memctrl_util_pct']!='NA']
 z[f]={
  'wall':statistics.median(float(r['wall_s']) for r in g),
  'high':statistics.median(float(r['high_s']) for r in g),
  'mc':statistics.median(mc) if mc else float('nan'),
  'fr':int(g[0]['forward_regs'] or 0),'rr':int(g[0]['reverse_regs'] or 0),
  'fb':int(g[0]['forward_blocks_per_sm'] or 0),'rb':int(g[0]['reverse_blocks_per_sm'] or 0)}
best='1' if z['1']['wall'] < z['0']['wall'] else '0'
for f in ('0','1'):
 print('DYNAMIC_FUSE',f,f"wall_s={z[f]['wall']:.6f}",f"high_s={z[f]['high']:.6f}",f"mc_avg_pct={z[f]['mc']:.3f}",f"regs={z[f]['fr']}/{z[f]['rr']}",f"active_blocks_sm={z[f]['fb']}/{z[f]['rb']}")
updates={
 'ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP':best,
 'ORBIT_DYNAMIC_FUSE_PROFILE':'fuse'+best,
 'ORBIT_DYNAMIC_FUSE_WALL_S':f'{z[best]["wall"]:.9f}',
 'ORBIT_DYNAMIC_FUSE_HIGH_S':f'{z[best]["high"]:.9f}',
}
out=[]; seen=set()
for line in open(pin):
 s=line.rstrip('\n')
 if '=' not in s or s.lstrip().startswith('#'):
  out.append(s); continue
 k,v=s.split('=',1)
 if k=='ORBIT_PROFILE':
  v=re.sub(r'_df[01]$','',v)
  out.append('ORBIT_PROFILE='+v+('_df1' if best=='1' else ''))
  continue
 if k in updates:
  if k not in seen: out.append(k+'='+updates[k]); seen.add(k)
  continue
 out.append(s)
for k in ('ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP','ORBIT_DYNAMIC_FUSE_PROFILE','ORBIT_DYNAMIC_FUSE_WALL_S','ORBIT_DYNAMIC_FUSE_HIGH_S'):
 if k not in seen: out.append(k+'='+updates[k])
open(pout,'w').write('\n'.join(out)+'\n')
print('BEST_DYNAMIC_FUSE='+best,f"speedup={z['0']['wall']/z['1']['wall']:.6f}",f'profile_file={pout}')
PY
cat "$PROFILE_OUT"
echo "dynamic fuse refinement OK input=$PROFILE_IN output=$PROFILE_OUT resources=$RESOURCE" >&2
