#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROFILE_IN="${PROFILE_IN:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_fuse21.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_adaptive21.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_dynamic_adaptive21}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
REPEATS="${REPEATS:-1}"; WAVES_VALUES="${WAVES_VALUES:-0 1 2 4}"
N=21; MOD=4294967291; EXPECT=998035516; NGPU="${NGPU:-8}"
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
[[ "$ORBITCTA_FLAT_DYNAMIC" == 1 ]] || { echo 'adaptive refine requires selected dynamic queue' >&2; exit 2; }
case "$ORBITCTA_FLAT_DYNAMIC_BATCH" in 2|4|8|16) ;; *) echo 'adaptive refine requires dynamic batch 2,4,8,16' >&2; exit 2;; esac
case "$ORBIT_COL_ILP" in 2|4) ;; *) echo 'adaptive refine expects ORBIT_COL_ILP=2 or 4' >&2; exit 2;; esac
for w in $WAVES_VALUES; do case "$w" in 0|1|2|4) ;; *) echo "bad adaptive waves=$w" >&2; exit 2;; esac; done
for x in ORBIT_SPARSE64 ORBIT_CPASYNC_PAIR ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE ORBIT_PRECTX_COMPACT ORBIT_PRECTX_FLAT_BID ORBIT_PRECTX_FLAT_BID_FUSED PM_ACCUM; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= NGPU )) || { echo "need $NGPU visible GPUs" >&2; exit 2; }
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"

COMMON=(N="$N" ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$ORBIT_SPARSE64" DIRECTGATHER_SORT_RANKS=0 RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR="$ORBIT_CPASYNC_PAIR" ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK=1 ORBITCTA_FLAT_DYNAMIC=1 ORBITCTA_FLAT_DYNAMIC_BATCH="$ORBITCTA_FLAT_DYNAMIC_BATCH" ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="$ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP" ORBITCTA_COL_ILP="$ORBIT_COL_ILP" QUAD_MLP=0 QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP=0 QUAD_OVERLAP_BYPASS_LOCAL0=0 QUAD_CPASYNC_PREFETCH_BYTES=0 QUAD_CPASYNC_GROUP_COLS=1 PRECTX_FORWARD="$ORBIT_PRECTX_FORWARD" PRECTX_REVERSE="$ORBIT_PRECTX_REVERSE" PRECTX_COMPACT="$ORBIT_PRECTX_COMPACT" PRECTX_FLAT_BID="$ORBIT_PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED="$ORBIT_PRECTX_FLAT_BID_FUSED" PRECTX_WARPCOOP=0 PTXAS_VERBOSE=1)
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
for waves in $WAVES_VALUES; do
  bin="$ONEESAN_BUILD_DIR/b300_dynamic_adaptive_w${waves}_b${ORBITCTA_FLAT_DYNAMIC_BATCH}_f${ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP}_n21"
  env "${COMMON[@]}" ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="$waves" OUT="$bin" \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/w${waves}.build.out" 2>"$LOGDIR/w${waves}.build.err"
  grep -q "flat_dynamic=1 flat_dynamic_batch=$ORBITCTA_FLAT_DYNAMIC_BATCH flat_dynamic_fuse_lease_prep=$ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP flat_dynamic_adaptive_waves=$waves" "$LOGDIR/w${waves}.build.err" || { echo "waves=$waves build marker mismatch" >&2; exit 6; }
  python3 "$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py" "$LOGDIR/w${waves}.build.err" --label "waves${waves}" >>"$RESOURCE" || true
done

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }
printf 'waves\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
run_one(){
  local waves="$1" rep="$2"
  local bin="$ONEESAN_BUILD_DIR/b300_dynamic_adaptive_w${waves}_b${ORBITCTA_FLAT_DYNAMIC_BATCH}_f${ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP}_n21"
  local so="$LOGDIR/w${waves}_r${rep}.out" se="$LOGDIR/w${waves}_r${rep}.err" util="$LOGDIR/w${waves}_r${rep}.util"
  env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
    "$bin" "$N" "$TARGET_MIB" "$MAX_WINDOW" "$NGPU" "$MOD" >"$so" 2>"$se" &
  local pid=$!; sample "$pid" "$util" & local sp=$!; set +e; wait "$pid"; local rc=$?; set -e; wait "$sp" || true
  ((rc==0)) || { echo "waves=$waves failed rc=$rc" >&2; exit "$rc"; }
  local line="$(grep '^residue=' "$so"|tail -n1||true)"; [[ -n "$line" ]] || exit 3
  local residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "waves=$waves residue=$residue expected=$EXPECT" >&2; exit 4; }
  local d="$(grep 'snake_onepass_graph_batch modulus=' "$se"|tail -n1||true)" fh rh high ag am mm
  fh="$(field forward_high_s "$d")"; rh="$(field reverse_high_s "$d")"; [[ -n "$fh" && -n "$rh" ]] || exit 5
  high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  read -r ag am mm < <(awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$waves" "$rep" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$mm" >>"$RESULT"
}
for waves in $WAVES_VALUES; do for ((r=1;r<=REPEATS;++r)); do run_one "$waves" "$r"; done; done
cat "$RESULT"
python3 - "$PROFILE_IN" "$RESULT" "$PROFILE_OUT" <<'PY'
import csv,statistics,sys,re
pin,result,pout=sys.argv[1:]; rows=list(csv.DictReader(open(result),delimiter='\t')); z={}
for w in dict.fromkeys(r['waves'] for r in rows):
 g=[r for r in rows if r['waves']==w]
 z[w]=(statistics.median(float(r['wall_s']) for r in g),statistics.median(float(r['high_s']) for r in g))
best=min(z,key=lambda w:z[w][0]); lines=open(pin).read().splitlines(); out=[]; seen=False
for line in lines:
 if line.startswith('ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES='):
  out.append('ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES='+best); seen=True
 elif line.startswith('ORBIT_PROFILE='):
  val=re.sub(r'_daw[0124]$','',line.split('=',1)[1])
  out.append('ORBIT_PROFILE='+val+('' if best=='0' else '_daw'+best))
 else: out.append(line)
if not seen: out.append('ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES='+best)
out += [f'ORBIT_DYNAMIC_ADAPTIVE_WALL_S={z[best][0]:.9f}',f'ORBIT_DYNAMIC_ADAPTIVE_HIGH_S={z[best][1]:.9f}']
open(pout,'w').write('\n'.join(out)+'\n')
for w,(wall,high) in sorted(z.items(),key=lambda kv:kv[1][0]): print('DYNAMIC_ADAPTIVE',w,f'wall_s={wall:.6f}',f'high_s={high:.6f}')
print('BEST_DYNAMIC_ADAPTIVE_WAVES='+best,f'profile_file={pout}')
PY
cat "$PROFILE_OUT"
echo "dynamic adaptive refinement OK input=$PROFILE_IN output=$PROFILE_OUT" >&2
