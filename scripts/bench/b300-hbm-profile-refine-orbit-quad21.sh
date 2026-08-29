#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROFILE_IN="${PROFILE_IN:-$ONEESAN_ROOT/work/b300_hbm_profile_scheduler21.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_quad21}"
[[ -f "$PROFILE_IN" ]] || { echo "missing PROFILE_IN=$PROFILE_IN" >&2; exit 2; }
# shellcheck disable=SC1090
source "$PROFILE_IN"

N=21; MOD=4294967291; EXPECT=998035516; NGPU=8
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; ORBIT_GY="${BUCKET_ORBITCTA_GRID_Y:-128}"
LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"; PM_ACCUM="${PM_ACCUM:-1}"
REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.20}"
CHUNKS="${CHUNKS:-2 4 8}"; POOLS="${POOLS:-auto 1 2}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; CURRENT_RESULT="${CURRENT_RESULT:-${PREFIX}_current.tsv}"
QUAD_WINNER_ENV="${QUAD_WINNER_ENV:-${PREFIX}_quad_winner.env}"
mkdir -p "$LOGDIR" "$(dirname "$PROFILE_OUT")"

for n in ORBIT_PROFILE ORBIT_COL_ILP ORBIT_SPARSE64 ORBIT_SORTED ORBIT_CPASYNC_PAIR ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE ORBIT_PRECTX_COMPACT ORBITCTA_FLAT ORBITCTA_FLAT_BLOCKS_PER_SM; do
  [[ -n "${!n+x}" ]] || { echo "profile missing $n" >&2; exit 2; }
done
[[ "$ORBIT_SORTED" == 0 ]] || { echo 'orbit quad refine requires unsorted orbit-CTA profile' >&2; exit 2; }
for x in ORBIT_SPARSE64 ORBIT_CPASYNC_PAIR ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE ORBIT_PRECTX_COMPACT ORBITCTA_FLAT; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
case "$ORBIT_COL_ILP" in 1|2|4) ;; *) echo 'bad ORBIT_COL_ILP' >&2; exit 2;; esac
[[ "$ORBITCTA_FLAT_BLOCKS_PER_SM" =~ ^[0-9]+$ ]] || { echo 'bad ORBITCTA_FLAT_BLOCKS_PER_SM' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

if [[ "$ORBIT_CPASYNC_PAIR" == 1 ]]; then
  ARCH="$ARCH" NGPU=8 THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/current-cpasync.out" 2>"$LOGDIR/current-cpasync.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/current-cpasync.out" || { echo 'current cp.async peer gate failed' >&2; exit 5; }
fi
if [[ "$ORBIT_PRECTX_COMPACT" == 1 ]]; then
  ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" >"$LOGDIR/current-prectx.out" 2>"$LOGDIR/current-prectx.err"
fi

CURRENT_BIN="$ONEESAN_BUILD_DIR/b300_orbit_quad_refine_current_n21"
N=21 ARCH="$ARCH" OUT="$CURRENT_BIN" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$ORBIT_SPARSE64" DIRECTGATHER_SORT_RANKS=0 \
  RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR="$ORBIT_CPASYNC_PAIR" ORBITCTA_COL_ILP="$ORBIT_COL_ILP" \
  ORBITCTA_FLAT="$ORBITCTA_FLAT" ORBITCTA_FLAT_CHUNK=1 QUAD_MLP=0 QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 \
  PRECTX_FORWARD="$ORBIT_PRECTX_FORWARD" PRECTX_REVERSE="$ORBIT_PRECTX_REVERSE" PRECTX_COMPACT="$ORBIT_PRECTX_COMPACT" PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/current.build.out" 2>"$LOGDIR/current.build.err"

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }

printf 'repeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$CURRENT_RESULT"
for ((r=1;r<=REPEATS;++r)); do
  so="$LOGDIR/current_r${r}.out"; se="$LOGDIR/current_r${r}.err"; util="$LOGDIR/current_r${r}.util"
  if [[ "$ORBITCTA_FLAT" == 1 && "$ORBITCTA_FLAT_BLOCKS_PER_SM" == 0 ]]; then
    env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM \
      BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$CURRENT_BIN" 21 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
  elif [[ "$ORBITCTA_FLAT" == 1 ]]; then
    env -u BUCKET_ORBITCTA_FLAT_BLOCKS BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM" \
      BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$CURRENT_BIN" 21 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
  else
    BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$CURRENT_BIN" 21 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
  fi
  pid=$!; sample "$pid" "$util" & sp=$!; set +e; wait "$pid"; rc=$?; set -e; wait "$sp" || true
  ((rc==0)) || { echo "current orbit repeat=$r failed rc=$rc" >&2; exit "$rc"; }
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo 'current orbit missing residue' >&2; exit 3; }
  residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "current orbit residue=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"; fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
  [[ -n "$fh" && -n "$rh" ]] || { echo 'current orbit missing HIGH timing' >&2; exit 6; }
  high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  read -r ag am mm < <(awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$mm" >>"$CURRENT_RESULT"
done

QUAD_PREFIX="${QUAD_PREFIX:-${PREFIX}_quad}"
N=21 ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" THREADS="$THREADS" CHUNKS="$CHUNKS" POOLS="$POOLS" REPEATS="$REPEATS" \
  SPARSE64="$ORBIT_SPARSE64" PRECTX_FORWARD="$ORBIT_PRECTX_FORWARD" PRECTX_REVERSE="$ORBIT_PRECTX_REVERSE" PRECTX_COMPACT="$ORBIT_PRECTX_COMPACT" \
  LOW_GX="$LOW_GX" LOW_GY="$LOW_GY" PREFIX="$QUAD_PREFIX" WINNER_ENV="$QUAD_WINNER_ENV" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-quad-sweep.sh"
[[ -f "$QUAD_WINNER_ENV" ]] || { echo "missing quad winner env=$QUAD_WINNER_ENV" >&2; exit 7; }
# shellcheck disable=SC1090
source "$QUAD_WINNER_ENV"

python3 - "$CURRENT_RESULT" "$PROFILE_IN" "$PROFILE_OUT" "$QUAD_WINNER_ENV" <<'PY'
import csv,statistics,sys,re
cur,profile_in,profile_out,quad_env=sys.argv[1:]
rows=list(csv.DictReader(open(cur),delimiter='\t'))
current_wall=statistics.median(float(r['wall_s']) for r in rows)
current_high=statistics.median(float(r['high_s']) for r in rows)
q={}
for line in open(quad_env):
 s=line.strip()
 if s and not s.startswith('#') and '=' in s:
  k,v=s.split('=',1);q[k]=v.strip('"')
quad_wall=float(q['ORBIT_QUAD_WALL_S']);quad_high=float(q['ORBIT_QUAD_HIGH_S'])
kv={};order=[]
for line in open(profile_in):
 s=line.strip()
 if not s or s.startswith('#') or '=' not in s: continue
 k,v=s.split('=',1)
 if k not in kv: order.append(k)
 kv[k]=v.strip('"')
for k,v in {
 'ORBITCTA_FLAT_CHUNK':'1',
 'ORBIT_QUAD_MLP':'0',
 'ORBIT_QUAD_OVERLAP_LOCAL':'0',
 'ORBIT_QUAD_LOCAL_DIRECT_MAX':'0',
}.items():
 kv.setdefault(k,v)
 if k not in order: order.append(k)
if quad_wall < current_wall:
 for k in ('ORBITCTA_FLAT','ORBITCTA_FLAT_CHUNK','ORBITCTA_FLAT_BLOCKS_PER_SM','ORBIT_QUAD_MLP','ORBIT_QUAD_OVERLAP_LOCAL','ORBIT_QUAD_LOCAL_DIRECT_MAX','ORBIT_CPASYNC_PAIR','ORBIT_COL_ILP'):
  kv[k]=q[k]
  if k not in order: order.append(k)
 root=re.sub(r'_quad_.*$','',kv['ORBIT_PROFILE'])
 kv['ORBIT_PROFILE']=root+'_quad_'+q['ORBIT_QUAD_PROFILE']
 winner='quad'
else:
 winner='current'
with open(profile_out,'w') as f:
 f.write('# generated by b300-hbm-profile-refine-orbit-quad21.sh\n')
 for k in order:
  v=kv[k]
  if k=='CANDIDATES': f.write(f'{k}="{v}"\n')
  else: f.write(f'{k}={v}\n')
print(f'CURRENT_ORBIT wall_s={current_wall:.6f} high_s={current_high:.6f}')
print(f'QUAD_ORBIT wall_s={quad_wall:.6f} high_s={quad_high:.6f}')
print('BEST_ORBIT_QUAD_REFINE='+winner,'profile_file='+profile_out)
PY
cat "$PROFILE_OUT"
echo "orbit quad refine OK input=$PROFILE_IN output=$PROFILE_OUT current=$CURRENT_RESULT quad=$QUAD_WINNER_ENV" >&2
