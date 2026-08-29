#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROFILE_IN="${PROFILE_IN:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_quad21.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_producer21.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_dynamic_producer21}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
[[ -f "$PROFILE_IN" ]] || { echo "missing PROFILE_IN=$PROFILE_IN" >&2; exit 2; }
# shellcheck disable=SC1090
source "$PROFILE_IN"

: "${ORBIT_SPARSE64:?profile missing ORBIT_SPARSE64}"
: "${ORBIT_CPASYNC_PAIR:?profile missing ORBIT_CPASYNC_PAIR}"
: "${ORBIT_COL_ILP:?profile missing ORBIT_COL_ILP}"
ORBITCTA_FLAT_DYNAMIC="${ORBITCTA_FLAT_DYNAMIC:-0}"
ORBITCTA_FLAT_DYNAMIC_BATCH="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"
ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="${ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP:-0}"
ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="${ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES:-0}"
ORBITCTA_FLAT_DYNAMIC_PIPE2="${ORBITCTA_FLAT_DYNAMIC_PIPE2:-0}"
ORBITCTA_FLAT_BLOCKS_PER_SM="${ORBITCTA_FLAT_BLOCKS_PER_SM:-0}"
ORBIT_QUAD_MLP="${ORBIT_QUAD_MLP:-0}"
ORBIT_PRECTX_FORWARD="${ORBIT_PRECTX_FORWARD:-0}"
ORBIT_PRECTX_REVERSE="${ORBIT_PRECTX_REVERSE:-0}"
ORBIT_PRECTX_COMPACT="${ORBIT_PRECTX_COMPACT:-0}"
ORBIT_PRECTX_FLAT_BID="${ORBIT_PRECTX_FLAT_BID:-0}"
ORBIT_PRECTX_FLAT_BID_FUSED="${ORBIT_PRECTX_FLAT_BID_FUSED:-0}"

[[ "$ORBITCTA_FLAT_DYNAMIC" == 1 ]] || { echo 'producer refine requires dynamic queue' >&2; exit 2; }
[[ "$ORBITCTA_FLAT_DYNAMIC_PIPE2" == 1 ]] || { echo 'producer refine requires selected PIPE2' >&2; exit 2; }
[[ "$ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP" == 0 ]] || { echo 'producer refine requires PIPE2 fusion=0' >&2; exit 2; }
for x in ORBIT_SPARSE64 ORBIT_CPASYNC_PAIR ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP ORBITCTA_FLAT_DYNAMIC_PIPE2 ORBIT_QUAD_MLP ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE ORBIT_PRECTX_COMPACT ORBIT_PRECTX_FLAT_BID ORBIT_PRECTX_FLAT_BID_FUSED; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
case "$ORBIT_COL_ILP" in 1|2|4) ;; *) echo 'ORBIT_COL_ILP must be 1,2,4' >&2; exit 2;; esac
case "$ORBITCTA_FLAT_DYNAMIC_BATCH" in 1|2|4|8|16) ;; *) exit 2;; esac
case "$ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES" in 0|1|2|4) ;; *) exit 2;; esac
[[ "$ORBITCTA_FLAT_BLOCKS_PER_SM" =~ ^[0-9]+$ ]] || { echo 'ORBITCTA_FLAT_BLOCKS_PER_SM must be non-negative integer' >&2; exit 2; }
[[ "$ORBIT_QUAD_MLP" == 0 || "$ORBIT_COL_ILP" == 4 ]] || { echo 'dynamic quad producer requires ILP4' >&2; exit 2; }

runenv=(
  N=21 ARCH="${ARCH:-sm_103}" TARGET_MIB="${TARGET_MIB:-16384}" MAX_WINDOW="${MAX_WINDOW:-14}" REPEATS="${REPEATS:-1}" PM_ACCUM="${PM_ACCUM:-1}"
  DIRECTGATHER_SPARSE64="$ORBIT_SPARSE64" CPASYNC_PAIR="$ORBIT_CPASYNC_PAIR" ORBITCTA_COL_ILP="$ORBIT_COL_ILP" QUAD_MLP="$ORBIT_QUAD_MLP"
  ORBITCTA_FLAT_DYNAMIC_BATCH="$ORBITCTA_FLAT_DYNAMIC_BATCH" ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="$ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES"
  PRECTX_FORWARD="$ORBIT_PRECTX_FORWARD" PRECTX_REVERSE="$ORBIT_PRECTX_REVERSE" PRECTX_COMPACT="$ORBIT_PRECTX_COMPACT"
  PRECTX_FLAT_BID="$ORBIT_PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED="$ORBIT_PRECTX_FLAT_BID_FUSED"
  PREFIX="$PREFIX" WINNER_ENV="$WINNER_ENV"
)
if (( ORBITCTA_FLAT_BLOCKS_PER_SM > 0 )); then
  runenv+=(BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM")
fi
env "${runenv[@]}" bash "$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-dynamic-pipe2-producer-selected-ab.sh"
[[ -f "$WINNER_ENV" ]] || { echo "missing producer winner=$WINNER_ENV" >&2; exit 3; }

python3 - "$PROFILE_IN" "$WINNER_ENV" "$PROFILE_OUT" <<'PY'
import re,sys
pin,wenv,pout=sys.argv[1:]
def env(path):
 d={}
 for line in open(path):
  s=line.strip()
  if s and not s.startswith('#') and '=' in s:
   k,v=s.split('=',1); d[k]=v.strip('"')
 return d
w=env(wenv)
prod=w['ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP']
weight=w.get('ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT','0')
if prod not in ('0','1'): raise SystemExit('bad producer winner '+prod)
if weight not in ('0','1','2','3','4'): raise SystemExit('bad producer weight '+weight)
if prod=='0' and weight!='0': raise SystemExit('non-producer winner must have weight=0')
repl={
 'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP':prod,
 'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT':weight,
 'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP':'0',
}
lines=open(pin).read().splitlines(); out=[]; seen=set()
for line in lines:
 if '=' in line and not line.startswith('#'):
  k=line.split('=',1)[0]
  if k in repl:
   out.append(k+'='+repl[k]); seen.add(k); continue
 if line.startswith('ORBIT_PROFILE='):
  val=line.split('=',1)[1]
  val=re.sub(r'_dpw(?:[0-4])?$','',val)
  out.append('ORBIT_PROFILE='+val+(f'_dpw{weight}' if prod=='1' else ''))
  continue
 out.append(line)
for k,v in repl.items():
 if k not in seen: out.append(k+'='+v)
for k in ('DYNAMIC_PRODUCER_WALL_S','DYNAMIC_PRODUCER_HIGH_S'):
 if k in w: out.append('ORBIT_'+k+'='+w[k])
open(pout,'w').write('\n'.join(out)+'\n')
print('DYNAMIC_PRODUCER_REFINE',f'producer={prod}',f'worker_weight={weight}',f'profile_file={pout}')
PY
cat "$PROFILE_OUT"
echo "dynamic producer refine OK input=$PROFILE_IN output=$PROFILE_OUT winner=$WINNER_ENV" >&2
