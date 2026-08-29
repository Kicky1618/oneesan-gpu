#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROFILE_IN="${PROFILE_IN:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_pipe221.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_quad21.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_dynamic_quad21}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
[[ -f "$PROFILE_IN" ]] || { echo "missing PROFILE_IN=$PROFILE_IN" >&2; exit 2; }
source "$PROFILE_IN"
: "${ORBIT_SPARSE64:?profile missing ORBIT_SPARSE64}"
: "${ORBIT_CPASYNC_PAIR:?profile missing ORBIT_CPASYNC_PAIR}"
ORBITCTA_FLAT_DYNAMIC="${ORBITCTA_FLAT_DYNAMIC:-0}"; ORBITCTA_FLAT_DYNAMIC_BATCH="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"
ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="${ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP:-0}"; ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="${ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES:-0}"; ORBITCTA_FLAT_DYNAMIC_PIPE2="${ORBITCTA_FLAT_DYNAMIC_PIPE2:-0}"
ORBIT_PRECTX_FORWARD="${ORBIT_PRECTX_FORWARD:-0}"; ORBIT_PRECTX_REVERSE="${ORBIT_PRECTX_REVERSE:-0}"; ORBIT_PRECTX_COMPACT="${ORBIT_PRECTX_COMPACT:-0}"
ORBIT_PRECTX_FLAT_BID="${ORBIT_PRECTX_FLAT_BID:-0}"; ORBIT_PRECTX_FLAT_BID_FUSED="${ORBIT_PRECTX_FLAT_BID_FUSED:-0}"
[[ "$ORBITCTA_FLAT_DYNAMIC" == 1 ]] || { echo 'dynamic quad refine requires dynamic queue' >&2; exit 2; }
for x in ORBIT_SPARSE64 ORBIT_CPASYNC_PAIR ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP ORBITCTA_FLAT_DYNAMIC_PIPE2 ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE ORBIT_PRECTX_COMPACT ORBIT_PRECTX_FLAT_BID ORBIT_PRECTX_FLAT_BID_FUSED; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
case "$ORBITCTA_FLAT_DYNAMIC_BATCH" in 1|2|4|8|16);;*)exit 2;;esac
case "$ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES" in 0|1|2|4);;*)exit 2;;esac

N=21 ARCH="${ARCH:-sm_103}" TARGET_MIB="${TARGET_MIB:-16384}" MAX_WINDOW="${MAX_WINDOW:-14}" REPEATS="${REPEATS:-1}" PM_ACCUM="${PM_ACCUM:-1}" \
DIRECTGATHER_SPARSE64="$ORBIT_SPARSE64" CPASYNC_PAIR="$ORBIT_CPASYNC_PAIR" \
ORBITCTA_FLAT_DYNAMIC_BATCH="$ORBITCTA_FLAT_DYNAMIC_BATCH" ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="$ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP" ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="$ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES" ORBITCTA_FLAT_DYNAMIC_PIPE2="$ORBITCTA_FLAT_DYNAMIC_PIPE2" \
PRECTX_FORWARD="$ORBIT_PRECTX_FORWARD" PRECTX_REVERSE="$ORBIT_PRECTX_REVERSE" PRECTX_COMPACT="$ORBIT_PRECTX_COMPACT" PRECTX_FLAT_BID="$ORBIT_PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED="$ORBIT_PRECTX_FLAT_BID_FUSED" \
PREFIX="$PREFIX" WINNER_ENV="$WINNER_ENV" bash "$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-dynamic-quad-ab.sh"
[[ -f "$WINNER_ENV" ]] || { echo "missing dynamic quad winner=$WINNER_ENV" >&2; exit 3; }

python3 - "$PROFILE_IN" "$WINNER_ENV" "$PROFILE_OUT" <<'PY'
import sys,re
pin,wenv,pout=sys.argv[1:]
def env(path):
 d={}
 for line in open(path):
  s=line.strip()
  if s and not s.startswith('#') and '=' in s:
   k,v=s.split('=',1);d[k]=v.strip('"')
 return d
w=env(wenv); q=w['QUAD_MLP']
lines=open(pin).read().splitlines();out=[];seen=set()
repl={
 'ORBIT_QUAD_MLP':q,
 'ORBIT_COL_ILP':'4',
 'ORBIT_QUAD_OVERLAP_LOCAL':'0',
 'ORBIT_QUAD_LOCAL_DIRECT_MAX':'0',
 'ORBIT_QUAD_SPARSE_DESC_MLP':'0',
 'ORBIT_QUAD_OVERLAP_BYPASS_LOCAL0':'0',
 'ORBIT_QUAD_CPASYNC_GROUP_COLS':'1',
 'ORBIT_QUAD_CPASYNC_PREFETCH_BYTES':'0',
}
for line in lines:
 if '=' in line and not line.startswith('#'):
  k=line.split('=',1)[0]
  if k in repl: out.append(k+'='+repl[k]);seen.add(k);continue
 if line.startswith('ORBIT_PROFILE='):
  val=re.sub(r'_dq$','',line.split('=',1)[1]);out.append('ORBIT_PROFILE='+val+('_dq' if q=='1' else ''));continue
 out.append(line)
for k,v in repl.items():
 if k not in seen: out.append(k+'='+v)
for k in ('DYNAMIC_QUAD_WALL_S','DYNAMIC_QUAD_HIGH_S'):
 if k in w: out.append('ORBIT_'+k+'='+w[k])
open(pout,'w').write('\n'.join(out)+'\n')
print('DYNAMIC_QUAD_REFINE',f'quad={q}',f'profile_file={pout}')
PY
cat "$PROFILE_OUT"
echo "dynamic quad refine OK input=$PROFILE_IN output=$PROFILE_OUT winner=$WINNER_ENV" >&2
