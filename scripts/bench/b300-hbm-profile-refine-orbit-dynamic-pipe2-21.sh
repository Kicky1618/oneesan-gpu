#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROFILE_IN="${PROFILE_IN:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_adaptive21.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_pipe221.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_dynamic_pipe221}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"
[[ -f "$PROFILE_IN" ]] || { echo "missing PROFILE_IN=$PROFILE_IN" >&2; exit 2; }
# shellcheck disable=SC1090
source "$PROFILE_IN"
: "${ORBIT_COL_ILP:?profile missing ORBIT_COL_ILP}"
: "${ORBIT_SPARSE64:?profile missing ORBIT_SPARSE64}"
: "${ORBIT_CPASYNC_PAIR:?profile missing ORBIT_CPASYNC_PAIR}"
ORBITCTA_FLAT_DYNAMIC="${ORBITCTA_FLAT_DYNAMIC:-0}"
ORBITCTA_FLAT_DYNAMIC_BATCH="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"
ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="${ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP:-0}"
ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="${ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES:-0}"
ORBIT_PRECTX_FORWARD="${ORBIT_PRECTX_FORWARD:-0}"; ORBIT_PRECTX_REVERSE="${ORBIT_PRECTX_REVERSE:-0}"; ORBIT_PRECTX_COMPACT="${ORBIT_PRECTX_COMPACT:-0}"
ORBIT_PRECTX_FLAT_BID="${ORBIT_PRECTX_FLAT_BID:-0}"; ORBIT_PRECTX_FLAT_BID_FUSED="${ORBIT_PRECTX_FLAT_BID_FUSED:-0}"
[[ "$ORBITCTA_FLAT_DYNAMIC" == 1 ]] || { echo 'pipe2 refine requires selected dynamic queue' >&2; exit 2; }
case "$ORBITCTA_FLAT_DYNAMIC_BATCH" in 1|2|4|8|16) ;; *) echo 'bad dynamic batch' >&2; exit 2;; esac
case "$ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES" in 0|1|2|4) ;; *) echo 'bad adaptive waves' >&2; exit 2;; esac
for x in ORBIT_SPARSE64 ORBIT_CPASYNC_PAIR ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE ORBIT_PRECTX_COMPACT ORBIT_PRECTX_FLAT_BID ORBIT_PRECTX_FLAT_BID_FUSED; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done

N=21 ARCH="${ARCH:-sm_103}" TARGET_MIB="${TARGET_MIB:-16384}" MAX_WINDOW="${MAX_WINDOW:-14}" REPEATS="${REPEATS:-1}" PM_ACCUM="${PM_ACCUM:-1}" \
  DIRECTGATHER_SPARSE64="$ORBIT_SPARSE64" ORBITCTA_COL_ILP="$ORBIT_COL_ILP" CPASYNC_PAIR="$ORBIT_CPASYNC_PAIR" \
  ORBITCTA_FLAT_DYNAMIC_BATCH="$ORBITCTA_FLAT_DYNAMIC_BATCH" ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="$ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP" \
  ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="$ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES" \
  PRECTX_FORWARD="$ORBIT_PRECTX_FORWARD" PRECTX_REVERSE="$ORBIT_PRECTX_REVERSE" PRECTX_COMPACT="$ORBIT_PRECTX_COMPACT" \
  PRECTX_FLAT_BID="$ORBIT_PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED="$ORBIT_PRECTX_FLAT_BID_FUSED" \
  PREFIX="$PREFIX" WINNER_ENV="$WINNER_ENV" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-dynamic-pipe2-ab.sh"
[[ -f "$WINNER_ENV" ]] || { echo "missing pipe2 winner env=$WINNER_ENV" >&2; exit 3; }

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
w=env(wenv); pipe=w['ORBITCTA_FLAT_DYNAMIC_PIPE2']; fuse=w['ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP']
lines=open(pin).read().splitlines();out=[];seen_pipe=False;seen_fuse=False
for line in lines:
 if line.startswith('ORBITCTA_FLAT_DYNAMIC_PIPE2='):
  out.append('ORBITCTA_FLAT_DYNAMIC_PIPE2='+pipe);seen_pipe=True
 elif line.startswith('ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP='):
  out.append('ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP='+fuse);seen_fuse=True
 elif line.startswith('ORBIT_PROFILE='):
  val=re.sub(r'_dp2$','',line.split('=',1)[1]);out.append('ORBIT_PROFILE='+val+('_dp2' if pipe=='1' else ''))
 else: out.append(line)
if not seen_pipe: out.append('ORBITCTA_FLAT_DYNAMIC_PIPE2='+pipe)
if not seen_fuse: out.append('ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP='+fuse)
for k in ('ORBIT_DYNAMIC_PIPE2_PROFILE','ORBIT_DYNAMIC_PIPE2_WALL_S','ORBIT_DYNAMIC_PIPE2_HIGH_S'):
 if k in w: out.append(k+'='+w[k])
open(pout,'w').write('\n'.join(out)+'\n')
print('DYNAMIC_PIPE2_REFINE',f'pipe2={pipe}',f'fuse={fuse}',f'profile_file={pout}')
PY
cat "$PROFILE_OUT"
echo "dynamic pipe2 refine OK input=$PROFILE_IN output=$PROFILE_OUT winner=$WINNER_ENV" >&2
