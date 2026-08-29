#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROFILE_IN="${PROFILE_IN:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_producer21.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_producer_prectx21.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_dynamic_producer_prectx21}"
RESULT="${RESULT:-${PREFIX}.tsv}"
[[ -f "$PROFILE_IN" ]] || { echo "missing PROFILE_IN=$PROFILE_IN" >&2; exit 2; }
# shellcheck disable=SC1090
source "$PROFILE_IN"

: "${ORBIT_SPARSE64:?profile missing ORBIT_SPARSE64}"
: "${ORBIT_CPASYNC_PAIR:?profile missing ORBIT_CPASYNC_PAIR}"
: "${ORBIT_COL_ILP:?profile missing ORBIT_COL_ILP}"
ORBITCTA_FLAT_DYNAMIC="${ORBITCTA_FLAT_DYNAMIC:-0}"
ORBITCTA_FLAT_DYNAMIC_BATCH="${ORBITCTA_FLAT_DYNAMIC_BATCH:-1}"
ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="${ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES:-0}"
ORBITCTA_FLAT_DYNAMIC_PIPE2="${ORBITCTA_FLAT_DYNAMIC_PIPE2:-0}"
ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP="${ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP:-0}"
ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP="${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP:-0}"
ORBITCTA_FLAT_BLOCKS_PER_SM="${ORBITCTA_FLAT_BLOCKS_PER_SM:-0}"
ORBIT_QUAD_MLP="${ORBIT_QUAD_MLP:-0}"
ORBIT_PRECTX_FORWARD="${ORBIT_PRECTX_FORWARD:-0}"
ORBIT_PRECTX_REVERSE="${ORBIT_PRECTX_REVERSE:-0}"
ORBIT_PRECTX_COMPACT="${ORBIT_PRECTX_COMPACT:-0}"
ORBIT_PRECTX_FLAT_BID="${ORBIT_PRECTX_FLAT_BID:-0}"
ORBIT_PRECTX_FLAT_BID_FUSED="${ORBIT_PRECTX_FLAT_BID_FUSED:-0}"

[[ "$ORBITCTA_FLAT_DYNAMIC" == 1 && "$ORBITCTA_FLAT_DYNAMIC_PIPE2" == 1 && "$ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP" == 1 ]] || { echo 'producer prectx refine requires selected dynamic PIPE2 producer warp' >&2; exit 2; }
[[ "$ORBITCTA_FLAT_DYNAMIC_FUSE_LEASE_PREP" == 0 ]] || { echo 'producer prectx refine requires fusion=0' >&2; exit 2; }
[[ "$ORBIT_QUAD_MLP" == 1 && "$ORBIT_COL_ILP" == 4 && "$ORBIT_CPASYNC_PAIR" == 1 ]] || { echo 'producer prectx exact runner currently requires quad=1 ILP4 cp.async=1' >&2; exit 2; }
[[ "$ORBIT_PRECTX_FORWARD" == 1 && "$ORBIT_PRECTX_REVERSE" == 1 && "$ORBIT_PRECTX_COMPACT" == 1 ]] || { echo 'producer prectx exact runner requires compact forward+reverse prectx' >&2; exit 2; }
[[ "$ORBIT_PRECTX_FLAT_BID_FUSED" == 0 ]] || { echo 'producer prectx exact runner currently requires flat-bid-fused=0' >&2; exit 2; }
[[ "$ORBITCTA_FLAT_BLOCKS_PER_SM" == 0 ]] || { echo 'producer prectx exact runner currently requires occupancy-derived pool' >&2; exit 2; }
for x in ORBIT_PRECTX_FLAT_BID ORBIT_PRECTX_FLAT_BID_FUSED; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
case "$ORBITCTA_FLAT_DYNAMIC_BATCH" in 1|2|4|8|16) ;; *) exit 2;; esac
case "$ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES" in 0|1|2|4) ;; *) exit 2;; esac

N=21 ARCH="${ARCH:-sm_103}" TARGET_MIB="${TARGET_MIB:-16384}" MAX_WINDOW="${MAX_WINDOW:-14}" REPEATS="${REPEATS:-1}" PM_ACCUM="${PM_ACCUM:-1}" \
DIRECTGATHER_SPARSE64="$ORBIT_SPARSE64" CPASYNC_PAIR=1 \
ORBITCTA_FLAT_DYNAMIC_BATCH="$ORBITCTA_FLAT_DYNAMIC_BATCH" ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="$ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES" \
PRECTX_FLAT_BID="$ORBIT_PRECTX_FLAT_BID" PRECTX_FLAT_BID_FUSED="$ORBIT_PRECTX_FLAT_BID_FUSED" \
PREFIX="$PREFIX" RESULT="$RESULT" bash "$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-dynamic-pipe2-producer-prectx-warpcoop-ab.sh"
[[ -f "$RESULT" ]] || { echo "missing producer prectx result=$RESULT" >&2; exit 3; }

python3 - "$PROFILE_IN" "$RESULT" "$PROFILE_OUT" <<'PY'
import csv,re,statistics,sys
pin,result,pout=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t'))
def med(v,k):
 vals=[float(r[k]) for r in rows if r['variant']==v and r[k] not in ('','NA')]
 if not vals: raise SystemExit(f'missing {v}/{k}')
 return statistics.median(vals)
serial=med('serial','wall_s'); coop=med('warpcoop','wall_s')
flag='1' if coop < serial else '0'; winner='warpcoop' if flag=='1' else 'serial'
repl={'ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP':flag}
lines=open(pin).read().splitlines();out=[];seen=set()
for line in lines:
 if '=' in line and not line.startswith('#'):
  k=line.split('=',1)[0]
  if k in repl:
   out.append(k+'='+repl[k]); seen.add(k); continue
 if line.startswith('ORBIT_PROFILE='):
  val=line.split('=',1)[1]
  val=re.sub(r'_ppw$','',val)
  out.append('ORBIT_PROFILE='+val+('_ppw' if flag=='1' else ''))
  continue
 out.append(line)
for k,v in repl.items():
 if k not in seen: out.append(k+'='+v)
out.append(f'ORBIT_DYNAMIC_PRODUCER_PRECTX_WALL_S={med(winner,"wall_s"):.9f}')
out.append(f'ORBIT_DYNAMIC_PRODUCER_PRECTX_HIGH_S={med(winner,"high_s"):.9f}')
open(pout,'w').write('\n'.join(out)+'\n')
print('DYNAMIC_PRODUCER_PRECTX_REFINE',f'warpcoop={flag}',f'serial_wall={serial:.9f}',f'warpcoop_wall={coop:.9f}',f'profile_file={pout}')
PY
cat "$PROFILE_OUT"
echo "dynamic producer prectx refine OK input=$PROFILE_IN output=$PROFILE_OUT result=$RESULT cached_bid=$ORBIT_PRECTX_FLAT_BID" >&2
