#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROFILE_IN="${PROFILE_IN:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_producer_prectx21.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_dynamic_producer_max21.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_dynamic_producer_max21}"
FLATBID_RESULT="${FLATBID_RESULT:-${PREFIX}_flatbid.tsv}"
OVERLAP_RESULT="${OVERLAP_RESULT:-${PREFIX}_overlap.tsv}"
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
ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP="${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP:-0}"
ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP="${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP:-0}"
ORBIT_QUAD_MLP="${ORBIT_QUAD_MLP:-0}"
ORBIT_QUAD_OVERLAP_LOCAL="${ORBIT_QUAD_OVERLAP_LOCAL:-0}"
ORBIT_PRECTX_FORWARD="${ORBIT_PRECTX_FORWARD:-0}"
ORBIT_PRECTX_REVERSE="${ORBIT_PRECTX_REVERSE:-0}"
ORBIT_PRECTX_COMPACT="${ORBIT_PRECTX_COMPACT:-0}"
ORBIT_PRECTX_FLAT_BID="${ORBIT_PRECTX_FLAT_BID:-0}"
ORBIT_PRECTX_FLAT_BID_FUSED="${ORBIT_PRECTX_FLAT_BID_FUSED:-0}"

[[ "$ORBITCTA_FLAT_DYNAMIC" == 1 && "$ORBITCTA_FLAT_DYNAMIC_PIPE2" == 1 ]] || { echo 'producer max refine requires dynamic PIPE2' >&2; exit 2; }
[[ "$ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP" == 1 && "$ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP" == 1 ]] || { echo 'producer max refine requires producer warp + producer prectx warpcoop' >&2; exit 2; }
[[ "$ORBIT_QUAD_MLP" == 1 && "$ORBIT_COL_ILP" == 4 && "$ORBIT_CPASYNC_PAIR" == 1 ]] || { echo 'producer max refine requires quad ILP4 cp.async' >&2; exit 2; }
[[ "$ORBIT_PRECTX_FORWARD" == 1 && "$ORBIT_PRECTX_REVERSE" == 1 && "$ORBIT_PRECTX_COMPACT" == 1 ]] || { echo 'producer max refine requires compact forward+reverse prectx' >&2; exit 2; }
[[ "$ORBIT_PRECTX_FLAT_BID_FUSED" == 0 ]] || { echo 'producer max refine requires fused flat-bid load off' >&2; exit 2; }
[[ "$ORBIT_QUAD_OVERLAP_LOCAL" == 0 ]] || { echo 'producer max refine expects overlap-local not yet selected' >&2; exit 2; }
case "$ORBITCTA_FLAT_DYNAMIC_BATCH" in 1|2|4|8|16) ;; *) exit 2;; esac
case "$ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES" in 0|1|2|4) ;; *) exit 2;; esac

COMMON=(N=21 ARCH="${ARCH:-sm_103}" TARGET_MIB="${TARGET_MIB:-16384}" MAX_WINDOW="${MAX_WINDOW:-14}" REPEATS="${REPEATS:-1}" PM_ACCUM="${PM_ACCUM:-1}"
  DIRECTGATHER_SPARSE64="$ORBIT_SPARSE64" ORBITCTA_FLAT_DYNAMIC_BATCH="$ORBITCTA_FLAT_DYNAMIC_BATCH"
  ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES="$ORBITCTA_FLAT_DYNAMIC_ADAPTIVE_WAVES")

env "${COMMON[@]}" PREFIX="${PREFIX}_flatbid" RESULT="$FLATBID_RESULT" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-dynamic-pipe2-producer-flatbid-ab.sh"
[[ -f "$FLATBID_RESULT" ]] || { echo "missing flatbid result=$FLATBID_RESULT" >&2; exit 3; }

FLATBID_WIN="$(python3 - "$FLATBID_RESULT" <<'PY'
import csv,statistics,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
def m(v): return statistics.median(float(x['wall_s']) for x in r if x['variant']==v)
print(1 if m('cached') < m('binary') else 0)
PY
)"

env "${COMMON[@]}" PRECTX_FLAT_BID="$FLATBID_WIN" PREFIX="${PREFIX}_overlap" RESULT="$OVERLAP_RESULT" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-orbitcta-pipe2-producer-quad-overlap-ab.sh"
[[ -f "$OVERLAP_RESULT" ]] || { echo "missing overlap result=$OVERLAP_RESULT" >&2; exit 3; }

python3 - "$PROFILE_IN" "$FLATBID_RESULT" "$OVERLAP_RESULT" "$PROFILE_OUT" "$FLATBID_WIN" <<'PY'
import csv,re,statistics,sys
pin,fbfile,qolfile,pout,fbflag=sys.argv[1:]
def rows(path): return list(csv.DictReader(open(path),delimiter='\t'))
def med(rs,v,k):
 vals=[float(x[k]) for x in rs if x['variant']==v and x[k] not in ('','NA')]
 if not vals: raise SystemExit(f'missing {v}/{k} in result')
 return statistics.median(vals)
fb=rows(fbfile); qol=rows(qolfile)
qolflag='1' if med(qol,'overlap','wall_s') < med(qol,'base','wall_s') else '0'
qwin='overlap' if qolflag=='1' else 'base'
repl={'ORBIT_PRECTX_FLAT_BID':fbflag,'ORBIT_QUAD_OVERLAP_LOCAL':qolflag}
out=[]; seen=set()
for line in open(pin).read().splitlines():
 if '=' in line and not line.startswith('#'):
  k=line.split('=',1)[0]
  if k in repl:
   out.append(k+'='+repl[k]);seen.add(k);continue
 if line.startswith('ORBIT_PROFILE='):
  val=line.split('=',1)[1]
  val=re.sub(r'_(fb|qol|maxp)+$','',val)
  suffix=('_fb' if fbflag=='1' else '')+('_qol' if qolflag=='1' else '')+'_maxp'
  out.append('ORBIT_PROFILE='+val+suffix);continue
 out.append(line)
for k,v in repl.items():
 if k not in seen: out.append(k+'='+v)
out.append(f'ORBIT_DYNAMIC_PRODUCER_MAX_WALL_S={med(qol,qwin,"wall_s"):.9f}')
out.append(f'ORBIT_DYNAMIC_PRODUCER_MAX_HIGH_S={med(qol,qwin,"high_s"):.9f}')
out.append(f'ORBIT_DYNAMIC_PRODUCER_MAX_MEMCTRL_PCT={med(qol,qwin,"avg_memctrl_pct"):.6f}')
out.append(f'ORBIT_DYNAMIC_PRODUCER_FLATBID_WALL_BINARY_S={med(fb,"binary","wall_s"):.9f}')
out.append(f'ORBIT_DYNAMIC_PRODUCER_FLATBID_WALL_CACHED_S={med(fb,"cached","wall_s"):.9f}')
open(pout,'w').write('\n'.join(out)+'\n')
print('DYNAMIC_PRODUCER_MAX_REFINE',f'flat_bid={fbflag}',f'overlap_local={qolflag}',f'profile_file={pout}')
PY
cat "$PROFILE_OUT"
echo "dynamic producer max refine OK input=$PROFILE_IN output=$PROFILE_OUT flatbid_result=$FLATBID_RESULT overlap_result=$OVERLAP_RESULT" >&2
