#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROFILE_IN="${PROFILE_IN:-$ONEESAN_ROOT/work/b300_hbm_profile_advanced21.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_warpcoop21.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_warpcoop21}"
QUAD_WINNER_ENV="${QUAD_WINNER_ENV:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_advanced21_quad_winner.env}"
[[ -f "$PROFILE_IN" ]] || { echo "missing PROFILE_IN=$PROFILE_IN" >&2; exit 2; }
[[ -f "$QUAD_WINNER_ENV" ]] || { echo "missing QUAD_WINNER_ENV=$QUAD_WINNER_ENV" >&2; exit 2; }
# shellcheck disable=SC1090
source "$PROFILE_IN"

N=21; MOD=4294967291; EXPECT=998035516; NGPU=8
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; ORBIT_GY="${BUCKET_ORBITCTA_GRID_Y:-128}"
LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"; PM_ACCUM="${PM_ACCUM:-1}"
REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.20}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; CURRENT_RESULT="${CURRENT_RESULT:-${PREFIX}_current.tsv}"
WARPCOOP_RESULT="${WARPCOOP_RESULT:-${PREFIX}_ab.tsv}"; WARPCOOP_SUMMARY="${WARPCOOP_SUMMARY:-${PREFIX}_ab_summary.tsv}"
SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$PROFILE_OUT")"

ORBIT_PRECTX_COMPACT="${ORBIT_PRECTX_COMPACT:-0}"
ORBIT_PRECTX_FLAT_BID="${ORBIT_PRECTX_FLAT_BID:-0}"
ORBIT_PRECTX_FLAT_BID_FUSED="${ORBIT_PRECTX_FLAT_BID_FUSED:-0}"
ORBIT_PRECTX_WARPCOOP="${ORBIT_PRECTX_WARPCOOP:-0}"
ORBITCTA_FLAT="${ORBITCTA_FLAT:-0}"
ORBITCTA_FLAT_CHUNK="${ORBITCTA_FLAT_CHUNK:-1}"
ORBITCTA_FLAT_BLOCKS_PER_SM="${ORBITCTA_FLAT_BLOCKS_PER_SM:-0}"
ORBIT_QUAD_MLP="${ORBIT_QUAD_MLP:-0}"
ORBIT_QUAD_OVERLAP_LOCAL="${ORBIT_QUAD_OVERLAP_LOCAL:-0}"
ORBIT_QUAD_LOCAL_DIRECT_MAX="${ORBIT_QUAD_LOCAL_DIRECT_MAX:-0}"
ORBIT_QUAD_SPARSE_DESC_MLP="${ORBIT_QUAD_SPARSE_DESC_MLP:-0}"
for n in ORBIT_PROFILE ORBIT_COL_ILP ORBIT_SPARSE64 ORBIT_SORTED ORBIT_CPASYNC_PAIR ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE; do
  [[ -n "${!n+x}" ]] || { echo "profile missing $n" >&2; exit 2; }
done
for x in ORBIT_SPARSE64 ORBIT_SORTED ORBIT_CPASYNC_PAIR ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE ORBIT_PRECTX_COMPACT ORBIT_PRECTX_FLAT_BID ORBIT_PRECTX_FLAT_BID_FUSED ORBIT_PRECTX_WARPCOOP ORBITCTA_FLAT ORBIT_QUAD_MLP ORBIT_QUAD_OVERLAP_LOCAL ORBIT_QUAD_SPARSE_DESC_MLP; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
[[ "$ORBIT_SORTED" == 0 ]] || { echo 'warpcoop refine requires unsorted orbit-CTA profile' >&2; exit 2; }
case "$ORBIT_COL_ILP" in 2|4) ;; *) echo 'bad ORBIT_COL_ILP' >&2; exit 2;; esac
case "$ORBITCTA_FLAT_CHUNK" in 1|2|4|8|16|32) ;; *) echo 'bad ORBITCTA_FLAT_CHUNK' >&2; exit 2;; esac
[[ "$ORBITCTA_FLAT_BLOCKS_PER_SM" =~ ^[0-9]+$ ]] || { echo 'bad ORBITCTA_FLAT_BLOCKS_PER_SM' >&2; exit 2; }
[[ "$ORBIT_QUAD_LOCAL_DIRECT_MAX" =~ ^[0-9]+$ ]] && (( ORBIT_QUAD_LOCAL_DIRECT_MAX <= 8 )) || { echo 'bad ORBIT_QUAD_LOCAL_DIRECT_MAX' >&2; exit 2; }
# Advanced selector must hand us an unrefined descriptor profile. This keeps the
# comparison rooted before qsd and prevents applying warpcoop twice.
[[ "$ORBIT_PRECTX_WARPCOOP" == 0 && "$ORBIT_QUAD_SPARSE_DESC_MLP" == 0 ]] || {
  echo 'PROFILE_IN must precede warpcoop/qsd refinement' >&2; exit 2;
}
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

# Snapshot the advanced winner settings before sourcing the quad sidecar.
CUR_COL_ILP="$ORBIT_COL_ILP"; CUR_SPARSE64="$ORBIT_SPARSE64"; CUR_CPASYNC_PAIR="$ORBIT_CPASYNC_PAIR"
CUR_PRE_F="$ORBIT_PRECTX_FORWARD"; CUR_PRE_R="$ORBIT_PRECTX_REVERSE"; CUR_PRE_C="$ORBIT_PRECTX_COMPACT"
CUR_BID="$ORBIT_PRECTX_FLAT_BID"; CUR_BID_FUSED="$ORBIT_PRECTX_FLAT_BID_FUSED"
CUR_FLAT="$ORBITCTA_FLAT"; CUR_CHUNK="$ORBITCTA_FLAT_CHUNK"; CUR_PSM="$ORBITCTA_FLAT_BLOCKS_PER_SM"
CUR_QUAD="$ORBIT_QUAD_MLP"; CUR_QOL="$ORBIT_QUAD_OVERLAP_LOCAL"; CUR_QLD="$ORBIT_QUAD_LOCAL_DIRECT_MAX"

if [[ "$CUR_CPASYNC_PAIR" == 1 ]]; then
  ARCH="$ARCH" NGPU=8 THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/current-cpasync.out" 2>"$LOGDIR/current-cpasync.err"
  grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/current-cpasync.out" || { echo 'current cp.async peer gate failed' >&2; exit 5; }
fi
if [[ "$CUR_PRE_C" == 1 ]]; then
  ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" >"$LOGDIR/current-prectx.out" 2>"$LOGDIR/current-prectx.err"
fi

CURRENT_BIN="$ONEESAN_BUILD_DIR/b300_orbit_warpcoop_refine_current_n21"
N=21 ARCH="$ARCH" OUT="$CURRENT_BIN" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$CUR_SPARSE64" DIRECTGATHER_SORT_RANKS=0 \
  RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR="$CUR_CPASYNC_PAIR" ORBITCTA_COL_ILP="$CUR_COL_ILP" \
  ORBITCTA_FLAT="$CUR_FLAT" ORBITCTA_FLAT_CHUNK="$CUR_CHUNK" QUAD_MLP="$CUR_QUAD" QUAD_OVERLAP_LOCAL="$CUR_QOL" QUAD_LOCAL_DIRECT_MAX="$CUR_QLD" QUAD_SPARSE_DESC_MLP=0 \
  PRECTX_FORWARD="$CUR_PRE_F" PRECTX_REVERSE="$CUR_PRE_R" PRECTX_COMPACT="$CUR_PRE_C" PRECTX_FLAT_BID="$CUR_BID" PRECTX_FLAT_BID_FUSED="$CUR_BID_FUSED" PRECTX_WARPCOOP=0 PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/current.build.out" 2>"$LOGDIR/current.build.err"

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }
printf 'repeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$CURRENT_RESULT"
for ((r=1;r<=REPEATS;++r)); do
  so="$LOGDIR/current_r${r}.out"; se="$LOGDIR/current_r${r}.err"; util="$LOGDIR/current_r${r}.util"
  if [[ "$CUR_FLAT" == 1 && "$CUR_PSM" == 0 ]]; then
    env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM \
      BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$CURRENT_BIN" 21 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
  elif [[ "$CUR_FLAT" == 1 ]]; then
    env -u BUCKET_ORBITCTA_FLAT_BLOCKS BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$CUR_PSM" \
      BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$CURRENT_BIN" 21 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
  else
    env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM \
      BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$CURRENT_BIN" 21 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
  fi
  pid=$!; sample "$pid" "$util" & sp=$!; set +e; wait "$pid"; rc=$?; set -e; wait "$sp" || true
  ((rc==0)) || { echo "current repeat=$r failed rc=$rc" >&2; exit "$rc"; }
  line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo 'current missing residue' >&2; exit 3; }
  residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "current residue=$residue expected=$EXPECT" >&2; exit 4; }
  detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"; fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
  [[ -n "$fh" && -n "$rh" ]] || { echo 'current missing HIGH timing' >&2; exit 6; }
  high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
  read -r ag am mm < <(awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' "$util")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$mm" >>"$CURRENT_RESULT"
done

# The advanced stage always writes this sidecar from its exact chunked-quad
# sweep, even if another branch won overall.
# shellcheck disable=SC1090
source "$QUAD_WINNER_ENV"
Q_CHUNK="$ORBITCTA_FLAT_CHUNK"; Q_PSM="$ORBITCTA_FLAT_BLOCKS_PER_SM"
case "$Q_CHUNK" in 2|4|8|16|32) ;; *) echo "bad quad sidecar chunk=$Q_CHUNK" >&2; exit 7;; esac
[[ "$Q_PSM" =~ ^[0-9]+$ ]] || { echo "bad quad sidecar psm=$Q_PSM" >&2; exit 7; }

POOL_ENV=(env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM)
if [[ "$Q_PSM" != 0 ]]; then POOL_ENV+=(BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$Q_PSM"); fi
"${POOL_ENV[@]}" N=21 ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" REPEATS="$REPEATS" \
  DIRECTGATHER_SPARSE64="$CUR_SPARSE64" ORBITCTA_FLAT_CHUNK="$Q_CHUNK" ORBITCTA_COL_ILP=4 PAIR_MLP=1 CPASYNC_PAIR=1 \
  QUAD_MLP=1 QUAD_OVERLAP_LOCAL=1 QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP=0 RANKFORMULA_MLP_WINDOW4=1 PM_ACCUM="$PM_ACCUM" \
  PREFIX="${PREFIX}_ab" RESULT="$WARPCOOP_RESULT" SUMMARY="$WARPCOOP_SUMMARY" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-prectx-warpcoop-ab.sh"

python3 - "$CURRENT_RESULT" "$WARPCOOP_RESULT" "$PROFILE_IN" "$PROFILE_OUT" "$SUMMARY" "$Q_CHUNK" "$Q_PSM" <<'PY'
import csv,statistics,sys,re
cur,ab,pin,pout,summary,qchunk,qpsm=sys.argv[1:]
def med(path, selector=None):
 rows=list(csv.DictReader(open(path),delimiter='\t'))
 if selector is not None: rows=[r for r in rows if r['variant']==selector]
 if not rows: raise SystemExit(f'no rows {path} selector={selector}')
 return {'wall_s':statistics.median(float(r['wall_s']) for r in rows),
         'high_s':statistics.median(float(r['high_s']) for r in rows)}
current=med(cur); serial=med(ab,'serial'); coop=med(ab,'warpcoop')
candidates=[('current',current),('quad_compact_serial',serial),('quad_compact_warpcoop',coop)]
best_name,best=min(candidates,key=lambda z:z[1]['wall_s'])
kv={};order=[]
for line in open(pin):
 s=line.strip()
 if not s or s.startswith('#') or '=' not in s: continue
 k,v=s.split('=',1)
 if k not in kv: order.append(k)
 kv[k]=v.strip('"')
kv.setdefault('ORBIT_PRECTX_WARPCOOP','0')
if 'ORBIT_PRECTX_WARPCOOP' not in order: order.append('ORBIT_PRECTX_WARPCOOP')
if best_name!='current':
 kv.update({'ORBITCTA_FLAT':'1','ORBITCTA_FLAT_CHUNK':qchunk,'ORBITCTA_FLAT_BLOCKS_PER_SM':qpsm,
            'ORBIT_COL_ILP':'4','ORBIT_CPASYNC_PAIR':'1',
            'ORBIT_PRECTX_FORWARD':'1','ORBIT_PRECTX_REVERSE':'1','ORBIT_PRECTX_COMPACT':'1',
            'ORBIT_PRECTX_FLAT_BID':'0','ORBIT_PRECTX_FLAT_BID_FUSED':'0',
            'ORBIT_PRECTX_WARPCOOP':'1' if best_name=='quad_compact_warpcoop' else '0',
            'ORBIT_QUAD_MLP':'1','ORBIT_QUAD_OVERLAP_LOCAL':'1','ORBIT_QUAD_LOCAL_DIRECT_MAX':'0','ORBIT_QUAD_SPARSE_DESC_MLP':'0'})
else:
 kv['ORBIT_PRECTX_WARPCOOP']='0'
kv['ORBIT_WARPCOOP_PROFILE']=best_name
if 'ORBIT_WARPCOOP_PROFILE' not in order: order.append('ORBIT_WARPCOOP_PROFILE')
if best_name!='current':
 root=re.sub(r'_wc_.*$','',kv.get('ORBIT_PROFILE','orbit'))
 kv['ORBIT_PROFILE']=root+'_wc_'+best_name
with open(pout,'w') as f:
 f.write('# generated by b300-hbm-profile-refine-orbit-warpcoop21.sh\n')
 for k in order:
  v=kv[k]
  if k=='CANDIDATES': f.write(f'{k}="{v}"\n')
  else: f.write(f'{k}={v}\n')
with open(summary,'w') as f:
 f.write('candidate\twall_s\thigh_s\n')
 for name,z in sorted(candidates,key=lambda x:x[1]['wall_s']): f.write(f'{name}\t{z["wall_s"]:.9f}\t{z["high_s"]:.9f}\n')
for name,z in sorted(candidates,key=lambda x:x[1]['wall_s']): print('WARPCOOP_ORBIT',name,f'wall_s={z["wall_s"]:.6f}',f'high_s={z["high_s"]:.6f}')
print('BEST_ORBIT_WARPCOOP='+best_name,f'wall_s={best["wall_s"]:.6f}',f'profile_file={pout}',f'summary={summary}')
PY
cat "$SUMMARY"
cat "$PROFILE_OUT"
echo "orbit warpcoop refine OK input=$PROFILE_IN output=$PROFILE_OUT quad_sidecar=$QUAD_WINNER_ENV summary=$SUMMARY" >&2
