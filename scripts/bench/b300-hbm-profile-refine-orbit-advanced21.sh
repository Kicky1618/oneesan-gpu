#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROFILE_IN="${PROFILE_IN:-$ONEESAN_ROOT/work/b300_hbm_profile_scheduler21.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_advanced21}"
[[ -f "$PROFILE_IN" ]] || { echo "missing PROFILE_IN=$PROFILE_IN" >&2; exit 2; }
# shellcheck disable=SC1090
source "$PROFILE_IN"

N=21; MOD=4294967291; EXPECT=998035516; NGPU=8
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; ORBIT_GY="${BUCKET_ORBITCTA_GRID_Y:-128}"
LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"; PM_ACCUM="${PM_ACCUM:-1}"
REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.20}"
QUAD_CHUNKS="${ADV_QUAD_CHUNKS:-2 4 8}"; QUAD_POOLS="${ADV_QUAD_POOLS:-auto 1 2 4}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; CURRENT_RESULT="${CURRENT_RESULT:-${PREFIX}_current.tsv}"
BID_PREFIX="${BID_PREFIX:-${PREFIX}_bid}"; FUSED_PREFIX="${FUSED_PREFIX:-${PREFIX}_fused}"; QUAD_PREFIX="${QUAD_PREFIX:-${PREFIX}_quad}"
BID_RESULT="${BID_RESULT:-${BID_PREFIX}.tsv}"; FUSED_RESULT="${FUSED_RESULT:-${FUSED_PREFIX}.tsv}"
QUAD_WINNER_ENV="${QUAD_WINNER_ENV:-${QUAD_PREFIX}_winner.env}"; SUMMARY="${SUMMARY:-${PREFIX}_summary.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$PROFILE_OUT")"

ORBIT_PRECTX_COMPACT="${ORBIT_PRECTX_COMPACT:-0}"
ORBITCTA_FLAT="${ORBITCTA_FLAT:-0}"
ORBITCTA_FLAT_CHUNK="${ORBITCTA_FLAT_CHUNK:-1}"
ORBITCTA_FLAT_BLOCKS_PER_SM="${ORBITCTA_FLAT_BLOCKS_PER_SM:-0}"
ORBIT_QUAD_MLP="${ORBIT_QUAD_MLP:-0}"
ORBIT_QUAD_OVERLAP_LOCAL="${ORBIT_QUAD_OVERLAP_LOCAL:-0}"
ORBIT_QUAD_LOCAL_DIRECT_MAX="${ORBIT_QUAD_LOCAL_DIRECT_MAX:-0}"
ORBIT_PRECTX_FLAT_BID="${ORBIT_PRECTX_FLAT_BID:-0}"
ORBIT_PRECTX_FLAT_BID_FUSED="${ORBIT_PRECTX_FLAT_BID_FUSED:-0}"
for n in ORBIT_PROFILE ORBIT_COL_ILP ORBIT_SPARSE64 ORBIT_SORTED ORBIT_CPASYNC_PAIR ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE; do
  [[ -n "${!n+x}" ]] || { echo "profile missing $n" >&2; exit 2; }
done
for x in ORBIT_SPARSE64 ORBIT_SORTED ORBIT_CPASYNC_PAIR ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE ORBIT_PRECTX_COMPACT ORBITCTA_FLAT ORBIT_QUAD_MLP ORBIT_QUAD_OVERLAP_LOCAL ORBIT_PRECTX_FLAT_BID ORBIT_PRECTX_FLAT_BID_FUSED; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
[[ "$ORBIT_SORTED" == 0 ]] || { echo 'advanced orbit refine requires unsorted orbit-CTA profile' >&2; exit 2; }
case "$ORBIT_COL_ILP" in 1|2|4) ;; *) echo 'bad ORBIT_COL_ILP' >&2; exit 2;; esac
case "$ORBITCTA_FLAT_CHUNK" in 1|2|4|8|16|32) ;; *) echo 'bad ORBITCTA_FLAT_CHUNK' >&2; exit 2;; esac
[[ "$ORBITCTA_FLAT_BLOCKS_PER_SM" =~ ^[0-9]+$ ]] || { echo 'bad ORBITCTA_FLAT_BLOCKS_PER_SM' >&2; exit 2; }
[[ "$ORBIT_QUAD_LOCAL_DIRECT_MAX" =~ ^[0-9]+$ ]] && (( ORBIT_QUAD_LOCAL_DIRECT_MAX <= 8 )) || { echo 'bad ORBIT_QUAD_LOCAL_DIRECT_MAX' >&2; exit 2; }
# This stage is intentionally rooted at the scheduler winner. Advanced flags in
# the input would make branch timing ambiguous and are rejected.
[[ "$ORBITCTA_FLAT_CHUNK" == 1 && "$ORBIT_QUAD_MLP" == 0 && "$ORBIT_QUAD_OVERLAP_LOCAL" == 0 && "$ORBIT_QUAD_LOCAL_DIRECT_MAX" == 0 && "$ORBIT_PRECTX_FLAT_BID" == 0 && "$ORBIT_PRECTX_FLAT_BID_FUSED" == 0 ]] || {
  echo 'PROFILE_IN must be the pre-advanced scheduler winner (chunk=1, quad=0, flat-bid=0)' >&2; exit 2;
}
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

CURRENT_BIN="$ONEESAN_BUILD_DIR/b300_orbit_advanced_current_n21"
N=21 ARCH="$ARCH" OUT="$CURRENT_BIN" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64="$ORBIT_SPARSE64" DIRECTGATHER_SORT_RANKS=0 \
  RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR="$ORBIT_CPASYNC_PAIR" ORBITCTA_COL_ILP="$ORBIT_COL_ILP" \
  ORBITCTA_FLAT="$ORBITCTA_FLAT" ORBITCTA_FLAT_CHUNK=1 QUAD_MLP=0 QUAD_OVERLAP_LOCAL=0 QUAD_LOCAL_DIRECT_MAX=0 \
  PRECTX_FORWARD="$ORBIT_PRECTX_FORWARD" PRECTX_REVERSE="$ORBIT_PRECTX_REVERSE" PRECTX_COMPACT="$ORBIT_PRECTX_COMPACT" PRECTX_FLAT_BID=0 PRECTX_FLAT_BID_FUSED=0 PTXAS_VERBOSE=1 \
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
    env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM \
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

# Flat-bid and compact-both branches use the scheduler winner's pool request
# when it was a tuned flat pool. If the scheduler winner was ordinary or used
# occupancy-derived flat pools, these branches use occupancy mode.
POOL_ARGS=(env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM)
if [[ "$ORBITCTA_FLAT" == 1 && "$ORBITCTA_FLAT_BLOCKS_PER_SM" != 0 ]]; then
  POOL_ARGS+=(BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM")
fi
"${POOL_ARGS[@]}" N=21 ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" REPEATS="$REPEATS" \
  DIRECTGATHER_SPARSE64="$ORBIT_SPARSE64" ORBITCTA_COL_ILP="$ORBIT_COL_ILP" PAIR_MLP=1 CPASYNC_PAIR="$ORBIT_CPASYNC_PAIR" RANKFORMULA_MLP_WINDOW4=1 PM_ACCUM="$PM_ACCUM" \
  PREFIX="$BID_PREFIX" RESULT="$BID_RESULT" bash "$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-prectx-bid-ab.sh"

"${POOL_ARGS[@]}" N=21 ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" REPEATS="$REPEATS" \
  DIRECTGATHER_SPARSE64="$ORBIT_SPARSE64" ORBITCTA_COL_ILP="$ORBIT_COL_ILP" PAIR_MLP=1 CPASYNC_PAIR="$ORBIT_CPASYNC_PAIR" RANKFORMULA_MLP_WINDOW4=1 PM_ACCUM="$PM_ACCUM" \
  PREFIX="$FUSED_PREFIX" RESULT="$FUSED_RESULT" bash "$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-prectx-fused-ab.sh"

N=21 ARCH="$ARCH" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" THREADS="$THREADS" CHUNKS="$QUAD_CHUNKS" POOLS="$QUAD_POOLS" REPEATS="$REPEATS" \
  SPARSE64="$ORBIT_SPARSE64" PRECTX_FORWARD="$ORBIT_PRECTX_FORWARD" PRECTX_REVERSE="$ORBIT_PRECTX_REVERSE" PRECTX_COMPACT="$ORBIT_PRECTX_COMPACT" \
  LOW_GX="$LOW_GX" LOW_GY="$LOW_GY" PREFIX="$QUAD_PREFIX" WINNER_ENV="$QUAD_WINNER_ENV" \
  bash "$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-quad-sweep.sh"
[[ -f "$QUAD_WINNER_ENV" ]] || { echo "missing quad winner env=$QUAD_WINNER_ENV" >&2; exit 7; }

python3 - "$CURRENT_RESULT" "$BID_RESULT" "$FUSED_RESULT" "$QUAD_WINNER_ENV" "$PROFILE_IN" "$PROFILE_OUT" "$SUMMARY" "$ORBITCTA_FLAT" "$ORBITCTA_FLAT_BLOCKS_PER_SM" <<'PY'
import csv,statistics,sys,re
cur,bid,fused,qenv,pin,pout,summary,current_flat,current_psm=sys.argv[1:]
def median_rows(path, key, value):
 rows=list(csv.DictReader(open(path),delimiter='\t')); g=[r for r in rows if r[key]==value]
 if not g: raise SystemExit(f'no rows {path} {key}={value}')
 return {k:statistics.median(float(r[k]) for r in g) for k in ('wall_s','high_s','forward_high_s','reverse_high_s')}
rows=list(csv.DictReader(open(cur),delimiter='\t'))
current={k:statistics.median(float(r[k]) for r in rows) for k in ('wall_s','high_s','forward_high_s','reverse_high_s')}
plain=median_rows(bid,'mode','base')
bidsplit=median_rows(fused,'fused','0')
bidfused=median_rows(fused,'fused','1')
q={}
for line in open(qenv):
 s=line.strip()
 if s and not s.startswith('#') and '=' in s:
  k,v=s.split('=',1);q[k]=v.strip('"')
quad={'wall_s':float(q['ORBIT_QUAD_WALL_S']),'high_s':float(q['ORBIT_QUAD_HIGH_S'])}
candidates=[('current',current),('flat_compact_both',plain),('flat_bid_split',bidsplit),('flat_bid_fused',bidfused),('chunked_quad',quad)]
best_name,best=min(candidates,key=lambda z:z[1]['wall_s'])
kv={};order=[]
for line in open(pin):
 s=line.strip()
 if not s or s.startswith('#') or '=' not in s: continue
 k,v=s.split('=',1)
 if k not in kv: order.append(k)
 kv[k]=v.strip('"')
advanced_defaults={
 'ORBITCTA_FLAT_CHUNK':'1','ORBIT_QUAD_MLP':'0','ORBIT_QUAD_OVERLAP_LOCAL':'0','ORBIT_QUAD_LOCAL_DIRECT_MAX':'0',
 'ORBIT_PRECTX_FLAT_BID':'0','ORBIT_PRECTX_FLAT_BID_FUSED':'0'
}
for k,v in advanced_defaults.items():
 kv.setdefault(k,v)
 if k not in order: order.append(k)
used_psm=current_psm if current_flat=='1' and current_psm!='0' else '0'
if best_name=='flat_compact_both':
 kv.update({'ORBITCTA_FLAT':'1','ORBITCTA_FLAT_CHUNK':'1','ORBITCTA_FLAT_BLOCKS_PER_SM':used_psm,
            'ORBIT_PRECTX_FORWARD':'1','ORBIT_PRECTX_REVERSE':'1','ORBIT_PRECTX_COMPACT':'1',
            'ORBIT_PRECTX_FLAT_BID':'0','ORBIT_PRECTX_FLAT_BID_FUSED':'0',
            'ORBIT_QUAD_MLP':'0','ORBIT_QUAD_OVERLAP_LOCAL':'0','ORBIT_QUAD_LOCAL_DIRECT_MAX':'0'})
elif best_name in ('flat_bid_split','flat_bid_fused'):
 kv.update({'ORBITCTA_FLAT':'1','ORBITCTA_FLAT_CHUNK':'1','ORBITCTA_FLAT_BLOCKS_PER_SM':used_psm,
            'ORBIT_PRECTX_FORWARD':'1','ORBIT_PRECTX_REVERSE':'1','ORBIT_PRECTX_COMPACT':'1',
            'ORBIT_PRECTX_FLAT_BID':'1','ORBIT_PRECTX_FLAT_BID_FUSED':'1' if best_name=='flat_bid_fused' else '0',
            'ORBIT_QUAD_MLP':'0','ORBIT_QUAD_OVERLAP_LOCAL':'0','ORBIT_QUAD_LOCAL_DIRECT_MAX':'0'})
elif best_name=='chunked_quad':
 kv.update({'ORBITCTA_FLAT':'1','ORBITCTA_FLAT_CHUNK':q['ORBITCTA_FLAT_CHUNK'],'ORBITCTA_FLAT_BLOCKS_PER_SM':q['ORBITCTA_FLAT_BLOCKS_PER_SM'],
            'ORBIT_CPASYNC_PAIR':q['ORBIT_CPASYNC_PAIR'],'ORBIT_COL_ILP':q['ORBIT_COL_ILP'],
            'ORBIT_PRECTX_FLAT_BID':'0','ORBIT_PRECTX_FLAT_BID_FUSED':'0',
            'ORBIT_QUAD_MLP':q['ORBIT_QUAD_MLP'],'ORBIT_QUAD_OVERLAP_LOCAL':q['ORBIT_QUAD_OVERLAP_LOCAL'],'ORBIT_QUAD_LOCAL_DIRECT_MAX':q['ORBIT_QUAD_LOCAL_DIRECT_MAX']})
for k in ('ORBITCTA_FLAT','ORBITCTA_FLAT_BLOCKS_PER_SM'):
 if k not in order: order.append(k)
kv['ORBIT_ADVANCED_PROFILE']=best_name
if 'ORBIT_ADVANCED_PROFILE' not in order: order.append('ORBIT_ADVANCED_PROFILE')
root=re.sub(r'_adv_.*$','',kv['ORBIT_PROFILE'])
kv['ORBIT_PROFILE']=root+'_adv_'+best_name
with open(pout,'w') as f:
 f.write('# generated by b300-hbm-profile-refine-orbit-advanced21.sh\n')
 for k in order:
  v=kv[k]
  if k=='CANDIDATES': f.write(f'{k}="{v}"\n')
  else: f.write(f'{k}={v}\n')
with open(summary,'w') as f:
 f.write('candidate\twall_s\thigh_s\n')
 for name,z in sorted(candidates,key=lambda x:x[1]['wall_s']): f.write(f'{name}\t{z["wall_s"]:.9f}\t{z["high_s"]:.9f}\n')
for name,z in sorted(candidates,key=lambda x:x[1]['wall_s']): print('ADV_ORBIT',name,f'wall_s={z["wall_s"]:.6f}',f'high_s={z["high_s"]:.6f}')
print('BEST_ORBIT_ADVANCED='+best_name,f'wall_s={best["wall_s"]:.6f}',f'profile_file={pout}',f'summary={summary}')
PY
cat "$SUMMARY"
cat "$PROFILE_OUT"
echo "advanced orbit refine OK input=$PROFILE_IN output=$PROFILE_OUT summary=$SUMMARY" >&2
