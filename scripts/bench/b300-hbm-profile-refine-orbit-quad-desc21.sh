#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

PROFILE_IN="${PROFILE_IN:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_refined_desc21.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_hbm_profile_orbit_quad_desc21}"
[[ -f "$PROFILE_IN" ]] || { echo "missing PROFILE_IN=$PROFILE_IN" >&2; exit 2; }
# shellcheck disable=SC1090
source "$PROFILE_IN"

N=21; MOD=4294967291; EXPECT=998035516; NGPU=8
ARCH="${ARCH:-sm_103}"; TARGET_MIB="${TARGET_MIB:-16384}"; MAX_WINDOW="${MAX_WINDOW:-14}"
THREADS="${BUCKET_THREADS:-256}"; LOW_GX="${BUCKET_LOW_GRID_X:-16}"; LOW_GY="${BUCKET_LOW_GRID_Y:-8}"
PM_ACCUM="${PM_ACCUM:-1}"; REPEATS="${REPEATS:-1}"; SAMPLE_INTERVAL="${SAMPLE_INTERVAL:-0.20}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; RESOURCE="${RESOURCE:-${PREFIX}_ptxas.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$PROFILE_OUT")"

for n in ORBIT_COL_ILP ORBIT_SPARSE64 ORBIT_CPASYNC_PAIR ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE ORBIT_PRECTX_COMPACT ORBITCTA_FLAT ORBITCTA_FLAT_CHUNK ORBITCTA_FLAT_BLOCKS_PER_SM ORBIT_QUAD_MLP ORBIT_QUAD_OVERLAP_LOCAL ORBIT_QUAD_LOCAL_DIRECT_MAX; do
  [[ -n "${!n+x}" ]] || { echo "profile missing $n" >&2; exit 2; }
done
for x in ORBIT_SPARSE64 ORBIT_CPASYNC_PAIR ORBIT_PRECTX_FORWARD ORBIT_PRECTX_REVERSE ORBIT_PRECTX_COMPACT ORBITCTA_FLAT ORBIT_QUAD_MLP ORBIT_QUAD_OVERLAP_LOCAL; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
[[ "$ORBIT_QUAD_MLP" == 1 && "$ORBIT_QUAD_OVERLAP_LOCAL" == 1 ]] || { echo 'descriptor refine requires quad overlap-local winner' >&2; exit 2; }
[[ "$ORBIT_SPARSE64" == 1 ]] || { echo 'descriptor refine only applies to sparse64 winner' >&2; exit 2; }
[[ "$ORBIT_CPASYNC_PAIR" == 1 && "$ORBIT_COL_ILP" == 4 ]] || { echo 'descriptor refine requires cp.async quad ILP4' >&2; exit 2; }
[[ "$ORBITCTA_FLAT" == 1 ]] && (( ORBITCTA_FLAT_CHUNK > 1 )) || { echo 'descriptor refine requires chunked flat orbit CTA' >&2; exit 2; }
[[ "$ORBITCTA_FLAT_BLOCKS_PER_SM" =~ ^[0-9]+$ ]] || { echo 'bad ORBITCTA_FLAT_BLOCKS_PER_SM' >&2; exit 2; }
[[ "$ORBIT_QUAD_LOCAL_DIRECT_MAX" == 0 ]] || { echo 'descriptor refine expects QOL direct local path' >&2; exit 2; }
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

bash "$ONEESAN_ROOT/scripts/bench/directgather64-quad-proof.sh" >"$LOGDIR/quad-proof.out" 2>"$LOGDIR/quad-proof.err"
ARCH="$ARCH" NGPU=8 THREADS="$THREADS" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-remote-peer-microprobe.sh" >"$LOGDIR/cpasync-peer.out" 2>"$LOGDIR/cpasync-peer.err"
grep -q 'cp_async_remote_peer=OK exact=OK' "$LOGDIR/cpasync-peer.out" || { echo 'cp.async remote-peer gate failed' >&2; exit 5; }
if [[ "$ORBIT_PRECTX_COMPACT" == 1 ]]; then
  ARCH="$ARCH" PM_ACCUM="$PM_ACCUM" bash "$ONEESAN_ROOT/scripts/bench/compact-prectx-selftest.sh" >"$LOGDIR/compact-prectx.out" 2>"$LOGDIR/compact-prectx.err"
fi

printf 'desc_mlp\trepeat\tresidue\twall_s\tforward_high_s\treverse_high_s\thigh_s\tavg_gpu_util_pct\tavg_memctrl_util_pct\tmax_memctrl_util_pct\n' >"$RESULT"
printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$RESOURCE"
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample(){ local pid="$1" out="$2"; : >"$out"; while kill -0 "$pid" 2>/dev/null; do nvidia-smi --query-gpu=utilization.gpu,utilization.memory --format=csv,noheader,nounits 2>/dev/null | awk -F',' '{g=$1+0;m=$2+0;sg+=g;sm+=m;if(m>mm)mm=m;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' >>"$out" || true; sleep "$SAMPLE_INTERVAL"; done; }

for qsd in 0 1; do
  bin="$ONEESAN_BUILD_DIR/b300_orbit_quad_desc${qsd}_n21"
  N=21 ARCH="$ARCH" OUT="$bin" PM_ACCUM="$PM_ACCUM" DIRECTGATHER64=1 DIRECTGATHER_SPARSE64=1 DIRECTGATHER_SORT_RANKS=0 \
    RANKFORMULA_MLP_WINDOW4=1 PAIR_MLP=1 CPASYNC_PAIR=1 QUAD_MLP=1 QUAD_OVERLAP_LOCAL=1 QUAD_LOCAL_DIRECT_MAX=0 QUAD_SPARSE_DESC_MLP="$qsd" \
    ORBITCTA_FLAT=1 ORBITCTA_FLAT_CHUNK="$ORBITCTA_FLAT_CHUNK" ORBITCTA_COL_ILP=4 \
    PRECTX_FORWARD="$ORBIT_PRECTX_FORWARD" PRECTX_REVERSE="$ORBIT_PRECTX_REVERSE" PRECTX_COMPACT="$ORBIT_PRECTX_COMPACT" PRECTX_FLAT_BID=0 PRECTX_FLAT_BID_FUSED=0 PTXAS_VERBOSE=1 \
    bash "$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta.sh" >"$LOGDIR/qsd${qsd}.build.out" 2>"$LOGDIR/qsd${qsd}.build.err"
  python3 "$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py" "$LOGDIR/qsd${qsd}.build.err" --label "qsd${qsd}" >>"$RESOURCE" || true

  for ((r=1;r<=REPEATS;++r)); do
    so="$LOGDIR/qsd${qsd}_r${r}.out"; se="$LOGDIR/qsd${qsd}_r${r}.err"; util="$LOGDIR/qsd${qsd}_r${r}.util"
    if [[ "$ORBITCTA_FLAT_BLOCKS_PER_SM" == 0 ]]; then
      env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM \
        BUCKET_THREADS="$THREADS" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
        "$bin" 21 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
    else
      env -u BUCKET_ORBITCTA_FLAT_BLOCKS BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM" \
        BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" \
        "$bin" 21 "$TARGET_MIB" "$MAX_WINDOW" 8 "$MOD" >"$so" 2>"$se" &
    fi
    pid=$!; sample "$pid" "$util" & sp=$!; set +e; wait "$pid"; rc=$?; set -e; wait "$sp" || true
    (( rc == 0 )) || { echo "qsd$qsd repeat=$r failed rc=$rc" >&2; exit "$rc"; }
    line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "qsd$qsd missing residue" >&2; exit 3; }
    residue="$(field residue "$line")"; [[ "$residue" == "$EXPECT" ]] || { echo "qsd$qsd residue=$residue expected=$EXPECT" >&2; exit 4; }
    detail="$(grep 'snake_onepass_graph_batch modulus=' "$se" | tail -n1 || true)"; fh="$(field forward_high_s "$detail")"; rh="$(field reverse_high_s "$detail")"
    [[ -n "$fh" && -n "$rh" ]] || { echo "qsd$qsd missing HIGH timing" >&2; exit 6; }
    high="$(python3 - "$fh" "$rh" <<'PY'
import sys
print(f'{float(sys.argv[1])+float(sys.argv[2]):.9f}')
PY
)"
    read -r ag am mm < <(awk '{sg+=$1;sm+=$2;if($3>mm)mm=$3;n++}END{if(n)printf "%.6f %.6f %d\n",sg/n,sm/n,mm;else print "NA NA NA"}' "$util")
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$qsd" "$r" "$residue" "$(field wall_s "$line")" "$fh" "$rh" "$high" "$ag" "$am" "$mm" >>"$RESULT"
  done
done

cat "$RESULT"
python3 - "$RESULT" "$PROFILE_IN" "$PROFILE_OUT" <<'PY'
import csv,statistics,sys
result,profile_in,profile_out=sys.argv[1:]
rows=list(csv.DictReader(open(result),delimiter='\t')); by={}
for r in rows: by.setdefault(int(r['desc_mlp']),[]).append(r)
if set(by)!={0,1}: raise SystemExit(f'missing variants {set(by)}')
res={k:{r['residue'] for r in g} for k,g in by.items()}
if any(len(v)!=1 for v in res.values()) or next(iter(res[0]))!=next(iter(res[1])): raise SystemExit(f'RESIDUE MISMATCH {res}')
stats={}
for k,g in by.items():
 stats[k]={
  'wall':statistics.median(float(r['wall_s']) for r in g),
  'high':statistics.median(float(r['high_s']) for r in g),
  'mc':statistics.median(float(r['avg_memctrl_util_pct']) for r in g if r['avg_memctrl_util_pct']!='NA')}
winner=min(stats,key=lambda k:stats[k]['wall'])
kv={};order=[]
for line in open(profile_in):
 s=line.strip()
 if not s or s.startswith('#') or '=' not in s: continue
 k,v=s.split('=',1)
 if k not in kv: order.append(k)
 kv[k]=v.strip('"')
kv['ORBIT_QUAD_SPARSE_DESC_MLP']=str(winner)
if 'ORBIT_QUAD_SPARSE_DESC_MLP' not in order: order.append('ORBIT_QUAD_SPARSE_DESC_MLP')
if winner:
 kv['ORBIT_PROFILE']=kv.get('ORBIT_PROFILE','orbit')+'_qsd1'
with open(profile_out,'w') as f:
 f.write('# generated by b300-hbm-profile-refine-orbit-quad-desc21.sh\n')
 for k in order:
  v=kv[k]
  if k=='CANDIDATES': f.write(f'{k}="{v}"\n')
  else: f.write(f'{k}={v}\n')
for k in (0,1): print(f'QSD{k}',f"wall_s={stats[k]['wall']:.6f}",f"high_s={stats[k]['high']:.6f}",f"mc_avg_pct={stats[k]['mc']:.3f}")
print('BEST_QUAD_SPARSE_DESC_MLP='+str(winner),f"wall_speedup={stats[0]['wall']/stats[1]['wall']:.6f}",f"high_speedup={stats[0]['high']/stats[1]['high']:.6f}",f'profile_file={profile_out}')
PY
cat "$PROFILE_OUT"
echo "orbit quad sparse descriptor refine OK input=$PROFILE_IN output=$PROFILE_OUT result=$RESULT resources=$RESOURCE" >&2
