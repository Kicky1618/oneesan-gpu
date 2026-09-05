#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; (($#==0)) || shift
[[ "$N" == 27 ]] || { echo 'forced-set single-pass race targets n=27' >&2; exit 2; }
: "${FORCED_SET_FILE:?FORCED_SET_FILE is required}"
[[ -s "$FORCED_SET_FILE" ]] || { echo "missing forced set: $FORCED_SET_FILE" >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
[[ -f "$PROFILE_FILE" ]] || { echo "missing profile: $PROFILE_FILE" >&2; exit 2; }
ARCH="${ARCH:-native}"; PRIME="${SMOKE_PRIME:-4294967291}"; MAX_WINDOW="${MAX_WINDOW:-14}"
FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}"; BUCKET_TARGET_MIB="${BUCKET_TARGET_MIB:-16384}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"; SELECT_ONLY="${SELECT_ONLY:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_forced_set_profiled_once_n27}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; WORK_ROOT="${WORK_ROOT:-$ONEESAN_ROOT/work}"
BUCKET_ENV="${BUCKET_ENV:-${PREFIX}.buckets.env}"; BUCKET_PREFIX="${BUCKET_PREFIX:-${PREFIX}.buckets}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
for x in REBUILD_BUCKETS SELECT_ONLY; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || exit 2; done
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

FORCED_CANON="$LOGDIR/forced-set.tsv"
FORCED_COUNT="$(python3 - "$FORCED_SET_FILE" "$FORCED_CANON" <<'PY'
import os,re,sys
src,dst=sys.argv[1:3]
lines=[x.rstrip('\n') for x in open(src) if x.strip() and not x.lstrip().startswith('#')]
if lines and lines[0].split('\t')[:3]==['label','binary','threads']: lines=lines[1:]
out=[];labels=set();bins=set()
for line in lines:
    a=line.split('\t')
    if len(a)<3: raise SystemExit('forced row needs label,binary,threads: '+line)
    label,binary,ts=a[:3]
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',label): raise SystemExit('unsafe label '+label)
    if label in labels: raise SystemExit('duplicate label '+label)
    binary=os.path.realpath(binary)
    if not os.path.isfile(binary) or not os.access(binary,os.X_OK): raise SystemExit('binary not executable '+binary)
    if binary in bins: raise SystemExit('duplicate binary '+binary)
    try:t=int(ts)
    except ValueError: raise SystemExit('bad threads '+ts)
    if t<32 or t>1024 or t%32: raise SystemExit('threads must be warp multiple 32..1024')
    labels.add(label);bins.add(binary);out.append((label,binary,t))
if not out: raise SystemExit('no forced candidates')
with open(dst,'w') as f:
    f.write('label\tbinary\tthreads\n')
    for r in out:f.write('\t'.join(map(str,r))+'\n')
print(len(out))
PY
)"
[[ "$FORCED_COUNT" =~ ^[1-9][0-9]*$ ]] || { echo 'bad forced candidate count' >&2; exit 3; }

# Build current profiled bucket binaries only; no complete-prime preselection.
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" REBUILD="$REBUILD_BUCKETS" PREFIX="$BUCKET_PREFIX" OUT_ENV="$BUCKET_ENV" \
  bash "$ONEESAN_ROOT/scripts/build/b300-profiled-buckets-only.sh" >"$LOGDIR/buckets.build.out" 2>"$LOGDIR/buckets.build.err"
[[ -s "$BUCKET_ENV" ]] || exit 3
# shellcheck disable=SC1090
source "$BUCKET_ENV"
[[ "${B300_PROFILED_BUCKETS_BUILD_ONLY:-0}" == 1 && -x "$WARP_BIN" && -x "$ORBIT_BIN" ]] || exit 3
[[ "$(sha256sum "$PROFILE_FILE" | awk '{print $1}')" == "$PROFILE_SHA256" ]] || { echo 'profile changed during bucket build' >&2; exit 3; }
# shellcheck disable=SC1090
source "$PROFILE_FILE"
THREADS="${BUCKET_THREADS:-256}"; WARP_GX="${WARP_GX:-32}"; WARP_GY="${WARP_GY:-8}"; ORBIT_GY="${ORBIT_GY:-128}"; LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
ORBITCTA_FLAT="${ORBITCTA_FLAT:-0}"; ORBITCTA_FLAT_BLOCKS_PER_SM="${ORBITCTA_FLAT_BLOCKS_PER_SM:-0}"

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
mc_summary(){ awk '$1~/^[0-9]+$/&&$3~/^[0-9]+$/{s+=$3;n++;if($3>m)m=$3}END{if(n)printf "%.3f\t%d\t%d\n",s/n,m,n;else print "NA\tNA\t0"}' "$1"; }
monitor_start(){ local f="$1"; :>"$f"; nvidia-smi dmon -s u -d 1 >"$f" 2>/dev/null & MON_PID=$!; }
monitor_stop(){ kill "$MON_PID" 2>/dev/null || true; wait "$MON_PID" 2>/dev/null || true; }
printf 'family\tbackend\tprofile\tbinary\tthreads\tstatus\tresidue\twall_s\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"

run_forced(){
  local label="$1" bin="$2" t="$3" out="$LOGDIR/forced_${label}.out" err="$LOGDIR/forced_${label}.err" dm="$LOGDIR/forced_${label}.dmon"
  monitor_start "$dm"; set +e; GRIDFP_THREADS="$t" "$bin" 27 "$FORCED_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$out" 2>"$err"; local rc=$?; set -e; monitor_stop
  if ((rc)); then printf 'forced\t%s\tt%s\t%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\t0\n' "$label" "$t" "$bin" "$t" "$rc" >>"$RESULT"; return; fi
  local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$out" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$label missing forced result" >&2; exit 4; }
  local avg mx ns; IFS=$'\t' read -r avg mx ns <<<"$(mc_summary "$dm")"
  printf 'forced\t%s\tt%s\t%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\n' "$label" "$t" "$bin" "$t" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" "$ns" >>"$RESULT"
}
run_warp(){
  local out="$LOGDIR/warp.out" err="$LOGDIR/warp.err" dm="$LOGDIR/warp.dmon"; monitor_start "$dm"; set +e
  BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$WARP_BIN" 27 "$BUCKET_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$out" 2>"$err"; local rc=$?; set -e; monitor_stop
  if ((rc)); then printf 'bucket\twarp_tuned\t%s\t%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\t0\n' "$WARP_PROFILE" "$WARP_BIN" "$THREADS" "$rc" >>"$RESULT"; return; fi
  local line="$(grep '^residue=' "$out" | tail -n1 || true)"; [[ -n "$line" ]] || { echo 'warp result missing' >&2; exit 4; }; local avg mx ns; IFS=$'\t' read -r avg mx ns <<<"$(mc_summary "$dm")"
  printf 'bucket\twarp_tuned\t%s\t%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\n' "$WARP_PROFILE" "$WARP_BIN" "$THREADS" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" "$ns" >>"$RESULT"
}
run_orbit(){
  local out="$LOGDIR/orbit.out" err="$LOGDIR/orbit.err" dm="$LOGDIR/orbit.dmon"; monitor_start "$dm"; set +e
  if [[ "$ORBITCTA_FLAT" == 1 && "$ORBITCTA_FLAT_BLOCKS_PER_SM" == 0 ]]; then
    env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$ORBIT_BIN" 27 "$BUCKET_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$out" 2>"$err"
  else
    env -u BUCKET_ORBITCTA_FLAT_BLOCKS BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$ORBIT_BIN" 27 "$BUCKET_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$out" 2>"$err"
  fi
  local rc=$?; set -e; monitor_stop
  if ((rc)); then printf 'bucket\torbit_tuned\t%s\t%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\t0\n' "$ORBIT_PROFILE" "$ORBIT_BIN" "$THREADS" "$rc" >>"$RESULT"; return; fi
  local line="$(grep '^residue=' "$out" | tail -n1 || true)"; [[ -n "$line" ]] || { echo 'orbit result missing' >&2; exit 4; }; local avg mx ns; IFS=$'\t' read -r avg mx ns <<<"$(mc_summary "$dm")"
  printf 'bucket\torbit_tuned\t%s\t%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\n' "$ORBIT_PROFILE" "$ORBIT_BIN" "$THREADS" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" "$ns" >>"$RESULT"
}

while IFS=$'\t' read -r label bin t; do [[ "$label" == label ]] && continue; echo "=== forced-set smoke $label threads=$t ===" >&2; run_forced "$label" "$bin" "$t"; done <"$FORCED_CANON"
echo '=== bucket smoke warp ===' >&2; run_warp
echo '=== bucket smoke orbit ===' >&2; run_orbit
EXPECTED=$((FORCED_COUNT + 2))
WIN="$(python3 - "$RESULT" "$EXPECTED" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));expected=int(sys.argv[2]);ok=[r for r in rows if r['status']=='ok']
if len(ok)!=expected:raise SystemExit(f'all {expected} candidates must succeed; got {len(ok)}')
res={r['residue'] for r in ok}
if len(res)!=1:raise SystemExit('FATAL forced-set/profiled residue mismatch '+repr({r['backend']:r['residue'] for r in ok}))
for r in sorted(ok,key=lambda x:float(x['wall_s'])):print('FORCED_SET_RACE',r['family'],r['backend'],'profile='+r['profile'],'wall_s='+r['wall_s'],'mc_avg='+r['mc_avg_pct'],file=sys.stderr)
b=min(ok,key=lambda x:float(x['wall_s']));print('\t'.join([b['family'],b['backend'],b['profile'],b['binary'],b['threads'],b['residue'],b['wall_s']]))
PY
)"
IFS=$'\t' read -r BEST_FAMILY BEST BEST_PROFILE BEST_BIN BEST_THREADS BEST_RES BEST_WALL <<<"$WIN"
echo "FORCED SET PROFILED SELECTED family=$BEST_FAMILY backend=$BEST profile=$BEST_PROFILE wall_s=$BEST_WALL residue=$BEST_RES" >&2; cat "$RESULT" >&2
SHA12="$(sha256sum "$BEST_BIN" | awk '{print substr($1,1,12)}')"; BEST_WORK="$WORK_ROOT/b300_exact_forcedset_${BEST}_${SHA12}_n27"; mkdir -p "$BEST_WORK"
python3 - "$BEST_WORK" "$BEST_BIN" "$PRIME" "$BEST_RES" "$BEST_WALL" <<'PY'
import hashlib,json,sys
from pathlib import Path
w=Path(sys.argv[1]);b=Path(sys.argv[2]).resolve();p=int(sys.argv[3]);r=int(sys.argv[4]);wall=float(sys.argv[5]);h=hashlib.sha256()
with b.open('rb') as f:
 for z in iter(lambda:f.read(1<<20),b''):h.update(z)
fp={'schema':2,'binary_sha256':h.hexdigest()};cp=w/'checkpoint.json';res={}
if cp.exists():
 old=json.loads(cp.read_text())
 if int(old.get('n',-1))!=27 or old.get('solver_fingerprint')!=fp:raise SystemExit(f'checkpoint incompatible: {cp}')
 res=dict(old.get('residues',{}))
if str(p) in res and int(res[str(p)]['residue'])!=r:raise SystemExit('smoke/checkpoint residue disagreement')
res[str(p)]={'residue':r,'wall_s':wall};tmp=cp.with_suffix('.json.tmp');tmp.write_text(json.dumps({'n':27,'solver_fingerprint':fp,'residues':res},indent=2,sort_keys=True)+'\n');tmp.replace(cp)
PY
if [[ "$SELECT_ONLY" == 1 ]]; then echo "SELECT_ONLY=1: selected $BEST; CRT not continued" >&2; exit 0; fi
if [[ "$BEST_FAMILY" == forced ]]; then export GRIDFP_THREADS="$BEST_THREADS"; RUN_TARGET="$FORCED_TARGET_MIB"
elif [[ "$BEST" == warp_tuned ]]; then export BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"; RUN_TARGET="$BUCKET_TARGET_MIB"
else export BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"; unset BUCKET_ORBITCTA_FLAT_BLOCKS; if [[ "$ORBITCTA_FLAT" == 1 && "$ORBITCTA_FLAT_BLOCKS_PER_SM" != 0 ]]; then export BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM"; else unset BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM; fi; RUN_TARGET="$BUCKET_TARGET_MIB"; fi
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" 27 --binary "$BEST_BIN" --target-mib "$RUN_TARGET" --max-window "$MAX_WINDOW" --gpus 8 --work-dir "$BEST_WORK" "$@"
