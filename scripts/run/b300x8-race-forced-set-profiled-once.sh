#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}";if(($#>0));then shift;fi
[[ "$N" == 27 ]]||{ echo 'forced-set single-pass race targets n=27' >&2;exit 2; }
: "${FORCED_SET_FILE:?FORCED_SET_FILE is required}"
[[ -s "$FORCED_SET_FILE" ]]||{ echo "missing forced set: $FORCED_SET_FILE" >&2;exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}";[[ -f "$PROFILE_FILE" ]]||{ echo "missing profile: $PROFILE_FILE" >&2;exit 2; }
ARCH="${ARCH:-native}";PRIME="${SMOKE_PRIME:-4294967291}";MAX_WINDOW="${MAX_WINDOW:-14}";FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}";BUCKET_TARGET_MIB="${BUCKET_TARGET_MIB:-16384}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}";SELECT_ONLY="${SELECT_ONLY:-1}";PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_forced_set_profiled_once_n27}";LOGDIR="${LOGDIR:-${PREFIX}_logs}";RESULT="${RESULT:-${PREFIX}.tsv}";WORK_ROOT="${WORK_ROOT:-$ONEESAN_ROOT/work}"
BUCKET_ENV="${BUCKET_ENV:-${PREFIX}.buckets.env}";BUCKET_PREFIX="${BUCKET_PREFIX:-${PREFIX}.buckets}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
for x in REBUILD_BUCKETS SELECT_ONLY;do v="${!x}";[[ "$v" == 0 || "$v" == 1 ]]||exit 2;done
command -v nvidia-smi >/dev/null||{ echo 'nvidia-smi required' >&2;exit 2; };(( $(nvidia-smi --query-gpu=index --format=csv,noheader|wc -l)>=8 ))||{ echo 'need 8 visible GPUs' >&2;exit 2; }

# Validate and canonicalize the forced set. Header is optional.
FORCED_CANON="$LOGDIR/forced-set.tsv"
python3 - "$FORCED_SET_FILE" "$FORCED_CANON" <<'PY'
import csv,os,re,sys
src,dst=sys.argv[1:3]
lines=[x.rstrip('\n') for x in open(src) if x.strip() and not x.lstrip().startswith('#')]
if not lines:raise SystemExit('empty forced set')
if lines[0].split('\t')[:3]==['label','binary','threads']:lines=lines[1:]
out=[];seen=set();bins=set()
for ln in lines:
 a=ln.split('\t')
 if len(a)<3:raise SystemExit('forced set row needs label,binary,threads: '+ln)
 label,binary,threads=a[:3]
 if not re.fullmatch(r'[A-Za-z0-9_.-]+',label):raise SystemExit('unsafe forced label '+label)
 if label in seen:raise SystemExit('duplicate forced label '+label)
 if not os.path.isfile(binary) or not os.access(binary,os.X_OK):raise SystemExit('forced binary not executable '+binary)
 try:t=int(threads)
 except ValueError:raise SystemExit('bad forced threads '+threads)
 if t<32 or t>1024 or t%32:raise SystemExit('forced threads must be warp multiple 32..1024')
 rp=os.path.realpath(binary)
 if rp in bins:raise SystemExit('duplicate forced binary '+rp)
 seen.add(label);bins.add(rp);out.append((label,rp,t))
if not out:raise SystemExit('no forced candidates')
with open(dst,'w') as f:
 f.write('label\tbinary\tthreads\n')
 for r in out:f.write('\t'.join(map(str,r))+'\n')
print(f'forced_set_candidates={len(out)}',file=sys.stderr)
PY

# Build the two current profile families without consuming a complete prime.
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" REBUILD="$REBUILD_BUCKETS" PREFIX="$BUCKET_PREFIX" OUT_ENV="$BUCKET_ENV" \
  bash "$ONEESAN_ROOT/scripts/build/b300-profiled-buckets-only.sh" >"$LOGDIR/buckets.build.out" 2>"$LOGDIR/buckets.build.err"
[[ -s "$BUCKET_ENV" ]]||exit 3
# shellcheck disable=SC1090
source "$BUCKET_ENV"
[[ "${B300_PROFILED_BUCKETS_BUILD_ONLY:-0}" == 1 && -x "$WARP_BIN" && -x "$ORBIT_BIN" ]]||exit 3
[[ "$(sha256sum "$PROFILE_FILE"|awk '{print $1}')" == "$PROFILE_SHA256" ]]||{ echo 'profile changed during bucket build' >&2;exit 3; }
# shellcheck disable=SC1090
source "$PROFILE_FILE"
THREADS="${BUCKET_THREADS:-256}";WARP_GX="${WARP_GX:-32}";WARP_GY="${WARP_GY:-8}";ORBIT_GY="${ORBIT_GY:-128}";LOW_GX="${LOW_GX:-16}";LOW_GY="${LOW_GY:-8}";ORBITCTA_FLAT="${ORBITCTA_FLAT:-0}";ORBITCTA_FLAT_BLOCKS_PER_SM="${ORBITCTA_FLAT_BLOCKS_PER_SM:-0}"

field(){ local k="$1" l="$2";sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p"<<<"$l"|tail -n1; }
sample(){ python3 - "$1" <<'PY'
import sys
v=[]
try:
 for l in open(sys.argv[1],errors='replace'):
  a=l.split()
  if len(a)>=3:
   try:v.append(float(a[2]))
   except:pass
except:pass
if len(v)>16:v=v[8:]
print('NA\tNA\t0' if not v else f'{sum(v)/len(v):.3f}\t{max(v):.3f}\t{len(v)}')
PY
}
printf 'family\tbackend\tprofile\tbinary\tthreads\tstatus\tresidue\twall_s\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"

smoke_forced(){ local label="$1" bin="$2" t="$3" tag="forced_${label}" so="$LOGDIR/${tag}.out" se="$LOGDIR/${tag}.err" dm="$LOGDIR/${tag}.dmon";nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null&pid=$!;set +e;GRIDFP_THREADS="$t" "$bin" 27 "$FORCED_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se";rc=$?;set -e;kill "$pid" 2>/dev/null||true;wait "$pid" 2>/dev/null||true;if((rc));then printf 'forced\t%s\tt%s\t%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\t0\n' "$label" "$t" "$bin" "$t" "$rc">>"$RESULT";return;fi;line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so"|tail -n1||true)";[[ -n "$line" ]]||{ echo "$label missing forced result" >&2;exit 4; };IFS=$'\t' read -r avg mx ns<<<"$(sample "$dm")";printf 'forced\t%s\tt%s\t%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\n' "$label" "$t" "$bin" "$t" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" "$ns">>"$RESULT"; }
smoke_warp(){ local so="$LOGDIR/warp.out" se="$LOGDIR/warp.err" dm="$LOGDIR/warp.dmon";nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null&pid=$!;set +e;BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$WARP_BIN" 27 "$BUCKET_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se";rc=$?;set -e;kill "$pid" 2>/dev/null||true;wait "$pid" 2>/dev/null||true;if((rc));then printf 'bucket\twarp_tuned\t%s\t%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\t0\n' "$WARP_PROFILE" "$WARP_BIN" "$THREADS" "$rc">>"$RESULT";return;fi;line="$(grep '^residue=' "$so"|tail -n1||true)";[[ -n "$line" ]]||exit 4;IFS=$'\t' read -r avg mx ns<<<"$(sample "$dm")";printf 'bucket\twarp_tuned\t%s\t%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\n' "$WARP_PROFILE" "$WARP_BIN" "$THREADS" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" "$ns">>"$RESULT"; }
smoke_orbit(){ local so="$LOGDIR/orbit.out" se="$LOGDIR/orbit.err" dm="$LOGDIR/orbit.dmon";nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null&pid=$!;set +e;if [[ "$ORBITCTA_FLAT" == 1 && "$ORBITCTA_FLAT_BLOCKS_PER_SM" == 0 ]];then env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$ORBIT_BIN" 27 "$BUCKET_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se";else env -u BUCKET_ORBITCTA_FLAT_BLOCKS BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$ORBIT_BIN" 27 "$BUCKET_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se";fi;rc=$?;set -e;kill "$pid" 2>/dev/null||true;wait "$pid" 2>/dev/null||true;if((rc));then printf 'bucket\torbit_tuned\t%s\t%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\t0\n' "$ORBIT_PROFILE" "$ORBIT_BIN" "$THREADS" "$rc">>"$RESULT";return;fi;line="$(grep '^residue=' "$so"|tail -n1||true)";[[ -n "$line" ]]||exit 4;IFS=$'\t' read -r avg mx ns<<<"$(sample "$dm")";printf 'bucket\torbit_tuned\t%s\t%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\n' "$ORBIT_PROFILE" "$ORBIT_BIN" "$THREADS" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" "$ns">>"$RESULT"; }

while IFS=$'\t' read -r label bin t;do [[ "$label" == label ]]&&continue;echo "=== forced-set smoke $label threads=$t ===" >&2;smoke_forced "$label" "$bin" "$t";done<"$FORCED_CANON"
echo '=== bucket smoke warp ===' >&2;smoke_warp
echo '=== bucket smoke orbit ===' >&2;smoke_orbit
EXPECTED=$(( $(($(wc -l <"$FORCED_CANON")-1)) + 2 ))
WIN="$(python3 - "$RESULT" "$EXPECTED" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));expected=int(sys.argv[2]);ok=[x for x in r if x['status']=='ok']
if len(ok)!=expected:raise SystemExit(f'all {expected} candidates must complete one-prime smoke; got {len(ok)}')
res={x['residue'] for x in ok}
if len(res)!=1:raise SystemExit('FATAL forced-set/profiled residue mismatch '+repr({x['backend']:x['residue'] for x in ok}))
for x in sorted(ok,key=lambda z:float(z['wall_s'])):print('FORCED_SET_RACE',x['family'],x['backend'],'profile='+x['profile'],'wall_s='+x['wall_s'],'mc_avg='+x['mc_avg_pct'],file=sys.stderr)
b=min(ok,key=lambda z:float(z['wall_s']));print('\t'.join([b['family'],b['backend'],b['profile'],b['binary'],b['threads'],b['residue'],b['wall_s']]))
PY
)"
IFS=$'\t' read -r BEST_FAMILY BEST BEST_PROFILE BEST_BIN BEST_THREADS BEST_RES BEST_WALL<<<"$WIN"
echo "FORCED SET PROFILED SELECTED family=$BEST_FAMILY backend=$BEST profile=$BEST_PROFILE wall_s=$BEST_WALL residue=$BEST_RES" >&2;cat "$RESULT" >&2
SHA12="$(sha256sum "$BEST_BIN"|awk '{print substr($1,1,12)}')";BEST_WORK="$WORK_ROOT/b300_exact_forcedset_${BEST}_${SHA12}_n27";mkdir -p "$BEST_WORK"
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
if [[ "$SELECT_ONLY" == 1 ]];then echo "SELECT_ONLY=1: selected $BEST; CRT not continued" >&2;exit 0;fi
if [[ "$BEST_FAMILY" == forced ]];then export GRIDFP_THREADS="$BEST_THREADS";RUN_TARGET="$FORCED_TARGET_MIB"
elif [[ "$BEST" == warp_tuned ]];then export BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY";RUN_TARGET="$BUCKET_TARGET_MIB"
else export BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY";unset BUCKET_ORBITCTA_FLAT_BLOCKS;if [[ "$ORBITCTA_FLAT" == 1 && "$ORBITCTA_FLAT_BLOCKS_PER_SM" != 0 ]];then export BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM";else unset BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM;fi;RUN_TARGET="$BUCKET_TARGET_MIB";fi
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" 27 --binary "$BEST_BIN" --target-mib "$RUN_TARGET" --max-window "$MAX_WINDOW" --gpus 8 --work-dir "$BEST_WORK" "$@"
