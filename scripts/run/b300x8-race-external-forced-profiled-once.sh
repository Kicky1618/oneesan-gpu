#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'single-pass external/profiled race targets n=27' >&2; exit 2; }
FORCED_OVERRIDE_BIN="${FORCED_OVERRIDE_BIN:-}"; FORCED_OVERRIDE_LABEL="${FORCED_OVERRIDE_LABEL:-external_forced}"; FORCED_OVERRIDE_THREADS="${FORCED_OVERRIDE_THREADS:-256}"
FORCED_BASE_BIN="${FORCED_BASE_BIN:-}"; FORCED_BASE_LABEL="${FORCED_BASE_LABEL:-external_forced_base}"; FORCED_BASE_THREADS="${FORCED_BASE_THREADS:-256}"
[[ -n "$FORCED_OVERRIDE_BIN" && -x "$FORCED_OVERRIDE_BIN" ]] || { echo 'FORCED_OVERRIDE_BIN must be executable' >&2; exit 2; }
for v in FORCED_OVERRIDE_THREADS FORCED_BASE_THREADS; do x="${!v}"; [[ "$x" =~ ^[0-9]+$ ]] && ((x>=32&&x<=1024&&x%32==0)) || { echo "$v must be warp multiple 32..1024" >&2; exit 2; }; done
HAS_FORCED_BASE=0
if [[ -n "$FORCED_BASE_BIN" && "$FORCED_BASE_BIN" != "$FORCED_OVERRIDE_BIN" ]]; then [[ -x "$FORCED_BASE_BIN" ]] || { echo 'FORCED_BASE_BIN must be executable' >&2; exit 2; }; HAS_FORCED_BASE=1; fi

PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"; [[ -f "$PROFILE_FILE" ]] || { echo "missing profile $PROFILE_FILE" >&2; exit 2; }
ARCH="${ARCH:-native}"; PRIME="${SMOKE_PRIME:-4294967291}"; MAX_WINDOW="${MAX_WINDOW:-14}"; FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}"; BUCKET_TARGET_MIB="${BUCKET_TARGET_MIB:-16384}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"; SELECT_ONLY="${SELECT_ONLY:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_external_forced_profiled_once_n27}"; LOGDIR="${LOGDIR:-${PREFIX}_logs}"; RESULT="${RESULT:-${PREFIX}.tsv}"; WORK_ROOT="${WORK_ROOT:-$ONEESAN_ROOT/work}"
BUCKET_BUILD_ENV="${BUCKET_BUILD_ENV:-${PREFIX}.buckets.env}"; BUCKET_BUILD_PREFIX="${BUCKET_BUILD_PREFIX:-${PREFIX}.buckets}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
for x in REBUILD_BUCKETS SELECT_ONLY; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }

# Build both current profiled bucket families, but do not run a full-prime
# preselection. The final race below is the only complete-prime execution.
echo '=== single-pass race: build profiled warp/orbit only ===' >&2
PROFILE_FILE="$PROFILE_FILE" ARCH="$ARCH" REBUILD="$REBUILD_BUCKETS" PREFIX="$BUCKET_BUILD_PREFIX" OUT_ENV="$BUCKET_BUILD_ENV" \
  bash "$ONEESAN_ROOT/scripts/build/b300-profiled-buckets-only.sh" >"$LOGDIR/buckets.build.out" 2>"$LOGDIR/buckets.build.err"
[[ -s "$BUCKET_BUILD_ENV" ]] || { echo 'bucket build env missing' >&2; exit 3; }
# shellcheck disable=SC1090
source "$BUCKET_BUILD_ENV"
[[ "${B300_PROFILED_BUCKETS_BUILD_ONLY:-0}" == 1 ]] || { echo 'bucket build-only marker missing' >&2; exit 3; }
[[ -x "$WARP_BIN" && -x "$ORBIT_BIN" ]] || { echo 'bucket binary missing after build-only' >&2; exit 3; }
NOW_PROFILE_SHA="$(sha256sum "$PROFILE_FILE" | awk '{print $1}')"
[[ "$NOW_PROFILE_SHA" == "$PROFILE_SHA256" ]] || { echo 'profile changed during bucket build; refusing mixed race' >&2; exit 3; }

# shellcheck disable=SC1090
source "$PROFILE_FILE"
THREADS="${BUCKET_THREADS:-256}"; WARP_GX="${WARP_GX:-32}"; WARP_GY="${WARP_GY:-8}"; ORBIT_GY="${ORBIT_GY:-128}"; LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"
ORBITCTA_FLAT="${ORBITCTA_FLAT:-0}"; ORBITCTA_FLAT_BLOCKS_PER_SM="${ORBITCTA_FLAT_BLOCKS_PER_SM:-0}"
[[ "$THREADS" =~ ^[0-9]+$ ]] && ((THREADS>=32&&THREADS<=1024&&THREADS%32==0)) || { echo 'bad BUCKET_THREADS' >&2; exit 3; }

field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }
sample_summary(){ python3 - "$1" <<'PY'
import sys
v=[]
try:
 for line in open(sys.argv[1],encoding='utf-8',errors='replace'):
  s=line.strip()
  if not s or s.startswith('#'): continue
  a=s.split()
  if len(a)>=3:
   try:v.append(float(a[2]))
   except ValueError:pass
except FileNotFoundError:pass
if len(v)>16:v=v[8:]
print('NA\tNA\t0' if not v else f'{sum(v)/len(v):.3f}\t{max(v):.3f}\t{len(v)}')
PY
}
printf 'backend\tprofile\tbinary\tstatus\tresidue\twall_s\tmc_avg_pct\tmc_max_pct\tmc_samples\n' >"$RESULT"

smoke_forced(){
 local slot="$1" label="$2" bin="$3" threads="$4"; local so="$LOGDIR/$slot.out" se="$LOGDIR/$slot.err" dm="$LOGDIR/$slot.dmon"
 nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null & local dp=$!; set +e
 GRIDFP_THREADS="$threads" "$bin" 27 "$FORCED_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"; local rc=$?; set -e
 kill "$dp" 2>/dev/null || true; wait "$dp" 2>/dev/null || true
 if ((rc)); then printf '%s\tt%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\t0\n' "$label" "$threads" "$bin" "$rc" >>"$RESULT"; return; fi
 local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo "$slot forced result missing" >&2; exit 4; }
 local avg mx ns; IFS=$'\t' read -r avg mx ns <<<"$(sample_summary "$dm")"
 printf '%s\tt%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\n' "$label" "$threads" "$bin" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" "$ns" >>"$RESULT"
}
smoke_warp(){
 local so="$LOGDIR/warp.out" se="$LOGDIR/warp.err" dm="$LOGDIR/warp.dmon"; nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null & local dp=$!; set +e
 BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$WARP_BIN" 27 "$BUCKET_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"; local rc=$?; set -e
 kill "$dp" 2>/dev/null || true; wait "$dp" 2>/dev/null || true
 if ((rc)); then printf 'warp_tuned\t%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\t0\n' "$WARP_PROFILE" "$WARP_BIN" "$rc" >>"$RESULT"; return; fi
 local line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo 'warp result missing' >&2; exit 4; }; local avg mx ns; IFS=$'\t' read -r avg mx ns <<<"$(sample_summary "$dm")"
 printf 'warp_tuned\t%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\n' "$WARP_PROFILE" "$WARP_BIN" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" "$ns" >>"$RESULT"
}
smoke_orbit(){
 local so="$LOGDIR/orbit.out" se="$LOGDIR/orbit.err" dm="$LOGDIR/orbit.dmon"; nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null & local dp=$!; set +e
 if [[ "$ORBITCTA_FLAT" == 1 && "$ORBITCTA_FLAT_BLOCKS_PER_SM" == 0 ]]; then
   env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$ORBIT_BIN" 27 "$BUCKET_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
 else
   env -u BUCKET_ORBITCTA_FLAT_BLOCKS BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" "$ORBIT_BIN" 27 "$BUCKET_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
 fi
 local rc=$?; set -e; kill "$dp" 2>/dev/null || true; wait "$dp" 2>/dev/null || true
 if ((rc)); then printf 'orbit_tuned\t%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\t0\n' "$ORBIT_PROFILE" "$ORBIT_BIN" "$rc" >>"$RESULT"; return; fi
 local line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo 'orbit result missing' >&2; exit 4; }; local avg mx ns; IFS=$'\t' read -r avg mx ns <<<"$(sample_summary "$dm")"
 printf 'orbit_tuned\t%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\n' "$ORBIT_PROFILE" "$ORBIT_BIN" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" "$ns" >>"$RESULT"
}

echo '=== single-pass race: one complete prime per candidate ===' >&2
smoke_forced forced_primary "$FORCED_OVERRIDE_LABEL" "$FORCED_OVERRIDE_BIN" "$FORCED_OVERRIDE_THREADS"
if [[ "$HAS_FORCED_BASE" == 1 ]]; then smoke_forced forced_base "$FORCED_BASE_LABEL" "$FORCED_BASE_BIN" "$FORCED_BASE_THREADS"; fi
smoke_warp
smoke_orbit
EXPECTED_OK=$((3+HAS_FORCED_BASE))
WIN="$(python3 - "$RESULT" "$EXPECTED_OK" <<'PY'
import csv,sys
r=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'));expected=int(sys.argv[2]);ok=[x for x in r if x['status']=='ok']
if len(ok)!=expected:raise SystemExit(f'all {expected} single-pass candidates must succeed; got {len(ok)}')
res={x['residue'] for x in ok}
if len(res)!=1:raise SystemExit('FATAL single-pass residue mismatch '+repr({x['backend']:x['residue'] for x in ok}))
for x in sorted(ok,key=lambda z:float(z['wall_s'])):print('SINGLE_PASS_RACE',x['backend'],'profile='+x['profile'],'wall_s='+x['wall_s'],'mc_avg='+x['mc_avg_pct'],file=sys.stderr)
b=min(ok,key=lambda z:float(z['wall_s']));print('\t'.join([b['backend'],b['profile'],b['binary'],b['residue'],b['wall_s']]))
PY
)"
IFS=$'\t' read -r BEST BEST_PROFILE BEST_BIN BEST_RES BEST_WALL <<<"$WIN"
echo "SINGLE PASS SELECTED backend=$BEST profile=$BEST_PROFILE wall_s=$BEST_WALL residue=$BEST_RES" >&2; cat "$RESULT" >&2

SHA12="$(sha256sum "$BEST_BIN" | awk '{print substr($1,1,12)}')"; BEST_WORK="$WORK_ROOT/b300_exact_singlepass_${BEST}_${BEST_PROFILE}_${SHA12}_n27"; mkdir -p "$BEST_WORK"
python3 - "$BEST_WORK" "$BEST_BIN" "$PRIME" "$BEST_RES" "$BEST_WALL" "$PROFILE_SHA256" <<'PY'
import hashlib,json,sys
from pathlib import Path
w=Path(sys.argv[1]);b=Path(sys.argv[2]).resolve();p=int(sys.argv[3]);r=int(sys.argv[4]);wall=float(sys.argv[5]);profile_sha=sys.argv[6];h=hashlib.sha256()
with b.open('rb') as f:
 for z in iter(lambda:f.read(1<<20),b''):h.update(z)
fp={'schema':3,'binary_sha256':h.hexdigest(),'profile_sha256':profile_sha};cp=w/'checkpoint.json';res={}
if cp.exists():
 old=json.loads(cp.read_text())
 if int(old.get('n',-1))!=27 or old.get('solver_fingerprint')!=fp:raise SystemExit(f'checkpoint incompatible: {cp}')
 res=dict(old.get('residues',{}))
if str(p) in res and int(res[str(p)]['residue'])!=r:raise SystemExit('smoke/checkpoint residue disagreement')
res[str(p)]={'residue':r,'wall_s':wall};tmp=cp.with_suffix('.json.tmp');tmp.write_text(json.dumps({'n':27,'solver_fingerprint':fp,'residues':res},indent=2,sort_keys=True)+'\n');tmp.replace(cp)
print(f'seeded {cp} cached_residues={len(res)} profile_sha256={profile_sha}',file=sys.stderr)
PY
if [[ "$SELECT_ONLY" == 1 ]]; then echo "SELECT_ONLY=1: selected $BEST/$BEST_PROFILE; CRT not continued" >&2; exit 0; fi
if [[ "$BEST_BIN" == "$FORCED_OVERRIDE_BIN" ]]; then export GRIDFP_THREADS="$FORCED_OVERRIDE_THREADS"; RUN_TARGET="$FORCED_TARGET_MIB"
elif [[ "$HAS_FORCED_BASE" == 1 && "$BEST_BIN" == "$FORCED_BASE_BIN" ]]; then export GRIDFP_THREADS="$FORCED_BASE_THREADS"; RUN_TARGET="$FORCED_TARGET_MIB"
elif [[ "$BEST" == warp_tuned ]]; then export BUCKET_THREADS="$THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"; RUN_TARGET="$BUCKET_TARGET_MIB"
else export BUCKET_THREADS="$THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"; unset BUCKET_ORBITCTA_FLAT_BLOCKS; if [[ "$ORBITCTA_FLAT" == 1 && "$ORBITCTA_FLAT_BLOCKS_PER_SM" != 0 ]]; then export BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM"; else unset BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM; fi; RUN_TARGET="$BUCKET_TARGET_MIB"; fi
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" 27 --binary "$BEST_BIN" --target-mib "$RUN_TARGET" --max-window "$MAX_WINDOW" --gpus 8 --work-dir "$BEST_WORK" "$@"
