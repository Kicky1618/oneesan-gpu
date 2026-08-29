#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'calibrated profiled selector currently targets n=27' >&2; exit 2; }

ARCH="${ARCH:-native}"
PRIME="${SMOKE_PRIME:-4294967291}"
MAX_WINDOW="${MAX_WINDOW:-14}"
FORCED_TARGET_MIB="${FORCED_TARGET_MIB:-65536}"
BUCKET_TARGET_MIB="${BUCKET_TARGET_MIB:-16384}"
SELECT_ONLY="${SELECT_ONLY:-0}"
RECALIBRATE="${RECALIBRATE:-1}"
REBUILD_BUCKETS="${REBUILD_BUCKETS:-1}"
DUALMASK_MIN_SPEEDUP="${DUALMASK_MIN_SPEEDUP:-1.01}"
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_exact_calibrated_profiled_n27}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
WORK_ROOT="${WORK_ROOT:-$ONEESAN_ROOT/work}"
CAL_PREFIX="${CAL_PREFIX:-${PREFIX}_calibration}"
CAL_LOG="${CAL_LOG:-${CAL_PREFIX}.log}"
BUCKET_PREFIX="${BUCKET_PREFIX:-${PREFIX}_profiled_buckets}"
BUCKET_RESULT="${BUCKET_PREFIX}.tsv"
RESULT="${RESULT:-${PREFIX}.tsv}"
mkdir -p "$LOGDIR" "$(dirname "$RESULT")"
for x in SELECT_ONLY RECALIBRATE REBUILD_BUCKETS; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }; done
command -v nvcc >/dev/null || { echo 'nvcc required' >&2; exit 2; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi required' >&2; exit 2; }
(( $(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l) >= 8 )) || { echo 'need 8 visible GPUs' >&2; exit 2; }
[[ -f "$PROFILE_FILE" ]] || { echo "missing profile file: $PROFILE_FILE" >&2; exit 2; }

getv(){ local k="$1" f="$2"; sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f" | tail -n1; }
field(){ local k="$1" l="$2"; sed -nE "s/(^|.*[[:space:]])${k}=([^[:space:]]+).*/\\2/p" <<<"$l" | tail -n1; }

if [[ "$RECALIBRATE" == 1 || ! -s "$CAL_LOG" ]]; then
  echo '=== calibrated selector: forced nextgen calibration ===' >&2
  ROWS="${CAL_ROWS:-1}" ARCH="$ARCH" TARGET_MIB="$FORCED_TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" MOD="$PRIME" \
    THREADS_LIST="${THREADS_LIST:-128 256 512}" HIGHDROP_LIST="${HIGHDROP_LIST:-0 1}" SAMPLE_LOG2="${SAMPLE_LOG2:-20}" \
    PREFIX="$CAL_PREFIX" bash "$ONEESAN_ROOT/scripts/bench/b300-forced-nextgen-calibrate.sh" | tee "$CAL_LOG"
fi
[[ "$(getv b300_forced_nextgen_exact_calibration "$CAL_LOG")" == 1 ]] || { echo 'forced calibration exact gate missing' >&2; exit 3; }
FORCED_THREADS="$(getv b300_forced_nextgen_best_threads "$CAL_LOG")"
FORCED_HIGH="$(getv b300_forced_nextgen_best_high_drop_chunk "$CAL_LOG")"
DUAL_SPEED_RAW="$(getv b300_forced_nextgen_dualmask_speedup "$CAL_LOG")"
DUAL_SPEED="${DUAL_SPEED_RAW%x}"
[[ "$FORCED_THREADS" =~ ^[0-9]+$ ]] || { echo "invalid calibrated threads=$FORCED_THREADS" >&2; exit 3; }
[[ "$FORCED_HIGH" == 0 || "$FORCED_HIGH" == 1 ]] || { echo "invalid calibrated highdrop=$FORCED_HIGH" >&2; exit 3; }
FORCED_DUAL="$(python3 - "$DUAL_SPEED" "$DUALMASK_MIN_SPEEDUP" <<'PY'
import sys
x=float(sys.argv[1]); threshold=float(sys.argv[2]); print(1 if x>=threshold else 0)
PY
)"
echo "CALIBRATED FORCED threads=$FORCED_THREADS high_drop=$FORCED_HIGH dualmask=$FORCED_DUAL measured_dualmask_speedup=${DUAL_SPEED}x threshold=${DUALMASK_MIN_SPEEDUP}x" >&2

FORCED_BIN="$ONEESAN_BUILD_DIR/b300_calibrated_forced_hd${FORCED_HIGH}_dual${FORCED_DUAL}_n27"
FORCED_BUILD_OUT="$LOGDIR/forced.build.out"; FORCED_BUILD_ERR="$LOGDIR/forced.build.err"
echo '=== calibrated selector: build forced winner ===' >&2
N=27 ARCH="$ARCH" OUT="$FORCED_BIN" HIGH_DROP_CHUNK="$FORCED_HIGH" DUALMASK="$FORCED_DUAL" BUILD_ERR="$FORCED_BUILD_ERR" PTXAS_VERBOSE=1 \
  bash "$ONEESAN_ROOT/scripts/build/b300-forced-calibrated.sh" >"$FORCED_BUILD_OUT" 2>"$LOGDIR/forced.build.driver.err"
grep -Fq 'calibrated_forced=1' "$FORCED_BUILD_OUT"
grep -Fq "high_drop_chunk=$FORCED_HIGH dualmask=$FORCED_DUAL" "$FORCED_BUILD_OUT"

# Let the actively maintained profiled selector build and preselect only the
# current warp/orbit families. It deliberately does not see the forced binary.
echo '=== calibrated selector: preselect latest profiled bucket family ===' >&2
PROFILE_FILE="$PROFILE_FILE" CANDIDATES='warp_tuned orbit_tuned' SELECT_ONLY=1 REBUILD="$REBUILD_BUCKETS" \
  ARCH="$ARCH" SMOKE_PRIME="$PRIME" MAX_WINDOW="$MAX_WINDOW" BUCKET_TARGET_MIB="$BUCKET_TARGET_MIB" \
  PREFIX="$BUCKET_PREFIX" bash "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled.sh" 27 \
  >"$LOGDIR/profiled.selector.out" 2>"$LOGDIR/profiled.selector.err"
[[ -s "$BUCKET_RESULT" ]] || { echo "profiled selector did not produce $BUCKET_RESULT" >&2; exit 4; }

BUCKET_SELECTION="$(python3 - "$BUCKET_RESULT" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); ok=[r for r in rows if r.get('status')=='ok' and r.get('backend') in ('warp_tuned','orbit_tuned')]
if not ok: raise SystemExit('no successful profiled bucket candidate')
res={r['residue'] for r in ok}
if len(res)!=1: raise SystemExit('profiled bucket residue mismatch '+repr({r['backend']:r['residue'] for r in ok}))
b=min(ok,key=lambda r:float(r['wall_s']))
print('\t'.join([b['backend'],b['profile'],b['binary'],b['residue'],b['wall_s']]))
PY
)"
IFS=$'\t' read -r BUCKET_BACKEND BUCKET_PROFILE BUCKET_BIN BUCKET_PRE_RES BUCKET_PRE_WALL <<<"$BUCKET_SELECTION"
[[ -x "$BUCKET_BIN" ]] || { echo "profiled bucket binary missing: $BUCKET_BIN" >&2; exit 4; }
echo "PROFILED BUCKET PRESELECT backend=$BUCKET_BACKEND profile=$BUCKET_PROFILE wall_s=$BUCKET_PRE_WALL residue=$BUCKET_PRE_RES" >&2

# Runtime geometry must match the selected profile. Source only after all
# calibrated-forced decisions have been captured to avoid profile variables
# shadowing them.
# shellcheck disable=SC1090
source "$PROFILE_FILE"
ORBITCTA_FLAT="${ORBITCTA_FLAT:-0}"
ORBITCTA_FLAT_BLOCKS_PER_SM="${ORBITCTA_FLAT_BLOCKS_PER_SM:-0}"
BUCKET_THREADS="${BUCKET_THREADS:-256}"
WARP_GX="${WARP_GX:-32}"; WARP_GY="${WARP_GY:-8}"
ORBIT_GY="${ORBIT_GY:-128}"; LOW_GX="${LOW_GX:-16}"; LOW_GY="${LOW_GY:-8}"

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
  local so="$LOGDIR/forced.smoke.out" se="$LOGDIR/forced.smoke.err" dm="$LOGDIR/forced.smoke.dmon"
  nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null & local dp=$!; set +e
  GRIDFP_THREADS="$FORCED_THREADS" "$FORCED_BIN" 27 "$FORCED_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"; local rc=$?; set -e
  kill "$dp" 2>/dev/null || true; wait "$dp" 2>/dev/null || true
  if ((rc)); then printf 'forced_calibrated\tmainrec-hd%s-dual%s-t%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\t0\n' "$FORCED_HIGH" "$FORCED_DUAL" "$FORCED_THREADS" "$FORCED_BIN" "$rc" >>"$RESULT"; return 0; fi
  local line="$(grep '^backend=gridfp-b300-hbm32-forced2window-opt-batch ' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo 'forced calibrated result line missing' >&2; return 5; }
  local avg mx ns; IFS=$'\t' read -r avg mx ns <<<"$(sample_summary "$dm")"
  printf 'forced_calibrated\tmainrec-hd%s-dual%s-t%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\n' "$FORCED_HIGH" "$FORCED_DUAL" "$FORCED_THREADS" "$FORCED_BIN" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" "$ns" >>"$RESULT"
}
smoke_bucket(){
  local so="$LOGDIR/bucket.smoke.out" se="$LOGDIR/bucket.smoke.err" dm="$LOGDIR/bucket.smoke.dmon"
  nvidia-smi dmon -s u -d 1 >"$dm" 2>/dev/null & local dp=$!; set +e
  if [[ "$BUCKET_BACKEND" == warp_tuned ]]; then
    BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$BUCKET_BIN" 27 "$BUCKET_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
  elif [[ "$ORBITCTA_FLAT" == 1 && "$ORBITCTA_FLAT_BLOCKS_PER_SM" == 0 ]]; then
    env -u BUCKET_ORBITCTA_FLAT_BLOCKS -u BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM \
      BUCKET_THREADS="$BUCKET_THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$BUCKET_BIN" 27 "$BUCKET_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
  else
    env -u BUCKET_ORBITCTA_FLAT_BLOCKS BUCKET_THREADS="$BUCKET_THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM" \
      BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY" \
      "$BUCKET_BIN" 27 "$BUCKET_TARGET_MIB" "$MAX_WINDOW" 8 "$PRIME" >"$so" 2>"$se"
  fi
  local rc=$?; set -e; kill "$dp" 2>/dev/null || true; wait "$dp" 2>/dev/null || true
  if ((rc)); then printf '%s\t%s\t%s\tfailed:%s\tNA\tNA\tNA\tNA\t0\n' "$BUCKET_BACKEND" "$BUCKET_PROFILE" "$BUCKET_BIN" "$rc" >>"$RESULT"; return 0; fi
  local line="$(grep '^residue=' "$so" | tail -n1 || true)"; [[ -n "$line" ]] || { echo 'profiled bucket result line missing' >&2; return 5; }
  local avg mx ns; IFS=$'\t' read -r avg mx ns <<<"$(sample_summary "$dm")"
  printf '%s\t%s\t%s\tok\t%s\t%s\t%s\t%s\t%s\n' "$BUCKET_BACKEND" "$BUCKET_PROFILE" "$BUCKET_BIN" "$(field residue "$line")" "$(field wall_s "$line")" "$avg" "$mx" "$ns" >>"$RESULT"
}

echo '=== calibrated selector: final same-session smoke ===' >&2
smoke_forced
smoke_bucket

SEL="$(python3 - "$RESULT" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); ok=[r for r in rows if r['status']=='ok']
if len(ok)<2: raise SystemExit('need both calibrated forced and profiled bucket smoke results')
res={r['residue'] for r in ok}
if len(res)!=1: raise SystemExit('FATAL calibrated/profiled complete residue mismatch '+repr({r['backend']:r['residue'] for r in ok}))
for r in sorted(ok,key=lambda x:float(x['wall_s'])): print('CALIBRATED_FINAL',r['backend'],'profile='+r['profile'],'wall_s='+r['wall_s'],'mc_avg='+r['mc_avg_pct'],file=sys.stderr)
b=min(ok,key=lambda x:float(x['wall_s'])); print('\t'.join([b['backend'],b['profile'],b['binary'],b['residue'],b['wall_s']]))
PY
)"
IFS=$'\t' read -r BEST BEST_PROFILE BEST_BIN BEST_RES BEST_WALL <<<"$SEL"
echo "CALIBRATED PROFILED SELECTED backend=$BEST profile=$BEST_PROFILE wall_s=$BEST_WALL residue=$BEST_RES" >&2
cat "$RESULT" >&2

SHA12="$(sha256sum "$BEST_BIN" | awk '{print substr($1,1,12)}')"
BEST_WORK="$WORK_ROOT/b300_exact_calibrated_${BEST}_${BEST_PROFILE}_${SHA12}_n27"; mkdir -p "$BEST_WORK"
python3 - "$BEST_WORK" "$BEST_BIN" "$PRIME" "$BEST_RES" "$BEST_WALL" <<'PY'
import hashlib,json,sys
from pathlib import Path
work=Path(sys.argv[1]); binary=Path(sys.argv[2]).resolve(); p=int(sys.argv[3]); r=int(sys.argv[4]); wall=float(sys.argv[5]); h=hashlib.sha256()
with binary.open('rb') as f:
 for z in iter(lambda:f.read(1<<20),b''): h.update(z)
fp={'schema':2,'binary_sha256':h.hexdigest()}; cp=work/'checkpoint.json'; res={}
if cp.exists():
 old=json.loads(cp.read_text())
 if int(old.get('n',-1))!=27 or old.get('solver_fingerprint')!=fp: raise SystemExit(f'checkpoint incompatible: {cp}')
 res=dict(old.get('residues',{}))
if str(p) in res and int(res[str(p)]['residue'])!=r: raise SystemExit('smoke/checkpoint residue disagreement')
res[str(p)]={'residue':r,'wall_s':wall}; tmp=cp.with_suffix('.json.tmp'); tmp.write_text(json.dumps({'n':27,'solver_fingerprint':fp,'residues':res},indent=2,sort_keys=True)+'\n'); tmp.replace(cp)
print(f'seeded {cp} cached_residues={len(res)}',file=sys.stderr)
PY

if [[ "$SELECT_ONLY" == 1 ]]; then echo "SELECT_ONLY=1: selected $BEST/$BEST_PROFILE; CRT not continued" >&2; exit 0; fi
if [[ "$BEST" == forced_calibrated ]]; then
  export GRIDFP_THREADS="$FORCED_THREADS"; RUN_TARGET="$FORCED_TARGET_MIB"
elif [[ "$BEST" == warp_tuned ]]; then
  export BUCKET_THREADS="$BUCKET_THREADS" BUCKET_GRID_X="$WARP_GX" BUCKET_GRID_Y="$WARP_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"; RUN_TARGET="$BUCKET_TARGET_MIB"
else
  export BUCKET_THREADS="$BUCKET_THREADS" BUCKET_ORBITCTA_GRID_Y="$ORBIT_GY" BUCKET_GRID_X="$LOW_GX" BUCKET_GRID_Y="$LOW_GY" BUCKET_LOW_GRID_X="$LOW_GX" BUCKET_LOW_GRID_Y="$LOW_GY"
  unset BUCKET_ORBITCTA_FLAT_BLOCKS
  if [[ "$ORBITCTA_FLAT" == 1 && "$ORBITCTA_FLAT_BLOCKS_PER_SM" != 0 ]]; then export BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM="$ORBITCTA_FLAT_BLOCKS_PER_SM"; else unset BUCKET_ORBITCTA_FLAT_BLOCKS_PER_SM; fi
  RUN_TARGET="$BUCKET_TARGET_MIB"
fi
exec python3 "$ONEESAN_ROOT/scripts/solve/solve_b300_exact_batch.py" 27 --binary "$BEST_BIN" --target-mib "$RUN_TARGET" --max-window "$MAX_WINDOW" --gpus 8 --work-dir "$BEST_WORK" "$@"
