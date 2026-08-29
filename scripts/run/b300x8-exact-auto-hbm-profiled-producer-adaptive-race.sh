#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'producer-adaptive race currently targets n=27' >&2; exit 2; }

BASE_RUN="$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled.sh"
PROFILE_IN="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21_n27pww.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21_n27pa.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_n27_producer_adaptive_race}"
THRESHOLDS="${PRODUCER_ADAPTIVE_THRESHOLDS:-0 64 128 256 512 1024 2048 4096 8192 16384}"
ADAPTIVE_REBUILD="${ADAPTIVE_REBUILD:-1}"
ADAPTIVE_REPEATS="${ADAPTIVE_REPEATS:-1}"
ADAPTIVE_RACE_ONLY="${ADAPTIVE_RACE_ONLY:-0}"
SUMMARY="${SUMMARY:-${PREFIX}.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$LOGDIR" "$(dirname "$PROFILE_OUT")" "$(dirname "$SUMMARY")"

for x in ADAPTIVE_REBUILD ADAPTIVE_RACE_ONLY; do
  v="${!x}"
  [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
[[ "$ADAPTIVE_REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo 'ADAPTIVE_REPEATS must be positive integer' >&2; exit 2; }
[[ -f "$PROFILE_IN" ]] || { echo "missing profile: $PROFILE_IN" >&2; exit 2; }
# shellcheck disable=SC1090
source "$PROFILE_IN"
ORBITCTA_FLAT_DYNAMIC="${ORBITCTA_FLAT_DYNAMIC:-0}"
ORBITCTA_FLAT_DYNAMIC_PIPE2="${ORBITCTA_FLAT_DYNAMIC_PIPE2:-0}"
ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP="${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP:-0}"
BASE_WEIGHT="${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT:-0}"
case "$BASE_WEIGHT" in 0|1|2|3|4) ;; *) echo "bad producer worker weight: $BASE_WEIGHT" >&2; exit 2;; esac

if [[ "$ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP" != 1 ]]; then
  echo 'producer-adaptive race skipped: selected profile does not use PIPE2 producer warp' >&2
  cp "$PROFILE_IN" "$PROFILE_OUT"
  if [[ "$ADAPTIVE_RACE_ONLY" == 1 ]]; then exit 0; fi
  exec env PROFILE_FILE="$PROFILE_OUT" "$BASE_RUN" 27 "$@"
fi
[[ "$ORBITCTA_FLAT_DYNAMIC" == 1 && "$ORBITCTA_FLAT_DYNAMIC_PIPE2" == 1 ]] || {
  echo 'producer-adaptive race requires dynamic PIPE2' >&2; exit 2;
}
if [[ "$BASE_WEIGHT" == 1 ]]; then
  echo 'producer-adaptive race skipped: base weight=1 makes every threshold equivalent' >&2
  cp "$PROFILE_IN" "$PROFILE_OUT"
  if [[ "$ADAPTIVE_RACE_ONLY" == 1 ]]; then exit 0; fi
  exec env PROFILE_FILE="$PROFILE_OUT" "$BASE_RUN" 27 "$@"
fi

read -r -a threshold_list <<<"$THRESHOLDS"
(( ${#threshold_list[@]} > 0 )) || { echo 'PRODUCER_ADAPTIVE_THRESHOLDS is empty' >&2; exit 2; }
declare -A seen=()
for t in "${threshold_list[@]}"; do
  [[ "$t" =~ ^[0-9]+$ ]] || { echo "bad producer adaptive threshold: $t" >&2; exit 2; }
  [[ -z "${seen[$t]+x}" ]] || { echo "duplicate producer adaptive threshold: $t" >&2; exit 2; }
  seen[$t]=1
done
[[ -n "${seen[0]+x}" ]] || { echo 'producer adaptive race must include threshold 0 baseline' >&2; exit 2; }

make_profile() {
  local threshold="$1" out="$2"
  python3 - "$PROFILE_IN" "$out" "$threshold" <<'PY'
import sys
src,out,t=sys.argv[1:]
key='ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS'
metadata={
    'ORBIT_N27_PRODUCER_ADAPTIVE_COLS',
    'ORBIT_N27_PRODUCER_ADAPTIVE_WALL_S',
    'ORBIT_N27_PRODUCER_ADAPTIVE_BASE_WEIGHT',
    'ORBIT_N27_PRODUCER_ADAPTIVE_REPEATS',
}
lines=open(src).read().splitlines(); dst=[]; found=False
for line in lines:
    if line.startswith(key+'='):
        dst.append(key+'='+t); found=True
    elif any(line.startswith(k+'=') for k in metadata):
        continue
    else:
        dst.append(line)
if not found: dst.append(key+'='+t)
open(out,'w').write('\n'.join(dst)+'\n')
PY
}

printf 'threshold\tbase_weight\trepeat\tstatus\tresidue\twall_s\tprofile\tresult\n' >"$SUMMARY"
for t in "${threshold_list[@]}"; do
  pf="$LOGDIR/profile_t${t}.env"
  make_profile "$t" "$pf"
  for ((rep=1; rep<=ADAPTIVE_REPEATS; ++rep)); do
    tp="${PREFIX}_t${t}_r${rep}"
    rebuild="$ADAPTIVE_REBUILD"; (( rep > 1 )) && rebuild=0
    echo "=== n27 producer-adaptive smoke base_weight=$BASE_WEIGHT threshold=$t repeat=$rep/$ADAPTIVE_REPEATS ===" >&2
    set +e
    PRODUCER_ADAPTIVE_COLS="$t" PROFILE_FILE="$pf" CANDIDATES=orbit_tuned SELECT_ONLY=1 REBUILD="$rebuild" PREFIX="$tp" \
      "$BASE_RUN" 27 >"$LOGDIR/t${t}_r${rep}.out" 2>"$LOGDIR/t${t}_r${rep}.err"
    rc=$?
    set -e
    if (( rc != 0 )); then
      printf '%s\t%s\t%s\tfailed:%s\tNA\tNA\t%s\t%s\n' "$t" "$BASE_WEIGHT" "$rep" "$rc" "$pf" "${tp}.tsv" >>"$SUMMARY"
      echo "producer-adaptive smoke failed threshold=$t repeat=$rep rc=$rc" >&2
      exit "$rc"
    fi
    result="${tp}.tsv"
    [[ -f "$result" ]] || { echo "missing result for threshold=$t repeat=$rep: $result" >&2; exit 3; }
    read -r residue wall < <(python3 - "$result" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
ok=[r for r in rows if r.get('backend')=='orbit_tuned' and r.get('status')=='ok']
if len(ok)!=1: raise SystemExit(f'expected one successful orbit_tuned row, got {len(ok)}')
print(ok[0]['residue'],ok[0]['wall_s'])
PY
)
    printf '%s\t%s\t%s\tok\t%s\t%s\t%s\t%s\n' "$t" "$BASE_WEIGHT" "$rep" "$residue" "$wall" "$pf" "$result" >>"$SUMMARY"
  done
done

read -r best_threshold best_residue best_wall < <(python3 - "$SUMMARY" "$ADAPTIVE_REPEATS" <<'PY'
import csv,statistics,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t')); repeats=int(sys.argv[2])
ok=[r for r in rows if r['status']=='ok']
if not ok: raise SystemExit('no successful producer-adaptive candidate')
res={r['residue'] for r in ok}
if len(res)!=1: raise SystemExit('FATAL producer-adaptive residue mismatch '+repr({(r['threshold'],r['repeat']):r['residue'] for r in ok}))
thresholds=sorted({r['threshold'] for r in ok},key=int)
score={}
for t in thresholds:
    vals=[float(r['wall_s']) for r in ok if r['threshold']==t]
    if len(vals)!=repeats: raise SystemExit(f'threshold {t} expected {repeats} repeats got {len(vals)}')
    score[t]=statistics.median(vals)
for t in sorted(thresholds,key=lambda x:score[x]):
    vals=','.join(r['wall_s'] for r in ok if r['threshold']==t)
    row=next(r for r in ok if r['threshold']==t)
    print('PAC_CANDIDATE','threshold='+t,'base_weight='+row['base_weight'],f'median_wall_s={score[t]:.9f}','samples='+vals,'residue='+row['residue'],file=sys.stderr)
b=min(thresholds,key=lambda t:score[t])
print(b,next(r['residue'] for r in ok if r['threshold']==b),f'{score[b]:.9f}')
PY
)

make_profile "$best_threshold" "$PROFILE_OUT"
cat >>"$PROFILE_OUT" <<EOF
ORBIT_N27_PRODUCER_ADAPTIVE_COLS=$best_threshold
ORBIT_N27_PRODUCER_ADAPTIVE_WALL_S=$best_wall
ORBIT_N27_PRODUCER_ADAPTIVE_BASE_WEIGHT=$BASE_WEIGHT
ORBIT_N27_PRODUCER_ADAPTIVE_REPEATS=$ADAPTIVE_REPEATS
EOF

echo "N27 PRODUCER ADAPTIVE SELECTED threshold=$best_threshold base_weight=$BASE_WEIGHT median_wall_s=$best_wall repeats=$ADAPTIVE_REPEATS residue=$best_residue profile=$PROFILE_OUT" >&2
cat "$SUMMARY" >&2
if [[ "$ADAPTIVE_RACE_ONLY" == 1 ]]; then
  echo "ADAPTIVE_RACE_ONLY=1: selected threshold=$best_threshold; final selector not run" >&2
  exit 0
fi

# The canonical selector still consumes this feature through the producer build
# wrapper environment; export the selected threshold explicitly for this run.
exec env PRODUCER_ADAPTIVE_COLS="$best_threshold" PROFILE_FILE="$PROFILE_OUT" REBUILD=1 "$BASE_RUN" 27 "$@"
