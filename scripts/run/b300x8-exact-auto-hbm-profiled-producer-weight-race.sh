#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'producer-weight race currently targets n=27' >&2; exit 2; }

BASE_RUN="$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled.sh"
PROFILE_IN="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21_n27pww.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_n27_producer_weight_race}"
WEIGHTS="${PRODUCER_WEIGHTS:-0 1 2 3 4}"
WEIGHT_REBUILD="${WEIGHT_REBUILD:-1}"
WEIGHT_RACE_ONLY="${WEIGHT_RACE_ONLY:-0}"
SUMMARY="${SUMMARY:-${PREFIX}.tsv}"
LOGDIR="${LOGDIR:-${PREFIX}_logs}"
mkdir -p "$LOGDIR" "$(dirname "$PROFILE_OUT")" "$(dirname "$SUMMARY")"

for x in WEIGHT_REBUILD WEIGHT_RACE_ONLY; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }
done
[[ -f "$PROFILE_IN" ]] || { echo "missing profile: $PROFILE_IN" >&2; exit 2; }
# shellcheck disable=SC1090
source "$PROFILE_IN"
ORBITCTA_FLAT_DYNAMIC="${ORBITCTA_FLAT_DYNAMIC:-0}"
ORBITCTA_FLAT_DYNAMIC_PIPE2="${ORBITCTA_FLAT_DYNAMIC_PIPE2:-0}"
ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP="${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP:-0}"
if [[ "$ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP" != 1 ]]; then
  echo 'producer-weight race skipped: selected profile does not use PIPE2 producer warp' >&2
  cp "$PROFILE_IN" "$PROFILE_OUT"
  if [[ "$WEIGHT_RACE_ONLY" == 1 ]]; then exit 0; fi
  exec env PROFILE_FILE="$PROFILE_OUT" "$BASE_RUN" 27 "$@"
fi
[[ "$ORBITCTA_FLAT_DYNAMIC" == 1 && "$ORBITCTA_FLAT_DYNAMIC_PIPE2" == 1 ]] || {
  echo 'producer-weight race requires dynamic PIPE2' >&2; exit 2;
}

read -r -a weight_list <<<"$WEIGHTS"
(( ${#weight_list[@]} > 0 )) || { echo 'PRODUCER_WEIGHTS is empty' >&2; exit 2; }
declare -A seen=()
for w in "${weight_list[@]}"; do
  case "$w" in 0|1|2|3|4) ;; *) echo "bad producer weight: $w" >&2; exit 2;; esac
  [[ -z "${seen[$w]+x}" ]] || { echo "duplicate producer weight: $w" >&2; exit 2; }
  seen[$w]=1
done

make_profile() {
  local weight="$1" out="$2"
  python3 - "$PROFILE_IN" "$out" "$weight" <<'PY'
import sys
src,out,w=sys.argv[1:]
key='ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT'
lines=open(src).read().splitlines(); dst=[]; found=False
for line in lines:
    if line.startswith(key+'='):
        dst.append(key+'='+w); found=True
    elif line.startswith('ORBIT_N27_PRODUCER_WORKER_WEIGHT=') or line.startswith('ORBIT_N27_PRODUCER_WEIGHT_WALL_S='):
        continue
    else:
        dst.append(line)
if not found: dst.append(key+'='+w)
open(out,'w').write('\n'.join(dst)+'\n')
PY
}

printf 'weight\tstatus\tresidue\twall_s\tprofile\tresult\n' >"$SUMMARY"
for w in "${weight_list[@]}"; do
  pf="$LOGDIR/profile_w${w}.env"
  wp="${PREFIX}_w${w}"
  make_profile "$w" "$pf"
  echo "=== n27 producer-weight smoke weight=$w ===" >&2
  set +e
  PROFILE_FILE="$pf" CANDIDATES=orbit_tuned SELECT_ONLY=1 REBUILD="$WEIGHT_REBUILD" PREFIX="$wp" \
    "$BASE_RUN" 27 >"$LOGDIR/w${w}.out" 2>"$LOGDIR/w${w}.err"
  rc=$?
  set -e
  if (( rc != 0 )); then
    printf '%s\tfailed:%s\tNA\tNA\t%s\t%s\n' "$w" "$rc" "$pf" "${wp}.tsv" >>"$SUMMARY"
    echo "producer-weight smoke failed weight=$w rc=$rc" >&2
    exit "$rc"
  fi
  result="${wp}.tsv"
  [[ -f "$result" ]] || { echo "missing result for weight=$w: $result" >&2; exit 3; }
  read -r residue wall < <(python3 - "$result" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
ok=[r for r in rows if r.get('backend')=='orbit_tuned' and r.get('status')=='ok']
if len(ok)!=1: raise SystemExit(f'expected one successful orbit_tuned row, got {len(ok)}')
print(ok[0]['residue'],ok[0]['wall_s'])
PY
)
  printf '%s\tok\t%s\t%s\t%s\t%s\n' "$w" "$residue" "$wall" "$pf" "$result" >>"$SUMMARY"
done

read -r best_weight best_residue best_wall < <(python3 - "$SUMMARY" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
ok=[r for r in rows if r['status']=='ok']
if not ok: raise SystemExit('no successful producer-weight candidate')
res={r['residue'] for r in ok}
if len(res)!=1: raise SystemExit('FATAL producer-weight residue mismatch '+repr({r['weight']:r['residue'] for r in ok}))
b=min(ok,key=lambda r:float(r['wall_s']))
for r in sorted(ok,key=lambda r:float(r['wall_s'])):
    print('PWW_CANDIDATE', 'weight='+r['weight'], 'wall_s='+r['wall_s'], 'residue='+r['residue'], file=sys.stderr)
print(b['weight'],b['residue'],b['wall_s'])
PY
)

make_profile "$best_weight" "$PROFILE_OUT"
cat >>"$PROFILE_OUT" <<EOF
ORBIT_N27_PRODUCER_WORKER_WEIGHT=$best_weight
ORBIT_N27_PRODUCER_WEIGHT_WALL_S=$best_wall
EOF

echo "N27 PRODUCER WEIGHT SELECTED weight=$best_weight wall_s=$best_wall residue=$best_residue profile=$PROFILE_OUT" >&2
cat "$SUMMARY" >&2
if [[ "$WEIGHT_RACE_ONLY" == 1 ]]; then
  echo "WEIGHT_RACE_ONLY=1: selected weight=$best_weight; final selector not run" >&2
  exit 0
fi
exec env PROFILE_FILE="$PROFILE_OUT" "$BASE_RUN" 27 "$@"
