#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'producer coordinate refinement targets n=27' >&2; exit 2; }

PROFILE_IN="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
PROFILE_OUT="${PROFILE_OUT:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21_n27coord.env}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_n27_producer_coordinate}"
ROUNDS="${COORDINATE_ROUNDS:-1}"
REPEATS="${COORDINATE_REPEATS:-1}"
REBUILD="${COORDINATE_REBUILD:-1}"
COORDINATE_RACE_ONLY="${COORDINATE_RACE_ONLY:-1}"
mkdir -p "$(dirname "$PROFILE_OUT")"

[[ -f "$PROFILE_IN" ]] || { echo "missing profile: $PROFILE_IN" >&2; exit 2; }
[[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]] || { echo 'COORDINATE_ROUNDS must be positive integer' >&2; exit 2; }
[[ "$REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo 'COORDINATE_REPEATS must be positive integer' >&2; exit 2; }
for x in REBUILD COORDINATE_RACE_ONLY; do v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0/1" >&2; exit 2; }; done

getv(){ local k="$1" f="$2"; sed -nE "s/^${k}=([^[:space:]]+).*/\\1/p" "$f" | tail -n1; }
read_pair(){
  local f="$1" w t
  w="$(getv ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT "$f")"; w="${w:-0}"
  t="$(getv ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS "$f")"; t="${t:-0}"
  case "$w" in 0|1|2|3|4) ;; *) echo "bad producer weight in $f: $w" >&2; exit 3;; esac
  [[ "$t" =~ ^[0-9]+$ ]] || { echo "bad adaptive threshold in $f: $t" >&2; exit 3; }
  printf '%s %s\n' "$w" "$t"
}

cur="$PROFILE_IN"
for ((round=1; round<=ROUNDS; ++round)); do
  read -r old_w old_t < <(read_pair "$cur")
  echo "=== producer coordinate round=$round/$ROUNDS start weight=$old_w threshold=$old_t ===" >&2

  wp="${PREFIX}.r${round}.weight.env"
  PROFILE_FILE="$cur" PROFILE_OUT="$wp" PREFIX="${PREFIX}.r${round}.weight" \
    WEIGHT_ADAPTIVE_COLS="$old_t" WEIGHT_REPEATS="$REPEATS" WEIGHT_REBUILD="$REBUILD" WEIGHT_RACE_ONLY=1 \
    bash "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled-producer-weight-race.sh" 27
  [[ -s "$wp" ]] || { echo "coordinate weight profile missing: $wp" >&2; exit 4; }

  ap="${PREFIX}.r${round}.adaptive.env"
  PROFILE_FILE="$wp" PROFILE_OUT="$ap" PREFIX="${PREFIX}.r${round}.adaptive" \
    ADAPTIVE_REPEATS="$REPEATS" ADAPTIVE_REBUILD="$REBUILD" ADAPTIVE_RACE_ONLY=1 \
    bash "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled-producer-adaptive-race.sh" 27
  [[ -s "$ap" ]] || { echo "coordinate adaptive profile missing: $ap" >&2; exit 4; }

  read -r new_w new_t < <(read_pair "$ap")
  echo "PRODUCER COORDINATE round=$round old_weight=$old_w old_threshold=$old_t new_weight=$new_w new_threshold=$new_t" >&2
  cur="$ap"
  if [[ "$new_w" == "$old_w" && "$new_t" == "$old_t" ]]; then
    echo "PRODUCER COORDINATE converged round=$round weight=$new_w threshold=$new_t" >&2
    break
  fi
done

cp "$cur" "$PROFILE_OUT"
read -r final_w final_t < <(read_pair "$PROFILE_OUT")
cat >>"$PROFILE_OUT" <<EOF
ORBIT_N27_PRODUCER_COORDINATE_WEIGHT=$final_w
ORBIT_N27_PRODUCER_COORDINATE_ADAPTIVE_COLS=$final_t
ORBIT_N27_PRODUCER_COORDINATE_ROUNDS_MAX=$ROUNDS
ORBIT_N27_PRODUCER_COORDINATE_REPEATS=$REPEATS
EOF

echo "N27 PRODUCER COORDINATE SELECTED weight=$final_w threshold=$final_t profile=$PROFILE_OUT" >&2
if [[ "$COORDINATE_RACE_ONLY" == 1 ]]; then
  exit 0
fi
# Standalone continuation uses the adaptive wrapper so the selected threshold is
# explicitly carried even though the legacy canonical selector lacks pac in its
# binary fingerprint. The joint single-pass selector uses the build-only shim.
exec env PROFILE_FILE="$PROFILE_OUT" REBUILD=1 "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled-producer-adaptive.sh" 27 "$@"
