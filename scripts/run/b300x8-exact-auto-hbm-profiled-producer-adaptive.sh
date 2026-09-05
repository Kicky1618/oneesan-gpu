#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'adaptive profiled wrapper targets n=27' >&2; exit 2; }
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
[[ -f "$PROFILE_FILE" ]] || { echo "missing PROFILE_FILE=$PROFILE_FILE" >&2; exit 2; }
# shellcheck disable=SC1090
source "$PROFILE_FILE"

THRESHOLD="${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_ADAPTIVE_COLS:-0}"
PRODUCER="${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WARP:-0}"
WEIGHT="${ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_WORKER_WEIGHT:-0}"
PIPE2="${ORBITCTA_FLAT_DYNAMIC_PIPE2:-0}"
DYNAMIC="${ORBITCTA_FLAT_DYNAMIC:-0}"
[[ "$THRESHOLD" =~ ^[0-9]+$ ]] || { echo 'bad producer adaptive threshold in profile' >&2; exit 2; }
case "$PRODUCER" in 0|1) ;; *) exit 2;; esac
case "$WEIGHT" in 0|1|2|3|4) ;; *) exit 2;; esac
if (( THRESHOLD > 0 )); then
  [[ "$DYNAMIC" == 1 && "$PIPE2" == 1 && "$PRODUCER" == 1 ]] || { echo 'adaptive producer threshold requires selected dynamic PIPE2 producer warp' >&2; exit 2; }
  [[ "$WEIGHT" != 1 ]] || { echo 'adaptive producer threshold with base weight=1 is redundant' >&2; exit 2; }
fi

export PRODUCER_ADAPTIVE_COLS="$THRESHOLD"
export PROFILE_FILE
echo "adaptive producer n27 threshold=$THRESHOLD producer=$PRODUCER base_weight=$WEIGHT profile=$PROFILE_FILE" >&2
exec "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm-profiled.sh" 27 "$@"
