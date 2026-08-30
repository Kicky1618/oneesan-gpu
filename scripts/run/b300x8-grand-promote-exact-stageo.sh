#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'Stage-O exact promotion targets n=27' >&2; exit 2; }
FIRSTPASS_PREFIX="${FIRSTPASS_PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"
SELECTED_ENV="${SELECTED_ENV:-${FIRSTPASS_PREFIX}.selected.env}"
BASE_PROMOTER="${BASE_PROMOTER:-$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stagen.sh}"
[[ -s "$SELECTED_ENV" ]] || { echo "missing selected env=$SELECTED_ENV" >&2; exit 2; }
[[ -f "$BASE_PROMOTER" ]] || { echo "missing Stage-N exact promoter=$BASE_PROMOTER" >&2; exit 2; }
command -v sha256sum >/dev/null || exit 2
# shellcheck disable=SC1090
source "$SELECTED_ENV"

HAS_O=0
[[ -n "${B300_GRAND_SELECTED_STAGEO_ENABLED+x}" ]] && HAS_O=1
if ((HAS_O)); then
  req=(B300_GRAND_SELECTED_STAGEO_ENABLED B300_GRAND_SELECTED_STAGEO_MIN_SPEEDUP B300_GRAND_SELECTED_STAGEO_ACCEPTED B300_GRAND_SELECTED_STAGEO_PAIR_L2_BYTES B300_GRAND_SELECTED_STAGEO_BLOCK_L2_BYTES B300_GRAND_SELECTED_STAGEO_BASE_PAIR_L2_BYTES B300_GRAND_SELECTED_STAGEO_BASE_BLOCK_L2_BYTES B300_GRAND_SELECTED_STAGEO_STAGED_SPEEDUP B300_GRAND_SELECTED_STAGEO_SEARCH_PAIR_L2 B300_GRAND_SELECTED_STAGEO_SEARCH_BLOCK_L2)
  for k in "${req[@]}"; do [[ -n "${!k+x}" ]] || { echo "Stage-O selected contract missing $k" >&2; exit 3; }; done
  [[ "$B300_GRAND_SELECTED_STAGEO_ENABLED" == 0 || "$B300_GRAND_SELECTED_STAGEO_ENABLED" == 1 ]] || { echo 'bad Stage-O enabled flag' >&2; exit 3; }
  [[ "$B300_GRAND_SELECTED_STAGEO_ACCEPTED" == 0 || "$B300_GRAND_SELECTED_STAGEO_ACCEPTED" == 1 ]] || { echo 'bad Stage-O accepted flag' >&2; exit 3; }
  (( B300_GRAND_SELECTED_STAGEO_ENABLED || ! B300_GRAND_SELECTED_STAGEO_ACCEPTED )) || { echo 'Stage-O accepted while disabled' >&2; exit 3; }
  for b in "$B300_GRAND_SELECTED_STAGEO_PAIR_L2_BYTES" "$B300_GRAND_SELECTED_STAGEO_BLOCK_L2_BYTES" "$B300_GRAND_SELECTED_STAGEO_BASE_PAIR_L2_BYTES" "$B300_GRAND_SELECTED_STAGEO_BASE_BLOCK_L2_BYTES"; do case "$b" in 0|64|128|256) ;; *) echo "bad Stage-O L2 bytes=$b" >&2; exit 3;; esac; done
  python3 - "$B300_GRAND_SELECTED_STAGEO_MIN_SPEEDUP" "$B300_GRAND_SELECTED_STAGEO_STAGED_SPEEDUP" <<'PY'
import sys
m,s=map(float,sys.argv[1:])
if m < 1.0: raise SystemExit('Stage-O minimum speedup must be >=1')
if s <= 0.0: raise SystemExit('Stage-O staged speedup must be positive')
PY
  pair_seen=0; for b in $B300_GRAND_SELECTED_STAGEO_SEARCH_PAIR_L2; do case "$b" in 0|64|128|256) ;; *) echo "bad Stage-O pair search L2=$b" >&2; exit 3;; esac; [[ "$b" == "$B300_GRAND_SELECTED_STAGEO_BASE_PAIR_L2_BYTES" ]] && pair_seen=1; done
  block_seen=0; for b in $B300_GRAND_SELECTED_STAGEO_SEARCH_BLOCK_L2; do case "$b" in 0|64|128|256) ;; *) echo "bad Stage-O block search L2=$b" >&2; exit 3;; esac; [[ "$b" == "$B300_GRAND_SELECTED_STAGEO_BASE_BLOCK_L2_BYTES" ]] && block_seen=1; done
  ((pair_seen&&block_seen)) || { echo 'Stage-O search L2 set omits inherited baseline' >&2; exit 3; }

  if [[ "$B300_GRAND_SELECTED_STAGEO_ACCEPTED" == 1 ]]; then
    [[ "${B300_GRAND_SELECTED_STAGEN_ACCEPTED:-0}" == 1 ]] || { echo 'Stage O accepted without Stage-N acceptance' >&2; exit 3; }
    [[ "$B300_GRAND_SELECTED_STAGEO_PAIR_L2_BYTES" != "$B300_GRAND_SELECTED_STAGEO_BASE_PAIR_L2_BYTES" || "$B300_GRAND_SELECTED_STAGEO_BLOCK_L2_BYTES" != "$B300_GRAND_SELECTED_STAGEO_BASE_BLOCK_L2_BYTES" ]] || { echo 'accepted Stage O retained inherited L2 baseline' >&2; exit 3; }
    python3 - "$B300_GRAND_SELECTED_STAGEO_MIN_SPEEDUP" "$B300_GRAND_SELECTED_STAGEO_STAGED_SPEEDUP" <<'PY'
import sys
m,s=map(float,sys.argv[1:])
if s < m: raise SystemExit('accepted Stage-O speedup below threshold')
PY
    case "${B300_GRAND_SELECTED_STAGEN_PAIR_POLICY:-default}" in cg) ;; *) [[ "$B300_GRAND_SELECTED_STAGEO_PAIR_L2_BYTES" == 0 && "$B300_GRAND_SELECTED_STAGEO_BASE_PAIR_L2_BYTES" == 0 ]] || { echo 'Stage O changed L2 on non-CG pair axis' >&2; exit 3; };; esac
    case "${B300_GRAND_SELECTED_STAGEN_BLOCK_POLICY:-default}" in cg) ;; *) [[ "$B300_GRAND_SELECTED_STAGEO_BLOCK_L2_BYTES" == 0 && "$B300_GRAND_SELECTED_STAGEO_BASE_BLOCK_L2_BYTES" == 0 ]] || { echo 'Stage O changed L2 on non-CG block axis' >&2; exit 3; };; esac
  fi

  [[ -n "${B300_GRAND_SELECTED_GRAND_SUMMARY_ENV+x}" && -s "$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV" ]] || { echo 'Stage-O selection missing grand summary' >&2; exit 3; }
  # shellcheck disable=SC1090
  source "$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV"
  [[ "${B300_GRAND_STAGEO_INTEGRATED:-0}" == 1 ]] || { echo 'grand summary missing Stage-O integration' >&2; exit 3; }
  [[ "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || { echo 'Stage-O grand summary does not prove one complete-prime race' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEO_OK:-0}" == "$B300_GRAND_SELECTED_STAGEO_ACCEPTED" ]] || { echo 'Stage-O accepted flag differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEO_PAIR_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGEO_PAIR_L2_BYTES" ]] || { echo 'Stage-O pair L2 differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEO_BLOCK_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGEO_BLOCK_L2_BYTES" ]] || { echo 'Stage-O block L2 differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEO_BASE_PAIR_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGEO_BASE_PAIR_L2_BYTES" && "${B300_GRAND_STAGEO_BASE_BLOCK_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGEO_BASE_BLOCK_L2_BYTES" ]] || { echo 'Stage-O inherited L2 baseline differs from grand summary' >&2; exit 3; }
  if [[ "$B300_GRAND_SELECTED_STAGEO_ACCEPTED" == 1 ]]; then
    [[ -s "${B300_GRAND_STAGEO_MANIFEST:-}" ]] || { echo 'accepted Stage-O manifest missing from grand summary' >&2; exit 3; }
    sha256sum -c "$B300_GRAND_STAGEO_MANIFEST" >/dev/null || { echo 'Stage-O promotion manifest failed before exact continuation' >&2; exit 3; }
  fi
fi

echo "Stage-O exact provenance OK has_o=$HAS_O" >&2
exec env SELECTED_ENV="$SELECTED_ENV" "$BASE_PROMOTER" 27 "$@"
