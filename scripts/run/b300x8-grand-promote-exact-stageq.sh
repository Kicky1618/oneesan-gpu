#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || exit 2
FIRSTPASS_PREFIX="${FIRSTPASS_PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"
SELECTED_ENV="${SELECTED_ENV:-${FIRSTPASS_PREFIX}.selected.env}"
BASE_PROMOTER="${BASE_PROMOTER:-$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stagep.sh}"
[[ -s "$SELECTED_ENV" && -f "$BASE_PROMOTER" ]] || exit 2
command -v sha256sum >/dev/null || exit 2
# shellcheck disable=SC1090
source "$SELECTED_ENV"
HAS_Q=0; [[ -n "${B300_GRAND_SELECTED_STAGEQ_ENABLED+x}" ]] && HAS_Q=1
if ((HAS_Q)); then
  req=(B300_GRAND_SELECTED_STAGEQ_ENABLED B300_GRAND_SELECTED_STAGEQ_MIN_SPEEDUP B300_GRAND_SELECTED_STAGEQ_ACCEPTED B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND B300_GRAND_SELECTED_STAGEQ_UPSTREAM_PAIR_L2_BYTES B300_GRAND_SELECTED_STAGEQ_UPSTREAM_BLOCK_L2_BYTES B300_GRAND_SELECTED_STAGEQ_PAIR_L2_BYTES B300_GRAND_SELECTED_STAGEQ_BLOCK_L2_BYTES B300_GRAND_SELECTED_STAGEQ_STAGED_SPEEDUP B300_GRAND_SELECTED_STAGEQ_SEARCH_PAIR_L2 B300_GRAND_SELECTED_STAGEQ_SEARCH_BLOCK_L2)
  for k in "${req[@]}"; do [[ -n "${!k+x}" ]] || { echo "Stage-Q selected contract missing $k" >&2; exit 3; }; done
  [[ "$B300_GRAND_SELECTED_STAGEQ_ENABLED" == 0 || "$B300_GRAND_SELECTED_STAGEQ_ENABLED" == 1 ]] || exit 3
  [[ "$B300_GRAND_SELECTED_STAGEQ_ACCEPTED" == 0 || "$B300_GRAND_SELECTED_STAGEQ_ACCEPTED" == 1 ]] || exit 3
  ((B300_GRAND_SELECTED_STAGEQ_ENABLED || !B300_GRAND_SELECTED_STAGEQ_ACCEPTED)) || { echo 'Stage-Q accepted while disabled' >&2; exit 3; }
  case "$B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND" in ''|stagen|stageo|stagep) ;; *) echo 'bad Stage-Q upstream kind' >&2; exit 3;; esac
  for b in "$B300_GRAND_SELECTED_STAGEQ_UPSTREAM_PAIR_L2_BYTES" "$B300_GRAND_SELECTED_STAGEQ_UPSTREAM_BLOCK_L2_BYTES" "$B300_GRAND_SELECTED_STAGEQ_PAIR_L2_BYTES" "$B300_GRAND_SELECTED_STAGEQ_BLOCK_L2_BYTES"; do case "$b" in 0|64|128|256) ;; *) echo "bad Stage-Q Count L2=$b" >&2; exit 3;; esac; done
  for list in B300_GRAND_SELECTED_STAGEQ_SEARCH_PAIR_L2 B300_GRAND_SELECTED_STAGEQ_SEARCH_BLOCK_L2; do
    n=0; for b in ${!list}; do case "$b" in 0|64|128|256) ;; *) echo "bad Stage-Q search L2=$b" >&2; exit 3;; esac; n=$((n+1)); done; ((n>0)) || { echo "$list empty" >&2; exit 3; }
  done
  python3 - "$B300_GRAND_SELECTED_STAGEQ_MIN_SPEEDUP" "$B300_GRAND_SELECTED_STAGEQ_STAGED_SPEEDUP" <<'PY'
import sys
m,s=map(float,sys.argv[1:])
if m<1.0 or s<=0: raise SystemExit('bad Stage-Q speed contract')
PY
  if [[ "$B300_GRAND_SELECTED_STAGEQ_ACCEPTED" == 1 ]]; then
    [[ "${B300_GRAND_SELECTED_STAGEN_ACCEPTED:-0}" == 1 ]] || { echo 'Stage Q accepted without Stage-N acceptance' >&2; exit 3; }
    if [[ "${B300_GRAND_SELECTED_STAGEP_ACCEPTED:-0}" == 1 ]]; then
      [[ "$B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND" == stagep ]] || { echo 'Stage Q ignored accepted Stage P upstream' >&2; exit 3; }
    elif [[ "${B300_GRAND_SELECTED_STAGEO_ACCEPTED:-0}" == 1 ]]; then
      [[ "$B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND" == stageo ]] || { echo 'Stage Q ignored accepted Stage O upstream' >&2; exit 3; }
    else
      [[ "$B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND" == stagen ]] || { echo 'Stage Q upstream must fall back to Stage N' >&2; exit 3; }
    fi
    [[ "$B300_GRAND_SELECTED_STAGEQ_PAIR_L2_BYTES" != "$B300_GRAND_SELECTED_STAGEQ_UPSTREAM_PAIR_L2_BYTES" || "$B300_GRAND_SELECTED_STAGEQ_BLOCK_L2_BYTES" != "$B300_GRAND_SELECTED_STAGEQ_UPSTREAM_BLOCK_L2_BYTES" ]] || { echo 'accepted Stage Q retained exact upstream Count L2 tuple' >&2; exit 3; }
    python3 - "$B300_GRAND_SELECTED_STAGEQ_MIN_SPEEDUP" "$B300_GRAND_SELECTED_STAGEQ_STAGED_SPEEDUP" <<'PY'
import sys
m,s=map(float,sys.argv[1:])
if s<m: raise SystemExit('accepted Stage-Q speedup below threshold')
PY
  fi
  [[ -n "${B300_GRAND_SELECTED_GRAND_SUMMARY_ENV+x}" && -s "$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV" ]] || { echo 'Stage-Q selection missing grand summary' >&2; exit 3; }
  # shellcheck disable=SC1090
  source "$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV"
  [[ "${B300_GRAND_STAGEQ_INTEGRATED:-0}" == 1 && "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || { echo 'grand summary missing Stage-Q single-race integration' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEQ_OK:-0}" == "$B300_GRAND_SELECTED_STAGEQ_ACCEPTED" ]] || { echo 'Stage-Q accepted flag differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEQ_UPSTREAM_KIND:-}" == "$B300_GRAND_SELECTED_STAGEQ_UPSTREAM_KIND" ]] || { echo 'Stage-Q upstream differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEQ_UPSTREAM_PAIR_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGEQ_UPSTREAM_PAIR_L2_BYTES" && "${B300_GRAND_STAGEQ_UPSTREAM_BLOCK_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGEQ_UPSTREAM_BLOCK_L2_BYTES" ]] || { echo 'Stage-Q upstream Count L2 differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEQ_PAIR_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGEQ_PAIR_L2_BYTES" && "${B300_GRAND_STAGEQ_BLOCK_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGEQ_BLOCK_L2_BYTES" ]] || { echo 'Stage-Q selected Count L2 differs from grand summary' >&2; exit 3; }
  if [[ "$B300_GRAND_SELECTED_STAGEQ_ACCEPTED" == 1 ]]; then
    [[ -s "${B300_GRAND_STAGEQ_MANIFEST:-}" ]] || { echo 'accepted Stage-Q manifest missing from grand summary' >&2; exit 3; }
    sha256sum -c "$B300_GRAND_STAGEQ_MANIFEST" >/dev/null || { echo 'Stage-Q promotion manifest failed before exact continuation' >&2; exit 3; }
  fi
fi
echo "Stage-Q exact provenance OK has_q=$HAS_Q" >&2
exec env SELECTED_ENV="$SELECTED_ENV" "$BASE_PROMOTER" 27 "$@"
