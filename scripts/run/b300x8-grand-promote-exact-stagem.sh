#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'Stage-M exact promotion targets n=27' >&2; exit 2; }
FIRSTPASS_PREFIX="${FIRSTPASS_PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"
SELECTED_ENV="${SELECTED_ENV:-${FIRSTPASS_PREFIX}.selected.env}"
BASE_PROMOTER="${BASE_PROMOTER:-$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact.sh}"
[[ -s "$SELECTED_ENV" ]] || { echo "missing selected env=$SELECTED_ENV" >&2; exit 2; }
[[ -f "$BASE_PROMOTER" ]] || { echo "missing base exact promoter=$BASE_PROMOTER" >&2; exit 2; }
command -v sha256sum >/dev/null || exit 2
# shellcheck disable=SC1090
source "$SELECTED_ENV"

# Old schema-3 selections did not expose Stage L/M fields. Keep them compatible;
# when the new fields are present, validate the complete semantic chain.
HAS_L=0; HAS_M=0
[[ -n "${B300_GRAND_SELECTED_STAGEL_ENABLED+x}" ]] && HAS_L=1
[[ -n "${B300_GRAND_SELECTED_STAGEM_ENABLED+x}" ]] && HAS_M=1
if ((HAS_M && !HAS_L)); then
  echo 'Stage-M provenance exists without Stage-L provenance' >&2; exit 3
fi
if ((HAS_L)); then
  req_l=(B300_GRAND_SELECTED_STAGEL_ENABLED B300_GRAND_SELECTED_STAGEL_MIN_SPEEDUP B300_GRAND_SELECTED_STAGEL_ACCEPTED B300_GRAND_SELECTED_STAGEL_PROFILE B300_GRAND_SELECTED_STAGEL_SELF_GUARD B300_GRAND_SELECTED_STAGEL_MATE_GUARD B300_GRAND_SELECTED_STAGEL_STAGED_SPEEDUP B300_GRAND_SELECTED_STAGEL_SEARCH_PROFILES)
  for k in "${req_l[@]}"; do [[ -n "${!k+x}" ]] || { echo "Stage-L selected contract missing $k" >&2; exit 3; }; done
  [[ "$B300_GRAND_SELECTED_STAGEL_ENABLED" == 0 || "$B300_GRAND_SELECTED_STAGEL_ENABLED" == 1 ]] || exit 3
  [[ "$B300_GRAND_SELECTED_STAGEL_ACCEPTED" == 0 || "$B300_GRAND_SELECTED_STAGEL_ACCEPTED" == 1 ]] || exit 3
  (( B300_GRAND_SELECTED_STAGEL_ENABLED || ! B300_GRAND_SELECTED_STAGEL_ACCEPTED )) || { echo 'Stage-L accepted while disabled' >&2; exit 3; }
  case "$B300_GRAND_SELECTED_STAGEL_PROFILE" in bb|pb|bp|pp) ;; *) echo 'bad Stage-L profile' >&2; exit 3;; esac
  for g in "$B300_GRAND_SELECTED_STAGEL_SELF_GUARD" "$B300_GRAND_SELECTED_STAGEL_MATE_GUARD"; do case "$g" in branch|predicated) ;; *) echo "bad Stage-L guard=$g" >&2; exit 3;; esac; done
  if [[ "$B300_GRAND_SELECTED_STAGEL_ACCEPTED" == 1 && "$B300_GRAND_SELECTED_STAGEL_PROFILE" == bb ]]; then
    echo 'accepted Stage L cannot retain bb control profile' >&2; exit 3
  fi
fi
if ((HAS_M)); then
  req_m=(B300_GRAND_SELECTED_STAGEM_ENABLED B300_GRAND_SELECTED_STAGEM_MIN_SPEEDUP B300_GRAND_SELECTED_STAGEM_ACCEPTED B300_GRAND_SELECTED_STAGEM_POLICY B300_GRAND_SELECTED_STAGEM_STAGED_SPEEDUP B300_GRAND_SELECTED_STAGEM_SEARCH_POLICIES)
  for k in "${req_m[@]}"; do [[ -n "${!k+x}" ]] || { echo "Stage-M selected contract missing $k" >&2; exit 3; }; done
  [[ "$B300_GRAND_SELECTED_STAGEM_ENABLED" == 0 || "$B300_GRAND_SELECTED_STAGEM_ENABLED" == 1 ]] || exit 3
  [[ "$B300_GRAND_SELECTED_STAGEM_ACCEPTED" == 0 || "$B300_GRAND_SELECTED_STAGEM_ACCEPTED" == 1 ]] || exit 3
  (( B300_GRAND_SELECTED_STAGEM_ENABLED || ! B300_GRAND_SELECTED_STAGEM_ACCEPTED )) || { echo 'Stage-M accepted while disabled' >&2; exit 3; }
  if [[ "$B300_GRAND_SELECTED_STAGEM_ACCEPTED" == 1 ]]; then
    [[ "$B300_GRAND_SELECTED_STAGEL_ACCEPTED" == 1 ]] || { echo 'Stage-M accepted without Stage-L acceptance' >&2; exit 3; }
    case "$B300_GRAND_SELECTED_STAGEM_POLICY" in cg|cs) ;; *) echo 'accepted Stage-M policy must be cg/cs' >&2; exit 3;; esac
  else
    case "$B300_GRAND_SELECTED_STAGEM_POLICY" in default|cg|cs) ;; *) echo 'bad Stage-M policy' >&2; exit 3;; esac
  fi
fi

if ((HAS_L || HAS_M)); then
  [[ -n "${B300_GRAND_SELECTED_GRAND_SUMMARY_ENV+x}" && -s "$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV" ]] || { echo 'L/M selection missing grand summary' >&2; exit 3; }
  # shellcheck disable=SC1090
  source "$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV"
  [[ "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || { echo 'L/M grand summary does not prove one complete-prime race' >&2; exit 3; }
  if ((HAS_L)); then
    [[ "${B300_GRAND_STAGEL_INTEGRATED:-0}" == 1 ]] || { echo 'grand summary missing Stage-L integration' >&2; exit 3; }
    [[ "${B300_GRAND_STAGEL_OK:-0}" == "$B300_GRAND_SELECTED_STAGEL_ACCEPTED" ]] || { echo 'Stage-L accepted flag differs from grand summary' >&2; exit 3; }
    [[ "${B300_GRAND_STAGEL_PROFILE:-bb}" == "$B300_GRAND_SELECTED_STAGEL_PROFILE" ]] || { echo 'Stage-L profile differs from grand summary' >&2; exit 3; }
    [[ "${B300_GRAND_STAGEL_SELF_GUARD:-branch}" == "$B300_GRAND_SELECTED_STAGEL_SELF_GUARD" && "${B300_GRAND_STAGEL_MATE_GUARD:-branch}" == "$B300_GRAND_SELECTED_STAGEL_MATE_GUARD" ]] || { echo 'Stage-L guards differ from grand summary' >&2; exit 3; }
  fi
  if ((HAS_M)); then
    [[ "${B300_GRAND_STAGEM_INTEGRATED:-0}" == 1 ]] || { echo 'grand summary missing Stage-M integration' >&2; exit 3; }
    [[ "${B300_GRAND_STAGEM_OK:-0}" == "$B300_GRAND_SELECTED_STAGEM_ACCEPTED" ]] || { echo 'Stage-M accepted flag differs from grand summary' >&2; exit 3; }
    [[ "${B300_GRAND_STAGEM_POLICY:-default}" == "$B300_GRAND_SELECTED_STAGEM_POLICY" ]] || { echo 'Stage-M policy differs from grand summary' >&2; exit 3; }
  fi
fi

echo "Stage-L/M exact provenance OK has_l=$HAS_L has_m=$HAS_M" >&2
exec env SELECTED_ENV="$SELECTED_ENV" "$BASE_PROMOTER" 27 "$@"
