#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
if (( $# > 0 )); then shift; fi
[[ "$N" == 27 ]] || { echo 'grand exact promotion currently targets n=27' >&2; exit 2; }

FIRSTPASS_PREFIX="${FIRSTPASS_PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"
SELECTED_ENV="${SELECTED_ENV:-${FIRSTPASS_PREFIX}.selected.env}"
CORE_PROMOTER="${CORE_PROMOTER:-$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-core.sh}"
[[ -s "$SELECTED_ENV" ]] || { echo "missing first-pass selection contract: $SELECTED_ENV" >&2; exit 2; }
[[ -f "$CORE_PROMOTER" ]] || { echo "missing exact promotion core: $CORE_PROMOTER" >&2; exit 2; }
# shellcheck disable=SC1090
source "$SELECTED_ENV"

stagel_keys=(
  B300_GRAND_SELECTED_STAGEL_ENABLED B300_GRAND_SELECTED_STAGEL_MIN_SPEEDUP
  B300_GRAND_SELECTED_STAGEL_ACCEPTED B300_GRAND_SELECTED_STAGEL_PROFILE
  B300_GRAND_SELECTED_STAGEL_SELF_GUARD B300_GRAND_SELECTED_STAGEL_MATE_GUARD
  B300_GRAND_SELECTED_STAGEL_STAGED_SPEEDUP B300_GRAND_SELECTED_STAGEL_SEARCH_PROFILES
)
stagem_keys=(
  B300_GRAND_SELECTED_STAGEM_ENABLED B300_GRAND_SELECTED_STAGEM_MIN_SPEEDUP
  B300_GRAND_SELECTED_STAGEM_ACCEPTED B300_GRAND_SELECTED_STAGEM_POLICY
  B300_GRAND_SELECTED_STAGEM_STAGED_SPEEDUP B300_GRAND_SELECTED_STAGEM_SEARCH_POLICIES
)
HAS_L=0
HAS_M=0
for key in "${stagel_keys[@]}"; do [[ -n "${!key+x}" ]] && HAS_L=1; done
for key in "${stagem_keys[@]}"; do [[ -n "${!key+x}" ]] && HAS_M=1; done
(( HAS_M == 0 || HAS_L == 1 )) || { echo 'Stage-M provenance exists without Stage-L provenance' >&2; exit 3; }

require_keys(){ local label="$1"; shift; local key; for key in "$@"; do [[ -n "${!key+x}" ]] || { echo "$label selected contract missing $key" >&2; exit 3; }; done; }
require_bool(){ local key="$1" v="${!1}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "bad $key=$v" >&2; exit 3; }; }
contains_word(){ local haystack="$1" needle="$2"; case " $haystack " in *" $needle "*) return 0;; *) return 1;; esac; }
validate_speedups(){ python3 - "$@" <<'PY'
import math,sys
for name,value in zip(sys.argv[1::2], sys.argv[2::2]):
    try: v=float(value)
    except ValueError: raise SystemExit(f'bad {name}={value}')
    if not math.isfinite(v) or v <= 0: raise SystemExit(f'bad {name}={value}')
    if name.endswith('_MIN_SPEEDUP') and v < 1.0: raise SystemExit(f'bad {name}={value}')
PY
}

if (( HAS_L )); then
  require_keys Stage-L "${stagel_keys[@]}"
  require_bool B300_GRAND_SELECTED_STAGEL_ENABLED
  require_bool B300_GRAND_SELECTED_STAGEL_ACCEPTED
  (( B300_GRAND_SELECTED_STAGEL_ENABLED || ! B300_GRAND_SELECTED_STAGEL_ACCEPTED )) || { echo 'Stage-L accepted while disabled' >&2; exit 3; }
  case "$B300_GRAND_SELECTED_STAGEL_PROFILE" in
    bb) expected_self=branch; expected_mate=branch ;;
    pb) expected_self=predicated; expected_mate=branch ;;
    bp) expected_self=branch; expected_mate=predicated ;;
    pp) expected_self=predicated; expected_mate=predicated ;;
    *) echo "bad Stage-L profile=$B300_GRAND_SELECTED_STAGEL_PROFILE" >&2; exit 3 ;;
  esac
  [[ "$B300_GRAND_SELECTED_STAGEL_SELF_GUARD" == "$expected_self" && "$B300_GRAND_SELECTED_STAGEL_MATE_GUARD" == "$expected_mate" ]] || { echo 'Stage-L profile/guard mapping mismatch' >&2; exit 3; }
  contains_word "$B300_GRAND_SELECTED_STAGEL_SEARCH_PROFILES" bb || { echo 'Stage-L search profiles omit bb control' >&2; exit 3; }
  contains_word "$B300_GRAND_SELECTED_STAGEL_SEARCH_PROFILES" "$B300_GRAND_SELECTED_STAGEL_PROFILE" || { echo 'Stage-L selected profile missing from search profiles' >&2; exit 3; }
  if [[ "$B300_GRAND_SELECTED_STAGEL_ACCEPTED" == 1 && "$B300_GRAND_SELECTED_STAGEL_PROFILE" == bb ]]; then echo 'accepted Stage L cannot retain bb control profile' >&2; exit 3; fi
  validate_speedups B300_GRAND_SELECTED_STAGEL_MIN_SPEEDUP "$B300_GRAND_SELECTED_STAGEL_MIN_SPEEDUP" B300_GRAND_SELECTED_STAGEL_STAGED_SPEEDUP "$B300_GRAND_SELECTED_STAGEL_STAGED_SPEEDUP"
  if [[ "$B300_GRAND_SELECTED_STAGEL_ACCEPTED" == 1 ]]; then python3 - "$B300_GRAND_SELECTED_STAGEL_STAGED_SPEEDUP" "$B300_GRAND_SELECTED_STAGEL_MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < float(sys.argv[2]): raise SystemExit('accepted Stage-L speedup is below threshold')
PY
  fi
fi

if (( HAS_M )); then
  require_keys Stage-M "${stagem_keys[@]}"
  require_bool B300_GRAND_SELECTED_STAGEM_ENABLED
  require_bool B300_GRAND_SELECTED_STAGEM_ACCEPTED
  (( B300_GRAND_SELECTED_STAGEM_ENABLED || ! B300_GRAND_SELECTED_STAGEM_ACCEPTED )) || { echo 'Stage-M accepted while disabled' >&2; exit 3; }
  if [[ "$B300_GRAND_SELECTED_STAGEM_ENABLED" == 1 && "$B300_GRAND_SELECTED_STAGEL_ENABLED" != 1 ]]; then echo 'Stage-M enabled while Stage-L disabled' >&2; exit 3; fi
  case "$B300_GRAND_SELECTED_STAGEM_POLICY" in default|cg|cs) ;; *) echo "bad Stage-M policy=$B300_GRAND_SELECTED_STAGEM_POLICY" >&2; exit 3;; esac
  contains_word "$B300_GRAND_SELECTED_STAGEM_SEARCH_POLICIES" default || { echo 'Stage-M search policies omit default control' >&2; exit 3; }
  contains_word "$B300_GRAND_SELECTED_STAGEM_SEARCH_POLICIES" "$B300_GRAND_SELECTED_STAGEM_POLICY" || { echo 'Stage-M selected policy missing from search policies' >&2; exit 3; }
  validate_speedups B300_GRAND_SELECTED_STAGEM_MIN_SPEEDUP "$B300_GRAND_SELECTED_STAGEM_MIN_SPEEDUP" B300_GRAND_SELECTED_STAGEM_STAGED_SPEEDUP "$B300_GRAND_SELECTED_STAGEM_STAGED_SPEEDUP"
  if [[ "$B300_GRAND_SELECTED_STAGEM_ACCEPTED" == 1 ]]; then
    [[ "$B300_GRAND_SELECTED_STAGEL_ACCEPTED" == 1 ]] || { echo 'Stage-M accepted without Stage-L acceptance' >&2; exit 3; }
    case "$B300_GRAND_SELECTED_STAGEM_POLICY" in cg|cs) ;; *) echo 'accepted Stage-M policy must be cg/cs' >&2; exit 3;; esac
    python3 - "$B300_GRAND_SELECTED_STAGEM_STAGED_SPEEDUP" "$B300_GRAND_SELECTED_STAGEM_MIN_SPEEDUP" <<'PY'
import sys
if float(sys.argv[1]) < float(sys.argv[2]): raise SystemExit('accepted Stage-M speedup is below threshold')
PY
  fi
fi

if (( HAS_L || HAS_M )); then
  [[ -n "${B300_GRAND_SELECTED_GRAND_SUMMARY_ENV+x}" && -n "${B300_GRAND_SELECTED_GRAND_SUMMARY_SHA256+x}" ]] || { echo 'L/M selection missing grand-summary provenance' >&2; exit 3; }
  summary="$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV"
  [[ -s "$summary" ]] || { echo "grand summary missing: $summary" >&2; exit 3; }
  command -v sha256sum >/dev/null || { echo 'sha256sum required' >&2; exit 2; }
  actual_summary_sha="$(sha256sum "$summary" | awk '{print $1}')"
  [[ "$actual_summary_sha" == "$B300_GRAND_SELECTED_GRAND_SUMMARY_SHA256" ]] || { echo 'grand summary fingerprint mismatch' >&2; exit 4; }
  # shellcheck disable=SC1090
  source "$summary"
  [[ "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || { echo 'L/M grand summary does not prove one complete-prime race' >&2; exit 3; }
  if (( HAS_L )); then
    [[ "${B300_GRAND_STAGEL_INTEGRATED:-0}" == 1 ]] || { echo 'grand summary missing Stage-L integration' >&2; exit 3; }
    [[ "${B300_GRAND_STAGEL_OK:-0}" == "$B300_GRAND_SELECTED_STAGEL_ACCEPTED" ]] || { echo 'Stage-L accepted flag differs from grand summary' >&2; exit 3; }
    [[ "${B300_GRAND_STAGEL_PROFILE:-bb}" == "$B300_GRAND_SELECTED_STAGEL_PROFILE" ]] || { echo 'Stage-L profile differs from grand summary' >&2; exit 3; }
    [[ "${B300_GRAND_STAGEL_SELF_GUARD:-branch}" == "$B300_GRAND_SELECTED_STAGEL_SELF_GUARD" && "${B300_GRAND_STAGEL_MATE_GUARD:-branch}" == "$B300_GRAND_SELECTED_STAGEL_MATE_GUARD" ]] || { echo 'Stage-L guards differ from grand summary' >&2; exit 3; }
  fi
  if (( HAS_M )); then
    [[ "${B300_GRAND_STAGEM_INTEGRATED:-0}" == 1 ]] || { echo 'grand summary missing Stage-M integration' >&2; exit 3; }
    [[ "${B300_GRAND_STAGEM_OK:-0}" == "$B300_GRAND_SELECTED_STAGEM_ACCEPTED" ]] || { echo 'Stage-M accepted flag differs from grand summary' >&2; exit 3; }
    [[ "${B300_GRAND_STAGEM_POLICY:-default}" == "$B300_GRAND_SELECTED_STAGEM_POLICY" ]] || { echo 'Stage-M policy differs from grand summary' >&2; exit 3; }
  fi
fi

# Static compatibility markers consumed by existing promotion preflights; implementation remains in CORE_PROMOTER.
# B300_GRAND_SELECTED_VALIDATED
# B300_GRAND_SELECTED_BINARY_SHA256
# B300_GRAND_SELECTED_PROFILE_SHA256
# B300_GRAND_SELECTED_RACE_RESULT_SHA256
# unsupported grand selection schema
# schema-3 grand summary Stage-I/J/K single-race proof missing
# selected binary fingerprint mismatch
# selected profile fingerprint mismatch
# single-pass TSV fingerprint mismatch
# checkpoint fingerprint mismatch
# race winner contract mismatch
# solver_fingerprint
# ALLOW_HEAD_DRIFT
# ALLOW_DIRTY_FIRSTPASS
# ALLOW_WORKTREE_DIRTY
# DRY_RUN
# solve_b300_exact_batch.py
# B300 GRAND EXACT PROMOTION VALIDATED

echo "Stage-L/M exact provenance OK has_l=$HAS_L has_m=$HAS_M" >&2
exec env SELECTED_ENV="$SELECTED_ENV" "$CORE_PROMOTER" 27 "$@"
