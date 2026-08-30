#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'Stage-N exact promotion targets n=27' >&2; exit 2; }
FIRSTPASS_PREFIX="${FIRSTPASS_PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"
SELECTED_ENV="${SELECTED_ENV:-${FIRSTPASS_PREFIX}.selected.env}"
BASE_PROMOTER="${BASE_PROMOTER:-$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stagem.sh}"
[[ -s "$SELECTED_ENV" ]] || { echo "missing selected env=$SELECTED_ENV" >&2; exit 2; }
[[ -f "$BASE_PROMOTER" ]] || { echo "missing Stage-L/M exact promoter=$BASE_PROMOTER" >&2; exit 2; }
command -v sha256sum >/dev/null || exit 2
# shellcheck disable=SC1090
source "$SELECTED_ENV"

HAS_N=0
[[ -n "${B300_GRAND_SELECTED_STAGEN_ENABLED+x}" ]] && HAS_N=1
if ((HAS_N)); then
  req=(
    B300_GRAND_SELECTED_STAGEN_ENABLED
    B300_GRAND_SELECTED_STAGEN_MIN_SPEEDUP
    B300_GRAND_SELECTED_STAGEN_ACCEPTED
    B300_GRAND_SELECTED_STAGEN_UPSTREAM_KIND
    B300_GRAND_SELECTED_STAGEN_PAIR_POLICY
    B300_GRAND_SELECTED_STAGEN_BLOCK_POLICY
    B300_GRAND_SELECTED_STAGEN_BASE_COUNT_POLICY
    B300_GRAND_SELECTED_STAGEN_STAGED_SPEEDUP
    B300_GRAND_SELECTED_STAGEN_SEARCH_PAIR_POLICIES
    B300_GRAND_SELECTED_STAGEN_SEARCH_BLOCK_POLICIES
  )
  for k in "${req[@]}"; do [[ -n "${!k+x}" ]] || { echo "Stage-N selected contract missing $k" >&2; exit 3; }; done
  [[ "$B300_GRAND_SELECTED_STAGEN_ENABLED" == 0 || "$B300_GRAND_SELECTED_STAGEN_ENABLED" == 1 ]] || { echo 'bad Stage-N enabled flag' >&2; exit 3; }
  [[ "$B300_GRAND_SELECTED_STAGEN_ACCEPTED" == 0 || "$B300_GRAND_SELECTED_STAGEN_ACCEPTED" == 1 ]] || { echo 'bad Stage-N accepted flag' >&2; exit 3; }
  (( B300_GRAND_SELECTED_STAGEN_ENABLED || ! B300_GRAND_SELECTED_STAGEN_ACCEPTED )) || { echo 'Stage-N accepted while disabled' >&2; exit 3; }
  for p in "$B300_GRAND_SELECTED_STAGEN_PAIR_POLICY" "$B300_GRAND_SELECTED_STAGEN_BLOCK_POLICY" "$B300_GRAND_SELECTED_STAGEN_BASE_COUNT_POLICY"; do
    case "$p" in default|cg|cs) ;; *) echo "bad Stage-N count-load policy=$p" >&2; exit 3;; esac
  done
  case "$B300_GRAND_SELECTED_STAGEN_UPSTREAM_KIND" in ''|stagel|stagem) ;; *) echo 'bad Stage-N upstream kind' >&2; exit 3;; esac

  python3 - "$B300_GRAND_SELECTED_STAGEN_MIN_SPEEDUP" "$B300_GRAND_SELECTED_STAGEN_STAGED_SPEEDUP" <<'PY'
import sys
m,s=map(float,sys.argv[1:])
if m < 1.0: raise SystemExit('Stage-N minimum speedup must be >=1')
if s <= 0.0: raise SystemExit('Stage-N staged speedup must be positive')
PY
  for raw in "$B300_GRAND_SELECTED_STAGEN_SEARCH_PAIR_POLICIES" "$B300_GRAND_SELECTED_STAGEN_SEARCH_BLOCK_POLICIES"; do
    seen=0
    for p in $raw; do
      case "$p" in default|cg|cs) ;; *) echo "bad Stage-N search policy=$p" >&2; exit 3;; esac
      [[ "$p" == "$B300_GRAND_SELECTED_STAGEN_BASE_COUNT_POLICY" ]] && seen=1
    done
    ((seen)) || { echo 'Stage-N search policy set omits inherited baseline' >&2; exit 3; }
  done

  if [[ "$B300_GRAND_SELECTED_STAGEN_ACCEPTED" == 1 ]]; then
    [[ -n "$B300_GRAND_SELECTED_STAGEN_UPSTREAM_KIND" ]] || { echo 'accepted Stage N has no upstream kind' >&2; exit 3; }
    [[ "${B300_GRAND_SELECTED_STAGEL_ACCEPTED:-0}" == 1 ]] || { echo 'Stage N accepted without Stage-L acceptance' >&2; exit 3; }
    [[ "$B300_GRAND_SELECTED_STAGEN_PAIR_POLICY" != "$B300_GRAND_SELECTED_STAGEN_BASE_COUNT_POLICY" || \
       "$B300_GRAND_SELECTED_STAGEN_BLOCK_POLICY" != "$B300_GRAND_SELECTED_STAGEN_BASE_COUNT_POLICY" ]] || {
      echo 'accepted Stage N retained inherited pair/block baseline' >&2; exit 3;
    }
    python3 - "$B300_GRAND_SELECTED_STAGEN_MIN_SPEEDUP" "$B300_GRAND_SELECTED_STAGEN_STAGED_SPEEDUP" <<'PY'
import sys
m,s=map(float,sys.argv[1:])
if s < m: raise SystemExit('accepted Stage-N speedup below threshold')
PY
    case "$B300_GRAND_SELECTED_STAGEN_UPSTREAM_KIND" in
      stagem)
        [[ "${B300_GRAND_SELECTED_STAGEM_ACCEPTED:-0}" == 1 ]] || { echo 'Stage N uses Stage M without Stage-M acceptance' >&2; exit 3; }
        ;;
      stagel)
        [[ "${B300_GRAND_SELECTED_STAGEM_ACCEPTED:-0}" != 1 ]] || { echo 'Stage N used Stage L despite accepted Stage M' >&2; exit 3; }
        ;;
    esac
  fi

  [[ -n "${B300_GRAND_SELECTED_GRAND_SUMMARY_ENV+x}" && -s "$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV" ]] || { echo 'Stage-N selection missing grand summary' >&2; exit 3; }
  # shellcheck disable=SC1090
  source "$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV"
  [[ "${B300_GRAND_STAGEN_INTEGRATED:-0}" == 1 ]] || { echo 'grand summary missing Stage-N integration' >&2; exit 3; }
  [[ "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || { echo 'Stage-N grand summary does not prove one complete-prime race' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEN_OK:-0}" == "$B300_GRAND_SELECTED_STAGEN_ACCEPTED" ]] || { echo 'Stage-N accepted flag differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEN_UPSTREAM_KIND:-}" == "$B300_GRAND_SELECTED_STAGEN_UPSTREAM_KIND" ]] || { echo 'Stage-N upstream differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEN_PAIR_POLICY:-default}" == "$B300_GRAND_SELECTED_STAGEN_PAIR_POLICY" ]] || { echo 'Stage-N pair policy differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEN_BLOCK_POLICY:-default}" == "$B300_GRAND_SELECTED_STAGEN_BLOCK_POLICY" ]] || { echo 'Stage-N block policy differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEN_BASE_COUNT_POLICY:-default}" == "$B300_GRAND_SELECTED_STAGEN_BASE_COUNT_POLICY" ]] || { echo 'Stage-N baseline policy differs from grand summary' >&2; exit 3; }
  if [[ "$B300_GRAND_SELECTED_STAGEN_ACCEPTED" == 1 ]]; then
    [[ -s "${B300_GRAND_STAGEN_MANIFEST:-}" ]] || { echo 'accepted Stage-N manifest missing from grand summary' >&2; exit 3; }
    sha256sum -c "$B300_GRAND_STAGEN_MANIFEST" >/dev/null || { echo 'Stage-N promotion manifest failed before exact continuation' >&2; exit 3; }
  fi
fi

echo "Stage-N exact provenance OK has_n=$HAS_N" >&2
exec env SELECTED_ENV="$SELECTED_ENV" "$BASE_PROMOTER" 27 "$@"
