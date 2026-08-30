#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || exit 2
FIRSTPASS_PREFIX="${FIRSTPASS_PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"
SELECTED_ENV="${SELECTED_ENV:-${FIRSTPASS_PREFIX}.selected.env}"
BASE_PROMOTER="${BASE_PROMOTER:-$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stager.sh}"
[[ -s "$SELECTED_ENV" && -f "$BASE_PROMOTER" ]] || exit 2
command -v sha256sum >/dev/null || exit 2
# Pin control paths before sourcing any selected artifact.
PINNED_SELECTED_ENV="$SELECTED_ENV"; PINNED_BASE_PROMOTER="$BASE_PROMOTER"
# shellcheck disable=SC1090
source "$PINNED_SELECTED_ENV"
SELECTED_ENV="$PINNED_SELECTED_ENV"; BASE_PROMOTER="$PINNED_BASE_PROMOTER"
HAS_S=0; [[ -n "${B300_GRAND_SELECTED_STAGES_ENABLED+x}" ]] && HAS_S=1
if ((HAS_S)); then
  req=(B300_GRAND_SELECTED_STAGES_ENABLED B300_GRAND_SELECTED_STAGES_MIN_SPEEDUP B300_GRAND_SELECTED_STAGES_ACCEPTED B300_GRAND_SELECTED_STAGES_STAGER_UPSTREAM_KIND B300_GRAND_SELECTED_STAGES_LOW_PAIR_POLICY B300_GRAND_SELECTED_STAGES_LOW_BLOCK_POLICY B300_GRAND_SELECTED_STAGES_HIGH_PAIR_POLICY B300_GRAND_SELECTED_STAGES_HIGH_BLOCK_POLICY B300_GRAND_SELECTED_STAGES_HIGH_PAIR_L2_BYTES B300_GRAND_SELECTED_STAGES_HIGH_BLOCK_L2_BYTES B300_GRAND_SELECTED_STAGES_PAIR_L2_BYTES B300_GRAND_SELECTED_STAGES_BLOCK_L2_BYTES B300_GRAND_SELECTED_STAGES_STAGED_SPEEDUP B300_GRAND_SELECTED_STAGES_SEARCH_PAIR_L2 B300_GRAND_SELECTED_STAGES_SEARCH_BLOCK_L2)
  for k in "${req[@]}"; do [[ -n "${!k+x}" ]] || { echo "Stage-S selected contract missing $k" >&2; exit 3; }; done
  [[ "$B300_GRAND_SELECTED_STAGES_ENABLED" == 0 || "$B300_GRAND_SELECTED_STAGES_ENABLED" == 1 ]] || exit 3
  [[ "$B300_GRAND_SELECTED_STAGES_ACCEPTED" == 0 || "$B300_GRAND_SELECTED_STAGES_ACCEPTED" == 1 ]] || exit 3
  ((B300_GRAND_SELECTED_STAGES_ENABLED || !B300_GRAND_SELECTED_STAGES_ACCEPTED)) || { echo 'Stage-S accepted while disabled' >&2; exit 3; }
  case "$B300_GRAND_SELECTED_STAGES_STAGER_UPSTREAM_KIND" in ''|stagen|stageo|stagep|stageq) ;; *) echo 'bad Stage-S Stage-R upstream kind' >&2; exit 3;; esac
  for p in "$B300_GRAND_SELECTED_STAGES_LOW_PAIR_POLICY" "$B300_GRAND_SELECTED_STAGES_LOW_BLOCK_POLICY" "$B300_GRAND_SELECTED_STAGES_HIGH_PAIR_POLICY" "$B300_GRAND_SELECTED_STAGES_HIGH_BLOCK_POLICY"; do case "$p" in default|cg|cs) ;; *) echo "bad Stage-S policy=$p" >&2; exit 3;; esac; done
  for b in "$B300_GRAND_SELECTED_STAGES_HIGH_PAIR_L2_BYTES" "$B300_GRAND_SELECTED_STAGES_HIGH_BLOCK_L2_BYTES" "$B300_GRAND_SELECTED_STAGES_PAIR_L2_BYTES" "$B300_GRAND_SELECTED_STAGES_BLOCK_L2_BYTES"; do case "$b" in 0|64|128|256) ;; *) echo "bad Stage-S L2=$b" >&2; exit 3;; esac; done
  for list in B300_GRAND_SELECTED_STAGES_SEARCH_PAIR_L2 B300_GRAND_SELECTED_STAGES_SEARCH_BLOCK_L2; do
    n=0; has0=0
    for b in ${!list}; do case "$b" in 0|64|128|256) ;; *) echo "bad Stage-S search L2=$b" >&2; exit 3;; esac; [[ "$b" == 0 ]] && has0=1; n=$((n+1)); done
    ((n>0 && has0==1)) || { echo "$list must include zero baseline" >&2; exit 3; }
  done
  python3 - "$B300_GRAND_SELECTED_STAGES_MIN_SPEEDUP" "$B300_GRAND_SELECTED_STAGES_STAGED_SPEEDUP" <<'PY'
import sys
m,s=map(float,sys.argv[1:])
if m<1.0 or s<=0: raise SystemExit('bad Stage-S speed contract')
PY
  if [[ "$B300_GRAND_SELECTED_STAGES_ACCEPTED" == 1 ]]; then
    [[ "${B300_GRAND_SELECTED_STAGER_ACCEPTED:-0}" == 1 ]] || { echo 'Stage S accepted without Stage-R acceptance' >&2; exit 3; }
    [[ "$B300_GRAND_SELECTED_STAGES_STAGER_UPSTREAM_KIND" == "${B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND:-}" ]] || { echo 'Stage-S Stage-R upstream provenance drift' >&2; exit 3; }
    [[ "$B300_GRAND_SELECTED_STAGES_LOW_PAIR_POLICY" == "${B300_GRAND_SELECTED_STAGER_PAIR_POLICY:-}" && "$B300_GRAND_SELECTED_STAGES_LOW_BLOCK_POLICY" == "${B300_GRAND_SELECTED_STAGER_BLOCK_POLICY:-}" ]] || { echo 'Stage-S low policy drift from accepted Stage R' >&2; exit 3; }
    [[ "$B300_GRAND_SELECTED_STAGES_HIGH_PAIR_POLICY" == "${B300_GRAND_SELECTED_STAGER_HIGH_PAIR_POLICY:-}" && "$B300_GRAND_SELECTED_STAGES_HIGH_BLOCK_POLICY" == "${B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_POLICY:-}" && "$B300_GRAND_SELECTED_STAGES_HIGH_PAIR_L2_BYTES" == "${B300_GRAND_SELECTED_STAGER_HIGH_PAIR_L2_BYTES:-}" && "$B300_GRAND_SELECTED_STAGES_HIGH_BLOCK_L2_BYTES" == "${B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_L2_BYTES:-}" ]] || { echo 'Stage-S high-state tuple drift from accepted Stage R' >&2; exit 3; }
    [[ "$B300_GRAND_SELECTED_STAGES_PAIR_L2_BYTES" != 0 || "$B300_GRAND_SELECTED_STAGES_BLOCK_L2_BYTES" != 0 ]] || { echo 'accepted Stage S retained exact Stage-R zero-hint tuple' >&2; exit 3; }
    [[ "$B300_GRAND_SELECTED_STAGES_LOW_PAIR_POLICY" == cg || "$B300_GRAND_SELECTED_STAGES_PAIR_L2_BYTES" == 0 ]] || { echo 'Stage-S pair hint active on non-cg low policy' >&2; exit 3; }
    [[ "$B300_GRAND_SELECTED_STAGES_LOW_BLOCK_POLICY" == cg || "$B300_GRAND_SELECTED_STAGES_BLOCK_L2_BYTES" == 0 ]] || { echo 'Stage-S block hint active on non-cg low policy' >&2; exit 3; }
    python3 - "$B300_GRAND_SELECTED_STAGES_MIN_SPEEDUP" "$B300_GRAND_SELECTED_STAGES_STAGED_SPEEDUP" <<'PY'
import sys
m,s=map(float,sys.argv[1:])
if s<m: raise SystemExit('accepted Stage-S speedup below threshold')
PY
  fi
  [[ -n "${B300_GRAND_SELECTED_GRAND_SUMMARY_ENV+x}" && -s "$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV" ]] || { echo 'Stage-S selection missing grand summary' >&2; exit 3; }
  SUMMARY_ENV="$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV"
  # shellcheck disable=SC1090
  source "$SUMMARY_ENV"
  [[ "${B300_GRAND_STAGES_INTEGRATED:-0}" == 1 && "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || { echo 'grand summary missing Stage-S single-race integration' >&2; exit 3; }
  [[ "${B300_GRAND_STAGES_OK:-0}" == "$B300_GRAND_SELECTED_STAGES_ACCEPTED" ]] || { echo 'Stage-S accepted flag differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGES_STAGER_UPSTREAM_KIND:-}" == "$B300_GRAND_SELECTED_STAGES_STAGER_UPSTREAM_KIND" ]] || { echo 'Stage-S upstream differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGES_LOW_PAIR_POLICY:-default}" == "$B300_GRAND_SELECTED_STAGES_LOW_PAIR_POLICY" && "${B300_GRAND_STAGES_LOW_BLOCK_POLICY:-default}" == "$B300_GRAND_SELECTED_STAGES_LOW_BLOCK_POLICY" ]] || { echo 'Stage-S low policy differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGES_HIGH_PAIR_POLICY:-default}" == "$B300_GRAND_SELECTED_STAGES_HIGH_PAIR_POLICY" && "${B300_GRAND_STAGES_HIGH_BLOCK_POLICY:-default}" == "$B300_GRAND_SELECTED_STAGES_HIGH_BLOCK_POLICY" && "${B300_GRAND_STAGES_HIGH_PAIR_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGES_HIGH_PAIR_L2_BYTES" && "${B300_GRAND_STAGES_HIGH_BLOCK_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGES_HIGH_BLOCK_L2_BYTES" ]] || { echo 'Stage-S high-state provenance differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGES_PAIR_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGES_PAIR_L2_BYTES" && "${B300_GRAND_STAGES_BLOCK_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGES_BLOCK_L2_BYTES" ]] || { echo 'Stage-S selected low L2 tuple differs from grand summary' >&2; exit 3; }
  if [[ "$B300_GRAND_SELECTED_STAGES_ACCEPTED" == 1 ]]; then
    [[ -s "${B300_GRAND_STAGES_MANIFEST:-}" ]] || { echo 'accepted Stage-S manifest missing from grand summary' >&2; exit 3; }
    sha256sum -c "$B300_GRAND_STAGES_MANIFEST" >/dev/null || { echo 'Stage-S promotion manifest failed before exact continuation' >&2; exit 3; }
  fi
fi
echo "Stage-S exact provenance OK has_s=$HAS_S" >&2
exec env SELECTED_ENV="$PINNED_SELECTED_ENV" "$PINNED_BASE_PROMOTER" 27 "$@"
