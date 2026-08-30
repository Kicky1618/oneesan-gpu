#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || exit 2
FIRSTPASS_PREFIX="${FIRSTPASS_PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"; SELECTED_ENV="${SELECTED_ENV:-${FIRSTPASS_PREFIX}.selected.env}"; BASE_PROMOTER="${BASE_PROMOTER:-$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stageq.sh}"
[[ -s "$SELECTED_ENV" && -f "$BASE_PROMOTER" ]] || exit 2; command -v sha256sum >/dev/null || exit 2
source "$SELECTED_ENV"
HAS_R=0; [[ -n "${B300_GRAND_SELECTED_STAGER_ENABLED+x}" ]] && HAS_R=1
if ((HAS_R)); then
  req=(B300_GRAND_SELECTED_STAGER_ENABLED B300_GRAND_SELECTED_STAGER_MIN_SPEEDUP B300_GRAND_SELECTED_STAGER_ACCEPTED B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND B300_GRAND_SELECTED_STAGER_UPSTREAM_PAIR_POLICY B300_GRAND_SELECTED_STAGER_UPSTREAM_BLOCK_POLICY B300_GRAND_SELECTED_STAGER_HIGH_PAIR_POLICY B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_POLICY B300_GRAND_SELECTED_STAGER_HIGH_PAIR_L2_BYTES B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_L2_BYTES B300_GRAND_SELECTED_STAGER_PAIR_POLICY B300_GRAND_SELECTED_STAGER_BLOCK_POLICY B300_GRAND_SELECTED_STAGER_STAGED_SPEEDUP B300_GRAND_SELECTED_STAGER_SEARCH_PAIR_POLICIES B300_GRAND_SELECTED_STAGER_SEARCH_BLOCK_POLICIES)
  for k in "${req[@]}"; do [[ -n "${!k+x}" ]] || { echo "Stage-R selected contract missing $k" >&2; exit 3; }; done
  [[ "$B300_GRAND_SELECTED_STAGER_ENABLED" == 0 || "$B300_GRAND_SELECTED_STAGER_ENABLED" == 1 ]] || exit 3; [[ "$B300_GRAND_SELECTED_STAGER_ACCEPTED" == 0 || "$B300_GRAND_SELECTED_STAGER_ACCEPTED" == 1 ]] || exit 3
  ((B300_GRAND_SELECTED_STAGER_ENABLED || !B300_GRAND_SELECTED_STAGER_ACCEPTED)) || { echo 'Stage-R accepted while disabled' >&2; exit 3; }
  case "$B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND" in ''|stagen|stageo|stagep|stageq) ;; *) echo 'bad Stage-R upstream kind' >&2; exit 3;; esac
  for p in "$B300_GRAND_SELECTED_STAGER_UPSTREAM_PAIR_POLICY" "$B300_GRAND_SELECTED_STAGER_UPSTREAM_BLOCK_POLICY" "$B300_GRAND_SELECTED_STAGER_HIGH_PAIR_POLICY" "$B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_POLICY" "$B300_GRAND_SELECTED_STAGER_PAIR_POLICY" "$B300_GRAND_SELECTED_STAGER_BLOCK_POLICY"; do case "$p" in default|cg|cs) ;; *) echo "bad Stage-R policy=$p" >&2; exit 3;; esac; done
  for b in "$B300_GRAND_SELECTED_STAGER_HIGH_PAIR_L2_BYTES" "$B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_L2_BYTES"; do case "$b" in 0|64|128|256) ;; *) echo "bad Stage-R high L2=$b" >&2; exit 3;; esac; done
  for list in B300_GRAND_SELECTED_STAGER_SEARCH_PAIR_POLICIES B300_GRAND_SELECTED_STAGER_SEARCH_BLOCK_POLICIES; do n=0; for p in ${!list}; do case "$p" in default|cg|cs) ;; *) echo "bad Stage-R search policy=$p" >&2; exit 3;; esac; n=$((n+1)); done; ((n>0)) || exit 3; done
  python3 - "$B300_GRAND_SELECTED_STAGER_MIN_SPEEDUP" "$B300_GRAND_SELECTED_STAGER_STAGED_SPEEDUP" <<'PY'
import sys
m,s=map(float,sys.argv[1:])
if m<1.0 or s<=0: raise SystemExit('bad Stage-R speed contract')
PY
  if [[ "$B300_GRAND_SELECTED_STAGER_ACCEPTED" == 1 ]]; then
    [[ "${B300_GRAND_SELECTED_STAGEN_ACCEPTED:-0}" == 1 ]] || { echo 'Stage R accepted without Stage-N acceptance' >&2; exit 3; }
    if [[ "${B300_GRAND_SELECTED_STAGEQ_ACCEPTED:-0}" == 1 ]]; then want=stageq
    elif [[ "${B300_GRAND_SELECTED_STAGEP_ACCEPTED:-0}" == 1 ]]; then want=stagep
    elif [[ "${B300_GRAND_SELECTED_STAGEO_ACCEPTED:-0}" == 1 ]]; then want=stageo
    else want=stagen; fi
    [[ "$B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND" == "$want" ]] || { echo "Stage R ignored maximal accepted upstream expected=$want got=$B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND" >&2; exit 3; }
    [[ "$B300_GRAND_SELECTED_STAGER_PAIR_POLICY" != "$B300_GRAND_SELECTED_STAGER_UPSTREAM_PAIR_POLICY" || "$B300_GRAND_SELECTED_STAGER_BLOCK_POLICY" != "$B300_GRAND_SELECTED_STAGER_UPSTREAM_BLOCK_POLICY" ]] || { echo 'accepted Stage R retained exact upstream ILP2 tuple' >&2; exit 3; }
    # High-state policy is always the Stage-N policy; Q only changes its ILP8 CG L2 tuple.
    [[ "$B300_GRAND_SELECTED_STAGER_HIGH_PAIR_POLICY" == "${B300_GRAND_SELECTED_STAGEN_PAIR_POLICY:-$B300_GRAND_SELECTED_STAGER_HIGH_PAIR_POLICY}" && "$B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_POLICY" == "${B300_GRAND_SELECTED_STAGEN_BLOCK_POLICY:-$B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_POLICY}" ]] || { echo 'Stage-R high policy drift from Stage N' >&2; exit 3; }
    if [[ "$want" == stageq ]]; then
      [[ "$B300_GRAND_SELECTED_STAGER_HIGH_PAIR_L2_BYTES" == "${B300_GRAND_SELECTED_STAGEQ_PAIR_L2_BYTES:-}" && "$B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_L2_BYTES" == "${B300_GRAND_SELECTED_STAGEQ_BLOCK_L2_BYTES:-}" ]] || { echo 'Stage-R did not preserve accepted Stage-Q high L2 tuple' >&2; exit 3; }
    fi
    python3 - "$B300_GRAND_SELECTED_STAGER_MIN_SPEEDUP" "$B300_GRAND_SELECTED_STAGER_STAGED_SPEEDUP" <<'PY'
import sys
m,s=map(float,sys.argv[1:])
if s<m: raise SystemExit('accepted Stage-R speedup below threshold')
PY
  fi
  [[ -n "${B300_GRAND_SELECTED_GRAND_SUMMARY_ENV+x}" && -s "$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV" ]] || { echo 'Stage-R selection missing grand summary' >&2; exit 3; }
  source "$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV"
  [[ "${B300_GRAND_STAGER_INTEGRATED:-0}" == 1 && "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || { echo 'grand summary missing Stage-R single-race integration' >&2; exit 3; }
  [[ "${B300_GRAND_STAGER_OK:-0}" == "$B300_GRAND_SELECTED_STAGER_ACCEPTED" ]] || { echo 'Stage-R accepted flag differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGER_UPSTREAM_KIND:-}" == "$B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND" ]] || { echo 'Stage-R upstream differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGER_PAIR_POLICY:-default}" == "$B300_GRAND_SELECTED_STAGER_PAIR_POLICY" && "${B300_GRAND_STAGER_BLOCK_POLICY:-default}" == "$B300_GRAND_SELECTED_STAGER_BLOCK_POLICY" ]] || { echo 'Stage-R selected tuple differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGER_HIGH_PAIR_POLICY:-default}" == "$B300_GRAND_SELECTED_STAGER_HIGH_PAIR_POLICY" && "${B300_GRAND_STAGER_HIGH_BLOCK_POLICY:-default}" == "$B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_POLICY" && "${B300_GRAND_STAGER_HIGH_PAIR_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGER_HIGH_PAIR_L2_BYTES" && "${B300_GRAND_STAGER_HIGH_BLOCK_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_L2_BYTES" ]] || { echo 'Stage-R high-state provenance differs from grand summary' >&2; exit 3; }
  if [[ "$B300_GRAND_SELECTED_STAGER_ACCEPTED" == 1 ]]; then [[ -s "${B300_GRAND_STAGER_MANIFEST:-}" ]] || { echo 'accepted Stage-R manifest missing from grand summary' >&2; exit 3; }; sha256sum -c "$B300_GRAND_STAGER_MANIFEST" >/dev/null || { echo 'Stage-R promotion manifest failed before exact continuation' >&2; exit 3; }; fi
fi
echo "Stage-R exact provenance OK has_r=$HAS_R" >&2
exec env SELECTED_ENV="$SELECTED_ENV" "$BASE_PROMOTER" 27 "$@"
