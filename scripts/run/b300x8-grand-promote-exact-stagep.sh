#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || exit 2
FIRSTPASS_PREFIX="${FIRSTPASS_PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"; SELECTED_ENV="${SELECTED_ENV:-${FIRSTPASS_PREFIX}.selected.env}"; BASE_PROMOTER="${BASE_PROMOTER:-$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stageo.sh}"
[[ -s "$SELECTED_ENV" && -f "$BASE_PROMOTER" ]] || exit 2; command -v sha256sum >/dev/null || exit 2; source "$SELECTED_ENV"
HAS_P=0; [[ -n "${B300_GRAND_SELECTED_STAGEP_ENABLED+x}" ]] && HAS_P=1
if ((HAS_P)); then
  req=(B300_GRAND_SELECTED_STAGEP_ENABLED B300_GRAND_SELECTED_STAGEP_MIN_SPEEDUP B300_GRAND_SELECTED_STAGEP_ACCEPTED B300_GRAND_SELECTED_STAGEP_COUNT_UPSTREAM B300_GRAND_SELECTED_STAGEP_BASE_MATE_L2_BYTES B300_GRAND_SELECTED_STAGEP_MATE_L2_BYTES B300_GRAND_SELECTED_STAGEP_STAGED_SPEEDUP B300_GRAND_SELECTED_STAGEP_SEARCH_MATE_L2)
  for k in "${req[@]}"; do [[ -n "${!k+x}" ]] || { echo "Stage-P selected contract missing $k" >&2; exit 3; }; done
  [[ "$B300_GRAND_SELECTED_STAGEP_ENABLED" == 0 || "$B300_GRAND_SELECTED_STAGEP_ENABLED" == 1 ]] || exit 3; [[ "$B300_GRAND_SELECTED_STAGEP_ACCEPTED" == 0 || "$B300_GRAND_SELECTED_STAGEP_ACCEPTED" == 1 ]] || exit 3; ((B300_GRAND_SELECTED_STAGEP_ENABLED || !B300_GRAND_SELECTED_STAGEP_ACCEPTED)) || { echo 'Stage-P accepted while disabled' >&2; exit 3; }
  case "$B300_GRAND_SELECTED_STAGEP_COUNT_UPSTREAM" in ''|stagen|stageo) ;; *) echo 'bad Stage-P count upstream' >&2; exit 3;; esac
  for b in "$B300_GRAND_SELECTED_STAGEP_BASE_MATE_L2_BYTES" "$B300_GRAND_SELECTED_STAGEP_MATE_L2_BYTES"; do case "$b" in 0|64|128|256) ;; *) echo "bad Stage-P mate L2=$b" >&2; exit 3;; esac; done
  seen0=0; for b in $B300_GRAND_SELECTED_STAGEP_SEARCH_MATE_L2; do case "$b" in 0|64|128|256) ;; *) exit 3;; esac; [[ "$b" == 0 ]] && seen0=1; done; ((seen0)) || { echo 'Stage-P search list omits inherited 0B baseline' >&2; exit 3; }
  python3 - "$B300_GRAND_SELECTED_STAGEP_MIN_SPEEDUP" "$B300_GRAND_SELECTED_STAGEP_STAGED_SPEEDUP" <<'PY'
import sys
m,s=map(float,sys.argv[1:])
if m<1.0 or s<=0: raise SystemExit('bad Stage-P speed contract')
PY
  if [[ "$B300_GRAND_SELECTED_STAGEP_ACCEPTED" == 1 ]]; then
    [[ "${B300_GRAND_SELECTED_STAGEN_ACCEPTED:-0}" == 1 ]] || { echo 'Stage P accepted without Stage-N acceptance' >&2; exit 3; }
    [[ "${B300_GRAND_SELECTED_STAGEN_MATE_LOAD_POLICY:-default}" == cg || "${B300_GRAND_SELECTED_STAGEM_POLICY:-default}" == cg ]] || { echo 'Stage P accepted without inherited mate cg policy' >&2; exit 3; }
    [[ "$B300_GRAND_SELECTED_STAGEP_BASE_MATE_L2_BYTES" == 0 && "$B300_GRAND_SELECTED_STAGEP_MATE_L2_BYTES" != 0 ]] || { echo 'accepted Stage P retained inherited mate L2 baseline' >&2; exit 3; }
    if [[ "${B300_GRAND_SELECTED_STAGEO_ACCEPTED:-0}" == 1 ]]; then [[ "$B300_GRAND_SELECTED_STAGEP_COUNT_UPSTREAM" == stageo ]] || { echo 'Stage P ignored accepted Stage O upstream' >&2; exit 3; }; else [[ "$B300_GRAND_SELECTED_STAGEP_COUNT_UPSTREAM" == stagen ]] || { echo 'Stage P used Stage O without Stage-O acceptance' >&2; exit 3; }; fi
    python3 - "$B300_GRAND_SELECTED_STAGEP_MIN_SPEEDUP" "$B300_GRAND_SELECTED_STAGEP_STAGED_SPEEDUP" <<'PY'
import sys
m,s=map(float,sys.argv[1:])
if s<m: raise SystemExit('accepted Stage-P speedup below threshold')
PY
  fi
  [[ -n "${B300_GRAND_SELECTED_GRAND_SUMMARY_ENV+x}" && -s "$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV" ]] || { echo 'Stage-P selection missing grand summary' >&2; exit 3; }; source "$B300_GRAND_SELECTED_GRAND_SUMMARY_ENV"
  [[ "${B300_GRAND_STAGEP_INTEGRATED:-0}" == 1 && "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || { echo 'grand summary missing Stage-P single-race integration' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEP_OK:-0}" == "$B300_GRAND_SELECTED_STAGEP_ACCEPTED" ]] || { echo 'Stage-P accepted flag differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEP_COUNT_UPSTREAM:-}" == "$B300_GRAND_SELECTED_STAGEP_COUNT_UPSTREAM" ]] || { echo 'Stage-P upstream differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGEP_BASE_MATE_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGEP_BASE_MATE_L2_BYTES" && "${B300_GRAND_STAGEP_MATE_L2_BYTES:-0}" == "$B300_GRAND_SELECTED_STAGEP_MATE_L2_BYTES" ]] || { echo 'Stage-P mate L2 differs from grand summary' >&2; exit 3; }
  if [[ "$B300_GRAND_SELECTED_STAGEP_ACCEPTED" == 1 ]]; then [[ -s "${B300_GRAND_STAGEP_MANIFEST:-}" ]] || { echo 'accepted Stage-P manifest missing from grand summary' >&2; exit 3; }; sha256sum -c "$B300_GRAND_STAGEP_MANIFEST" >/dev/null || { echo 'Stage-P promotion manifest failed before exact continuation' >&2; exit 3; }; fi
fi
echo "Stage-P exact provenance OK has_p=$HAS_P" >&2
exec env SELECTED_ENV="$SELECTED_ENV" "$BASE_PROMOTER" 27 "$@"
