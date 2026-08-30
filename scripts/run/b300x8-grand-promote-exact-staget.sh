#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || exit 2
FIRSTPASS_PREFIX="${FIRSTPASS_PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"
SELECTED_ENV="${SELECTED_ENV:-${FIRSTPASS_PREFIX}.selected.env}"
BASE_PROMOTER="${BASE_PROMOTER:-$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stages.sh}"
[[ -s "$SELECTED_ENV" && -f "$BASE_PROMOTER" ]] || exit 2
command -v sha256sum >/dev/null || exit 2
PINNED_SELECTED_ENV="$SELECTED_ENV"; PINNED_BASE_PROMOTER="$BASE_PROMOTER"
readonly PINNED_SELECTED_ENV PINNED_BASE_PROMOTER
# shellcheck disable=SC1090
source "$PINNED_SELECTED_ENV"
SELECTED_ENV="$PINNED_SELECTED_ENV"; BASE_PROMOTER="$PINNED_BASE_PROMOTER"
HAS_T=0; [[ -n "${B300_GRAND_SELECTED_STAGET_ENABLED+x}" ]] && HAS_T=1
if ((HAS_T)); then
  req=(B300_GRAND_SELECTED_STAGET_ENABLED B300_GRAND_SELECTED_STAGET_MIN_SPEEDUP B300_GRAND_SELECTED_STAGET_ACCEPTED B300_GRAND_SELECTED_STAGET_UPSTREAM_KIND B300_GRAND_SELECTED_STAGET_STAGER_UPSTREAM_KIND B300_GRAND_SELECTED_STAGET_LOW_PAIR_POLICY B300_GRAND_SELECTED_STAGET_LOW_BLOCK_POLICY B300_GRAND_SELECTED_STAGET_LOW_PAIR_L2_BYTES B300_GRAND_SELECTED_STAGET_LOW_BLOCK_L2_BYTES B300_GRAND_SELECTED_STAGET_HIGH_PAIR_POLICY B300_GRAND_SELECTED_STAGET_HIGH_BLOCK_POLICY B300_GRAND_SELECTED_STAGET_HIGH_PAIR_L2_BYTES B300_GRAND_SELECTED_STAGET_HIGH_BLOCK_L2_BYTES B300_GRAND_SELECTED_STAGET_HIGH_MATE_POLICY B300_GRAND_SELECTED_STAGET_HIGH_MATE_L2_BYTES B300_GRAND_SELECTED_STAGET_POLICY B300_GRAND_SELECTED_STAGET_STAGED_SPEEDUP B300_GRAND_SELECTED_STAGET_SEARCH_POLICIES)
  for k in "${req[@]}"; do [[ -n "${!k+x}" ]] || { echo "Stage-T selected contract missing $k" >&2; exit 3; }; done
  T_ENABLED="$B300_GRAND_SELECTED_STAGET_ENABLED"; T_MIN="$B300_GRAND_SELECTED_STAGET_MIN_SPEEDUP"; T_ACCEPTED="$B300_GRAND_SELECTED_STAGET_ACCEPTED"; T_UP="$B300_GRAND_SELECTED_STAGET_UPSTREAM_KIND"; T_RUP="$B300_GRAND_SELECTED_STAGET_STAGER_UPSTREAM_KIND"; T_LP="$B300_GRAND_SELECTED_STAGET_LOW_PAIR_POLICY"; T_LB="$B300_GRAND_SELECTED_STAGET_LOW_BLOCK_POLICY"; T_LPL2="$B300_GRAND_SELECTED_STAGET_LOW_PAIR_L2_BYTES"; T_LBL2="$B300_GRAND_SELECTED_STAGET_LOW_BLOCK_L2_BYTES"; T_HP="$B300_GRAND_SELECTED_STAGET_HIGH_PAIR_POLICY"; T_HB="$B300_GRAND_SELECTED_STAGET_HIGH_BLOCK_POLICY"; T_HPL2="$B300_GRAND_SELECTED_STAGET_HIGH_PAIR_L2_BYTES"; T_HBL2="$B300_GRAND_SELECTED_STAGET_HIGH_BLOCK_L2_BYTES"; T_HM="$B300_GRAND_SELECTED_STAGET_HIGH_MATE_POLICY"; T_HML2="$B300_GRAND_SELECTED_STAGET_HIGH_MATE_L2_BYTES"; T_POLICY="$B300_GRAND_SELECTED_STAGET_POLICY"; T_SPEED="$B300_GRAND_SELECTED_STAGET_STAGED_SPEEDUP"; T_SEARCH="$B300_GRAND_SELECTED_STAGET_SEARCH_POLICIES"; T_SUMMARY="${B300_GRAND_SELECTED_GRAND_SUMMARY_ENV:-}"
  readonly T_ENABLED T_MIN T_ACCEPTED T_UP T_RUP T_LP T_LB T_LPL2 T_LBL2 T_HP T_HB T_HPL2 T_HBL2 T_HM T_HML2 T_POLICY T_SPEED T_SEARCH T_SUMMARY
  [[ "$T_ENABLED" == 0 || "$T_ENABLED" == 1 ]] || exit 3
  [[ "$T_ACCEPTED" == 0 || "$T_ACCEPTED" == 1 ]] || exit 3
  ((T_ENABLED || !T_ACCEPTED)) || { echo 'Stage-T accepted while disabled' >&2; exit 3; }
  case "$T_UP" in ''|stager|stages) ;; *) echo 'bad Stage-T immediate upstream' >&2; exit 3;; esac
  case "$T_RUP" in ''|stagen|stageo|stagep|stageq) ;; *) echo 'bad Stage-T Stage-R upstream' >&2; exit 3;; esac
  for p in "$T_LP" "$T_LB" "$T_HP" "$T_HB" "$T_HM"; do case "$p" in default|cg|cs) ;; *) echo "bad Stage-T inherited policy=$p" >&2; exit 3;; esac; done
  for b in "$T_LPL2" "$T_LBL2" "$T_HPL2" "$T_HBL2" "$T_HML2"; do case "$b" in 0|64|128|256) ;; *) echo "bad Stage-T L2=$b" >&2; exit 3;; esac; done
  seen_default=0; for p in $T_SEARCH; do case "$p" in default|cg|cs) ;; *) echo "bad Stage-T search policy=$p" >&2; exit 3;; esac; [[ "$p" == default ]] && seen_default=1; done; ((seen_default)) || { echo 'Stage-T search list omits exact default baseline' >&2; exit 3; }
  python3 - "$T_MIN" "$T_SPEED" <<'PY'
import sys
m,s=map(float,sys.argv[1:])
if m<1.0 or s<=0: raise SystemExit('bad Stage-T speed contract')
PY
  if [[ "$T_ACCEPTED" == 1 ]]; then
    [[ "${B300_GRAND_SELECTED_STAGER_ACCEPTED:-0}" == 1 ]] || { echo 'Stage T accepted without Stage-R acceptance' >&2; exit 3; }
    if [[ "${B300_GRAND_SELECTED_STAGES_ACCEPTED:-0}" == 1 ]]; then WANT_UP=stages; else WANT_UP=stager; fi
    [[ "$T_UP" == "$WANT_UP" ]] || { echo "Stage T ignored maximal immediate upstream expected=$WANT_UP got=$T_UP" >&2; exit 3; }
    case "$T_POLICY" in cg|cs) ;; *) echo 'accepted Stage T retained exact default mate policy' >&2; exit 3;; esac
    if [[ "$T_UP" == stages ]]; then
      [[ "$T_LP" == "${B300_GRAND_SELECTED_STAGES_LOW_PAIR_POLICY:-}" && "$T_LB" == "${B300_GRAND_SELECTED_STAGES_LOW_BLOCK_POLICY:-}" && "$T_LPL2" == "${B300_GRAND_SELECTED_STAGES_PAIR_L2_BYTES:-}" && "$T_LBL2" == "${B300_GRAND_SELECTED_STAGES_BLOCK_L2_BYTES:-}" ]] || { echo 'Stage-T low Count tuple drift from accepted Stage S' >&2; exit 3; }
    else
      [[ "$T_LP" == "${B300_GRAND_SELECTED_STAGER_PAIR_POLICY:-}" && "$T_LB" == "${B300_GRAND_SELECTED_STAGER_BLOCK_POLICY:-}" && "$T_LPL2" == 0 && "$T_LBL2" == 0 ]] || { echo 'Stage-T low Count tuple drift from Stage R' >&2; exit 3; }
    fi
    [[ "$T_RUP" == "${B300_GRAND_SELECTED_STAGER_UPSTREAM_KIND:-}" && "$T_HP" == "${B300_GRAND_SELECTED_STAGER_HIGH_PAIR_POLICY:-}" && "$T_HB" == "${B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_POLICY:-}" && "$T_HPL2" == "${B300_GRAND_SELECTED_STAGER_HIGH_PAIR_L2_BYTES:-}" && "$T_HBL2" == "${B300_GRAND_SELECTED_STAGER_HIGH_BLOCK_L2_BYTES:-}" ]] || { echo 'Stage-T high Count/Stage-R provenance drift' >&2; exit 3; }
    EXPECT_HM="${B300_GRAND_SELECTED_STAGEN_MATE_LOAD_POLICY:-${B300_GRAND_SELECTED_STAGEM_POLICY:-default}}"; EXPECT_HML2=0
    if [[ "${B300_GRAND_SELECTED_STAGEP_ACCEPTED:-0}" == 1 ]]; then EXPECT_HML2="${B300_GRAND_SELECTED_STAGEP_MATE_L2_BYTES:-0}"; fi
    [[ "$T_HM" == "$EXPECT_HM" && "$T_HML2" == "$EXPECT_HML2" ]] || { echo "Stage-T high mate provenance drift expected=$EXPECT_HM/$EXPECT_HML2 got=$T_HM/$T_HML2" >&2; exit 3; }
    [[ "$T_HM" == cg || "$T_HML2" == 0 ]] || { echo 'Stage-T high mate L2 active on non-cg policy' >&2; exit 3; }
    python3 - "$T_MIN" "$T_SPEED" <<'PY'
import sys
m,s=map(float,sys.argv[1:])
if s<m: raise SystemExit('accepted Stage-T speedup below threshold')
PY
  fi
  [[ -n "$T_SUMMARY" && -s "$T_SUMMARY" ]] || { echo 'Stage-T selection missing grand summary' >&2; exit 3; }
  # shellcheck disable=SC1090
  source "$T_SUMMARY"
  SELECTED_ENV="$PINNED_SELECTED_ENV"; BASE_PROMOTER="$PINNED_BASE_PROMOTER"
  [[ "${B300_GRAND_STAGET_INTEGRATED:-0}" == 1 && "${B300_GRAND_COMPLETE_PRIME_RACES:-0}" == 1 ]] || { echo 'grand summary missing Stage-T single-race integration' >&2; exit 3; }
  [[ "${B300_GRAND_STAGET_OK:-0}" == "$T_ACCEPTED" && "${B300_GRAND_STAGET_UPSTREAM_KIND:-}" == "$T_UP" && "${B300_GRAND_STAGET_STAGER_UPSTREAM_KIND:-}" == "$T_RUP" ]] || { echo 'Stage-T acceptance/upstream differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGET_LOW_PAIR_POLICY:-default}" == "$T_LP" && "${B300_GRAND_STAGET_LOW_BLOCK_POLICY:-default}" == "$T_LB" && "${B300_GRAND_STAGET_LOW_PAIR_L2_BYTES:-0}" == "$T_LPL2" && "${B300_GRAND_STAGET_LOW_BLOCK_L2_BYTES:-0}" == "$T_LBL2" ]] || { echo 'Stage-T low tuple differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGET_HIGH_PAIR_POLICY:-default}" == "$T_HP" && "${B300_GRAND_STAGET_HIGH_BLOCK_POLICY:-default}" == "$T_HB" && "${B300_GRAND_STAGET_HIGH_PAIR_L2_BYTES:-0}" == "$T_HPL2" && "${B300_GRAND_STAGET_HIGH_BLOCK_L2_BYTES:-0}" == "$T_HBL2" && "${B300_GRAND_STAGET_HIGH_MATE_POLICY:-default}" == "$T_HM" && "${B300_GRAND_STAGET_HIGH_MATE_L2_BYTES:-0}" == "$T_HML2" ]] || { echo 'Stage-T high-state tuple differs from grand summary' >&2; exit 3; }
  [[ "${B300_GRAND_STAGET_POLICY:-default}" == "$T_POLICY" ]] || { echo 'Stage-T selected policy differs from grand summary' >&2; exit 3; }
  if [[ "$T_ACCEPTED" == 1 ]]; then
    [[ -s "${B300_GRAND_STAGET_MANIFEST:-}" ]] || { echo 'accepted Stage-T manifest missing from grand summary' >&2; exit 3; }
    sha256sum -c "$B300_GRAND_STAGET_MANIFEST" >/dev/null || { echo 'Stage-T promotion manifest failed before exact continuation' >&2; exit 3; }
  fi
fi
echo "Stage-T exact provenance OK has_t=$HAS_T" >&2
exec env -u BASE_PROMOTER SELECTED_ENV="$PINNED_SELECTED_ENV" "$PINNED_BASE_PROMOTER" 27 "$@"
