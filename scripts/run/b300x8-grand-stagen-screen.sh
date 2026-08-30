#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"; if (($#>0)); then shift; fi
[[ "$N" == 27 ]] || { echo 'grand Stage-N screen targets n=27' >&2; exit 2; }
FIRSTPASS_PREFIX="${FIRSTPASS_PREFIX:-$ONEESAN_ROOT/work/b300_grand_firstpass_n27}"
SELECTED_ENV="${SELECTED_ENV:-${FIRSTPASS_PREFIX}.selected.env}"
PROFILE_FILE="${PROFILE_FILE:-$ONEESAN_ROOT/work/b300_hbm_profile_refined21.env}"
STAGE_F_ENV="${STAGE_F_ENV:-${FIRSTPASS_PREFIX}.hybrid8-nextself_winner.env}"
STAGEL_WINNER_ENV="${STAGEL_WINNER_ENV:-${FIRSTPASS_PREFIX}.stagel-guard_winner.env}"
STAGEL_PREPARE_ENV="${STAGEL_PREPARE_ENV:-${FIRSTPASS_PREFIX}.stagel-guard.prepared.env}"
STAGEM_WINNER_ENV="${STAGEM_WINNER_ENV:-${FIRSTPASS_PREFIX}.stagem-mateload_winner.env}"
STAGEM_PREPARE_ENV="${STAGEM_PREPARE_ENV:-${FIRSTPASS_PREFIX}.stagem-mateload.prepared.env}"
PREFIX="${PREFIX:-${FIRSTPASS_PREFIX}.stagen-pairblock}"
WINNER_ENV="${WINNER_ENV:-${PREFIX}_winner.env}"; PREPARE_ENV="${PREPARE_ENV:-${PREFIX}.prepared.env}"
MIN_SPEEDUP="${MIN_SPEEDUP:-1.002}"; PAIR_POLICY_LIST="${PAIR_POLICY_LIST:-default cg cs}"; BLOCK_POLICY_LIST="${BLOCK_POLICY_LIST:-default cg cs}"
TARGET_MIB="${TARGET_MIB:-65536}"; MAX_WINDOW="${MAX_WINDOW:-14}"; NGPU="${NGPU:-8}"; ARCH="${ARCH:-native}"
[[ "$NGPU" =~ ^[1-8]$ ]] || exit 2
for f in "$SELECTED_ENV" "$PROFILE_FILE" "$STAGE_F_ENV" "$STAGEL_WINNER_ENV" "$STAGEL_PREPARE_ENV"; do [[ -s "$f" ]] || { echo "missing Stage-N screen input=$f" >&2; exit 2; }; done
# shellcheck disable=SC1090
source "$SELECTED_ENV"
[[ "${B300_GRAND_SELECTED_VALIDATED:-0}" == 1 && "${B300_GRAND_SELECTED_N:-0}" == 27 ]] || { echo 'invalid grand selected contract for Stage N' >&2; exit 3; }
MOD="${MOD:-$B300_GRAND_SELECTED_SMOKE_PRIME}"
[[ "$MOD" =~ ^[1-9][0-9]*$ ]] || exit 3

# Stage N is intentionally screened only after Stage L survived. If Stage M
# survived too, use it as the exact control; otherwise use Stage L directly.
L_ACCEPT="${B300_GRAND_SELECTED_STAGEL_ACCEPTED:-${B300_GRAND_STAGEL_OK:-0}}"
M_ACCEPT="${B300_GRAND_SELECTED_STAGEM_ACCEPTED:-${B300_GRAND_STAGEM_OK:-0}}"
[[ "$L_ACCEPT" == 0 || "$L_ACCEPT" == 1 ]] || exit 3; [[ "$M_ACCEPT" == 0 || "$M_ACCEPT" == 1 ]] || exit 3
if [[ "$L_ACCEPT" != 1 ]]; then
  echo 'Stage N screen not applicable: Stage L was not accepted' >&2
  exit 4
fi
UPSTREAM_KIND=stagel
if [[ "$M_ACCEPT" == 1 ]]; then
  [[ -s "$STAGEM_WINNER_ENV" && -s "$STAGEM_PREPARE_ENV" ]] || { echo 'selected Stage M but artifacts missing' >&2; exit 3; }
  UPSTREAM_KIND=stagem
fi

echo "=== grand Stage N staged screen upstream=$UPSTREAM_KIND pair=[$PAIR_POLICY_LIST] block=[$BLOCK_POLICY_LIST] min_speedup=$MIN_SPEEDUP ===" >&2
PROFILE_FILE="$PROFILE_FILE" STAGE_F_ENV="$STAGE_F_ENV" STAGEL_WINNER_ENV="$STAGEL_WINNER_ENV" STAGEL_PREPARE_ENV="$STAGEL_PREPARE_ENV" STAGEM_WINNER_ENV="$STAGEM_WINNER_ENV" STAGEM_PREPARE_ENV="$STAGEM_PREPARE_ENV" UPSTREAM_KIND="$UPSTREAM_KIND" ARCH="$ARCH" MOD="$MOD" TARGET_MIB="$TARGET_MIB" MAX_WINDOW="$MAX_WINDOW" NGPU="$NGPU" RUN_STAGED=1 PREPARE_ONLY=1 MIN_SPEEDUP="$MIN_SPEEDUP" PAIR_POLICY_LIST="$PAIR_POLICY_LIST" BLOCK_POLICY_LIST="$BLOCK_POLICY_LIST" STAGED_PREFIX="$PREFIX" WINNER_ENV="$WINNER_ENV" RACE_PREFIX="${PREFIX}.promote" PREPARE_ENV="$PREPARE_ENV" bash "$ONEESAN_ROOT/scripts/run/b300x8-nextgen-hybrid8-pair-block-load-stagen-staged-fullprime-race.sh" 27 "$@"
[[ -s "$PREPARE_ENV" ]] || { echo 'Stage-N prepared env missing after screen' >&2; exit 3; }
# shellcheck disable=SC1090
source "$PREPARE_ENV"
[[ "${B300_STAGEN_PREPARED:-0}" == 1 ]] || exit 3
cat "$PREPARE_ENV"
echo "GRAND STAGE N SCREEN PASSED upstream=$B300_STAGEN_PREPARED_UPSTREAM_KIND pair=$B300_STAGEN_PREPARED_PAIR_POLICY block=$B300_STAGEN_PREPARED_BLOCK_POLICY speedup=${B300_STAGEN_PREPARED_STAGED_SPEEDUP}x prepared=$PREPARE_ENV" >&2
