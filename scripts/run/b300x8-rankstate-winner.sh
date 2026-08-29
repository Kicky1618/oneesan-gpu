#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

WINNER_ENV="${WINNER_ENV:-$ONEESAN_ROOT/work/b300_rankstate_ilp4_hotd32_race_row1_winner.env}"
[[ -f "$WINNER_ENV" ]] || { echo "missing WINNER_ENV=$WINNER_ENV; run scripts/bench/b300x8-rankstate-ilp4-hotd32-race.sh first" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
source "$WINNER_ENV"
set +a

for k in RANK_STATE_ILP2 RANK_STATE_ILP4 HOT_DELTA_TABLE BLOCK_CLOSURE_QUAD GRIDFP_THREADS; do
  [[ -n "${!k+x}" ]] || { echo "winner env missing $k" >&2; exit 3; }
done
for k in RANK_STATE_ILP2 RANK_STATE_ILP4 HOT_DELTA_TABLE BLOCK_CLOSURE_QUAD; do
  [[ "${!k}" == 0 || "${!k}" == 1 ]] || { echo "winner env invalid $k=${!k}" >&2; exit 3; }
done
(( RANK_STATE_ILP2 + RANK_STATE_ILP4 == 1 )) || { echo 'winner must select exactly one of ILP2/ILP4' >&2; exit 3; }
[[ "$BLOCK_CLOSURE_QUAD" == 0 || "$RANK_STATE_ILP4" == 1 ]] || { echo 'closure quad winner requires ILP4' >&2; exit 3; }
[[ "$GRIDFP_THREADS" =~ ^[0-9]+$ ]] && (( GRIDFP_THREADS>=32 && GRIDFP_THREADS<=1024 && GRIDFP_THREADS%32==0 )) || {
  echo "bad GRIDFP_THREADS=$GRIDFP_THREADS" >&2; exit 3;
}

export ROWS="${ROWS:-28}"
export REBUILD="${REBUILD:-1}"
export CONCURRENT_GROUP_IO="${CONCURRENT_GROUP_IO:-1}"
export RANK_DELTA_CACHE=1 RANK_STATE_PACKED=1
export MAIN_PULL=1 BLOCK_PULL=1 MAIN_MATE_CACHE=1 BLOCK_MATE_CACHE=1 FAST_SHARD_ADDRESS8=1

echo "rankstate_winner env=$WINNER_ENV ilp2=$RANK_STATE_ILP2 ilp4=$RANK_STATE_ILP4 hotd32=$HOT_DELTA_TABLE closureq=$BLOCK_CLOSURE_QUAD threads=$GRIDFP_THREADS rows=$ROWS rebuild=$REBUILD" >&2
exec "$ONEESAN_ROOT/scripts/run/b300x8.sh" "${1:-27}" "${2:-4294967291}"
