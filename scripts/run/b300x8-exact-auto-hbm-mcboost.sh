#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Low-memory-controller entry point. Select the warp pair memory schedule on the
# exact, cheap n=21 instance, cache the winner, then pass only that configuration
# into the existing n=27 staged selector. Forced/orbit families remain unchanged.
AUTO_CPASYNC_PAIR_MODE="${AUTO_CPASYNC_PAIR_MODE:-1}"
CPASYNC_PAIR_RETUNE="${CPASYNC_PAIR_RETUNE:-0}"
CPASYNC_PAIR_REPEATS="${CPASYNC_PAIR_REPEATS:-1}"
CPASYNC_PAIR_PROFILE="${CPASYNC_PAIR_PROFILE:-$ONEESAN_ROOT/work/b300_overlap_local_pipe2_n21_winner.env}"
CPASYNC_PAIR_PROFILE_REV_EXPECT="pipe2-wait1-ignore-src-prefetchB-v3"
COL_ILP="${COL_ILP:-2}"
PAIR_MLP="${PAIR_MLP:-1}"

for x in AUTO_CPASYNC_PAIR_MODE CPASYNC_PAIR_RETUNE; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
[[ "$CPASYNC_PAIR_REPEATS" =~ ^[1-9][0-9]*$ ]] || { echo 'CPASYNC_PAIR_REPEATS must be positive integer' >&2; exit 2; }
case "$COL_ILP" in 1|2|4) ;; *) echo 'COL_ILP must be 1,2,4' >&2; exit 2;; esac
[[ "$PAIR_MLP" == 0 || "$PAIR_MLP" == 1 ]] || { echo 'PAIR_MLP must be 0 or 1' >&2; exit 2; }

MANUAL=0
for x in CPASYNC_PAIR CPASYNC_LOCAL_PAIR CPASYNC_OVERLAP_LOCAL_PAIR CPASYNC_OVERLAP_LOCAL_PIPE2; do
  [[ -v "$x" ]] && MANUAL=1
done

profile_current(){
  [[ -s "$CPASYNC_PAIR_PROFILE" ]] &&
    grep -Fxq "CPASYNC_PAIR_PROFILE_REV=$CPASYNC_PAIR_PROFILE_REV_EXPECT" "$CPASYNC_PAIR_PROFILE"
}

if [[ "$AUTO_CPASYNC_PAIR_MODE" == 1 && "$MANUAL" == 0 && "$PAIR_MLP" == 1 && "$COL_ILP" == 2 ]]; then
  if [[ "$CPASYNC_PAIR_RETUNE" == 1 ]] || ! profile_current; then
    echo "MCBOOST cpasync-preselect exact_n=21 repeats=$CPASYNC_PAIR_REPEATS profile=$CPASYNC_PAIR_PROFILE rev=$CPASYNC_PAIR_PROFILE_REV_EXPECT" >&2
    ARCH="${ARCH:-native}" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-ignore-src-microprobe.sh"
    WINNER_ENV="$CPASYNC_PAIR_PROFILE" REPEATS="$CPASYNC_PAIR_REPEATS" \
      SORTED=0 DIRECTGATHER_SPARSE64=0 PRECTX_FORWARD=0 PRECTX_REVERSE=0 \
      bash "$ONEESAN_ROOT/scripts/bench/b300-overlap-local-pipe2-ab.sh"
    printf 'CPASYNC_PAIR_PROFILE_REV=%s\n' "$CPASYNC_PAIR_PROFILE_REV_EXPECT" >>"$CPASYNC_PAIR_PROFILE"
  else
    echo "MCBOOST cpasync-preselect reuse profile=$CPASYNC_PAIR_PROFILE rev=$CPASYNC_PAIR_PROFILE_REV_EXPECT" >&2
  fi
  [[ -s "$CPASYNC_PAIR_PROFILE" ]] || { echo "missing/empty cpasync profile: $CPASYNC_PAIR_PROFILE" >&2; exit 3; }
  # shellcheck disable=SC1090
  source "$CPASYNC_PAIR_PROFILE"
else
  CPASYNC_PAIR="${CPASYNC_PAIR:-0}"
  CPASYNC_LOCAL_PAIR="${CPASYNC_LOCAL_PAIR:-0}"
  CPASYNC_OVERLAP_LOCAL_PAIR="${CPASYNC_OVERLAP_LOCAL_PAIR:-0}"
  CPASYNC_OVERLAP_LOCAL_PIPE2="${CPASYNC_OVERLAP_LOCAL_PIPE2:-0}"
  CPASYNC_PAIR_MODE="${CPASYNC_PAIR_MODE:-manual_or_register}"
fi

for x in CPASYNC_PAIR CPASYNC_LOCAL_PAIR CPASYNC_OVERLAP_LOCAL_PAIR CPASYNC_OVERLAP_LOCAL_PIPE2; do
  v="${!x}"; [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
[[ "$CPASYNC_LOCAL_PAIR" == 0 || "$CPASYNC_PAIR" == 1 ]] || { echo 'local pair requires cpasync pair' >&2; exit 2; }
[[ "$CPASYNC_OVERLAP_LOCAL_PAIR" == 0 || "$CPASYNC_PAIR" == 1 ]] || { echo 'overlap pair requires cpasync pair' >&2; exit 2; }
[[ "$CPASYNC_OVERLAP_LOCAL_PIPE2" == 0 || "$CPASYNC_OVERLAP_LOCAL_PAIR" == 1 ]] || { echo 'pipe2 requires overlap pair' >&2; exit 2; }

if [[ "$MANUAL" == 1 && "$CPASYNC_OVERLAP_LOCAL_PIPE2" == 1 ]]; then
  ARCH="${ARCH:-native}" bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-ignore-src-microprobe.sh"
fi

export COL_ILP PAIR_MLP
export CPASYNC_PAIR CPASYNC_LOCAL_PAIR CPASYNC_OVERLAP_LOCAL_PAIR CPASYNC_OVERLAP_LOCAL_PIPE2 CPASYNC_PAIR_MODE

echo "MCBOOST selected mode=$CPASYNC_PAIR_MODE cpasync=$CPASYNC_PAIR local=$CPASYNC_LOCAL_PAIR overlap=$CPASYNC_OVERLAP_LOCAL_PAIR pipe2=$CPASYNC_OVERLAP_LOCAL_PIPE2" >&2
exec "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-hbm.sh" "$@"
