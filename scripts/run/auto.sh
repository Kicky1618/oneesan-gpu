#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${1:-27}"
MOD="${2:-4294967291}"

PLAN_EXPORTS="$(python3 "$ONEESAN_ROOT/scripts/run/auto-plan.py" "$N" --format shell)"
eval "$PLAN_EXPORTS"

cat >&2 <<EOF
oneesan auto plan:
  n=$N
  runner=$ONEESAN_RUNNER
  storage=$ONEESAN_STORAGE
  gpus=$NGPU
  target_mib=$TARGET_MIB
  planner_target_mib=$GRIDFP_PLAN_TARGET_MIB
  reserve_mib=$GRIDFP_VRAM_RESERVE_MIB
  max_window=$MAX_WINDOW
EOF

if [[ "${ONEESAN_AUTO_DRY_RUN:-0}" == 1 ]]; then
  exit 0
fi

case "$ONEESAN_RUNNER" in
  optimized)
    export FAST_SHARD_ADDRESS8="${FAST_SHARD_ADDRESS8:-1}"
    exec "$ONEESAN_ROOT/scripts/run/b300x8.sh" "$N" "$MOD"
    ;;
  adaptive)
    export FAST_SHARD_ADDRESS8=0
    exec "$ONEESAN_ROOT/scripts/run/adaptive.sh" "$N" "$MOD"
    ;;
  *)
    echo "unknown auto runner: $ONEESAN_RUNNER" >&2
    exit 2
    ;;
esac
