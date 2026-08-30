#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
RUN_LATEST_PREFLIGHT="${RUN_LATEST_PREFLIGHT:-1}"
[[ "$RUN_LATEST_PREFLIGHT" == 0 || "$RUN_LATEST_PREFLIGHT" == 1 ]] || { echo 'RUN_LATEST_PREFLIGHT must be 0/1' >&2; exit 2; }
N="${1:-27}"
if [[ "$RUN_LATEST_PREFLIGHT" == 1 ]]; then
  bash "$ONEESAN_ROOT/scripts/run/b300x8-grand-latest-preflight.sh" "$N"
fi
exec bash "$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-stageq.sh" "$@"
