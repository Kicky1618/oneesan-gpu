#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
RUN_LATEST_PREFLIGHT="${RUN_LATEST_PREFLIGHT:-1}"
[[ "$RUN_LATEST_PREFLIGHT" == 0 || "$RUN_LATEST_PREFLIGHT" == 1 ]] || { echo 'RUN_LATEST_PREFLIGHT must be 0/1' >&2; exit 2; }
N="${1:-27}"
for rel in \
  scripts/run/b300x8-grand-firstpass-stageq.sh \
  scripts/run/b300x8-joint-nextself-hybrid8-select-stageq.sh \
  scripts/run/b300x8-nextgen-hybrid8-stageq-ilp8-count-cg-l2-staged-fullprime-race.sh \
  scripts/bench/b300-grand-stageq-contract-preflight.sh \
  scripts/bench/b300-grand-stageq-firstpass-preflight.sh; do
  [[ -s "$ONEESAN_ROOT/$rel" ]] || { echo "missing latest Stage-Q artifact=$rel" >&2; exit 3; }
done
if [[ "$RUN_LATEST_PREFLIGHT" == 1 ]]; then
  bash "$ONEESAN_ROOT/scripts/run/b300x8-grand-latest-preflight.sh" "$N"
fi
exec bash "$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-stageq.sh" "$@"
