#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SELECTED_ENV="${SELECTED_ENV:-$ONEESAN_ROOT/work/b300_grand_refine_firstpass_n27.selected.env}"
[[ -s "$SELECTED_ENV" ]] || { echo "missing refinement selected env=$SELECTED_ENV" >&2; exit 2; }
exec env SELECTED_ENV="$SELECTED_ENV" "$ONEESAN_ROOT/scripts/run/b300x8-grand-stageh-continue.sh" "$@"
