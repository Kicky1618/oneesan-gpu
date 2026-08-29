#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Safe entrypoint for experimental calibrated selection. Full CRT continuation
# requires invoking b300x8-exact-auto-calibrated-profiled.sh explicitly with
# SELECT_ONLY=0 after reviewing the same-prime complete-residue race.
export SELECT_ONLY=1
exec "$ONEESAN_ROOT/scripts/run/b300x8-exact-auto-calibrated-profiled.sh" "${@:-27}"
