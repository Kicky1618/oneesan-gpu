#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

# Compatibility entry point.  The canonical benchmark now compares the
# production candidates directly:
#   depth4 LUT resolved
#   depthcode payload thread
#   depthcode payload resolved
#   depthcode payload warp
# It also requires all residues to match the known/reference residue.
exec bash "$ONEESAN_ROOT/scripts/bench/b300-depth4-depthcode-graph-ab.sh" "$@"
