#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

bash "$ONEESAN_ROOT/scripts/bench/directgather64-quad-proof.sh"
exec bash "$ONEESAN_ROOT/scripts/bench/b300-cpasync-quad-vs-pair-ab.sh"
