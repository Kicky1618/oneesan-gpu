#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
exec bash "$ONEESAN_ROOT/scripts/run/b300x8-grand-promote-exact-stagep.sh" "$@"
