#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
out="$(python3 "$ONEESAN_ROOT/scripts/bench/rankchunk32-block64-align-proof.py")"
printf '%s\n' "$out"
grep -Fq 'rankchunk32-block64-align-proof OK' <<<"$out"
grep -Fq 'height_align=64 block_base_loads_per_warp_max=1' <<<"$out"
grep -Fq 'padding_per_height_max=63' <<<"$out"
grep -Fq 'padding_per_owner_entries_max=1890' <<<"$out"
echo 'rankchunk32-block64-align-proof runner OK' >&2
