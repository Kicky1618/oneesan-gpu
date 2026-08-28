#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
out="$(python3 "$ONEESAN_ROOT/scripts/bench/rankchunk32-basepair64-warp-proof.py")"
printf '%s\n' "$out"
grep -Fq 'rankchunk32-basepair64-warp-proof OK' <<<"$out"
grep -Fq 'fullwarp_one_pair_offsets=33/64' <<<"$out"
grep -Fq 'fullwarp_two_pair_offsets=31/64' <<<"$out"
grep -Fq 'pair_loads_per_warp_max=2' <<<"$out"
grep -Fq 'partial_source_active=1 pair_selection_exact=1' <<<"$out"
echo 'rankchunk32-basepair64-warp-proof runner OK' >&2
