#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
out="$(python3 "$ONEESAN_ROOT/scripts/bench/rankchunk32-basepair64-proof.py")"
printf '%s\n' "$out"
grep -Fq 'rankchunk32-basepair64-proof OK' <<<"$out"
grep -Fq 'low_codes=1201917 total_l_digits=3720805' <<<"$out"
grep -Fq 'base_bits=22 base_limit=4194304' <<<"$out"
grep -Fq 'half_delta_max=224 delta_bits=8' <<<"$out"
grep -Fq 'packed_bits=30 pair_bytes=4 codes_per_pair=64' <<<"$out"
grep -Fq 'block_base_bytes_per_code=0.0625' <<<"$out"
grep -Fq 'unaligned_one_word_offsets=33/64' <<<"$out"
grep -Fq 'aligned32_words_per_warp=1' <<<"$out"
echo 'rankchunk32-basepair64-proof runner OK' >&2
