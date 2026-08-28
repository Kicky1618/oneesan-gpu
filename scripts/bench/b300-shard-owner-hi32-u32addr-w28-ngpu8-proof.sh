#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_shard_owner_hi32_u32addr_w28_ngpu8_proof.cpp"
OUT="${OUT:-$ONEESAN_BUILD_DIR/b300_shard_owner_hi32_u32addr_w28_ngpu8_proof}"
"$CXX" -O3 -std=c++17 "$SRC" -o "$OUT"
text="$($OUT)"
printf '%s\n' "$text"
grep -Fq 'main total=385719506620 chunk=48214938328 chunk_hi=11 chunk_lo=970298072 hmax=89' <<<"$text"
grep -Fq 'block total=135015505407 chunk=16876938176 chunk_hi=3 chunk_lo=3992036288 hmax=31' <<<"$text"
grep -Fq 'seed_hi32=1 base_u32_bits=1 correction_u32_compare=1 local_u32_subborrow=1 device_div64=0 device_mul64=0 device_setp64=0 device_sub64=0 exact=1' <<<"$text"
