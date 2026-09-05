#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_shard_owner_hi32_seed_w28_ngpu8_proof.cpp"
OUT="${OUT:-$ONEESAN_BUILD_DIR/b300_shard_owner_hi32_seed_w28_ngpu8_proof}"
"$CXX" -O3 -std=c++17 "$SRC" -o "$OUT"
text="$($OUT)"
printf '%s\n' "$text"
grep -Fq 'main total=385719506620 chunk=48214938328 hmax=89' <<<"$text"
grep -Fq 'block total=135015505407 chunk=16876938176 hmax=31' <<<"$text"
grep -Fq 'main_seed=(hi32*365)>>12 main_seed_shiftadd_exact=1 block_seed=hi32>>2 correction_compare64=1 correction_max=1 device_div64=0 device_mul64=0 base_table=0 exact=1' <<<"$text"
