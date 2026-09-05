#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_shard_owner_u32shift_w28_ngpu8_proof.cpp"
OUT="${OUT:-$ONEESAN_BUILD_DIR/b300_shard_owner_u32shift_w28_ngpu8_proof}"
"$CXX" -O3 -std=c++17 "$SRC" -o "$OUT"
text="$($OUT)"
printf '%s\n' "$text"
grep -Fq 'main total=385719506620 chunk=48214938328 magic=195888106327 shift=73 high_shift=9 magic_hi=45 shiftadd_hi=1' <<<"$text"
grep -Fq 'block total=135015505407 chunk=16876938176 magic=139905900989 shift=71 high_shift=7 magic_hi=32 shiftadd_hi=1' <<<"$text"
grep -Fq 'device_mul64=0 device_div64=0 device_table_load=0 umulhi_u32_per_owner=2 mullo_u32_per_owner=1' <<<"$text"
grep -Fq 'magic_hi_mul_replaced_by_shiftadd=1 masked_base_exact=1 dense_samples_per_case=2000001 exact=1' <<<"$text"
