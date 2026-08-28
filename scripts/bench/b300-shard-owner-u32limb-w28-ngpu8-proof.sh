#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_shard_owner_u32limb_w28_ngpu8_proof.cpp"
OUT="${OUT:-$ONEESAN_BUILD_DIR/b300_shard_owner_u32limb_w28_ngpu8_proof}"
"$CXX" -O3 -std=c++17 "$SRC" -o "$OUT"
text="$($OUT)"
printf '%s\n' "$text"
grep -Fq 'main total=385719506620 chunk=48214938328 magic=195888106327 shift=73 high_shift=9 g_hi_max=89 magic_hi=45 magic_lo=2614578007 high_bound=4139' <<<"$text"
grep -Fq 'block total=135015505407 chunk=16876938176 magic=139905900989 shift=71 high_shift=7 g_hi_max=31 magic_hi=32 magic_lo=2466947517 high_bound=1055' <<<"$text"
grep -Fq 'device_mul64=0 device_div64=0 device_table_load=0' <<<"$text"
grep -Fq 'umulhi_u32_per_owner=3 mullo_u32_per_owner=3 masked_base_exact=1 high64_fits_u32=1 dense_samples_per_case=1000001 exact=1' <<<"$text"
