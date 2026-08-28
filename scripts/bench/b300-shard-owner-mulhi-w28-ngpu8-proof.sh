#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

CXX="${CXX:-g++}"
command -v "$CXX" >/dev/null || { echo "$CXX not found" >&2; exit 2; }
SRC="$ONEESAN_ROOT/src/cpp/probes/b300_shard_owner_mulhi_w28_ngpu8_proof.cpp"
OUT="${OUT:-$ONEESAN_BUILD_DIR/b300_shard_owner_mulhi_w28_ngpu8_proof}"
"$CXX" -O3 -std=c++17 "$SRC" -o "$OUT"
text="$($OUT)"
printf '%s\n' "$text"
grep -Fq 'main total=385719506620 chunk=48214938328 shift=73 high_shift=9 magic=195888106327' <<<"$text"
grep -Fq 'block total=135015505407 chunk=16876938176 shift=71 high_shift=7 magic=139905900989' <<<"$text"
grep -Fq 'magic_unique=1 monotone_interval_proof=1 exact=1' <<<"$text"
