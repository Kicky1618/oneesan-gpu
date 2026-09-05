#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BASE_FIRSTPASS="${BASE_FIRSTPASS:-$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass.sh}"
GENERATOR="${GENERATOR:-$ONEESAN_ROOT/scripts/build/gen-b300-grand-firstpass-stagem.py}"
PATCH_ONLY="${PATCH_ONLY:-0}"
[[ "$PATCH_ONLY" == 0 || "$PATCH_ONLY" == 1 ]] || { echo 'PATCH_ONLY must be 0/1' >&2; exit 2; }
[[ -s "$BASE_FIRSTPASS" && -s "$GENERATOR" ]] || { echo 'Stage-M firstpass source/generator missing' >&2; exit 2; }
command -v sha256sum >/dev/null || exit 2
command -v python3 >/dev/null || exit 2
BASE_SHA="$(sha256sum "$BASE_FIRSTPASS" | awk '{print $1}')"
GEN_SHA="$(sha256sum "$GENERATOR" | awk '{print $1}')"
KEY="${BASE_SHA:0:12}-${GEN_SHA:0:12}"
OUT_DIR="${STAGEM_FIRSTPASS_BUILD_DIR:-$ONEESAN_BUILD_DIR/generated-grand-firstpass}"
OUT="${STAGEM_FIRSTPASS_OUT:-$OUT_DIR/b300x8-grand-firstpass-stagem-${KEY}.sh}"
mkdir -p "$OUT_DIR"
python3 "$GENERATOR" "$BASE_FIRSTPASS" "$OUT" >"${OUT}.transform.out"
chmod +x "$OUT"
bash -n "$OUT"
OUT_SHA="$(sha256sum "$OUT" | awk '{print $1}')"
export ONEESAN_ROOT ONEESAN_BUILD_DIR
export B300_STAGEM_FIRSTPASS_BASE="$BASE_FIRSTPASS"
export B300_STAGEM_FIRSTPASS_BASE_SHA256="$BASE_SHA"
export B300_STAGEM_FIRSTPASS_GENERATOR="$GENERATOR"
export B300_STAGEM_FIRSTPASS_GENERATOR_SHA256="$GEN_SHA"
export B300_STAGEM_FIRSTPASS_GENERATED="$OUT"
export B300_STAGEM_FIRSTPASS_GENERATED_SHA256="$OUT_SHA"
echo "Stage-M firstpass base_sha=${BASE_SHA:0:12} generator_sha=${GEN_SHA:0:12} generated_sha=${OUT_SHA:0:12} path=$OUT" >&2
if [[ "$PATCH_ONLY" == 1 ]]; then
  printf 'B300_STAGEM_FIRSTPASS_PATCHED=1\n'
  printf 'B300_STAGEM_FIRSTPASS_BASE_SHA256=%q\n' "$BASE_SHA"
  printf 'B300_STAGEM_FIRSTPASS_GENERATOR_SHA256=%q\n' "$GEN_SHA"
  printf 'B300_STAGEM_FIRSTPASS_GENERATED=%q\n' "$OUT"
  printf 'B300_STAGEM_FIRSTPASS_GENERATED_SHA256=%q\n' "$OUT_SHA"
  exit 0
fi
exec bash "$OUT" "$@"
