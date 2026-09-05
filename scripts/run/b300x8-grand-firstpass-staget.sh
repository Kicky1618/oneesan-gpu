#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
STAGES_WRAP="${STAGES_WRAP:-$ONEESAN_ROOT/scripts/run/b300x8-grand-firstpass-stages.sh}"
GENERATOR="${GENERATOR:-$ONEESAN_ROOT/scripts/build/gen-b300-grand-firstpass-staget-compat.py}"
PATCH_ONLY="${PATCH_ONLY:-0}"
[[ "$PATCH_ONLY" == 0 || "$PATCH_ONLY" == 1 ]] || exit 2
[[ -s "$STAGES_WRAP" && -s "$GENERATOR" ]] || exit 2
command -v sha256sum >/dev/null || exit 2
command -v python3 >/dev/null || exit 2
if [[ -n "${STAGET_BASE_FIRSTPASS:-}" ]]; then
  BASE_FIRSTPASS="$STAGET_BASE_FIRSTPASS"
else
  tmp_env="$(mktemp "${TMPDIR:-/tmp}/oneesan-staget-firstpass.XXXXXX")"
  trap 'rm -f "$tmp_env"' EXIT
  PATCH_ONLY=1 bash "$STAGES_WRAP" 27 >"$tmp_env"
  # shellcheck disable=SC1090
  source "$tmp_env"
  [[ "${B300_STAGES_FIRSTPASS_PATCHED:-0}" == 1 && -s "${B300_STAGES_FIRSTPASS_GENERATED:-}" ]] || exit 3
  BASE_FIRSTPASS="$B300_STAGES_FIRSTPASS_GENERATED"
fi
BASE_SHA="$(sha256sum "$BASE_FIRSTPASS"|awk '{print $1}')"
GEN_SHA="$(sha256sum "$GENERATOR"|awk '{print $1}')"
NATIVE=0
if grep -Fq 'RUN_STAGET=' "$BASE_FIRSTPASS" && grep -Fq 'B300_GRAND_SELECTED_STAGET_ACCEPTED' "$BASE_FIRSTPASS" && grep -Fq 'b300x8-joint-nextself-hybrid8-select-staget.sh' "$BASE_FIRSTPASS"; then
  NATIVE=1; OUT="$BASE_FIRSTPASS"; OUT_SHA="$BASE_SHA"
else
  KEY="${BASE_SHA:0:12}-${GEN_SHA:0:12}"
  OUT_DIR="${STAGET_FIRSTPASS_BUILD_DIR:-$ONEESAN_BUILD_DIR/generated-grand-firstpass}"
  OUT="${STAGET_FIRSTPASS_OUT:-$OUT_DIR/b300x8-grand-firstpass-staget-${KEY}.sh}"
  mkdir -p "$OUT_DIR"
  python3 "$GENERATOR" "$BASE_FIRSTPASS" "$OUT" >"${OUT}.transform.out"
  chmod +x "$OUT"
  bash -n "$OUT"
  OUT_SHA="$(sha256sum "$OUT"|awk '{print $1}')"
fi
export ONEESAN_ROOT ONEESAN_BUILD_DIR
export B300_STAGET_FIRSTPASS_BASE="$BASE_FIRSTPASS" B300_STAGET_FIRSTPASS_BASE_SHA256="$BASE_SHA"
export B300_STAGET_FIRSTPASS_GENERATOR="$GENERATOR" B300_STAGET_FIRSTPASS_GENERATOR_SHA256="$GEN_SHA"
export B300_STAGET_FIRSTPASS_GENERATED="$OUT" B300_STAGET_FIRSTPASS_GENERATED_SHA256="$OUT_SHA" B300_STAGET_FIRSTPASS_NATIVE="$NATIVE"
echo "Stage-T firstpass native=$NATIVE base_sha=${BASE_SHA:0:12} generator_sha=${GEN_SHA:0:12} selected_sha=${OUT_SHA:0:12} path=$OUT" >&2
if [[ "$PATCH_ONLY" == 1 ]]; then
  printf 'B300_STAGET_FIRSTPASS_PATCHED=1\n'
  printf 'B300_STAGET_FIRSTPASS_NATIVE=%q\n' "$NATIVE"
  printf 'B300_STAGET_FIRSTPASS_GENERATED=%q\n' "$OUT"
  printf 'B300_STAGET_FIRSTPASS_GENERATED_SHA256=%q\n' "$OUT_SHA"
  exit 0
fi
exec bash "$OUT" "$@"
