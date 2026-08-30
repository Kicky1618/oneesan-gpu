#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
STAGES_WRAP="${STAGES_WRAP:-$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stages.sh}"
GENERATOR="${GENERATOR:-$ONEESAN_ROOT/scripts/build/gen-b300-grand-selector-staget.py}"
PATCH_ONLY="${PATCH_ONLY:-0}"
[[ "$PATCH_ONLY" == 0 || "$PATCH_ONLY" == 1 ]] || exit 2
[[ -s "$STAGES_WRAP" && -s "$GENERATOR" ]] || exit 2
command -v sha256sum >/dev/null || exit 2
command -v python3 >/dev/null || exit 2
if [[ -n "${STAGET_BASE_SELECTOR:-}" ]]; then
  BASE_SELECTOR="$STAGET_BASE_SELECTOR"
else
  tmp_env="$(mktemp "${TMPDIR:-/tmp}/oneesan-staget-selector.XXXXXX")"
  trap 'rm -f "$tmp_env"' EXIT
  PATCH_ONLY=1 bash "$STAGES_WRAP" 27 >"$tmp_env"
  # shellcheck disable=SC1090
  source "$tmp_env"
  [[ "${B300_STAGES_SELECTOR_PATCHED:-0}" == 1 && -s "${B300_STAGES_SELECTOR_GENERATED:-}" ]] || exit 3
  BASE_SELECTOR="$B300_STAGES_SELECTOR_GENERATED"
fi
BASE_SHA="$(sha256sum "$BASE_SELECTOR"|awk '{print $1}')"
GEN_SHA="$(sha256sum "$GENERATOR"|awk '{print $1}')"
NATIVE=0
if grep -Fq 'RUN_STAGET=' "$BASE_SELECTOR" && grep -Fq 'B300_GRAND_STAGET_INTEGRATED=1' "$BASE_SELECTOR" && grep -Fq 'MODE=staget_ilp2_mate_grand' "$BASE_SELECTOR"; then
  NATIVE=1; OUT="$BASE_SELECTOR"; OUT_SHA="$BASE_SHA"
else
  KEY="${BASE_SHA:0:12}-${GEN_SHA:0:12}"
  OUT_DIR="${STAGET_SELECTOR_BUILD_DIR:-$ONEESAN_BUILD_DIR/generated-grand-selectors}"
  OUT="${STAGET_SELECTOR_OUT:-$OUT_DIR/b300x8-joint-nextself-hybrid8-select-staget-${KEY}.sh}"
  mkdir -p "$OUT_DIR"
  python3 "$GENERATOR" "$BASE_SELECTOR" "$OUT" >"${OUT}.transform.out"
  chmod +x "$OUT"
  bash -n "$OUT"
  OUT_SHA="$(sha256sum "$OUT"|awk '{print $1}')"
fi
export ONEESAN_ROOT ONEESAN_BUILD_DIR
export B300_STAGET_SELECTOR_BASE="$BASE_SELECTOR" B300_STAGET_SELECTOR_BASE_SHA256="$BASE_SHA"
export B300_STAGET_SELECTOR_GENERATOR="$GENERATOR" B300_STAGET_SELECTOR_GENERATOR_SHA256="$GEN_SHA"
export B300_STAGET_SELECTOR_GENERATED="$OUT" B300_STAGET_SELECTOR_GENERATED_SHA256="$OUT_SHA" B300_STAGET_SELECTOR_NATIVE="$NATIVE"
echo "Stage-T selector native=$NATIVE base_sha=${BASE_SHA:0:12} generator_sha=${GEN_SHA:0:12} selected_sha=${OUT_SHA:0:12} path=$OUT" >&2
if [[ "$PATCH_ONLY" == 1 ]]; then
  printf 'B300_STAGET_SELECTOR_PATCHED=1\n'
  printf 'B300_STAGET_SELECTOR_NATIVE=%q\n' "$NATIVE"
  printf 'B300_STAGET_SELECTOR_BASE=%q\n' "$BASE_SELECTOR"
  printf 'B300_STAGET_SELECTOR_BASE_SHA256=%q\n' "$BASE_SHA"
  printf 'B300_STAGET_SELECTOR_GENERATED=%q\n' "$OUT"
  printf 'B300_STAGET_SELECTOR_GENERATED_SHA256=%q\n' "$OUT_SHA"
  exit 0
fi
exec bash "$OUT" "$@"
