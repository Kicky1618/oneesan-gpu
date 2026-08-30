#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

STAGEN_WRAP="${STAGEN_WRAP:-$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stagen.sh}"
GENERATOR="${GENERATOR:-$ONEESAN_ROOT/scripts/build/gen-b300-grand-selector-stageo.py}"
PATCH_ONLY="${PATCH_ONLY:-0}"
[[ "$PATCH_ONLY" == 0 || "$PATCH_ONLY" == 1 ]] || { echo 'PATCH_ONLY must be 0/1' >&2; exit 2; }
[[ -s "$STAGEN_WRAP" && -s "$GENERATOR" ]] || { echo 'Stage-O selector source/generator missing' >&2; exit 2; }
command -v sha256sum >/dev/null || exit 2; command -v python3 >/dev/null || exit 2

if [[ -n "${STAGEO_BASE_SELECTOR:-}" ]]; then
  BASE_SELECTOR="$STAGEO_BASE_SELECTOR"
else
  tmp_env="$(mktemp "${TMPDIR:-/tmp}/oneesan-stageo-selector.XXXXXX")"; trap 'rm -f "$tmp_env"' EXIT
  PATCH_ONLY=1 bash "$STAGEN_WRAP" 27 >"$tmp_env"
  # shellcheck disable=SC1090
  source "$tmp_env"
  [[ "${B300_STAGEN_SELECTOR_PATCHED:-0}" == 1 && -s "${B300_STAGEN_SELECTOR_GENERATED:-}" ]] || { echo 'Stage-N selector overlay did not produce a base' >&2; exit 3; }
  BASE_SELECTOR="$B300_STAGEN_SELECTOR_GENERATED"
fi
[[ -s "$BASE_SELECTOR" ]] || { echo "missing Stage-O base selector=$BASE_SELECTOR" >&2; exit 2; }
BASE_SHA="$(sha256sum "$BASE_SELECTOR" | awk '{print $1}')"; GEN_SHA="$(sha256sum "$GENERATOR" | awk '{print $1}')"
NATIVE=0
if grep -Fq 'RUN_STAGEO=' "$BASE_SELECTOR" && grep -Fq 'B300_GRAND_STAGEO_INTEGRATED=1' "$BASE_SELECTOR" && grep -Fq 'MODE=stageo_cgl2_grand' "$BASE_SELECTOR" && grep -Fq 'b300x8-nextgen-hybrid8-stageo-cg-l2-staged-fullprime-race.sh' "$BASE_SELECTOR"; then
  NATIVE=1; OUT="$BASE_SELECTOR"; OUT_SHA="$BASE_SHA"
else
  KEY="${BASE_SHA:0:12}-${GEN_SHA:0:12}"; OUT_DIR="${STAGEO_SELECTOR_BUILD_DIR:-$ONEESAN_BUILD_DIR/generated-grand-selectors}"
  OUT="${STAGEO_SELECTOR_OUT:-$OUT_DIR/b300x8-joint-nextself-hybrid8-select-stageo-${KEY}.sh}"
  mkdir -p "$OUT_DIR"; python3 "$GENERATOR" "$BASE_SELECTOR" "$OUT" >"${OUT}.transform.out"; chmod +x "$OUT"; bash -n "$OUT"; OUT_SHA="$(sha256sum "$OUT" | awk '{print $1}')"
fi
export ONEESAN_ROOT ONEESAN_BUILD_DIR
export B300_STAGEO_SELECTOR_BASE="$BASE_SELECTOR" B300_STAGEO_SELECTOR_BASE_SHA256="$BASE_SHA" B300_STAGEO_SELECTOR_GENERATOR="$GENERATOR" B300_STAGEO_SELECTOR_GENERATOR_SHA256="$GEN_SHA" B300_STAGEO_SELECTOR_GENERATED="$OUT" B300_STAGEO_SELECTOR_GENERATED_SHA256="$OUT_SHA" B300_STAGEO_SELECTOR_NATIVE="$NATIVE"
echo "Stage-O selector native=$NATIVE base_sha=${BASE_SHA:0:12} generator_sha=${GEN_SHA:0:12} selected_sha=${OUT_SHA:0:12} path=$OUT" >&2
if [[ "$PATCH_ONLY" == 1 ]]; then
  printf 'B300_STAGEO_SELECTOR_PATCHED=1\n'; printf 'B300_STAGEO_SELECTOR_NATIVE=%q\n' "$NATIVE"; printf 'B300_STAGEO_SELECTOR_BASE=%q\n' "$BASE_SELECTOR"; printf 'B300_STAGEO_SELECTOR_BASE_SHA256=%q\n' "$BASE_SHA"; printf 'B300_STAGEO_SELECTOR_GENERATOR=%q\n' "$GENERATOR"; printf 'B300_STAGEO_SELECTOR_GENERATOR_SHA256=%q\n' "$GEN_SHA"; printf 'B300_STAGEO_SELECTOR_GENERATED=%q\n' "$OUT"; printf 'B300_STAGEO_SELECTOR_GENERATED_SHA256=%q\n' "$OUT_SHA"; exit 0
fi
exec bash "$OUT" "$@"
