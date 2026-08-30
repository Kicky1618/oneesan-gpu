#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

BASE_SELECTOR="${BASE_SELECTOR:-$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select.sh}"
GENERATOR="${GENERATOR:-$ONEESAN_ROOT/scripts/build/gen-b300-grand-selector-stagen.py}"
PATCH_ONLY="${PATCH_ONLY:-0}"
[[ "$PATCH_ONLY" == 0 || "$PATCH_ONLY" == 1 ]] || { echo 'PATCH_ONLY must be 0/1' >&2; exit 2; }
[[ -s "$BASE_SELECTOR" && -s "$GENERATOR" ]] || { echo 'Stage-N selector source/generator missing' >&2; exit 2; }
command -v sha256sum >/dev/null || exit 2
command -v python3 >/dev/null || exit 2

BASE_SHA="$(sha256sum "$BASE_SELECTOR" | awk '{print $1}')"
GEN_SHA="$(sha256sum "$GENERATOR" | awk '{print $1}')"
NATIVE=0
if grep -Fq 'RUN_STAGEN=' "$BASE_SELECTOR" && \
   grep -Fq 'B300_GRAND_STAGEN_INTEGRATED=1' "$BASE_SELECTOR" && \
   grep -Fq 'MODE=stagen_pairblock_grand' "$BASE_SELECTOR" && \
   grep -Fq 'b300x8-nextgen-hybrid8-pair-block-load-stagen-staged-fullprime-race.sh' "$BASE_SELECTOR"; then
  NATIVE=1
  OUT="$BASE_SELECTOR"
  OUT_SHA="$BASE_SHA"
else
  KEY="${BASE_SHA:0:12}-${GEN_SHA:0:12}"
  OUT_DIR="${STAGEN_SELECTOR_BUILD_DIR:-$ONEESAN_BUILD_DIR/generated-grand-selectors}"
  OUT="${STAGEN_SELECTOR_OUT:-$OUT_DIR/b300x8-joint-nextself-hybrid8-select-stagen-${KEY}.sh}"
  mkdir -p "$OUT_DIR"
  python3 "$GENERATOR" "$BASE_SELECTOR" "$OUT" >"${OUT}.transform.out"
  chmod +x "$OUT"
  bash -n "$OUT"
  OUT_SHA="$(sha256sum "$OUT" | awk '{print $1}')"
fi

export ONEESAN_ROOT ONEESAN_BUILD_DIR
export B300_STAGEN_SELECTOR_BASE="$BASE_SELECTOR"
export B300_STAGEN_SELECTOR_BASE_SHA256="$BASE_SHA"
export B300_STAGEN_SELECTOR_GENERATOR="$GENERATOR"
export B300_STAGEN_SELECTOR_GENERATOR_SHA256="$GEN_SHA"
export B300_STAGEN_SELECTOR_GENERATED="$OUT"
export B300_STAGEN_SELECTOR_GENERATED_SHA256="$OUT_SHA"
export B300_STAGEN_SELECTOR_NATIVE="$NATIVE"

echo "Stage-N selector native=$NATIVE base_sha=${BASE_SHA:0:12} generator_sha=${GEN_SHA:0:12} selected_sha=${OUT_SHA:0:12} path=$OUT" >&2
if [[ "$PATCH_ONLY" == 1 ]]; then
  printf 'B300_STAGEN_SELECTOR_PATCHED=1\n'
  printf 'B300_STAGEN_SELECTOR_NATIVE=%q\n' "$NATIVE"
  printf 'B300_STAGEN_SELECTOR_BASE=%q\n' "$BASE_SELECTOR"
  printf 'B300_STAGEN_SELECTOR_BASE_SHA256=%q\n' "$BASE_SHA"
  printf 'B300_STAGEN_SELECTOR_GENERATOR=%q\n' "$GENERATOR"
  printf 'B300_STAGEN_SELECTOR_GENERATOR_SHA256=%q\n' "$GEN_SHA"
  printf 'B300_STAGEN_SELECTOR_GENERATED=%q\n' "$OUT"
  printf 'B300_STAGEN_SELECTOR_GENERATED_SHA256=%q\n' "$OUT_SHA"
  exit 0
fi
exec bash "$OUT" "$@"
