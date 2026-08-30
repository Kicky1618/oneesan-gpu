#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
STAGEQ_WRAP="${STAGEQ_WRAP:-$ONEESAN_ROOT/scripts/run/b300x8-joint-nextself-hybrid8-select-stageq.sh}"
GENERATOR="${GENERATOR:-$ONEESAN_ROOT/scripts/build/gen-b300-grand-selector-stager.py}"
PATCH_ONLY="${PATCH_ONLY:-0}"
[[ "$PATCH_ONLY" == 0 || "$PATCH_ONLY" == 1 ]] || exit 2
[[ -s "$STAGEQ_WRAP" && -s "$GENERATOR" ]] || exit 2
command -v sha256sum >/dev/null || exit 2; command -v python3 >/dev/null || exit 2
if [[ -n "${STAGER_BASE_SELECTOR:-}" ]]; then BASE_SELECTOR="$STAGER_BASE_SELECTOR"; else
  tmp_env="$(mktemp "${TMPDIR:-/tmp}/oneesan-stager-selector.XXXXXX")"; trap 'rm -f "$tmp_env"' EXIT
  PATCH_ONLY=1 bash "$STAGEQ_WRAP" 27 >"$tmp_env"; source "$tmp_env"
  [[ "${B300_STAGEQ_SELECTOR_PATCHED:-0}" == 1 && -s "${B300_STAGEQ_SELECTOR_GENERATED:-}" ]] || exit 3
  BASE_SELECTOR="$B300_STAGEQ_SELECTOR_GENERATED"
fi
BASE_SHA="$(sha256sum "$BASE_SELECTOR"|awk '{print $1}')"; GEN_SHA="$(sha256sum "$GENERATOR"|awk '{print $1}')"; NATIVE=0
if grep -Fq 'RUN_STAGER=' "$BASE_SELECTOR" && grep -Fq 'B300_GRAND_STAGER_INTEGRATED=1' "$BASE_SELECTOR" && grep -Fq 'MODE=stager_ilp2_grand' "$BASE_SELECTOR"; then NATIVE=1; OUT="$BASE_SELECTOR"; OUT_SHA="$BASE_SHA"; else
  KEY="${BASE_SHA:0:12}-${GEN_SHA:0:12}"; OUT_DIR="${STAGER_SELECTOR_BUILD_DIR:-$ONEESAN_BUILD_DIR/generated-grand-selectors}"; OUT="${STAGER_SELECTOR_OUT:-$OUT_DIR/b300x8-joint-nextself-hybrid8-select-stager-${KEY}.sh}"; mkdir -p "$OUT_DIR"; python3 "$GENERATOR" "$BASE_SELECTOR" "$OUT" >"${OUT}.transform.out"; chmod +x "$OUT"; bash -n "$OUT"; OUT_SHA="$(sha256sum "$OUT"|awk '{print $1}')"; fi
export ONEESAN_ROOT ONEESAN_BUILD_DIR B300_STAGER_SELECTOR_BASE="$BASE_SELECTOR" B300_STAGER_SELECTOR_BASE_SHA256="$BASE_SHA" B300_STAGER_SELECTOR_GENERATOR="$GENERATOR" B300_STAGER_SELECTOR_GENERATOR_SHA256="$GEN_SHA" B300_STAGER_SELECTOR_GENERATED="$OUT" B300_STAGER_SELECTOR_GENERATED_SHA256="$OUT_SHA" B300_STAGER_SELECTOR_NATIVE="$NATIVE"
echo "Stage-R selector native=$NATIVE base_sha=${BASE_SHA:0:12} generator_sha=${GEN_SHA:0:12} selected_sha=${OUT_SHA:0:12} path=$OUT" >&2
if [[ "$PATCH_ONLY" == 1 ]]; then printf 'B300_STAGER_SELECTOR_PATCHED=1\n'; printf 'B300_STAGER_SELECTOR_NATIVE=%q\n' "$NATIVE"; printf 'B300_STAGER_SELECTOR_GENERATED=%q\n' "$OUT"; printf 'B300_STAGER_SELECTOR_GENERATED_SHA256=%q\n' "$OUT_SHA"; exit 0; fi
exec bash "$OUT" "$@"
