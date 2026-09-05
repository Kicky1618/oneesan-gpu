#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
if ! command -v nvcc >/dev/null; then
  echo "nvcc is required" >&2
  exit 2
fi

LOG="${LOG:-$ONEESAN_ROOT/work/gridfp_build_nvcc_prepend_smoke.log}"
OUT="${OUT:-$ONEESAN_BUILD_DIR/gridfp_build_nvcc_prepend_smoke_should_not_exist}"
mkdir -p "$(dirname "$LOG")" "$(dirname "$OUT")"
rm -f "$OUT"

# PACKED=1 without LAST_R=1 is intentionally invalid. The static_assert in
# gridfp_reduced_production_codec_device.cuh must fire only if
# NVCC_PREPEND_FLAGS actually reaches nvcc. If the build unexpectedly succeeds,
# the injection path is broken and exact A/B scripts that rely on it are invalid.
set +e
NVCC_PREPEND_FLAGS='-DRP_FAST_MATERIALIZE_PRIMITIVE_PACKED=1' \
  MODE=forward ARCH="$ARCH" PTXAS_VERBOSE=0 OUT="$OUT" \
  bash "$ONEESAN_ROOT/scripts/build/gridfp-reduced-component-probe.sh" \
  >"$LOG" 2>&1
status=$?
set -e

if (( status == 0 )); then
  echo "NVCC_PREPEND_FLAGS smoke unexpectedly compiled successfully" >&2
  cat "$LOG" >&2
  exit 3
fi
if ! grep -Fq 'packed primitive materialize requires forced final R' "$LOG"; then
  echo "NVCC_PREPEND_FLAGS smoke failed for an unexpected reason" >&2
  cat "$LOG" >&2
  exit 4
fi
if [[ -e "$OUT" ]]; then
  echo "unexpected output binary exists: $OUT" >&2
  exit 5
fi

printf '%s\n' \
  'gridfp-build-nvcc-prepend-smoke OK' \
  'injected_macro=RP_FAST_MATERIALIZE_PRIMITIVE_PACKED=1' \
  'expected_static_assert=packed_requires_forced_final_R' \
  'compiler_injection_exact=1'
echo "gridfp-build-nvcc-prepend-smoke OK log=$LOG" >&2
