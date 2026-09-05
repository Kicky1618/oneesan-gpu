#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
WIDTHS="${WIDTHS:-8 9 10 11 12}"
PM_ACCUM="${PM_ACCUM:-0}"
[[ "$PM_ACCUM" == 0 || "$PM_ACCUM" == 1 ]] || { echo "PM_ACCUM must be 0 or 1" >&2; exit 2; }
command -v nvcc >/dev/null || { echo "nvcc required" >&2; exit 2; }

SRC="$(repo_path src/cuda/gridfp/probes/ramstream32_bucket_compact_prectx_selftest.cu)"
for W in $WIDTHS; do
  [[ "$W" =~ ^[0-9]+$ ]] || { echo "bad width: $W" >&2; exit 2; }
  (( W >= 5 && W <= 12 )) || { echo "width must be 5..12, got $W" >&2; exit 2; }
  LOW=$((W / 2))
  HIGH=$((W - LOW - 1))
  OUT="$(build_path "compact_prectx_selftest_w${W}")"
  echo "build compact-prectx selftest W=$W low=$LOW high=$HIGH" >&2
  TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
    -DTARGET_W="$W" -DLOW_LUT_K="$LOW" -DHIGH_LUT_K="$HIGH" \
    -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" -DBKCZ_TERNARY_KEY4=1 \
    "$SRC" -o "$OUT"
  "$OUT"
done

echo "compact-prectx selftests passed widths=[$WIDTHS] pm_accum=$PM_ACCUM" >&2
