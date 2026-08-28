#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

W="${W:-10}"; ARCH="${ARCH:-sm_80}"; PM_ACCUM="${PM_ACCUM:-0}"
DECODE_LOAD="${DECODE_LOAD:-ldg}"; RANKSTREAM_LUT_LOAD="${RANKSTREAM_LUT_LOAD:-ldg}"
RANKFORMULA_NOMETA_BLOCK="${RANKFORMULA_NOMETA_BLOCK:-4}"
RANKFORMULA_NOMETA_WARPSHARE="${RANKFORMULA_NOMETA_WARPSHARE:-0}"
LOW_LUT_K="${LOW_LUT_K:-$((W / 2))}"; HIGH_LUT_K="${HIGH_LUT_K:-$((W - LOW_LUT_K - 1))}"
if (( W > 12 || LOW_LUT_K <= 0 || HIGH_LUT_K <= 0 || LOW_LUT_K + HIGH_LUT_K + 1 != W )); then exit 2; fi
case "$RANKFORMULA_NOMETA_BLOCK" in 4|8|16) ;; *) echo "RANKFORMULA_NOMETA_BLOCK must be 4, 8, or 16" >&2; exit 2;; esac
[[ "$RANKFORMULA_NOMETA_WARPSHARE" == 0 || "$RANKFORMULA_NOMETA_WARPSHARE" == 1 ]] || { echo "RANKFORMULA_NOMETA_WARPSHARE must be 0 or 1" >&2; exit 2; }
case "$DECODE_LOAD" in global) P10DC_DECODE_LDG=0 ;; ldg) P10DC_DECODE_LDG=1 ;; *) exit 2;; esac
P10DC_RANKSTREAM_LUT_LDG=0; P10DC_RANKSTREAM_LUT_PAD256=0
case "$RANKSTREAM_LUT_LOAD" in constant) ;; ldg) P10DC_RANKSTREAM_LUT_LDG=1 ;; ldg256) P10DC_RANKSTREAM_LUT_LDG=1; P10DC_RANKSTREAM_LUT_PAD256=1 ;; *) exit 2;; esac

SRC="$ONEESAN_ROOT/src/cuda/gridfp/probes/ramstream32_bucket_orbit_closure_pattern10_depthcode_rankformula_nometa4_abstract_block_selftest.cu"
BIN="${BIN:-$ONEESAN_BUILD_DIR/pattern10_depthcode_rankformula_nometa4_abstract_block${RANKFORMULA_NOMETA_BLOCK}_warpshare${RANKFORMULA_NOMETA_WARPSHARE}_selftest_w${W}}"
mkdir -p "$(dirname "$BIN")"
TMPDIR="$ONEESAN_TMP_DIR" nvcc -O3 -std=c++17 -lineinfo -arch="$ARCH" \
  -DTARGET_W="$W" -DLOW_LUT_K="$LOW_LUT_K" -DHIGH_LUT_K="$HIGH_LUT_K" \
  -DP10DC_RANKFORMULA_NOMETA_BLOCK="$RANKFORMULA_NOMETA_BLOCK" \
  -DP10DC_RANKFORMULA_NOMETA_WARPSHARE="$RANKFORMULA_NOMETA_WARPSHARE" \
  -DGPU_DIRECT_PM_ACCUM="$PM_ACCUM" -DP10DC_DECODE_LDG="$P10DC_DECODE_LDG" \
  -DP10DC_RANKSTREAM_LUT_LDG="$P10DC_RANKSTREAM_LUT_LDG" -DP10DC_RANKSTREAM_LUT_PAD256="$P10DC_RANKSTREAM_LUT_PAD256" \
  -DP10DC_RANKCHUNK32_ONESHFL=1 -DP10DC_RANKCHUNK32_FUSED16=0 -DP10DC_RANKCHUNK32_BYTEPACK=0 \
  -DP10DC_RANKCHUNK32_ALIGN32=0 -DP10DC_RANKCHUNK32_BLOCK64=0 -DP10DC_RANKDELTA8_FUSED13=1 \
  -DP10DC_RANKFORMULA_SPARSE_BASE=1 -DP10DC_RANKFORMULA_RAWCODE=1 -DP10DC_RANKFORMULA_INLINE_CROSS=1 \
  -DP10DC_RANKFORMULA_BASE_DELTA=0 -DP10DC_RANKFORMULA_SLOTMETA=0 -DP10DC_RANKFORMULA_SLOTROW32=0 \
  "$SRC" -o "$BIN"
out="$($BIN)"
printf '%s\n' "$out"
grep -Eq "bucket-closure-pattern10-depthcode-rankformula-nometa4-abstract-selftest (OK W=$W|SKIP no CUDA device)" <<<"$out"
grep -Fq "rankformula-nometa4-abstract-block-selftest compiled_block=$RANKFORMULA_NOMETA_BLOCK warpshare=$RANKFORMULA_NOMETA_WARPSHARE max_locator_steps_bound=$((RANKFORMULA_NOMETA_BLOCK - 1)) wrapper_ok=1" <<<"$out"
if grep -Fq "OK W=$W" <<<"$out"; then
  grep -Fq 'forward_exact=1 reverse_exact=1' <<<"$out"
  grep -Fq 'abstract_lut_bytes=94206 abstract_states=7060 abstract_transitions=32743' <<<"$out"
  grep -Fq 'ballot_runtime_loads=0 source_local_lut=1' <<<"$out"
fi
echo "pattern10-depthcode-rankformula-nometa4-abstract-block-selftest OK W=$W block=$RANKFORMULA_NOMETA_BLOCK warpshare=$RANKFORMULA_NOMETA_WARPSHARE" >&2
