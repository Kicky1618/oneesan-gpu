#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
require_uint N "$N" || exit 2
if (( N < 2 || N > 27 )); then echo "N must be in 2..27 for the B300 production solvers" >&2; exit 2; fi
W=$((N + 1))
ARCH="${ARCH:-native}"
NVCC="$(command -v nvcc)"
SRC="$(repo_path "${SRC:-src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu}")"
ROW6_CRT20_SRC="$(repo_path "src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu")"
PROVENANCE_AUX=()
if [[ "$SRC" == "$ROW6_CRT20_SRC" ]]; then
  python3 "$ONEESAN_ROOT/scripts/tools/verify_row6_crt20.py" >/dev/null
  PROVENANCE_AUX+=(
    --auxiliary-dependency row6-crt20-generator "$ONEESAN_ROOT/scripts/tools/gen_row6_crt20.py"
    --auxiliary-dependency row6-crt20-verifier "$ONEESAN_ROOT/scripts/tools/verify_row6_crt20.py"
    --auxiliary-dependency row6-path-bound-source "$ONEESAN_ROOT/scripts/solve/path_bound.py"
    --auxiliary-dependency row6-rational-certificate "$ONEESAN_ROOT/formal/certificates/row6_rational_dump.txt.xz"
  )
fi
OUT="$(build_path "${OUT:-oneesan_cuda_gridfp_b300_hbm32_n${N}}")"
LOW_LUT_K="${LOW_LUT_K:-}"
HIGH_LUT_K="${HIGH_LUT_K:-}"

if [[ -z "$LOW_LUT_K" ]]; then
  if (( N >= 27 )); then LOW_LUT_K=13; else LOW_LUT_K=0; fi
fi
if [[ -z "$HIGH_LUT_K" ]]; then
  if (( N >= 27 )); then HIGH_LUT_K=13; else HIGH_LUT_K=0; fi
fi

TMPDIR="$ONEESAN_TMP_DIR" "$NVCC" \
  -O3 -std=c++17 -lineinfo \
  -arch="$ARCH" \
  -DTARGET_W="$W" \
  -DLOW_LUT_K="$LOW_LUT_K" \
  -DHIGH_LUT_K="$HIGH_LUT_K" \
  "$SRC" -o "$OUT"

python3 "$ONEESAN_ROOT/scripts/solve/build_provenance.py" create \
  --root "$ONEESAN_ROOT" \
  --binary "$OUT" \
  --source "$SRC" \
  --compiler "$NVCC" \
  --compile-arg=-O3 \
  --compile-arg=-std=c++17 \
  --compile-arg=-lineinfo \
  --compile-arg=-arch="$ARCH" \
  --compile-arg=-DTARGET_W="$W" \
  --compile-arg=-DLOW_LUT_K="$LOW_LUT_K" \
  --compile-arg=-DHIGH_LUT_K="$HIGH_LUT_K" \
  --compile-arg="$SRC" \
  --compile-arg=-o \
  --compile-arg="$OUT" \
  "${PROVENANCE_AUX[@]}" \
  >/dev/null

echo "built $OUT"
echo "  source=$SRC"
echo "  n=$N width=$W arch=$ARCH low_lut_k=$LOW_LUT_K high_lut_k=$HIGH_LUT_K"
echo "  provenance=${OUT}.provenance.json"
