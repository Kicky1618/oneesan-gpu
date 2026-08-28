#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_80}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-hi32-seed-w28-ngpu8-proof.sh"

SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GENERATOR="$ONEESAN_ROOT/scripts/build/gen-b300-shard-address-hi32-seed.py"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_shard_address_hi32_seed_production_ptx}"
GENSRC="$OUTDIR/generated.cu"
PROBE="$OUTDIR/probe.cu"
mkdir -p "$OUTDIR"
python3 "$GENERATOR" "$SRC" "$GENSRC"

python3 - "$GENSRC" "$PROBE" <<'PY'
import json,pathlib,sys
src=pathlib.Path(sys.argv[1]).resolve(); out=pathlib.Path(sys.argv[2]); inc=json.dumps(src.as_posix())
out.write_text(f'''#pragma push_macro("main")
#undef main
#define main gridfp_b300_generated_main_unused
#include {inc}
#pragma pop_macro("main")
extern "C" __global__ void b300_hi32_generated_main_probe(const Code* g, Count* out){{int i=int(blockIdx.x*blockDim.x+threadIdx.x);out[i]=global_load_main(g[i]);}}
extern "C" __global__ void b300_hi32_generated_block_probe(const Code* g, Count* out){{int i=int(blockIdx.x*blockDim.x+threadIdx.x);out[i]=global_load_block(g[i]);}}
''')
PY

extract(){
  local ptx="$1" kernel="$2" out="$3"
  awk -v k="$kernel" '$0 ~ "\\.entry[[:space:]]+" k {inside=1} inside{print} inside&&/^}/{exit}' "$ptx" > "$out"
  [[ -s "$out" ]] || { echo "missing PTX body for $kernel" >&2; exit 3; }
}
metric(){ local re="$1" file="$2"; grep -Ec "$re" "$file" || true; }

for mode in 1 5; do
  ptx="$OUTDIR/mode_${mode}.ptx"
  "$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" \
    -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 \
    -DB300_FAST_SHARD_ADDRESS8=0 -DB300_SHARD_ADDRESS_MODE="$mode" \
    "$PROBE" -o "$ptx"
  extract "$ptx" b300_hi32_generated_main_probe "$OUTDIR/mode_${mode}_main.body.ptx"
  extract "$ptx" b300_hi32_generated_block_probe "$OUTDIR/mode_${mode}_block.body.ptx"
  for kind in main block; do
    body="$OUTDIR/mode_${mode}_${kind}.body.ptx"
    div64="$(metric '\b(div|rem)\.u64\b' "$body")"
    mul64="$(metric '\bmul\.(hi|lo|wide)\.u64\b' "$body")"
    setp64="$(metric '\bsetp\..*\.u64\b' "$body")"
    mul32="$(metric '\bmul\.(hi|lo|wide)\.u32\b|\bmad\.lo\.u32\b' "$body")"
    shl32="$(metric '\bshl\.b32\b' "$body")"
    bra="$(metric '\bbra\b' "$body")"
    selp="$(metric '\bselp\.' "$body")"
    if (( div64 != 0 || mul64 != 0 )); then
      echo "mode=$mode kind=$kind contains forbidden u64 divide/multiply" >&2
      exit 4
    fi
    printf 'mode%s_%s_divrem_u64=%s\nmode%s_%s_mul_u64=%s\nmode%s_%s_setp_u64=%s\nmode%s_%s_mul_u32=%s\nmode%s_%s_shl_b32=%s\nmode%s_%s_bra=%s\nmode%s_%s_selp=%s\n' \
      "$mode" "$kind" "$div64" "$mode" "$kind" "$mul64" "$mode" "$kind" "$setp64" \
      "$mode" "$kind" "$mul32" "$mode" "$kind" "$shl32" "$mode" "$kind" "$bra" "$mode" "$kind" "$selp"
  done
done

for kind in main block; do
  compare="$OUTDIR/mode_1_${kind}.body.ptx"
  seed="$OUTDIR/mode_5_${kind}.body.ptx"
  c="$(metric '\bsetp\..*\.u64\b' "$compare")"
  s="$(metric '\bsetp\..*\.u64\b' "$seed")"
  if (( c < 3 )); then echo "$kind compare path missing expected three u64 compares" >&2; exit 5; fi
  if (( s < 1 || s >= c )); then echo "$kind hi32 seed did not reduce u64 compare count compare=$c seed=$s" >&2; exit 6; fi
done

grep -Fq 'B300_SHARD_ADDRESS_MODE must be 0..5' "$GENSRC"
grep -Fq 'shard_hi32_seed_main' "$GENSRC"
grep -Fq 'shard_hi32_seed_block' "$GENSRC"
grep -Fq 'hi32-seed shard address is specialized for TARGET_W=28' "$GENSRC"
grep -Fq 'B300_SHARD_ADDRESS_MODE>=2 requires exactly 8 GPUs' "$GENSRC"
grep -Fq 'B300_SHARD_ADDRESS_MODE>=2 W28x8 constants mismatch' "$GENSRC"

echo "b300-shard-address-hi32-seed-production-ptx-proof OK arch=$ARCH modes=1,5 main_seed=(hi32*365)>>12 block_seed=hi32>>2 correction_max=1 device_div64=0 device_mul64=0 exact_proof=1"
