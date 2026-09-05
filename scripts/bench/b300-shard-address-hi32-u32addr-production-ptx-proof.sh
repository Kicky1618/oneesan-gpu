#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}"; ARCH="${ARCH:-sm_80}"; command -v "$NVCC" >/dev/null || exit 2
bash "$ONEESAN_ROOT/scripts/bench/b300-shard-owner-hi32-u32addr-w28-ngpu8-proof.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-shard-address-hi32-u32addr.py"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_hi32_u32addr_production_ptx}"; mkdir -p "$OUTDIR"
GENSRC="$OUTDIR/generated.cu"; PROBE="$OUTDIR/probe.cu"; python3 "$GEN" "$SRC" "$GENSRC"
python3 - "$GENSRC" "$PROBE" <<'PY'
import json,pathlib,sys
s=pathlib.Path(sys.argv[1]).resolve();o=pathlib.Path(sys.argv[2]);inc=json.dumps(s.as_posix())
o.write_text(f'''#pragma push_macro("main")
#undef main
#define main b300_generated_main_unused
#include {inc}
#pragma pop_macro("main")
extern "C" __global__ void b300_u32addr_main_probe(const Code*g,Count*out){{int i=int(blockIdx.x*blockDim.x+threadIdx.x);out[i]=global_load_main(g[i]);}}
extern "C" __global__ void b300_u32addr_block_probe(const Code*g,Count*out){{int i=int(blockIdx.x*blockDim.x+threadIdx.x);out[i]=global_load_block(g[i]);}}
''')
PY
extract(){ local p="$1" k="$2" o="$3"; awk -v k="$k" '$0~"\\.entry[[:space:]]+"k{f=1}f{print}f&&/^}/{exit}' "$p">"$o"; [[ -s "$o" ]]||exit 3; }
metric(){ grep -Ec "$1" "$2" || true; }
for mode in 1 5 6; do
 ptx="$OUTDIR/mode_${mode}.ptx"; "$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=0 -DB300_SHARD_ADDRESS_MODE="$mode" "$PROBE" -o "$ptx"
 extract "$ptx" b300_u32addr_main_probe "$OUTDIR/m${mode}_main.ptx"; extract "$ptx" b300_u32addr_block_probe "$OUTDIR/m${mode}_block.ptx"
 for kind in main block; do
  b="$OUTDIR/m${mode}_${kind}.ptx"; div="$(metric '\b(div|rem)\.u64\b' "$b")"; mul="$(metric '\bmul\.(hi|lo|wide)\.u64\b' "$b")"; setp="$(metric '\bsetp\..*\.u64\b' "$b")"; sub="$(metric '\bsub\.u64\b' "$b")"; setp32="$(metric '\bsetp\..*\.u32\b' "$b")"
  ((div==0&&mul==0)) || { echo "mode=$mode $kind has u64 div/mul" >&2; exit 4; }
  printf 'mode%s_%s_divrem_u64=%s\nmode%s_%s_mul_u64=%s\nmode%s_%s_setp_u64=%s\nmode%s_%s_sub_u64=%s\nmode%s_%s_setp_u32=%s\n' "$mode" "$kind" "$div" "$mode" "$kind" "$mul" "$mode" "$kind" "$setp" "$mode" "$kind" "$sub" "$mode" "$kind" "$setp32"
 done
done
for kind in main block; do
 c="$(metric '\bsetp\..*\.u64\b' "$OUTDIR/m1_${kind}.ptx")"; s="$(metric '\bsetp\..*\.u64\b' "$OUTDIR/m5_${kind}.ptx")"; u="$(metric '\bsetp\..*\.u64\b' "$OUTDIR/m6_${kind}.ptx")"; us="$(metric '\bsub\.u64\b' "$OUTDIR/m6_${kind}.ptx")"
 ((c>=3&&s>=1&&s<c&&u==0&&us==0)) || { echo "$kind compare chain unexpected mode1=$c mode5=$s mode6=$u sub6=$us" >&2; exit 5; }
done
grep -Fq 'B300_SHARD_ADDRESS_MODE must be 0..6' "$GENSRC"; grep -Fq 'fully-u32 hi32-seed shard address is specialized for TARGET_W=28' "$GENSRC"
echo "b300-shard-address-hi32-u32addr-production-ptx-proof OK arch=$ARCH modes=1,5,6 mode6_setp64=0 mode6_sub64=0 device_div64=0 device_mul64=0 exact_proof=1"
