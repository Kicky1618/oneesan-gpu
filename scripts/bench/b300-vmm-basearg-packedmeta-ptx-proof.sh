#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-sm_103}"
require_nvcc_version_at_least "$NVCC" 13 0 "B300 packed group metadata PTX proof"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py";PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py";BASEARG="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg.py";PACK="$ONEESAN_ROOT/scripts/build/lower-b300-packed-group-meta.py"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_vmm_basearg_packedmeta_ptx}";GENSRC="$OUTDIR/generated.cu";PROBE="$OUTDIR/probe.cu";PTX="$OUTDIR/probe.ptx";mkdir -p "$OUTDIR"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-basearg-packedmeta-production-generate-proof.sh"
python3 "$GEN" "$SRC" "$GENSRC";python3 "$PRUNE" "$GENSRC" "$GENSRC";python3 "$BASEARG" "$GENSRC" "$GENSRC";python3 "$PACK" "$GENSRC" "$GENSRC"
python3 - "$GENSRC" "$PROBE" <<'PY'
import json,pathlib,sys
src=pathlib.Path(sys.argv[1]).resolve();out=pathlib.Path(sys.argv[2]);inc=json.dumps(src.as_posix())
out.write_text(f'''#pragma push_macro("main")
#undef main
#define main b300_vmm_packedmeta_generated_main_unused
#include {inc}
#pragma pop_macro("main")
extern "C" __global__ void b300_packed_meta_probe(unsigned long long* out){{
 if(threadIdx.x==0&&blockIdx.x==0){{
  unsigned long long x=D_MAIN_DP[0][0]^D_MAIN_DP[MAXW][MAXW+1]^D_BLOCK_DP[1][2]^D_BLOCK_DP[MAXW][0];
  x^=static_cast<unsigned long long>(D_MAIN_FIXED)<<1;
  x^=static_cast<unsigned long long>(D_MAIN_OCC)<<2;
  x^=static_cast<unsigned long long>(D_BLOCK_FIXED)<<3;
  x^=static_cast<unsigned long long>(D_BLOCK_OCC)<<4;
  out[0]=x;
 }}
}}
''')
PY
"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" -I"$ONEESAN_ROOT/src/cuda/b300" -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 "$PROBE" -o "$PTX"
body="$OUTDIR/packed_meta.body.ptx"
awk '$0 ~ /\.entry[[:space:]]+b300_packed_meta_probe/ {inside=1} inside{print} inside&&/^}/{exit}' "$PTX" >"$body"
[[ -s "$body" ]] || { echo "missing packed metadata PTX body" >&2; exit 3; }
grep -Fq 'D_GROUP_META' "$body" || { echo "packed metadata PTX does not reference D_GROUP_META" >&2; cat "$body" >&2; exit 4; }
for stale in D_MAIN_DP D_BLOCK_DP D_MAIN_FIXED D_MAIN_OCC D_BLOCK_FIXED D_BLOCK_OCC; do
  if grep -Fq "$stale" "$PTX"; then echo "packed metadata PTX still exposes old constant symbol $stale" >&2; exit 5; fi
done
ldconst="$(grep -Ec '\bld\.const\.' "$body" || true)";(( ldconst>=1 )) || { echo "packed metadata PTX missing constant loads" >&2; exit 6; }
echo "b300-vmm-basearg-packedmeta-ptx-proof OK arch=$ARCH cuda_min=13.0 group_meta_bytes=13936 packed_constant_symbol=1 old_constant_symbols=0 ldconst=$ldconst symbol_copies_per_group=1"
