#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

NVCC="${NVCC:-nvcc}"
ARCH="${ARCH:-sm_103}"
command -v "$NVCC" >/dev/null || { echo "$NVCC not found" >&2; exit 2; }

SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py"
PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_vmm_production_ptx}"
GENSRC="$OUTDIR/generated.cu"
PROBE="$OUTDIR/probe.cu"
PTX="$OUTDIR/probe.ptx"
mkdir -p "$OUTDIR"

bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-generate-proof.sh"
python3 "$GEN" "$SRC" "$GENSRC"
python3 "$PRUNE" "$GENSRC" "$GENSRC"
for stale in D_MAIN_PTR D_BLOCK_PTR D_MAIN_CHUNK D_BLOCK_CHUNK D_NGPU; do
  if grep -Fq "$stale" "$GENSRC"; then
    echo "PTX input still contains stale shard symbol $stale" >&2
    exit 3
  fi
done
python3 - "$GENSRC" "$PROBE" <<'PY'
import json,pathlib,sys
src=pathlib.Path(sys.argv[1]).resolve()
out=pathlib.Path(sys.argv[2])
inc=json.dumps(src.as_posix())
out.write_text(f'''#pragma push_macro("main")
#undef main
#define main b300_vmm_generated_main_unused
#include {inc}
#pragma pop_macro("main")
extern "C" __global__ void b300_vmm_main_load_probe(Count* out, Code g){{if(threadIdx.x==0&&blockIdx.x==0)out[0]=global_load_main(g);}}
extern "C" __global__ void b300_vmm_block_load_probe(Count* out, Code g){{if(threadIdx.x==0&&blockIdx.x==0)out[0]=global_load_block(g);}}
extern "C" __global__ void b300_vmm_main_store_probe(Code g, Count v){{if(threadIdx.x==0&&blockIdx.x==0)global_store_main(g,v);}}
extern "C" __global__ void b300_vmm_block_store_probe(Code g, Count v){{if(threadIdx.x==0&&blockIdx.x==0)global_store_block(g,v);}}
''')
PY

"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" \
  -I"$ONEESAN_ROOT/src/cuda/b300" \
  -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 -DB300_FAST_SHARD_ADDRESS8=0 \
  "$PROBE" -o "$PTX"

extract(){
  local kernel="$1" out="$2"
  awk -v k="$kernel" '$0 ~ "\\.entry[[:space:]]+" k {inside=1} inside{print} inside&&/^}/{exit}' "$PTX" >"$out"
  [[ -s "$out" ]] || { echo "missing PTX body for $kernel" >&2; exit 4; }
}
metric(){ grep -Ec "$1" "$2" || true; }

for k in main_load block_load main_store block_store; do
  kernel="b300_vmm_${k}_probe"
  body="$OUTDIR/${k}.body.ptx"
  extract "$kernel" "$body"
  div="$(metric '\b(div|rem)\.u64\b' "$body")"
  mul="$(metric '\bmul\.(hi|lo|wide)\.u64\b' "$body")"
  setp="$(metric '\bsetp\..*\.u64\b' "$body")"
  ldconst="$(metric '\bld\.const\.u64\b' "$body")"
  if (( div != 0 || mul != 0 || setp != 0 )); then
    echo "$k VMM direct path still has owner-style u64 arithmetic: div=$div mul=$mul setp=$setp" >&2
    cat "$body" >&2
    exit 5
  fi
  if (( ldconst < 1 )); then
    echo "$k VMM direct path missing constant base pointer load" >&2
    exit 6
  fi
  for stale in D_MAIN_PTR D_BLOCK_PTR D_MAIN_CHUNK D_BLOCK_CHUNK D_NGPU; do
    if grep -Fq "$stale" "$body"; then
      echo "$k VMM direct PTX still references stale shard symbol $stale" >&2
      exit 7
    fi
  done
  printf '%s_divrem_u64=%s\n%s_mul_u64=%s\n%s_setp_u64=%s\n%s_ldconst_u64=%s\n' "$k" "$div" "$k" "$mul" "$k" "$setp" "$k" "$ldconst"
done

grep -Fq 'D_MAIN_VBASE' "$OUTDIR/main_load.body.ptx"
grep -Fq 'D_BLOCK_VBASE' "$OUTDIR/block_load.body.ptx"
grep -Fq 'D_MAIN_VBASE' "$OUTDIR/main_store.body.ptx"
grep -Fq 'D_BLOCK_VBASE' "$OUTDIR/block_store.body.ptx"

echo "b300-vmm-production-ptx-proof OK arch=$ARCH owner_div64=0 owner_mul64=0 owner_compare64=0 dynamic_shard_ptr_index=0 stale_shard_symbols=0 direct_global_index=1"
