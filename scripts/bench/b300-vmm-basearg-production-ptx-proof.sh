#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-sm_103}"
require_nvcc_version_at_least "$NVCC" 13 0 "B300 sm_103 VMM basearg PTX proof"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py";PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py";LOWER="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg.py"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_vmm_basearg_production_ptx}";GENSRC="$OUTDIR/generated.cu";PROBE="$OUTDIR/probe.cu";PTX="$OUTDIR/probe.ptx";mkdir -p "$OUTDIR"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-basearg-production-generate-proof.sh"
python3 "$GEN" "$SRC" "$GENSRC";python3 "$PRUNE" "$GENSRC" "$GENSRC";python3 "$LOWER" "$GENSRC" "$GENSRC"
python3 - "$GENSRC" "$PROBE" <<'PY'
import json,pathlib,sys
src=pathlib.Path(sys.argv[1]).resolve();out=pathlib.Path(sys.argv[2]);inc=json.dumps(src.as_posix())
out.write_text(f'''#pragma push_macro("main")
#undef main
#define main b300_vmm_basearg_generated_main_unused
#include {inc}
#pragma pop_macro("main")
extern "C" __global__ void b300_vmm_basearg_main_load_probe(Count*out,const Count*base,Code g){{if(threadIdx.x==0&&blockIdx.x==0)out[0]=global_load_main(base,g);}}
extern "C" __global__ void b300_vmm_basearg_block_load_probe(Count*out,const Count*base,Code g){{if(threadIdx.x==0&&blockIdx.x==0)out[0]=global_load_block(base,g);}}
extern "C" __global__ void b300_vmm_basearg_main_store_probe(Count*base,Code g,Count v){{if(threadIdx.x==0&&blockIdx.x==0)global_store_main(base,g,v);}}
extern "C" __global__ void b300_vmm_basearg_block_store_probe(Count*base,Code g,Count v){{if(threadIdx.x==0&&blockIdx.x==0)global_store_block(base,g,v);}}
''')
PY
"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" -I"$ONEESAN_ROOT/src/cuda/b300" -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 "$PROBE" -o "$PTX"
extract(){ local k="$1" o="$2";awk -v k="$k" '$0 ~ "\\.entry[[:space:]]+" k {inside=1} inside{print} inside&&/^}/{exit}' "$PTX" >"$o";[[ -s "$o" ]]||{ echo "missing PTX body $k" >&2;exit 3;};}
metric(){ grep -Ec "$1" "$2" || true; }
for k in main_load block_load main_store block_store;do
  body="$OUTDIR/${k}.body.ptx";extract "b300_vmm_basearg_${k}_probe" "$body"
  const="$(metric '\bld\.const\.u64\b' "$body")";param="$(metric '\bld\.param\.u64\b' "$body")";div="$(metric '\b(div|rem)\.u64\b' "$body")";mul="$(metric '\bmul\.(hi|lo|wide)\.u64\b' "$body")";setp="$(metric '\bsetp\..*\.u64\b' "$body")"
  (( const==0 && param>=1 && div==0 && mul==0 && setp==0 )) || { echo "$k basearg PTX gate failed const=$const param=$param div=$div mul=$mul setp=$setp" >&2;cat "$body" >&2;exit 4; }
  printf '%s_ldconst_u64=%s\n%s_ldparam_u64=%s\n' "$k" "$const" "$k" "$param"
done
if grep -Fq 'D_MAIN_VBASE' "$PTX" || grep -Fq 'D_BLOCK_VBASE' "$PTX";then echo "basearg PTX still contains VMM base symbols" >&2;exit 5;fi
echo "b300-vmm-basearg-production-ptx-proof OK arch=$ARCH cuda_min=13.0 vmm_base_source=kernel_param vmm_base_symbols=0 base_ldconst_u64=0 owner_div64=0 owner_mul64=0 owner_compare64=0 direct_global_index=1"
