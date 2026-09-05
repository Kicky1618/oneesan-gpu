#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-sm_103}"
require_nvcc_version_at_least "$NVCC" 13 0 "B300 metadata kernel-arg PTX proof"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py";PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py";BASEARG="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg.py";PACK="$ONEESAN_ROOT/scripts/build/lower-b300-packed-group-meta.py";STAGE="$ONEESAN_ROOT/scripts/build/lower-b300-staged-group-meta.py";STATIC="$ONEESAN_ROOT/scripts/build/lower-b300-static-lpt-staged-meta.py";METAARG="$ONEESAN_ROOT/scripts/build/lower-b300-staged-meta-kernelarg.py"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_static_lpt_metakernelarg_ptx}";GENSRC="$OUTDIR/generated.cu";PROBE="$OUTDIR/probe.cu";PTX="$OUTDIR/probe.ptx";BODY="$OUTDIR/probe.body.ptx";mkdir -p "$OUTDIR"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-metakernelarg-production-generate-proof.sh"
python3 "$GEN" "$SRC" "$GENSRC";python3 "$PRUNE" "$GENSRC" "$GENSRC";python3 "$BASEARG" "$GENSRC" "$GENSRC";python3 "$PACK" "$GENSRC" "$GENSRC";python3 "$STAGE" "$GENSRC" "$GENSRC";python3 "$STATIC" "$GENSRC" "$GENSRC";python3 "$METAARG" "$GENSRC" "$GENSRC"
python3 - "$GENSRC" "$PROBE" <<'PY'
import json,pathlib,sys
src=pathlib.Path(sys.argv[1]).resolve();out=pathlib.Path(sys.argv[2]);inc=json.dumps(src.as_posix())
out.write_text(f'''#pragma push_macro("main")
#undef main
#define main b300_metakernelarg_generated_main_unused
#include {inc}
#pragma pop_macro("main")
extern "C" __global__ void b300_metakernelarg_probe(Code*out,Code i,const DeviceGroupMeta*meta){{if(threadIdx.x==0&&blockIdx.x==0){{Code g=0;MateID m=unrank_group_global_t<TARGET_W>(i,meta->main_fixed,meta->main_occ,meta->main_dp,g);out[0]=g^Code(m);}}}}
''')
PY
"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" -I"$ONEESAN_ROOT/src/cuda/b300" -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 "$PROBE" -o "$PTX"
awk '$0 ~ /\.entry[[:space:]]+b300_metakernelarg_probe/ {inbody=1} inbody{print} inbody&&/^}/{exit}' "$PTX" >"$BODY"
[[ -s "$BODY" ]]||{ echo "missing metadata kernel-arg PTX body" >&2;exit 3; }
param="$(grep -Ec '\bld\.param\.(u64|b64)\b' "$BODY"||true)";global="$(grep -Ec '\bld\.global(\.[a-z]+)*\.(u32|u64|b32|b64)\b' "$BODY"||true)"
((param>=1&&global>=1))||{ echo "metadata kernel-arg PTX gate failed param=$param global=$global" >&2;cat "$BODY" >&2;exit 4; }
if grep -Fq 'D_GROUP_META' "$PTX";then echo "metadata kernel-arg PTX still contains group metadata symbol" >&2;exit 5;fi
printf 'metakernelarg_ldparam=%s\nmetakernelarg_ldglobal=%s\n' "$param" "$global"
echo "b300-vmm-static-lpt-metakernelarg-ptx-proof OK arch=$ARCH metadata_symbol=0 per_group_meta_copy_bytes=0 metadata_pointer_kernel_arg=1 staged_global_metadata_loads=1"
