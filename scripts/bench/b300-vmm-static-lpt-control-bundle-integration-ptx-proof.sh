#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-sm_103}";require_nvcc_version_at_least "$NVCC" 13 0 "B300 control bundle PTX proof"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_control_bundle_ptx}";GENSRC="$OUTDIR/generated.cu";PROBE="$OUTDIR/probe.cu";PTX="$OUTDIR/probe.ptx";BODY="$OUTDIR/body.ptx";mkdir -p "$OUTDIR"
OUT="$GENSRC" bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-control-bundle-production-generate-proof.sh"
python3 - "$GENSRC" "$PROBE" <<'PY'
import json,pathlib,sys
src=pathlib.Path(sys.argv[1]).resolve();out=pathlib.Path(sys.argv[2]);inc=json.dumps(src.as_posix())
out.write_text(f'''#pragma push_macro("main")\n#undef main\n#define main b300_bundle_generated_main_unused\n#include {inc}\n#pragma pop_macro("main")\nextern "C" __global__ void b300_bundle_meta_probe(Code*out,Code i,const DeviceGroupMeta*meta){{if(threadIdx.x==0&&blockIdx.x==0){{Code g=0;MateID m=unrank_group_global_t<TARGET_W>(i,meta->main_fixed,meta->main_occ,meta->main_dp,g);out[0]=g^Code(m);}}}}\n''')
PY
"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" -I"$ONEESAN_ROOT/src/cuda/b300" -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 "$PROBE" -o "$PTX"
awk '$0 ~ /\.entry[[:space:]]+b300_bundle_meta_probe/ {inbody=1} inbody{print} inbody&&/^}/{exit}' "$PTX" >"$BODY";[[ -s "$BODY" ]]||exit 3
param="$(grep -Ec '\bld\.param\.(u64|b64)\b' "$BODY"||true)";global="$(grep -Ec '\bld\.global(\.[a-z]+)*\.(u32|u64|b32|b64)\b' "$BODY"||true)";((param>=1&&global>=1))||{ cat "$BODY" >&2;exit 4; }
for stale in D_GROUP_META D_MAIN_PTR D_BLOCK_PTR D_MAIN_CHUNK D_BLOCK_CHUNK D_NGPU;do grep -Fq "$stale" "$PTX"&&{ echo "stale PTX symbol $stale" >&2;exit 5;}||true;done
echo "b300-vmm-static-lpt-control-bundle-integration-ptx-proof OK arch=$ARCH metadata_kernelarg=1 staged_global_metadata=1 stale_shard_symbols=0"
