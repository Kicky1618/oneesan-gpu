#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
NVCC="${NVCC:-nvcc}";ARCH="${ARCH:-sm_103}"
require_nvcc_version_at_least "$NVCC" 13 0 "B300 static LPT metadata-pointer PTX proof"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py";PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py";BASEARG="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg.py";PACK="$ONEESAN_ROOT/scripts/build/lower-b300-packed-group-meta.py";STAGE="$ONEESAN_ROOT/scripts/build/lower-b300-staged-group-meta.py";STATIC="$ONEESAN_ROOT/scripts/build/lower-b300-static-lpt-staged-meta.py";METAPTR="$ONEESAN_ROOT/scripts/build/lower-b300-staged-meta-pointer.py"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_static_lpt_metaptr_ptx}";GENSRC="$OUTDIR/generated.cu";PROBE="$OUTDIR/probe.cu";PTX="$OUTDIR/probe.ptx";BODY="$OUTDIR/probe.body.ptx";mkdir -p "$OUTDIR"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-static-lpt-metaptr-production-generate-proof.sh"
python3 "$GEN" "$SRC" "$GENSRC";python3 "$PRUNE" "$GENSRC" "$GENSRC";python3 "$BASEARG" "$GENSRC" "$GENSRC";python3 "$PACK" "$GENSRC" "$GENSRC";python3 "$STAGE" "$GENSRC" "$GENSRC";python3 "$STATIC" "$GENSRC" "$GENSRC";python3 "$METAPTR" "$GENSRC" "$GENSRC"
python3 - "$GENSRC" "$PROBE" <<'PY'
import json,pathlib,sys
src=pathlib.Path(sys.argv[1]).resolve();out=pathlib.Path(sys.argv[2]);inc=json.dumps(src.as_posix())
out.write_text(f'''#pragma push_macro("main")
#undef main
#define main b300_static_lpt_metaptr_generated_main_unused
#include {inc}
#pragma pop_macro("main")
extern "C" __global__ void b300_static_lpt_metaptr_probe(Code*out,Code i){{if(threadIdx.x==0&&blockIdx.x==0){{Code g=0;MateID m=unrank_group_global_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP,g);out[0]=g^Code(m);}}}}
''')
PY
"$NVCC" -O3 -std=c++17 -ptx -arch="$ARCH" -I"$ONEESAN_ROOT/src/cuda/b300" -DTARGET_W=28 -DLOW_LUT_K=13 -DHIGH_LUT_K=13 "$PROBE" -o "$PTX"
awk '$0 ~ /\.entry[[:space:]]+b300_static_lpt_metaptr_probe/ {inbody=1} inbody{print} inbody&&/^}/{exit}' "$PTX" >"$BODY"
[[ -s "$BODY" ]] || { echo "missing metaptr PTX body" >&2; exit 3; }
ldconst="$(grep -Ec '\bld\.const\.(u64|b64)\b' "$BODY" || true)"
ldglobal="$(grep -Ec '\bld\.global(\.[a-z]+)*\.(u32|u64|b32|b64)\b' "$BODY" || true)"
grep -Fq 'D_GROUP_META_PTR' "$BODY" || { echo "metaptr PTX missing constant metadata pointer symbol" >&2; cat "$BODY" >&2; exit 4; }
(( ldconst>=1 )) || { echo "metaptr PTX missing constant pointer load" >&2; cat "$BODY" >&2; exit 5; }
(( ldglobal>=1 )) || { echo "metaptr PTX missing staged global metadata loads" >&2; cat "$BODY" >&2; exit 6; }
if grep -Eq '[^A-Z_]D_GROUP_META([^_A-Z]|$)' "$BODY";then echo "metaptr PTX unexpectedly references full constant metadata object" >&2;cat "$BODY" >&2;exit 7;fi
printf 'metaptr_ldconst_pointer=%s\nmetaptr_ldglobal_metadata=%s\n' "$ldconst" "$ldglobal"
echo "b300-vmm-static-lpt-metaptr-ptx-proof OK arch=$ARCH constant_metadata_payload_bytes=8 full_constant_metadata_object=0 staged_global_metadata_loads=1 scheduler=static_lpt"
