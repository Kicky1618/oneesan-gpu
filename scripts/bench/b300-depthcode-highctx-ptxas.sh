#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
ARCH="${ARCH:-native}"
TRANSPOSE_MODE="${TRANSPOSE_MODE:-pipeline}"
PM_ACCUM="${PM_ACCUM:-0}"
TERNARY_KEY4="${TERNARY_KEY4:-1}"
OUT="${OUT:-$ONEESAN_ROOT/work/b300_depthcode_highctx_ptxas_n${N}_${TRANSPOSE_MODE}_pm${PM_ACCUM}_key4${TERNARY_KEY4}.tsv}"
LOGDIR="${LOGDIR:-$ONEESAN_ROOT/work/b300_depthcode_highctx_ptxas_n${N}_${TRANSPOSE_MODE}_pm${PM_ACCUM}_key4${TERNARY_KEY4}_logs}"
PARSER="$ONEESAN_ROOT/scripts/bench/parse-ptxas-resources.py"

case "$TRANSPOSE_MODE" in sync|events|pipeline) ;; *) echo "invalid TRANSPOSE_MODE" >&2; exit 2;; esac
if [[ "$PM_ACCUM" != 0 && "$PM_ACCUM" != 1 ]]; then echo "PM_ACCUM must be 0 or 1" >&2; exit 2; fi
if [[ "$TERNARY_KEY4" != 0 && "$TERNARY_KEY4" != 1 ]]; then echo "TERNARY_KEY4 must be 0 or 1" >&2; exit 2; fi
if ! command -v nvcc >/dev/null; then echo "nvcc not found" >&2; exit 2; fi
mkdir -p "$(dirname "$OUT")" "$LOGDIR"

build_one(){
  local ctx="$1"
  local load="${2:-global}"
  local label="depthcode_payload_${ctx}"
  [[ "$load" == ldg ]] && label="${label}_ldg"
  local log="$LOGDIR/${label}.ptxas.log"
  echo "=== ptxas build $label ===" >&2
  N="$N" ARCH="$ARCH" TRANSPOSE_MODE="$TRANSPOSE_MODE" HIGH_CTX="$ctx" DEPTHCODE_DECODE_LOAD="$load" \
    PM_ACCUM="$PM_ACCUM" TERNARY_KEY4="$TERNARY_KEY4" PTXAS_VERBOSE=1 \
    OUT="$ONEESAN_BUILD_DIR/ptxas_${label}_${TRANSPOSE_MODE}_n${N}" \
    bash "$ONEESAN_ROOT/scripts/build/b300-bucket-snake-pattern10-depthcode-graph-batch.sh" \
    >"$LOGDIR/${label}.out" 2>"$log"
  python3 "$PARSER" "$log" --label "$label"
}

printf 'backend\tkernel\tregisters\tstack_bytes\tspill_store_bytes\tspill_load_bytes\tsmem_bytes\tcmem0_bytes\n' >"$OUT"
{
  build_one thread
  build_one resolved
  build_one resolved_delta
  build_one warp
  build_one warp ldg
  build_one warpstriped
  build_one warpstriped ldg
  build_one warpstriped_delta
  build_one warpstriped_delta ldg
  build_one warpstriped_delta_cross5
  build_one warpstriped_delta_cross5 ldg
  build_one warpstriped_delta_direct_cross5
  build_one warpstriped_delta_direct_cross5 ldg
  build_one warpstriped_delta_direct_affine_cross5
  build_one warpstriped_delta_direct_affine_cross5 ldg
  build_one warpstriped_delta_direct_affine_prekey_cross5
  build_one warpstriped_delta_direct_affine_prekey_cross5 ldg
  build_one warpstriped_delta_direct_affine_prekey_rank16_cross5
  build_one warpstriped_delta_direct_affine_prekey_rank16_cross5 ldg
  build_one warpstriped_delta_direct_affine_prekey_rankstream_cross5
  build_one warpstriped_delta_direct_affine_prekey_rankstream_cross5 ldg
  build_one warpstriped_delta_direct_affine_rankstream32_cross5
  build_one warpstriped_delta_direct_affine_rankstream32_cross5 ldg
  build_one warpstriped_delta_direct_affine_rankchunk32_cross5
  build_one warpstriped_delta_direct_affine_rankchunk32_cross5 ldg
} >>"$OUT"

cat "$OUT"
python3 - "$OUT" <<'PY'
import csv, sys
rows=list(csv.DictReader(open(sys.argv[1]),delimiter='\t'))
def ints(group,key): return [int(r[key]) for r in group if r[key] != 'NA']
backends=(
    'depthcode_payload_thread','depthcode_payload_resolved','depthcode_payload_resolved_delta',
    'depthcode_payload_warp','depthcode_payload_warp_ldg',
    'depthcode_payload_warpstriped','depthcode_payload_warpstriped_ldg',
    'depthcode_payload_warpstriped_delta','depthcode_payload_warpstriped_delta_ldg',
    'depthcode_payload_warpstriped_delta_cross5','depthcode_payload_warpstriped_delta_cross5_ldg',
    'depthcode_payload_warpstriped_delta_direct_cross5','depthcode_payload_warpstriped_delta_direct_cross5_ldg',
    'depthcode_payload_warpstriped_delta_direct_affine_cross5','depthcode_payload_warpstriped_delta_direct_affine_cross5_ldg',
    'depthcode_payload_warpstriped_delta_direct_affine_prekey_cross5','depthcode_payload_warpstriped_delta_direct_affine_prekey_cross5_ldg',
    'depthcode_payload_warpstriped_delta_direct_affine_prekey_rank16_cross5','depthcode_payload_warpstriped_delta_direct_affine_prekey_rank16_cross5_ldg',
    'depthcode_payload_warpstriped_delta_direct_affine_prekey_rankstream_cross5','depthcode_payload_warpstriped_delta_direct_affine_prekey_rankstream_cross5_ldg',
    'depthcode_payload_warpstriped_delta_direct_affine_rankstream32_cross5','depthcode_payload_warpstriped_delta_direct_affine_rankstream32_cross5_ldg',
    'depthcode_payload_warpstriped_delta_direct_affine_rankchunk32_cross5','depthcode_payload_warpstriped_delta_direct_affine_rankchunk32_cross5_ldg')
for backend in backends:
    all_rows=[r for r in rows if r['backend']==backend]; high=[r for r in all_rows if 'high' in r['kernel'].lower()]
    if not high: continue
    regs=ints(high,'registers'); smem=ints(high,'smem_bytes'); stores=ints(high,'spill_store_bytes'); loads=ints(high,'spill_load_bytes'); cmem=ints(all_rows,'cmem0_bytes')
    print(f'{backend}_high_max_registers={max(regs) if regs else "NA"}')
    print(f'{backend}_high_max_static_smem_bytes={max(smem) if smem else "NA"}')
    print(f'{backend}_high_spill_store_bytes={sum(stores) if stores else "NA"}')
    print(f'{backend}_high_spill_load_bytes={sum(loads) if loads else "NA"}')
    print(f'{backend}_max_cmem0_bytes={max(cmem) if cmem else "NA"}')
    if cmem and max(cmem) >= 60*1024: print(f'{backend}_cmem0_headroom_warning=1')
print('rankstream32_model=key23+prefix9_per_code+blockbase32+rank16_per_L')
print('rankstream32_per_code_meta_bytes=4')
print('rankstream32_block_base_bytes_per_code=0.125')
print('rankstream32_cross_runtime_divmod=1')
print('rankchunk32_model=chunk24+prefix8_per_code+blockbase16+rank16_per_L')
print('rankchunk32_per_code_meta_bytes=4')
print('rankchunk32_block_base_bytes_per_code=0.25')
print('rankchunk32_extra_block_base_bytes_per_code_vs_rankstream32=0.125')
print('rankchunk32_cross_runtime_divmod=0')
print('rankchunk32_cross_runtime_direct_lookup=0')
PY

echo "depthcode-highctx-ptxas OK n=$N arch=$ARCH transpose=$TRANSPOSE_MODE result=$OUT logs=$LOGDIR rankstream32=1 rankchunk32=1" >&2
