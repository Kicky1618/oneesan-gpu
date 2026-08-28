#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py"
OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_vmm_production_proof.cu}"

bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-contiguous-layout-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-balanced-physical-layout-proof.sh"
python3 "$GEN" "$SRC" "$OUT"

grep -Fq '#include "b300_vmm_contiguous_storage.cuh"' "$OUT"
grep -Fq '__constant__ Count* D_MAIN_VBASE;' "$OUT"
grep -Fq '__constant__ Count* D_BLOCK_VBASE;' "$OUT"
grep -Fq 'Count global_load_main(Code g){return D_MAIN_VBASE[g];}' "$OUT"
grep -Fq 'Count global_load_block(Code g){return D_BLOCK_VBASE[g];}' "$OUT"
grep -Fq 'void global_store_main(Code g,Count v){D_MAIN_VBASE[g]=v;}' "$OUT"
grep -Fq 'void global_store_block(Code g,Count v){D_BLOCK_VBASE[g]=v;}' "$OUT"
grep -Fq 'b300_vmm::ContiguousStorage main_store,block_store;' "$OUT"
grep -Fq 'block_store.create(blockN,ng,int(main_store.mapped_units%size_t(ng)),"auth block");' "$OUT"
grep -Fq 'mp[d]=main_base+Code(d)*mc;bp[d]=block_base+Code(d)*bc;' "$OUT"
grep -Fq 'cudaMemcpyToSymbol(D_MAIN_VBASE,&main_base,sizeof(main_base))' "$OUT"
grep -Fq 'cudaMemcpyToSymbol(D_BLOCK_VBASE,&block_base,sizeof(block_base))' "$OUT"
grep -Fq 'main_store.destroy();block_store.destroy();' "$OUT"
grep -Fq 'backend=gridfp-b300-hbm32-fullmate-dropN-vmm n=' "$OUT"

if grep -Fq 'cudaMalloc(&mp[d]' "$OUT" || grep -Fq 'cudaMalloc(&bp[d]' "$OUT"; then
  echo "generated VMM source still cudaMallocs authoritative shards" >&2
  exit 3
fi
if grep -Fq 'cudaFree(mp[d])' "$OUT" || grep -Fq 'cudaFree(bp[d])' "$OUT"; then
  echo "generated VMM source still cudaFrees VMM logical views" >&2
  exit 4
fi

echo "b300-vmm-production-generate-proof OK direct_global_index=1 logical_shard_views=1 authoritative_cudaMalloc=0 authoritative_cudaFree=0 balanced_physical_rotation=1 combined_imbalance_le_one_granularity=1"
