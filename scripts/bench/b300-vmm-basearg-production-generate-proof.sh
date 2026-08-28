#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py"
PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py"
LOWER="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg.py"
OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_vmm_basearg_proof.cu}"

bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-production-generate-proof.sh"
python3 "$GEN" "$SRC" "$OUT"
python3 "$PRUNE" "$OUT" "$OUT"
python3 "$LOWER" "$OUT" "$OUT"

grep -Fq 'global_load_main(const Count* base,Code g)' "$OUT"
grep -Fq 'global_load_block(const Count* base,Code g)' "$OUT"
grep -Fq 'global_store_main(Count* base,Code g,Count v)' "$OUT"
grep -Fq 'global_store_block(Count* base,Code g,Count v)' "$OUT"
grep -Fq 'gather_main_kernel(Count*out,MateID*mates,Code n,const Count* auth)' "$OUT"
grep -Fq 'gather_block_kernel(Count*out,Code n,const Count* auth)' "$OUT"
grep -Fq 'scatter_main_kernel(const Count*in,Code n,Count* auth)' "$OUT"
grep -Fq 'scatter_block_kernel(const Count*in,Code n,Count* auth)' "$OUT"
grep -Fq 'template<bool SCATTER>' "$OUT"
grep -Fq 'interval_io_kernel(Count*buf,const PeerInterval*iv,size_t niv,Count*auth)' "$OUT"
grep -Fq 'Count*authMain=nullptr,*authBlock=nullptr' "$OUT"
grep -Fq 'void init(int d,Count mod,Count*main_base,Count*block_base)' "$OUT"
grep -Fq 'ctx[d].init(d,mod,main_base,block_base)' "$OUT"
grep -Fq 'c.authMain' "$OUT"
grep -Fq 'c.authBlock' "$OUT"
for stale in 'template<bool BLOCK,bool SCATTER>' D_MAIN_VBASE D_BLOCK_VBASE 'cudaMemcpyToSymbol(D_MAIN_VBASE' 'cudaMemcpyToSymbol(D_BLOCK_VBASE' B300_FAST_SHARD_ADDRESS8 ShardAddress8 D_MAIN_PTR D_BLOCK_PTR D_MAIN_CHUNK D_BLOCK_CHUNK D_NGPU D_MAIN_W D_BLOCK_W; do
  if grep -Fq "$stale" "$OUT"; then echo "basearg generated source still contains $stale" >&2; exit 3; fi
done

echo "b300-vmm-basearg-production-generate-proof OK vmm_base_source=kernel_param vmm_base_symbols=0 vmm_base_symbol_copies=0 interval_template_axes=1 logical_shard_chunks=0 logical_shard_views=0 legacy_shard_address_scaffolding=0 compact_interval_bytes=24 direct_global_index=1"
