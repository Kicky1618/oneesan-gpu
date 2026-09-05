#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py"
PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py"
BASEARG="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg.py"
PACK="$ONEESAN_ROOT/scripts/build/lower-b300-packed-group-meta.py"
OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_vmm_basearg_packedmeta_proof.cu}"

bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-basearg-production-generate-proof.sh"
python3 "$GEN" "$SRC" "$OUT"
python3 "$PRUNE" "$OUT" "$OUT"
python3 "$BASEARG" "$OUT" "$OUT"
python3 "$PACK" "$OUT" "$OUT"

grep -Fq 'struct DeviceGroupMeta{' "$OUT"
grep -Fq 'static_assert(sizeof(DeviceGroupMeta)==13936' "$OUT"
grep -Fq '__constant__ DeviceGroupMeta D_GROUP_META;' "$OUT"
grep -Fq 'cudaMemcpyToSymbol(D_GROUP_META,&hmeta,sizeof(hmeta))' "$OUT"
grep -Fq '#define D_MAIN_DP (D_GROUP_META.main_dp)' "$OUT"
grep -Fq '#define D_BLOCK_DP (D_GROUP_META.block_dp)' "$OUT"
grep -Fq '#define D_MAIN_FIXED (D_GROUP_META.main_fixed)' "$OUT"
for stale in 'cudaMemcpyToSymbol(D_MAIN_DP' 'cudaMemcpyToSymbol(D_BLOCK_DP' 'cudaMemcpyToSymbol(D_MAIN_FIXED' 'cudaMemcpyToSymbol(D_MAIN_OCC' 'cudaMemcpyToSymbol(D_BLOCK_FIXED' 'cudaMemcpyToSymbol(D_BLOCK_OCC'; do
  if grep -Fq "$stale" "$OUT"; then echo "packedmeta source still contains unpacked copy $stale" >&2; exit 3; fi
done
copies="$(grep -Fo 'cudaMemcpyToSymbol(D_GROUP_META' "$OUT" | wc -l)"
[[ "$copies" == 1 ]] || { echo "expected one packed group metadata copy site, got $copies" >&2; exit 4; }

echo "b300-vmm-basearg-packedmeta-production-generate-proof OK group_meta_bytes=13936 symbol_copies_per_group=1 old_symbol_copies_per_group=6 copy_call_reduction=6x vmm_base_source=kernel_param compact_interval_bytes=24"
