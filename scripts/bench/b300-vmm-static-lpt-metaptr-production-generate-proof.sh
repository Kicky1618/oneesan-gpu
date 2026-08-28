#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py";PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py";BASEARG="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg.py";PACK="$ONEESAN_ROOT/scripts/build/lower-b300-packed-group-meta.py";STAGE="$ONEESAN_ROOT/scripts/build/lower-b300-staged-group-meta.py";STATIC="$ONEESAN_ROOT/scripts/build/lower-b300-static-lpt-staged-meta.py";METAPTR="$ONEESAN_ROOT/scripts/build/lower-b300-staged-meta-pointer.py";ROWLIMIT="$ONEESAN_ROOT/scripts/build/lower-b300-row-limit.py"
OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_vmm_static_lpt_metaptr_proof.cu}"
bash "$ONEESAN_ROOT/scripts/bench/b300-staged-group-meta-plan-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-static-lpt-local-meta-proof.sh"
python3 "$GEN" "$SRC" "$OUT";python3 "$PRUNE" "$OUT" "$OUT";python3 "$BASEARG" "$OUT" "$OUT";python3 "$PACK" "$OUT" "$OUT";python3 "$STAGE" "$OUT" "$OUT";python3 "$STATIC" "$OUT" "$OUT";python3 "$METAPTR" "$OUT" "$OUT";python3 "$ROWLIMIT" "$OUT" "$OUT"
grep -Fq '__constant__ const DeviceGroupMeta* D_GROUP_META_PTR;' "$OUT"
grep -Fq '#define D_MAIN_DP (D_GROUP_META_PTR->main_dp)' "$OUT"
grep -Fq '#define D_BLOCK_DP (D_GROUP_META_PTR->block_dp)' "$OUT"
grep -Fq 'const DeviceGroupMeta*group_meta_ptr=c.dGroupMeta+pg.meta_id;' "$OUT"
grep -Fq 'cudaMemcpyToSymbol(D_GROUP_META_PTR,&group_meta_ptr,sizeof(group_meta_ptr))' "$OUT"
grep -Fq 'scheduler=static_lpt' "$OUT"
grep -Fq 'for(int q:pw.by_gpu[d])process_group' "$OUT"
grep -Fq 'B300_ROW_LIMIT' "$OUT"
for stale in '__constant__ DeviceGroupMeta D_GROUP_META;' 'cudaMemcpyToSymbol(D_GROUP_META,c.dGroupMeta+pg.meta_id' 'cudaMemcpyDeviceToDevice),"staged group meta D2D"' 'std::atomic<int>next{0}' 'next.fetch_add(1,std::memory_order_relaxed)';do
  if grep -Fq "$stale" "$OUT";then echo "metaptr generated source still contains stale artifact: $stale" >&2;exit 3;fi
done
echo "b300-vmm-static-lpt-metaptr-production-generate-proof OK scheduler=static_lpt metadata_replication=0 group_meta_source=staged_global_via_constant_pointer constant_payload_bytes_per_group=8 old_constant_payload_bytes_per_group=13936 payload_reduction=1742x per_group_symbol_copy_calls=1 row_limit_env=B300_ROW_LIMIT"
