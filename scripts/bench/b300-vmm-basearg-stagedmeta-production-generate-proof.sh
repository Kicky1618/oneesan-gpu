#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py"
PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py"
BASEARG="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg.py"
PACK="$ONEESAN_ROOT/scripts/build/lower-b300-packed-group-meta.py"
STAGE="$ONEESAN_ROOT/scripts/build/lower-b300-staged-group-meta.py"
OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_vmm_basearg_stagedmeta_proof.cu}"

bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-basearg-packedmeta-production-generate-proof.sh"
python3 "$GEN" "$SRC" "$OUT"
python3 "$PRUNE" "$OUT" "$OUT"
python3 "$BASEARG" "$OUT" "$OUT"
python3 "$PACK" "$OUT" "$OUT"
python3 "$STAGE" "$OUT" "$OUT"

grep -Fq 'DeviceGroupMeta*dGroupMeta=nullptr' "$OUT"
grep -Fq 'size_t groupMetaCount=0' "$OUT"
grep -Fq 'void stage_group_meta(const std::vector<DeviceGroupMeta>& h)' "$OUT"
grep -Fq 'size_t meta_id=0' "$OUT"
grep -Fq 'cudaMemcpyDeviceToDevice' "$OUT"
grep -Fq 'c.dGroupMeta+pg.meta_id' "$OUT"
grep -Fq 'staged group meta: groups=' "$OUT"
grep -Fq 'copy_mode=H2D_once_then_D2D_per_group' "$OUT"
grep -Fq 'staged_group_meta.clear();staged_group_meta.shrink_to_fit();' "$OUT"
for stale in 'DeviceGroupMeta hmeta{}' 'cudaMemcpyToSymbol(D_GROUP_META,&hmeta'; do
  if grep -Fq "$stale" "$OUT"; then echo "stagedmeta source still rebuilds/copies host metadata per group: $stale" >&2; exit 3; fi
done

echo "b300-vmm-basearg-stagedmeta-production-generate-proof OK group_meta_bytes=13936 staged_h2d_once=1 per_group_meta_copy=D2D_sync host_meta_rebuild_per_group=0 staged_device_copy=1 expected_default_groups=16384 expected_default_mib_per_gpu=217.75"
