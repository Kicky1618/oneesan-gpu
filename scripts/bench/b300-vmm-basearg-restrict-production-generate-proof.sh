#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"
SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py"
PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py"
BASEARG="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg.py"
RESTRICT="$ONEESAN_ROOT/scripts/build/lower-b300-vmm-basearg-restrict.py"
OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_vmm_basearg_restrict_proof.cu}"

bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-basearg-production-generate-proof.sh"
python3 "$GEN" "$SRC" "$OUT"
python3 "$PRUNE" "$OUT" "$OUT"
python3 "$BASEARG" "$OUT" "$OUT"
python3 "$RESTRICT" "$OUT" "$OUT"

grep -Fq 'gather_main_kernel(Count* __restrict__ out,MateID* __restrict__ mates,Code n,const Count* __restrict__ auth)' "$OUT"
grep -Fq 'gather_block_kernel(Count* __restrict__ out,Code n,const Count* __restrict__ auth)' "$OUT"
grep -Fq 'scatter_main_kernel(const Count* __restrict__ in,Code n,Count* __restrict__ auth)' "$OUT"
grep -Fq 'scatter_block_kernel(const Count* __restrict__ in,Code n,Count* __restrict__ auth)' "$OUT"
grep -Fq 'interval_io_kernel(Count* __restrict__ buf,const PeerInterval* __restrict__ iv,size_t niv,Count* __restrict__ auth)' "$OUT"
for stale in 'gather_main_kernel(Count*out' 'gather_block_kernel(Count*out' 'scatter_main_kernel(const Count*in' 'scatter_block_kernel(const Count*in' 'interval_io_kernel(Count*buf'; do
  if grep -Fq "$stale" "$OUT"; then echo "restrict generated source still contains non-restrict signature $stale" >&2; exit 3; fi
done

echo "b300-vmm-basearg-restrict-production-generate-proof OK vmm_base_source=kernel_param restrict=1 alias_contract=scratch_interval_auth_disjoint interval_template_axes=1 compact_interval_bytes=24"
