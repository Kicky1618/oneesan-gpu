#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
OUTDIR="${OUTDIR:-$ONEESAN_BUILD_DIR/b300_vmm_main_pull_generate_proof}"
MATE="$OUTDIR/main_mate.cu"
PULL="$OUTDIR/main_pull.cu"
VMM="$OUTDIR/vmm.cu"
mkdir -p "$OUTDIR"

bash "$ONEESAN_ROOT/scripts/bench/b300-main-pull-operator-proof.sh" >/dev/null
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-mate-cache.py" "$SRC" "$MATE"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull.py" "$MATE" "$PULL"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py" "$PULL" "$VMM"
python3 "$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py" "$VMM" "$VMM"

for required in \
  'template<bool CACHED_MATE>' \
  '__global__ void main_pull_kernel' \
  '__global__ void main_to_block_kernel' \
  'main_pull_kernel<true>' \
  'main_to_block_kernel<true>' \
  'materialize_main_mates_kernel' \
  'if(nblock&&mget(m,p)==N)' \
  'if(j<nblock)acc+=in_block[j]' \
  'return D_MAIN_VBASE[g]' \
  'return D_BLOCK_VBASE[g]' \
  'Count*peer=(BLOCK?D_BLOCK_VBASE:D_MAIN_VBASE)+x.remote' \
  'b300_vmm::ContiguousStorage main_store,block_store' \
  'backend=gridfp-b300-hbm32-fullmate-dropN-vmm'; do
  grep -Fq "$required" "$VMM" || {
    echo "VMM main-pull composition missing: $required" >&2
    exit 3
  }
done

# p>1 must not execute the old main identity-copy / main scatter / blocked->main
# kernel. These operations are retained only in the p==1 fallback branch.
python3 - "$VMM" <<'PY'
from pathlib import Path
import sys
s=Path(sys.argv[1]).read_text()
a=s.find('if(p>1){')
b=s.find('}else{',a)
if a<0 or b<0: raise SystemExit('missing p>1 pull branch')
q=s[a:b]
for bad in ('cudaMemcpyAsync(nxt,cur','main_group_kernel<','blocked_group_kernel<<<'):
    if bad in q: raise SystemExit(f'p>1 pull branch retained stale operation: {bad}')
for good in ('main_pull_kernel<true>','main_to_block_kernel<true>','clear next D pull'):
    if good not in q: raise SystemExit(f'p>1 pull branch missing: {good}')
# Pulling main values must itself stay free of modular atomic updates.
a=s.find('__global__ void main_pull_kernel')
b=s.find('template<bool CACHED_MATE>\n__global__ void main_to_block_kernel',a)
body=s[a:b]
if 'atomic_add_mod' in body: raise SystemExit('main_pull_kernel contains atomic_add_mod')
if 'out_main[i]=Count(acc)' not in body: raise SystemExit('main_pull_kernel lost direct output store')
PY

for stale in B300_FAST_SHARD_ADDRESS8 ShardAddress8 shard_address8 D_MAIN_PTR D_BLOCK_PTR D_MAIN_CHUNK D_BLOCK_CHUNK D_NGPU D_MAIN_W D_BLOCK_W; do
  if grep -Fq "$stale" "$VMM"; then
    echo "VMM main-pull generated source retained stale shard artifact: $stale" >&2
    exit 4
  fi
done

echo "b300-vmm-main-pull-production-generate-proof OK mate_cache=1 main_pull=1 p_gt_1_main_atomic=0 p_gt_1_identity_copy=0 p_gt_1_blocked_to_main_kernel=0 block_rank_guard=1 vmm_direct_global=1 stale_shard_symbols=0 exact_operator_proof=1"
