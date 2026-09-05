#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

SRC="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
GEN="$ONEESAN_ROOT/scripts/build/gen-b300-vmm-production.py"
PRUNE="$ONEESAN_ROOT/scripts/build/prune-b300-vmm-stale-shard-symbols.py"
OUT="${OUT:-$ONEESAN_BUILD_DIR/generated_b300_vmm_production_proof.cu}"

bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-contiguous-layout-proof.sh"
bash "$ONEESAN_ROOT/scripts/bench/b300-vmm-balanced-physical-layout-proof.sh"
python3 "$GEN" "$SRC" "$OUT"
python3 "$PRUNE" "$OUT" "$OUT"

grep -Fq '#include "b300_vmm_contiguous_storage.cuh"' "$OUT"
grep -Fq 'static_assert(sizeof(Count)==4,"B300 VMM authoritative storage requires 32-bit Count");' "$OUT"
grep -Fq '__constant__ Count* D_MAIN_VBASE;' "$OUT"
grep -Fq '__constant__ Count* D_BLOCK_VBASE;' "$OUT"
grep -Fq 'struct PeerInterval{Code remote,local,len;};' "$OUT"
grep -Fq 'static_assert(sizeof(PeerInterval)==24,"VMM PeerInterval must stay three 64-bit words");' "$OUT"
grep -Fq 'static std::vector<PeerInterval> make_peer_intervals(const GroupSpec&s,bool& use_interval)' "$OUT"
grep -Fq 'static PreparedGroup prepare_group(int W,const WindowPlan&wp,int g)' "$OUT"
grep -Fq 'tiled.push_back({x.global+off,x.local+off,take});' "$OUT"
grep -Fq 'Count global_load_main(Code g){return D_MAIN_VBASE[g];}' "$OUT"
grep -Fq 'Count global_load_block(Code g){return D_BLOCK_VBASE[g];}' "$OUT"
grep -Fq 'void global_store_main(Code g,Count v){D_MAIN_VBASE[g]=v;}' "$OUT"
grep -Fq 'void global_store_block(Code g,Count v){D_BLOCK_VBASE[g]=v;}' "$OUT"
grep -Fq 'void init(int d,Count mod)' "$OUT"
grep -Fq 'ctx[d].init(d,mod);' "$OUT"
grep -Fq 'VMM exposes one contiguous authoritative VA' "$OUT"
grep -Fq 'Count*peer=(BLOCK?D_BLOCK_VBASE:D_MAIN_VBASE)+x.remote;' "$OUT"
grep -Fq 'b300_vmm::ContiguousStorage main_store,block_store;' "$OUT"
grep -Fq 'block_store.create(blockN,ng,int(main_store.mapped_units%size_t(ng)),"auth block");' "$OUT"
grep -Fq 'main_store.granularity!=block_store.granularity' "$OUT"
grep -Fq 'vmm_phys_max-vmm_phys_min>main_store.granularity' "$OUT"
grep -Fq 'VMM32 combined: granularity_kib=' "$OUT"
grep -Fq 'cudaMemcpyToSymbol(D_MAIN_VBASE,&main_base,sizeof(main_base))' "$OUT"
grep -Fq 'cudaMemcpyToSymbol(D_BLOCK_VBASE,&block_base,sizeof(block_base))' "$OUT"
grep -Fq 'cudaMemcpy(main_base+ig,&one,sizeof(one),cudaMemcpyHostToDevice)' "$OUT"
grep -Fq 'cudaMemcpy(&ans,main_base+fg,sizeof(ans),cudaMemcpyDeviceToHost)' "$OUT"
grep -Fq 'prepare_group(W,pw.wp,g)' "$OUT"
grep -Fq 'main_store.destroy();block_store.destroy();' "$OUT"
grep -Fq 'backend=gridfp-b300-hbm32-fullmate-dropN-vmm n=' "$OUT"

for stale in B300_FAST_SHARD_ADDRESS8 ShardAddress8 shard_address8 D_MAIN_PTR D_BLOCK_PTR D_MAIN_CHUNK D_BLOCK_CHUNK D_NGPU D_MAIN_W D_BLOCK_W; do
  if grep -Fq "$stale" "$OUT"; then
    echo "generated VMM source still contains stale shard scaffolding/symbol $stale" >&2
    exit 3
  fi
done
for stale in 'Code mc=' 'bc=(blockN+ng-1)/ng' 'Count*mp[MAXGPU]' '*bp[MAXGPU]' 'std::vector<Code>ml' 'int io=int(ig/mc)' 'int fo=int(fg/mc)' 'prepare_group(W,pw.wp,g,mc,bc,ng)' 'make_peer_intervals(pg.ms,mc,ng'; do
  if grep -Fq "$stale" "$OUT"; then
    echo "generated VMM source still contains logical shard artifact $stale" >&2
    exit 4
  fi
done
if grep -Fq 'uint32_t owner,pad' "$OUT"; then
  echo "generated VMM PeerInterval still contains owner/pad" >&2
  exit 5
fi
if grep -Fq 'cudaMalloc(&mp[d]' "$OUT" || grep -Fq 'cudaMalloc(&bp[d]' "$OUT"; then
  echo "generated VMM source still cudaMallocs authoritative shards" >&2
  exit 6
fi
if grep -Fq 'cudaFree(mp[d])' "$OUT" || grep -Fq 'cudaFree(bp[d])' "$OUT"; then
  echo "generated VMM source still cudaFrees VMM logical views" >&2
  exit 7
fi
if grep -Fq 'Every shard boundary can split at most one globally ordered interval.' "$OUT" || grep -Fq 'int owner=int(g/chunk)' "$OUT"; then
  echo "generated VMM interval planner still performs logical shard splitting" >&2
  exit 8
fi

echo "b300-vmm-production-generate-proof OK count_bytes=4 direct_global_index=1 shard_free_interval_io=1 compact_interval_bytes=24 compact_interval_vs_old_pct=75 interval_host_owner_div=0 interval_device_ptr_index=0 legacy_shard_address_scaffolding=0 stale_shard_symbols=0 stale_shard_symbol_copies=0 stale_shard_init_args=0 stale_width_symbols=0 per_group_width_symbol_copies=0 logical_shard_chunks=0 logical_shard_views=0 host_owner_div=0 direct_init_answer=1 authoritative_cudaMalloc=0 authoritative_cudaFree=0 balanced_physical_rotation=1 combined_imbalance_le_one_granularity=1 runtime_physical_balance_guard=1"
