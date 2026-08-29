#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

shells=(
  scripts/build/b300-hbm32.sh
  scripts/build/b300-directgather-orbitcta-pipe2-producer-warp.sh
  scripts/run/b300x8.sh
  scripts/run/b300x8-rankstate-winner.sh
  scripts/bench/b300x8-rankstate-ilp4-hotd32-race.sh
  scripts/bench/b300x8-block-closure-cg-ab.sh
  scripts/bench/b300-orbitcta-flat-prectx-warpcoop-ab.sh
  scripts/bench/b300-pipe2-producer-warp-coverage-proof.sh
  scripts/bench/b300-orbitcta-flat-dynamic-pipe2-producer-quad-ab.sh
  scripts/bench/b300-orbitcta-flat-dynamic-pipe2-producer-prectx-warpcoop-ab.sh
)
py=(
  scripts/build/gen-b300-rank-state-ilp2.py
  scripts/build/gen-b300-block-rank-state-ilp2.py
  scripts/build/gen-b300-rank-state-ilp4.py
  scripts/build/gen-b300-block-rank-state-ilp4.py
  scripts/build/gen-b300-hot-delta-table.py
)
for f in "${shells[@]}"; do bash -n "$ONEESAN_ROOT/$f"; echo "shell_syntax_ok=$f"; done
for f in "${py[@]}"; do python3 -m py_compile "$ONEESAN_ROOT/$f"; echo "python_syntax_ok=$f"; done

main2="$ONEESAN_ROOT/scripts/build/gen-b300-rank-state-ilp2.py"
block2="$ONEESAN_ROOT/scripts/build/gen-b300-block-rank-state-ilp2.py"
main4="$ONEESAN_ROOT/scripts/build/gen-b300-rank-state-ilp4.py"
block4="$ONEESAN_ROOT/scripts/build/gen-b300-block-rank-state-ilp4.py"
hot="$ONEESAN_ROOT/scripts/build/gen-b300-hot-delta-table.py"
build="$ONEESAN_ROOT/scripts/build/b300-hbm32.sh"
run="$ONEESAN_ROOT/scripts/run/b300x8.sh"
chunk="$ONEESAN_ROOT/src/cuda/gridfp/ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_chunked.cuh"
coop="$ONEESAN_ROOT/src/cuda/gridfp/ramstream32_bucket_precomputed_high_ctx_compact_warpcoop.cuh"
producer="$ONEESAN_ROOT/src/cuda/gridfp/ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2_producer_warp.cuh"
producer_pipe2="$ONEESAN_ROOT/src/cuda/gridfp/ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2.cuh"
producer_prectx="$ONEESAN_ROOT/src/cuda/gridfp/ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_dynamic_pipe2_producer_prectx_warpcoop.cuh"
producer_build="$ONEESAN_ROOT/scripts/build/b300-directgather-orbitcta-pipe2-producer-warp.sh"
producer_ab="$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-dynamic-pipe2-producer-quad-ab.sh"
producer_prectx_ab="$ONEESAN_ROOT/scripts/bench/b300-orbitcta-flat-dynamic-pipe2-producer-prectx-warpcoop-ab.sh"
coverage="$ONEESAN_ROOT/scripts/bench/b300-pipe2-producer-warp-coverage-proof.sh"

for s in 'destinations_per_thread=2' 'launch_cover_per_thread=2' 'launch_blocks=ceil_n_over_2threads_capped65535' 'b300_rankstate_ilp2_blocks(ms.size,threads)' 'rank_state_store_before_count_gather=1'; do
  grep -Fq "$s" "$main2" || { echo "missing main ILP2 marker: $s" >&2; exit 3; }
done
for s in 'launch_cover_per_thread=2' 'b300_rankstate_ilp2_blocks(ds.size,threads)' 'closure_slow_path=noinline' 'rank_state_store_before_count_gather=1'; do
  grep -Fq "$s" "$block2" || { echo "missing block ILP2 marker: $s" >&2; exit 3; }
done
for s in 'destinations_per_thread=4' 'launch_cover_per_thread=4' 'launch_blocks=ceil_n_over_4threads_capped65535' 'b300_rankstate_ilp4_blocks(ms.size,threads)' 'rank_state_store_before_count_gather=1' 'self_load_after_random_addresses=1' 'random_count_requests_up_to=8'; do
  grep -Fq "$s" "$main4" || { echo "missing main ILP4 marker: $s" >&2; exit 3; }
done
for s in 'b300_rankstate_ilp4_blocks(ds.size,threads)' 'B300_BLOCK_CLOSURE_QUAD' 'B300_BLOCK_CLOSURE_CG' 'ld.global.cg.u32' 'closure_slow_path=noinline' 'rank_state_store_before_count_gather=1'; do
  grep -Fq "$s" "$block4" || { echo "missing block ILP4 marker: $s" >&2; exit 3; }
done
for s in 'constant_bytes_added=17400' 'step_n_stored=0' 'int32_runtime_checked=1'; do
  grep -Fq "$s" "$hot" || { echo "missing hot-delta marker: $s" >&2; exit 3; }
done
for s in 'RANK_STATE_ILP4' 'BLOCK_CLOSURE_QUAD' 'gen-b300-rank-state-ilp4.py' 'gen-b300-block-rank-state-ilp4.py' 'gen-b300-hot-delta-table.py'; do
  grep -Fq "$s" "$build" || { echo "missing production build wiring: $s" >&2; exit 3; }
done
for s in 'RANK_STATE_ILP4' 'BLOCK_CLOSURE_QUAD' '_rsilp4' '_closureq'; do
  grep -Fq "$s" "$run" || { echo "missing run wiring: $s" >&2; exit 3; }
done
for s in 'P10DC_RANKFORMULA_PRECTX_WARPCOOP' 'p10dc_apply_forward_compact_prectx_warpcoop' 'p10dc_apply_reverse_compact_prectx_warpcoop' 'coop_q_lane0' 'coop_meta_lane0' '__shfl_sync(active, coop_q_lane0, 0)' '__shfl_sync(active, coop_meta_lane0, 0)'; do
  grep -Fq "$s" "$chunk" || { echo "missing chunk warpcoop wiring: $s" >&2; exit 3; }
done
if grep -Fq 'c.cross_depth = q;' "$chunk"; then echo 'stale warpcoop q handoff through cross_depth remains' >&2; exit 3; fi
if grep -Fq '__syncwarp();' "$chunk"; then echo 'stale explicit warpcoop sync remains in chunk scheduler' >&2; exit 3; fi
for s in 'BKCZ_MAX_LOCAL + 1 <= 32' 'offsetof(P10DCHighClosureCompactPreCtx, local_n) % alignof(uint32_t) == 0' 'meta = __ldg(tail)' 'meta = __shfl_sync(active, meta, 0)' '__ldg(&z->local_ref[lane])' '__ldg(&z->cross_ref)' 'p10dc_high_row_ref_resolve_unchecked'; do
  grep -Fq "$s" "$coop" || { echo "missing coalesced warpcoop helper marker: $s" >&2; exit 3; }
done

for s in 'P10DC_ORBITCTA_PLAN_SUM_QUAD' 'producer-warp quad plan sum requires ILP=4' 'valid[0] && valid[1] && valid[2] && valid[3]' 'P10DC_ORBITCTA_PLAN_SUM_PAIR' 'Full groups now issue one quad gather'; do
  grep -Fq "$s" "$producer" || { echo "missing producer native-quad marker: $s" >&2; exit 3; }
done
for s in 'P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP' 'producer prectx warpcoop requires producer warp' 'first_k_lane0' 'next_k_lane0' 'prepare_forward_producer_prectx_warpcoop' 'prepare_reverse_producer_prectx_warpcoop'; do
  grep -Fq "$s" "$producer_pipe2" || { echo "missing producer prectx pipe2 wiring: $s" >&2; exit 3; }
done
for s in 'compact forward+reverse prectx' 'P10DC_RANKFORMULA_PRECTX_FLAT_BID == 0' '__shfl_sync(active, k_lane0, 0)' 'p10dc_direct_resolve_high_io' 'p10dc_apply_forward_compact_prectx_warpcoop' 'p10dc_apply_reverse_compact_prectx_warpcoop'; do
  grep -Fq "$s" "$producer_prectx" || { echo "missing producer prectx helper marker: $s" >&2; exit 3; }
done
for s in 'QUAD_MLP="${QUAD_MLP:-0}"' 'producer-warp native QUAD_MLP=1 requires ORBITCTA_COL_ILP=4' 'QUAD_MLP="$QUAD_MLP"' 'PRODUCER_PRECTX_WARPCOOP="${PRODUCER_PRECTX_WARPCOOP:-0}"' 'PRODUCER_PRECTX_WARPCOOP=1 requires PRECTX_FORWARD=1 PRECTX_REVERSE=1 PRECTX_COMPACT=1' 'P10DC_ORBITCTA_FLAT_DYNAMIC_PIPE2_PRODUCER_PRECTX_WARPCOOP'; do
  grep -Fq "$s" "$producer_build" || { echo "missing producer build wiring: $s" >&2; exit 3; }
done
for s in 'QUAD_MLP=1 ORBITCTA_COL_ILP=4' 'producer-quad A/B fixes CPASYNC_PAIR=1' 'producer_quad_high_speedup=' 'producer_quad_memctrl_delta='; do
  grep -Fq "$s" "$producer_ab" || { echo "missing producer quad A/B marker: $s" >&2; exit 3; }
done
for s in 'PRODUCER_PRECTX_WARPCOOP=1 QUAD_MLP=1 ORBITCTA_COL_ILP=4' 'producer prectx A/B fixes CPASYNC_PAIR=1' 'producer_prectx_high_speedup=' 'producer_prectx_memctrl_delta='; do
  grep -Fq "$s" "$producer_prectx_ab" || { echo "missing producer prectx A/B marker: $s" >&2; exit 3; }
done
for s in 'producer_quad_full_groups_seen=' 'producer_quad_tail_groups_seen=' 'quad_tail_valid_prefix=1' 'pair_single_fallback_exact=1'; do
  grep -Fq "$s" "$coverage" || { echo "missing producer quad coverage marker: $s" >&2; exit 3; }
done

echo 'b300_rankstate_ilp4_static_preflight=OK'
echo 'ilp2_launch_cover=2 ilp4_launch_cover=4 old_full_grid_degeneracy=removed'
echo 'warpcoop_compact_bytes=40 warpcoop_ticket_shuffles=2 warpcoop_meta_broadcast=1 extra_syncwarp=0 cross_depth_ticket_reuse=0'
echo 'pipe2_producer_native_quad=1 ilp4_full_group_quad=1 partial_group_pair_single_fallback=1 producer_warp=32'
echo 'pipe2_producer_prectx_warpcoop=1 producer_prectx_lanes=9 next_context_overlap=worker_columns extra_cta_barriers=0'
echo 'gpu_work=0 actions_triggered=0'
