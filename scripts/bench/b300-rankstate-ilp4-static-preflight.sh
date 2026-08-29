#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

shells=(
  scripts/build/b300-hbm32.sh
  scripts/run/b300x8.sh
  scripts/bench/b300x8-rankstate-ilp4-hotd32-race.sh
  scripts/bench/b300x8-block-closure-cg-ab.sh
  scripts/bench/b300-orbitcta-flat-prectx-warpcoop-ab.sh
)
py=(
  scripts/build/gen-b300-rank-state-ilp4.py
  scripts/build/gen-b300-block-rank-state-ilp4.py
  scripts/build/gen-b300-hot-delta-table.py
)
for f in "${shells[@]}"; do
  bash -n "$ONEESAN_ROOT/$f"
  echo "shell_syntax_ok=$f"
done
for f in "${py[@]}"; do
  python3 -m py_compile "$ONEESAN_ROOT/$f"
  echo "python_syntax_ok=$f"
done

main4="$ONEESAN_ROOT/scripts/build/gen-b300-rank-state-ilp4.py"
block4="$ONEESAN_ROOT/scripts/build/gen-b300-block-rank-state-ilp4.py"
hot="$ONEESAN_ROOT/scripts/build/gen-b300-hot-delta-table.py"
build="$ONEESAN_ROOT/scripts/build/b300-hbm32.sh"
run="$ONEESAN_ROOT/scripts/run/b300x8.sh"
chunk="$ONEESAN_ROOT/src/cuda/gridfp/ramstream32_bucket_orbit_closure_pattern10_depthcode_orbitcta_flat_chunked.cuh"
coop="$ONEESAN_ROOT/src/cuda/gridfp/ramstream32_bucket_precomputed_high_ctx_compact_warpcoop.cuh"

for s in \
  'destinations_per_thread=4' \
  'rank_state_store_before_count_gather=1' \
  'self_load_after_random_addresses=1' \
  'random_count_requests_up_to=8'; do grep -Fq "$s" "$main4" || { echo "missing main ILP4 marker: $s" >&2; exit 3; }; done
for s in \
  'B300_BLOCK_CLOSURE_QUAD' \
  'B300_BLOCK_CLOSURE_CG' \
  'ld.global.cg.u32' \
  'closure_slow_path=noinline' \
  'rank_state_store_before_count_gather=1'; do grep -Fq "$s" "$block4" || { echo "missing block ILP4 marker: $s" >&2; exit 3; }; done
for s in \
  'constant_bytes_added=17400' \
  'step_n_stored=0' \
  'int32_runtime_checked=1'; do grep -Fq "$s" "$hot" || { echo "missing hot-delta marker: $s" >&2; exit 3; }; done
for s in \
  'RANK_STATE_ILP4' \
  'BLOCK_CLOSURE_QUAD' \
  'gen-b300-rank-state-ilp4.py' \
  'gen-b300-block-rank-state-ilp4.py' \
  'gen-b300-hot-delta-table.py'; do grep -Fq "$s" "$build" || { echo "missing production build wiring: $s" >&2; exit 3; }; done
for s in 'RANK_STATE_ILP4' 'BLOCK_CLOSURE_QUAD' '_rsilp4' '_closureq'; do grep -Fq "$s" "$run" || { echo "missing run wiring: $s" >&2; exit 3; }; done
for s in 'P10DC_RANKFORMULA_PRECTX_WARPCOOP' 'p10dc_apply_forward_compact_prectx_warpcoop' 'p10dc_apply_reverse_compact_prectx_warpcoop'; do grep -Fq "$s" "$chunk" || { echo "missing chunk warpcoop wiring: $s" >&2; exit 3; }; done
for s in 'BKCZ_MAX_LOCAL + 1 <= 32' 'p10dc_high_row_ref_resolve_unchecked'; do grep -Fq "$s" "$coop" || { echo "missing warpcoop helper marker: $s" >&2; exit 3; }; done

echo 'b300_rankstate_ilp4_static_preflight=OK'
echo 'gpu_work=0 actions_triggered=0'
