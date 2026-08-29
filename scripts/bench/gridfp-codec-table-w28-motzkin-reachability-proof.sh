#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

DEVICE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_reduced_production_device.cuh"
OWNER_PLAN="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_reduced_production_owner_component_plan_device.cuh"
GROUP="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_reduced_production_group_context_device.cuh"
W28_PROBE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_codec_table_w28_rank_microprobe.cu"
for f in "$DEVICE" "$OWNER_PLAN" "$GROUP" "$W28_PROBE"; do
  [[ -f "$f" ]] || { echo "missing W28 Motzkin reachability input: $f" >&2; exit 2; }
done

# Motzkin remains a real production table, but its only primitive accessor is
# the generic motzkin_unrank_device path in production_device.cuh.
grep -Fq '__constant__ Rank64 RP_MOTZKIN[RP_MAX_W + 1][RP_MAX_W + 2];' "$DEVICE"
grep -Fq '__device__ __forceinline__ MateID motzkin_unrank_device' "$DEVICE"
grep -Fq 'RP_MOTZKIN[rem][h]' "$DEVICE"

# The W28 measurement/production owner path deliberately bypasses generic
# Motzkin label unranking: planned owner unrank reconstructs support+primitive
# directly, then grouped context/rank handles the state index.
grep -Fq 'owner_component_label_unrank_planned_device(' "$W28_PROBE"
grep -Fq 'owner_component_label_unrank_planned_device(' "$OWNER_PLAN"
grep -Fq 'grouped_component_context_device(' "$W28_PROBE"
grep -Fq 'grouped_rank_in_component_device(' "$W28_PROBE"
for f in "$OWNER_PLAN" "$GROUP" "$W28_PROBE"; do
  if grep -Eq 'RP_MOTZKIN\[|motzkin_unrank_device\(' "$f"; then
    echo "W28 planned rank path acquired a Motzkin dependency: $f" >&2
    grep -En 'RP_MOTZKIN\[|motzkin_unrank_device\(' "$f" >&2 || true
    exit 3
  fi
done

echo "gridfp-codec-table-w28-motzkin-reachability-proof OK w28_rank_motzkin_reads=0 motzkin_physical_priority=footprint_only exact=1"
