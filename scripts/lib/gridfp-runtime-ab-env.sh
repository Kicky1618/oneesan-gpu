#!/usr/bin/env bash
# Canonical unrelated runtime stack for isolated two-row exact A/B experiments.
# Source this after scripts/lib/common.sh, then invoke builds with:
#   env "${GRIDFP_RUNTIME_AB_ENV[@]}" NVCC_PREPEND_FLAGS=... MODE=... bash ...
# Keep every knob here explicit so a future production-default change cannot
# silently alter the baseline of existing experiments.

GRIDFP_RUNTIME_AB_ENV=(
  RUNTIME_CACHE_EDGES=1
  RUNTIME_FAST_P32M5_MOD=1
  RUNTIME_POLL_GLOBAL_ERROR=0
  RUNTIME_PACK_SHARED_KEYS=1
  RUNTIME_FAST_DIV64=1
  RUNTIME_PRIMITIVE_RANK_SETBITS=1
  RUNTIME_MATERIALIZE_PRIMITIVE_SETBITS=1
  RUNTIME_SUPPORT_UNRANK_EARLY_EXIT=1
  RUNTIME_BROADWORD_SUPPORT=1
  RUNTIME_OWNER_FROM_BOUNDARIES=1
  RUNTIME_OWNER_RECIPROCAL=1
  RUNTIME_OWNER_FIXED54=0
  RUNTIME_OWNER_FIXED52=1
  RUNTIME_OWNER_U32LIMB=0
  RUNTIME_OWNER_W28_NGPU8_DIRECT=0
  RUNTIME_OWNER_SUPPORT_BITPACK=1
  RUNTIME_OWNER_PREFIX_BINARY=1
  RUNTIME_OWNER_LOCAL_SECTOR_TABLE=1
  RUNTIME_OWNER_LOCAL_SECTOR_PARITY=1
  RUNTIME_OWNER_LOCAL_SECTOR_W28_TREE=0
  RUNTIME_TURN_LOCAL_SECTOR_TABLE=1
  RUNTIME_TURN_LOCAL_SECTOR_W28_TREE=0
  RUNTIME_TURN_DISCOVERY_NONN_SCAN=0
  RUNTIME_TURN_DIRECT_COMPRESS_INVERSE=1
  RUNTIME_SUPPORT_RANK_SETBITS=1
  RUNTIME_FUSE_PRIMITIVE_SUPPORT_RANK=1
  RUNTIME_DIRECT_BLOCKED_RANK=1
  RUNTIME_SECTOR_OFFSET_TABLE=1
  RUNTIME_CACHE_SECTOR_ROW_BASE=1
  RUNTIME_OUTER_GROUP_TABLE=1
  RUNTIME_FAST_OUTSIDE_COMPACT=1
  RUNTIME_FAST_ERASE_TWO_BITS=1
  RUNTIME_FAST_DISCOVERY_VALIDITY=1
  RUNTIME_DISCOVERY_ENDPOINT_SCAN=1
  RUNTIME_FAST_CLOSURE_NONN_SCAN=1
  RUNTIME_FIND_RECENT_FIRST=0
  RUNTIME_FIND_SIGNATURE_FILTER=0
  RUNTIME_FIND_INDEX_CACHE=0
  RUNTIME_FIND_INDEX_BUCKETS=64
  RUNTIME_FIND_INDEX_WAYS=1
  RUNTIME_FAST_MIRROR_MATE=1
  RUNTIME_FAST_INCLUDE_HORIZONTAL_REVERSE=1
  RUNTIME_FAST_BLOCKED_EXCLUDE_REVERSE=1
  RUNTIME_DIRECT_REVERSE_SMALL_STEP=1
)

GRIDFP_RUNTIME_AB_BUILD_TOKENS=(
  runtime_cache_edges=1
  runtime_fast_p32m5_mod=1
  runtime_poll_global_error=0
  runtime_pack_shared_keys=1
  runtime_fast_div64=1
  runtime_primitive_rank_setbits=1
  runtime_materialize_primitive_setbits=1
  runtime_support_unrank_early_exit=1
  runtime_broadword_support=1
  runtime_owner_from_boundaries=1
  runtime_owner_fixed54=0
  runtime_owner_fixed52=1
  runtime_owner_u32limb=0
  runtime_owner_w28_ngpu8_direct=0
  runtime_owner_support_bitpack=1
  runtime_owner_prefix_binary=1
  runtime_owner_local_sector_table=1
  runtime_owner_local_sector_parity=1
  runtime_owner_local_sector_w28_tree=0
  runtime_turn_local_sector_table=1
  runtime_turn_local_sector_w28_tree=0
  runtime_turn_discovery_nonn_scan=0
  runtime_turn_direct_compress_inverse=1
  runtime_support_rank_setbits=1
  runtime_fuse_primitive_support_rank=1
  runtime_direct_blocked_rank=1
  runtime_sector_offset_table=1
  runtime_cache_sector_row_base=1
  runtime_outer_group_table=1
  runtime_fast_outside_compact=1
  runtime_fast_erase_two_bits=1
  runtime_fast_discovery_validity=1
  runtime_discovery_endpoint_scan=1
  runtime_fast_closure_nonn_scan=1
  runtime_find_recent_first=0
  runtime_find_signature_filter=0
  runtime_find_index_cache=0
  runtime_find_index_storage_bytes=64
  runtime_find_index_ways=1
  runtime_fast_mirror_mate=1
  runtime_fast_include_horizontal_reverse=1
  runtime_fast_blocked_exclude_reverse=1
  runtime_direct_reverse_small_step=1
)

gridfp_runtime_ab_assert_build() {
  local path="$1"
  [[ -f "$path" ]] || { echo "missing build log: $path" >&2; return 2; }
  local token
  for token in "${GRIDFP_RUNTIME_AB_BUILD_TOKENS[@]}"; do
    grep -Fq "$token" "$path" || {
      echo "isolated runtime A/B build drift: missing $token in $path" >&2
      return 3
    }
  done
}
