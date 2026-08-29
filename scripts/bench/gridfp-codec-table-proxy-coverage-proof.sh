#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

DEVICE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_reduced_production_device.cuh"
GROUP="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_reduced_production_group_context_device.cuh"
CHOOSE_PRE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_choose_sym_u32_preinclude.cuh"
PRIMITIVE_PRE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_primitive_sym_u32_preinclude.cuh"
CODEC_PRE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_runtime_codec_tables_sym_u32_preinclude.cuh"
PHYSICAL="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_reduced_production_codec_physical_device.cuh"
ABENV="$ONEESAN_ROOT/scripts/lib/gridfp-runtime-ab-env.sh"
for f in "$DEVICE" "$GROUP" "$CHOOSE_PRE" "$PRIMITIVE_PRE" "$CODEC_PRE" "$PHYSICAL" "$ABENV"; do
  [[ -f "$f" ]] || { echo "missing proxy coverage input: $f" >&2; exit 2; }
done

# A force-included proxy intentionally includes device.cuh once under renamed
# canonical symbols. Any direct table access defined inside that header is
# therefore frozen to the original table; accesses compiled in later headers
# see the proxy macro. Table declarations now live in the physical-layout
# header, so only actual reads remain in device.cuh.
choose_tokens="$(grep -o 'RP_CHOOSE\[' "$DEVICE" | wc -l | tr -d ' ')"
primitive_tokens="$(grep -o 'RP_PRIMITIVE\[' "$DEVICE" | wc -l | tr -d ' ')"
[[ "$choose_tokens" == 2 ]] || {
  echo "unexpected RP_CHOOSE read count in device.cuh: $choose_tokens (expected 2 legacy rank reads)" >&2
  grep -n 'RP_CHOOSE\[' "$DEVICE" >&2 || true
  exit 3
}
[[ "$primitive_tokens" == 1 ]] || {
  echo "unexpected RP_PRIMITIVE read count in device.cuh: $primitive_tokens (expected 1 legacy rank read)" >&2
  grep -n 'RP_PRIMITIVE\[' "$DEVICE" >&2 || true
  exit 4
}
grep -Fq '#include "gridfp_reduced_production_codec_physical_device.cuh"' "$DEVICE"
grep -Fq 'rank += RP_CHOOSE[rem][left];' "$DEVICE"
grep -Fq 'rank += RP_PRIMITIVE[rem][h - 1];' "$DEVICE"
grep -Fq 'const Rank64 pr = primitive_rank_device(k.mate, len, occupied);' "$DEVICE"

# Default physical modes are zero, so the old preinclude experiments still
# rename the canonical u64 declarations exactly as before.
grep -Fq '#define RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE 0' "$PHYSICAL"
grep -Fq '#define RP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE 0' "$PHYSICAL"

# Current exact-runtime baseline must select the later-header fused rank path,
# which is compiled after the proxy is defined and therefore reads candidate
# RP_CHOOSE / RP_PRIMITIVE storage.
for token in \
  'RUNTIME_PRIMITIVE_RANK_SETBITS=1' \
  'RUNTIME_SUPPORT_RANK_SETBITS=1' \
  'RUNTIME_FUSE_PRIMITIVE_SUPPORT_RANK=1' \
  'RUNTIME_DIRECT_BLOCKED_RANK=1'; do
  grep -Fq "$token" "$ABENV" || {
    echo "proxy coverage baseline no longer guarantees fused rank path: missing $token" >&2
    exit 5
  }
done

grep -Fq 'runtime_primitive_local_ranks_fused_device(' "$GROUP"
grep -Fq 'runtime_primitive_local_ranks_blocked_compressed_device(' "$GROUP"
grep -Fq 'out.primitive += RP_PRIMITIVE[rem][h - 1];' "$GROUP"
grep -Fq 'out.support += choose_device(' "$GROUP"
grep -Fq 'const Rank64 pc = RP_PRIMITIVE[occupied][1];' "$GROUP"

# Verify each preinclude proxy has the intended include-before-redirect structure.
grep -Fq '#define RP_CHOOSE RP_CHOOSE_PREINCLUDE_ORIG' "$CHOOSE_PRE"
grep -Fq '#define RP_CHOOSE RuntimeChooseU32Proxy{}' "$CHOOSE_PRE"
grep -Fq '#define RP_PRIMITIVE RP_PRIMITIVE_PREINCLUDE_ORIG' "$PRIMITIVE_PRE"
grep -Fq '#define RP_PRIMITIVE RuntimePrimitiveU32Proxy{}' "$PRIMITIVE_PRE"
grep -Fq '#define RP_CHOOSE RP_CHOOSE_CODEC_TABLES_ORIG' "$CODEC_PRE"
grep -Fq '#define RP_CHOOSE RuntimeCodecChooseU32Proxy{}' "$CODEC_PRE"
grep -Fq '#define RP_PRIMITIVE RP_PRIMITIVE_CODEC_TABLES_ORIG' "$CODEC_PRE"
grep -Fq '#define RP_PRIMITIVE RuntimeCodecPrimitiveU32Proxy{}' "$CODEC_PRE"

# factor_rank_device is the transitive user of the frozen legacy rank helpers.
# It must not become part of the current grouped runtime rank implementation
# without this audit being revisited.
if grep -Fq 'factor_rank_device(' "$GROUP"; then
  echo "grouped runtime rank path started using frozen factor_rank_device" >&2
  exit 6
fi

echo "gridfp-codec-table-proxy-coverage-proof OK" \
  "device_choose_reads=$choose_tokens" \
  "device_primitive_reads=$primitive_tokens" \
  "frozen_direct_choose_reads=2" \
  "frozen_direct_primitive_reads=1" \
  "frozen_rank_helpers=primitive_rank_device,support_rank_main_device,support_rank_block_device,factor_rank_device" \
  "runtime_fused_rank_proxy_covered=1" \
  "preinclude_proxy_physical_replacement=0 exact=1"
