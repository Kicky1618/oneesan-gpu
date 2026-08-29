#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

DEVICE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_reduced_production_device.cuh"
PHYSICAL="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_reduced_production_codec_physical_device.cuh"
CHOOSE_PROOF="$ONEESAN_ROOT/scripts/bench/gridfp-choose-sym-u32-table-proof.sh"
PRIMITIVE_PROOF="$ONEESAN_ROOT/scripts/bench/gridfp-primitive-sym-u32-table-proof.sh"
for f in "$DEVICE" "$PHYSICAL" "$CHOOSE_PROOF" "$PRIMITIVE_PROOF"; do
  [[ -f "$f" ]] || { echo "missing physical replacement proof input: $f" >&2; exit 2; }
done

# Value exactness is independently proved against generated DP tables.
bash "$CHOOSE_PROOF" >/dev/null
bash "$PRIMITIVE_PROOF" >/dev/null

# The production header owns no codec declaration directly anymore; one
# physical-layout header is the single declaration/redirect point.
grep -Fq '#include "gridfp_reduced_production_codec_physical_device.cuh"' "$DEVICE"
if grep -Eq '^__constant__ Rank64 RP_(CHOOSE|PRIMITIVE)' "$DEVICE"; then
  echo "legacy codec constant declaration escaped physical-layout header" >&2
  exit 3
fi

# Default mode must preserve the exact old u64 constant layout.
grep -Fq '#define RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE 0' "$PHYSICAL"
grep -Fq '#define RP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE 0' "$PHYSICAL"
grep -Fq '__constant__ Rank64 RP_CHOOSE[RP_MAX_W + 1][RP_MAX_W + 1];' "$PHYSICAL"
grep -Fq '__constant__ Rank64 RP_PRIMITIVE[RP_MAX_W + 1][RP_MAX_W + 2];' "$PHYSICAL"

# In physical modes, legacy host upload targets are plain device-global sinks,
# not constant-memory arrays. Device compilation alone sees the proxy macro.
choose_sink_count="$(grep -Fc '__device__ Rank64 RP_CHOOSE[RP_MAX_W + 1][RP_MAX_W + 1];' "$PHYSICAL")"
primitive_sink_count="$(grep -Fc '__device__ Rank64 RP_PRIMITIVE[RP_MAX_W + 1][RP_MAX_W + 2];' "$PHYSICAL")"
[[ "$choose_sink_count" == 3 ]] || { echo "expected 3 choose upload-sink branches, got $choose_sink_count" >&2; exit 4; }
[[ "$primitive_sink_count" == 2 ]] || { echo "expected 2 primitive upload-sink branches, got $primitive_sink_count" >&2; exit 5; }
grep -Fq '#if defined(__CUDA_ARCH__)' "$PHYSICAL"
grep -Fq '#define RP_CHOOSE CodecPhysicalChooseProxy{}' "$PHYSICAL"
grep -Fq '#define RP_PRIMITIVE CodecPhysicalPrimitiveProxy{}' "$PHYSICAL"
grep -Fq 'physical choose layout cannot be combined with an RP_CHOOSE preinclude remap' "$PHYSICAL"
grep -Fq 'physical primitive layout cannot be combined with an RP_PRIMITIVE preinclude remap' "$PHYSICAL"

python3 - <<'PY'
choose={0:6728,1:900,2:1740,3:3364}
primitive={0:6960,1:900,2:3480}
layouts={
  0:(0,0,'baseline'),
  1:(1,1,'max_compact'),
  2:(2,1,'tri_choose_compact_primitive'),
  3:(3,1,'full_choose_compact_primitive'),
  4:(1,2,'sym_choose_full_primitive'),
  5:(3,2,'full_shape_both'),
}
expected={0:13688,1:1800,2:2640,3:4264,4:4380,5:6844}
for mode,(c,p,name) in layouts.items():
    total=choose[c]+primitive[p]
    assert total==expected[mode], (mode,total,expected[mode])
    print(f'physical_layout_mode{mode}_name={name} choose_mode={c} primitive_mode={p} constant_bytes={total} saved_bytes={13688-total}')
print('physical_layout_upload_sink_bytes_when_both_replaced=13688')
print('physical_layout_upload_sink_memory_space=global')
print('physical_layout_device_reads_legacy_sink=0')
print('physical_layout_exact=1')
PY

echo "gridfp-codec-table-physical-replacement-proof OK default_legacy_constant=1 physical_old_constant_removed=1 exact=1"
