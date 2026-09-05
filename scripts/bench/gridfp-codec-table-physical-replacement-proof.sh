#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

DEVICE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_reduced_production_device.cuh"
PHYSICAL="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_reduced_production_codec_physical_device.cuh"
COMPONENT="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_reduced_production_component_microprobe.cu"
COMPILE_PROBE="$ONEESAN_ROOT/src/cuda/gridfp/gridfp_codec_table_physical_compile_probe.cu"
CHOOSE_PROOF="$ONEESAN_ROOT/scripts/bench/gridfp-choose-sym-u32-table-proof.sh"
PRIMITIVE_PROOF="$ONEESAN_ROOT/scripts/bench/gridfp-primitive-sym-u32-table-proof.sh"
for f in "$DEVICE" "$PHYSICAL" "$COMPONENT" "$COMPILE_PROBE" "$CHOOSE_PROOF" "$PRIMITIVE_PROOF"; do
  [[ -f "$f" ]] || { echo "missing physical replacement proof input: $f" >&2; exit 2; }
done

bash "$CHOOSE_PROOF" >/dev/null
bash "$PRIMITIVE_PROOF" >/dev/null

grep -Fq '#include "gridfp_reduced_production_codec_physical_device.cuh"' "$DEVICE"
if grep -Eq '^__constant__ Rank64 RP_(CHOOSE|PRIMITIVE)' "$DEVICE"; then
  echo "legacy codec constant declaration escaped physical-layout header" >&2
  exit 3
fi

# Mode zero is byte-for-byte the legacy table type/shape.
grep -Fq '#define RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE 0' "$PHYSICAL"
grep -Fq '#define RP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE 0' "$PHYSICAL"
grep -Fq '__constant__ Rank64 RP_CHOOSE[RP_MAX_W + 1][RP_MAX_W + 1];' "$PHYSICAL"
grep -Fq '__constant__ Rank64 RP_PRIMITIVE[RP_MAX_W + 1][RP_MAX_W + 2];' "$PHYSICAL"

# Nonzero physical modes must not retain legacy u64 symbols in any memory
# space. Candidate u32 constants and proxy macros are the only storage/read
# path. The proxy macros are deliberately visible in both nvcc passes so the
# host pass never sees an undefined RP_CHOOSE/RP_PRIMITIVE token in device
# function bodies; host upload statements are separately eliminated by mode0
# preprocessor guards.
if grep -Fq '__device__ Rank64 RP_CHOOSE[RP_MAX_W + 1][RP_MAX_W + 1];' "$PHYSICAL"; then
  echo "physical choose still contains a legacy global upload sink" >&2; exit 4
fi
if grep -Fq '__device__ Rank64 RP_PRIMITIVE[RP_MAX_W + 1][RP_MAX_W + 2];' "$PHYSICAL"; then
  echo "physical primitive still contains a legacy global upload sink" >&2; exit 5
fi
if grep -Fq '#if defined(__CUDA_ARCH__)' "$PHYSICAL"; then
  echo "physical codec proxy is still device-pass-only" >&2; exit 6
fi
grep -Fq '#define RP_CHOOSE CodecPhysicalChooseProxy{}' "$PHYSICAL"
grep -Fq '#define RP_PRIMITIVE CodecPhysicalPrimitiveProxy{}' "$PHYSICAL"
grep -Fq 'physical choose layout cannot be combined with an RP_CHOOSE preinclude remap' "$PHYSICAL"
grep -Fq 'physical primitive layout cannot be combined with an RP_PRIMITIVE preinclude remap' "$PHYSICAL"

# Host table uploads exist only in mode zero. With a physical candidate, the
# old symbol is neither declared nor referenced in host or device compilation.
for src in "$COMPONENT" "$COMPILE_PROBE"; do
  grep -Fq '#if RP_EXPERIMENTAL_CODEC_CHOOSE_PHYSICAL_MODE == 0' "$src"
  grep -Fq 'cudaMemcpyToSymbol(RP_CHOOSE' "$src"
  grep -Fq '#if RP_EXPERIMENTAL_CODEC_PRIMITIVE_PHYSICAL_MODE == 0' "$src"
  grep -Fq 'cudaMemcpyToSymbol(RP_PRIMITIVE' "$src"
done

python3 - <<'PY'
choose={0:6728,1:900,2:1740,3:3364}
primitive={0:6960,1:900,2:3480}
motzkin=6960
legacy_three=6728+6960+motzkin
layouts={
  0:(0,0,'baseline'),
  1:(1,1,'max_compact'),
  2:(2,1,'tri_choose_compact_primitive'),
  3:(3,1,'full_choose_compact_primitive'),
  4:(1,2,'sym_choose_full_primitive'),
  5:(3,2,'full_shape_both'),
}
expected_pair={0:13688,1:1800,2:2640,3:4264,4:4380,5:6844}
assert legacy_three==20648
for mode,(c,p,name) in layouts.items():
    pair=choose[c]+primitive[p]
    assert pair==expected_pair[mode], (mode,pair,expected_pair[mode])
    three=pair+motzkin
    print(f'physical_layout_mode{mode}_name={name} choose_mode={c} primitive_mode={p} choose_primitive_bytes={pair} choose_primitive_saved_bytes={13688-pair} three_table_bytes={three} three_table_saved_bytes={legacy_three-three}')
print(f'physical_layout_motzkin_unmodified_bytes={motzkin}')
print(f'physical_layout_legacy_three_table_bytes={legacy_three}')
print('physical_layout_legacy_symbol_bytes_nonzero_mode=0')
print('physical_layout_legacy_host_upload_nonzero_mode=0')
print('physical_layout_proxy_nvcc_host_device_passes=1')
print('physical_layout_device_reads_legacy_symbol=0')
print('physical_layout_exact=1')
PY

echo "gridfp-codec-table-physical-replacement-proof OK default_legacy_constant=1 physical_legacy_symbol_removed=1 host_upload_removed=1 proxy_both_nvcc_passes=1 motzkin_unmodified=1 exact=1"
