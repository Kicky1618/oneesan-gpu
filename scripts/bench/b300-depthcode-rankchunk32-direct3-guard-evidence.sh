#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
EXACT_ARCH="${EXACT_ARCH:-sm_80}"
N="${N:-21}"
MOD="${MOD:-4294967291}"
NGPU="${NGPU:-8}"
HOST_EXACT="${HOST_EXACT:-1}"
PROFILE="${PROFILE:-1}"
PROFILE_LOG2="${PROFILE_LOG2:-8}"
MICROBENCH="${MICROBENCH:-1}"
PTXAS="${PTXAS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankchunk32_direct3_guard_evidence_n${N}_s${PROFILE_LOG2}}"
EXACT_LOG="${EXACT_LOG:-${PREFIX}.exact.log}"
PROFILE_LOG="${PROFILE_LOG:-${PREFIX}.profile.log}"
MICRO_LOG="${MICRO_LOG:-${PREFIX}.micro.log}"
PTXAS_LOG="${PTXAS_LOG:-${PREFIX}.ptxas.log}"

for x in HOST_EXACT PROFILE MICROBENCH PTXAS; do
  v="${!x}"
  [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
if ! [[ "$PROFILE_LOG2" =~ ^[0-9]+$ ]] || (( PROFILE_LOG2 > 16 )); then
  echo "PROFILE_LOG2 must be in [0,16]" >&2; exit 2
fi
mkdir -p "$(dirname "$PREFIX")"

if [[ "$HOST_EXACT" == 1 ]]; then
  echo "=== exact host production rankmask traffic ===" >&2
  ARCH="$EXACT_ARCH" N="$N" \
    bash "$ONEESAN_ROOT/scripts/bench/rankchunk32-exact-rankmask-traffic.sh" \
    | tee "$EXACT_LOG"
fi

if [[ "$PROFILE" == 1 ]]; then
  echo "=== real-traffic rankmask profile sample=1/$((1 << PROFILE_LOG2)) warps ===" >&2
  ARCH="$ARCH" N="$N" MOD="$MOD" NGPU="$NGPU" PROFILE_LOG2="$PROFILE_LOG2" \
    bash "$ONEESAN_ROOT/scripts/bench/b300-depthcode-rankchunk32-rankmask-profile.sh" \
    | tee "$PROFILE_LOG"
fi

if [[ "$MICROBENCH" == 1 ]]; then
  echo "=== ordered vs shuffled direct3-guard microbench ===" >&2
  ARCH="$ARCH" \
    bash "$ONEESAN_ROOT/scripts/bench/rankmask5-decode-order-ab.sh" \
    | tee "$MICRO_LOG"
fi

if [[ "$PTXAS" == 1 ]]; then
  echo "=== ptxas decoder resource comparison ===" >&2
  ARCH="$ARCH" LOG="${PTXAS_LOG}.raw" \
    bash "$ONEESAN_ROOT/scripts/bench/rankmask5-decode-ptxas.sh" \
    | tee "$PTXAS_LOG"
fi

python3 - "$HOST_EXACT" "$PROFILE" "$MICROBENCH" "$PTXAS" \
  "$EXACT_LOG" "$PROFILE_LOG" "$MICRO_LOG" "$PTXAS_LOG" <<'PY'
import re
import sys

run_exact, run_profile, run_micro, run_ptxas = map(int, sys.argv[1:5])
exact_path, profile_path, micro_path, ptxas_path = sys.argv[5:9]

def fields(line):
    return dict(re.findall(r'([A-Za-z0-9_]+)=([^\s]+)', line))

print('rankchunk32-direct3-guard-evidence')
exact = {}
profile = {}

if run_exact:
    text = open(exact_path, encoding='utf-8', errors='replace').read()
    line = next((x for x in text.splitlines()
                 if x.startswith('rankchunk32_exact_rankmask_traffic scope=one_residue ')), '')
    exact = fields(line)
    for key in ('resolved_calls','chunk_calls','m0','m1','m2','m3','m5','m7',
                'zero_frac','nonzero_frac','avg_popcount','chunks_per_resolved','halt_frac'):
        if key in exact:
            print(f'host_exact_{key}={exact[key]}')

if run_profile:
    text = open(profile_path, encoding='utf-8', errors='replace').read()
    for src, dst in (
        ('sample_log2','sample_log2'),
        ('sample_one_in','sample_one_in'),
        ('actual_zero_frac','zero_frac'),
        ('actual_nonzero_frac','nonzero_frac'),
        ('actual_avg_popcount','avg_popcount'),
        ('warp_all_zero_frac','warp_all_zero_frac'),
        ('warp_all_nonzero_frac','warp_all_nonzero_frac'),
        ('warp_mixed_frac','warp_mixed_frac'),
        ('avg_active_lanes','avg_active_lanes'),
    ):
        m = re.search(rf'\b{src}=([^\s]+)', text)
        if m:
            profile[dst] = m.group(1)
            print(f'profile_{dst}={m.group(1)}')

if exact and profile:
    if 'zero_frac' in exact and 'zero_frac' in profile:
        print(f'profile_zero_abs_error={abs(float(profile["zero_frac"])-float(exact["zero_frac"])):.9g}')
    if 'avg_popcount' in exact and 'avg_popcount' in profile:
        print(f'profile_avg_popcount_abs_error={abs(float(profile["avg_popcount"])-float(exact["avg_popcount"])):.9g}')

if run_micro:
    text = open(micro_path, encoding='utf-8', errors='replace').read()
    ordered = next((x for x in text.splitlines() if x.startswith('ordered ')), '')
    shuffled = next((x for x in text.splitlines() if x.startswith('shuffled ')), '')
    of = fields(ordered)
    sf = fields(shuffled)
    if 'direct3_to_guard_speedup' in of:
        print('micro_ordered_direct3_to_guard_speedup=' + of['direct3_to_guard_speedup'])
    if 'direct3_to_guard_speedup' in sf:
        print('micro_shuffled_direct3_to_guard_speedup=' + sf['direct3_to_guard_speedup'])
    m = re.search(r'\bguard_order_sensitivity_ratio=([^\s]+)', text)
    if m:
        print('micro_guard_order_sensitivity_ratio=' + m.group(1))

if run_ptxas:
    text = open(ptxas_path, encoding='utf-8', errors='replace').read()
    for key in ('direct3_guard_register_delta','direct3_guard_same_registers','all_modes_spill_free'):
        m = re.search(rf'\b{key}=([^\s]+)', text)
        if m:
            print(f'ptxas_{key}={m.group(1)}')

print('interpretation=host_exact_mask_frequencies_are_authoritative;sampled_gpu_profile_validates_runtime_mapping_and_supplies_warp_coherence;guard_is_promising_when_shuffled_speedup_exceeds_1_and_register_pressure_does_not_regress;high_all_zero_warp_fraction_strengthens_the_case_while_high_mixed_fraction_weakens_it')
print('production_guard_enabled=0')
PY

echo "b300-depthcode-rankchunk32-direct3-guard-evidence OK host_exact=$HOST_EXACT profile=$PROFILE profile_log2=$PROFILE_LOG2 microbench=$MICROBENCH ptxas=$PTXAS prefix=$PREFIX" >&2
