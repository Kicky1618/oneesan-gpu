#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

ARCH="${ARCH:-native}"
N="${N:-21}"
MOD="${MOD:-4294967291}"
NGPU="${NGPU:-8}"
PROFILE="${PROFILE:-1}"
MICROBENCH="${MICROBENCH:-1}"
PTXAS="${PTXAS:-1}"
PREFIX="${PREFIX:-$ONEESAN_ROOT/work/b300_rankchunk32_direct3_guard_evidence_n${N}}"
PROFILE_LOG="${PROFILE_LOG:-${PREFIX}.profile.log}"
MICRO_LOG="${MICRO_LOG:-${PREFIX}.micro.log}"
PTXAS_LOG="${PTXAS_LOG:-${PREFIX}.ptxas.log}"

for x in PROFILE MICROBENCH PTXAS; do
  v="${!x}"
  [[ "$v" == 0 || "$v" == 1 ]] || { echo "$x must be 0 or 1" >&2; exit 2; }
done
mkdir -p "$(dirname "$PREFIX")"

if [[ "$PROFILE" == 1 ]]; then
  echo "=== real-traffic rankmask profile ===" >&2
  ARCH="$ARCH" N="$N" MOD="$MOD" NGPU="$NGPU" \
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

python3 - "$PROFILE" "$MICROBENCH" "$PTXAS" "$PROFILE_LOG" "$MICRO_LOG" "$PTXAS_LOG" <<'PY'
import re
import sys

run_profile, run_micro, run_ptxas = map(int, sys.argv[1:4])
profile_path, micro_path, ptxas_path = sys.argv[4:7]

def fields(line):
    return dict(re.findall(r'([A-Za-z0-9_]+)=([^\s]+)', line))

print('rankchunk32-direct3-guard-evidence')

if run_profile:
    text = open(profile_path, encoding='utf-8', errors='replace').read()
    for src, dst in (
        ('actual_zero_frac','profile_zero_frac'),
        ('actual_nonzero_frac','profile_nonzero_frac'),
        ('actual_avg_popcount','profile_avg_popcount'),
        ('warp_all_zero_frac','profile_warp_all_zero_frac'),
        ('warp_all_nonzero_frac','profile_warp_all_nonzero_frac'),
        ('warp_mixed_frac','profile_warp_mixed_frac'),
        ('avg_active_lanes','profile_avg_active_lanes'),
    ):
        m = re.search(rf'\b{src}=([^\s]+)', text)
        if m:
            print(f'{dst}={m.group(1)}')

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

print('interpretation=guard_is_promising_when_shuffled_speedup_exceeds_1_and_register_pressure_does_not_regress;real_warp_all_zero_fraction_strengthens_the_case_while_high_mixed_fraction_weakens_it')
print('production_guard_enabled=0')
PY

echo "b300-depthcode-rankchunk32-direct3-guard-evidence OK profile=$PROFILE microbench=$MICROBENCH ptxas=$PTXAS prefix=$PREFIX" >&2
