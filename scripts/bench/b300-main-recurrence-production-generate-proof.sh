#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/../lib/common.sh"

N="${N:-27}"
[[ "$N" == 27 ]] || { echo "production composition proof currently targets n=27" >&2; exit 2; }
TMP="${TMP:-$ONEESAN_BUILD_DIR/b300_main_recurrence_generate_proof}"
mkdir -p "$TMP"
BASE="$ONEESAN_ROOT/src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_forced2window_opt_batch.cu"
S1="$TMP/01_main_mate.cu";S2="$TMP/02_main_pull.cu";S3="$TMP/03_block_pull.cu";S4="$TMP/04_block_mate.cu";S5="$TMP/05_low_cache.cu";S6="$TMP/06_low_chunk.cu";S7="$TMP/07_low_block.cu";S8="$TMP/08_ilp2.cu";S9="$TMP/09_main_recurrence_raw.cu";S10="$TMP/10_main_recurrence_gated.cu"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-mate-cache.py" "$BASE" "$S1"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull.py" "$S1" "$S2"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-pull.py" "$S2" "$S3"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-block-mate-cache.py" "$S3" "$S4"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-low-window-drop-cache.py" "$S4" "$S5"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-low-drop-chunk.py" "$S5" "$S6"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-low-block-cache.py" "$S6" "$S7"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-pull-ilp2.py" "$S7" "$S8"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-main-recurrence.py" "$S8" "$S9"
python3 "$ONEESAN_ROOT/scripts/build/gen-b300-high-recurrence-fixed-gate.py" "$S9" "$S10"

python3 - "$S10" <<'PY'
from pathlib import Path
import sys
p=Path(sys.argv[1]);s=p.read_text()
def need(x):
    if x not in s: raise SystemExit(f'missing unified production artifact: {x}')
for x in (
    'b300_pack_main_transition_cache',
    'b300_low_state_advance(m0,p)',
    'b300_low_state_advance(m1,p)',
    'b300_high_state_advance(m0,p)',
    'b300_high_state_advance(m1,p)',
    'b300_high_state_drop_rank',
    'b300_low_cached_drop_rank(i,m,p)',
    'b300_init_low_main_state_kernel',
    'main_pull_kernel_ilp2',
    'MateID* __restrict__ mates',
    '__popc(D_MAIN_FIXED)>=7',
): need(x)
helper=s.find('__device__ __forceinline__ MateID b300_pack_main_transition_cache')
gather=s.find('__global__ void gather_main_kernel')
if helper<0 or gather<0 or helper>=gather:raise SystemExit(f'helper ordering invalid helper={helper} gather={gather}')
if s.count('b300_pack_main_transition_cache(m)') != 2:raise SystemExit('expected exactly two production unified packing calls')
if s.count('__popc(D_MAIN_FIXED)>=7') != 1:raise SystemExit('expected exactly one high recurrence fixed-bit gate')
start=s.find('if(p>1){');end=s.find('}else{',start)
if start<0 or end<0:raise SystemExit('p>1 process block not found')
hot=s[start:end]
if 'main_pull_kernel_ilp2' not in hot:raise SystemExit('p>1 hot path does not launch ILP2')
if 'main_to_block_kernel' in hot:raise SystemExit('stale main_to_block scatter remains')
prep=s.find('__device__ __forceinline__ void b300_main_pull_prepare');prep_end=s.find('__global__ void main_pull_kernel_ilp2',prep);q=s[prep:prep_end]
if 'high?b300_high_state_drop_rank' not in q:raise SystemExit('high recurrent drop not selected')
if 'low?b300_low_cached_drop_rank' not in q:raise SystemExit('low recurrent drop not selected')
print('b300-main-recurrence-production-generate-proof OK helper_before_gather=1 packing_calls=2 ilp2=1 low_recurrence=1 high_recurrence=1 high_min_fixed=7 fixed_lt7_fallback=1 stale_main_to_block=0 exact_structure=1')
PY

echo "b300-main-recurrence-production-generate-proof OK source=$S10" >&2
