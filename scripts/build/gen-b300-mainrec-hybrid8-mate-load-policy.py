#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

if len(sys.argv) != 4:
    raise SystemExit('usage: gen-b300-mainrec-hybrid8-mate-load-policy.py INPUT.cu OUTPUT.cu POLICY')
src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
policy = sys.argv[3]
if policy not in ('cg', 'cs'):
    raise SystemExit('POLICY must be cg or cs')

s = src.read_text()
marker_name = 'b300_mainrec_hybrid8_mate_load_policy_'
if marker_name in s:
    raise SystemExit('source already contains hybrid8 mate-load policy')
for req in (
    'main_pull_kernel_ilp8_hybrid',
    'MateID* __restrict__ mates',
    'const MateID m0=mates[i0];',
    'const MateID m7=v7?mates[i7]:MateID(0);',
    'const Count self7=',
    'const uint64_t mod=D_MOD;',
):
    if req not in s:
        raise SystemExit(f'hybrid8 mate-load policy requires artifact: {req}')

# The helper must be declared before the ILP8 kernel definition because the
# generated kernel calls it directly. The block-count helper is the stable
# insertion point immediately before that definition.
kernel_marker = 'static inline int b300_main_recurrence_ilp8_hybrid_blocks(Code n,int threads)'
kernel_definition = '__global__ void main_pull_kernel_ilp8_hybrid('
if s.count(kernel_marker) != 1:
    raise SystemExit(f'hybrid8 kernel preamble expected one match got {s.count(kernel_marker)}')
if s.count(kernel_definition) != 1:
    raise SystemExit(f'hybrid8 kernel definition expected one match got {s.count(kernel_definition)}')
helper = f'b300_mainrec_hybrid8_mate_load_policy_{policy}'
intrinsic = '__ldcg' if policy == 'cg' else '__ldcs'
helper_src = f'''__device__ __forceinline__ MateID {helper}(const MateID* p){{
    return {intrinsic}(p);
}}

'''
s = s.replace(kernel_marker, helper_src + kernel_marker, 1)

for k in range(8):
    old = f'        const MateID m{k}=' + ('mates[i0];' if k == 0 else f'v{k}?mates[i{k}]:MateID(0);')
    new = f'        const MateID m{k}=' + (f'{helper}(mates+i0);' if k == 0 else f'v{k}?{helper}(mates+i{k}):MateID(0);')
    if s.count(old) != 1:
        raise SystemExit(f'mate load anchor m{k} expected one match got {s.count(old)}')
    s = s.replace(old, new, 1)

# Guard against accidentally changing the low-state ILP2 kernel or mate writes.
if 'const MateID m0=mates[i0];' in s:
    raise SystemExit('ILP8 mate load m0 was not rewritten')
for k in range(8):
    expected = f'{helper}(mates+i{k})'
    if s.count(expected) != 1:
        raise SystemExit(f'expected one policy load for lane {k}, got {s.count(expected)}')
for req in ('main_pull_kernel_ilp2', 'mates[i7]=b300_high_state_advance', 'const Count self7='):
    if req not in s:
        raise SystemExit(f'mate-load policy damaged required artifact: {req}')
helper_pos = s.find(f'__device__ __forceinline__ MateID {helper}(')
kernel_pos = s.find(kernel_definition)
if helper_pos < 0 or kernel_pos < 0 or helper_pos >= kernel_pos:
    raise SystemExit('mate-load helper must precede ILP8 kernel definition')

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(s)
print(
    f'generated {out} from {src}: '
    f'b300_mainrec_hybrid8_mate_load_policy=1 policy={policy} intrinsic={intrinsic} '
    'scope=ilp8_mate_reads_only lanes=8 helper_before_kernel=1 ilp2_unchanged=1 mate_writes_unchanged=1 semantics_unchanged=1'
)
