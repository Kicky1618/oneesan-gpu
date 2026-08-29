#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

if len(sys.argv) != 3:
    raise SystemExit('usage: gen-b300-mainrec-hybrid8-next-self-prefetch.py INPUT.cu OUTPUT.cu')

src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
s = src.read_text()

for req in (
    'main_pull_kernel_ilp8_hybrid',
    'base+=Code(8)*grid',
    'const Count self7=',
    'const uint64_t mod=D_MOD;',
):
    if req not in s:
        raise SystemExit(f'hybrid8 next-self prefetch requires artifact: {req}')

helper_name = 'b300_mainrec_hybrid8_prefetch_next_self_l2'
if helper_name in s:
    raise SystemExit('source already contains hybrid8 next-self prefetch')

marker = '\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s:
    raise SystemExit('rank_full marker not found')

helper = r'''

__device__ __forceinline__ void b300_mainrec_hybrid8_prefetch_next_self_l2(const Count* p,bool valid){
#if __CUDA_ARCH__ >= 700
    if(valid){
        const unsigned long long a=reinterpret_cast<unsigned long long>(p);
        asm volatile("prefetch.global.L2 [%0];" :: "l"(a));
    }
#else
    (void)p;(void)valid;
#endif
}
'''
s = s.replace(marker, helper + marker, 1)

anchor = '        const Count self7=v7?in[i7]:Count(0);\n'
if s.count(anchor) != 1:
    raise SystemExit(f'hybrid8 self7 anchor expected one match got {s.count(anchor)}')

lines = [anchor.rstrip('\n'), '        const Code next_base=base+Code(8)*grid;']
for k in range(8):
    lines.append(f'        const Code ni{k}=next_base+Code({k})*grid;')
    lines.append(
        f'        {helper_name}(ni{k}<n?in+ni{k}:in,ni{k}<n);'
    )
insert = '\n'.join(lines) + '\n'
s = s.replace(anchor, insert, 1)

pref = s.find(f'{helper_name}(ni7<n?in+ni7:in,ni7<n);')
reduce_anchor = s.find('        const uint64_t mod=D_MOD;', pref)
if pref < 0 or reduce_anchor < 0 or pref >= reduce_anchor:
    raise SystemExit('hybrid8 next-self prefetch must precede current-iteration reduction')

for req in (
    'prefetch.global.L2',
    'const Code next_base=base+Code(8)*grid',
    f'{helper_name}(ni0<n?in+ni0:in,ni0<n)',
    f'{helper_name}(ni7<n?in+ni7:in,ni7<n)',
):
    if req not in s:
        raise SystemExit(f'missing hybrid8 next-self prefetch artifact: {req}')

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(s)
print(
    f'generated {out} from {src}: '
    'b300_mainrec_hybrid8_next_self_prefetch=1 '
    'next_iteration_self_prefetches_per_thread=8 cache=L2 '
    'prefetch_before_current_reduction=1 coalesced_per_k=1 '
    'semantics_unchanged=1 extra_shared_bytes=0'
)
