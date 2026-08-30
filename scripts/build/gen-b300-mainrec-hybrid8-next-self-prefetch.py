#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

if len(sys.argv) not in (3, 4, 5, 6, 7):
    raise SystemExit('usage: gen-b300-mainrec-hybrid8-next-self-prefetch.py INPUT.cu OUTPUT.cu [WIDTH] [DISTANCE] [EVICT] [GUARD]')

src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
try:
    width = int(sys.argv[3], 0) if len(sys.argv) >= 4 else 8
    distance = int(sys.argv[4], 0) if len(sys.argv) >= 5 else 1
except ValueError:
    raise SystemExit('WIDTH must be 1,2,4,8 and DISTANCE must be 1,2,4')
evict = sys.argv[5] if len(sys.argv) >= 6 else 'default'
guard = sys.argv[6] if len(sys.argv) >= 7 else 'branch'
if width not in (1, 2, 4, 8):
    raise SystemExit('WIDTH must be one of 1,2,4,8')
if distance not in (1, 2, 4):
    raise SystemExit('DISTANCE must be one of 1,2,4')
if evict not in ('default', 'normal', 'last'):
    raise SystemExit('EVICT must be one of default,normal,last')
if guard not in ('branch', 'predicated'):
    raise SystemExit('GUARD must be one of branch,predicated')
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

if evict == 'default':
    asm80 = asm70 = 'prefetch.global.L2'
else:
    asm80 = f'prefetch.global.L2::evict_{evict}'
    # The eviction-priority qualifier requires sm_80. Keep older targets legal.
    asm70 = 'prefetch.global.L2'

if guard == 'branch':
    helper = f'''

__device__ __forceinline__ void {helper_name}(const Count* p,bool valid){{
#if __CUDA_ARCH__ >= 800
    if(valid){{
        const unsigned long long a=reinterpret_cast<unsigned long long>(p);
        asm volatile("{asm80} [%0];" :: "l"(a));
    }}
#elif __CUDA_ARCH__ >= 700
    if(valid){{
        const unsigned long long a=reinterpret_cast<unsigned long long>(p);
        asm volatile("{asm70} [%0];" :: "l"(a));
    }}
#else
    (void)p;(void)valid;
#endif
}}
'''
else:
    # Do address arithmetic in PTX integer registers so invalid lanes never form
    # an out-of-range C++ pointer. The prefetch itself is guarded with @p rather
    # than a C++ if(valid), keeping semantics identical while removing the
    # source-level branch/conditional pointer from the hot ILP8 loop.
    helper = f'''

__device__ __forceinline__ void {helper_name}(const Count* base,Code idx,bool valid){{
#if __CUDA_ARCH__ >= 800
    const unsigned long long b=reinterpret_cast<unsigned long long>(base);
    const unsigned long long off=static_cast<unsigned long long>(idx)*sizeof(Count);
    const unsigned int v=valid?1u:0u;
    asm volatile("{{ .reg .pred p; .reg .b64 a; setp.ne.u32 p, %2, 0; add.u64 a, %0, %1; @p {asm80} [a]; }}" :: "l"(b), "l"(off), "r"(v));
#elif __CUDA_ARCH__ >= 700
    const unsigned long long b=reinterpret_cast<unsigned long long>(base);
    const unsigned long long off=static_cast<unsigned long long>(idx)*sizeof(Count);
    const unsigned int v=valid?1u:0u;
    asm volatile("{{ .reg .pred p; .reg .b64 a; setp.ne.u32 p, %2, 0; add.u64 a, %0, %1; @p {asm70} [a]; }}" :: "l"(b), "l"(off), "r"(v));
#else
    (void)base;(void)idx;(void)valid;
#endif
}}
'''
s = s.replace(marker, helper + marker, 1)

anchor = '        const Count self7=v7?in[i7]:Count(0);\n'
if s.count(anchor) != 1:
    raise SystemExit(f'hybrid8 self7 anchor expected one match got {s.count(anchor)}')

advance = 8 * distance
lines = [anchor.rstrip('\n'), f'        const Code next_base=base+Code({advance})*grid;']
for k in range(width):
    lines.append(f'        const Code ni{k}=next_base+Code({k})*grid;')
    if guard == 'branch':
        lines.append(f'        {helper_name}(ni{k}<n?in+ni{k}:in,ni{k}<n);')
    else:
        lines.append(f'        {helper_name}(in,ni{k},ni{k}<n);')
insert = '\n'.join(lines) + '\n'
s = s.replace(anchor, insert, 1)

if guard == 'branch':
    first_prefetch = f'{helper_name}(ni0<n?in+ni0:in,ni0<n);'
    last_prefetch = f'{helper_name}(ni{width-1}<n?in+ni{width-1}:in,ni{width-1}<n);'
else:
    first_prefetch = f'{helper_name}(in,ni0,ni0<n);'
    last_prefetch = f'{helper_name}(in,ni{width-1},ni{width-1}<n);'
pref = s.find(last_prefetch)
reduce_anchor = s.find('        const uint64_t mod=D_MOD;', pref)
if pref < 0 or reduce_anchor < 0 or pref >= reduce_anchor:
    raise SystemExit('hybrid8 next-self prefetch must precede current-iteration reduction')

for req in (
    'prefetch.global.L2',
    f'const Code next_base=base+Code({advance})*grid',
    first_prefetch,
    last_prefetch,
):
    if req not in s:
        raise SystemExit(f'missing hybrid8 next-self prefetch artifact: {req}')
if evict != 'default' and f'prefetch.global.L2::evict_{evict}' not in s:
    raise SystemExit('requested eviction-priority instruction missing')
if guard == 'predicated':
    for req in ('setp.ne.u32 p, %2, 0', '@p prefetch.global.L2', 'add.u64 a, %0, %1'):
        if req not in s:
            raise SystemExit(f'missing predicated guard artifact: {req}')
    if '?in+ni' in '\n'.join(lines):
        raise SystemExit('predicated guard unexpectedly emitted conditional C++ pointer')
if width < 8 and f'const Code ni{width}=' in s:
    raise SystemExit('hybrid8 next-self prefetch emitted more lanes than requested')

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(s)
print(
    f'generated {out} from {src}: '
    'b300_mainrec_hybrid8_next_self_prefetch=1 '
    f'next_iteration_self_prefetches_per_thread={width} prefetch_width={width} '
    f'prefetch_distance_iterations={distance} prefetch_advance={advance}grid cache=L2 '
    f'evict_priority={evict} eviction_hint_sm80_only={int(evict != "default")} '
    f'guard_mode={guard} ptx_predicated_guard={int(guard == "predicated")} '
    'prefetch_before_current_reduction=1 coalesced_per_k=1 '
    'semantics_unchanged=1 extra_shared_bytes=0'
)
