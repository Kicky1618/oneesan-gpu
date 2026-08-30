#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys

if len(sys.argv) not in (3,4,5,6,7):
    raise SystemExit('usage: gen-b300-mainrec-hybrid8-next-mate-prefetch.py INPUT.cu OUTPUT.cu [WIDTH] [DISTANCE] [EVICT] [GUARD]')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2])
try:
    width=int(sys.argv[3],0) if len(sys.argv)>=4 else 8
    distance=int(sys.argv[4],0) if len(sys.argv)>=5 else 1
except ValueError:
    raise SystemExit('WIDTH must be 1,2,4,8 and DISTANCE must be 1,2,4')
evict=sys.argv[5] if len(sys.argv)>=6 else 'default'
guard=sys.argv[6] if len(sys.argv)>=7 else 'branch'
if width not in (1,2,4,8): raise SystemExit('WIDTH must be one of 1,2,4,8')
if distance not in (1,2,4): raise SystemExit('DISTANCE must be one of 1,2,4')
if evict not in ('default','normal','last'): raise SystemExit('EVICT must be one of default,normal,last')
if guard not in ('branch','predicated'): raise SystemExit('GUARD must be one of branch,predicated')
s=src.read_text()
for req in ('main_pull_kernel_ilp8_hybrid','MateID* __restrict__ mates','base+=Code(8)*grid','const Count self7=','const uint64_t mod=D_MOD;'):
    if req not in s: raise SystemExit(f'hybrid8 next-mate prefetch requires artifact: {req}')
helper='b300_mainrec_hybrid8_prefetch_next_mate_l2'
if helper in s: raise SystemExit('source already contains hybrid8 next-mate prefetch')
marker='\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s: raise SystemExit('rank_full marker not found')
if evict=='default':
    asm80=asm70='prefetch.global.L2'
else:
    asm80=f'prefetch.global.L2::evict_{evict}'
    asm70='prefetch.global.L2'
if guard=='branch':
    helper_src=f'''

__device__ __forceinline__ void {helper}(const MateID* p,bool valid){{
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
    helper_src=f'''

__device__ __forceinline__ void {helper}(const MateID* base,Code idx,bool valid){{
#if __CUDA_ARCH__ >= 800
    const unsigned long long b=reinterpret_cast<unsigned long long>(base);
    const unsigned long long off=static_cast<unsigned long long>(idx)*sizeof(MateID);
    const unsigned int v=valid?1u:0u;
    asm volatile("{{ .reg .pred p; .reg .b64 a; setp.ne.u32 p, %2, 0; add.u64 a, %0, %1; @p {asm80} [a]; }}" :: "l"(b), "l"(off), "r"(v));
#elif __CUDA_ARCH__ >= 700
    const unsigned long long b=reinterpret_cast<unsigned long long>(base);
    const unsigned long long off=static_cast<unsigned long long>(idx)*sizeof(MateID);
    const unsigned int v=valid?1u:0u;
    asm volatile("{{ .reg .pred p; .reg .b64 a; setp.ne.u32 p, %2, 0; add.u64 a, %0, %1; @p {asm70} [a]; }}" :: "l"(b), "l"(off), "r"(v));
#else
    (void)base;(void)idx;(void)valid;
#endif
}}
'''
s=s.replace(marker,helper_src+marker,1)
anchor='        const Count self7=v7?in[i7]:Count(0);\n'
if s.count(anchor)!=1: raise SystemExit(f'hybrid8 self7 anchor expected one match got {s.count(anchor)}')
advance=8*distance
lines=[anchor.rstrip('\n'),f'        const Code next_mate_base=base+Code({advance})*grid;']
for k in range(width):
    lines.append(f'        const Code nmi{k}=next_mate_base+Code({k})*grid;')
    if guard=='branch':
        lines.append(f'        {helper}(nmi{k}<n?mates+nmi{k}:mates,nmi{k}<n);')
    else:
        lines.append(f'        {helper}(mates,nmi{k},nmi{k}<n);')
s=s.replace(anchor,'\n'.join(lines)+'\n',1)
if guard=='branch':
    first=f'{helper}(nmi0<n?mates+nmi0:mates,nmi0<n);'
    last=f'{helper}(nmi{width-1}<n?mates+nmi{width-1}:mates,nmi{width-1}<n);'
else:
    first=f'{helper}(mates,nmi0,nmi0<n);'
    last=f'{helper}(mates,nmi{width-1},nmi{width-1}<n);'
a=s.find(last); b=s.find('        const uint64_t mod=D_MOD;',a)
if a<0 or b<0 or a>=b: raise SystemExit('hybrid8 next-mate prefetch must precede current reduction')
for req in ('prefetch.global.L2',f'const Code next_mate_base=base+Code({advance})*grid',first,last):
    if req not in s: raise SystemExit(f'missing hybrid8 next-mate artifact: {req}')
if evict!='default' and f'prefetch.global.L2::evict_{evict}' not in s: raise SystemExit('requested mate eviction-priority instruction missing')
if guard=='predicated':
    for req in ('setp.ne.u32 p, %2, 0','@p prefetch.global.L2','add.u64 a, %0, %1'):
        if req not in s: raise SystemExit(f'missing predicated mate guard artifact: {req}')
    if '?mates+nmi' in '\n'.join(lines): raise SystemExit('predicated mate guard unexpectedly emitted conditional C++ pointer')
if width<8 and f'const Code nmi{width}=' in s: raise SystemExit('next-mate emitted more lanes than requested')
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(s)
print(f'generated {out} from {src}: b300_mainrec_hybrid8_next_mate_prefetch=1 prefetch_width={width} prefetch_distance_iterations={distance} prefetch_advance={advance}grid cache=L2 evict_priority={evict} eviction_hint_sm80_only={int(evict != "default")} guard_mode={guard} ptx_predicated_guard={int(guard == "predicated")} prefetch_before_current_reduction=1 semantics_unchanged=1 extra_shared_bytes=0')
