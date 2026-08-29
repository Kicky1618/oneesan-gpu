#!/usr/bin/env python3
import pathlib, sys

if len(sys.argv) != 3:
    raise SystemExit('usage: gen-b300-main-mate-cache.py INPUT.cu OUTPUT.cu')
src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
s = src.read_text()

old = '__global__ void main_group_kernel(const Count*in,const MateID*mates,Code n,Count*out_main,Count*out_block,int p){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Count c=in[i];if(!c)continue;MateID m=mates?mates[i]:unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);'
new = 'template<bool CACHED_MATE>\n__global__ void main_group_kernel(const Count*in,const MateID*mates,Code n,Count*out_main,Count*out_block,int p){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Count c=in[i];if(!c)continue;MateID m;if constexpr(CACHED_MATE)m=mates[i];else m=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);'
if old not in s:
    raise SystemExit('main_group_kernel marker not found')
s = s.replace(old, new, 1)

marker = '__global__ void blocked_group_kernel('
insert = '''__global__ void materialize_main_mates_kernel(MateID* mates,Code n){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride)mates[i]=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);
}
'''
if marker not in s:
    raise SystemExit('blocked_group_kernel marker not found')
s = s.replace(marker, insert + marker, 1)

old = 'size_t countBytes=size_t(2*ms.size+2*ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID);bool useMate=!pg.use_mi&&(countBytes+mateBytes<=target);'
new = 'size_t countBytes=size_t(2*ms.size+2*ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID);bool useMate=(countBytes+mateBytes<=target);'
if old not in s:
    raise SystemExit('useMate marker not found')
s = s.replace(old, new, 1)

old = 'if(ms.size){if(pg.use_mi)interval_io_kernel<false,false><<<interval_blocks(pg.mi.size(),threads),threads>>>(c.dA,c.dIM,pg.mi.size());else gather_main_kernel<<<bm,threads>>>(c.dA,useMate?c.dMate:nullptr,ms.size);}'
new = 'if(ms.size){if(pg.use_mi){interval_io_kernel<false,false><<<interval_blocks(pg.mi.size(),threads),threads>>>(c.dA,c.dIM,pg.mi.size());if(useMate)materialize_main_mates_kernel<<<bm,threads>>>(c.dMate,ms.size);}else gather_main_kernel<<<bm,threads>>>(c.dA,useMate?c.dMate:nullptr,ms.size);}'
if old not in s:
    raise SystemExit('main gather marker not found')
s = s.replace(old, new, 1)

old = 'if(ms.size)main_group_kernel<<<bm,threads,0,c.sMain>>>(cur,useMate?c.dMate:nullptr,ms.size,nxt,dnext,p);'
new = 'if(ms.size){if(useMate)main_group_kernel<true><<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,nxt,dnext,p);else main_group_kernel<false><<<bm,threads,0,c.sMain>>>(cur,nullptr,ms.size,nxt,dnext,p);}'
if old not in s:
    raise SystemExit('main launch marker not found')
s = s.replace(old, new, 1)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(s)
print(f'generated {out} from {src}: b300_main_mate_cache=1 compile_time_cached_kernel=1 interval_materialize=1')
