#!/usr/bin/env python3
import pathlib, sys

if len(sys.argv) != 3:
    raise SystemExit('usage: gen-b300-main-mate-cache.py INPUT.cu OUTPUT.cu')
src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
s = src.read_text()

def once(old,new,label):
    global s
    n=s.count(old)
    if n!=1:
        raise SystemExit(f'{label}: expected one match got {n}')
    s=s.replace(old,new,1)

once(
'__global__ void main_group_kernel(const Count*in,const MateID*mates,Code n,Count*out_main,Count*out_block,int p){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Count c=in[i];if(!c)continue;MateID m=mates?mates[i]:unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);',
'__global__ void main_group_kernel_body_marker_unused(const Count*,const MateID*,Code,Count*,Count*,int);\ntemplate<bool CACHED_MATE>\n__global__ void main_group_kernel(const Count*in,const MateID*mates,Code n,Count*out_main,Count*out_block,int p){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Count c=in[i];if(!c)continue;MateID m;if constexpr(CACHED_MATE)m=mates[i];else m=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);',
'main cached template')
s=s.replace('__global__ void main_group_kernel_body_marker_unused(const Count*,const MateID*,Code,Count*,Count*,int);\n','',1)

marker='__global__ void blocked_group_kernel('
if s.count(marker)!=1:raise SystemExit('blocked_group_kernel marker not unique')
insert='''__global__ void materialize_main_mates_kernel(MateID* mates,Code n){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride)mates[i]=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);
}
'''
s=s.replace(marker,insert+marker,1)

once(
'size_t b=size_t(2*ms.size+2*ds.size)*sizeof(Count);',
'size_t b=size_t(2*ms.size+2*ds.size)*sizeof(Count)+size_t(ms.size)*sizeof(MateID);',
'planner main mate bytes')

once(
'size_t countBytes=size_t(2*ms.size+2*ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID);bool useMate=!pg.use_mi&&(countBytes+mateBytes<=target);',
'size_t countBytes=size_t(2*ms.size+2*ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID);bool useMate=(countBytes+mateBytes<=target);if(!useMate){std::cerr<<"planner regression: main MateID cache does not fit selected group bytes="<<(countBytes+mateBytes)<<" target="<<target<<"\\n";std::exit(19);}',
'useMate guaranteed by planner')

once(
'if(ms.size){if(pg.use_mi)interval_io_kernel<false,false><<<interval_blocks(pg.mi.size(),threads),threads>>>(c.dA,c.dIM,pg.mi.size());else gather_main_kernel<<<bm,threads>>>(c.dA,useMate?c.dMate:nullptr,ms.size);}',
'if(ms.size){if(pg.use_mi){interval_io_kernel<false,false><<<interval_blocks(pg.mi.size(),threads),threads>>>(c.dA,c.dIM,pg.mi.size());if(useMate)materialize_main_mates_kernel<<<bm,threads>>>(c.dMate,ms.size);}else gather_main_kernel<<<bm,threads>>>(c.dA,useMate?c.dMate:nullptr,ms.size);}',
'main gather mate materialization')

once(
'if(ms.size)main_group_kernel<<<bm,threads,0,c.sMain>>>(cur,useMate?c.dMate:nullptr,ms.size,nxt,dnext,p);',
'if(ms.size){if(useMate)main_group_kernel<true><<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,nxt,dnext,p);else main_group_kernel<false><<<bm,threads,0,c.sMain>>>(cur,nullptr,ms.size,nxt,dnext,p);}',
'main cached launch')

for stale in ('bool useMate=!pg.use_mi', 'size_t b=size_t(2*ms.size+2*ds.size)*sizeof(Count);'):
    if stale in s:raise SystemExit(f'stale main-mate artifact remains: {stale}')

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(s)
print(f'generated {out} from {src}: b300_main_mate_cache=1 compile_time_cached_kernel=1 interval_materialize=1 planner_cache_aware=1 cache_fit_required=1')
