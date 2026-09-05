#!/usr/bin/env python3
import pathlib, sys

if len(sys.argv) != 3:
    raise SystemExit('usage: gen-b300-block-mate-cache.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
if 'block_pull_kernel' not in s or 'doubleD full pull transition' not in s:
    raise SystemExit('block mate cache requires gen-b300-block-pull.py first')

def once(old,new,label):
    global s
    n=s.count(old)
    if n!=1:raise SystemExit(f'{label}: expected one match got {n}')
    s=s.replace(old,new,1)

once(
'__global__ void gather_block_kernel(Count*out,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g;unrank_group_global_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP,g);out[i]=global_load_block(g);}}',
'__global__ void gather_block_kernel(Count*out,MateID*mates,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Code g;MateID m=unrank_group_global_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP,g);out[i]=global_load_block(g);if(mates)mates[i]=m;}}',
'gather_block mate fusion')

once(
'__global__ void block_pull_kernel(const Count*in_main,Code n,Count*out_block,int p){\n    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;\n    for(;i<n;i+=stride){\n        MateID b=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);',
'__global__ void block_pull_kernel_body_marker_unused(const Count*,Code,Count*,int);\ntemplate<bool CACHED_BLOCK_MATE>\n__global__ void block_pull_kernel(const Count*in_main,const MateID*block_mates,Code n,Count*out_block,int p){\n    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;\n    for(;i<n;i+=stride){\n        MateID b;if constexpr(CACHED_BLOCK_MATE)b=block_mates[i];else b=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);',
'block pull cached template')
s=s.replace('__global__ void block_pull_kernel_body_marker_unused(const Count*,Code,Count*,int);\n','''__global__ void materialize_block_mates_kernel(MateID*mates,Code n){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride)mates[i]=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);
}
''',1)

# The main-mate transform already made plan_window account for main MateID
# storage. Full-pull also requires one cached blocked MateID per blocked state.
once(
'size_t b=size_t(2*ms.size+2*ds.size)*sizeof(Count)+size_t(ms.size)*sizeof(MateID);',
'size_t b=size_t(2*ms.size+2*ds.size)*sizeof(Count)+size_t(ms.size+ds.size)*sizeof(MateID);',
'planner block mate bytes')

once('MateID*dMate=nullptr;PeerInterval*dIM=nullptr,*dID=nullptr;',
     'MateID*dMate=nullptr,*dBlockMate=nullptr;PeerInterval*dIM=nullptr,*dID=nullptr;',
     'DeviceCtx block mate field')
once(
'void ensure(Code m,Code b,bool useMate,size_t im,size_t id){ck(cudaSetDevice(dev),"set ensure");auto al=[](size_t x){return(x+255)&~size_t(255);};size_t ab=al(size_t(m)*sizeof(Count)),db=al(size_t(b)*sizeof(Count)),mb=useMate?al(size_t(m)*sizeof(MateID)):0,need=2*ab+2*db+mb;if(need>capArena){if(arena)cudaFree(arena);capArena=need;ck(cudaMalloc(&arena,capArena),"scratch arena");}size_t off=0;dA=(Count*)(arena+off);off+=ab;dB=(Count*)(arena+off);off+=ab;dD=(Count*)(arena+off);off+=db;dE=(Count*)(arena+off);off+=db;dMate=useMate?(MateID*)(arena+off):nullptr;',
'void ensure(Code m,Code b,bool useMate,bool useBlockMate,size_t im,size_t id){ck(cudaSetDevice(dev),"set ensure");auto al=[](size_t x){return(x+255)&~size_t(255);};size_t ab=al(size_t(m)*sizeof(Count)),db=al(size_t(b)*sizeof(Count)),mb=useMate?al(size_t(m)*sizeof(MateID)):0,bmb=useBlockMate?al(size_t(b)*sizeof(MateID)):0,need=2*ab+2*db+mb+bmb;if(need>capArena){if(arena)cudaFree(arena);capArena=need;ck(cudaMalloc(&arena,capArena),"scratch arena");}size_t off=0;dA=(Count*)(arena+off);off+=ab;dB=(Count*)(arena+off);off+=ab;dD=(Count*)(arena+off);off+=db;dE=(Count*)(arena+off);off+=db;dMate=useMate?(MateID*)(arena+off):nullptr;off+=mb;dBlockMate=useBlockMate?(MateID*)(arena+off):nullptr;',
'DeviceCtx ensure block mate allocation')

once(
'    c.ensure(ms.size,ds.size,useMate,pg.mi.size(),pg.di.size());',
'    size_t blockMateBytes=size_t(ds.size)*sizeof(MateID);bool useBlockMate=(countBytes+mateBytes+blockMateBytes<=target);if(!useBlockMate){std::cerr<<"planner regression: blocked MateID cache does not fit selected group bytes="<<(countBytes+mateBytes+blockMateBytes)<<" target="<<target<<"\\n";std::exit(20);}\n    c.ensure(ms.size,ds.size,useMate,useBlockMate,pg.mi.size(),pg.di.size());',
'process_group block mate sizing')

once(
'if(ds.size){if(pg.use_di)interval_io_kernel<true,false><<<interval_blocks(pg.di.size(),threads),threads>>>(c.dD,c.dID,pg.di.size());else gather_block_kernel<<<bd,threads>>>(c.dD,ds.size);}',
'if(ds.size){if(pg.use_di){interval_io_kernel<true,false><<<interval_blocks(pg.di.size(),threads),threads>>>(c.dD,c.dID,pg.di.size());if(useBlockMate)materialize_block_mates_kernel<<<bd,threads>>>(c.dBlockMate,ds.size);}else gather_block_kernel<<<bd,threads>>>(c.dD,useBlockMate?c.dBlockMate:nullptr,ds.size);}',
'block gather fused mate materialization')

once(
'if(ds.size)block_pull_kernel<<<bd,threads,0,c.sBlock>>>(cur,ds.size,dnext,p);',
'if(ds.size){if(useBlockMate)block_pull_kernel<true><<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p);else block_pull_kernel<false><<<bd,threads,0,c.sBlock>>>(cur,nullptr,ds.size,dnext,p);}',
'block pull cached launch')

for stale in (
    'gather_block_kernel<<<bd,threads>>>(c.dD,ds.size)',
    'block_pull_kernel<<<bd,threads,0,c.sBlock>>>',
    'size_t b=size_t(2*ms.size+2*ds.size)*sizeof(Count)+size_t(ms.size)*sizeof(MateID);'
):
    if stale in s:raise SystemExit(f'stale block-mate artifact remains: {stale}')

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_block_mate_cache=1 window_reuse=1 conditional_scratch=0 gather_fused_materialize=1 interval_materialize=1 planner_cache_aware=1 cache_fit_required=1')
