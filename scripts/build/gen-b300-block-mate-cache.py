#!/usr/bin/env python3
import pathlib, sys

if len(sys.argv) != 3:
    raise SystemExit('usage: gen-b300-block-mate-cache.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); s=src.read_text()
if 'block_pull_kernel' not in s or 'doubleD full pull transition' not in s:
    raise SystemExit('blocked-mate transform requires gen-b300-block-pull.py first')

old='__global__ void block_pull_kernel(const Count*in_main,Code n,Count*out_block,int p){\n    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;\n    for(;i<n;i+=stride){\n        MateID b=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);'
new='template<bool CACHED_BLOCK_MATE>\n__global__ void block_pull_kernel(const Count*in_main,const MateID*block_mates,Code n,Count*out_block,int p){\n    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;\n    for(;i<n;i+=stride){\n        MateID b;if constexpr(CACHED_BLOCK_MATE)b=block_mates[i];else b=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);'
if old not in s: raise SystemExit('block_pull_kernel anchor not found')
s=s.replace(old,new,1)

marker='__global__ void block_pull_kernel'
pos=s.find(marker)
if pos<0: raise SystemExit('block pull marker missing after template')
insert='''__global__ void materialize_block_mates_kernel(MateID*mates,Code n){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride)mates[i]=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);
}
'''
s=s[:pos]+insert+s[pos:]

old='MateID*dMate=nullptr;PeerInterval*dIM=nullptr,*dID=nullptr;'
new='MateID*dMate=nullptr,*dBlockMate=nullptr;PeerInterval*dIM=nullptr,*dID=nullptr;'
if old not in s: raise SystemExit('DeviceCtx mate field anchor not found')
s=s.replace(old,new,1)

old='void ensure(Code m,Code b,bool useMate,size_t im,size_t id){ck(cudaSetDevice(dev),"set ensure");auto al=[](size_t x){return(x+255)&~size_t(255);};size_t ab=al(size_t(m)*sizeof(Count)),db=al(size_t(b)*sizeof(Count)),mb=useMate?al(size_t(m)*sizeof(MateID)):0,need=2*ab+2*db+mb;if(need>capArena){if(arena)cudaFree(arena);capArena=need;ck(cudaMalloc(&arena,capArena),"scratch arena");}size_t off=0;dA=(Count*)(arena+off);off+=ab;dB=(Count*)(arena+off);off+=ab;dD=(Count*)(arena+off);off+=db;dE=(Count*)(arena+off);off+=db;dMate=useMate?(MateID*)(arena+off):nullptr;'
new='void ensure(Code m,Code b,bool useMate,bool useBlockMate,size_t im,size_t id){ck(cudaSetDevice(dev),"set ensure");auto al=[](size_t x){return(x+255)&~size_t(255);};size_t ab=al(size_t(m)*sizeof(Count)),db=al(size_t(b)*sizeof(Count)),mb=useMate?al(size_t(m)*sizeof(MateID)):0,bmb=useBlockMate?al(size_t(b)*sizeof(MateID)):0,need=2*ab+2*db+mb+bmb;if(need>capArena){if(arena)cudaFree(arena);capArena=need;ck(cudaMalloc(&arena,capArena),"scratch arena");}size_t off=0;dA=(Count*)(arena+off);off+=ab;dB=(Count*)(arena+off);off+=ab;dD=(Count*)(arena+off);off+=db;dE=(Count*)(arena+off);off+=db;dMate=useMate?(MateID*)(arena+off):nullptr;off+=mb;dBlockMate=useBlockMate?(MateID*)(arena+off):nullptr;'
if old not in s: raise SystemExit('DeviceCtx ensure anchor not found')
s=s.replace(old,new,1)

old='size_t countBytes=size_t(2*ms.size+2*ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID);bool useMate=(countBytes+mateBytes<=target);\n    c.ensure(ms.size,ds.size,useMate,pg.mi.size(),pg.di.size());'
new='size_t countBytes=size_t(2*ms.size+2*ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID),blockMateBytes=size_t(ds.size)*sizeof(MateID);bool useMate=(countBytes+mateBytes<=target);bool useBlockMate=(countBytes+(useMate?mateBytes:0)+blockMateBytes<=target);\n    c.ensure(ms.size,ds.size,useMate,useBlockMate,pg.mi.size(),pg.di.size());'
if old not in s: raise SystemExit('process_group cache sizing anchor not found')
s=s.replace(old,new,1)

old='if(ds.size){if(pg.use_di)interval_io_kernel<true,false><<<interval_blocks(pg.di.size(),threads),threads>>>(c.dD,c.dID,pg.di.size());else gather_block_kernel<<<bd,threads>>>(c.dD,ds.size);}ck(cudaGetLastError(),"gather");ck(cudaDeviceSynchronize(),"gather sync");'
new='if(ds.size){if(pg.use_di)interval_io_kernel<true,false><<<interval_blocks(pg.di.size(),threads),threads>>>(c.dD,c.dID,pg.di.size());else gather_block_kernel<<<bd,threads>>>(c.dD,ds.size);if(useBlockMate)materialize_block_mates_kernel<<<bd,threads>>>(c.dBlockMate,ds.size);}ck(cudaGetLastError(),"gather");ck(cudaDeviceSynchronize(),"gather sync");'
if old not in s: raise SystemExit('blocked gather anchor not found')
s=s.replace(old,new,1)

old='if(ds.size)block_pull_kernel<<<bd,threads,0,c.sBlock>>>(cur,ds.size,dnext,p);'
new='if(ds.size){if(useBlockMate)block_pull_kernel<true><<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p);else block_pull_kernel<false><<<bd,threads,0,c.sBlock>>>(cur,nullptr,ds.size,dnext,p);}'
if old not in s: raise SystemExit('block pull launch anchor not found')
s=s.replace(old,new,1)

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_block_mate_cache=1 window_reuse=1 conditional_scratch=1')
