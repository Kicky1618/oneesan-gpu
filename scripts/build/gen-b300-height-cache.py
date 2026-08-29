#!/usr/bin/env python3
from __future__ import annotations
import pathlib,sys

if len(sys.argv)!=3:
    raise SystemExit('usage: gen-b300-height-cache.py INPUT.cu OUTPUT.cu')
src=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);s=src.read_text()
if 'template<bool CACHED_MATE>\n__global__ void main_pull_kernel' not in s:
    raise SystemExit('height cache requires main-pull transform')
if 'template<bool CACHED_BLOCK_MATE>\n__global__ void block_pull_kernel' not in s or 'dBlockMate' not in s:
    raise SystemExit('height cache requires block-pull + block-mate-cache transforms')

def once(old:str,new:str,label:str)->None:
    global s
    n=s.count(old)
    if n!=1:raise SystemExit(f'{label}: expected one match got {n}')
    s=s.replace(old,new,1)

marker='\n\nstatic Code rank_full(MateID m,int width)'
if marker not in s:raise SystemExit('rank_full marker not found')
helpers=r'''

using HeightCode = unsigned char;
__device__ __forceinline__ int b300_height_advance(int h,MateValue v){
    return h+(v==L)-(v==R);
}
__global__ void b300_init_main_height_kernel(const MateID*mates,Code n,HeightCode*out,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride)out[i]=HeightCode(height_before_rank_pos<TARGET_W>(mates[i],p));
}
__global__ void b300_init_block_height_kernel(const MateID*mates,Code n,HeightCode*out,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride)out[i]=HeightCode(height_before_rank_pos<TARGET_W-1>(mates[i],p-1));
}
'''
s=s.replace(marker,helpers+marker,1)

once(
'''__device__ __forceinline__ Code main_pull_direct_pair_source_rank(Code dst_rank,MateID m,int p){
    const int h=height_before_rank_pos<TARGET_W>(m,p);''',
'''__device__ __forceinline__ Code main_pull_direct_pair_source_rank(Code dst_rank,MateID m,int p,int h){''',
'main pair rank accepts cached height')
once(
'''template<bool CACHED_MATE>
__global__ void main_pull_kernel(const Count*in,const MateID*mates,Code n,const Count*in_block,Code nblock,Count*out_main,int p){''',
'''template<bool CACHED_MATE,bool CACHED_HEIGHT>
__global__ void main_pull_kernel(const Count*in,const MateID*mates,Code n,const Count*in_block,Code nblock,Count*out_main,int p,HeightCode*height){''',
'main height template')
once(
'''        else m=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);
        uint64_t acc=in[i];''',
'''        else m=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);
        int h;if constexpr(CACHED_HEIGHT)h=int(height[i]);else h=height_before_rank_pos<TARGET_W>(m,p);
        uint64_t acc=in[i];''',
'main height load')
once(
'''            const Code j=main_pull_direct_pair_source_rank(i,m,p);''',
'''            const Code j=main_pull_direct_pair_source_rank(i,m,p,h);''',
'main pair cached height use')
once(
'''        uint64_t mod=D_MOD;if(acc>=mod)acc-=mod;if(acc>=mod)acc-=mod;out_main[i]=Count(acc);
    }
}''',
'''        uint64_t mod=D_MOD;if(acc>=mod)acc-=mod;if(acc>=mod)acc-=mod;out_main[i]=Count(acc);
        if constexpr(CACHED_HEIGHT)height[i]=HeightCode(b300_height_advance(h,mget(m,p)));
    }
}''',
'main height recurrence store')

once(
'''template<bool CACHED_BLOCK_MATE>
__global__ void block_pull_kernel(const Count*in_main,const MateID*block_mates,Code n,Count*out_block,int p){''',
'''template<bool CACHED_BLOCK_MATE,bool CACHED_HEIGHT>
__global__ void block_pull_kernel(const Count*in_main,const MateID*block_mates,Code n,Count*out_block,int p,HeightCode*height){''',
'block height template')
once(
'''        MateID b;if constexpr(CACHED_BLOCK_MATE)b=block_mates[i];else b=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);
        Count acc=0;''',
'''        MateID b;if constexpr(CACHED_BLOCK_MATE)b=block_mates[i];else b=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);
        int h;if constexpr(CACHED_HEIGHT)h=int(height[i]);else h=height_before_rank_pos<TARGET_W-1>(b,p-1);
        Count acc=0;''',
'block height load')
once(
'''            const int H=height_before_rank_pos<TARGET_W>(d,p);''',
'''            const int H=h;''',
'block closure cached height use')
once(
'''        out_block[i]=acc;
    }
}''',
'''        out_block[i]=acc;
        if constexpr(CACHED_HEIGHT)height[i]=HeightCode(b300_height_advance(h,mget(b,p-1)));
    }
}''',
'block height recurrence store')

# Planner: both one-byte streams are part of the selected group footprint, so
# enabling the cache never silently falls back to recomputing prefix popcounts.
once(
'''size_t b=size_t(2*ms.size+2*ds.size)*sizeof(Count)+size_t(ms.size+ds.size)*sizeof(MateID);''',
'''size_t b=size_t(2*ms.size+2*ds.size)*sizeof(Count)+size_t(ms.size+ds.size)*sizeof(MateID)+size_t(ms.size+ds.size)*sizeof(HeightCode);''',
'planner height bytes')

once(
'MateID*dMate=nullptr,*dBlockMate=nullptr;PeerInterval*dIM=nullptr,*dID=nullptr;',
'MateID*dMate=nullptr,*dBlockMate=nullptr;HeightCode*dMainHeight=nullptr,*dBlockHeight=nullptr;PeerInterval*dIM=nullptr,*dID=nullptr;',
'DeviceCtx height fields')
once(
'void ensure(Code m,Code b,bool useMate,bool useBlockMate,size_t im,size_t id){',
'void ensure(Code m,Code b,bool useMate,bool useBlockMate,bool useHeight,size_t im,size_t id){',
'DeviceCtx height flag')
once(
'bmb=useBlockMate?al(size_t(b)*sizeof(MateID)):0,need=2*ab+2*db+mb+bmb;',
'bmb=useBlockMate?al(size_t(b)*sizeof(MateID)):0,mhb=useHeight?al(size_t(m)*sizeof(HeightCode)):0,bhb=useHeight?al(size_t(b)*sizeof(HeightCode)):0,need=2*ab+2*db+mb+bmb+mhb+bhb;',
'DeviceCtx height bytes')
once(
'dBlockMate=useBlockMate?(MateID*)(arena+off):nullptr;',
'dBlockMate=useBlockMate?(MateID*)(arena+off):nullptr;off+=bmb;dMainHeight=useHeight?(HeightCode*)(arena+off):nullptr;off+=mhb;dBlockHeight=useHeight?(HeightCode*)(arena+off):nullptr;',
'DeviceCtx height pointers')

old_size='''size_t countBytes=size_t(2*ms.size+2*ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID);bool useMate=(countBytes+mateBytes<=target);if(!useMate){std::cerr<<"planner regression: main MateID cache does not fit selected group bytes="<<(countBytes+mateBytes)<<" target="<<target<<"\\n";std::exit(19);}
    size_t blockMateBytes=size_t(ds.size)*sizeof(MateID);bool useBlockMate=(countBytes+mateBytes+blockMateBytes<=target);if(!useBlockMate){std::cerr<<"planner regression: blocked MateID cache does not fit selected group bytes="<<(countBytes+mateBytes+blockMateBytes)<<" target="<<target<<"\\n";std::exit(20);}
    c.ensure(ms.size,ds.size,useMate,useBlockMate,pg.mi.size(),pg.di.size());'''
new_size='''size_t countBytes=size_t(2*ms.size+2*ds.size)*sizeof(Count),mateBytes=size_t(ms.size)*sizeof(MateID);bool useMate=(countBytes+mateBytes<=target);if(!useMate){std::cerr<<"planner regression: main MateID cache does not fit selected group bytes="<<(countBytes+mateBytes)<<" target="<<target<<"\\n";std::exit(19);}
    size_t blockMateBytes=size_t(ds.size)*sizeof(MateID);bool useBlockMate=(countBytes+mateBytes+blockMateBytes<=target);if(!useBlockMate){std::cerr<<"planner regression: blocked MateID cache does not fit selected group bytes="<<(countBytes+mateBytes+blockMateBytes)<<" target="<<target<<"\\n";std::exit(20);}
    size_t heightBytes=size_t(ms.size+ds.size)*sizeof(HeightCode);bool useHeight=(countBytes+mateBytes+blockMateBytes+heightBytes<=target);if(!useHeight){std::cerr<<"planner regression: height cache does not fit selected group bytes="<<(countBytes+mateBytes+blockMateBytes+heightBytes)<<" target="<<target<<"\\n";std::exit(21);}
    c.ensure(ms.size,ds.size,useMate,useBlockMate,useHeight,pg.mi.size(),pg.di.size());'''
once(old_size,new_size,'process_group height sizing')

once(
'''    Count*cur=c.dA,*nxt=c.dB,*dcur=c.dD,*dnext=c.dE;''',
'''    if(useHeight){if(ms.size)b300_init_main_height_kernel<<<bm,threads>>>(c.dMate,ms.size,c.dMainHeight,wp.p_hi);if(ds.size)b300_init_block_height_kernel<<<bd,threads>>>(c.dBlockMate,ds.size,c.dBlockHeight,wp.p_hi);ck(cudaGetLastError(),"height cache init");ck(cudaDeviceSynchronize(),"height cache init sync");}
    Count*cur=c.dA,*nxt=c.dB,*dcur=c.dD,*dnext=c.dE;''',
'height initialization')

once(
'''if(useMate)main_pull_kernel<true><<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p);
                else main_pull_kernel<false><<<bm,threads,0,c.sMain>>>(cur,nullptr,ms.size,dcur,ds.size,nxt,p);''',
'''if(useHeight)main_pull_kernel<true,true><<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p,c.dMainHeight);
                else if(useMate)main_pull_kernel<true,false><<<bm,threads,0,c.sMain>>>(cur,c.dMate,ms.size,dcur,ds.size,nxt,p,nullptr);
                else main_pull_kernel<false,false><<<bm,threads,0,c.sMain>>>(cur,nullptr,ms.size,dcur,ds.size,nxt,p,nullptr);''',
'main height launch')
once(
'''if(ds.size){if(useBlockMate)block_pull_kernel<true><<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p);else block_pull_kernel<false><<<bd,threads,0,c.sBlock>>>(cur,nullptr,ds.size,dnext,p);}''',
'''if(ds.size){if(useHeight)block_pull_kernel<true,true><<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p,c.dBlockHeight);else if(useBlockMate)block_pull_kernel<true,false><<<bd,threads,0,c.sBlock>>>(cur,c.dBlockMate,ds.size,dnext,p,nullptr);else block_pull_kernel<false,false><<<bd,threads,0,c.sBlock>>>(cur,nullptr,ds.size,dnext,p,nullptr);}''',
'block height launch')

for required in ('dMainHeight','b300_init_main_height_kernel','main_pull_kernel<true,true>','block_pull_kernel<true,true>','height[i]=HeightCode'):
    if required not in s:raise SystemExit(f'missing height-cache artifact: {required}')

out.parent.mkdir(parents=True,exist_ok=True);out.write_text(s)
print(f'generated {out} from {src}: b300_height_cache=1 bytes_per_state=1 hbm_rw_per_state_step=2 prefix_popcount_per_state_step=0 recurrence=O1 planner_cache_aware=1 exact_fallback=1')
