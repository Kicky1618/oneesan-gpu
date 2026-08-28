#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

INCLUDE_OLD = '#include <vector>\n'
INCLUDE_NEW = '#include <vector>\n#include "b300_vmm_contiguous_storage.cuh"\n'

SYMBOL_OLD = '''__constant__ Count* D_MAIN_PTR[MAXGPU];
__constant__ Count* D_BLOCK_PTR[MAXGPU];
__constant__ Code D_MAIN_CHUNK,D_BLOCK_CHUNK;'''
SYMBOL_NEW = '''static_assert(sizeof(Count)==4,"B300 VMM authoritative storage requires 32-bit Count");
__constant__ Count* D_MAIN_PTR[MAXGPU];
__constant__ Count* D_BLOCK_PTR[MAXGPU];
__constant__ Count* D_MAIN_VBASE;
__constant__ Count* D_BLOCK_VBASE;
__constant__ Code D_MAIN_CHUNK,D_BLOCK_CHUNK;'''

INTERVAL_OLD = '''struct PeerInterval{Code remote,local,len;uint32_t owner,pad;};
static std::vector<PeerInterval> make_peer_intervals(const GroupSpec&s,Code chunk,int ng,bool& use_interval){
    constexpr Code MIN_AVG_INTERVAL_ELEMS = 65536;
    Code est=interval_leaf_upper(s);
    // Every shard boundary can split at most one globally ordered interval.
    Code est_peer=est+Code(std::max(0,ng-1));
    use_interval = est_peer==0 || s.size >= est_peer*MIN_AVG_INTERVAL_ELEMS;
    if(!use_interval)return {};
    auto base=make_intervals(s);std::vector<PeerInterval>out;out.reserve(base.size()+ng);
    for(auto const&x:base){Code g=x.global,l=x.local,left=x.len;while(left){int owner=int(g/chunk);if(owner>=ng)owner=ng-1;Code shard0=Code(owner)*chunk;Code shard_end=(owner+1<ng)?shard0+chunk:~Code(0);Code take=left;if(shard_end!=~Code(0)&&g+take>shard_end)take=shard_end-g;Code remote=g-shard0;
            if(!out.empty()&&out.back().owner==(uint32_t)owner&&out.back().remote+out.back().len==remote&&out.back().local+out.back().len==l)out.back().len+=take;else out.push_back({remote,l,take,(uint32_t)owner,0});
            g+=take;l+=take;left-=take;}}
    // Interval I/O wins only when the canonical layout contains reasonably long
    // contiguous runs.  Low fixed bits can fragment a group into hundreds of
    // thousands of tiny runs; in that case rank/unrank gather/scatter is faster.
    use_interval = out.empty() || s.size >= Code(out.size()) * MIN_AVG_INTERVAL_ELEMS;
    if(!use_interval) return {};

    std::vector<PeerInterval> tiled;
    size_t nt=0;
    for(auto const& x:out) nt += size_t((x.len + IO_TILE_ELEMS - 1) / IO_TILE_ELEMS);
    tiled.reserve(nt);
    for(auto const& x:out){
        Code off=0;
        while(off<x.len){
            Code take=std::min<Code>(x.len-off,Code(IO_TILE_ELEMS));
            tiled.push_back({x.remote+off,x.local+off,take,x.owner,0});
            off+=take;
        }
    }
    return tiled;
}'''
INTERVAL_NEW = '''struct PeerInterval{Code remote,local,len;uint32_t owner,pad;};
static std::vector<PeerInterval> make_peer_intervals(const GroupSpec&s,Code chunk,int ng,bool& use_interval){
    (void)chunk;(void)ng;
    constexpr Code MIN_AVG_INTERVAL_ELEMS = 65536;
    Code est=interval_leaf_upper(s);
    // VMM exposes one contiguous authoritative VA, so no logical shard boundary
    // can split an interval. remote stores the global authoritative offset.
    use_interval = est==0 || s.size >= est*MIN_AVG_INTERVAL_ELEMS;
    if(!use_interval)return {};
    auto base=make_intervals(s);
    use_interval = base.empty() || s.size >= Code(base.size()) * MIN_AVG_INTERVAL_ELEMS;
    if(!use_interval)return {};

    std::vector<PeerInterval> tiled;
    size_t nt=0;
    for(auto const& x:base)nt+=size_t((x.len+IO_TILE_ELEMS-1)/IO_TILE_ELEMS);
    tiled.reserve(nt);
    for(auto const& x:base){
        Code off=0;
        while(off<x.len){
            Code take=std::min<Code>(x.len-off,Code(IO_TILE_ELEMS));
            tiled.push_back({x.global+off,x.local+off,take,0,0});
            off+=take;
        }
    }
    return tiled;
}'''

SHARD_OLD = '''struct ShardAddress8{int owner;Code local;};
__device__ __forceinline__ ShardAddress8 shard_address8(Code g,Code chunk){int o=0;Code c4=chunk<<2;if(g>=c4){g-=c4;o|=4;}Code c2=chunk<<1;if(g>=c2){g-=c2;o|=2;}if(g>=chunk){g-=chunk;o|=1;}return{o,g};}
#if B300_FAST_SHARD_ADDRESS8
__device__ __forceinline__ Count global_load_main(Code g){auto a=shard_address8(g,D_MAIN_CHUNK);return D_MAIN_PTR[a.owner][a.local];}
__device__ __forceinline__ Count global_load_block(Code g){auto a=shard_address8(g,D_BLOCK_CHUNK);return D_BLOCK_PTR[a.owner][a.local];}
__device__ __forceinline__ void global_store_main(Code g,Count v){auto a=shard_address8(g,D_MAIN_CHUNK);D_MAIN_PTR[a.owner][a.local]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){auto a=shard_address8(g,D_BLOCK_CHUNK);D_BLOCK_PTR[a.owner][a.local]=v;}
#else
__device__ __forceinline__ Count global_load_main(Code g){int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_MAIN_PTR[o][g-Code(o)*D_MAIN_CHUNK];}
__device__ __forceinline__ Count global_load_block(Code g){int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_BLOCK_PTR[o][g-Code(o)*D_BLOCK_CHUNK];}
__device__ __forceinline__ void global_store_main(Code g,Count v){int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;D_MAIN_PTR[o][g-Code(o)*D_MAIN_CHUNK]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;D_BLOCK_PTR[o][g-Code(o)*D_BLOCK_CHUNK]=v;}
#endif'''
SHARD_NEW = '''struct ShardAddress8{int owner;Code local;};
__device__ __forceinline__ ShardAddress8 shard_address8(Code g,Code chunk){int o=0;Code c4=chunk<<2;if(g>=c4){g-=c4;o|=4;}Code c2=chunk<<1;if(g>=c2){g-=c2;o|=2;}if(g>=chunk){g-=chunk;o|=1;}return{o,g};}
__device__ __forceinline__ Count global_load_main(Code g){return D_MAIN_VBASE[g];}
__device__ __forceinline__ Count global_load_block(Code g){return D_BLOCK_VBASE[g];}
__device__ __forceinline__ void global_store_main(Code g,Count v){D_MAIN_VBASE[g]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){D_BLOCK_VBASE[g]=v;}'''

INTERVAL_KERNEL_OLD = '''template<bool BLOCK,bool SCATTER>
__global__ void interval_io_kernel(Count*buf,const PeerInterval*iv,size_t niv){
    for(size_t k=blockIdx.x;k<niv;k+=gridDim.x){
        PeerInterval x=iv[k];
        Count*peer=(BLOCK?D_BLOCK_PTR[x.owner]:D_MAIN_PTR[x.owner])+x.remote;
        for(Code off=threadIdx.x;off<x.len;off+=blockDim.x){
            if constexpr(SCATTER) peer[off]=buf[x.local+off];
            else buf[x.local+off]=peer[off];
        }
    }
}'''
INTERVAL_KERNEL_NEW = '''template<bool BLOCK,bool SCATTER>
__global__ void interval_io_kernel(Count*buf,const PeerInterval*iv,size_t niv){
    for(size_t k=blockIdx.x;k<niv;k+=gridDim.x){
        PeerInterval x=iv[k];
        Count*peer=(BLOCK?D_BLOCK_VBASE:D_MAIN_VBASE)+x.remote;
        for(Code off=threadIdx.x;off<x.len;off+=blockDim.x){
            if constexpr(SCATTER) peer[off]=buf[x.local+off];
            else buf[x.local+off]=peer[off];
        }
    }
}'''

ALLOC_OLD = '''Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];Code mc=(mainN+ng-1)/ng,bc=(blockN+ng-1)/ng;Count*mp[MAXGPU]{},*bp[MAXGPU]{};std::vector<Code>ml(ng),bl(ng);
    for(int d=0;d<ng;++d){ml[d]=std::min<Code>(mc,mainN-std::min<Code>(mainN,Code(d)*mc));bl[d]=std::min<Code>(bc,blockN-std::min<Code>(blockN,Code(d)*bc));cudaSetDevice(d);if(ml[d]){ck(cudaMalloc(&mp[d],size_t(ml[d])*sizeof(Count)),"auth main");ck(cudaMemset(mp[d],0,size_t(ml[d])*sizeof(Count)),"zero main");}if(bl[d]){ck(cudaMalloc(&bp[d],size_t(bl[d])*sizeof(Count)),"auth block");ck(cudaMemset(bp[d],0,size_t(bl[d])*sizeof(Count)),"zero block");}}
    std::vector<DeviceCtx>ctx(ng);for(int d=0;d<ng;++d)ctx[d].init(d,mod,mp,bp,mc,bc,ng);'''
ALLOC_NEW = '''Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];Code mc=(mainN+ng-1)/ng,bc=(blockN+ng-1)/ng;
    b300_vmm::ContiguousStorage main_store,block_store;
    main_store.create(mainN,ng,0,"auth main");
    block_store.create(blockN,ng,int(main_store.mapped_units%size_t(ng)),"auth block");
    if(main_store.granularity!=block_store.granularity){std::cerr<<"VMM main/block granularity mismatch "<<main_store.granularity<<" != "<<block_store.granularity<<"\n";return 10;}
    size_t vmm_phys_min=~size_t(0),vmm_phys_max=0;
    for(int d=0;d<ng;++d){size_t bytes=main_store.segment_bytes[size_t(d)]+block_store.segment_bytes[size_t(d)];vmm_phys_min=std::min(vmm_phys_min,bytes);vmm_phys_max=std::max(vmm_phys_max,bytes);}
    if(vmm_phys_max-vmm_phys_min>main_store.granularity){std::cerr<<"VMM combined physical imbalance exceeds one granularity: "<<(vmm_phys_max-vmm_phys_min)<<" > "<<main_store.granularity<<"\n";return 10;}
    std::cerr<<"VMM32 combined: granularity_kib="<<double(main_store.granularity)/1024.0<<" physical_min_gib="<<double(vmm_phys_min)/(1ull<<30)<<" physical_max_gib="<<double(vmm_phys_max)/(1ull<<30)<<" imbalance_kib="<<double(vmm_phys_max-vmm_phys_min)/1024.0<<" main_padding_kib="<<double(main_store.mapped_bytes-main_store.logical_bytes)/1024.0<<" block_padding_kib="<<double(block_store.mapped_bytes-block_store.logical_bytes)/1024.0<<"\n";
    main_store.zero_local_segments();block_store.zero_local_segments();
    Count*main_base=main_store.base_as<Count>();Count*block_base=block_store.base_as<Count>();
    Count*mp[MAXGPU]{},*bp[MAXGPU]{};std::vector<Code>ml(ng),bl(ng);
    for(int d=0;d<ng;++d){ml[d]=std::min<Code>(mc,mainN-std::min<Code>(mainN,Code(d)*mc));bl[d]=std::min<Code>(bc,blockN-std::min<Code>(blockN,Code(d)*bc));mp[d]=main_base+Code(d)*mc;bp[d]=block_base+Code(d)*bc;}
    std::vector<DeviceCtx>ctx(ng);for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"set VMM symbols");ck(cudaMemcpyToSymbol(D_MAIN_VBASE,&main_base,sizeof(main_base)),"main VMM base");ck(cudaMemcpyToSymbol(D_BLOCK_VBASE,&block_base,sizeof(block_base)),"block VMM base");ctx[d].init(d,mod,mp,bp,mc,bc,ng);}'''

CLEANUP_OLD = '''for(auto&c:ctx)c.destroy();for(int d=0;d<ng;++d){cudaSetDevice(d);if(mp[d])cudaFree(mp[d]);if(bp[d])cudaFree(bp[d]);if(lowE[d])cudaFree(lowE[d]);if(lowO[d])cudaFree(lowO[d]);if(lowR[d])cudaFree(lowR[d]);if(highEM[d])cudaFree(highEM[d]);if(highEB[d])cudaFree(highEB[d]);if(highOM[d])cudaFree(highOM[d]);if(highOB[d])cudaFree(highOB[d]);}'''
CLEANUP_NEW = '''for(auto&c:ctx)c.destroy();for(int d=0;d<ng;++d){cudaSetDevice(d);if(lowE[d])cudaFree(lowE[d]);if(lowO[d])cudaFree(lowO[d]);if(lowR[d])cudaFree(lowR[d]);if(highEM[d])cudaFree(highEM[d]);if(highEB[d])cudaFree(highEB[d]);if(highOM[d])cudaFree(highOM[d]);if(highOB[d])cudaFree(highOB[d]);}main_store.destroy();block_store.destroy();'''

BACKEND_OLD = 'backend=gridfp-b300-hbm32-fullmate-dropN n='
BACKEND_NEW = 'backend=gridfp-b300-hbm32-fullmate-dropN-vmm n='


def once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected exactly one source match, got {count}')
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('src', type=Path)
    ap.add_argument('out', type=Path)
    args = ap.parse_args()
    text = args.src.read_text()
    text = once(text, INCLUDE_OLD, INCLUDE_NEW, 'include')
    text = once(text, SYMBOL_OLD, SYMBOL_NEW, 'VMM symbols')
    text = once(text, INTERVAL_OLD, INTERVAL_NEW, 'VMM interval planning')
    text = once(text, SHARD_OLD, SHARD_NEW, 'global shard access')
    text = once(text, INTERVAL_KERNEL_OLD, INTERVAL_KERNEL_NEW, 'VMM interval kernel')
    text = once(text, ALLOC_OLD, ALLOC_NEW, 'authoritative allocation')
    text = once(text, CLEANUP_OLD, CLEANUP_NEW, 'authoritative cleanup')
    text = once(text, BACKEND_OLD, BACKEND_NEW, 'backend label')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f'generated {args.out} from {args.src} vmm_contiguous_authoritative=1 direct_global_index=1 logical_shard_views=1 shard_free_interval_io=1 count_bytes=4 runtime_physical_balance_guard=1')


if __name__ == '__main__':
    main()
