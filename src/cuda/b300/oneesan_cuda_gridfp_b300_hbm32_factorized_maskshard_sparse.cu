#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../gridfp/oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../gridfp/ramstream32_high_orbit.cuh"
#include "../gridfp/ramstream32_cpu_low_inplace.hpp"
#include "../gridfp/ramstream32_b300_direct_maskshard.cuh"
#include "../gridfp/ramstream32_b300_direct_system_atomic.cuh"

struct MaskLoc { uint32_t bid=0,hr=0,lr=0; };
static MaskLoc locate_mask(Code rank,const std::vector<StorageBlock>& bs){
    for(uint32_t b=0;b<bs.size();++b){const auto&x=bs[b];Code n=Code(x.rows)*x.cols;
        if(x.valid&&rank>=x.off&&rank<x.off+n){Code r=rank-x.off;return{b,uint32_t(r/x.cols),uint32_t(r%x.cols)};}}
    std::cerr<<"maskshard rank not found "<<rank<<"\n";std::exit(470);
}
static Code local_mask_index(const MaskLoc&x,const B300DirectMaskShardHost&s,
                             const StorageFactorHost&f,const StorageBlock&b,bool blocked){
    uint32_t ai=f.high_all_off[b.he]+x.hr;int g=s.high_owner[ai];uint32_t lh=s.high_local[ai];
    Code base=blocked?s.block_off[g][x.bid]:s.main_off[g][x.bid];return base+Code(lh)*b.cols+x.lr;
}

struct MaskGpu{
    int dev=-1;Count*main=nullptr,*block=nullptr;
    BidescMaskDeviceTables mask;B300DirectStorageDeviceTables storage;
    B300DirectMaskMapDeviceTables map;B300DirectSparseDeviceTables sparse;
    void alloc(Code m,Code b){ck(cudaSetDevice(dev),"mask alloc dev");
        if(m)ck(cudaMalloc(&main,size_t(m)*sizeof(Count)),"mask main");
        if(b)ck(cudaMalloc(&block,size_t(b)*sizeof(Count)),"mask block");
        if(m)ck(cudaMemset(main,0,size_t(m)*sizeof(Count)),"mask zero main");
        if(b)ck(cudaMemset(block,0,size_t(b)*sizeof(Count)),"mask zero block");}
    void install(const StorageFactorHost&f,const StorageLayout&l,const B300DirectMaskShardHost&s,
                 const B300SparseActionsHost&a,const B300DirectSparsePartitionHost&p,Count mod,
                 Count**mp,Count**bp){ck(cudaSetDevice(dev),"mask install dev");mask.install(G_FACTOR);
        storage.install(f);map.install(s,dev);sparse.install(a,p,dev);b300_mask_install_layout(l,s,dev,mp,bp);
        ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"mask mod");}
    void release(){ck(cudaSetDevice(dev),"mask release");sparse.release();map.release();storage.release();mask.release();
        if(main)cudaFree(main);if(block)cudaFree(block);main=block=nullptr;}
};

static void peer_atomic_mesh(int n){for(int a=0;a<n;++a){ck(cudaSetDevice(a),"peer src");for(int b=0;b<n;++b)if(a!=b){
    int can=0,atom=0;ck(cudaDeviceCanAccessPeer(&can,a,b),"peer access query");
    ck(cudaDeviceGetP2PAttribute(&atom,cudaDevP2PAttrNativeAtomicSupported,a,b),"peer atomic query");
    if(!can||!atom){std::cerr<<"maskshard requires peer native atomics "<<a<<" -> "<<b<<"\n";std::exit(471);}
    cudaError_t e=cudaDeviceEnablePeerAccess(b,0);if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else ck(e,"enable peer");}}}
static void sync_all(int n,const char*w){for(int g=0;g<n;++g){ck(cudaSetDevice(g),w);ck(cudaDeviceSynchronize(),w);}}

static void run_high(const B300DirectSparsePartitionHost&p,int ng,int edge,int th){
    for(int g=0;g<ng;++g){ck(cudaSetDevice(g),"high orbit dev");uint32_t n=b300_direct_high_orbit_count(p,g,edge);
        if(n)b300_mask_high_orbit_kernel<<<n,th>>>(edge);ck(cudaGetLastError(),"high orbit");}sync_all(ng,"high orbit sync");
    for(int g=0;g<ng;++g){ck(cudaSetDevice(g),"high closure dev");uint32_t n=b300_direct_high_closure_count(p,g,edge);
        if(n)b300_mask_high_closure_scoped_kernel<<<n,th>>>(edge);ck(cudaGetLastError(),"high closure");}sync_all(ng,"high closure sync");
}
static void run_low(const B300SparseActionsHost&s,int ng,int edge,int th){
    uint32_t on=b300_sparse_low_orbit_count(s,edge);for(int g=0;g<ng;++g){ck(cudaSetDevice(g),"low orbit dev");
        if(on)b300_mask_low_orbit_kernel<<<on,th>>>(edge);ck(cudaGetLastError(),"low orbit");}sync_all(ng,"low orbit sync");
    uint32_t cn=b300_sparse_low_closure_count(s,edge);for(int g=0;g<ng;++g){ck(cudaSetDevice(g),"low closure dev");
        if(cn)b300_mask_low_closure_kernel<<<cn,th>>>(edge);ck(cudaGetLastError(),"low closure");}sync_all(ng,"low closure sync");
}
static double GiB(long double x){return double(x/(1ull<<30));}static double MiB(long double x){return double(x/(1ull<<20));}

int main(int argc,char**argv){
    int n=argc>1?std::atoi(argv[1]):TARGET_W-1;Count mod=argc>2?Count(std::strtoul(argv[2],nullptr,10)):4294967291u;
    int ng=argc>3?std::max(1,std::atoi(argv[3])):8,th=argc>4?std::max(32,std::atoi(argv[4])):256;
    bool plan=argc>5&&std::strcmp(argv[5],"--plan-only")==0;int W=n+1;
    if(W!=TARGET_W||W>MAXW||n<2||ng>MAXGPU)return 1;if constexpr(LOW_LUT_K+HIGH_LUT_K+1!=TARGET_W)return 1;

    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost f=build_storage_factor_tables(G_FACTOR);StorageLayout l=build_storage_layout(f);
    LowDescHost ld=build_low_descriptors(f,l);HighDescHost hd=build_high_descriptors(f,l);
    LowOrbitHost lo=build_cpu_low_orbit(f,l,ld);HighOrbitHost ho=build_high_orbit(f,l);
    B300SparseActionsHost sparse=build_b300_sparse_actions(l,ld,lo,hd,ho);
    B300DirectMaskShardHost shard=build_b300_direct_mask_shards(f,l,ng);
    B300DirectSparsePartitionHost part=b300_direct_partition_high_by_mask(sparse,f,l,shard);

    Code ip=storage_rank_main_host(MateID(R)<<(2*(W-1)),f,l),ap=storage_rank_main_host(MateID(R),f,l);
    MaskLoc il=locate_mask(ip,l.main_blocks),al=locate_mask(ap,l.main_blocks);
    long double maskb=(long double)(G_FACTOR.low_mask_codes.size()+G_FACTOR.low_mask_off.size()+G_FACTOR.high_mask_codes.size()+G_FACTOR.high_mask_off.size())*4;
    long double storeb=(long double)(f.low_all_codes.size()+f.high_all_codes.size()+f.low_mask_begin.size()+f.high_mask_begin.size())*4;
    long double mapcommon=(long double)shard.high_owner.size()+ (long double)shard.high_local.size()*4;
    long double lows=(long double)sparse.low_orbit.size()*sizeof(B300SparseOrbitOp)+(long double)sparse.low_closure.size()*8
                    +(long double)(sparse.low_orbit_off.size()+sparse.low_closure_off.size())*4;
    long double layoutb=(long double)(l.main_blocks.size()+l.block_blocks.size())*sizeof(StorageBlock)+sizeof(shard.main_off)+sizeof(shard.block_off);
    std::array<long double,MAXGPU> need{},hs{};long double maxneed=0,minauth=1e100L,maxauth=0;
    for(int g=0;g<ng;++g){hs[g]=(long double)part.high_orbit[g].size()*sizeof(B300SparseOrbitOp)+(long double)part.high_closure[g].size()*8
        +(long double)(part.high_orbit_off[g].size()+part.high_closure_off[g].size())*4;
        long double auth=(long double)(shard.main_count[g]+shard.block_count[g])*4;minauth=std::min(minauth,auth);maxauth=std::max(maxauth,auth);
        need[g]=auth+maskb+storeb+mapcommon+(long double)shard.owned_rows[g].size()*4+lows+hs[g]+layoutb;maxneed=std::max(maxneed,need[g]);}
    long double cgs=(long double)(4*l.main_size+2*l.block_size)*4*W,rp=(long double)(2*l.main_size+l.block_size)*4*W;
    if(plan){std::cout<<std::fixed<<std::setprecision(6)
        <<"backend=gridfp-b300-hbm32-factorized-maskshard-sparse-plan n="<<n<<" gpus="<<ng
        <<" auth_total_gib="<<GiB((long double)(l.main_size+l.block_size)*4)<<" auth_min_gib="<<GiB(minauth)<<" auth_max_gib="<<GiB(maxauth)
        <<" auth_imbalance="<<double(maxauth/minauth)<<" runtime_groups=0 scratch_gib=0.000 low_p2p_bytes=0"
        <<" gather_scatter_tib_per_residue=0.000 host_pcie_tib_per_residue=0.000 eliminated_canonical_gs_tib="<<double(cgs/(1ull<<40))
        <<" eliminated_ramstream_pcie_tib="<<double(rp/(1ull<<40))<<" low_sparse_replicated_mib="<<MiB(lows)
        <<" common_meta_mib="<<MiB(maskb+storeb+mapcommon+layoutb)<<" max_need_gib="<<GiB(maxneed)
        <<" headroom_288GB_gib="<<GiB(288.0e9L-maxneed)<<" headroom_279GB_gib="<<GiB(279.0e9L-maxneed)<<"\n";
        for(int g=0;g<ng;++g)std::cout<<"mask_gpu="<<g<<" auth_gib="<<GiB((long double)(shard.main_count[g]+shard.block_count[g])*4)
            <<" owned_high_rows="<<shard.owned_rows[g].size()<<" high_sparse_mib="<<MiB(hs[g])<<" need_gib="<<GiB(need[g])<<"\n";return 0;}

    int visible=0;ck(cudaGetDeviceCount(&visible),"device count");if(visible<ng){std::cerr<<"need "<<ng<<" GPUs visible="<<visible<<"\n";return 3;}
    ld.main_desc.clear();ld.block_desc.clear();hd.main_desc.clear();hd.block_desc.clear();lo.rec.clear();ho.rec.clear();
    G_FACTOR.low_packed_rank.clear();G_FACTOR.low_packed_rank.shrink_to_fit();G_FACTOR.high_packed_rank.clear();G_FACTOR.high_packed_rank.shrink_to_fit();
    f.low_packed_rank.clear();f.low_packed_rank.shrink_to_fit();f.high_packed_rank.clear();f.high_packed_rank.shrink_to_fit();

    std::vector<std::unique_ptr<MaskGpu>> gpu;for(int g=0;g<ng;++g){auto c=std::make_unique<MaskGpu>();c->dev=g;c->alloc(shard.main_count[g],shard.block_count[g]);gpu.push_back(std::move(c));}
    peer_atomic_mesh(ng);std::array<Count*,MAXGPU>mp{},bp{};for(int g=0;g<ng;++g){mp[g]=gpu[g]->main;bp[g]=gpu[g]->block;}
    for(int g=0;g<ng;++g)gpu[g]->install(f,l,shard,sparse,part,mod,mp.data(),bp.data());
    uint32_t iai=f.high_all_off[l.main_blocks[il.bid].he]+il.hr;int io=shard.high_owner[iai];Code ix=local_mask_index(il,shard,f,l.main_blocks[il.bid],false);Count one=1;
    ck(cudaSetDevice(io),"init dev");ck(cudaMemcpy(gpu[io]->main+ix,&one,4,cudaMemcpyHostToDevice),"init");

    auto t0=std::chrono::steady_clock::now();for(int row=0;row<W;++row){for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p)run_high(part,ng,p,th);
        for(int p=LOW_LUT_K;p>=1;--p)run_low(sparse,ng,p,th);std::cerr<<"maskshard row "<<row+1<<'/'<<W<<"\n";}sync_all(ng,"final sync");double sec=ram_seconds_since(t0);
    uint32_t aai=f.high_all_off[l.main_blocks[al.bid].he]+al.hr;int ao=shard.high_owner[aai];Code ax=local_mask_index(al,shard,f,l.main_blocks[al.bid],false);Count ans=0;
    ck(cudaSetDevice(ao),"answer dev");ck(cudaMemcpy(&ans,gpu[ao]->main+ax,4,cudaMemcpyDeviceToHost),"answer");
    std::cout<<"backend=gridfp-b300-hbm32-factorized-maskshard-sparse n="<<n<<" residue="<<ans<<" modulus="<<mod<<" gpus="<<ng
             <<" groups=0 low_p2p_bytes=0 bulk_transfer_bytes=0 wall_s="<<sec<<"\n";
    for(auto&c:gpu)c->release();return 0;
}
