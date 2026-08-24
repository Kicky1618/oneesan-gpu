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
#include "../gridfp/ramstream32_b300_dual_tile_kernels.cuh"
#include "../gridfp/ramstream32_b300_dual_tile_shuffle.cuh"
#include "../gridfp/ramstream32_b300_dual_tile_precomputed_w28.cuh"

struct DTMainLoc { uint32_t bid=0,hr=0,lr=0; int hi=0,lo=0; Code high_index=0,low_index=0; };

static DTMainLoc dt_main_loc_host(
    MateID m, const StorageFactorHost& f, const StorageLayout& l,
    const B300DualTileHost& z
) {
    constexpr uint32_t LM=(1u<<(2*LOW_LUT_K))-1u;
    constexpr uint32_t HM=(1u<<(2*HIGH_LUT_K))-1u;
    uint32_t lc=uint32_t(m)&LM;
    uint32_t hc=uint32_t((m>>(2*(LOW_LUT_K+1)))&HM);
    int he=seg_end_height_host(hc,HIGH_LUT_K),cv=int(mget(m,LOW_LUT_K));
    uint32_t bid=uint32_t(3*he+cv);
    const auto& b=l.main_blocks[bid];
    uint32_t lp=f.low_packed_rank[lc],hp=f.high_packed_rank[hc];
    if(!b.valid||lp==0xffffffffu||hp==0xffffffffu)std::exit(530);
    uint32_t lr=lp>>LOW_LUT_K,hr=hp>>HIGH_LUT_K;
    uint32_t hai=f.high_all_off[b.he]+hr,lai=f.low_all_off[b.hs]+lr;
    int hi=z.high.high_owner[hai],lo=z.low_owner[lai];
    uint32_t hl=z.high.high_local[hai],ll=z.low_local[lai];
    size_t pix=(size_t(hi)*z.ngpu+lo)*l.main_blocks.size()+bid;
    Code k=z.pair_main_off[pix]+Code(hl)*z.low_count[lo][b.hs]+ll;
    DTMainLoc x; x.bid=bid;x.hr=hr;x.lr=lr;x.hi=hi;x.lo=lo;
    x.high_index=z.main_slot_base[hi][lo]+k;
    x.low_index=z.main_slot_base[lo][hi]+k;
    return x;
}

struct DTGpu {
    int dev=-1;
    Count *main=nullptr,*block=nullptr,*scratch=nullptr;
    Code main_n=0,block_n=0,scratch_n=0;
    BidescMaskDeviceTables mask;
    B300DirectStorageDeviceTables storage;
    B300DualTileDeviceTables dual;
    B300SparseActionsDeviceTables sparse;

    void allocate(Code mn,Code bn,Code sn){
        main_n=mn;block_n=bn;scratch_n=sn;
        ck(cudaSetDevice(dev),"dual alloc device");
        if(mn)ck(cudaMalloc(&main,size_t(mn)*sizeof(Count)),"dual main arena");
        if(bn)ck(cudaMalloc(&block,size_t(bn)*sizeof(Count)),"dual block arena");
        if(sn)ck(cudaMalloc(&scratch,size_t(sn)*sizeof(Count)),"dual shuffle scratch");
        if(mn)ck(cudaMemset(main,0,size_t(mn)*sizeof(Count)),"dual zero main");
        if(bn)ck(cudaMemset(block,0,size_t(bn)*sizeof(Count)),"dual zero block");
    }
    void release(){
        ck(cudaSetDevice(dev),"dual release device");
        sparse.release();dual.release();storage.release();mask.release();
        if(scratch)cudaFree(scratch);if(block)cudaFree(block);if(main)cudaFree(main);
        main=block=scratch=nullptr;
    }
};

static void dt_enable_peer_mesh(int ngpu){
    for(int a=0;a<ngpu;++a){
        ck(cudaSetDevice(a),"dual peer source");
        for(int b=0;b<ngpu;++b)if(a!=b){
            int can=0;ck(cudaDeviceCanAccessPeer(&can,a,b),"dual peer query");
            if(!can){std::cerr<<"dual tile requires peer access "<<a<<" -> "<<b<<'\n';std::exit(531);}
            cudaError_t e=cudaDeviceEnablePeerAccess(b,0);
            if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else ck(e,"dual enable peer");
        }
    }
}

static void dt_install_layout(
    const StorageLayout& l,const B300DualTileHost& z,int self,
    Count** mp,Count** bp
){
    int mn=int(l.main_blocks.size()),bn=int(l.block_blocks.size());
    ck(cudaMemcpyToSymbol(D_DR_MAIN_BLOCKS,l.main_blocks.data(),l.main_blocks.size()*sizeof(StorageBlock)),"dual main blocks");
    ck(cudaMemcpyToSymbol(D_DR_BLOCK_BLOCKS,l.block_blocks.data(),l.block_blocks.size()*sizeof(StorageBlock)),"dual block blocks");
    ck(cudaMemcpyToSymbol(D_DR_MAIN_NBLOCKS,&mn,sizeof(mn)),"dual main nblocks");
    ck(cudaMemcpyToSymbol(D_DR_BLOCK_NBLOCKS,&bn,sizeof(bn)),"dual block nblocks");
    ck(cudaMemcpyToSymbol(D_DR_SELF,&self,sizeof(self)),"dual self");
    ck(cudaMemcpyToSymbol(D_NGPU,&z.ngpu,sizeof(z.ngpu)),"dual ngpu");
    ck(cudaMemcpyToSymbol(D_MAIN_PTR,mp,sizeof(Count*)*MAXGPU),"dual main ptrs");
    ck(cudaMemcpyToSymbol(D_BLOCK_PTR,bp,sizeof(Count*)*MAXGPU),"dual block ptrs");
}

static void dt_install_gpu(
    DTGpu& c,const StorageFactorHost& f,const StorageLayout& l,
    const B300DualTileHost& z,const B300SparseActionsHost& s,Count mod,
    Count**mp,Count**bp
){
    ck(cudaSetDevice(c.dev),"dual install device");
    c.mask.install(G_FACTOR);
    c.storage.install(f);
    c.dual.install(z,l,c.dev);
    c.sparse.install(s);
    dt_install_layout(l,z,c.dev,mp,bp);
    ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"dual modulus");
}

static void dt_run_high_local(const B300SparseActionsHost&s,int ngpu,int threads){
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){
        uint32_t on=b300_sparse_high_orbit_count(s,p),cn=b300_sparse_high_closure_count(s,p);
        for(int g=0;g<ngpu;++g){
            ck(cudaSetDevice(g),"dual high device");
            if(on)b300_dt_high_orbit_kernel<<<on,threads>>>(p);
            if(cn)b300_dt_high_closure_kernel<<<cn,threads>>>(p);
            ck(cudaGetLastError(),"dual high launch");
        }
    }
    b300_dt_sync_all(ngpu,"dual high sync");
}
static void dt_run_low_local(const B300SparseActionsHost&s,int ngpu,int threads){
    for(int p=LOW_LUT_K;p>=1;--p){
        uint32_t on=b300_sparse_low_orbit_count(s,p),cn=b300_sparse_low_closure_count(s,p);
        for(int g=0;g<ngpu;++g){
            ck(cudaSetDevice(g),"dual low device");
            if(on)b300_dt_low_orbit_kernel<<<on,threads>>>(p);
            if(cn)b300_dt_low_closure_kernel<<<cn,threads>>>(p);
            ck(cudaGetLastError(),"dual low launch");
        }
    }
    b300_dt_sync_all(ngpu,"dual low sync");
}

static double dt_gib(long double x){return double(x/(1ull<<30));}
static double dt_mib(long double x){return double(x/(1ull<<20));}
static double dt_tib(long double x){return double(x/(1ull<<40));}

int main(int argc,char**argv){
    int n=argc>1?std::atoi(argv[1]):TARGET_W-1;
    Count mod=argc>2?Count(std::strtoul(argv[2],nullptr,10)):4294967291u;
    int ngpu=argc>3?std::max(1,std::atoi(argv[3])):8;
    int threads=argc>4?std::max(32,std::atoi(argv[4])):256;
    bool plan=argc>5&&std::strcmp(argv[5],"--plan-only")==0;
    int chunk_mib=argc>6?std::max(64,std::atoi(argv[6])):512;
    int W=n+1;
    if(W!=TARGET_W||W>MAXW||n<2||ngpu>MAXGPU)return 1;
    if constexpr(LOW_LUT_K+HIGH_LUT_K+1!=TARGET_W)return 1;

    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost f=build_storage_factor_tables(G_FACTOR);
    StorageLayout l=build_storage_layout(f);
    LowDescHost ld=build_low_descriptors(f,l);HighDescHost hd=build_high_descriptors(f,l);
    LowOrbitHost lo=build_cpu_low_orbit(f,l,ld);HighOrbitHost ho=build_high_orbit(f,l);
    B300SparseActionsHost sparse=build_b300_sparse_actions(l,ld,lo,hd,ho);
    B300DualTileHost dual=build_b300_dual_tile_layout_w28_precomputed(f,l,ngpu);

    Code chunk_elems=Code(chunk_mib)*(1ull<<20)/sizeof(Count);
    long double chunk_bytes=(long double)chunk_elems*sizeof(Count);
    long double mask_bytes=(long double)(G_FACTOR.low_mask_codes.size()+G_FACTOR.low_mask_off.size()+G_FACTOR.high_mask_codes.size()+G_FACTOR.high_mask_off.size())*sizeof(uint32_t);
    long double storage_bytes=(long double)(f.low_all_codes.size()+f.high_all_codes.size()+f.low_mask_begin.size()+f.high_mask_begin.size())*sizeof(uint32_t);
    long double sparse_bytes=(long double)sparse.bytes();
    long double common_dual=(long double)dual.high.high_owner.size()*sizeof(uint8_t)+(long double)dual.high.high_local.size()*sizeof(uint32_t)
        +(long double)dual.low_owner.size()*sizeof(uint8_t)+(long double)dual.low_local.size()*sizeof(uint32_t)
        +(long double)(dual.pair_main_off.size()+dual.pair_block_off.size())*sizeof(Code);
    long double max_need=0,min_arena=1e100L,max_arena=0;
    std::array<long double,MAXGPU> need{};
    for(int g=0;g<ngpu;++g){
        long double local_lists=(long double)(dual.high.owned_rows[g].size()+dual.owned_low_cols[g].size())*sizeof(uint32_t);
        long double arena=(long double)(dual.main_count[g]+dual.block_count[g])*sizeof(Count);
        min_arena=std::min(min_arena,arena);max_arena=std::max(max_arena,arena);
        need[g]=arena+chunk_bytes+mask_bytes+storage_bytes+sparse_bytes+common_dual+local_lists;
        max_need=std::max(max_need,need[g]);
    }

    long double l2h=0,h2l=0;
    std::array<long double,MAXGPU> l2h_send{},l2h_recv{},h2l_send{},h2l_recv{};
    for(int hi=0;hi<ngpu;++hi)for(int lo_=0;lo_<ngpu;++lo_)if(hi!=lo_){
        long double m=(long double)dual.pair_main_size[hi][lo_]*sizeof(Count);
        long double b=(long double)dual.pair_block_size[hi][lo_]*sizeof(Count);
        l2h+=m+b;h2l+=m;
        l2h_send[lo_]+=m+b;l2h_recv[hi]+=m+b;
        h2l_send[hi]+=m;h2l_recv[lo_]+=m;
    }
    long double residue_bytes=(long double)W*l2h+(long double)(W-1)*h2l;
    long double max_port=0;
    for(int g=0;g<ngpu;++g){
        long double x=(long double)W*std::max(l2h_send[g],l2h_recv[g])
                     +(long double)(W-1)*std::max(h2l_send[g],h2l_recv[g]);
        max_port=std::max(max_port,x);
    }

    DTMainLoc init=dt_main_loc_host(MateID(R)<<(2*(W-1)),f,l,dual);
    DTMainLoc answer=dt_main_loc_host(MateID(R),f,l,dual);

    if(plan){
        std::cout<<std::fixed<<std::setprecision(6)
            <<"backend=gridfp-b300-hbm32-dual-tile-sparse-plan n="<<n<<" gpus="<<ngpu
            <<" ownership_policy="<<((W==28&&ngpu==8)?"w28-popcount-milp-tight":"lpt-fallback")
            <<" runtime_groups=0 kernel_peer_access=0 remote_system_atomics=0"
            <<" chunk_mib="<<chunk_mib
            <<" pairslot_arena_min_gib="<<dt_gib(min_arena)
            <<" pairslot_arena_max_gib="<<dt_gib(max_arena)
            <<" sparse_actions_mib="<<dt_mib(sparse_bytes)
            <<" common_metadata_mib="<<dt_mib(mask_bytes+storage_bytes+common_dual)
            <<" max_need_gib="<<dt_gib(max_need)
            <<" headroom_288GB_gib="<<dt_gib(288.0e9L-max_need)
            <<" headroom_279GB_gib="<<dt_gib(279.0e9L-max_need)
            <<" low_to_high_offgpu_tib_per_row="<<dt_tib(l2h)
            <<" high_to_low_main_offgpu_tib_per_row="<<dt_tib(h2l)
            <<" low_to_high_count="<<W<<" high_to_low_count="<<(W-1)
            <<" offgpu_tib_per_residue="<<dt_tib(residue_bytes)
            <<" max_gpu_port_tib_per_residue="<<dt_tib(max_port)
            <<" ideal_1p8TBs_port_seconds_per_residue="<<double(max_port/1.8e12L)
            <<" final_shuffle_elided=1\n";
        for(int g=0;g<ngpu;++g)std::cout
            <<"dual_gpu="<<g<<" main_arena_gib="<<dt_gib((long double)dual.main_count[g]*sizeof(Count))
            <<" block_arena_gib="<<dt_gib((long double)dual.block_count[g]*sizeof(Count))
            <<" total_need_gib="<<dt_gib(need[g])
            <<" l2h_send_gib="<<dt_gib(l2h_send[g])<<" l2h_recv_gib="<<dt_gib(l2h_recv[g])
            <<" h2l_send_gib="<<dt_gib(h2l_send[g])<<" h2l_recv_gib="<<dt_gib(h2l_recv[g])<<'\n';
        return 0;
    }

    int visible=0;ck(cudaGetDeviceCount(&visible),"dual device count");
    if(visible<ngpu){std::cerr<<"need "<<ngpu<<" GPUs visible="<<visible<<'\n';return 3;}
    std::vector<std::unique_ptr<DTGpu>> gpu;
    gpu.reserve(ngpu);
    for(int g=0;g<ngpu;++g){auto c=std::make_unique<DTGpu>();c->dev=g;c->allocate(dual.main_count[g],dual.block_count[g],chunk_elems);gpu.push_back(std::move(c));}
    dt_enable_peer_mesh(ngpu);
    std::array<Count*,MAXGPU>mp{},bp{},sp{};
    for(int g=0;g<ngpu;++g){mp[g]=gpu[g]->main;bp[g]=gpu[g]->block;sp[g]=gpu[g]->scratch;}
    for(int g=0;g<ngpu;++g)dt_install_gpu(*gpu[g],f,l,dual,sparse,mod,mp.data(),bp.data());

    // Dense construction metadata is no longer needed after sparse streams and
    // the physical dual-tile maps have been installed.
    ld.main_desc.clear();ld.block_desc.clear();hd.main_desc.clear();hd.block_desc.clear();
    lo.rec.clear();ho.rec.clear();

    Count one=1;
    ck(cudaSetDevice(init.lo),"dual init device");
    ck(cudaMemcpy(mp[init.lo]+init.low_index,&one,sizeof(one),cudaMemcpyHostToDevice),"dual init");

    B300DualShuffleStats sh{};
    double high_s=0,low_s=0,shuffle_s=0,zero_s=0;
    auto wall0=std::chrono::steady_clock::now();
    for(int row=0;row<W;++row){
        auto t=std::chrono::steady_clock::now();
        dt_run_high_local(sparse,ngpu,threads);
        high_s+=ram_seconds_since(t);

        t=std::chrono::steady_clock::now();
        b300_dt_low_to_high(dual,mp.data(),bp.data(),sp.data(),chunk_elems,&sh);
        shuffle_s+=ram_seconds_since(t);

        t=std::chrono::steady_clock::now();
        dt_run_low_local(sparse,ngpu,threads);
        low_s+=ram_seconds_since(t);

        if(row+1<W){
            t=std::chrono::steady_clock::now();
            b300_dt_zero_block_arenas(dual,bp.data());
            zero_s+=ram_seconds_since(t);
            t=std::chrono::steady_clock::now();
            b300_dt_high_to_low_main(dual,mp.data(),sp.data(),chunk_elems,&sh);
            shuffle_s+=ram_seconds_since(t);
        }
        std::cerr<<"dual-tile row "<<row+1<<'/'<<W<<'\n';
    }
    b300_dt_sync_all(ngpu,"dual final sync");
    double wall_s=ram_seconds_since(wall0);

    Count ans=0;
    ck(cudaSetDevice(answer.hi),"dual answer device");
    ck(cudaMemcpy(&ans,mp[answer.hi]+answer.high_index,sizeof(ans),cudaMemcpyDeviceToHost),"dual answer");
    std::cout<<"backend=gridfp-b300-hbm32-dual-tile-sparse n="<<n
             <<" residue="<<ans<<" modulus="<<mod<<" gpus="<<ngpu
             <<" ownership_policy="<<((W==28&&ngpu==8)?"w28-popcount-milp-tight":"lpt-fallback")
             <<" kernel_peer_access=0 remote_system_atomics=0"
             <<" high_s="<<high_s<<" low_s="<<low_s<<" shuffle_s="<<shuffle_s<<" zero_s="<<zero_s
             <<" shuffle_tib="<<dt_tib(sh.main_bytes+sh.block_bytes)
             <<" shuffle_rounds="<<sh.rounds<<" chunk_steps="<<sh.chunk_barriers
             <<" wall_s="<<wall_s<<'\n';
    for(auto&c:gpu)c->release();
    return 0;
}
