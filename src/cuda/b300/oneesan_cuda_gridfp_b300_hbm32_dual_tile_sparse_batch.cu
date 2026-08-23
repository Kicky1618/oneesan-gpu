// Persistent multi-modulus wrapper around the dual-tile backend.
// Reuse all factor tables, sparse actions, GPU metadata and the ~1.94 TiB HBM
// allocation across CRT primes; only the state arrays and D_MOD change.
#define main(...) b300_dual_tile_single_main(__VA_ARGS__)
#include "oneesan_cuda_gridfp_b300_hbm32_dual_tile_sparse.cu"
#undef main
#include "../gridfp/ramstream32_b300_dual_tile_peer_swap.cuh"

static int dt_env_int(const char* name,int fallback,int lo){
    const char* s=std::getenv(name);if(!s||!*s)return fallback;
    return std::max(lo,std::atoi(s));
}
static bool dt_env_peer_shuffle(){
    const char* s=std::getenv("ONEESAN_DUAL_SHUFFLE");
    return s && (std::strcmp(s,"peer-kernel")==0 || std::strcmp(s,"peer")==0);
}

static void dt_batch_zero_state(
    const B300DualTileHost& dual,
    const std::vector<std::unique_ptr<DTGpu>>& gpu
){
    for(int g=0;g<dual.ngpu;++g){
        ck(cudaSetDevice(g),"dual batch zero device");
        if(dual.main_count[g])
            ck(cudaMemsetAsync(gpu[g]->main,0,size_t(dual.main_count[g])*sizeof(Count)),"dual batch zero main");
        if(dual.block_count[g])
            ck(cudaMemsetAsync(gpu[g]->block,0,size_t(dual.block_count[g])*sizeof(Count)),"dual batch zero block");
    }
    b300_dt_sync_all(dual.ngpu,"dual batch zero sync");
}

static void dt_batch_set_mod(int ngpu,Count mod){
    for(int g=0;g<ngpu;++g){
        ck(cudaSetDevice(g),"dual batch modulus device");
        ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"dual batch modulus");
    }
}

int main(int argc,char**argv){
    // Match scripts/solve/solve_b300_exact_batch.py:
    //   binary n target_mib max_window gpus p1 p2 ...
    // target_mib/max_window are compatibility arguments and are deliberately
    // ignored by the full-HBM dual-tile backend.
    if(argc<6){
        std::cerr<<"usage: "<<argv[0]<<" n target_mib max_window gpus prime...\n";
        return 2;
    }
    int n=std::atoi(argv[1]);
    int ngpu=std::max(1,std::atoi(argv[4]));
    int threads=dt_env_int("ONEESAN_DUAL_THREADS",256,32);
    int chunk_mib=dt_env_int("ONEESAN_DUAL_CHUNK_MIB",512,64);
    bool peer_shuffle=dt_env_peer_shuffle();
    int W=n+1;
    if(W!=TARGET_W||W>MAXW||n<2||ngpu>MAXGPU)return 1;
    if constexpr(LOW_LUT_K+HIGH_LUT_K+1!=TARGET_W)return 1;

    std::vector<Count> mods;
    mods.reserve(size_t(argc-5));
    for(int i=5;i<argc;++i){
        unsigned long x=std::strtoul(argv[i],nullptr,10);
        if(x<3||x>0xfffffffful){std::cerr<<"invalid modulus "<<argv[i]<<'\n';return 2;}
        mods.push_back(Count(x));
    }

    auto setup0=std::chrono::steady_clock::now();
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost f=build_storage_factor_tables(G_FACTOR);
    StorageLayout l=build_storage_layout(f);
    LowDescHost ld=build_low_descriptors(f,l);HighDescHost hd=build_high_descriptors(f,l);
    LowOrbitHost lo=build_cpu_low_orbit(f,l,ld);HighOrbitHost ho=build_high_orbit(f,l);
    B300SparseActionsHost sparse=build_b300_sparse_actions(l,ld,lo,hd,ho);
    B300DualTileHost dual=build_b300_dual_tile_layout_w28_precomputed(f,l,ngpu);
    Code chunk_elems=Code(chunk_mib)*(1ull<<20)/sizeof(Count);
    Code scratch_elems=peer_shuffle?Code(0):chunk_elems;

    DTMainLoc init=dt_main_loc_host(MateID(R)<<(2*(W-1)),f,l,dual);
    DTMainLoc answer=dt_main_loc_host(MateID(R),f,l,dual);

    int visible=0;ck(cudaGetDeviceCount(&visible),"dual batch device count");
    if(visible<ngpu){std::cerr<<"need "<<ngpu<<" GPUs visible="<<visible<<'\n';return 3;}

    std::vector<std::unique_ptr<DTGpu>> gpu;
    gpu.reserve(ngpu);
    for(int g=0;g<ngpu;++g){
        auto c=std::make_unique<DTGpu>();c->dev=g;
        c->allocate(dual.main_count[g],dual.block_count[g],scratch_elems);
        gpu.push_back(std::move(c));
    }
    dt_enable_peer_mesh(ngpu);
    std::array<Count*,MAXGPU>mp{},bp{},sp{};
    for(int g=0;g<ngpu;++g){mp[g]=gpu[g]->main;bp[g]=gpu[g]->block;sp[g]=gpu[g]->scratch;}
    // Install metadata once. D_MOD is replaced before every residue below.
    for(int g=0;g<ngpu;++g)dt_install_gpu(*gpu[g],f,l,dual,sparse,mods.front(),mp.data(),bp.data());

    B300DualShuffleContext copy_ctx;
    B300DualPeerSwapContext peer_ctx;
    if(peer_shuffle) peer_ctx.init(ngpu); else copy_ctx.init(ngpu);

    ld.main_desc.clear();ld.main_desc.shrink_to_fit();
    ld.block_desc.clear();ld.block_desc.shrink_to_fit();
    hd.main_desc.clear();hd.main_desc.shrink_to_fit();
    hd.block_desc.clear();hd.block_desc.shrink_to_fit();
    lo.rec.clear();lo.rec.shrink_to_fit();ho.rec.clear();ho.rec.shrink_to_fit();
    G_FACTOR.low_packed_rank.clear();G_FACTOR.low_packed_rank.shrink_to_fit();
    G_FACTOR.high_packed_rank.clear();G_FACTOR.high_packed_rank.shrink_to_fit();
    f.low_packed_rank.clear();f.low_packed_rank.shrink_to_fit();
    f.high_packed_rank.clear();f.high_packed_rank.shrink_to_fit();

    double setup_s=ram_seconds_since(setup0);
    std::cerr<<"dual-batch setup_s="<<setup_s<<" moduli="<<mods.size()
             <<" gpus="<<ngpu<<" threads="<<threads<<" chunk_mib="<<chunk_mib
             <<" shuffle="<<(peer_shuffle?"peer-kernel":"copy-pipeline")<<'\n';

    double all_wall=0.0;
    for(size_t mi=0;mi<mods.size();++mi){
        Count mod=mods[mi];
        auto residue0=std::chrono::steady_clock::now();
        dt_batch_set_mod(ngpu,mod);
        dt_batch_zero_state(dual,gpu);

        Count one=1;
        ck(cudaSetDevice(init.lo),"dual batch init device");
        ck(cudaMemcpy(mp[init.lo]+init.low_index,&one,sizeof(one),cudaMemcpyHostToDevice),"dual batch init");

        B300DualShuffleStats sh{};
        for(int row=0;row<W;++row){
            dt_run_high_local(sparse,ngpu,threads);
            if(peer_shuffle)
                b300_dt_peer_low_to_high(dual,mp.data(),bp.data(),peer_ctx,&sh);
            else
                b300_dt_low_to_high(dual,mp.data(),bp.data(),sp.data(),chunk_elems,&sh,&copy_ctx);
            dt_run_low_local(sparse,ngpu,threads);
            if(row+1<W){
                b300_dt_zero_block_arenas(dual,bp.data());
                if(peer_shuffle)
                    b300_dt_peer_high_to_low_main(dual,mp.data(),peer_ctx,&sh);
                else
                    b300_dt_high_to_low_main(dual,mp.data(),sp.data(),chunk_elems,&sh,&copy_ctx);
            }
        }
        b300_dt_sync_all(ngpu,"dual batch final sync");

        Count ans=0;
        ck(cudaSetDevice(answer.hi),"dual batch answer device");
        ck(cudaMemcpy(&ans,mp[answer.hi]+answer.high_index,sizeof(ans),cudaMemcpyDeviceToHost),"dual batch answer");
        double wall=ram_seconds_since(residue0);all_wall+=wall;
        std::cout<<"backend=gridfp-b300-hbm32-dual-tile-sparse-batch n="<<n
                 <<" residue="<<ans<<" modulus="<<mod
                 <<" wall_s="<<wall
                 <<" ordinal="<<(mi+1)<<'/'<<mods.size()
                 <<" shuffle_mode="<<(peer_shuffle?"peer-kernel":"copy-pipeline")
                 <<" shuffle_tib="<<dt_tib(sh.main_bytes+sh.block_bytes)<<'\n';
        std::cout.flush();
    }

    std::cerr<<"dual-batch complete residues="<<mods.size()
             <<" setup_s="<<setup_s<<" solver_wall_s_sum="<<all_wall<<'\n';
    if(peer_shuffle) peer_ctx.release(); else copy_ctx.release();
    for(auto&c:gpu)c->release();
    return 0;
}
