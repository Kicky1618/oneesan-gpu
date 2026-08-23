// Persistent multi-modulus wrapper around the dual-tile backend.
// Reuse all factor tables, sparse actions, GPU metadata and the ~1.94 TiB HBM
// allocation across CRT primes; only the state arrays and D_MOD change.
#define main(...) b300_dual_tile_single_main(__VA_ARGS__)
#include "oneesan_cuda_gridfp_b300_hbm32_dual_tile_sparse.cu"
#undef main
#include "../gridfp/ramstream32_b300_dual_tile_peer_swap.cuh"
#include "../gridfp/ramstream32_b300_dual_tile_pruned_plan.cuh"
#include "../gridfp/ramstream32_b300_dual_tile_dynamic_scan.cuh"

static int dt_env_int(const char* name,int fallback,int lo){
    const char* s=std::getenv(name);if(!s||!*s)return fallback;
    return std::max(lo,std::atoi(s));
}

enum class DTBatchShuffleMode { CopyPipeline, PeerKernel, ReachPruned, DynamicPruned };
static DTBatchShuffleMode dt_batch_shuffle_mode(){
    const char* s=std::getenv("ONEESAN_DUAL_SHUFFLE");
    if(!s||!*s)return DTBatchShuffleMode::CopyPipeline;
    if(std::strcmp(s,"peer-kernel")==0||std::strcmp(s,"peer")==0)
        return DTBatchShuffleMode::PeerKernel;
    if(std::strcmp(s,"peer-pruned")==0||std::strcmp(s,"pruned")==0
       ||std::strcmp(s,"reachability")==0)
        return DTBatchShuffleMode::ReachPruned;
    if(std::strcmp(s,"peer-dynamic")==0||std::strcmp(s,"dynamic")==0
       ||std::strcmp(s,"dynamic-pruned")==0)
        return DTBatchShuffleMode::DynamicPruned;
    std::cerr<<"unknown ONEESAN_DUAL_SHUFFLE="<<s
             <<" (use copy-pipeline, peer-kernel, peer-pruned, or peer-dynamic)\n";
    std::exit(656);
}
static const char* dt_batch_shuffle_name(DTBatchShuffleMode m){
    switch(m){
        case DTBatchShuffleMode::CopyPipeline:return "copy-pipeline";
        case DTBatchShuffleMode::PeerKernel:return "peer-kernel";
        case DTBatchShuffleMode::ReachPruned:return "peer-pruned";
        case DTBatchShuffleMode::DynamicPruned:return "peer-dynamic";
    }
    return "unknown";
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

static void dt_add_plan_ports(
    const B300DualPrunedArrayPlan&p,std::array<uint64_t,MAXGPU>&port
){
    for(int g=0;g<MAXGPU;++g)port[g]+=p.gpu_port_bytes[g];
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
    int prune_launch_kib=dt_env_int("ONEESAN_DUAL_PRUNE_LAUNCH_KIB",8192,0);
    int scan_chunk_mib=dt_env_int("ONEESAN_DUAL_SCAN_CHUNK_MIB",16,1);
    uint64_t prune_launch_bytes=uint64_t(prune_launch_kib)*1024ull;
    DTBatchShuffleMode shuffle_mode=dt_batch_shuffle_mode();
    bool peer_family=shuffle_mode!=DTBatchShuffleMode::CopyPipeline;
    bool uses_reach=shuffle_mode==DTBatchShuffleMode::ReachPruned
                 || shuffle_mode==DTBatchShuffleMode::DynamicPruned;
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
    B300DualReachSchedule reach;
    B300DualPrunedSchedulePlan prune_plan;
    if(uses_reach)reach=build_b300_dual_reach_schedule(sparse,f,l,dual);
    if(shuffle_mode==DTBatchShuffleMode::ReachPruned)
        prune_plan=build_b300_dual_pruned_schedule_plan(dual,l,reach,prune_launch_bytes);

    Code chunk_elems=Code(chunk_mib)*(1ull<<20)/sizeof(Count);
    Code scan_chunk_elems=Code(scan_chunk_mib)*(1ull<<20)/sizeof(Count);
    // All peer engines are genuinely scratch-free; only copy-pipeline needs it.
    Code scratch_elems=peer_family?Code(0):chunk_elems;

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
    B300DualPrunedPeerContext pruned_ctx;
    B300DualDynamicScanContext dynamic_scan;
    if(shuffle_mode==DTBatchShuffleMode::CopyPipeline)copy_ctx.init(ngpu);
    else if(shuffle_mode==DTBatchShuffleMode::PeerKernel)peer_ctx.init(ngpu);
    else pruned_ctx.init(ngpu);
    if(shuffle_mode==DTBatchShuffleMode::DynamicPruned)
        dynamic_scan.init(dual,l,uint32_t(scan_chunk_elems));

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
             <<" shuffle="<<dt_batch_shuffle_name(shuffle_mode)
             <<" scratch_mib_per_gpu="<<(scratch_elems*sizeof(Count)/(1ull<<20));
    if(shuffle_mode==DTBatchShuffleMode::ReachPruned){
        uint64_t max_port=prune_plan.max_gpu_port_bytes_per_residue();
        std::cerr<<" prune_launch_kib="<<prune_launch_kib
                 <<" prune_logical_tib="<<dt_tib((long double)prune_plan.logical_bytes_per_residue())
                 <<" prune_scheduled_tib="<<dt_tib((long double)prune_plan.scheduled_bytes_per_residue())
                 <<" prune_full_tib="<<dt_tib((long double)prune_plan.full_bytes_per_residue())
                 <<" prune_operation_units="<<prune_plan.launches_per_residue()
                 <<" prune_max_gpu_port_tib="<<dt_tib((long double)max_port)
                 <<" prune_ideal_1p8TBs_bidirectional_s="<<double((long double)max_port/1.8e12L);
    }else if(shuffle_mode==DTBatchShuffleMode::DynamicPruned){
        uint64_t tasks=0;
        for(int g=0;g<ngpu;++g)tasks+=dynamic_scan.main_low[g].ntask
            +dynamic_scan.block_low[g].ntask+dynamic_scan.main_high[g].ntask;
        std::cerr<<" prune_launch_kib="<<prune_launch_kib
                 <<" dynamic_scan_chunk_mib="<<scan_chunk_mib
                 <<" dynamic_scan_tasks="<<tasks;
    }
    std::cerr<<'\n';

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
        double dynamic_scan_s=0.0;
        uint64_t dynamic_logical_bytes=0,dynamic_scheduled_bytes=0,dynamic_ops=0;
        std::array<uint64_t,MAXGPU> dynamic_port{};
        for(int row=0;row<W;++row){
            dt_run_high_local(sparse,ngpu,threads);
            if(shuffle_mode==DTBatchShuffleMode::PeerKernel){
                b300_dt_peer_low_to_high(dual,mp.data(),bp.data(),peer_ctx,&sh);
            }else if(shuffle_mode==DTBatchShuffleMode::ReachPruned){
                b300_dt_execute_pruned_l2h(dual,mp.data(),bp.data(),
                    prune_plan.l2h_main[row],prune_plan.l2h_block[row],pruned_ctx,&sh);
            }else if(shuffle_mode==DTBatchShuffleMode::DynamicPruned){
                auto t=std::chrono::steady_clock::now();
                B300DualReachStage actual=b300_dt_dynamic_scan_l2h(
                    dual,l,mp.data(),bp.data(),reach.l2h[row],dynamic_scan);
                dynamic_scan_s+=ram_seconds_since(t);
                auto pm=b300_dt_build_pruned_array_plan(dual,l,false,true,actual,prune_launch_bytes);
                auto pd=b300_dt_build_pruned_array_plan(dual,l,true,true,actual,prune_launch_bytes);
                dynamic_logical_bytes+=pm.logical_bytes+pd.logical_bytes;
                dynamic_scheduled_bytes+=pm.scheduled_bytes+pd.scheduled_bytes;
                dynamic_ops+=pm.launches+pd.launches;dt_add_plan_ports(pm,dynamic_port);dt_add_plan_ports(pd,dynamic_port);
                b300_dt_execute_pruned_l2h(dual,mp.data(),bp.data(),pm,pd,pruned_ctx,&sh);
            }else{
                b300_dt_low_to_high(dual,mp.data(),bp.data(),sp.data(),chunk_elems,&sh,&copy_ctx);
            }

            dt_run_low_local(sparse,ngpu,threads);
            if(row+1<W){
                b300_dt_zero_block_arenas(dual,bp.data());
                if(shuffle_mode==DTBatchShuffleMode::PeerKernel){
                    b300_dt_peer_high_to_low_main(dual,mp.data(),peer_ctx,&sh);
                }else if(shuffle_mode==DTBatchShuffleMode::ReachPruned){
                    b300_dt_execute_pruned_h2l_main(dual,mp.data(),
                        prune_plan.h2l_main[row],pruned_ctx,&sh);
                }else if(shuffle_mode==DTBatchShuffleMode::DynamicPruned){
                    auto t=std::chrono::steady_clock::now();
                    B300DualReachStage actual=b300_dt_dynamic_scan_h2l_main(
                        dual,l,mp.data(),reach.h2l[row],dynamic_scan);
                    dynamic_scan_s+=ram_seconds_since(t);
                    auto pm=b300_dt_build_pruned_array_plan(dual,l,false,false,actual,prune_launch_bytes);
                    dynamic_logical_bytes+=pm.logical_bytes;dynamic_scheduled_bytes+=pm.scheduled_bytes;
                    dynamic_ops+=pm.launches;dt_add_plan_ports(pm,dynamic_port);
                    b300_dt_execute_pruned_h2l_main(dual,mp.data(),pm,pruned_ctx,&sh);
                }else{
                    b300_dt_high_to_low_main(dual,mp.data(),sp.data(),chunk_elems,&sh,&copy_ctx);
                }
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
                 <<" shuffle_mode="<<dt_batch_shuffle_name(shuffle_mode)
                 <<" shuffle_tib="<<dt_tib(sh.main_bytes+sh.block_bytes)
                 <<" shuffle_operation_units="<<sh.chunk_barriers;
        if(shuffle_mode==DTBatchShuffleMode::DynamicPruned){
            uint64_t max_port=*std::max_element(dynamic_port.begin(),dynamic_port.end());
            std::cout<<" dynamic_scan_s="<<dynamic_scan_s
                     <<" dynamic_logical_tib="<<dt_tib((long double)dynamic_logical_bytes)
                     <<" dynamic_scheduled_tib="<<dt_tib((long double)dynamic_scheduled_bytes)
                     <<" dynamic_operation_units="<<dynamic_ops
                     <<" dynamic_max_gpu_port_tib="<<dt_tib((long double)max_port)
                     <<" dynamic_ideal_1p8TBs_bidirectional_s="<<double((long double)max_port/1.8e12L);
        }
        std::cout<<'\n';
        std::cout.flush();
    }

    std::cerr<<"dual-batch complete residues="<<mods.size()
             <<" setup_s="<<setup_s<<" solver_wall_s_sum="<<all_wall<<'\n';
    if(shuffle_mode==DTBatchShuffleMode::CopyPipeline)copy_ctx.release();
    else if(shuffle_mode==DTBatchShuffleMode::PeerKernel)peer_ctx.release();
    else pruned_ctx.release();
    if(shuffle_mode==DTBatchShuffleMode::DynamicPruned)dynamic_scan.release();
    for(auto&c:gpu)c->release();
    return 0;
}
