#define main gpu_direct_gather_backend_reference_main
#include "oneesan_cuda_gridfp_gpu_direct_gather.cu"
#undef main

#include "ramstream32_gpu_direct_gather_cross.cuh"

int main(int argc,char**argv){
    int n=argc>1?std::atoi(argv[1]):TARGET_W-1;
    Count mod=argc>2?Count(std::strtoul(argv[2],nullptr,10)):4294967291u;
    int threads=argc>3?std::atoi(argv[3]):256;
    int grid_x=argc>4?std::atoi(argv[4]):16;
    int grid_y=argc>5?std::atoi(argv[5]):8;
    bool plan_only=gdg_has_arg(argc,argv,"--plan-only");int W=n+1;
    if(W!=TARGET_W||n<2||W>MAXW||threads<=0||threads>1024||grid_x<=0||grid_y<=0)return 1;

    auto prep0=std::chrono::steady_clock::now();
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);
    LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);
    GpuDirectCrossHost forward=build_gpu_direct_cross(storage);
    GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);
    GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);
    double prepare_s=gdg_seconds(prep0);

    size_t auth_bytes=size_t(layout.main_size+layout.block_size)*sizeof(Count);
    size_t base_resident=loworbit.rec.size()*sizeof(uint64_t)
        +(highdirect.orbit_ops.nn.size()+highdirect.orbit_ops.nrnl.size())*sizeof(CpuHighOrbitOp)
        +(highdirect.orbit_off.nn.size()+highdirect.orbit_off.nrnl.size())*sizeof(uint32_t);
    size_t ordinary_resident=ordinary.bytes()
        -ordinary.low_cross.size()*sizeof(GpuDirectLowCrossOp)
        -ordinary.low_cross_off.size()*sizeof(uint32_t);
    size_t resident_meta=base_resident+ordinary_resident+cross.bytes();
    size_t resident_total=auth_bytes+resident_meta;

    size_t base_transient=(lowdesc.main_desc.size()+lowdesc.block_desc.size())*sizeof(uint32_t)
        +(highdirect.closure_ops.block.size()+highdirect.closure_ops.cross.size())*sizeof(CpuHighClosureOp)
        +(highdirect.closure_off.block.size()+highdirect.closure_off.cross.size())*sizeof(uint32_t)
        +(forward.high_rank.size()+forward.low_rank.size())*sizeof(uint32_t);
    size_t ordinary_transient=ordinary.low_cross.size()*sizeof(GpuDirectLowCrossOp)
        +ordinary.low_cross_off.size()*sizeof(uint32_t);
    size_t transient_peak=resident_total+base_transient+ordinary_transient;

    if(plan_only){
        std::cout<<"backend=gridfp-gpu-direct-atomicfree-v0.1-plan"
                 <<" n="<<n<<" main_states="<<layout.main_size<<" blocked_states="<<layout.block_size
                 <<" authoritative_gib="<<double(auth_bytes)/double(1ULL<<30)
                 <<" ordinary_mib="<<double(ordinary_resident)/double(1ULL<<20)
                 <<" cross_mib="<<double(cross.bytes())/double(1ULL<<20)
                 <<" resident_metadata_mib="<<double(resident_meta)/double(1ULL<<20)
                 <<" resident_total_gib="<<double(resident_total)/double(1ULL<<30)
                 <<" transient_peak_gib="<<double(transient_peak)/double(1ULL<<30)
                 <<" low_local_edges="<<ordinary.low_src.size()<<" high_local_edges="<<ordinary.high_src.size()
                 <<" low_cross_ops="<<cross.low_op.size()<<" high_cross_ops="<<cross.high_op.size()
                 <<" low_max_indegree="<<ordinary.low_max_indegree<<" high_max_indegree="<<ordinary.high_max_indegree
                 <<" closure_atomic=0 scratch_bytes=0 prepare_s="<<prepare_s<<'\n';return 0;
    }

    int visible=0;ck(cudaGetDeviceCount(&visible),"gdx device count");if(visible<1)return 3;ck(cudaSetDevice(0),"gdx set device");
    size_t free_bytes=0,total_bytes=0;ck(cudaMemGetInfo(&free_bytes,&total_bytes),"gdx mem info");
    if(transient_peak>free_bytes){std::cerr<<"insufficient HBM: need_gib="<<double(transient_peak)/double(1ULL<<30)<<" free_gib="<<double(free_bytes)/double(1ULL<<30)<<'\n';return 4;}
    Count*dmain=nullptr,*dblock=nullptr;ck(cudaMalloc(&dmain,size_t(layout.main_size)*sizeof(Count)),"gdx alloc main");ck(cudaMalloc(&dblock,size_t(layout.block_size)*sizeof(Count)),"gdx alloc block");
    ck(cudaMemset(dmain,0,size_t(layout.main_size)*sizeof(Count)),"gdx zero main");ck(cudaMemset(dblock,0,size_t(layout.block_size)*sizeof(Count)),"gdx zero block");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"gdx modulus");
    GpuDirectDeviceTables base;base.install(storage,layout,lowdesc,loworbit,highdirect,forward);
    GpuDirectGatherDeviceTables ot;ot.install(ordinary);gpu_direct_gather_drop_redundant(base);
    GpuDirectCrossGatherDeviceTables xt;xt.install(cross);gpu_direct_cross_gather_drop_redundant(base,ot);
    MateID init=MateID(R)<<(2*(W-1));Code init_rank=storage_rank_main_host(init,storage,layout);Count one=1;ck(cudaMemcpy(dmain+init_rank,&one,sizeof(one),cudaMemcpyHostToDevice),"gdx init");
    double high_s=0,low_s=0;auto wall0=std::chrono::steady_clock::now();
    for(int row=0;row<W;++row){auto t=std::chrono::steady_clock::now();gpu_direct_run_high_atomicfree(dmain,dblock,layout,threads,grid_x,grid_y);high_s+=gdg_seconds(t);t=std::chrono::steady_clock::now();gpu_direct_run_low_atomicfree(dmain,dblock,layout,threads,grid_x,grid_y);low_s+=gdg_seconds(t);std::cerr<<"row "<<row+1<<'/'<<W<<" high_s="<<high_s<<" low_s="<<low_s<<'\n';}
    double wall_s=gdg_seconds(wall0);Code final_rank=storage_rank_main_host(MateID(R),storage,layout);Count answer=0;ck(cudaMemcpy(&answer,dmain+final_rank,sizeof(answer),cudaMemcpyDeviceToHost),"gdx answer");
    std::cout<<"backend=gridfp-gpu-direct-atomicfree-v0.1 n="<<n<<" residue="<<answer<<" modulus="<<mod
             <<" authoritative_gib="<<double(auth_bytes)/double(1ULL<<30)<<" resident_metadata_mib="<<double(resident_meta)/double(1ULL<<20)
             <<" threads="<<threads<<" grid_x="<<grid_x<<" grid_y="<<grid_y<<" high_s="<<high_s<<" low_s="<<low_s<<" prepare_s="<<prepare_s<<" wall_s="<<wall_s
             <<" closure_atomic=0 scratch_bytes=0\n";
    xt.release();ot.release();base.release();cudaFree(dmain);cudaFree(dblock);return 0;
}
