#define main gpu_direct_atomicfree_multigpu_events_v02_main_unused
#include "oneesan_cuda_gridfp_gpu_direct_atomicfree_multigpu_events.cu"
#undef main

#include "ramstream32_gpu_direct_atomicfree_multigpu_staged.cuh"

int main(int argc,char**argv) {
    int n=argc>1?std::atoi(argv[1]):TARGET_W-1;
    Count mod=argc>2?Count(std::strtoul(argv[2],nullptr,10)):4294967291u;
    int threads=argc>3?std::atoi(argv[3]):256;
    int grid_x=argc>4?std::atoi(argv[4]):32;
    int grid_y=argc>5?std::atoi(argv[5]):16;
    int requested=argc>6?std::atoi(argv[6]):0;
    bool plan_only=gdg_has_arg(argc,argv,"--plan-only");
    int W=n+1;
    if (W!=TARGET_W || n<2 || W>MAXW || threads<=0 || threads>1024 || grid_x<=0 || grid_y<=0) return 1;
    if constexpr (LOW_LUT_K+HIGH_LUT_K+1!=TARGET_W) return 1;
    if (requested<0 || requested>GDM_MAX_GPU) return 1;

    auto prep0=std::chrono::steady_clock::now();
    build_full_dp(); G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);
    StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);
    HighDescHost highdesc=build_high_descriptors(storage,layout);
    LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);
    CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);
    GpuDirectCrossHost forward=build_gpu_direct_cross(storage);
    GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);
    GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);

    size_t auth_bytes=size_t(layout.main_size+layout.block_size)*sizeof(Count);
    size_t stage_bytes=size_t(layout.main_size)*sizeof(Count);
    size_t base_resident=loworbit.rec.size()*sizeof(uint64_t)
        +(highdirect.orbit_ops.nn.size()+highdirect.orbit_ops.nrnl.size())*sizeof(CpuHighOrbitOp)
        +(highdirect.orbit_off.nn.size()+highdirect.orbit_off.nrnl.size())*sizeof(uint32_t);
    size_t ordinary_resident=ordinary.bytes()
        -ordinary.low_cross.size()*sizeof(GpuDirectLowCrossOp)
        -ordinary.low_cross_off.size()*sizeof(uint32_t);
    size_t resident_meta=base_resident+ordinary_resident+cross.bytes();
    size_t base_transient=(lowdesc.main_desc.size()+lowdesc.block_desc.size())*sizeof(uint32_t)
        +(highdirect.closure_ops.block.size()+highdirect.closure_ops.cross.size())*sizeof(CpuHighClosureOp)
        +(highdirect.closure_off.block.size()+highdirect.closure_off.cross.size())*sizeof(uint32_t)
        +(forward.high_rank.size()+forward.low_rank.size())*sizeof(uint32_t);
    size_t ordinary_transient=ordinary.low_cross.size()*sizeof(GpuDirectLowCrossOp)
        +ordinary.low_cross_off.size()*sizeof(uint32_t);
    size_t transient_meta=resident_meta+base_transient+ordinary_transient;

    int plan_ng=requested>0?requested:GDM_MAX_GPU;
    GdmShardHost plan_shard=build_gdm_shards(layout,plan_ng);
    GdmsStagePlan plan_stage=build_gdms_stage_plan(layout,plan_shard,ordinary,cross,plan_ng);
    double prepare_host_s=gdg_seconds(prep0);
    if (plan_only) {
        std::cout<<"backend=gridfp-gpu-direct-atomicfree-multigpu-staged-v0.3-plan"
                 <<" n="<<n<<" gpus="<<plan_ng
                 <<" main_states="<<layout.main_size<<" blocked_states="<<layout.block_size
                 <<" authoritative_gib="<<double(auth_bytes)/double(1ULL<<30)
                 <<" shard_max_gib="<<double(plan_shard.max_elems*sizeof(Count))/double(1ULL<<30)
                 <<" shard_min_gib="<<double(plan_shard.min_elems*sizeof(Count))/double(1ULL<<30)
                 <<" stage_mirror_gib_per_gpu="<<double(stage_bytes)/double(1ULL<<30)
                 <<" bulk_p2p_copy_gib_per_row="<<double(plan_stage.copy_bytes_per_row)/double(1ULL<<30)
                 <<" max_device_phase_copy_gib="<<double(plan_stage.max_device_phase_bytes)/double(1ULL<<30)
                 <<" resident_metadata_per_gpu_mib="<<double(resident_meta)/double(1ULL<<20)
                 <<" peak_per_gpu_gib="<<double(plan_shard.max_elems*sizeof(Count)+stage_bytes+transient_meta)/double(1ULL<<30)
                 <<" low_max_indegree="<<ordinary.low_max_indegree
                 <<" high_max_indegree="<<ordinary.high_max_indegree
                 <<" closure_atomic=0 scratch_bytes=0 shard=storage-block"
                 <<" remote_scalar_gather_loads=0 remote_orbit_accesses=1"
                 <<" device_fences_per_row="<<2*(LOW_LUT_K+HIGH_LUT_K)+1
                 <<" host_barrier_points_per_row=1 bulk_p2p_staging=1"
                 <<" prepare_s="<<prepare_host_s<<'\n';
        return 0;
    }

    int visible=0; ck(cudaGetDeviceCount(&visible),"gdms device count");
    int ngpu=requested>0?std::min(requested,visible):std::min(visible,GDM_MAX_GPU);
    if (ngpu<1 || ngpu>GDM_MAX_GPU) return 2;
    gdm_enable_full_p2p(ngpu);
    GdmShardHost shard=build_gdm_shards(layout,ngpu);
    GdmsStagePlan stage_plan=build_gdms_stage_plan(layout,shard,ordinary,cross,ngpu);

    std::array<size_t,GDM_MAX_GPU> free_before{},total_before{};
    for (int d=0;d<ngpu;++d) {
        ck(cudaSetDevice(d),"gdms mem device"); ck(cudaMemGetInfo(&free_before[d],&total_before[d]),"gdms mem info");
        size_t need=size_t(shard.total_elems[d])*sizeof(Count)+stage_bytes+transient_meta;
        if (need>free_before[d]) {
            std::cerr<<"gdms insufficient HBM device="<<d<<" need_gib="<<double(need)/double(1ULL<<30)
                     <<" free_gib="<<double(free_before[d])/double(1ULL<<30)<<'\n'; return 4;
        }
    }

    Count* main_ptr[GDM_MAX_GPU]{}; Count* block_ptr[GDM_MAX_GPU]{};
    for (int d=0;d<ngpu;++d) {
        ck(cudaSetDevice(d),"gdms alloc device");
        if (shard.main_elems[d]) ck(cudaMalloc(&main_ptr[d],size_t(shard.main_elems[d])*sizeof(Count)),"gdms alloc main");
        if (shard.block_elems[d]) ck(cudaMalloc(&block_ptr[d],size_t(shard.block_elems[d])*sizeof(Count)),"gdms alloc block");
        if (shard.main_elems[d]) ck(cudaMemset(main_ptr[d],0,size_t(shard.main_elems[d])*sizeof(Count)),"gdms zero main");
        if (shard.block_elems[d]) ck(cudaMemset(block_ptr[d],0,size_t(shard.block_elems[d])*sizeof(Count)),"gdms zero block");
    }

    std::vector<GpuDirectDeviceTables> base(ngpu);
    std::vector<GpuDirectGatherDeviceTables> ordinary_dev(ngpu);
    std::vector<GpuDirectCrossGatherDeviceTables> cross_dev(ngpu);
    for (int d=0;d<ngpu;++d) {
        ck(cudaSetDevice(d),"gdms metadata device"); ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"gdms modulus");
        base[d].install(storage,layout,lowdesc,loworbit,highdirect,forward);
        ordinary_dev[d].install(ordinary); gpu_direct_gather_drop_redundant(base[d]);
        cross_dev[d].install(cross); gpu_direct_cross_gather_drop_redundant(base[d],ordinary_dev[d]);
        gdm_install_shards(shard,main_ptr,block_ptr,ngpu,d);
    }

    GdmsStagePipeline pipe;
    pipe.init(ngpu,layout);
    double prepare_s=gdg_seconds(prep0);

    MateID init=MateID(R)<<(2*(W-1)); Code init_rank=storage_rank_main_host(init,storage,layout);
    auto init_loc=gdm_locate_main(init_rank,layout,shard); Count one=1;
    ck(cudaSetDevice(init_loc.first),"gdms init device");
    ck(cudaMemcpy(main_ptr[init_loc.first]+init_loc.second,&one,sizeof(one),cudaMemcpyHostToDevice),"gdms init");

    double submit_s=0,row_sync_s=0; auto wall0=std::chrono::steady_clock::now();
    for (int row=0;row<W;++row) {
        int slot=0;
        auto t=std::chrono::steady_clock::now();
        gdms_enqueue_high(pipe,stage_plan,layout,shard,main_ptr,threads,grid_x,grid_y,slot);
        gdms_enqueue_low(pipe,stage_plan,layout,shard,main_ptr,threads,grid_x,grid_y,slot);
        submit_s+=gdg_seconds(t);
        if (slot!=pipe.slots) { std::cerr<<"gdms slot mismatch "<<slot<<'/'<<pipe.slots<<'\n'; return 5; }
        t=std::chrono::steady_clock::now(); pipe.sync_row(); row_sync_s+=gdg_seconds(t);
        std::cerr<<"row "<<row+1<<'/'<<W<<" submit_s="<<submit_s<<" row_sync_s="<<row_sync_s<<'\n';
    }
    double wall_s=gdg_seconds(wall0);

    Code final_rank=storage_rank_main_host(MateID(R),storage,layout); auto final_loc=gdm_locate_main(final_rank,layout,shard); Count answer=0;
    ck(cudaSetDevice(final_loc.first),"gdms answer device");
    ck(cudaMemcpy(&answer,main_ptr[final_loc.first]+final_loc.second,sizeof(answer),cudaMemcpyDeviceToHost),"gdms answer");

    std::cout<<"backend=gridfp-gpu-direct-atomicfree-multigpu-staged-v0.3"
             <<" n="<<n<<" residue="<<answer<<" modulus="<<mod
             <<" gpus="<<ngpu<<" authoritative_gib="<<double(auth_bytes)/double(1ULL<<30)
             <<" shard_max_gib="<<double(shard.max_elems*sizeof(Count))/double(1ULL<<30)
             <<" shard_min_gib="<<double(shard.min_elems*sizeof(Count))/double(1ULL<<30)
             <<" stage_mirror_gib_per_gpu="<<double(stage_bytes)/double(1ULL<<30)
             <<" bulk_p2p_copy_gib_per_row="<<double(stage_plan.copy_bytes_per_row)/double(1ULL<<30)
             <<" resident_metadata_per_gpu_mib="<<double(resident_meta)/double(1ULL<<20)
             <<" threads="<<threads<<" grid_x="<<grid_x<<" grid_y="<<grid_y
             <<" submit_s="<<submit_s<<" row_sync_s="<<row_sync_s
             <<" prepare_s="<<prepare_s<<" wall_s="<<wall_s
             <<" closure_atomic=0 scratch_bytes=0 shard=storage-block"
             <<" remote_scalar_gather_loads=0 remote_orbit_accesses=1 bulk_p2p_staging=1"
             <<" device_fences_per_row="<<pipe.slots
             <<" host_barrier_points_per_row=1\n";

    pipe.destroy();
    for (int d=0;d<ngpu;++d) {
        ck(cudaSetDevice(d),"gdms release device"); cross_dev[d].release(); ordinary_dev[d].release(); base[d].release();
        if (main_ptr[d]) cudaFree(main_ptr[d]); if (block_ptr[d]) cudaFree(block_ptr[d]);
    }
    return 0;
}
