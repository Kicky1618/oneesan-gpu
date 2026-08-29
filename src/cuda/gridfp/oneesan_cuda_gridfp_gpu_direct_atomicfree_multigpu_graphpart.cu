#include "ramstream32_gpu_direct_atomicfree_base.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_staged.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_peerstreams.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_orbitstage.cuh"

#define gdow_low_orbit_kernel gdow_low_orbit_kernel_v06_buggy_graphpart
#define gdow_high_orbit_kernel gdow_high_orbit_kernel_v06_buggy_graphpart
#define gdow_enqueue_row gdow_enqueue_row_v06_buggy_graphpart
#include "ramstream32_gpu_direct_atomicfree_multigpu_ownerorbit.cuh"
#undef gdow_low_orbit_kernel
#undef gdow_high_orbit_kernel
#undef gdow_enqueue_row

#include "ramstream32_gpu_direct_atomicfree_multigpu_ownerorbit_safe.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_graphpart.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_graphselect.cuh"

static constexpr const char* GDPG_WORK_ID="f8614f64-01f6-4c12-9264-03b44a385e74";

static void gdpg_enable_full_p2p(int ngpu){
    int peers=0;for(int a=0;a<ngpu;++a)for(int b=0;b<ngpu;++b)if(a!=b){int can=0;ck(cudaDeviceCanAccessPeer(&can,a,b),"gdpg can peer");if(!can)continue;ck(cudaSetDevice(a),"gdpg set p2p");cudaError_t e=cudaDeviceEnablePeerAccess(b,0);if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else ck(e,"gdpg enable peer");++peers;}
    if(ngpu>1&&peers!=ngpu*(ngpu-1)){std::cerr<<"gdpg requires full P2P: "<<peers<<'/'<<ngpu*(ngpu-1)<<'\n';std::exit(188);}
}
static unsigned long long gdpg_active_visits(const GdowOrbitPlan&p,int ngpu){unsigned long long n=0;auto add=[&](const std::vector<GdowOrbitPhase>&v){for(const auto&ph:v)for(int d=0;d<ngpu;++d)n+=static_cast<unsigned long long>(__builtin_popcountll(ph.active_source_mask[d]));};add(p.high);add(p.low);return n;}
static double gdpg_reduction(unsigned long long before,unsigned long long after){return before?100.0*(double(before)-double(after))/double(before):0.0;}

int main(int argc,char**argv){
    int n=argc>1?std::atoi(argv[1]):TARGET_W-1;Count mod=argc>2?Count(std::strtoul(argv[2],nullptr,10)):4294967291u;int threads=argc>3?std::atoi(argv[3]):256;int grid_x=argc>4?std::atoi(argv[4]):32;int grid_y=argc>5?std::atoi(argv[5]):16;int requested=argc>6?std::atoi(argv[6]):0;bool plan_only=gdg_has_arg(argc,argv,"--plan-only");int W=n+1;
    if(W!=TARGET_W||n<2||W>MAXW||threads<=0||threads>1024||grid_x<=0||grid_y<=0)return 1;if constexpr(LOW_LUT_K+HIGH_LUT_K+1!=TARGET_W)return 1;if(requested<0||requested>GDM_MAX_GPU)return 1;

    auto prep0=std::chrono::steady_clock::now();build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectCrossHost forward=build_gpu_direct_cross(storage);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);

    size_t auth_bytes=size_t(layout.main_size+layout.block_size)*sizeof(Count),main_stage_bytes=size_t(layout.main_size)*sizeof(Count),block_stage_bytes=size_t(layout.block_size)*sizeof(Count);
    size_t base_resident=loworbit.rec.size()*sizeof(uint64_t)+(highdirect.orbit_ops.nn.size()+highdirect.orbit_ops.nrnl.size())*sizeof(CpuHighOrbitOp)+(highdirect.orbit_off.nn.size()+highdirect.orbit_off.nrnl.size())*sizeof(uint32_t);
    size_t ordinary_resident=ordinary.bytes()-ordinary.low_cross.size()*sizeof(GpuDirectLowCrossOp)-ordinary.low_cross_off.size()*sizeof(uint32_t);size_t resident_meta=base_resident+ordinary_resident+cross.bytes();
    size_t base_transient=(lowdesc.main_desc.size()+lowdesc.block_desc.size())*sizeof(uint32_t)+(highdirect.closure_ops.block.size()+highdirect.closure_ops.cross.size())*sizeof(CpuHighClosureOp)+(highdirect.closure_off.block.size()+highdirect.closure_off.cross.size())*sizeof(uint32_t)+(forward.high_rank.size()+forward.low_rank.size())*sizeof(uint32_t);size_t ordinary_transient=ordinary.low_cross.size()*sizeof(GpuDirectLowCrossOp)+ordinary.low_cross_off.size()*sizeof(uint32_t);size_t transient_meta=resident_meta+base_transient+ordinary_transient;

    int plan_ng=requested>0?requested:GDM_MAX_GPU;GdpgSelection ps=gdpg_select_exact(layout,loworbit,highdirect,ordinary,cross,plan_ng);GdmpPeerStats pp=gdmp_peer_stats(ps.gather,layout,ps.shard,plan_ng);double prepare_host_s=gdg_seconds(prep0);unsigned long long legacy_total=ps.legacy_gather_bytes+ps.legacy_orbit_bytes,graph_total=ps.graph_gather_bytes+ps.graph_orbit_bytes,selected_total=ps.gather.copy_bytes_per_row+ps.orbit.copy_bytes_per_row;
    if(plan_only){
        std::cout<<"backend=gridfp-gpu-direct-atomicfree-multigpu-graphpart-v0.7-plan work_id="<<GDPG_WORK_ID<<" n="<<n<<" gpus="<<plan_ng
                 <<" partition="<<(ps.graph_selected?"graph":"legacy-fallback")
                 <<" legacy_gather_p2p_gib_per_row="<<double(ps.legacy_gather_bytes)/double(1ULL<<30)
                 <<" legacy_orbit_p2p_gib_per_row="<<double(ps.legacy_orbit_bytes)/double(1ULL<<30)
                 <<" legacy_total_p2p_gib_per_row="<<double(legacy_total)/double(1ULL<<30)
                 <<" graph_gather_p2p_gib_per_row="<<double(ps.graph_gather_bytes)/double(1ULL<<30)
                 <<" graph_orbit_p2p_gib_per_row="<<double(ps.graph_orbit_bytes)/double(1ULL<<30)
                 <<" graph_total_p2p_gib_per_row="<<double(graph_total)/double(1ULL<<30)
                 <<" selected_total_p2p_gib_per_row="<<double(selected_total)/double(1ULL<<30)
                 <<" selected_p2p_reduction_pct="<<gdpg_reduction(legacy_total,selected_total)
                 <<" graph_cut_gib_weight="<<double(ps.graph_cut_bytes)/double(1ULL<<30)
                 <<" graph_max_to_avg="<<ps.graph_max_to_avg
                 <<" shard_max_gib="<<double(ps.shard.max_elems*sizeof(Count))/double(1ULL<<30)
                 <<" shard_min_gib="<<double(ps.shard.min_elems*sizeof(Count))/double(1ULL<<30)
                 <<" main_stage_mirror_gib_per_gpu="<<double(main_stage_bytes)/double(1ULL<<30)
                 <<" block_stage_mirror_gib_per_gpu="<<double(block_stage_bytes)/double(1ULL<<30)
                 <<" peak_per_gpu_gib="<<double(ps.shard.max_elems*sizeof(Count)+main_stage_bytes+block_stage_bytes+transient_meta)/double(1ULL<<30)
                 <<" gather_peer_block_copies_per_row="<<pp.copy_ops_per_row
                 <<" owner_orbit_peer_block_copies_per_row="<<ps.orbit.block_copies_per_row
                 <<" active_gather_peer_pairs="<<pp.active_peer_pairs
                 <<" owner_orbit_active_source_block_visits_per_row="<<gdpg_active_visits(ps.orbit,plan_ng)
                 <<" low_max_indegree="<<ordinary.low_max_indegree<<" high_max_indegree="<<ordinary.high_max_indegree
                 <<" closure_atomic=0 scratch_bytes=0 remote_scalar_gather_loads=0 remote_orbit_reads=0 remote_orbit_writes=0 fine_grain_p2p=0"
                 <<" host_barrier_points_per_row=1 prepare_s="<<prepare_host_s<<'\n';return 0;
    }

    int visible=0;ck(cudaGetDeviceCount(&visible),"gdpg device count");int ngpu=requested>0?std::min(requested,visible):std::min(visible,GDM_MAX_GPU);if(ngpu<1||ngpu>GDM_MAX_GPU)return 2;gdpg_enable_full_p2p(ngpu);
    GdpgSelection sel=gdpg_select_exact(layout,loworbit,highdirect,ordinary,cross,ngpu);GdmShardHost shard=std::move(sel.shard);GdmsStagePlan gather_plan=std::move(sel.gather);GdowOrbitPlan orbit_plan=std::move(sel.orbit);GdmpPeerStats peer_stats=gdmp_peer_stats(gather_plan,layout,shard,ngpu);unsigned long long runtime_legacy=sel.legacy_gather_bytes+sel.legacy_orbit_bytes,runtime_selected=gather_plan.copy_bytes_per_row+orbit_plan.copy_bytes_per_row;

    std::array<size_t,GDM_MAX_GPU>free_before{},total_before{};for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdpg mem device");ck(cudaMemGetInfo(&free_before[d],&total_before[d]),"gdpg mem info");size_t need=size_t(shard.total_elems[d])*sizeof(Count)+main_stage_bytes+block_stage_bytes+transient_meta;if(need>free_before[d]){std::cerr<<"gdpg insufficient HBM device="<<d<<" need_gib="<<double(need)/double(1ULL<<30)<<" free_gib="<<double(free_before[d])/double(1ULL<<30)<<'\n';return 4;}}

    Count*main_ptr[GDM_MAX_GPU]{};Count*block_ptr[GDM_MAX_GPU]{};for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdpg alloc device");if(shard.main_elems[d])ck(cudaMalloc(&main_ptr[d],size_t(shard.main_elems[d])*sizeof(Count)),"gdpg alloc main");if(shard.block_elems[d])ck(cudaMalloc(&block_ptr[d],size_t(shard.block_elems[d])*sizeof(Count)),"gdpg alloc block");if(shard.main_elems[d])ck(cudaMemset(main_ptr[d],0,size_t(shard.main_elems[d])*sizeof(Count)),"gdpg zero main");if(shard.block_elems[d])ck(cudaMemset(block_ptr[d],0,size_t(shard.block_elems[d])*sizeof(Count)),"gdpg zero block");}
    std::vector<GpuDirectDeviceTables>base(ngpu);std::vector<GpuDirectGatherDeviceTables>ordinary_dev(ngpu);std::vector<GpuDirectCrossGatherDeviceTables>cross_dev(ngpu);for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdpg metadata device");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"gdpg modulus");base[d].install(storage,layout,lowdesc,loworbit,highdirect,forward);ordinary_dev[d].install(ordinary);gpu_direct_gather_drop_redundant(base[d]);cross_dev[d].install(cross);gpu_direct_cross_gather_drop_redundant(base[d],ordinary_dev[d]);gdm_install_shards(shard,main_ptr,block_ptr,ngpu,d);}

    GdpoPipeline pipe;pipe.init(ngpu,layout);double prepare_s=gdg_seconds(prep0);MateID init=MateID(R)<<(2*(W-1));Code init_rank=storage_rank_main_host(init,storage,layout);auto init_loc=gdm_locate_main(init_rank,layout,shard);Count one=1;ck(cudaSetDevice(init_loc.first),"gdpg init device");ck(cudaMemcpy(main_ptr[init_loc.first]+init_loc.second,&one,sizeof(one),cudaMemcpyHostToDevice),"gdpg init");
    double submit_s=0,row_sync_s=0;auto wall0=std::chrono::steady_clock::now();for(int row=0;row<W;++row){int slot=0;auto t=std::chrono::steady_clock::now();gdow_enqueue_row(pipe,orbit_plan,gather_plan,layout,shard,main_ptr,block_ptr,threads,grid_x,grid_y,slot);submit_s+=gdg_seconds(t);if(slot!=pipe.slots){std::cerr<<"gdpg slot mismatch "<<slot<<'/'<<pipe.slots<<'\n';return 5;}t=std::chrono::steady_clock::now();pipe.sync_row();row_sync_s+=gdg_seconds(t);std::cerr<<"row "<<row+1<<'/'<<W<<" submit_s="<<submit_s<<" row_sync_s="<<row_sync_s<<'\n';}double wall_s=gdg_seconds(wall0);
    Code final_rank=storage_rank_main_host(MateID(R),storage,layout);auto final_loc=gdm_locate_main(final_rank,layout,shard);Count answer=0;ck(cudaSetDevice(final_loc.first),"gdpg answer device");ck(cudaMemcpy(&answer,main_ptr[final_loc.first]+final_loc.second,sizeof(answer),cudaMemcpyDeviceToHost),"gdpg answer");
    std::cout<<"backend=gridfp-gpu-direct-atomicfree-multigpu-graphpart-v0.7 work_id="<<GDPG_WORK_ID<<" n="<<n<<" residue="<<answer<<" modulus="<<mod<<" gpus="<<ngpu
             <<" partition="<<(sel.graph_selected?"graph":"legacy-fallback")
             <<" p2p_reduction_pct="<<gdpg_reduction(runtime_legacy,runtime_selected)
             <<" total_bulk_p2p_gib_per_row="<<double(runtime_selected)/double(1ULL<<30)
             <<" gather_bulk_p2p_gib_per_row="<<double(gather_plan.copy_bytes_per_row)/double(1ULL<<30)
             <<" owner_orbit_bulk_p2p_gib_per_row="<<double(orbit_plan.copy_bytes_per_row)/double(1ULL<<30)
             <<" shard_max_gib="<<double(shard.max_elems*sizeof(Count))/double(1ULL<<30)<<" shard_min_gib="<<double(shard.min_elems*sizeof(Count))/double(1ULL<<30)
             <<" gather_peer_block_copies_per_row="<<peer_stats.copy_ops_per_row<<" owner_orbit_peer_block_copies_per_row="<<orbit_plan.block_copies_per_row<<" active_gather_peer_pairs="<<peer_stats.active_peer_pairs
             <<" threads="<<threads<<" grid_x="<<grid_x<<" grid_y="<<grid_y<<" submit_s="<<submit_s<<" row_sync_s="<<row_sync_s<<" prepare_s="<<prepare_s<<" wall_s="<<wall_s
             <<" closure_atomic=0 scratch_bytes=0 remote_scalar_gather_loads=0 remote_orbit_reads=0 remote_orbit_writes=0 fine_grain_p2p=0 bulk_p2p_staging=1 peer_copy_streams=1 host_barrier_points_per_row=1\n";
    pipe.destroy();for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdpg release device");cross_dev[d].release();ordinary_dev[d].release();base[d].release();if(main_ptr[d])cudaFree(main_ptr[d]);if(block_ptr[d])cudaFree(block_ptr[d]);}return 0;
}
