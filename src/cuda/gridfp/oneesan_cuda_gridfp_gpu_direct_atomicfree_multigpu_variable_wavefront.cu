#include "ramstream32_gpu_direct_atomicfree_base.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_staged.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_peerstreams.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_orbitstage.cuh"

#define gdow_low_orbit_kernel gdow_low_orbit_kernel_v06_buggy_variable
#define gdow_high_orbit_kernel gdow_high_orbit_kernel_v06_buggy_variable
#define gdow_enqueue_row gdow_enqueue_row_v06_buggy_variable
#include "ramstream32_gpu_direct_atomicfree_multigpu_ownerorbit.cuh"
#undef gdow_low_orbit_kernel
#undef gdow_high_orbit_kernel
#undef gdow_enqueue_row

#include "ramstream32_gpu_direct_atomicfree_multigpu_ownerorbit_safe.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_graphpart.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_graphselect.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_phasepart.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_topomap.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_wavefront.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_adaptive_wavefront.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_online_wavefront.cuh"
#include "ramstream32_gpu_direct_atomicfree_multigpu_variable_wavefront_safe.cuh"

#include <string>

static constexpr const char* GDVW_WORK_ID="f8614f64-01f6-4c12-9264-03b44a385e74";

static void gdvw_enable_full_p2p(int ngpu){
    int peers=0;for(int a=0;a<ngpu;++a)for(int b=0;b<ngpu;++b)if(a!=b){int can=0;ck(cudaDeviceCanAccessPeer(&can,a,b),"gdvw can peer");if(!can)continue;ck(cudaSetDevice(a),"gdvw set p2p");cudaError_t e=cudaDeviceEnablePeerAccess(b,0);if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else ck(e,"gdvw enable peer");++peers;}
    if(ngpu>1&&peers!=ngpu*(ngpu-1)){std::cerr<<"gdvw requires full P2P: "<<peers<<'/'<<ngpu*(ngpu-1)<<'\n';std::exit(200);}
}
static std::string gdvw_map_text(const GdtpMapping&m,int ngpu){std::string s;for(int i=0;i<ngpu;++i){if(i)s.push_back(',');s+=std::to_string(unsigned(m.logical_to_physical[i]));}return s;}
static const char* gdvw_base_partition(const GdtpSelection&s){return s.phase_selected?"phase-peer":(s.graph_selected?"graph":"legacy-fallback");}
static std::string gdvw_hist_text(const GdvwPlan&p){std::string s;for(int k=GDVW_MIN_WAVES;k<=GDVW_MAX_WAVES;++k){if(k>GDVW_MIN_WAVES)s.push_back(',');s+=std::to_string(k);s.push_back(':');s+=std::to_string(p.phase_wave_hist[k]);}return s;}

int main(int argc,char**argv){
    int n=argc>1?std::atoi(argv[1]):TARGET_W-1;Count mod=argc>2?Count(std::strtoul(argv[2],nullptr,10)):4294967291u;int threads=argc>3?std::atoi(argv[3]):256;int grid_x=argc>4?std::atoi(argv[4]):32;int grid_y=argc>5?std::atoi(argv[5]):16;int requested=argc>6?std::atoi(argv[6]):0;bool plan_only=gdg_has_arg(argc,argv,"--plan-only");int W=n+1;
    if(W!=TARGET_W||n<2||W>MAXW||threads<=0||threads>1024||grid_x<=0||grid_y<=0)return 1;if constexpr(LOW_LUT_K+HIGH_LUT_K+1!=TARGET_W)return 1;if(requested<0||requested>GDM_MAX_GPU)return 1;

    auto prep0=std::chrono::steady_clock::now();build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectCrossHost forward=build_gpu_direct_cross(storage);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);
    size_t main_stage_bytes=size_t(layout.main_size)*sizeof(Count),block_stage_bytes=size_t(layout.block_size)*sizeof(Count);size_t base_resident=loworbit.rec.size()*sizeof(uint64_t)+(highdirect.orbit_ops.nn.size()+highdirect.orbit_ops.nrnl.size())*sizeof(CpuHighOrbitOp)+(highdirect.orbit_off.nn.size()+highdirect.orbit_off.nrnl.size())*sizeof(uint32_t);size_t ordinary_resident=ordinary.bytes()-ordinary.low_cross.size()*sizeof(GpuDirectLowCrossOp)-ordinary.low_cross_off.size()*sizeof(uint32_t);size_t resident_meta=base_resident+ordinary_resident+cross.bytes();size_t base_transient=(lowdesc.main_desc.size()+lowdesc.block_desc.size())*sizeof(uint32_t)+(highdirect.closure_ops.block.size()+highdirect.closure_ops.cross.size())*sizeof(CpuHighClosureOp)+(highdirect.closure_off.block.size()+highdirect.closure_off.cross.size())*sizeof(uint32_t)+(forward.high_rank.size()+forward.low_rank.size())*sizeof(uint32_t);size_t ordinary_transient=ordinary.low_cross.size()*sizeof(GpuDirectLowCrossOp)+ordinary.low_cross_off.size()*sizeof(uint32_t);size_t transient_meta=resident_meta+base_transient+ordinary_transient;

    int plan_ng=requested>0?requested:GDM_MAX_GPU;GdtpSelection ps=gdtp_select_exact(layout,loworbit,highdirect,ordinary,cross,plan_ng);GdwfPlan fixed=build_gdwf_plan(layout,ps.shard,ordinary,cross,ps.gather,plan_ng);GdvwPlan pv=gdvw_from_fixed(fixed,ps.gather,plan_ng);gdvw_recount(pv,ps.gather,plan_ng);GdmpPeerStats pp=gdmp_peer_stats(ps.gather,layout,ps.shard,plan_ng);double prepare_host_s=gdg_seconds(prep0);
    if(plan_only){std::cout<<"backend=gridfp-gpu-direct-atomicfree-multigpu-variable-wavefront-v1.3-plan work_id="<<GDVW_WORK_ID<<" n="<<n<<" gpus="<<plan_ng<<" base_partition="<<gdvw_base_partition(ps)<<" topology="<<(ps.topology.custom?"custom-gbps":"uniform")<<" topology_remapped="<<(ps.mapping.remapped?1:0)<<" topology_map="<<gdvw_map_text(ps.mapping,plan_ng)
        <<" wave_range="<<GDVW_MIN_WAVES<<'-'<<GDVW_MAX_WAVES<<" initial_waves=4 wave_group_model_us="<<gdvw_wave_group_ms()*1000.0<<" online_recalibration_rows=1,4,8 variable_wave_count_after_calibration=1"
        <<" total_p2p_gib_per_row="<<double(ps.byte_traffic.total_bytes)/double(1ULL<<30)<<" wavefront_eligible_p2p_gib_per_row="<<double(pv.eligible_copy_bytes_per_row)/double(1ULL<<30)<<" first_wave_p2p_gib_per_row="<<double(pv.first_wave_copy_bytes_per_row)/double(1ULL<<30)<<" overlap_candidate_p2p_gib_per_row="<<double(pv.overlap_candidate_bytes_per_row)/double(1ULL<<30)<<" serial_p1_p2p_gib_per_row="<<double(pv.serial_p1_copy_bytes_per_row)/double(1ULL<<30)
        <<" initial_phase_wave_hist="<<gdvw_hist_text(pv)<<" shard_max_gib="<<double(ps.shard.max_elems*sizeof(Count))/double(1ULL<<30)<<" shard_min_gib="<<double(ps.shard.min_elems*sizeof(Count))/double(1ULL<<30)<<" peak_per_gpu_gib="<<double(ps.shard.max_elems*sizeof(Count)+main_stage_bytes+block_stage_bytes+transient_meta)/double(1ULL<<30)<<" gather_peer_block_copies_per_row="<<pp.copy_ops_per_row
        <<" p2p_bytes_preserved=1 low_p1_serial_snapshot=1 variable_wave_count=1 online_wave_adaptation=1 closure_atomic=0 scratch_bytes=0 remote_scalar_gather_loads=0 remote_orbit_reads=0 remote_orbit_writes=0 fine_grain_p2p=0 host_barrier_points_per_row=1 prepare_s="<<prepare_host_s<<'\n';return 0;}

    int visible=0;ck(cudaGetDeviceCount(&visible),"gdvw device count");int ngpu=requested>0?std::min(requested,visible):std::min(visible,GDM_MAX_GPU);if(ngpu<1||ngpu>GDM_MAX_GPU)return 2;gdvw_enable_full_p2p(ngpu);
    GdtpSelection sel=gdtp_select_exact(layout,loworbit,highdirect,ordinary,cross,ngpu);GdmShardHost shard=std::move(sel.shard);GdmsStagePlan gather_plan=std::move(sel.gather);GdowOrbitPlan orbit_plan=std::move(sel.orbit);GdwfPlan fixed_runtime=build_gdwf_plan(layout,shard,ordinary,cross,gather_plan,ngpu);GdvwPlan wave_plan=gdvw_from_fixed(fixed_runtime,gather_plan,ngpu);gdvw_recount(wave_plan,gather_plan,ngpu);GdmpPeerStats peer_stats=gdmp_peer_stats(gather_plan,layout,shard,ngpu);GdptTraffic tr=gdpt_measure(gather_plan,orbit_plan,layout,shard,ngpu);

    std::array<size_t,GDM_MAX_GPU>free_before{},total_before{};for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdvw mem device");ck(cudaMemGetInfo(&free_before[d],&total_before[d]),"gdvw mem info");size_t need=size_t(shard.total_elems[d])*sizeof(Count)+main_stage_bytes+block_stage_bytes+transient_meta+GDVW_DBID_CAP;if(need>free_before[d]){std::cerr<<"gdvw insufficient HBM device="<<d<<" need_gib="<<double(need)/double(1ULL<<30)<<" free_gib="<<double(free_before[d])/double(1ULL<<30)<<'\n';return 4;}}
    Count*main_ptr[GDM_MAX_GPU]{};Count*block_ptr[GDM_MAX_GPU]{};for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdvw alloc device");if(shard.main_elems[d])ck(cudaMalloc(&main_ptr[d],size_t(shard.main_elems[d])*sizeof(Count)),"gdvw alloc main");if(shard.block_elems[d])ck(cudaMalloc(&block_ptr[d],size_t(shard.block_elems[d])*sizeof(Count)),"gdvw alloc block");if(shard.main_elems[d])ck(cudaMemset(main_ptr[d],0,size_t(shard.main_elems[d])*sizeof(Count)),"gdvw zero main");if(shard.block_elems[d])ck(cudaMemset(block_ptr[d],0,size_t(shard.block_elems[d])*sizeof(Count)),"gdvw zero block");}
    std::vector<GpuDirectDeviceTables>base(ngpu);std::vector<GpuDirectGatherDeviceTables>ordinary_dev(ngpu);std::vector<GpuDirectCrossGatherDeviceTables>cross_dev(ngpu);for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdvw metadata device");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"gdvw modulus");base[d].install(storage,layout,lowdesc,loworbit,highdirect,forward);ordinary_dev[d].install(ordinary);gpu_direct_gather_drop_redundant(base[d]);cross_dev[d].install(cross);gpu_direct_cross_gather_drop_redundant(base[d],ordinary_dev[d]);gdm_install_shards(shard,main_ptr,block_ptr,ngpu,d);}

    GdvwPipeline pipe;pipe.init(ngpu,layout);pipe.install_meta(wave_plan);double prepare_s=gdg_seconds(prep0);MateID init=MateID(R)<<(2*(W-1));Code init_rank=storage_rank_main_host(init,storage,layout);auto init_loc=gdm_locate_main(init_rank,layout,shard);Count one=1;ck(cudaSetDevice(init_loc.first),"gdvw init device");ck(cudaMemcpy(main_ptr[init_loc.first]+init_loc.second,&one,sizeof(one),cudaMemcpyHostToDevice),"gdvw init");

    GdorOnlineModel model;unsigned reconfigurations=0,calibration_rows=0,total_wave_count_changes=0;double last_gain_pct=0,last_base_ms=0,last_new_ms=0;double submit_s=0,row_sync_s=0;auto wall0=std::chrono::steady_clock::now();
    for(int row=0;row<W;++row){int slot=0;bool measure=gdor_measure_row(row);auto t=std::chrono::steady_clock::now();gdvw_enqueue_row_ready(pipe,orbit_plan,gather_plan,wave_plan,layout,shard,main_ptr,block_ptr,threads,grid_x,grid_y,slot,measure);submit_s+=gdg_seconds(t);if(slot!=pipe.slots){std::cerr<<"gdvw slot mismatch "<<slot<<'/'<<pipe.slots<<'\n';return 5;}t=std::chrono::steady_clock::now();pipe.sync_row();row_sync_s+=gdg_seconds(t);
        if(measure){++calibration_rows;GdawCalibration cs=gdvw_calibrate_compute(pipe,wave_plan,ngpu);GdorCopyCalibration xs=gdvw_calibrate_copy(pipe,wave_plan,layout,shard,ngpu);gdor_update_model(model,cs,xs,sel.topology,ngpu);GdvwReplan cand=gdvw_replan(wave_plan,model,layout,shard,ordinary,cross,gather_plan,ngpu);last_base_ms=cand.baseline_ms;last_new_ms=cand.candidate_ms;last_gain_pct=gdvw_replan_gain_pct(cand);bool accept=last_gain_pct>=GDOR_MIN_REPLAN_GAIN_PCT&&cand.candidate_ms+1e-6<cand.baseline_ms;unsigned wc=accept?gdvw_wave_count_changes(wave_plan,cand.plan):0;if(accept){wave_plan=std::move(cand.plan);pipe.refresh_meta(wave_plan);++reconfigurations;total_wave_count_changes+=wc;}
            std::cerr<<"variable-wave calibration row="<<row+1<<" compute_valid="<<(cs.valid?1:0)<<" copy_valid="<<(xs.valid?1:0)<<" model_updates="<<model.updates<<" accepted="<<(accept?1:0)<<" wave_count_changes="<<wc<<" phase_hist="<<gdvw_hist_text(wave_plan)<<" work_per_ms="<<model.compute.aggregate_work_per_ms<<" copy_mean_gbps="<<model.copy_mean_gbps<<" predicted_base_ms="<<last_base_ms<<" predicted_new_ms="<<last_new_ms<<" predicted_gain_pct="<<last_gain_pct<<'\n';}
        std::cerr<<"row "<<row+1<<'/'<<W<<" submit_s="<<submit_s<<" row_sync_s="<<row_sync_s<<'\n';}

    double wall_s=gdg_seconds(wall0);Code final_rank=storage_rank_main_host(MateID(R),storage,layout);auto final_loc=gdm_locate_main(final_rank,layout,shard);Count answer=0;ck(cudaSetDevice(final_loc.first),"gdvw answer device");ck(cudaMemcpy(&answer,main_ptr[final_loc.first]+final_loc.second,sizeof(answer),cudaMemcpyDeviceToHost),"gdvw answer");
    std::cout<<"backend=gridfp-gpu-direct-atomicfree-multigpu-variable-wavefront-v1.3 work_id="<<GDVW_WORK_ID<<" n="<<n<<" residue="<<answer<<" modulus="<<mod<<" gpus="<<ngpu<<" base_partition="<<gdvw_base_partition(sel)<<" topology="<<(sel.topology.custom?"custom-gbps":"runtime-calibrated")<<" topology_remapped="<<(sel.mapping.remapped?1:0)<<" topology_map="<<gdvw_map_text(sel.mapping,ngpu)
        <<" calibration_rows="<<calibration_rows<<" model_updates="<<model.updates<<" reconfigurations="<<reconfigurations<<" wave_count_changes="<<total_wave_count_changes<<" final_phase_wave_hist="<<gdvw_hist_text(wave_plan)<<" wave_group_model_us="<<gdvw_wave_group_ms()*1000.0
        <<" gather_work_per_ms="<<model.compute.aggregate_work_per_ms<<" effective_copy_gbps_mean="<<model.copy_mean_gbps<<" effective_copy_gbps_min="<<model.copy_min_gbps<<" effective_copy_gbps_max="<<model.copy_max_gbps<<" last_predicted_baseline_ms_per_row="<<last_base_ms<<" last_predicted_candidate_ms_per_row="<<last_new_ms<<" last_predicted_gain_pct="<<last_gain_pct
        <<" total_bulk_p2p_gib_per_row="<<double(tr.total_bytes)/double(1ULL<<30)<<" wavefront_eligible_p2p_gib_per_row="<<double(wave_plan.eligible_copy_bytes_per_row)/double(1ULL<<30)<<" first_wave_p2p_gib_per_row="<<double(wave_plan.first_wave_copy_bytes_per_row)/double(1ULL<<30)<<" overlap_candidate_p2p_gib_per_row="<<double(wave_plan.overlap_candidate_bytes_per_row)/double(1ULL<<30)<<" serial_p1_p2p_gib_per_row="<<double(wave_plan.serial_p1_copy_bytes_per_row)/double(1ULL<<30)<<" wave_launch_groups_per_row="<<wave_plan.wave_launch_groups_per_row
        <<" shard_max_gib="<<double(shard.max_elems*sizeof(Count))/double(1ULL<<30)<<" shard_min_gib="<<double(shard.min_elems*sizeof(Count))/double(1ULL<<30)<<" gather_peer_block_copies_per_row="<<peer_stats.copy_ops_per_row<<" owner_orbit_peer_block_copies_per_row="<<orbit_plan.block_copies_per_row<<" threads="<<threads<<" grid_x="<<grid_x<<" grid_y="<<grid_y<<" submit_s="<<submit_s<<" row_sync_s="<<row_sync_s<<" prepare_s="<<prepare_s<<" wall_s="<<wall_s
        <<" p2p_bytes_preserved=1 low_p1_serial_snapshot=1 variable_wave_count=1 online_wave_adaptation=1 ema_compute_and_copy=1 closure_atomic=0 scratch_bytes=0 remote_scalar_gather_loads=0 remote_orbit_reads=0 remote_orbit_writes=0 fine_grain_p2p=0 peer_copy_streams=1 host_barrier_points_per_row=1\n";

    pipe.destroy();for(int d=0;d<ngpu;++d){ck(cudaSetDevice(d),"gdvw release device");cross_dev[d].release();ordinary_dev[d].release();base[d].release();if(main_ptr[d])cudaFree(main_ptr[d]);if(block_ptr[d])cudaFree(block_ptr[d]);}return 0;
}
