#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_segment_major_builder_microprobe_main_unused2
#include "gridfp_reduced_production_p2p_segment_major_builder_microprobe.cu"
#pragma pop_macro("main")

namespace {

struct MajorPlanCount {
    std::vector<unsigned long long> net_segments;
    std::vector<unsigned long long> net_runs;
    std::vector<unsigned long long> local_cycles;
    std::vector<unsigned long long> local_runs;
    unsigned long long network_states = 0;
    float count_ms = 0.0f;
};

MajorPlanCount device_major_plan_count(
    const ProductionFactorTables& tables,
    int W,
    int K,
    bool reverse,
    int ngpu,
    int batches,
    unsigned blocks
) {
    ck(cudaSetDevice(0), "major planner set device");
    install_tables(tables);
    const int ngroups = ngpu * batches * MAJOR_PC_CLASSES;
    const int nlocal = ngpu * MAJOR_PC_CLASSES;
    unsigned long long *d_ns=nullptr,*d_nr=nullptr,*d_lc=nullptr,*d_lr=nullptr,*d_states=nullptr;
    int* d_error=nullptr;
    ck(cudaMalloc(&d_ns, ngroups*sizeof(unsigned long long)),"major planner alloc ns");
    ck(cudaMalloc(&d_nr, ngroups*sizeof(unsigned long long)),"major planner alloc nr");
    ck(cudaMalloc(&d_lc, nlocal*sizeof(unsigned long long)),"major planner alloc lc");
    ck(cudaMalloc(&d_lr, nlocal*sizeof(unsigned long long)),"major planner alloc lr");
    ck(cudaMalloc(&d_states,sizeof(unsigned long long)),"major planner alloc states");
    ck(cudaMalloc(&d_error,sizeof(int)),"major planner alloc error");
    ck(cudaMemset(d_ns,0,ngroups*sizeof(unsigned long long)),"major planner zero ns");
    ck(cudaMemset(d_nr,0,ngroups*sizeof(unsigned long long)),"major planner zero nr");
    ck(cudaMemset(d_lc,0,nlocal*sizeof(unsigned long long)),"major planner zero lc");
    ck(cudaMemset(d_lr,0,nlocal*sizeof(unsigned long long)),"major planner zero lr");
    ck(cudaMemset(d_states,0,sizeof(unsigned long long)),"major planner zero states");
    ck(cudaMemset(d_error,0,sizeof(int)),"major planner zero error");

    const Rank64 base_supports=Rank64(1)<<(W-2);
    const unsigned threads=256;
    const Rank64 one_pass=(base_supports+threads-1)/threads;
    const unsigned launch_blocks=static_cast<unsigned>(
        std::max<Rank64>(1,std::min<Rank64>(blocks,one_pass)));
    cudaEvent_t a{},b{};
    ck(cudaEventCreate(&a),"major planner event a");
    ck(cudaEventCreate(&b),"major planner event b");
    ck(cudaEventRecord(a),"major planner record a");
    segment_major_count_kernel<<<launch_blocks,threads>>>(
        base_supports,W,K,reverse,ngpu,batches,
        d_ns,d_nr,d_lc,d_lr,d_states,d_error);
    ck(cudaGetLastError(),"major planner count launch");
    ck(cudaEventRecord(b),"major planner record b");
    ck(cudaEventSynchronize(b),"major planner count sync");

    MajorPlanCount out;
    out.net_segments.resize(static_cast<std::size_t>(ngroups));
    out.net_runs.resize(static_cast<std::size_t>(ngroups));
    out.local_cycles.resize(static_cast<std::size_t>(nlocal));
    out.local_runs.resize(static_cast<std::size_t>(nlocal));
    ck(cudaMemcpy(out.net_segments.data(),d_ns,ngroups*sizeof(unsigned long long),cudaMemcpyDeviceToHost),"major planner copy ns");
    ck(cudaMemcpy(out.net_runs.data(),d_nr,ngroups*sizeof(unsigned long long),cudaMemcpyDeviceToHost),"major planner copy nr");
    ck(cudaMemcpy(out.local_cycles.data(),d_lc,nlocal*sizeof(unsigned long long),cudaMemcpyDeviceToHost),"major planner copy lc");
    ck(cudaMemcpy(out.local_runs.data(),d_lr,nlocal*sizeof(unsigned long long),cudaMemcpyDeviceToHost),"major planner copy lr");
    ck(cudaMemcpy(&out.network_states,d_states,sizeof(unsigned long long),cudaMemcpyDeviceToHost),"major planner copy states");
    int error=0;
    ck(cudaMemcpy(&error,d_error,sizeof(error),cudaMemcpyDeviceToHost),"major planner copy error");
    if(error) fail("segment-major planner device error="+std::to_string(error));
    ck(cudaEventElapsedTime(&out.count_ms,a,b),"major planner elapsed");

    cudaEventDestroy(a);cudaEventDestroy(b);
    cudaFree(d_error);cudaFree(d_states);cudaFree(d_lr);cudaFree(d_lc);cudaFree(d_nr);cudaFree(d_ns);
    return out;
}

struct MajorPlanMemory {
    std::vector<Rank64> scratch_states;
    std::vector<Rank64> metadata_bytes;
    Rank64 total_segments=0,total_net_runs=0,total_local_cycles=0,total_local_runs=0;
};

MajorPlanMemory summarize_major_plan(
    const MajorPlanCount& c,
    int ngpu,
    int batches
) {
    MajorPlanMemory m;
    m.scratch_states.assign(static_cast<std::size_t>(ngpu),0);
    m.metadata_bytes.assign(static_cast<std::size_t>(ngpu),0);
    for(int g=0;g<ngpu;++g){
        Rank64 local_cycles=0,local_runs=0;
        for(int cls=0;cls<MAJOR_PC_CLASSES;++cls){
            const int idx=g*MAJOR_PC_CLASSES+cls;
            local_cycles+=c.local_cycles[static_cast<std::size_t>(idx)];
            local_runs+=c.local_runs[static_cast<std::size_t>(idx)];
        }
        m.total_local_cycles+=local_cycles;m.total_local_runs+=local_runs;
        m.metadata_bytes[static_cast<std::size_t>(g)]+=
            (local_cycles+1)*4ULL+local_runs*5ULL+
            Rank64(sizeof(P2PMajorGroupMeta))*MAJOR_PC_CLASSES;
        for(int b=0;b<batches;++b){
            Rank64 nseg=0,nrun=0,scratch=0;
            for(int cls=0;cls<MAJOR_PC_CLASSES;++cls){
                const int idx=major_group_index(g,b,cls,batches);
                const Rank64 ns=c.net_segments[static_cast<std::size_t>(idx)];
                const Rank64 nr=c.net_runs[static_cast<std::size_t>(idx)];
                nseg+=ns;nrun+=nr;scratch+=ns*catalan(cls+1);
            }
            m.total_segments+=nseg;m.total_net_runs+=nrun;
            m.scratch_states[static_cast<std::size_t>(g)]=std::max(
                m.scratch_states[static_cast<std::size_t>(g)],scratch);
            m.metadata_bytes[static_cast<std::size_t>(g)]+=
                (nseg+1)*4ULL+nseg*5ULL+nrun*5ULL+
                Rank64(sizeof(P2PMajorGroupMeta))*MAJOR_PC_CLASSES;
        }
    }
    return m;
}

void run_major_memory_plan(
    int W,int ngpu,int batches,unsigned blocks,double scratch_cap_gib
){
    if((W&1)||W<8||W>RP_MAX_W) fail("major planner width");
    const int K=(W-2)/2;
    ProductionFactorTables tables(W);
    const HostTilePlan tile=make_host_tile_plan(tables,K,ngpu);
    const MajorPlanCount f=device_major_plan_count(tables,W,K,false,ngpu,batches,blocks);
    const MajorPlanCount r=device_major_plan_count(tables,W,K,true,ngpu,batches,blocks);
    const MajorPlanMemory fm=summarize_major_plan(f,ngpu,batches);
    const MajorPlanMemory rm=summarize_major_plan(r,ngpu,batches);
    if(f.network_states!=r.network_states||fm.total_segments!=rm.total_segments)
        fail("major planner forward/reverse asymmetry");
    if(W==28&&ngpu==8&&f.network_states!=409769189454ULL)
        fail("major planner W28 exact payload");

    const double gib=double(1ULL<<30);
    const double b300_gib=288e9/gib;
    Rank64 max_scratch=0,max_metadata=0,max_state=0;
    for(int g=0;g<ngpu;++g){
        const Rank64 scratch=std::max(fm.scratch_states[static_cast<std::size_t>(g)],
                                      rm.scratch_states[static_cast<std::size_t>(g)]);
        const Rank64 metadata=fm.metadata_bytes[static_cast<std::size_t>(g)]+rm.metadata_bytes[static_cast<std::size_t>(g)];
        const Rank64 state=tile.owner_size[static_cast<std::size_t>(g)]*4ULL;
        max_scratch=std::max(max_scratch,scratch);max_metadata=std::max(max_metadata,metadata);max_state=std::max(max_state,state);
        const double used=double(state+scratch*4ULL+metadata)/gib;
        std::cout<<"major-plan gpu="<<g
                 <<" state_GiB="<<double(state)/gib
                 <<" scratch_GiB="<<double(scratch)*4.0/gib
                 <<" metadata_both_GiB="<<double(metadata)/gib
                 <<" total_GiB="<<used
                 <<" B300_decimal_headroom_GiB="<<(b300_gib-used)<<"\n";
    }
    const double max_scratch_gib=double(max_scratch)*4.0/gib;
    std::cout<<"gridfp-reduced-production-p2p-segment-major-plan"
             <<" W="<<W<<" K="<<K<<" ngpu="<<ngpu<<" batches="<<batches
             <<" exact_network_u32_per_redistribution="<<f.network_states
             <<" network_TiB="<<double(f.network_states)*4.0/double(1ULL<<40)
             <<" network_segments="<<fm.total_segments
             <<" network_local_runs="<<fm.total_net_runs
             <<" local_cycles="<<fm.total_local_cycles
             <<" local_runs="<<fm.total_local_runs
             <<" max_state_GiB="<<double(max_state)/gib
             <<" max_scratch_GiB="<<max_scratch_gib
             <<" max_metadata_both_GiB="<<double(max_metadata)/gib
             <<" scratch_cap_GiB="<<scratch_cap_gib
             <<" scratch_cap_ok="<<(max_scratch_gib<=scratch_cap_gib)
             <<" B300_288GB_decimal_GiB="<<b300_gib
             <<" forward_count_ms="<<f.count_ms<<" reverse_count_ms="<<r.count_ms
             <<" state_allocated_bytes=0 plan_only=1\n";
    if(max_scratch_gib>scratch_cap_gib) return;
}

} // namespace

int main(int argc,char** argv){
    const int W=argc>1?std::atoi(argv[1]):28;
    const int ngpu=argc>2?std::atoi(argv[2]):8;
    const int batches=argc>3?std::atoi(argv[3]):8;
    const unsigned blocks=argc>4?static_cast<unsigned>(std::strtoul(argv[4],nullptr,10)):4096u;
    const double scratch_cap_gib=argc>5?std::atof(argv[5]):32.0;
    if(W<8||W>RP_MAX_W||(W&1)||ngpu<2||ngpu>P2P_MAX_GPU||batches<1||batches>64||!blocks||scratch_cap_gib<=0)return 2;
    int visible=0;ck(cudaGetDeviceCount(&visible),"major planner device count");if(visible<1)return 3;
    run_major_memory_plan(W,ngpu,batches,blocks,scratch_cap_gib);
    std::cout<<"ALL_OK production_p2p_segment_major_memory_plan=1\n";
    return 0;
}
