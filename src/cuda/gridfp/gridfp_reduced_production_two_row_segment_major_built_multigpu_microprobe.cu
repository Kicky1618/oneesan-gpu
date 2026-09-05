#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_two_row_segment_major_multigpu_microprobe_main_unused2
#include "gridfp_reduced_production_two_row_segment_major_multigpu_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_p2p_segment_major_builder.cuh"

namespace {

struct RtBuiltCount {
    std::vector<unsigned long long> net_segments;
    std::vector<unsigned long long> net_runs;
    std::vector<unsigned long long> local_cycles;
    std::vector<unsigned long long> local_runs;
    unsigned long long network_states = 0;
};

struct RtBuiltLayout {
    std::vector<Rank64> net_header_base, net_source_base, net_run_base;
    std::vector<Rank64> net_group_segment_begin, net_group_run_begin;
    std::vector<Rank64> local_header_base, local_run_base;
    std::vector<Rank64> local_group_cycle_begin, local_group_run_begin;
    std::vector<std::array<P2PMajorGroupMeta,RP_P2P_MAJOR_PC_CLASSES>> net_group;
    std::vector<std::array<P2PMajorGroupMeta,RP_P2P_MAJOR_PC_CLASSES>> local_group;
    std::vector<Rank64> scratch_states;
    std::vector<Rank64> net_segments_per_batch, net_runs_per_batch;
    std::vector<Rank64> local_cycles_per_gpu, local_runs_per_gpu;
    Rank64 total_net_headers = 0, total_net_segments = 0, total_net_runs = 0;
    Rank64 total_local_headers = 0, total_local_runs = 0;
};

unsigned rtbuild_setup_blocks(Rank64 work, unsigned cap) {
    const Rank64 one_pass = (work + 255) / 256;
    return static_cast<unsigned>(std::max<Rank64>(1, std::min<Rank64>(cap, one_pass)));
}

RtBuiltCount rtbuild_count_direction(
    const ProductionFactorTables& tables,
    int W, int K, bool reverse, int ngpu, int batches,
    std::uint32_t salt, unsigned blocks
) {
    ck(cudaSetDevice(0), "rtbuild count set device");
    install_tables(tables);
    const int ngroups = ngpu * batches * RP_P2P_MAJOR_PC_CLASSES;
    const int nlocal = ngpu * RP_P2P_MAJOR_PC_CLASSES;
    unsigned long long *d_ns=nullptr,*d_nr=nullptr,*d_lc=nullptr,*d_lr=nullptr,*d_states=nullptr;
    int* d_error=nullptr;
    ck(cudaMalloc(&d_ns,ngroups*sizeof(unsigned long long)),"rtbuild count ns");
    ck(cudaMalloc(&d_nr,ngroups*sizeof(unsigned long long)),"rtbuild count nr");
    ck(cudaMalloc(&d_lc,nlocal*sizeof(unsigned long long)),"rtbuild count lc");
    ck(cudaMalloc(&d_lr,nlocal*sizeof(unsigned long long)),"rtbuild count lr");
    ck(cudaMalloc(&d_states,sizeof(unsigned long long)),"rtbuild count states");
    ck(cudaMalloc(&d_error,sizeof(int)),"rtbuild count error");
    ck(cudaMemset(d_ns,0,ngroups*sizeof(unsigned long long)),"rtbuild zero ns");
    ck(cudaMemset(d_nr,0,ngroups*sizeof(unsigned long long)),"rtbuild zero nr");
    ck(cudaMemset(d_lc,0,nlocal*sizeof(unsigned long long)),"rtbuild zero lc");
    ck(cudaMemset(d_lr,0,nlocal*sizeof(unsigned long long)),"rtbuild zero lr");
    ck(cudaMemset(d_states,0,sizeof(unsigned long long)),"rtbuild zero states");
    ck(cudaMemset(d_error,0,sizeof(int)),"rtbuild zero error");

    const Rank64 base_supports=Rank64(1)<<(W-2);
    p2p_major_count_kernel<<<rtbuild_setup_blocks(base_supports,blocks),256>>>(
        base_supports,W,K,reverse,ngpu,batches,salt,
        d_ns,d_nr,d_lc,d_lr,d_states,d_error);
    ck(cudaGetLastError(),"rtbuild count launch");
    ck(cudaDeviceSynchronize(),"rtbuild count sync");

    RtBuiltCount out;
    out.net_segments.resize(static_cast<std::size_t>(ngroups));
    out.net_runs.resize(static_cast<std::size_t>(ngroups));
    out.local_cycles.resize(static_cast<std::size_t>(nlocal));
    out.local_runs.resize(static_cast<std::size_t>(nlocal));
    ck(cudaMemcpy(out.net_segments.data(),d_ns,ngroups*sizeof(unsigned long long),cudaMemcpyDeviceToHost),"rtbuild copy ns");
    ck(cudaMemcpy(out.net_runs.data(),d_nr,ngroups*sizeof(unsigned long long),cudaMemcpyDeviceToHost),"rtbuild copy nr");
    ck(cudaMemcpy(out.local_cycles.data(),d_lc,nlocal*sizeof(unsigned long long),cudaMemcpyDeviceToHost),"rtbuild copy lc");
    ck(cudaMemcpy(out.local_runs.data(),d_lr,nlocal*sizeof(unsigned long long),cudaMemcpyDeviceToHost),"rtbuild copy lr");
    ck(cudaMemcpy(&out.network_states,d_states,sizeof(unsigned long long),cudaMemcpyDeviceToHost),"rtbuild copy states");
    int error=0;
    ck(cudaMemcpy(&error,d_error,sizeof(error),cudaMemcpyDeviceToHost),"rtbuild copy count error");
    if(error) fail("rtbuild count device error="+std::to_string(error));
    cudaFree(d_error);cudaFree(d_states);cudaFree(d_lr);cudaFree(d_lc);cudaFree(d_nr);cudaFree(d_ns);
    return out;
}

RtBuiltLayout rtbuild_layout(
    const RtBuiltCount& c, int ngpu, int batches
) {
    RtBuiltLayout x;
    const int ngroups=ngpu*batches*RP_P2P_MAJOR_PC_CLASSES;
    const int nlocal=ngpu*RP_P2P_MAJOR_PC_CLASSES;
    x.net_header_base.resize(static_cast<std::size_t>(ngpu*batches));
    x.net_source_base.resize(static_cast<std::size_t>(ngpu*batches));
    x.net_run_base.resize(static_cast<std::size_t>(ngpu*batches));
    x.net_group_segment_begin.resize(static_cast<std::size_t>(ngroups));
    x.net_group_run_begin.resize(static_cast<std::size_t>(ngroups));
    x.local_header_base.resize(static_cast<std::size_t>(ngpu));
    x.local_run_base.resize(static_cast<std::size_t>(ngpu));
    x.local_group_cycle_begin.resize(static_cast<std::size_t>(nlocal));
    x.local_group_run_begin.resize(static_cast<std::size_t>(nlocal));
    x.net_group.resize(static_cast<std::size_t>(ngpu*batches));
    x.local_group.resize(static_cast<std::size_t>(ngpu));
    x.scratch_states.assign(static_cast<std::size_t>(ngpu),0);
    x.net_segments_per_batch.assign(static_cast<std::size_t>(ngpu*batches),0);
    x.net_runs_per_batch.assign(static_cast<std::size_t>(ngpu*batches),0);
    x.local_cycles_per_gpu.assign(static_cast<std::size_t>(ngpu),0);
    x.local_runs_per_gpu.assign(static_cast<std::size_t>(ngpu),0);

    for(int g=0;g<ngpu;++g){
        x.local_header_base[static_cast<std::size_t>(g)]=x.total_local_headers;
        x.local_run_base[static_cast<std::size_t>(g)]=x.total_local_runs;
        Rank64 cp=0,rp=0;
        for(int cls=0;cls<RP_P2P_MAJOR_PC_CLASSES;++cls){
            const int idx=g*RP_P2P_MAJOR_PC_CLASSES+cls;
            const Rank64 nc=c.local_cycles[static_cast<std::size_t>(idx)];
            const Rank64 nr=c.local_runs[static_cast<std::size_t>(idx)];
            x.local_group_cycle_begin[static_cast<std::size_t>(idx)]=cp;
            x.local_group_run_begin[static_cast<std::size_t>(idx)]=rp;
            auto& gm=x.local_group[static_cast<std::size_t>(g)][static_cast<std::size_t>(cls)];
            if(cp+nc>= (Rank64(1)<<32) || rp+nr >= (Rank64(1)<<32))
                fail("rtbuild local u32 range");
            gm.begin=static_cast<std::uint32_t>(cp);
            gm.end=static_cast<std::uint32_t>(cp+nc);
            gm.pc=catalan(cls+1);
            cp+=nc;rp+=nr;
        }
        x.local_cycles_per_gpu[static_cast<std::size_t>(g)]=cp;
        x.local_runs_per_gpu[static_cast<std::size_t>(g)]=rp;
        x.total_local_headers+=cp+1;x.total_local_runs+=rp;

        for(int b=0;b<batches;++b){
            const int gb=g*batches+b;
            x.net_header_base[static_cast<std::size_t>(gb)]=x.total_net_headers;
            x.net_source_base[static_cast<std::size_t>(gb)]=x.total_net_segments;
            x.net_run_base[static_cast<std::size_t>(gb)]=x.total_net_runs;
            Rank64 sp=0,rp2=0,scratch=0;
            for(int cls=0;cls<RP_P2P_MAJOR_PC_CLASSES;++cls){
                const int idx=p2p_major_group_index_device(g,b,cls,batches);
                const Rank64 ns=c.net_segments[static_cast<std::size_t>(idx)];
                const Rank64 nr=c.net_runs[static_cast<std::size_t>(idx)];
                x.net_group_segment_begin[static_cast<std::size_t>(idx)]=sp;
                x.net_group_run_begin[static_cast<std::size_t>(idx)]=rp2;
                auto& gm=x.net_group[static_cast<std::size_t>(gb)][static_cast<std::size_t>(cls)];
                if(sp+ns >= (Rank64(1)<<32) || rp2+nr >= (Rank64(1)<<32))
                    fail("rtbuild network u32 range");
                gm.begin=static_cast<std::uint32_t>(sp);
                gm.end=static_cast<std::uint32_t>(sp+ns);
                gm.scratch_base=scratch;
                gm.pc=catalan(cls+1);
                sp+=ns;rp2+=nr;scratch+=ns*gm.pc;
            }
            x.net_segments_per_batch[static_cast<std::size_t>(gb)]=sp;
            x.net_runs_per_batch[static_cast<std::size_t>(gb)]=rp2;
            x.scratch_states[static_cast<std::size_t>(g)]=std::max(
                x.scratch_states[static_cast<std::size_t>(g)],scratch);
            x.total_net_headers+=sp+1;x.total_net_segments+=sp;x.total_net_runs+=rp2;
        }
    }
    return x;
}

template<class T>
void rtbuild_upload_vector(T*& dst,const std::vector<T>& src,const char* what){
    ck(cudaMalloc(&dst,std::max<std::size_t>(1,src.size())*sizeof(T)),what);
    if(!src.empty())ck(cudaMemcpy(dst,src.data(),src.size()*sizeof(T),cudaMemcpyHostToDevice),what);
}

struct RtBuildTemp {
    std::uint32_t *net_header=nullptr,*source_low=nullptr,*net_local_low=nullptr;
    std::uint8_t *source_high=nullptr,*net_local_high=nullptr;
    std::uint32_t *local_header=nullptr,*local_low=nullptr;
    std::uint8_t* local_high=nullptr;
};

void rtbuild_set_sentinels(const RtBuiltLayout& x,RtBuildTemp& t,int ngpu,int batches){
    for(int g=0;g<ngpu;++g){
        const std::uint32_t end=static_cast<std::uint32_t>(x.local_runs_per_gpu[static_cast<std::size_t>(g)]);
        ck(cudaMemcpy(t.local_header+x.local_header_base[static_cast<std::size_t>(g)]+x.local_cycles_per_gpu[static_cast<std::size_t>(g)],&end,sizeof(end),cudaMemcpyHostToDevice),"rtbuild local sentinel");
        for(int b=0;b<batches;++b){
            const int gb=g*batches+b;
            const std::uint32_t bend=static_cast<std::uint32_t>(x.net_runs_per_batch[static_cast<std::size_t>(gb)]);
            ck(cudaMemcpy(t.net_header+x.net_header_base[static_cast<std::size_t>(gb)]+x.net_segments_per_batch[static_cast<std::size_t>(gb)],&bend,sizeof(bend),cudaMemcpyHostToDevice),"rtbuild net sentinel");
        }
    }
}

void rtbuild_validate_cursors(
    const RtBuiltCount& c,int ngpu,int batches,
    const std::vector<unsigned long long>& net,
    const std::vector<unsigned long long>& local
){
    for(std::size_t i=0;i<net.size();++i){
        if((net[i]>>32)!=c.net_segments[i] || (net[i]&0xffffffffULL)!=c.net_runs[i])
            fail("rtbuild paired network cursor mismatch");
    }
    for(std::size_t i=0;i<local.size();++i){
        if((local[i]>>32)!=c.local_cycles[i] || (local[i]&0xffffffffULL)!=c.local_runs[i])
            fail("rtbuild paired local cursor mismatch");
    }
}

RtBuildTemp rtbuild_fill_direction(
    const ProductionFactorTables& tables,const HostTilePlan& tile,
    const RtBuiltCount& c,const RtBuiltLayout& x,
    int W,int K,bool reverse,int ngpu,int batches,std::uint32_t salt,unsigned blocks
){
    ck(cudaSetDevice(0),"rtbuild fill set device");install_tables(tables);
    Rank64 *d_owner=nullptr,*d_nhb=nullptr,*d_nsb=nullptr,*d_nrb=nullptr,*d_ngs=nullptr,*d_ngr=nullptr,*d_lhb=nullptr,*d_lrb=nullptr,*d_lgc=nullptr,*d_lgr=nullptr;
    rtbuild_upload_vector(d_owner,tile.owner_begin,"rtbuild owner begin");
    rtbuild_upload_vector(d_nhb,x.net_header_base,"rtbuild nhb");rtbuild_upload_vector(d_nsb,x.net_source_base,"rtbuild nsb");rtbuild_upload_vector(d_nrb,x.net_run_base,"rtbuild nrb");
    rtbuild_upload_vector(d_ngs,x.net_group_segment_begin,"rtbuild ngs");rtbuild_upload_vector(d_ngr,x.net_group_run_begin,"rtbuild ngr");
    rtbuild_upload_vector(d_lhb,x.local_header_base,"rtbuild lhb");rtbuild_upload_vector(d_lrb,x.local_run_base,"rtbuild lrb");rtbuild_upload_vector(d_lgc,x.local_group_cycle_begin,"rtbuild lgc");rtbuild_upload_vector(d_lgr,x.local_group_run_begin,"rtbuild lgr");
    const int ngroups=ngpu*batches*RP_P2P_MAJOR_PC_CLASSES,nlocal=ngpu*RP_P2P_MAJOR_PC_CLASSES;
    unsigned long long *d_nc=nullptr,*d_lc=nullptr;ck(cudaMalloc(&d_nc,ngroups*sizeof(unsigned long long)),"rtbuild nc");ck(cudaMalloc(&d_lc,nlocal*sizeof(unsigned long long)),"rtbuild lc");ck(cudaMemset(d_nc,0,ngroups*sizeof(unsigned long long)),"rtbuild zero nc");ck(cudaMemset(d_lc,0,nlocal*sizeof(unsigned long long)),"rtbuild zero lc");
    RtBuildTemp t;
    ck(cudaMalloc(&t.net_header,std::max<Rank64>(1,x.total_net_headers)*4),"rtbuild net header");
    ck(cudaMalloc(&t.source_low,std::max<Rank64>(1,x.total_net_segments)*4),"rtbuild source low");ck(cudaMalloc(&t.source_high,std::max<Rank64>(1,x.total_net_segments)),"rtbuild source high");
    ck(cudaMalloc(&t.net_local_low,std::max<Rank64>(1,x.total_net_runs)*4),"rtbuild net low");ck(cudaMalloc(&t.net_local_high,std::max<Rank64>(1,x.total_net_runs)),"rtbuild net high");
    ck(cudaMalloc(&t.local_header,std::max<Rank64>(1,x.total_local_headers)*4),"rtbuild local header");ck(cudaMalloc(&t.local_low,std::max<Rank64>(1,x.total_local_runs)*4),"rtbuild local low");ck(cudaMalloc(&t.local_high,std::max<Rank64>(1,x.total_local_runs)),"rtbuild local high");
    int* d_error=nullptr;ck(cudaMalloc(&d_error,sizeof(int)),"rtbuild fill error");ck(cudaMemset(d_error,0,sizeof(int)),"rtbuild zero fill error");
    const Rank64 base_supports=Rank64(1)<<(W-2);
    p2p_major_fill_kernel<<<rtbuild_setup_blocks(base_supports,blocks),256>>>(
        base_supports,W,K,reverse,ngpu,batches,salt,d_owner,
        d_nhb,d_nsb,d_nrb,d_ngs,d_ngr,d_lhb,d_lrb,d_lgc,d_lgr,
        d_nc,d_lc,t.net_header,t.source_low,t.source_high,t.net_local_low,t.net_local_high,
        t.local_header,t.local_low,t.local_high,d_error);
    ck(cudaGetLastError(),"rtbuild fill launch");ck(cudaDeviceSynchronize(),"rtbuild fill sync");
    int error=0;ck(cudaMemcpy(&error,d_error,sizeof(error),cudaMemcpyDeviceToHost),"rtbuild fill error copy");if(error)fail("rtbuild fill device error="+std::to_string(error));
    std::vector<unsigned long long> hn(static_cast<std::size_t>(ngroups)),hl(static_cast<std::size_t>(nlocal));
    ck(cudaMemcpy(hn.data(),d_nc,ngroups*sizeof(unsigned long long),cudaMemcpyDeviceToHost),"rtbuild net cursor copy");ck(cudaMemcpy(hl.data(),d_lc,nlocal*sizeof(unsigned long long),cudaMemcpyDeviceToHost),"rtbuild local cursor copy");
    rtbuild_validate_cursors(c,ngpu,batches,hn,hl);rtbuild_set_sentinels(x,t,ngpu,batches);
    cudaFree(d_error);cudaFree(d_lc);cudaFree(d_nc);cudaFree(d_lgr);cudaFree(d_lgc);cudaFree(d_lrb);cudaFree(d_lhb);cudaFree(d_ngr);cudaFree(d_ngs);cudaFree(d_nrb);cudaFree(d_nsb);cudaFree(d_nhb);cudaFree(d_owner);
    return t;
}

void rtbuild_copy_peer(void* dst,int dg,const void* src,int sg,std::size_t bytes,const char* what){
    if(!bytes)return;
    if(dg==sg){ck(cudaSetDevice(dg),what);ck(cudaMemcpy(dst,src,bytes,cudaMemcpyDeviceToDevice),what);}
    else ck(cudaMemcpyPeer(dst,dg,src,sg,bytes),what);
}

template<class T>
T* rtbuild_alloc_on(int g,Rank64 n,const char* what){
    ck(cudaSetDevice(g),what);T* p=nullptr;ck(cudaMalloc(&p,std::max<Rank64>(1,n)*sizeof(T)),what);return p;
}

std::vector<RtMajorDevicePlan> rtbuild_distribute_direction(
    const RtBuiltLayout& x,const RtBuildTemp& t,int ngpu,int batches
){
    std::vector<RtMajorDevicePlan> out(static_cast<std::size_t>(ngpu));
    for(int g=0;g<ngpu;++g){
        auto& p=out[static_cast<std::size_t>(g)];p.batch.resize(static_cast<std::size_t>(batches));
        const Rank64 nc=x.local_cycles_per_gpu[static_cast<std::size_t>(g)],nr=x.local_runs_per_gpu[static_cast<std::size_t>(g)];
        p.local.cycles=nc;
        p.local.run_begin=rtbuild_alloc_on<std::uint32_t>(g,nc+1,"rtbuild alloc local header dst");p.local.local_low=rtbuild_alloc_on<std::uint32_t>(g,nr,"rtbuild alloc local low dst");p.local.local_high=rtbuild_alloc_on<std::uint8_t>(g,nr,"rtbuild alloc local high dst");p.local.group=rtbuild_alloc_on<P2PMajorGroupMeta>(g,RP_P2P_MAJOR_PC_CLASSES,"rtbuild alloc local group dst");
        rtbuild_copy_peer(p.local.run_begin,g,t.local_header+x.local_header_base[static_cast<std::size_t>(g)],0,(nc+1)*4ULL,"rtbuild copy local header peer");rtbuild_copy_peer(p.local.local_low,g,t.local_low+x.local_run_base[static_cast<std::size_t>(g)],0,nr*4ULL,"rtbuild copy local low peer");rtbuild_copy_peer(p.local.local_high,g,t.local_high+x.local_run_base[static_cast<std::size_t>(g)],0,nr,"rtbuild copy local high peer");ck(cudaSetDevice(g),"rtbuild local group host");ck(cudaMemcpy(p.local.group,x.local_group[static_cast<std::size_t>(g)].data(),sizeof(P2PMajorGroupMeta)*RP_P2P_MAJOR_PC_CLASSES,cudaMemcpyHostToDevice),"rtbuild local group host");
        for(int b=0;b<batches;++b){const int gb=g*batches+b;auto& d=p.batch[static_cast<std::size_t>(b)];const Rank64 ns=x.net_segments_per_batch[static_cast<std::size_t>(gb)],rr=x.net_runs_per_batch[static_cast<std::size_t>(gb)];d.segments=ns;d.run_begin=rtbuild_alloc_on<std::uint32_t>(g,ns+1,"rtbuild alloc net header dst");d.source_low=rtbuild_alloc_on<std::uint32_t>(g,ns,"rtbuild alloc source low dst");d.source_high=rtbuild_alloc_on<std::uint8_t>(g,ns,"rtbuild alloc source high dst");d.local_low=rtbuild_alloc_on<std::uint32_t>(g,rr,"rtbuild alloc net low dst");d.local_high=rtbuild_alloc_on<std::uint8_t>(g,rr,"rtbuild alloc net high dst");d.group=rtbuild_alloc_on<P2PMajorGroupMeta>(g,RP_P2P_MAJOR_PC_CLASSES,"rtbuild alloc net group dst");
            rtbuild_copy_peer(d.run_begin,g,t.net_header+x.net_header_base[static_cast<std::size_t>(gb)],0,(ns+1)*4ULL,"rtbuild copy net header peer");rtbuild_copy_peer(d.source_low,g,t.source_low+x.net_source_base[static_cast<std::size_t>(gb)],0,ns*4ULL,"rtbuild copy source low peer");rtbuild_copy_peer(d.source_high,g,t.source_high+x.net_source_base[static_cast<std::size_t>(gb)],0,ns,"rtbuild copy source high peer");rtbuild_copy_peer(d.local_low,g,t.net_local_low+x.net_run_base[static_cast<std::size_t>(gb)],0,rr*4ULL,"rtbuild copy net low peer");rtbuild_copy_peer(d.local_high,g,t.net_local_high+x.net_run_base[static_cast<std::size_t>(gb)],0,rr,"rtbuild copy net high peer");ck(cudaSetDevice(g),"rtbuild net group host");ck(cudaMemcpy(d.group,x.net_group[static_cast<std::size_t>(gb)].data(),sizeof(P2PMajorGroupMeta)*RP_P2P_MAJOR_PC_CLASSES,cudaMemcpyHostToDevice),"rtbuild net group host");}
    }
    return out;
}

void rtbuild_free_temp(RtBuildTemp& t){ck(cudaSetDevice(0),"rtbuild free temp");cudaFree(t.local_high);cudaFree(t.local_low);cudaFree(t.local_header);cudaFree(t.net_local_high);cudaFree(t.net_local_low);cudaFree(t.source_high);cudaFree(t.source_low);cudaFree(t.net_header);}

struct RtBuiltDirection {
    std::vector<RtMajorDevicePlan> device;
    RtBuiltLayout layout;
    unsigned long long network_states=0;
};

RtBuiltDirection rtbuild_compile_direction(
    const ProductionFactorTables& tables,const HostTilePlan& tile,
    int W,int K,bool reverse,int ngpu,int batches,std::uint32_t salt,unsigned blocks
){
    const RtBuiltCount count=rtbuild_count_direction(tables,W,K,reverse,ngpu,batches,salt,blocks);
    RtBuiltDirection out;out.layout=rtbuild_layout(count,ngpu,batches);out.network_states=count.network_states;
    RtBuildTemp temp=rtbuild_fill_direction(tables,tile,count,out.layout,W,K,reverse,ngpu,batches,salt,blocks);
    out.device=rtbuild_distribute_direction(out.layout,temp,ngpu,batches);rtbuild_free_temp(temp);return out;
}

void run_two_row_segment_major_built(
    int W,int ngpu,int batches,unsigned blocks,std::uint32_t mod,std::uint32_t salt
){
    const int K=(W-2)/2;ProductionFactorTables tables(W);const HostTilePlan tile=make_host_tile_plan(tables,K,ngpu);const OwnerPlan owner_plan{tile.owner_begin,tile.owner_size};const auto main_words=gen_words(W);
    ModMap initial;Rank64 serial=0;for(MateID m:main_words){const std::uint32_t v=static_cast<std::uint32_t>(1+(serial++*2654435761ULL)%(mod-1ULL));initial.emplace(Key{false,m},v);}ModMap ref=initial;for(int p=W-1;p>=1;--p)ref=schedule_raw_step(ref,W,p,false,mod);for(int p=1;p<W;++p)ref=schedule_raw_step(ref,W,p,true,mod);ref=schedule_projected_step(ref,W,W-1,false,mod);
    std::vector<std::uint32_t> flat_input(static_cast<std::size_t>(tables.size()),0),flat_expected(static_cast<std::size_t>(tables.size()),0);for(const auto&[k,v]:initial){const GroupedRank gr=grouped_rank(k,tables,W,W-1,false,W-1,K,ngpu,owner_plan);flat_input[static_cast<std::size_t>(tile.shard_base[gr.owner]+gr.local)]=v;}for(const auto&[k,v]:ref){const GroupedRank gr=grouped_rank(k,tables,W,W-2,false,W-1,K,ngpu,owner_plan);flat_expected[static_cast<std::size_t>(tile.shard_base[gr.owner]+gr.local)]=v;}

    schedule_enable_peer(ngpu);
    RtBuiltDirection fwd=rtbuild_compile_direction(tables,tile,W,K,false,ngpu,batches,salt,blocks);
    RtBuiltDirection rev=rtbuild_compile_direction(tables,tile,W,K,true,ngpu,batches,salt,blocks);
    if(fwd.network_states!=rev.network_states)fail("rtbuild forward/reverse payload asymmetry");

    std::vector<RuntimeScheduleDevice> dev(static_cast<std::size_t>(ngpu));std::vector<std::uint32_t*> peer_ptr(static_cast<std::size_t>(ngpu)),scratch(static_cast<std::size_t>(ngpu),nullptr);std::vector<HostOwnerComponentPlan> ip(static_cast<std::size_t>(ngpu)),cp(static_cast<std::size_t>(ngpu));
    for(int g=0;g<ngpu;++g){ck(cudaSetDevice(g),"rtbuild state set device");install_tables(tables);ip[static_cast<std::size_t>(g)]=make_host_owner_component_plan(tables,K,g,ngpu);cp[static_cast<std::size_t>(g)]=make_host_turn_compress_plan(tables,K,g,ngpu);auto& d=dev[static_cast<std::size_t>(g)];d.interior_components=ip[static_cast<std::size_t>(g)].prefix.back();d.compress_components=cp[static_cast<std::size_t>(g)].prefix.back();const Rank64 nstate=tile.owner_size[static_cast<std::size_t>(g)];ck(cudaMalloc(&d.state,nstate*4ULL),"rtbuild alloc state");peer_ptr[static_cast<std::size_t>(g)]=d.state;const Rank64 base=tile.shard_base[static_cast<std::size_t>(g)];ck(cudaMemcpy(d.state,flat_input.data()+base,nstate*4ULL,cudaMemcpyHostToDevice),"rtbuild copy state");ck(cudaMalloc(&d.owner_begin,ngpu*sizeof(Rank64)),"rtbuild state owner");ck(cudaMemcpy(d.owner_begin,tile.owner_begin.data(),ngpu*sizeof(Rank64),cudaMemcpyHostToDevice),"rtbuild state owner copy");runtime_upload_plan(ip[static_cast<std::size_t>(g)],d.iprefix,d.isr,d.icg);runtime_upload_plan(cp[static_cast<std::size_t>(g)],d.cprefix,d.csr,d.ccg);ck(cudaMalloc(&d.error,sizeof(int)),"rtbuild state error");ck(cudaMemset(d.error,0,sizeof(int)),"rtbuild state zero error");const Rank64 nscratch=std::max<Rank64>(1,std::max(fwd.layout.scratch_states[static_cast<std::size_t>(g)],rev.layout.scratch_states[static_cast<std::size_t>(g)]));ck(cudaMalloc(&scratch[static_cast<std::size_t>(g)],nscratch*4ULL),"rtbuild alloc scratch");}
    for(int g=0;g<ngpu;++g){ck(cudaSetDevice(g),"rtbuild peer table");auto& d=dev[static_cast<std::size_t>(g)];ck(cudaMalloc(&d.peer,ngpu*sizeof(std::uint32_t*)),"rtbuild peer alloc");ck(cudaMemcpy(d.peer,peer_ptr.data(),ngpu*sizeof(std::uint32_t*),cudaMemcpyHostToDevice),"rtbuild peer copy");}

    const auto t0=std::chrono::steady_clock::now();runtime_launch_turn_all(dev,W,K,true,true,ngpu,blocks,mod);for(int p=W-2;p>=K+2;--p)runtime_launch_interior_all(dev,W,p,false,W-1,K,ngpu,blocks,mod);runtime_sync_all(ngpu);rtmajor_launch_redistribution(dev,fwd.device,scratch,ngpu,batches,blocks);for(int p=K+1;p>=2;--p)runtime_launch_interior_all(dev,W,p,false,K+1,K,ngpu,blocks,mod);runtime_launch_turn_all(dev,W,K,false,false,ngpu,blocks,mod);runtime_launch_turn_all(dev,W,K,false,true,ngpu,blocks,mod);for(int p=2;p<=K;++p)runtime_launch_interior_all(dev,W,p,true,1,K,ngpu,blocks,mod);runtime_sync_all(ngpu);rtmajor_launch_redistribution(dev,rev.device,scratch,ngpu,batches,blocks);for(int p=K+1;p<=W-2;++p)runtime_launch_interior_all(dev,W,p,true,K+1,K,ngpu,blocks,mod);runtime_launch_turn_all(dev,W,K,true,false,ngpu,blocks,mod);runtime_launch_turn_all(dev,W,K,true,true,ngpu,blocks,mod);runtime_sync_all(ngpu);const double wall_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();

    std::vector<std::uint32_t> flat_output(flat_expected.size());Rank64 max_scratch=0;for(int g=0;g<ngpu;++g){ck(cudaSetDevice(g),"rtbuild result device");auto& d=dev[static_cast<std::size_t>(g)];int error=0;ck(cudaMemcpy(&error,d.error,sizeof(error),cudaMemcpyDeviceToHost),"rtbuild result error");if(error)fail("rtbuild two-row device error="+std::to_string(error));const Rank64 nstate=tile.owner_size[static_cast<std::size_t>(g)],base=tile.shard_base[static_cast<std::size_t>(g)];ck(cudaMemcpy(flat_output.data()+base,d.state,nstate*4ULL,cudaMemcpyDeviceToHost),"rtbuild result state");max_scratch=std::max(max_scratch,std::max(fwd.layout.scratch_states[static_cast<std::size_t>(g)],rev.layout.scratch_states[static_cast<std::size_t>(g)]));}if(flat_output!=flat_expected)fail("GPU-built segment-major two-row mismatch");
    std::cout<<"gridfp-reduced-two-row-segment-major-built-multigpu W="<<W<<" K="<<K<<" ngpu="<<ngpu<<" batches="<<batches<<" batch_salt="<<salt<<" states="<<tables.size()<<" exact_network_u32_per_redistribution="<<fwd.network_states<<" max_scratch_MiB_per_gpu="<<double(max_scratch)*4.0/double(1ULL<<20)<<" metadata_built_before_state=1 host_state_enumeration_for_metadata=0 paired_cursor_fill=1 temporary_builder_metadata_freed_before_state=1 wall_ms="<<wall_ms<<" exact=OK\n";
    for(int g=0;g<ngpu;++g){ck(cudaSetDevice(g),"rtbuild free device");rtmajor_free_plan(fwd.device[static_cast<std::size_t>(g)]);rtmajor_free_plan(rev.device[static_cast<std::size_t>(g)]);cudaFree(scratch[static_cast<std::size_t>(g)]);auto& d=dev[static_cast<std::size_t>(g)];cudaFree(d.error);cudaFree(d.ccg);cudaFree(d.csr);cudaFree(d.cprefix);cudaFree(d.icg);cudaFree(d.isr);cudaFree(d.iprefix);cudaFree(d.owner_begin);cudaFree(d.peer);cudaFree(d.state);}
}

} // namespace

int main(int argc,char** argv){const int W=argc>1?std::atoi(argv[1]):10;const int ngpu=argc>2?std::atoi(argv[2]):2;const int batches=argc>3?std::atoi(argv[3]):4;const unsigned blocks=argc>4?static_cast<unsigned>(std::strtoul(argv[4],nullptr,10)):256u;const std::uint32_t mod=argc>5?static_cast<std::uint32_t>(std::strtoul(argv[5],nullptr,10)):4294967291u;const std::uint32_t salt=argc>6?static_cast<std::uint32_t>(std::strtoul(argv[6],nullptr,0)):0u;if(W<8||W>11||(W&1)||ngpu<2||ngpu>SCHEDULE_MAX_GPU||batches<1||batches>64||!blocks||mod<3)return 2;int visible=0;ck(cudaGetDeviceCount(&visible),"rtbuild device count");if(visible<ngpu)return 3;run_two_row_segment_major_built(W,ngpu,batches,blocks,mod,salt);std::cout<<"ALL_OK production_two_row_segment_major_gpu_built=1\n";return 0;}
