#pragma once
// Same-geometry groups form a direct sum of identical integer operators.
// Each column retains its own values and global addresses; only T is shared.
namespace factor_transfer {
constexpr int MAX_BATCH=32;
#ifndef GRIDFP_BATCH_LANES
#define GRIDFP_BATCH_LANES 16
#endif
constexpr int BATCH_LANES=GRIDFP_BATCH_LANES;
static_assert(BATCH_LANES==1||BATCH_LANES==2||BATCH_LANES==4||BATCH_LANES==8||BATCH_LANES==16||BATCH_LANES==32);
__constant__ uint32_t batch_masks[MAX_BATCH];

template<bool Block,int BATCH_LANES>
__global__ void batch_gather_kernel(Count* out,Code* ranks,Code n,int count){
    const unsigned lane=threadIdx.x&(BATCH_LANES-1);
    const unsigned group=blockIdx.y*BATCH_LANES+lane;
    if(group>=count)return;
    const uint32_t mask=batch_masks[group];
    const Code base=Code(blockIdx.y)*n*BATCH_LANES+lane;
    for(Code i=(Code(blockIdx.x)*blockDim.x+threadIdx.x)/BATCH_LANES;i<n;i+=Code(gridDim.x)*blockDim.x/BATCH_LANES){
        Code g=factor_global_index<Block>(i,nullptr,mask);
        out[base+i*BATCH_LANES]=Block?global_load_block(g):global_load_main(g);
        ranks[base+i*BATCH_LANES]=g;
    }
}

template<bool Scatter,int BATCH_LANES>
__global__ void batch_apply_kernel(const Count* __restrict__ in,const Count* __restrict__ din,
        Count* __restrict__ out,Count* __restrict__ dout,Code n,Code dn,
        const uint32_t* __restrict__ offsets,const uint32_t* __restrict__ columns,
        const Code* __restrict__ ranks,const Code* __restrict__ dranks,int count){
    const unsigned lane=threadIdx.x&(BATCH_LANES-1);
    if(blockIdx.y*BATCH_LANES+lane>=count)return;
    const Code mb=Code(blockIdx.y)*n*BATCH_LANES+lane,db=Code(blockIdx.y)*dn*BATCH_LANES+lane;
    for(Code i=(Code(blockIdx.x)*blockDim.x+threadIdx.x)/BATCH_LANES;i<n+dn;i+=Code(gridDim.x)*blockDim.x/BATCH_LANES){
        Code sum=i<n?in[mb+i*BATCH_LANES]:0;
        const uint32_t end=offsets[i+1];
        for(uint32_t e=offsets[i];e<end;++e){uint32_t j=columns[e];sum+=j<n?in[mb+Code(j)*BATCH_LANES]:din[db+Code(j-n)*BATCH_LANES];}
        Count value=oneesan::invariant_divmod(sum,D_MOD,D_MOD_RECIPROCAL).remainder;
        if constexpr(Scatter){
            if(i<n)global_store_main(ranks[mb+i*BATCH_LANES],value);
            else global_store_block(dranks[db+(i-n)*BATCH_LANES],value);
        }else{
            if(i<n)out[mb+i*BATCH_LANES]=value;else dout[db+(i-n)*BATCH_LANES]=value;
        }
    }
}
}

struct PreparedTransferBatch {
    size_t begin=0,end=0;
    int lanes=1;
    std::array<uint32_t,factor_transfer::MAX_BATCH> masks{};
};

static bool same_transfer_shape(const PreparedGroup& a,const PreparedGroup& b){
    return a.factor.fix_low==b.factor.fix_low &&
        __builtin_popcount(a.factor.mask)==__builtin_popcount(b.factor.mask) &&
        a.ms.size==b.ms.size && a.ds.size==b.ds.size &&
        !a.use_mi&&!a.use_di&&!b.use_mi&&!b.use_di;
}

static std::vector<PreparedTransferBatch> prepare_transfer_batches(const PreparedWindow& pw,size_t target,int limit){
    std::vector<PreparedTransferBatch> batches;
    for(size_t i=0;i<pw.groups.size();){
        const auto& first=pw.groups[i];
        // Four Count arrays and two Code arrays; include each arena alignment.
        const size_t per_group=size_t(first.ms.size+first.ds.size)*16;
        size_t count=target>6*255&&per_group?(target-6*255)/per_group:1;
        count=std::max<size_t>(1,std::min<size_t>(limit,count));
        PreparedTransferBatch batch;
        while(batch.lanes*2<=factor_transfer::BATCH_LANES&&size_t(batch.lanes*2)<=count)batch.lanes*=2;
        count=count/batch.lanes*batch.lanes;
        batch.begin=i;batch.masks[0]=first.factor.mask;
        size_t j=i+1;
        while(j<pw.groups.size()&&j-i<count&&same_transfer_shape(first,pw.groups[j])){
            batch.masks[j-i]=pw.groups[j].factor.mask;++j;
        }
        batch.end=j;batches.push_back(batch);i=j;
    }
    return batches;
}

static void process_transfer_batch(DeviceCtx& c,int W,const PreparedWindow& pw,
        const PreparedTransferBatch& batch,int threads,size_t target){
    const int count=int(batch.end-batch.begin);
    const auto& first=pw.groups[batch.begin];
    auto fallback=[&](){for(size_t q=batch.begin;q<batch.end;++q)process_group(c,W,pw.wp,pw.groups[q],threads,target);};
    if(count<=1||!c.transfer_budget){fallback();return;}
    auto start=std::chrono::steady_clock::now();
    ck(cudaSetDevice(c.dev),"batch device");
    upload_factor_config(first.factor,c.sMain);
    Code n=first.ms.size,dn=first.ds.size;
    const int tiles=(count+batch.lanes-1)/batch.lanes;
    const int padded=tiles*batch.lanes;
    c.ensure(n*padded,dn*padded,true,0,0,true);
    ensure_transfer(c,first,pw.wp,target);
    if(c.transfer_steps.empty()){fallback();return;}
    if(c.verify_transfer){
        for(size_t q=batch.begin;q<batch.end;++q){
            upload_factor_config(pw.groups[q].factor,c.sMain);verify_transfer(c,n,dn);
        }
        upload_factor_config(first.factor,c.sMain);
    }
    ck(cudaMemcpyToSymbolAsync(factor_transfer::batch_masks,batch.masks.data(),count*sizeof(uint32_t),0,cudaMemcpyHostToDevice,c.sMain),"batch masks");
    threads=c.frontier_threads;
    auto blocks=[&](Code size){int b=int(std::min<Code>(65535,(size+threads-1)/threads));return std::max(1,c.frontier_blocks?std::min(b,c.frontier_blocks):b);};
    // A distinct key is necessary even if a one-group arena has equal byte size.
    DeviceCtx::GraphKey key{n,dn,pw.wp.p_hi,pw.wp.p_lo,threads,true,true,true,true,true,count*64+batch.lanes};
    cudaEvent_t timing[4]{};
    if(c.profile_batch)for(auto& event:timing)ck(cudaEventCreate(&event),"batch timing event");
    auto launch_lanes=[&](auto lane_tag){
        constexpr int lanes=decltype(lane_tag)::value;
        if(c.profile_batch)ck(cudaEventRecord(timing[0],c.sMain),"batch timing start");
        if(n)factor_transfer::batch_gather_kernel<false,lanes><<<dim3(blocks(n*lanes),tiles),threads,0,c.sMain>>>(c.dA,c.dMate,n,count);
        if(dn)factor_transfer::batch_gather_kernel<true,lanes><<<dim3(blocks(dn*lanes),tiles),threads,0,c.sMain>>>(c.dD,c.dBlockMate,dn,count);
        if(c.profile_batch)ck(cudaEventRecord(timing[1],c.sMain),"batch timing gathered");
        Count *a=c.dA,*b=c.dB,*d=c.dD,*e=c.dE;
        for(size_t s=0;s<c.transfer_steps.size();++s){
            const auto& step=c.transfer_steps[s];
            if(c.profile_batch&&s+1==c.transfer_steps.size())ck(cudaEventRecord(timing[2],c.sMain),"batch timing final");
            if(s+1==c.transfer_steps.size())
                factor_transfer::batch_apply_kernel<true,lanes><<<dim3(blocks((n+dn)*lanes),tiles),threads,0,c.sMain>>>(a,d,b,e,n,dn,step.offsets,step.columns,c.dMate,c.dBlockMate,count);
            else
                factor_transfer::batch_apply_kernel<false,lanes><<<dim3(blocks((n+dn)*lanes),tiles),threads,0,c.sMain>>>(a,d,b,e,n,dn,step.offsets,step.columns,c.dMate,c.dBlockMate,count);
            std::swap(a,b);std::swap(d,e);
        }
        if(c.profile_batch)ck(cudaEventRecord(timing[3],c.sMain),"batch timing finish");
    };
    auto launch=[&](){
        switch(batch.lanes){
        case 1:launch_lanes(std::integral_constant<int,1>{});break;
        case 2:launch_lanes(std::integral_constant<int,2>{});break;
        case 4:launch_lanes(std::integral_constant<int,4>{});break;
        case 8:launch_lanes(std::integral_constant<int,8>{});break;
        case 16:launch_lanes(std::integral_constant<int,16>{});break;
        case 32:launch_lanes(std::integral_constant<int,32>{});break;
        default:throw std::runtime_error("invalid batch lane count");
        }
    };
    if(c.use_graphs&&!c.profile_batch){
        auto it=c.transition_graphs.find(key);
        if(it==c.transition_graphs.end()){
            ck(cudaStreamBeginCapture(c.sMain,cudaStreamCaptureModeThreadLocal),"batch capture begin");launch();
            cudaGraph_t graph=nullptr;cudaGraphExec_t exec=nullptr;
            ck(cudaStreamEndCapture(c.sMain,&graph),"batch capture end");
            ck(cudaGraphInstantiate(&exec,graph,0),"batch instantiate");
            ck(cudaGraphDestroy(graph),"batch graph definition");
            it=c.transition_graphs.emplace(key,exec).first;++c.graph_builds;
        }else ++c.graph_replays;
        ck(cudaGraphLaunch(it->second,c.sMain),"batch graph launch");
    }else launch();
    ck(cudaGetLastError(),"batch kernels");
    if(c.profile_batch){
        ck(cudaEventSynchronize(timing[3]),"batch timing sync");
        double* times[]={&c.gather_s,&c.transition_s,&c.scatter_s};
        for(int i=0;i<3;++i){float ms=0;ck(cudaEventElapsedTime(&ms,timing[i],timing[i+1]),"batch timing read");*times[i]+=ms*0.001;}
        for(auto event:timing)ck(cudaEventDestroy(event),"batch timing destroy");
    }
    if(!c.pipeline_groups)ck(cudaStreamSynchronize(c.sMain),"batch sync");
    ++c.transfer_batches;c.transfer_batched_groups+=count;c.groups+=count;
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    c.active+=elapsed;if(c.use_graphs&&c.graph_io)c.group_graph_s+=elapsed;
}
