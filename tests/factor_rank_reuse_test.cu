// Compare the actual production kernel's two rank policies in one process.
#define main oneesan_solver_main
#include "../src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu"
#undef main

__global__ void fill_rank_test(Count* a, Code n, MateID* mates, bool main) {
    for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=Code(gridDim.x)*blockDim.x){
        a[i]=Count((i*6364136223846793005ULL+1442695040888963407ULL)%D_MOD);
        if(main)mates[i]=factor_unrank_main(i);
    }
}

__global__ void check_factor_roundtrip(Code n,Code dn,unsigned long long* errors){
    for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=Code(gridDim.x)*blockDim.x){
        MateID m=factor_unrank_main(i),got=0;Code rank=factor_global_rank_main(m);
        if(factor_rank_main(m)!=i||factor_global_index<false>(i,&got)!=rank||got!=m||factor_global_index<false>(i)!=rank)atomicAdd(errors,1ULL);
    }
    for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x;i<dn;i+=Code(gridDim.x)*blockDim.x){
        MateID m=factor_unrank_block(i),got=0;Code rank=factor_global_rank_block(m);
        if(factor_rank_block(m)!=i||factor_global_index<true>(i,&got)!=rank||got!=m||factor_global_index<true>(i)!=rank)atomicAdd(errors,1ULL);
    }
}

// Compare the delta formula with full ranking for every legal local predecessor.
__global__ void check_predecessor_ranks(Code n,unsigned long long* result){
    unsigned long long checked=0,errors=0;
    for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=Code(gridDim.x)*blockDim.x){
        MateID t=factor_unrank_main(i);
        for(int p=2;p<=LOW_LUT_K;++p){
            MateID pred=0;bool have=true;
            switch(oneesan::gridfp::mpair(t,p)){
            case oneesan::gridfp::LR:pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::NN);break;
            case oneesan::gridfp::NR:pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::RN);break;
            case oneesan::gridfp::NL:pred=oneesan::gridfp::msetpair(t,p,oneesan::gridfp::LN);break;
            default:have=false;
            }
            if(have){++checked;errors+=factor_predecessor_rank(t,pred,i,p)!=factor_rank_main(pred);}
        }
    }
    atomicAdd(result,checked);atomicAdd(result+1,errors);
}

int main(int argc,char**argv) {
    bool bench=argc==2&&std::strcmp(argv[1],"--bench")==0;
    if(argc>1&&!bench){std::cerr<<"usage: rank-reuse [--bench]\n";return 1;}
    static_assert(TARGET_W == 19 && LOW_LUT_K == 9 && HIGH_LUT_K == 9);
    build_full_dp(); G_FACTOR=build_factor_tables();upload_rank_index_table();
    std::vector<void*> allocations;
#define UPLOAD(symbol, vec) do { \
    using T=typename std::decay_t<decltype(vec)>::value_type; T* ptr=nullptr; \
    ck(cudaMalloc(&ptr,(vec).size()*sizeof(T)),"table alloc"); allocations.push_back(ptr); \
    ck(cudaMemcpy(ptr,(vec).data(),(vec).size()*sizeof(T),cudaMemcpyHostToDevice),"table copy"); \
    ck(cudaMemcpyToSymbol(symbol,&ptr,sizeof(ptr)),"table symbol"); \
} while(0)
    UPLOAD(D_F_LOW_ALL_CODES,G_FACTOR.low_all_codes);
    UPLOAD(D_F_LOW_MASK_CODES,G_FACTOR.low_mask_codes);
    UPLOAD(D_F_LOW_MASK_OFF,G_FACTOR.low_mask_off);
    UPLOAD(D_F_LOW_DENSE_PACKED_RANK,G_FACTOR.low_dense_packed_rank);
    UPLOAD(D_F_HIGH_ALL_CODES,G_FACTOR.high_all_codes);
    UPLOAD(D_F_HIGH_MASK_CODES,G_FACTOR.high_mask_codes);
    UPLOAD(D_F_HIGH_MASK_OFF,G_FACTOR.high_mask_off);
    UPLOAD(D_F_HIGH_PACKED_RANK,G_FACTOR.high_packed_rank);
    UPLOAD(D_F_HIGH_MAIN_BASE,G_FACTOR.high_main_base);
    UPLOAD(D_F_HIGH_BLOCK_BASE,G_FACTOR.high_block_base);
    ck(cudaMemcpyToSymbol(D_FULL_DP,H_DP,sizeof(H_DP)),"full rank dp");

#undef UPLOAD
    ck(cudaMemcpyToSymbol(D_F_LOW_ALL_OFF,G_FACTOR.low_all_off.data(),sizeof(uint32_t)*(MAXW+2)),"low off");
    ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_OFF,G_FACTOR.high_all_off.data(),sizeof(uint32_t)*(MAXW+2)),"high off");
    Count mod=4294967291u; ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"mod");
    cudaEvent_t start,stop; ck(cudaEventCreate(&start),"start"); ck(cudaEventCreate(&stop),"stop");
    for(int fixLow : {0,1})for(uint32_t mask : {0u,85u,511u}) {
        auto mb=make_factor_main_blocks(fixLow,mask), db=make_factor_block_blocks(fixLow,mask);
        int nm=mb.size(),nd=db.size();Code n=mb.back().end,dn=db.back().end;
        FactorRuntimeCfg cfg;
        std::copy(mb.begin(),mb.end(),cfg.main_blocks);std::copy(db.begin(),db.end(),cfg.block_blocks);
        cfg.main_n=nm;cfg.block_n=nd;cfg.mask=mask;cfg.fix_low=fixLow;
        upload_factor_config(cfg);
        {
            unsigned long long* errors=nullptr;unsigned long long host=0;
            ck(cudaMalloc(&errors,sizeof(host)),"roundtrip alloc");
            ck(cudaMemset(errors,0,sizeof(host)),"roundtrip clear");
            check_factor_roundtrip<<<256,256>>>(n,dn,errors);
            ck(cudaMemcpy(&host,errors,sizeof(host),cudaMemcpyDeviceToHost),"roundtrip result");
            cudaFree(errors);if(host){std::cerr<<"factor rank/unrank mismatch\n";return 4;}
            std::cout<<"PASS factor roundtrip fix_low="<<fixLow<<" mask="<<mask<<" states="<<n+dn<<"\n";
        }
        if(!fixLow){
            unsigned long long* device_result=nullptr;unsigned long long result[2]{};
            ck(cudaMalloc(&device_result,sizeof(result)),"delta result alloc");
            ck(cudaMemset(device_result,0,sizeof(result)),"delta result clear");
            check_predecessor_ranks<<<256,256>>>(n,device_result);
            ck(cudaMemcpy(result,device_result,sizeof(result),cudaMemcpyDeviceToHost),"delta result copy");
            cudaFree(device_result);
            if(result[1]||!result[0]){std::cerr<<"predecessor rank delta mismatch\n";return 3;}
            std::cout<<"PASS low rank delta mask="<<mask<<" predecessors="<<result[0]<<"\n";
        }
        Count *a,*d,*out[2];MateID* mates;
        ck(cudaMalloc(&a,n*4),"main");ck(cudaMalloc(&d,dn*4),"block");
        ck(cudaMalloc(&mates,n*sizeof(MateID)),"mates");
        for(auto& x:out)ck(cudaMalloc(&x,n*4),"output");
        fill_rank_test<<<256,256>>>(a,n,mates,true);fill_rank_test<<<256,256>>>(d,dn,nullptr,false);
        ck(cudaDeviceSynchronize(),"init");
        int p=fixLow?TARGET_W-1:LOW_LUT_K;
        for(bool cached : {false,true}) {
            std::vector<float> times[2];
            for(int repeat=0;repeat<(bench?11:1);++repeat)for(int order=0;order<2;++order){
                int mode=order^(repeat&1);ck(cudaEventRecord(start),"record");
                for(int j=0;j<(bench?50:1);++j){
                    if(mode)reverse2_main_group_kernel<true><<<256,256>>>(a,d,cached?mates:nullptr,n,out[mode],p);
                    else reverse2_main_group_kernel<false><<<256,256>>>(a,d,cached?mates:nullptr,n,out[mode],p);
                }
                ck(cudaGetLastError(),"launch");ck(cudaEventRecord(stop),"stop record");
                ck(cudaEventSynchronize(stop),"stop sync");float ms;ck(cudaEventElapsedTime(&ms,start,stop),"elapsed");
                if(repeat||!bench)times[mode].push_back(ms/(bench?50:1));
            }
            std::vector<Count>x(n),y(n);
            ck(cudaMemcpy(x.data(),out[0],n*4,cudaMemcpyDeviceToHost),"result 0");
            ck(cudaMemcpy(y.data(),out[1],n*4,cudaMemcpyDeviceToHost),"result 1");
            if(x!=y){std::cerr<<"rank reuse mismatch\n";return 2;}
            for(auto& t:times)std::sort(t.begin(),t.end());
            std::cout<<"fix_low="<<fixLow<<" mask="<<mask<<" cached="<<cached<<" states="<<n
                     <<" PASS";
            if(bench){float old=(times[0][4]+times[0][5])/2,neu=(times[1][4]+times[1][5])/2;
                std::cout<<" baseline_ms="<<old<<" reuse_ms="<<neu<<" speedup="<<old/neu;}
            std::cout<<"\n";
        }
        cudaFree(a);cudaFree(d);cudaFree(mates);for(auto x:out)cudaFree(x);
    }
    cudaEventDestroy(start);cudaEventDestroy(stop);for(auto p:allocations)cudaFree(p);
}
