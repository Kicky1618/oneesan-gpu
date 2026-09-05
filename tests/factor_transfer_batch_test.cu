// Compare all global coefficients with an independent CPU forward recurrence.
#define main oneesan_solver_main
#include "../src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu"
#undef main
#include <random>

static void legal_words(int pos,int height,MateID word,std::vector<MateID>& words){
    if(pos<0){if(!height)words.push_back(word);return;}
    legal_words(pos-1,height,word,words);
    if(height)legal_words(pos-1,height-1,word|(MateID(R)<<(2*pos)),words);
    if(height<pos+1)legal_words(pos-1,height+1,word|(MateID(L)<<(2*pos)),words);
}

int main(){
    static_assert(TARGET_W==10&&LOW_LUT_K==5&&HIGH_LUT_K==4);
    build_full_dp();G_FACTOR=build_factor_tables();upload_rank_index_table();
    std::vector<void*> allocations;
#define UPLOAD(symbol, vec) do { \
    using T=typename std::decay_t<decltype(vec)>::value_type;T* ptr=nullptr; \
    ck(cudaMalloc(&ptr,(vec).size()*sizeof(T)),"table alloc");allocations.push_back(ptr); \
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
#undef UPLOAD
    ck(cudaMemcpyToSymbol(D_F_LOW_ALL_OFF,G_FACTOR.low_all_off.data(),sizeof(uint32_t)*(MAXW+2)),"low off");
    ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_OFF,G_FACTOR.high_all_off.data(),sizeof(uint32_t)*(MAXW+2)),"high off");
    std::vector<MateID> main_words,block_words;
    legal_words(TARGET_W-1,1,0,main_words);legal_words(TARGET_W-2,1,0,block_words);
    Code n=H_DP[TARGET_W][1],dn=H_DP[TARGET_W-1][1];
    if(main_words.size()!=n||block_words.size()!=dn)throw std::runtime_error("CPU state generation");
    Count* mp[MAXGPU]{},*bp[MAXGPU]{};
    ck(cudaMalloc(&mp[0],n*sizeof(Count)),"main alloc");ck(cudaMalloc(&bp[0],dn*sizeof(Count)),"block alloc");
    DeviceCtx c;c.init(0,4294967291u,mp,bp,n,dn,1);c.verify_transfer=true;c.transfer_budget=8ull<<20;
    std::mt19937_64 rng(0x62c31f05);
    uint64_t cases=0,coefficients=0,total_batches=0;
    std::array<uint64_t,33> lane_cases{};
    for(Count mod:{2u,4294967291u,4294966997u}){
        ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"test modulus");
        uint64_t reciprocal=oneesan::division_reciprocal(mod);
        ck(cudaMemcpyToSymbol(D_MOD_RECIPROCAL,&reciprocal,sizeof(reciprocal)),"test reciprocal");
        for(bool fixed_low:{false,true}){
            PreparedWindow pw;pw.wp.p_hi=fixed_low?TARGET_W-1:LOW_LUT_K;pw.wp.p_lo=fixed_low?LOW_LUT_K+1:1;
            pw.wp.fixed_pos=window_candidates(TARGET_W,pw.wp.p_hi,pw.wp.p_lo);
            for(int g=0;g<(1<<pw.wp.fixed_pos.size());++g)pw.groups.push_back(prepare_group(TARGET_W,pw.wp,g,n,dn,1));
            std::sort(pw.groups.begin(),pw.groups.end(),[](const auto& a,const auto& b){return a.work>b.work;});
            std::vector<Count> input(n),dinput(dn);
            for(auto& value:input)value=Count(rng()%mod);
            for(auto& value:dinput)value=Count(rng()%mod);
            auto expected=input,dexpected=dinput;
            for(int p=pw.wp.p_hi;p>=pw.wp.p_lo;--p){
                auto next=expected;std::vector<Count> dnext(dn,0);
                auto add=[&](Count& a,Count b){a=Count((uint64_t(a)+b)%mod);};
                for(auto word:main_words){
                    auto edge=oneesan::gridfp::include_horizontal(word,TARGET_W,p);
                    if(!edge.valid)continue;
                    Count value=expected[rank_full(word,TARGET_W)];
                    if(edge.blocked)add(dnext[rank_full(edge.mate,TARGET_W-1)],value);
                    else add(next[rank_full(edge.mate,TARGET_W)],value);
                }
                for(auto word:block_words)add(next[rank_full(oneesan::gridfp::blocked_exclude(word,p),TARGET_W)],dexpected[rank_full(word,TARGET_W-1)]);
                expected.swap(next);dexpected.swap(dnext);
            }
            for(size_t target:{16ull<<10,256ull<<10})for(int limit:{1,2,3,8,32})for(bool graphs:{false,true})for(bool profiling:{false,true})for(int lane_override:{0,1,2,4,8,16,32}){
                if(lane_override&&!(target==(256ull<<10)&&limit==32&&!profiling))continue;
                c.clear_transfer();c.transfer_key_valid=false;c.clear_graphs();
                c.use_graphs=graphs;c.graph_io=graphs;c.pipeline_groups=(limit!=8);c.profile_batch=profiling;
                auto batches=prepare_transfer_batches(pw,target,limit);
                if(lane_override)for(auto& batch:batches){
                    size_t count=batch.end-batch.begin,padded=(count+lane_override-1)/lane_override*lane_override;
                    const auto& group=pw.groups[batch.begin];
                    if(16*padded*(group.ms.size+group.ds.size)+6*255<=target)batch.lanes=lane_override;
                }
                // Replay on another input copy to exercise cached graphs and masks.
                for(int replay=0;replay<2;++replay){
                    ck(cudaMemcpy(mp[0],input.data(),n*sizeof(Count),cudaMemcpyHostToDevice),"initial main");
                    ck(cudaMemcpy(bp[0],dinput.data(),dn*sizeof(Count),cudaMemcpyHostToDevice),"initial block");
                    ck(cudaDeviceSynchronize(),"input upload complete");
                    for(const auto& batch:batches){process_transfer_batch(c,TARGET_W,pw,batch,256,target);if(batch.end-batch.begin>1)++lane_cases[batch.lanes];}
                    c.drain_groups();
                    std::vector<Count> got(n),dgot(dn);
                    ck(cudaMemcpy(got.data(),mp[0],n*sizeof(Count),cudaMemcpyDeviceToHost),"result main");
                    ck(cudaMemcpy(dgot.data(),bp[0],dn*sizeof(Count),cudaMemcpyDeviceToHost),"result block");
                    if(got!=expected||dgot!=dexpected){std::cerr<<"FAIL mod="<<mod<<" fixed_low="<<fixed_low<<" target="<<target<<" limit="<<limit<<" graphs="<<graphs<<" profiling="<<profiling<<'\n';return 1;}
                    ++cases;coefficients+=n+dn;
                }
            }
        }
    }
    total_batches=c.transfer_batches;if(!total_batches)throw std::runtime_error("batch path not exercised");
    for(int lanes:{1,2,4,8,16,32})if(!lane_cases[lanes])throw std::runtime_error("batch lane variant not exercised");
    c.destroy();cudaFree(mp[0]);cudaFree(bp[0]);for(auto ptr:allocations)cudaFree(ptr);
    std::cout<<"PASS "<<cases<<" full-vector CPU/GPU cases, "<<coefficients<<" coefficients, "<<total_batches<<" batches; 3 moduli, both windows, scratch limits, graphs, profiling\n";
}
