// Compare precompiled operators with independent atomic FORWARD transitions.
#define main oneesan_solver_main
#include "../src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_reverse2_row6crt20_batch.cu"
#undef main
__global__ void map_test_fill(Count* a,Code n,unsigned salt){
    for(Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=Code(gridDim.x)*blockDim.x)
        a[i]=Count((i*6364136223846793005ULL+salt*1442695040888963407ULL)%D_MOD);
}
int main(){
    static_assert(TARGET_W==10&&LOW_LUT_K==5&&HIGH_LUT_K==4);
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

#undef UPLOAD
    ck(cudaMemcpyToSymbol(D_F_LOW_ALL_OFF,G_FACTOR.low_all_off.data(),sizeof(uint32_t)*(MAXW+2)),"low off");
    ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_OFF,G_FACTOR.high_all_off.data(),sizeof(uint32_t)*(MAXW+2)),"high off");

    uint64_t cases=0,states=0;
    for(Count mod:{2u,4294967291u,4294966997u}){
        DeviceCtx c;Count* nulls[MAXGPU]{};c.init(0,mod,nulls,nulls,1,1,1);c.transfer_budget=8ull<<20;c.verify_transfer=true;
        uint64_t reciprocal=oneesan::division_reciprocal(mod);ck(cudaMemcpyToSymbol(D_MOD_RECIPROCAL,&reciprocal,sizeof(reciprocal)),"test reciprocal");
        for(int fixed:{0,1})for(int depth=1;depth<=2;++depth){
            int width=fixed?LOW_LUT_K:HIGH_LUT_K;
            WindowPlan wp;wp.p_hi=fixed?TARGET_W-1:depth;wp.p_lo=wp.p_hi-depth+1;
            for(unsigned mask=1;mask<(1u<<width)-1;++mask){
                PreparedGroup pg;auto mb=make_factor_main_blocks(fixed,mask),db=make_factor_block_blocks(fixed,mask);
                pg.ms.size=mb.back().end;pg.ds.size=db.back().end;
                std::copy(mb.begin(),mb.end(),pg.factor.main_blocks);std::copy(db.begin(),db.end(),pg.factor.block_blocks);
                pg.factor.main_n=mb.size();pg.factor.block_n=db.size();pg.factor.mask=mask;pg.factor.fix_low=fixed;
                Code n=pg.ms.size,dn=pg.ds.size;
                upload_factor_config(pg.factor,c.sMain);c.ensure(n,dn,false,0,0);ensure_transfer(c,pg,wp,0);verify_transfer(c,n,dn);
                if(c.transfer_steps.size()!=1)throw std::runtime_error("test map not built");
                Count *mapped=nullptr;ck(cudaMalloc(&mapped,(n+dn)*sizeof(Count)),"test mapped output");
                map_test_fill<<<16,128,0,c.sMain>>>(c.dA,n,mask);map_test_fill<<<16,128,0,c.sMain>>>(c.dD,dn,mask+7);
                auto const& step=c.transfer_steps[0];
                factor_transfer::apply_kernel<<<16,128,0,c.sMain>>>(c.dA,c.dD,mapped,mapped+n,n,n+dn,step.offsets,step.columns);
                Count *a=c.dA,*b=c.dB,*d=c.dD,*e=c.dE;
                for(int p=wp.p_hi;p>=wp.p_lo;--p){
                    ck(cudaMemcpyAsync(b,a,n*sizeof(Count),cudaMemcpyDeviceToDevice,c.sMain),"oracle identity");
                    ck(cudaMemsetAsync(e,0,dn*sizeof(Count),c.sMain),"oracle clear");
                    main_group_kernel<<<16,128,0,c.sMain>>>(a,nullptr,n,b,e,p);
                    blocked_group_kernel<<<16,128,0,c.sMain>>>(d,dn,b,p);
                    std::swap(a,b);std::swap(d,e);
                }
                ck(cudaStreamSynchronize(c.sMain),"test result sync");
                std::vector<Count>want(n+dn),got(n+dn);
                ck(cudaMemcpy(want.data(),a,n*sizeof(Count),cudaMemcpyDeviceToHost),"oracle main");
                ck(cudaMemcpy(want.data()+n,d,dn*sizeof(Count),cudaMemcpyDeviceToHost),"oracle block");
                ck(cudaMemcpy(got.data(),mapped,(n+dn)*sizeof(Count),cudaMemcpyDeviceToHost),"map result");
                if(got!=want){std::cerr<<"Mismatch fixed="<<fixed<<" mask="<<mask<<" depth="<<depth<<" modulus="<<mod<<'\n';return 1;}
                cudaFree(mapped);++cases;states+=n+dn;
            }
        }
        c.destroy();
    }
    for(auto ptr:allocations)cudaFree(ptr);
    std::cout<<"PASS "<<cases<<" random-vector operator cases, "<<states<<" states; depths 1..2, both windows, 3 moduli\n";
}
