#pragma push_macro("main")
#undef main
#define main bucket_fused_reference_main
#include "ramstream32_gpu_direct_gather_selftest.cu"
#pragma pop_macro("main")

#include "../ramstream32_gpu_direct_gather_cross.cuh"
#include "../ramstream32_gpu_direct_fused.cuh"
#include "../ramstream32_gpu_direct_fused_validate.hpp"
#include "../ramstream32_bucket_layout.hpp"
#include "../ramstream32_bucket_direct.hpp"
#include "../ramstream32_bucket_fused.cuh"
#include "../ramstream32_bucket_fused_v2.cuh"

using BucketHostGrid=std::array<std::array<std::vector<Count>,BUCKET_NGPU>,BUCKET_NGPU>;

static BucketHostGrid bkft_make_grid(
    const std::vector<MateID>&ms,const std::vector<MateID>&bs,
    const std::vector<Count>&mv,const std::vector<Count>&bv,
    const StorageFactorHost&storage,const StorageLayout&layout,
    const BucketOwnerHost&owner,const BucketPhysicalLayoutHost&phy
){
    BucketHostGrid out;
    for(int a=0;a<BUCKET_NGPU;++a)for(int b=0;b<BUCKET_NGPU;++b)
        out[a][b].assign(size_t(phy.pair[a][b].size),0);
    for(size_t i=0;i<ms.size();++i){auto x=bucket_rank_main_host(ms[i],storage,layout,owner,phy);out[x.owner_h][x.owner_l][size_t(x.off)]=mv[i];}
    for(size_t i=0;i<bs.size();++i){auto x=bucket_rank_block_host(bs[i],storage,layout,owner,phy);out[x.owner_h][x.owner_l][size_t(x.off)]=bv[i];}
    return out;
}

static bool bkft_compare(
    const char*tag,const BucketHostGrid&g,
    const std::vector<MateID>&ms,const std::vector<MateID>&bs,
    const std::vector<Count>&mv,const std::vector<Count>&bv,
    const StorageFactorHost&storage,const StorageLayout&layout,
    const BucketOwnerHost&owner,const BucketPhysicalLayoutHost&phy
){
    for(size_t i=0;i<ms.size();++i){auto x=bucket_rank_main_host(ms[i],storage,layout,owner,phy);Count got=g[x.owner_h][x.owner_l][size_t(x.off)];if(got!=mv[i]){std::cerr<<"FAIL "<<tag<<" main i="<<i<<" got="<<got<<" expected="<<mv[i]<<'\n';return false;}}
    for(size_t i=0;i<bs.size();++i){auto x=bucket_rank_block_host(bs[i],storage,layout,owner,phy);Count got=g[x.owner_h][x.owner_l][size_t(x.off)];if(got!=bv[i]){std::cerr<<"FAIL "<<tag<<" block i="<<i<<" got="<<got<<" expected="<<bv[i]<<'\n';return false;}}
    return true;
}

static void bkft_alloc_slots(
    uint32_t fixed,const BucketPhysicalLayoutHost&phy,std::array<Count*,BUCKET_NGPU>&d
){
    for(uint32_t s=0;s<BUCKET_NGPU;++s){size_t n=size_t(std::max<Code>(Code(1),phy.slot_capacity[fixed][s]));ck(cudaMalloc(&d[s],n*sizeof(Count)),"bkft slot alloc");ck(cudaMemset(d[s],0,n*sizeof(Count)),"bkft slot zero");}
}
static void bkft_free_slots(std::array<Count*,BUCKET_NGPU>&d){for(auto&p:d){if(p)cudaFree(p);p=nullptr;}}

static void bkft_run_low(
    BucketHostGrid&g,const StorageLayout&layout,const BucketPhysicalLayoutHost&phy,
    BucketFusedDeviceTables&dt
){
    for(uint32_t fixed=0;fixed<BUCKET_NGPU;++fixed){
        std::array<Count*,BUCKET_NGPU>d{};bkft_alloc_slots(fixed,phy,d);
        for(uint32_t s=0;s<BUCKET_NGPU;++s){auto const&src=g[fixed][s];if(!src.empty())ck(cudaMemcpy(d[s],src.data(),src.size()*sizeof(Count),cudaMemcpyHostToDevice),"bkft low H2D");}
        dt.bind_owner(fixed,phy,d);bucket_run_low_fused_v2(layout,256,4,4);
        for(uint32_t s=0;s<BUCKET_NGPU;++s){auto&dst=g[fixed][s];if(!dst.empty())ck(cudaMemcpy(dst.data(),d[s],dst.size()*sizeof(Count),cudaMemcpyDeviceToHost),"bkft low D2H");}
        bkft_free_slots(d);
    }
}

static void bkft_run_high(
    BucketHostGrid&g,const StorageLayout&layout,const BucketPhysicalLayoutHost&phy,
    BucketFusedDeviceTables&dt
){
    for(uint32_t fixed=0;fixed<BUCKET_NGPU;++fixed){
        std::array<Count*,BUCKET_NGPU>d{};bkft_alloc_slots(fixed,phy,d);
        for(uint32_t s=0;s<BUCKET_NGPU;++s){auto const&src=g[s][fixed];if(!src.empty())ck(cudaMemcpy(d[s],src.data(),src.size()*sizeof(Count),cudaMemcpyHostToDevice),"bkft high H2D");}
        dt.bind_owner(fixed,phy,d);bucket_run_high_fused(layout,256,4,4);
        for(uint32_t s=0;s<BUCKET_NGPU;++s){auto&dst=g[s][fixed];if(!dst.empty())ck(cudaMemcpy(dst.data(),d[s],dst.size()*sizeof(Count),cudaMemcpyDeviceToHost),"bkft high D2H");}
        bkft_free_slots(d);
    }
}

int main(){
    constexpr Count mod=4294967291u;constexpr int W=TARGET_W;
    static_assert(W==LOW_LUT_K+HIGH_LUT_K+1);static_assert(W<=12,"bucket fused selftest intentionally uses small width");
    int visible=0;cudaError_t ce=cudaGetDeviceCount(&visible);if(ce!=cudaSuccess||visible<1){std::cout<<"gpu-bucket-fused-selftest SKIP no CUDA device\n";return 0;}ck(cudaSetDevice(0),"bkft set device");

    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);
    GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);
    CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketPhysicalLayoutHost phy=build_bucket_physical_layout(layout,owner);BucketOrbitStreamsHost borbit=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bfused=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);

    auto ms=gdg_enum_states(W),bs=gdg_enum_states(W-1);if(ms.size()!=layout.main_size||bs.size()!=layout.block_size)return 2;
    std::unordered_map<MateID,size_t>mi,di;for(size_t i=0;i<ms.size();++i)mi.emplace(ms[i],i);for(size_t i=0;i<bs.size();++i)di.emplace(bs[i],i);
    std::vector<Count>init_m(ms.size()),init_b(bs.size());std::mt19937_64 rng(1618);for(auto&x:init_m)x=Count(rng()%mod);for(auto&x:init_b)x=Count(rng()%mod);
    auto[low_m,low_b]=gdg_reference_window(W,LOW_LUT_K,1,mod,ms,bs,mi,di,init_m,init_b);auto[high_m,high_b]=gdg_reference_window(W,W-1,LOW_LUT_K+1,mod,ms,bs,mi,di,init_m,init_b);auto[row_m,row_b]=gdg_reference_window(W,LOW_LUT_K,1,mod,ms,bs,mi,di,high_m,high_b);

    ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"bkft modulus");BucketFusedDeviceTables dt;dt.install_metadata(layout,borbit,bfused);

    auto lowg=bkft_make_grid(ms,bs,init_m,init_b,storage,layout,owner,phy);bkft_run_low(lowg,layout,phy,dt);if(!bkft_compare("bucket-low",lowg,ms,bs,low_m,low_b,storage,layout,owner,phy))return 10;
    auto highg=bkft_make_grid(ms,bs,init_m,init_b,storage,layout,owner,phy);bkft_run_high(highg,layout,phy,dt);if(!bkft_compare("bucket-high",highg,ms,bs,high_m,high_b,storage,layout,owner,phy))return 11;
    auto rowg=bkft_make_grid(ms,bs,init_m,init_b,storage,layout,owner,phy);bkft_run_high(rowg,layout,phy,dt);bkft_run_low(rowg,layout,phy,dt);if(!bkft_compare("bucket-row",rowg,ms,bs,row_m,row_b,storage,layout,owner,phy))return 12;

    std::cout<<"gpu-bucket-fused-selftest OK W="<<W
             <<" main="<<ms.size()<<" block="<<bs.size()
             <<" owner_high_max="<<owner.max_high_count
             <<" owner_low_max="<<owner.max_low_count
             <<" orbit_mib="<<double(borbit.bytes())/double(1<<20)
             <<" fused_mib="<<double(bfused.bytes())/double(1<<20)
             <<" locator_bits="<<BUCKET_LOCATOR_BITS
             <<" closure_atomic=0 scratch_bytes=0\n";
    dt.release();return 0;
}
