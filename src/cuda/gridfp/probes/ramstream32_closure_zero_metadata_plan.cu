#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main closure_zero_metadata_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main

#include "../ramstream32_bucket_orbit_closure_packed18.cuh"
#include "../ramstream32_bucket_reverse_split18.hpp"
#include "../ramstream32_bucket_reverse_split54.hpp"
#include "../ramstream32_bucket_onepass_inline8.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>

static size_t czm_codec_bytes(const BucketFusedHost& f){
    return (f.high_codes.size()+f.low_codes.size()+f.high_code_off.size()+f.low_code_off.size()
        +f.high_direct.size()+f.low_direct.size())*sizeof(uint32_t);
}
static size_t czm_offsets_bytes(const BucketFusedHost& f){
    return (f.low_off.size()+f.high_off.size())*sizeof(uint32_t);
}
static size_t czm_offsets_bytes(const ReverseBucketFusedHost& f){
    return (f.low_off.size()+f.high_off.size())*sizeof(uint32_t);
}

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);
    StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);
    HighDescHost highdesc=build_high_descriptors(storage,layout);
    LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);
    CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);
    GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);
    GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);
    GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);
    CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);
    BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);
    BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);
    BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);

    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);
    ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);
    ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);
    ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);
    ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);

    // inline8 production representation.
    auto f18=build_bucket_forward_orbit_closure_attach18(layout,bo,bf);
    auto rfull=build_bucket_reverse_orbit_closure_attach_checked(layout,bo,bf,rb,rf);
    auto r18=build_reverse_split18(layout,rb,rf,rfull);
    auto fi8=build_bucket_forward_onepass_inline8(bf);
    auto ri8=build_bucket_reverse_onepass_inline8(rf);

    // closure-zero representation: same locator triples, no attachment or closure table.
    auto r54=build_reverse_split54(layout,rb,false);

    size_t view_bytes=2ull*BUCKET_NGPU*(layout.main_blocks.size()+layout.block_blocks.size())*sizeof(BucketPhysicalBlock);
    size_t orbit_bytes=bo.bytes();
    size_t codec_bytes=czm_codec_bytes(bf);
    size_t foff=czm_offsets_bytes(bf),roff=czm_offsets_bytes(rf);
    size_t inline8_closure=foff+f18.bytes()+fi8.bytes()+roff+ri8.bytes();
    size_t inline8_total=view_bytes+orbit_bytes+codec_bytes+inline8_closure+r18.bytes();
    size_t zero_closure=0;
    size_t zero_total=view_bytes+orbit_bytes+codec_bytes+r54.bytes();
    if(zero_total>inline8_total){std::cerr<<"closure-zero metadata unexpectedly larger\n";return 2;}
    size_t saved=inline8_total-zero_total;
    auto mib=[](size_t x){return double(x)/double(1<<20);};
    std::cout<<std::setprecision(12)
        <<"closure-zero-metadata-plan OK W="<<TARGET_W
        <<" views_mib="<<mib(view_bytes)
        <<" forward_orbit_mib="<<mib(orbit_bytes)
        <<" codec_mib="<<mib(codec_bytes)
        <<" inline8_forward_offsets_mib="<<mib(foff)
        <<" inline8_forward_attach_mib="<<mib(f18.bytes())
        <<" inline8_forward_record_mib="<<mib(fi8.bytes())
        <<" inline8_reverse_offsets_mib="<<mib(roff)
        <<" inline8_reverse_record_mib="<<mib(ri8.bytes())
        <<" reverse_split18_mib="<<mib(r18.bytes())
        <<" reverse_split54_mib="<<mib(r54.bytes())
        <<" inline8_closure_specific_mib="<<mib(inline8_closure)
        <<" zero_closure_specific_mib="<<mib(zero_closure)
        <<" inline8_total_metadata_mib="<<mib(inline8_total)
        <<" zero_total_metadata_mib="<<mib(zero_total)
        <<" saved_mib="<<mib(saved)
        <<" zero_vs_inline8_ratio="<<(inline8_total?double(zero_total)/double(inline8_total):0.0)
        <<'\n';
    return 0;
}
