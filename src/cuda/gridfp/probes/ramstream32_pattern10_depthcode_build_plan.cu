#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main pattern10_depthcode_plan_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_onepass_pattern10_depthcode.hpp"

#include <iomanip>
#include <iostream>

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);
    auto fh=build_bucket_forward_pattern10_depthcode_placeholder(layout,bo,bf);auto rh=build_bucket_reverse_pattern10_depthcode_zero_checked(layout,bo,bf,rb,rf);
    size_t fops=bo.low_nn.size()+bo.low_nr.size()+bo.low_nl.size()+bo.high_nn.size()+bo.high_nrnl.size();size_t rops=rh.split.low.ops()+rh.split.high.ops();size_t ops=fops+rops;size_t depth4=(ops+1)/2;

    uint64_t payload_checked=0,valid_payloads=0;
    p10dc_for_each_entry_direct(layout,bo,rh.split,bf,[&](P10DepthCodeEntryView e){
        uint32_t key=p10dc_key(e.rev,e.high,e.sid,e.p,e.h,rh.codebook.mode);if(key>=rh.codebook.base.size()){std::cerr<<"depthcode payload key overflow\n";std::exit(610);}uint32_t base=rh.codebook.base[key];if(base==P10DC_INVALID_BASE){std::cerr<<"depthcode payload missing context\n";std::exit(611);}uint32_t code=uint32_t(bkcp10_id(*e.op));if(base+code>=rh.codebook.decode.size()){std::cerr<<"depthcode payload code overflow\n";std::exit(612);}uint32_t got=rh.codebook.decode[base+code],want=p10dc_payload_host(e.pair,e.high,e.p);if(got!=want){std::cerr<<"depthcode payload mismatch rev="<<e.rev<<" high="<<e.high<<" p="<<e.p<<" h="<<e.h<<" sid="<<e.sid<<" code="<<code<<" got=0x"<<std::hex<<got<<" want=0x"<<want<<std::dec<<'\n';std::exit(613);}++payload_checked;valid_payloads+=p10dc_payload_valid(got);
    });
    if(payload_checked!=ops){std::cerr<<"depthcode payload count mismatch got="<<payload_checked<<" expected="<<ops<<'\n';return 614;}

    uint64_t lower54_hash=1469598103934665603ull;auto mix=[&](uint64_t x){lower54_hash^=x&BKCP10_BASE_MASK;lower54_hash*=1099511628211ull;};for(auto x:bo.low_nn)mix(x);for(auto x:bo.low_nr)mix(x);for(auto x:bo.low_nl)mix(x);for(auto x:bo.high_nn)mix(x);for(auto x:bo.high_nrnl)mix(x);for(auto x:rh.split.low.nn)mix(x);for(auto x:rh.split.low.nr)mix(x);for(auto x:rh.split.low.nl)mix(x);for(auto x:rh.split.high.nn)mix(x);for(auto x:rh.split.high.nr)mix(x);for(auto x:rh.split.high.nl)mix(x);
    std::cout<<std::setprecision(12)<<"pattern10-depthcode-build-plan OK W="<<TARGET_W<<" mode="<<rh.codebook.mode<<" ops="<<ops<<" forward_ops="<<fops<<" reverse_ops="<<rops<<" payload_checked="<<payload_checked<<" valid_payloads="<<valid_payloads<<" payload_exact=1 decode_payload_masks=1 decode_unrank=0 temporary_depth_bytes=0 codebook_mib="<<double(rh.codebook.bytes())/double(1<<20)<<" depth4_sidecar_mib="<<double(depth4)/double(1<<20)<<" codebook_vs_depth4_ratio="<<(depth4?double(rh.codebook.bytes())/double(depth4):0.0)<<" sidecar_bytes_per_orbit=0 lower54_hash=0x"<<std::hex<<lower54_hash<<std::dec<<" forward_attach_bytes="<<fh.bytes()<<"\n";
    return 0;
}
