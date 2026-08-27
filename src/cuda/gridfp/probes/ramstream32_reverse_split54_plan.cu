#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main rs54_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_reverse_split54.hpp"

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();
    StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);
    StorageLayout layout=build_storage_layout(storage);
    BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);
    ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);
    ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);
    ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);
    ReverseSplit54Host sp=build_reverse_split54(layout,rb,false);

    if(sp.nblocks!=rb.nblocks){std::cerr<<"reverse split54 nblocks mismatch\n";return 2;}
    const size_t pitch=size_t(rb.nblocks)+1;const uint64_t payload_mask=(1ull<<54)-1ull;uint64_t checked=0;
    auto check_side=[&](bool low,const ReverseSplit54SideHost&s){
        const auto&ov=low?rb.low_orbit:rb.high_orbit;const auto&oo=low?rb.low_orbit_off:rb.high_orbit_off;uint32_t steps=low?LOW_LUT_K:HIGH_LUT_K;
        for(uint32_t pi=0;pi<steps;++pi){
            for(uint32_t bid=0;bid<rb.nblocks;++bid){
                size_t oi=size_t(pi)*pitch+bid;uint32_t ni=s.nn_off[oi],ri=s.nr_off[oi],li=s.nl_off[oi];uint32_t ne=s.nn_off[oi+1],re=s.nr_off[oi+1],le=s.nl_off[oi+1];
                uint32_t a=oo[oi],b=oo[oi+1];
                for(uint32_t q=a;q<b;++q){
                    uint64_t w=ov[q];uint32_t kind=rb_orbit_kind(w);const std::vector<BucketOrbitOp>*v=nullptr;uint32_t idx=0;
                    if(kind==CPU_ORBIT_NN){v=&s.nn;idx=ni++;}
                    else if(kind==CPU_ORBIT_NR){v=&s.nr;idx=ri++;}
                    else if(kind==CPU_ORBIT_NL){v=&s.nl;idx=li++;}
                    else{std::cerr<<"reverse split54 invalid kind="<<kind<<'\n';std::exit(560);}
                    if(idx>=v->size()){std::cerr<<"reverse split54 stream offset overflow\n";std::exit(561);}
                    BucketOrbitOp got=(*v)[idx],expect=BucketOrbitOp(w&payload_mask);
                    if(got!=expect){std::cerr<<"reverse split54 payload mismatch side="<<(low?"low":"high")<<" pi="<<pi<<" bid="<<bid<<" q="<<q<<" got="<<got<<" expected="<<expect<<'\n';std::exit(562);}
                    if(bkf_orbit_src(got)!=rb_orbit_src(w)||bkf_orbit_partner(got)!=rb_orbit_partner(w)||bkf_orbit_drop(got)!=rb_orbit_drop(w)){std::cerr<<"reverse split54 locator decode mismatch\n";std::exit(563);}
                    ++checked;
                }
                if(ni!=ne||ri!=re||li!=le){std::cerr<<"reverse split54 interval consumption mismatch side="<<(low?"low":"high")<<" pi="<<pi<<" bid="<<bid<<'\n';std::exit(564);}
            }
        }
        if(s.ops()!=ov.size()){std::cerr<<"reverse split54 side op count mismatch\n";std::exit(565);}
    };
    check_side(true,sp.low);check_side(false,sp.high);

    size_t ops=sp.low.ops()+sp.high.ops();size_t split18_equiv=sp.bytes()+ops;size_t saved=ops;
    auto mib=[](size_t x){return double(x)/double(1<<20);};
    std::cout<<std::setprecision(12)
        <<"reverse-split54-plan OK W="<<TARGET_W
        <<" checked="<<checked
        <<" low_ops="<<sp.low.ops()
        <<" high_ops="<<sp.high.ops()
        <<" split54_mib="<<mib(sp.bytes())
        <<" split18_equiv_mib="<<mib(split18_equiv)
        <<" split18_sidecar_saved_mib="<<mib(saved)
        <<" payload_bits=54 storage_bytes_per_orbit="<<sizeof(BucketOrbitOp)
        <<" closure_attach_bytes=0 kind_bits=0 jblock_bits=0\n";
    return 0;
}
