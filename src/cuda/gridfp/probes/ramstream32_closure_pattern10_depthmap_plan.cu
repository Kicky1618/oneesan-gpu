#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main pattern10_depthmap_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_closure_pattern10_depth8.hpp"

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>

struct DepthMapCtx {
    // bit d is present for this pattern.  Depth is 0..15.
    std::array<uint16_t,1024> seen{};
};
struct DepthMapBooks {
    std::map<uint32_t,DepthMapCtx> coarse,phase,stream;
    uint64_t inserted=0;
    static uint32_t key(bool rev,bool high,uint32_t stream_id,int p,uint32_t h,int mode){
        uint32_t k=(uint32_t(high)<<23)|(uint32_t(p)<<8)|h;
        if(mode>=1)k|=uint32_t(rev)<<24;
        if(mode>=2)k|=(stream_id&7u)<<20;
        return k;
    }
    static void add_one(std::map<uint32_t,DepthMapCtx>&book,uint32_t k,uint16_t pat,uint8_t depth){
        if(pat>1023||depth>15){std::cerr<<"depthmap invalid pattern/depth pat="<<pat<<" depth="<<unsigned(depth)<<'\n';std::exit(590);}
        book[k].seen[pat]|=uint16_t(1u<<depth);
    }
    void add(bool rev,bool high,uint32_t sid,int p,uint32_t h,BucketOrbitOp op,uint8_t depth){
        uint16_t pat=bkcp10_id(op);add_one(coarse,key(rev,high,sid,p,h,0),pat,depth);add_one(phase,key(rev,high,sid,p,h,1),pat,depth);add_one(stream,key(rev,high,sid,p,h,2),pat,depth);++inserted;
    }
};
struct DepthMapStats {
    size_t contexts=0,used_patterns=0,ambiguous_patterns=0,max_depth_choices=0;
    uint32_t worst=0;uint16_t worst_pattern=0,worst_mask=0;
};
static DepthMapStats depthmap_stats(const std::map<uint32_t,DepthMapCtx>&m){
    DepthMapStats s;
    for(const auto&[k,c]:m){
        bool any=false;
        for(uint16_t p=0;p<1024;++p){uint16_t x=c.seen[p];if(!x)continue;any=true;++s.used_patterns;size_t n=size_t(__builtin_popcount(unsigned(x)));if(n>1)++s.ambiguous_patterns;if(n>s.max_depth_choices){s.max_depth_choices=n;s.worst=k;s.worst_pattern=p;s.worst_mask=x;}}
        s.contexts+=any;
    }
    return s;
}
static void print_worst(const char*tag,const DepthMapStats&s){
    uint32_t k=s.worst;std::cerr<<"depthmap context="<<tag<<" max_depth_choices="<<s.max_depth_choices<<" side="<<(((k>>23)&1u)?"high":"low")<<" phase="<<(((k>>24)&1u)?"reverse":"forward")<<" stream="<<((k>>20)&7u)<<" p="<<((k>>8)&0xfffu)<<" height="<<(k&0xffu)<<" pattern="<<s.worst_pattern<<" depth_mask=0x"<<std::hex<<s.worst_mask<<std::dec<<'\n';
}

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseSplit54Host rs=build_reverse_split54(layout,rb,false);
    build_bucket_forward_pattern10(layout,bo,bf);build_reverse_split54_pattern10(layout,bf,rs);BucketPattern10Depth8Host d=build_bucket_pattern10_depth8(layout,bf,bo,rs);

    DepthMapBooks books;size_t lp=size_t(bo.low_nblocks)+1,hp=size_t(bo.high_nblocks)+1;
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);for(uint32_t bid=0;bid<bo.low_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;auto scan=[&](const auto&ops,const auto&off,const auto&dep,uint32_t sid){for(uint32_t q=off[size_t(pi)*lp+bid];q<off[size_t(pi)*lp+bid+1];++q)books.add(false,false,sid,p,xb.he,ops[q],dep[q]);};scan(bo.low_nn,bo.low_nn_off,d.f_low_nn,0);scan(bo.low_nr,bo.low_nr_off,d.f_low_nr,1);scan(bo.low_nl,bo.low_nl_off,d.f_low_nl,2);}}
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){uint32_t pi=uint32_t((TARGET_W-1)-p);for(uint32_t bid=0;bid<bo.high_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;auto scan=[&](const auto&ops,const auto&off,const auto&dep,uint32_t sid){for(uint32_t q=off[size_t(pi)*hp+bid];q<off[size_t(pi)*hp+bid+1];++q)books.add(false,true,sid,p,xb.hs,ops[q],dep[q]);};scan(bo.high_nn,bo.high_nn_off,d.f_high_nn,0);scan(bo.high_nrnl,bo.high_nrnl_off,d.f_high_nrnl,3);}}
    size_t rp=size_t(rs.nblocks)+1;
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;auto scan=[&](const auto&ops,const auto&off,const auto&dep,uint32_t sid){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q)books.add(true,false,sid,p,xb.he,ops[q],dep[q]);};scan(rs.low.nn,rs.low.nn_off,d.r_low_nn,0);scan(rs.low.nr,rs.low.nr_off,d.r_low_nr,1);scan(rs.low.nl,rs.low.nl_off,d.r_low_nl,2);}}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;auto scan=[&](const auto&ops,const auto&off,const auto&dep,uint32_t sid){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q)books.add(true,true,sid,p,xb.hs,ops[q],dep[q]);};scan(rs.high.nn,rs.high.nn_off,d.r_high_nn,0);scan(rs.high.nr,rs.high.nr_off,d.r_high_nr,1);scan(rs.high.nl,rs.high.nl_off,d.r_high_nl,2);}}

    auto a=depthmap_stats(books.coarse),b=depthmap_stats(books.phase),c=depthmap_stats(books.stream);print_worst("coarse",a);print_worst("phase",b);print_worst("phase-stream",c);
    auto dense4_mib=[](const DepthMapStats&s){return double(s.contexts*512ull)/double(1<<20);}; // 1024 nibbles/context
    auto dense8_mib=[](const DepthMapStats&s){return double(s.contexts*1024ull)/double(1<<20);};
    std::cout<<std::setprecision(12)<<"closure-pattern10-depthmap-plan OK W="<<TARGET_W<<" ops="<<d.ops()<<" inserted="<<books.inserted
        <<" coarse_contexts="<<a.contexts<<" coarse_ambiguous="<<a.ambiguous_patterns<<" coarse_max_choices="<<a.max_depth_choices<<" coarse_function="<<(a.max_depth_choices<=1?1:0)<<" coarse_dense4_mib="<<dense4_mib(a)<<" coarse_dense8_mib="<<dense8_mib(a)
        <<" phase_contexts="<<b.contexts<<" phase_ambiguous="<<b.ambiguous_patterns<<" phase_max_choices="<<b.max_depth_choices<<" phase_function="<<(b.max_depth_choices<=1?1:0)<<" phase_dense4_mib="<<dense4_mib(b)<<" phase_dense8_mib="<<dense8_mib(b)
        <<" stream_contexts="<<c.contexts<<" stream_ambiguous="<<c.ambiguous_patterns<<" stream_max_choices="<<c.max_depth_choices<<" stream_function="<<(c.max_depth_choices<=1?1:0)<<" stream_dense4_mib="<<dense4_mib(c)<<" stream_dense8_mib="<<dense8_mib(c)
        <<" depth4_sidecar_mib="<<double((d.ops()+1)/2)/double(1<<20)<<'\n';
    return 0;
}
