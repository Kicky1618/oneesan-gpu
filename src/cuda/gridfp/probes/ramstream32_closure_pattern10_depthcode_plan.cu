#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main pattern10_depthcode_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_closure_pattern10_depth8.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>

struct DepthCodeBooks {
    std::map<uint32_t,std::set<uint16_t>> coarse,phase,stream;
    uint64_t inserted=0;
    static uint16_t pair(BucketOrbitOp op,uint8_t depth){return uint16_t((uint16_t(bkcp10_id(op))<<4)|uint16_t(depth));}
    static uint32_t key(bool rev,bool high,uint32_t stream_id,int p,uint32_t h,int mode){
        uint32_t k=(uint32_t(high)<<23)|(uint32_t(p)<<8)|h;
        if(mode>=1)k|=uint32_t(rev)<<24;
        if(mode>=2)k|=(stream_id&7u)<<20;
        return k;
    }
    void add(bool rev,bool high,uint32_t stream_id,int p,uint32_t h,BucketOrbitOp op,uint8_t depth){uint16_t x=pair(op,depth);coarse[key(rev,high,stream_id,p,h,0)].insert(x);phase[key(rev,high,stream_id,p,h,1)].insert(x);stream[key(rev,high,stream_id,p,h,2)].insert(x);++inserted;}
};
struct DepthCodeStats{size_t contexts=0,pairs=0,max_pairs=0;uint32_t worst=0;};
static DepthCodeStats depthcode_stats(const std::map<uint32_t,std::set<uint16_t>>&m){DepthCodeStats z;for(const auto&[k,s]:m){if(s.empty())continue;++z.contexts;z.pairs+=s.size();if(s.size()>z.max_pairs){z.max_pairs=s.size();z.worst=k;}}return z;}
static void depthcode_print_worst(const char*tag,const DepthCodeStats&s){uint32_t k=s.worst;std::cerr<<"depthcode context="<<tag<<" max_pairs="<<s.max_pairs<<" side="<<(((k>>23)&1u)?"high":"low")<<" phase="<<(((k>>24)&1u)?"reverse":"forward")<<" stream="<<((k>>20)&7u)<<" p="<<((k>>8)&0xfffu)<<" height="<<(k&0xffu)<<'\n';}

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseSplit54Host rs=build_reverse_split54(layout,rb,false);
    build_bucket_forward_pattern10(layout,bo,bf);build_reverse_split54_pattern10(layout,bf,rs);BucketPattern10Depth8Host d=build_bucket_pattern10_depth8(layout,bf,bo,rs);

    DepthCodeBooks books;size_t lp=size_t(bo.low_nblocks)+1,hp=size_t(bo.high_nblocks)+1;
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);for(uint32_t bid=0;bid<bo.low_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;auto scan=[&](const auto&ops,const auto&off,const auto&dep,uint32_t sid){for(uint32_t q=off[size_t(pi)*lp+bid];q<off[size_t(pi)*lp+bid+1];++q)books.add(false,false,sid,p,xb.he,ops[q],dep[q]);};scan(bo.low_nn,bo.low_nn_off,d.f_low_nn,0);scan(bo.low_nr,bo.low_nr_off,d.f_low_nr,1);scan(bo.low_nl,bo.low_nl_off,d.f_low_nl,2);}}
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){uint32_t pi=uint32_t((TARGET_W-1)-p);for(uint32_t bid=0;bid<bo.high_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;auto scan=[&](const auto&ops,const auto&off,const auto&dep,uint32_t sid){for(uint32_t q=off[size_t(pi)*hp+bid];q<off[size_t(pi)*hp+bid+1];++q)books.add(false,true,sid,p,xb.hs,ops[q],dep[q]);};scan(bo.high_nn,bo.high_nn_off,d.f_high_nn,0);scan(bo.high_nrnl,bo.high_nrnl_off,d.f_high_nrnl,3);}}

    size_t rp=size_t(rs.nblocks)+1;
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;auto scan=[&](const auto&ops,const auto&off,const auto&dep,uint32_t sid){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q)books.add(true,false,sid,p,xb.he,ops[q],dep[q]);};scan(rs.low.nn,rs.low.nn_off,d.r_low_nn,0);scan(rs.low.nr,rs.low.nr_off,d.r_low_nr,1);scan(rs.low.nl,rs.low.nl_off,d.r_low_nl,2);}}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;auto scan=[&](const auto&ops,const auto&off,const auto&dep,uint32_t sid){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q)books.add(true,true,sid,p,xb.hs,ops[q],dep[q]);};scan(rs.high.nn,rs.high.nn_off,d.r_high_nn,0);scan(rs.high.nr,rs.high.nr_off,d.r_high_nr,1);scan(rs.high.nl,rs.high.nl_off,d.r_high_nl,2);}}

    DepthCodeStats a=depthcode_stats(books.coarse),b=depthcode_stats(books.phase),c=depthcode_stats(books.stream);depthcode_print_worst("coarse",a);depthcode_print_worst("phase",b);depthcode_print_worst("phase-stream",c);
    auto dense_mib=[](const DepthCodeStats&s){return double(s.contexts*1024ull*sizeof(uint16_t))/double(1<<20);};auto compact_mib=[](const DepthCodeStats&s){return double(s.pairs*sizeof(uint16_t))/double(1<<20);};
    std::cout<<std::setprecision(12)
        <<"closure-pattern10-depthcode-plan OK W="<<TARGET_W
        <<" ops="<<d.ops()<<" inserted="<<books.inserted
        <<" coarse_contexts="<<a.contexts<<" coarse_max_pairs="<<a.max_pairs<<" coarse_fits10="<<(a.max_pairs<=1024?1:0)<<" coarse_dense_mib="<<dense_mib(a)<<" coarse_compact_mib="<<compact_mib(a)
        <<" phase_contexts="<<b.contexts<<" phase_max_pairs="<<b.max_pairs<<" phase_fits10="<<(b.max_pairs<=1024?1:0)<<" phase_dense_mib="<<dense_mib(b)<<" phase_compact_mib="<<compact_mib(b)
        <<" stream_contexts="<<c.contexts<<" stream_max_pairs="<<c.max_pairs<<" stream_fits10="<<(c.max_pairs<=1024?1:0)<<" stream_dense_mib="<<dense_mib(c)<<" stream_compact_mib="<<compact_mib(c)
        <<" depth8_sidecar_mib="<<double(d.bytes())/double(1<<20)<<" sidecar_bytes_per_orbit=1\n";
    return 0;
}
