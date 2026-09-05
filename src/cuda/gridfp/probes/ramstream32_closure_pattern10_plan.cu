#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main pattern10_plan_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main
#include "../ramstream32_bucket_closure_pattern10.cuh"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>

struct P10Hash{uint64_t x=0,s=0,n=0;};
static void p10_mix(P10Hash&h,uint64_t z){z&=BKCP10_BASE_MASK;h.x^=(z+0x9e3779b97f4a7c15ull+(h.n<<6)+(h.n>>2));h.s+=z*0x100000001b3ull+0x517cc1b727220a95ull;++h.n;}
static P10Hash p10_forward_hash(const BucketOrbitStreamsHost&o){P10Hash h;auto add=[&](const auto&v){for(auto z:v)p10_mix(h,z);};add(o.low_nn);add(o.low_nr);add(o.low_nl);add(o.high_nn);add(o.high_nrnl);return h;}
static P10Hash p10_reverse_hash(const ReverseSplit54Host&o){P10Hash h;auto add=[&](const auto&v){for(auto z:v)p10_mix(h,z);};add(o.low.nn);add(o.low.nr);add(o.low.nl);add(o.high.nn);add(o.high.nr);add(o.high.nl);return h;}
static bool p10_same(P10Hash a,P10Hash b){return a.x==b.x&&a.s==b.s&&a.n==b.n;}
static uint16_t p10_max_forward(const BucketOrbitStreamsHost&o,uint64_t&none){uint16_t m=0;none=0;auto add=[&](const auto&v){for(auto z:v){uint16_t id=bkcp10_id(z);if(id==oneesan::gridfp::CLOSURE_PATTERN10_NONE)++none;else m=std::max(m,id);}};add(o.low_nn);add(o.low_nr);add(o.low_nl);add(o.high_nn);add(o.high_nrnl);return m;}
static uint16_t p10_max_reverse(const ReverseSplit54Host&o,uint64_t&none){uint16_t m=0;none=0;auto add=[&](const auto&v){for(auto z:v){uint16_t id=bkcp10_id(z);if(id==oneesan::gridfp::CLOSURE_PATTERN10_NONE)++none;else m=std::max(m,id);}};add(o.low.nn);add(o.low.nr);add(o.low.nl);add(o.high.nn);add(o.high.nr);add(o.high.nl);return m;}

struct P10JointBucket{
    std::set<uint16_t> codes;
    bool has_none=false;
    uint8_t max_depth=0;
    void add(uint16_t id,int depth){
        if(id==oneesan::gridfp::CLOSURE_PATTERN10_NONE){has_none=true;return;}
        if(depth<0||depth>15){std::cerr<<"pattern10 joint invalid depth="<<depth<<'\n';std::exit(565);}
        codes.insert(uint16_t((id<<4)|uint16_t(depth)));
        max_depth=std::max(max_depth,uint8_t(depth));
    }
    size_t needed()const{return codes.size()+(has_none?1u:0u);}
};

struct P10JointStats{
    std::array<P10JointBucket,LOW_LUT_K> fl{},rl{};
    std::array<P10JointBucket,HIGH_LUT_K> fh{},rh{};
};

static P10JointStats p10_joint_stats(
    const StorageLayout&layout,const BucketFusedHost&bf,
    const BucketOrbitStreamsHost&bo,const ReverseSplit54Host&rs
){
    P10JointStats s;
    auto low_depth=[&](BucketOrbitOp op,const StorageBlock&db,int p,bool first){
        uint16_t id=bkcp10_id(op);if(id==oneesan::gridfp::CLOSURE_PATTERN10_NONE)return std::pair<uint16_t,int>{id,0};
        uint32_t loc=first?bkf_orbit_src(op):bkf_orbit_drop(op);uint32_t dc=bkcp10_low_code_host(bf,loc,db.hs);
        MateID d=first?(MateID(dc)|(MateID(db.c)<<(2*LOW_LUT_K))):minsert(MateID(dc),p,N),s0=0;
        int depth=oneesan::gridfp::low_cross_preimage_partial(d,LOW_LUT_K+1,p,s0);return std::pair<uint16_t,int>{id,depth};
    };
    auto high_depth=[&](BucketOrbitOp op,const StorageBlock&db,int rel,bool edge){
        uint16_t id=bkcp10_id(op);if(id==oneesan::gridfp::CLOSURE_PATTERN10_NONE)return std::pair<uint16_t,int>{id,0};
        uint32_t loc=edge?bkf_orbit_src(op):bkf_orbit_drop(op);uint32_t dc=bkcp10_high_code_host(bf,loc,db.he);
        MateID d=edge?(MateID(db.c)|(MateID(dc)<<2)):minsert(MateID(dc),rel,N),s0=0;
        int depth=oneesan::gridfp::high_cross_preimage_partial(d,HIGH_LUT_K+1,rel,s0);return std::pair<uint16_t,int>{id,depth};
    };

    size_t lp=size_t(bo.low_nblocks)+1,hp=size_t(bo.high_nblocks)+1;
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);auto&st=s.fl[size_t(p-1)];for(uint32_t bid=0;bid<bo.low_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=p==1?xb:layout.block_blocks[xb.he];auto scan=[&](const auto&v,const auto&off){for(uint32_t q=off[size_t(pi)*lp+bid];q<off[size_t(pi)*lp+bid+1];++q){auto[id,d]=low_depth(v[q],db,p,p==1);st.add(id,d);}};scan(bo.low_nn,bo.low_nn_off);scan(bo.low_nr,bo.low_nr_off);scan(bo.low_nl,bo.low_nl_off);}}
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){int rel=p-LOW_LUT_K;uint32_t pi=uint32_t((TARGET_W-1)-p);auto&st=s.fh[size_t(rel-1)];for(uint32_t bid=0;bid<bo.high_nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=layout.block_blocks[xb.hs];auto scan=[&](const auto&v,const auto&off){for(uint32_t q=off[size_t(pi)*hp+bid];q<off[size_t(pi)*hp+bid+1];++q){auto[id,d]=high_depth(v[q],db,rel,false);st.add(id,d);}};scan(bo.high_nn,bo.high_nn_off);scan(bo.high_nrnl,bo.high_nrnl_off);}}

    size_t rp=size_t(rs.nblocks)+1;
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);auto&st=s.rl[size_t(p-1)];for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;const auto&db=layout.block_blocks[xb.he];auto scan=[&](const auto&v,const auto&off){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q){uint16_t id=bkcp10_id(v[q]);if(id==oneesan::gridfp::CLOSURE_PATTERN10_NONE){st.add(id,0);continue;}uint32_t loc=bkf_orbit_drop(v[q]),dc=bkcp10_low_code_host(bf,loc,db.hs);MateID d=blocked_exclude_reverse(MateID(dc),LOW_LUT_K+1,p),s0=0;int depth=oneesan::gridfp::low_cross_preimage_partial(d,LOW_LUT_K+1,p,s0);st.add(id,depth);}};scan(rs.low.nn,rs.low.nn_off);scan(rs.low.nr,rs.low.nr_off);scan(rs.low.nl,rs.low.nl_off);}}
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){int rel=p-LOW_LUT_K;uint32_t pi=uint32_t(rel-1);bool edge=p==TARGET_W-1;auto&st=s.rh[size_t(rel-1)];for(uint32_t bid=0;bid<rs.nblocks;++bid){const auto&xb=layout.main_blocks[bid];if(!xb.valid)continue;auto scan=[&](const auto&v,const auto&off){for(uint32_t q=off[size_t(pi)*rp+bid];q<off[size_t(pi)*rp+bid+1];++q){uint16_t id=bkcp10_id(v[q]);if(id==oneesan::gridfp::CLOSURE_PATTERN10_NONE){st.add(id,0);continue;}const auto&db=edge?xb:layout.block_blocks[xb.hs];uint32_t loc=edge?bkf_orbit_src(v[q]):bkf_orbit_drop(v[q]),dc=bkcp10_high_code_host(bf,loc,db.he);MateID d=edge?(MateID(db.c)|(MateID(dc)<<2)):blocked_exclude_reverse(MateID(dc),HIGH_LUT_K+1,rel),s0=0;int depth=oneesan::gridfp::high_cross_preimage_partial(d,HIGH_LUT_K+1,rel,s0);st.add(id,depth);}};scan(rs.high.nn,rs.high.nn_off);scan(rs.high.nr,rs.high.nr_off);scan(rs.high.nl,rs.high.nl_off);}}
    return s;
}

template<size_t N>
static size_t p10_report_joint(const char*tag,const std::array<P10JointBucket,N>&a,int p0){
    size_t mx=0;for(size_t i=0;i<N;++i){mx=std::max(mx,a[i].needed());std::cout<<"pattern10-joint side="<<tag<<" p="<<(p0+int(i))<<" combinations="<<a[i].needed()<<" active_pairs="<<a[i].codes.size()<<" none="<<(a[i].has_none?1:0)<<" max_depth="<<unsigned(a[i].max_depth)<<'\n';}return mx;
}

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseSplit54Host rs=build_reverse_split54(layout,rb,false);
    P10Hash fb=p10_forward_hash(bo),rbh=p10_reverse_hash(rs);build_bucket_forward_pattern10(layout,bo,bf);build_reverse_split54_pattern10(layout,bf,rs);P10Hash fa=p10_forward_hash(bo),rah=p10_reverse_hash(rs);
    if(!p10_same(fb,fa)||!p10_same(rbh,rah)){std::cerr<<"pattern10 changed lower54 locator payload\n";return 2;}
    uint64_t fn=0,rn=0;uint16_t fm=p10_max_forward(bo,fn),rm=p10_max_reverse(rs,rn);if(fm>=oneesan::gridfp::CLOSURE_PATTERN10_NONE||rm>=oneesan::gridfp::CLOSURE_PATTERN10_NONE){std::cerr<<"pattern10 id overflow\n";return 3;}
    uint16_t bound_low=0,bound_high=0;for(int p=1;p<LOW_LUT_K+1;++p)bound_low=std::max(bound_low,oneesan::gridfp::closure_pattern10_count(LOW_LUT_K+1,p));for(int p=1;p<HIGH_LUT_K+1;++p)bound_high=std::max(bound_high,oneesan::gridfp::closure_pattern10_count(HIGH_LUT_K+1,p));if(bound_low>=oneesan::gridfp::CLOSURE_PATTERN10_NONE||bound_high>=oneesan::gridfp::CLOSURE_PATTERN10_NONE)return 4;
    P10JointStats js=p10_joint_stats(layout,bf,bo,rs);size_t jfl=p10_report_joint("forward-low",js.fl,1),jfh=p10_report_joint("forward-high",js.fh,LOW_LUT_K+1),jrl=p10_report_joint("reverse-low",js.rl,1),jrh=p10_report_joint("reverse-high",js.rh,LOW_LUT_K+1);size_t jmax=std::max(std::max(jfl,jfh),std::max(jrl,jrh));
    std::cout<<"closure-pattern10-plan OK W="<<TARGET_W<<" forward_ops="<<fa.n<<" reverse_ops="<<rah.n<<" forward_none="<<fn<<" reverse_none="<<rn<<" forward_max_id="<<fm<<" reverse_max_id="<<rm<<" low_bound="<<bound_low<<" high_bound="<<bound_high<<" lower54_unchanged=1 extra_bytes=0 joint_pattern_depth_max="<<jmax<<" joint_pattern_depth_fits10="<<(jmax<=1024?1:0)<<"\n";
    return 0;
}
