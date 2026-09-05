#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main ocf_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main

#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

using OcfKey=uint64_t;
static OcfKey ocf_key(uint32_t bid,uint32_t loc){return (uint64_t(bid)<<BUCKET_LOCATOR_BITS)|loc;}

struct OcfStats{
    uint64_t destinations=0;
    uint64_t source_refs=0;
    uint64_t orbit_targets=0;
    uint64_t untouched_sources=0;
};

static void ocf_add_unique(std::unordered_map<OcfKey,uint32_t>&m,OcfKey k,uint32_t tag,const char*what){
    auto [it,ok]=m.emplace(k,tag);if(!ok){std::cerr<<"fusion duplicate orbit target "<<what<<" key="<<k<<'\n';std::exit(350);}
}
static void ocf_add_mut(std::unordered_set<OcfKey>&s,OcfKey k){s.insert(k);}
static void ocf_check_source(const std::unordered_set<OcfKey>&mut,uint32_t packed,OcfStats&st,const char*what){
    OcfKey k=ocf_key(bkf_src_block(packed),bkf_src_locator(packed));
    if(mut.count(k)){std::cerr<<"fusion closure source modified by orbit "<<what<<" key="<<k<<'\n';std::exit(351);}
    ++st.source_refs;
}
static void ocf_check_dest(const std::unordered_map<OcfKey,uint32_t>&target,uint32_t bid,uint32_t loc,OcfStats&st,const char*what){
    OcfKey k=ocf_key(bid,loc);if(target.find(k)==target.end()){std::cerr<<"fusion destination lacks orbit target "<<what<<" bid="<<bid<<" loc="<<loc<<'\n';std::exit(352);}++st.destinations;
}

static OcfStats ocf_forward_low(const StorageLayout&layout,const BucketOrbitStreamsHost&o,const BucketFusedHost&f){
    OcfStats st;size_t pitch=size_t(o.low_nblocks)+1;
    for(int p=LOW_LUT_K;p>=1;--p){uint32_t pi=uint32_t(LOW_LUT_K-p);
        std::unordered_map<OcfKey,uint32_t> target;std::unordered_set<OcfKey> mut;
        for(uint32_t bid=0;bid<o.low_nblocks;++bid){const auto&xb=layout.main_blocks[bid];FBlock fx{};fx.he=xb.he;fx.hs=xb.hs;fx.c=xb.c;
            auto scan=[&](const std::vector<BucketOrbitOp>&v,const std::vector<uint32_t>&off,uint32_t kind){
                uint32_t a=off[size_t(pi)*pitch+bid],b=off[size_t(pi)*pitch+bid+1];uint32_t jbid=cpu_sparse_jblock(bid,fx,p,kind),dbid=uint32_t(xb.he);
                for(uint32_t q=a;q<b;++q){auto w=v[q];OcfKey sk=ocf_key(bid,bkf_orbit_src(w)),jk=ocf_key(jbid,bkf_orbit_partner(w));ocf_add_mut(mut,sk);ocf_add_mut(mut,jk);
                    if(p==1){if(kind==CPU_ORBIT_NN)ocf_add_unique(target,sk,q,"forward-low-main");}
                    else ocf_add_unique(target,ocf_key(dbid,bkf_orbit_drop(w)),q,"forward-low-drop");
                }};
            scan(o.low_nn,o.low_nn_off,CPU_ORBIT_NN);scan(o.low_nr,o.low_nr_off,CPU_ORBIT_NR);scan(o.low_nl,o.low_nl_off,CPU_ORBIT_NL);
        }
        st.orbit_targets+=target.size();
        bool tm=p==1;uint32_t nt=tm?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());
        for(uint32_t dbid=0;dbid<nt;++dbid){uint32_t a=f.low_off[size_t(pi)*f.low_pitch+dbid],b=f.low_off[size_t(pi)*f.low_pitch+dbid+1];for(uint32_t q=a;q<b;++q){const auto&r=f.low_dst[q];ocf_check_dest(target,dbid,r.dst_locator,st,"forward-low");uint32_t lc=r.counts&0xffffu,cc=r.counts>>16;for(uint32_t e=r.local_begin;e<r.local_begin+lc;++e)ocf_check_source(mut,f.low_local_src[e],st,"forward-low-local");for(uint32_t e=r.cross_begin;e<r.cross_begin+cc;++e)ocf_check_source(mut,f.low_cross_op[e],st,"forward-low-cross");}}
    }
    st.untouched_sources=st.source_refs;return st;
}

static OcfStats ocf_forward_high(const StorageLayout&layout,const BucketOrbitStreamsHost&o,const BucketFusedHost&f){
    OcfStats st;size_t pitch=size_t(o.high_nblocks)+1;
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){uint32_t pi=uint32_t((TARGET_W-1)-p);
        std::unordered_map<OcfKey,uint32_t> target;std::unordered_set<OcfKey> mut;
        for(uint32_t bid=0;bid<o.high_nblocks;++bid){const auto&xb=layout.main_blocks[bid];FBlock fx{};fx.he=xb.he;fx.hs=xb.hs;fx.c=xb.c;uint32_t dbid=cpu_high_orbit_drop_block(fx);
            auto scan=[&](const std::vector<BucketOrbitOp>&v,const std::vector<uint32_t>&off,bool nn){uint32_t a=off[size_t(pi)*pitch+bid],b=off[size_t(pi)*pitch+bid+1],jbid=cpu_high_orbit_partner_block(bid,fx,p,nn);for(uint32_t q=a;q<b;++q){auto w=v[q];ocf_add_mut(mut,ocf_key(bid,bkf_orbit_src(w)));ocf_add_mut(mut,ocf_key(jbid,bkf_orbit_partner(w)));ocf_add_unique(target,ocf_key(dbid,bkf_orbit_drop(w)),q,"forward-high-drop");}};
            scan(o.high_nn,o.high_nn_off,true);scan(o.high_nrnl,o.high_nrnl_off,false);
        }
        st.orbit_targets+=target.size();uint32_t nt=uint32_t(layout.block_blocks.size());
        for(uint32_t dbid=0;dbid<nt;++dbid){uint32_t a=f.high_off[size_t(pi)*f.high_pitch+dbid],b=f.high_off[size_t(pi)*f.high_pitch+dbid+1];for(uint32_t q=a;q<b;++q){const auto&r=f.high_dst[q];ocf_check_dest(target,dbid,r.dst_locator,st,"forward-high");uint32_t lc=r.counts&0xffffu,cc=r.counts>>16;for(uint32_t e=r.local_begin;e<r.local_begin+lc;++e)ocf_check_source(mut,f.high_local_src[e],st,"forward-high-local");for(uint32_t e=r.cross_begin;e<r.cross_begin+cc;++e)ocf_check_source(mut,f.high_cross_op[e],st,"forward-high-cross");}}
    }
    st.untouched_sources=st.source_refs;return st;
}

static OcfStats ocf_reverse_low(const StorageLayout&layout,const ReverseBucketAtomicHost&o,const ReverseBucketFusedHost&f){
    OcfStats st;size_t pitch=size_t(o.nblocks)+1;
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);std::unordered_map<OcfKey,uint32_t>target;std::unordered_set<OcfKey>mut;
        for(uint32_t bid=0;bid<o.nblocks;++bid){uint32_t a=o.low_orbit_off[size_t(pi)*pitch+bid],b=o.low_orbit_off[size_t(pi)*pitch+bid+1],dbid=uint32_t(layout.main_blocks[bid].he);for(uint32_t q=a;q<b;++q){auto w=o.low_orbit[q];ocf_add_mut(mut,ocf_key(bid,rb_orbit_src(w)));ocf_add_mut(mut,ocf_key(rb_orbit_jblock(w),rb_orbit_partner(w)));ocf_add_unique(target,ocf_key(dbid,rb_orbit_drop(w)),q,"reverse-low-drop");}}
        st.orbit_targets+=target.size();uint32_t nt=uint32_t(layout.block_blocks.size());for(uint32_t dbid=0;dbid<nt;++dbid){uint32_t a=f.low_off[size_t(pi)*f.low_pitch+dbid],b=f.low_off[size_t(pi)*f.low_pitch+dbid+1];for(uint32_t q=a;q<b;++q){const auto&r=f.low_dst[q];ocf_check_dest(target,dbid,r.dst_locator,st,"reverse-low");uint32_t lc=r.counts&0xffffu,cc=r.counts>>16;for(uint32_t e=r.local_begin;e<r.local_begin+lc;++e)ocf_check_source(mut,f.low_local_src[e],st,"reverse-low-local");for(uint32_t e=r.cross_begin;e<r.cross_begin+cc;++e)ocf_check_source(mut,f.low_cross_op[e],st,"reverse-low-cross");}}
    }
    st.untouched_sources=st.source_refs;return st;
}

static OcfStats ocf_reverse_high(const StorageLayout&layout,const ReverseBucketAtomicHost&o,const ReverseBucketFusedHost&f){
    OcfStats st;size_t pitch=size_t(o.nblocks)+1;
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));bool edge=p==TARGET_W-1;std::unordered_map<OcfKey,uint32_t>target;std::unordered_set<OcfKey>mut;
        for(uint32_t bid=0;bid<o.nblocks;++bid){uint32_t a=o.high_orbit_off[size_t(pi)*pitch+bid],b=o.high_orbit_off[size_t(pi)*pitch+bid+1],dbid=uint32_t(layout.main_blocks[bid].hs);for(uint32_t q=a;q<b;++q){auto w=o.high_orbit[q];OcfKey sk=ocf_key(bid,rb_orbit_src(w));ocf_add_mut(mut,sk);ocf_add_mut(mut,ocf_key(rb_orbit_jblock(w),rb_orbit_partner(w)));if(edge){if(rb_orbit_kind(w)==CPU_ORBIT_NN)ocf_add_unique(target,sk,q,"reverse-high-main");}else ocf_add_unique(target,ocf_key(dbid,rb_orbit_drop(w)),q,"reverse-high-drop");}}
        st.orbit_targets+=target.size();uint32_t nt=edge?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());for(uint32_t dbid=0;dbid<nt;++dbid){uint32_t a=f.high_off[size_t(pi)*f.high_pitch+dbid],b=f.high_off[size_t(pi)*f.high_pitch+dbid+1];for(uint32_t q=a;q<b;++q){const auto&r=f.high_dst[q];ocf_check_dest(target,dbid,r.dst_locator,st,"reverse-high");uint32_t lc=r.counts&0xffffu,cc=r.counts>>16;for(uint32_t e=r.local_begin;e<r.local_begin+lc;++e)ocf_check_source(mut,f.high_local_src[e],st,"reverse-high-local");for(uint32_t e=r.cross_begin;e<r.cross_begin+cc;++e)ocf_check_source(mut,f.high_cross_op[e],st,"reverse-high-cross");}}
    }
    st.untouched_sources=st.source_refs;return st;
}

static void ocf_print(const char*name,const OcfStats&s){std::cout<<"orbit-closure-fusion side="<<name<<" destinations="<<s.destinations<<" source_refs="<<s.source_refs<<" orbit_targets="<<s.orbit_targets<<" source_disjoint="<<s.untouched_sources<<" OK\n";}

int main(){
    build_full_dp();G_FACTOR=build_factor_tables();StorageFactorHost storage=build_storage_factor_tables(G_FACTOR);StorageLayout layout=build_storage_layout(storage);
    LowDescHost lowdesc=build_low_descriptors(storage,layout);HighDescHost highdesc=build_high_descriptors(storage,layout);LowOrbitHost loworbit=build_cpu_low_orbit(storage,layout,lowdesc);CpuHighDirectHost highdirect=build_cpu_high_direct(storage,layout,highdesc);GpuDirectGatherHost ordinary=build_gpu_direct_gather(layout,lowdesc,loworbit,highdirect);GpuDirectCrossGatherHost cross=build_gpu_direct_cross_gather(storage,layout,lowdesc,loworbit,highdirect);GpuDirectFusedHost fused=build_gpu_direct_fused_checked(layout,ordinary,cross);CpuLowSparseHost lowsparse=build_cpu_low_sparse(storage,layout,lowdesc,loworbit);BucketOwnerHost owner=build_bucket_owners(G_FACTOR,storage);BucketOrbitStreamsHost bo=build_bucket_orbits(storage,layout,owner,lowsparse,highdirect);BucketFusedHost bf=build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow=build_reverse_low_descriptors(storage,layout);ReverseHighDescHost rhigh=build_reverse_high_descriptors(storage,layout);ReverseOrbitHost rlo=build_reverse_orbit(storage,layout,true),rhi=build_reverse_orbit(storage,layout,false);ReverseBucketAtomicHost rb=build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi);ReverseBucketFusedHost rf=build_reverse_bucket_fused_checked(layout,owner,rb);
    ocf_print("forward-low",ocf_forward_low(layout,bo,bf));ocf_print("forward-high",ocf_forward_high(layout,bo,bf));ocf_print("reverse-low",ocf_reverse_low(layout,rb,rf));ocf_print("reverse-high",ocf_reverse_high(layout,rb,rf));
    std::cout<<"orbit-closure-fusion-plan OK W="<<TARGET_W<<"\n";return 0;
}
