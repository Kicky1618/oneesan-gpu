#pragma once

#include "ramstream32_bucket_reverse_fused.cuh"
#ifndef GPU_DIRECT_PM_ACCUM
#define GPU_DIRECT_PM_ACCUM 0
#endif
#if GPU_DIRECT_PM_ACCUM
#include "ramstream32_bucket_fused_pm.cuh"
#endif

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

// One-pass orbit + destination-closure executor.
//
// Every closure destination is attached to the unique orbit output that owns
// that cell at the same p.  Closure sources are disjoint from all orbit
// source/partner rows (verified by ramstream32_orbit_closure_fusion_plan.cu),
// so the orbit thread may gather the closure contribution before performing
// its in-place writes and then fold that contribution into source/drop output.
// This removes the orbit->closure kernel boundary without adding atomics or
// scratch storage.

static constexpr uint32_t BKOC_NONE = 0xffffffffu;
using BkocKey = uint64_t;
static inline BkocKey bkoc_key(uint32_t bid,uint32_t loc){
    return (uint64_t(bid)<<BUCKET_LOCATOR_BITS)|uint64_t(loc);
}

struct BucketForwardOrbitClosureAttachHost {
    std::vector<uint32_t> low_nn,low_nr,low_nl,high_nn,high_nrnl;
    size_t bytes()const{return (low_nn.size()+low_nr.size()+low_nl.size()+high_nn.size()+high_nrnl.size())*sizeof(uint32_t);}
};
struct BucketReverseOrbitClosureAttachHost {
    std::vector<uint32_t> low,high;
    size_t bytes()const{return (low.size()+high.size())*sizeof(uint32_t);}
};

static void bkoc_add_dst(std::unordered_map<BkocKey,uint32_t>&m,uint32_t bid,uint32_t loc,uint32_t q,const char*what){
    auto [it,ok]=m.emplace(bkoc_key(bid,loc),q);
    if(!ok){std::cerr<<"orbit-closure duplicate destination "<<what<<" bid="<<bid<<" loc="<<loc<<'\n';std::exit(360);}
}
static uint32_t bkoc_lookup(const std::unordered_map<BkocKey,uint32_t>&m,uint32_t bid,uint32_t loc){
    auto it=m.find(bkoc_key(bid,loc));return it==m.end()?BKOC_NONE:it->second;
}

static BucketForwardOrbitClosureAttachHost build_bucket_forward_orbit_closure_attach(
    const StorageLayout&layout,const BucketOrbitStreamsHost&o,const BucketFusedHost&f
){
    BucketForwardOrbitClosureAttachHost out;
    out.low_nn.assign(o.low_nn.size(),BKOC_NONE);out.low_nr.assign(o.low_nr.size(),BKOC_NONE);out.low_nl.assign(o.low_nl.size(),BKOC_NONE);
    out.high_nn.assign(o.high_nn.size(),BKOC_NONE);out.high_nrnl.assign(o.high_nrnl.size(),BKOC_NONE);
    std::vector<uint8_t> used_low(f.low_dst.size()),used_high(f.high_dst.size());
    uint64_t attached=0;
    size_t lpitch=size_t(o.low_nblocks)+1;
    for(int p=LOW_LUT_K;p>=1;--p){
        uint32_t pi=uint32_t(LOW_LUT_K-p);bool tm=p==1;
        uint32_t nt=tm?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());
        std::unordered_map<BkocKey,uint32_t> dst;
        for(uint32_t dbid=0;dbid<nt;++dbid){
            uint32_t a=f.low_off[size_t(pi)*f.low_pitch+dbid],b=f.low_off[size_t(pi)*f.low_pitch+dbid+1];
            for(uint32_t q=a;q<b;++q)bkoc_add_dst(dst,dbid,f.low_dst[q].dst_locator,q,"forward-low");
        }
        for(uint32_t bid=0;bid<o.low_nblocks;++bid){
            const auto&xb=layout.main_blocks[bid];FBlock fx{};fx.he=xb.he;fx.hs=xb.hs;fx.c=xb.c;uint32_t dbid=uint32_t(xb.he);
            auto scan=[&](const std::vector<BucketOrbitOp>&v,const std::vector<uint32_t>&off,std::vector<uint32_t>&att,uint32_t kind){
                uint32_t a=off[size_t(pi)*lpitch+bid],b=off[size_t(pi)*lpitch+bid+1];
                for(uint32_t q=a;q<b;++q){uint32_t rid=BKOC_NONE;
                    if(tm){if(kind==CPU_ORBIT_NN)rid=bkoc_lookup(dst,bid,bkf_orbit_src(v[q]));}
                    else rid=bkoc_lookup(dst,dbid,bkf_orbit_drop(v[q]));
                    if(rid!=BKOC_NONE){if(used_low[rid]++)std::exit(361);att[q]=rid;++attached;}
                }
            };
            scan(o.low_nn,o.low_nn_off,out.low_nn,CPU_ORBIT_NN);
            scan(o.low_nr,o.low_nr_off,out.low_nr,CPU_ORBIT_NR);
            scan(o.low_nl,o.low_nl_off,out.low_nl,CPU_ORBIT_NL);
        }
    }
    size_t hpitch=size_t(o.high_nblocks)+1;
    for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){
        uint32_t pi=uint32_t((TARGET_W-1)-p);std::unordered_map<BkocKey,uint32_t> dst;
        for(uint32_t dbid=0;dbid<layout.block_blocks.size();++dbid){
            uint32_t a=f.high_off[size_t(pi)*f.high_pitch+dbid],b=f.high_off[size_t(pi)*f.high_pitch+dbid+1];
            for(uint32_t q=a;q<b;++q)bkoc_add_dst(dst,dbid,f.high_dst[q].dst_locator,q,"forward-high");
        }
        for(uint32_t bid=0;bid<o.high_nblocks;++bid){
            const auto&xb=layout.main_blocks[bid];FBlock fx{};fx.he=xb.he;fx.hs=xb.hs;fx.c=xb.c;uint32_t dbid=cpu_high_orbit_drop_block(fx);
            auto scan=[&](const std::vector<BucketOrbitOp>&v,const std::vector<uint32_t>&off,std::vector<uint32_t>&att){
                uint32_t a=off[size_t(pi)*hpitch+bid],b=off[size_t(pi)*hpitch+bid+1];
                for(uint32_t q=a;q<b;++q){uint32_t rid=bkoc_lookup(dst,dbid,bkf_orbit_drop(v[q]));if(rid!=BKOC_NONE){if(used_high[rid]++)std::exit(362);att[q]=rid;++attached;}}
            };
            scan(o.high_nn,o.high_nn_off,out.high_nn);scan(o.high_nrnl,o.high_nrnl_off,out.high_nrnl);
        }
    }
    for(size_t q=0;q<used_low.size();++q)if(used_low[q]!=1){std::cerr<<"forward LOW unattached closure q="<<q<<" used="<<unsigned(used_low[q])<<'\n';std::exit(363);}
    for(size_t q=0;q<used_high.size();++q)if(used_high[q]!=1){std::cerr<<"forward HIGH unattached closure q="<<q<<" used="<<unsigned(used_high[q])<<'\n';std::exit(364);}
    if(attached!=f.low_dst.size()+f.high_dst.size())std::exit(365);
    std::cerr<<"bucket_forward_orbit_closure_attach attached="<<attached<<" mib="<<double(out.bytes())/double(1<<20)<<'\n';
    return out;
}

static BucketReverseOrbitClosureAttachHost build_bucket_reverse_orbit_closure_attach(
    const StorageLayout&layout,const ReverseBucketAtomicHost&o,const ReverseBucketFusedHost&f
){
    BucketReverseOrbitClosureAttachHost out;out.low.assign(o.low_orbit.size(),BKOC_NONE);out.high.assign(o.high_orbit.size(),BKOC_NONE);
    std::vector<uint8_t> used_low(f.low_dst.size()),used_high(f.high_dst.size());uint64_t attached=0;size_t pitch=size_t(o.nblocks)+1;
    for(int p=1;p<=LOW_LUT_K;++p){uint32_t pi=uint32_t(p-1);std::unordered_map<BkocKey,uint32_t>dst;
        for(uint32_t dbid=0;dbid<layout.block_blocks.size();++dbid){uint32_t a=f.low_off[size_t(pi)*f.low_pitch+dbid],b=f.low_off[size_t(pi)*f.low_pitch+dbid+1];for(uint32_t q=a;q<b;++q)bkoc_add_dst(dst,dbid,f.low_dst[q].dst_locator,q,"reverse-low");}
        for(uint32_t bid=0;bid<o.nblocks;++bid){uint32_t dbid=uint32_t(layout.main_blocks[bid].he);uint32_t a=o.low_orbit_off[size_t(pi)*pitch+bid],b=o.low_orbit_off[size_t(pi)*pitch+bid+1];for(uint32_t q=a;q<b;++q){uint32_t rid=bkoc_lookup(dst,dbid,rb_orbit_drop(o.low_orbit[q]));if(rid!=BKOC_NONE){if(used_low[rid]++)std::exit(366);out.low[q]=rid;++attached;}}}
    }
    for(int p=LOW_LUT_K+1;p<TARGET_W;++p){uint32_t pi=uint32_t(p-(LOW_LUT_K+1));bool edge=p==TARGET_W-1;uint32_t nt=edge?uint32_t(layout.main_blocks.size()):uint32_t(layout.block_blocks.size());std::unordered_map<BkocKey,uint32_t>dst;
        for(uint32_t dbid=0;dbid<nt;++dbid){uint32_t a=f.high_off[size_t(pi)*f.high_pitch+dbid],b=f.high_off[size_t(pi)*f.high_pitch+dbid+1];for(uint32_t q=a;q<b;++q)bkoc_add_dst(dst,dbid,f.high_dst[q].dst_locator,q,"reverse-high");}
        for(uint32_t bid=0;bid<o.nblocks;++bid){uint32_t dbid=uint32_t(layout.main_blocks[bid].hs);uint32_t a=o.high_orbit_off[size_t(pi)*pitch+bid],b=o.high_orbit_off[size_t(pi)*pitch+bid+1];for(uint32_t q=a;q<b;++q){uint64_t w=o.high_orbit[q];uint32_t rid=BKOC_NONE;if(edge){if(rb_orbit_kind(w)==CPU_ORBIT_NN)rid=bkoc_lookup(dst,bid,rb_orbit_src(w));}else rid=bkoc_lookup(dst,dbid,rb_orbit_drop(w));if(rid!=BKOC_NONE){if(used_high[rid]++)std::exit(367);out.high[q]=rid;++attached;}}}
    }
    for(size_t q=0;q<used_low.size();++q)if(used_low[q]!=1){std::cerr<<"reverse LOW unattached closure q="<<q<<" used="<<unsigned(used_low[q])<<'\n';std::exit(368);}
    for(size_t q=0;q<used_high.size();++q)if(used_high[q]!=1){std::cerr<<"reverse HIGH unattached closure q="<<q<<" used="<<unsigned(used_high[q])<<'\n';std::exit(369);}
    if(attached!=f.low_dst.size()+f.high_dst.size())std::exit(370);
    std::cerr<<"bucket_reverse_orbit_closure_attach attached="<<attached<<" mib="<<double(out.bytes())/double(1<<20)<<'\n';return out;
}

__constant__ uint32_t *D_BKOC_F_LOW_NN,*D_BKOC_F_LOW_NR,*D_BKOC_F_LOW_NL,*D_BKOC_F_HIGH_NN,*D_BKOC_F_HIGH_NRNL;
__constant__ uint32_t *D_BKOC_R_LOW,*D_BKOC_R_HIGH;

struct BucketForwardOrbitClosureAttachDeviceTables{
    uint32_t *low_nn=nullptr,*low_nr=nullptr,*low_nl=nullptr,*high_nn=nullptr,*high_nrnl=nullptr;
    template<class T>static void cp(T*&d,const std::vector<T>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()*sizeof(T)),w);ck(cudaMemcpy(d,s.data(),s.size()*sizeof(T),cudaMemcpyHostToDevice),w);}
    void install(const BucketForwardOrbitClosureAttachHost&h){cp(low_nn,h.low_nn,"bkoc f low nn");cp(low_nr,h.low_nr,"bkoc f low nr");cp(low_nl,h.low_nl,"bkoc f low nl");cp(high_nn,h.high_nn,"bkoc f high nn");cp(high_nrnl,h.high_nrnl,"bkoc f high nrnl");ck(cudaMemcpyToSymbol(D_BKOC_F_LOW_NN,&low_nn,sizeof(low_nn)),"bkoc f low nn ptr");ck(cudaMemcpyToSymbol(D_BKOC_F_LOW_NR,&low_nr,sizeof(low_nr)),"bkoc f low nr ptr");ck(cudaMemcpyToSymbol(D_BKOC_F_LOW_NL,&low_nl,sizeof(low_nl)),"bkoc f low nl ptr");ck(cudaMemcpyToSymbol(D_BKOC_F_HIGH_NN,&high_nn,sizeof(high_nn)),"bkoc f high nn ptr");ck(cudaMemcpyToSymbol(D_BKOC_F_HIGH_NRNL,&high_nrnl,sizeof(high_nrnl)),"bkoc f high nrnl ptr");}
    void release(){cudaFree(low_nn);cudaFree(low_nr);cudaFree(low_nl);cudaFree(high_nn);cudaFree(high_nrnl);low_nn=low_nr=low_nl=high_nn=high_nrnl=nullptr;}
};
struct BucketReverseOrbitClosureAttachDeviceTables{
    uint32_t *low=nullptr,*high=nullptr;
    template<class T>static void cp(T*&d,const std::vector<T>&s,const char*w){if(s.empty())return;ck(cudaMalloc(&d,s.size()*sizeof(T)),w);ck(cudaMemcpy(d,s.data(),s.size()*sizeof(T),cudaMemcpyHostToDevice),w);}
    void install(const BucketReverseOrbitClosureAttachHost&h){cp(low,h.low,"bkoc r low");cp(high,h.high,"bkoc r high");ck(cudaMemcpyToSymbol(D_BKOC_R_LOW,&low,sizeof(low)),"bkoc r low ptr");ck(cudaMemcpyToSymbol(D_BKOC_R_HIGH,&high,sizeof(high)),"bkoc r high ptr");}
    void release(){cudaFree(low);cudaFree(high);low=high=nullptr;}
};

__device__ __forceinline__ Count bkoc_f_low_extra(uint32_t rid,const BucketPhysicalBlock&db,uint32_t hr){
    if(rid==BKOC_NONE)return 0;BucketFusedDst rec=D_BKF_LOW_DST[rid];uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
#if GPU_DIRECT_PM_ACCUM
    uint64_t sum=0;
    for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){uint32_t x=D_BKF_LOW_LOCAL_SRC[e],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);BucketPhysicalBlock sb=bkf_low_main(ss,bkf_src_block(x));sum+=uint64_t(bkf_ptr(ss,sb.off+Code(hr)*sb.cols+bkf_loc_rank(sl))[0]);}
    if(cc){uint32_t dc=D_BKF_HIGH_CODES[D_BKF_HIGH_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.he]+hr];for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){uint32_t x=D_BKF_LOW_CROSS_OP[e],sl=bkf_src_locator(x);BucketPhysicalBlock sb=bkf_low_main(bkf_loc_owner(sl),bkf_src_block(x));sum+=bkf_sum_high_preimages_u64(dc,bkf_cross_depth(x),sb.he,bkf_src_block(x),sl);}}
    return gpu_direct_pm_reduce_u64(sum);
#else
    Count sum=0;for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){uint32_t x=D_BKF_LOW_LOCAL_SRC[e],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);BucketPhysicalBlock sb=bkf_low_main(ss,bkf_src_block(x));sum=gpu_direct_add(sum,bkf_ptr(ss,sb.off+Code(hr)*sb.cols+bkf_loc_rank(sl))[0]);}
    if(cc){uint32_t dc=D_BKF_HIGH_CODES[D_BKF_HIGH_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.he]+hr];for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){uint32_t x=D_BKF_LOW_CROSS_OP[e],sl=bkf_src_locator(x);BucketPhysicalBlock sb=bkf_low_main(bkf_loc_owner(sl),bkf_src_block(x));sum=gpu_direct_add(sum,bkf_sum_high_preimages(dc,bkf_cross_depth(x),sb.he,bkf_src_block(x),sl));}}return sum;
#endif
}
__device__ __forceinline__ Count bkoc_f_high_extra(uint32_t rid,const BucketPhysicalBlock&db,uint32_t lr){
    if(rid==BKOC_NONE)return 0;BucketFusedDst rec=D_BKF_HIGH_DST[rid];uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
#if GPU_DIRECT_PM_ACCUM
    uint64_t sum=0;for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){uint32_t x=D_BKF_HIGH_LOCAL_SRC[e],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);BucketPhysicalBlock sb=bkf_high_main(ss,bkf_src_block(x));sum+=uint64_t(bkf_ptr(ss,sb.off+Code(bkf_loc_rank(sl))*sb.cols+lr)[0]);}
    if(cc){uint32_t dc=D_BKF_LOW_CODES[D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.hs]+lr];for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){uint32_t x=D_BKF_HIGH_CROSS_OP[e],sl=bkf_src_locator(x);BucketPhysicalBlock sb=bkf_high_main(bkf_loc_owner(sl),bkf_src_block(x));sum+=bkf_sum_low_preimages_u64(dc,bkf_cross_depth(x),sb.hs,bkf_src_block(x),sl);}}return gpu_direct_pm_reduce_u64(sum);
#else
    Count sum=0;for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){uint32_t x=D_BKF_HIGH_LOCAL_SRC[e],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);BucketPhysicalBlock sb=bkf_high_main(ss,bkf_src_block(x));sum=gpu_direct_add(sum,bkf_ptr(ss,sb.off+Code(bkf_loc_rank(sl))*sb.cols+lr)[0]);}
    if(cc){uint32_t dc=D_BKF_LOW_CODES[D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.hs]+lr];for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){uint32_t x=D_BKF_HIGH_CROSS_OP[e],sl=bkf_src_locator(x);BucketPhysicalBlock sb=bkf_high_main(bkf_loc_owner(sl),bkf_src_block(x));sum=gpu_direct_add(sum,bkf_sum_low_preimages(dc,bkf_cross_depth(x),sb.hs,bkf_src_block(x),sl));}}return sum;
#endif
}
__device__ __forceinline__ Count bkoc_r_low_extra(uint32_t rid,const BucketPhysicalBlock&db,uint32_t hr){
    if(rid==BKOC_NONE)return 0;BucketFusedDst rec=D_RBF_LOW_DST[rid];uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
#if GPU_DIRECT_PM_ACCUM
    uint64_t sum=0;for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){uint32_t x=D_RBF_LOW_LOCAL_SRC[e],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);BucketPhysicalBlock sb=bkf_low_main(ss,bkf_src_block(x));sum+=uint64_t(bkf_ptr(ss,sb.off+Code(hr)*sb.cols+bkf_loc_rank(sl))[0]);}
    if(cc){uint32_t dc=D_BKF_HIGH_CODES[D_BKF_HIGH_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.he]+hr];for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){uint32_t x=D_RBF_LOW_CROSS_OP[e],sl=bkf_src_locator(x);BucketPhysicalBlock sb=bkf_low_main(bkf_loc_owner(sl),bkf_src_block(x));sum+=bkf_sum_high_preimages_u64(dc,bkf_cross_depth(x),sb.he,bkf_src_block(x),sl);}}return gpu_direct_pm_reduce_u64(sum);
#else
    Count sum=0;for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){uint32_t x=D_RBF_LOW_LOCAL_SRC[e],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);BucketPhysicalBlock sb=bkf_low_main(ss,bkf_src_block(x));sum=gpu_direct_add(sum,bkf_ptr(ss,sb.off+Code(hr)*sb.cols+bkf_loc_rank(sl))[0]);}
    if(cc){uint32_t dc=D_BKF_HIGH_CODES[D_BKF_HIGH_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.he]+hr];for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){uint32_t x=D_RBF_LOW_CROSS_OP[e],sl=bkf_src_locator(x);BucketPhysicalBlock sb=bkf_low_main(bkf_loc_owner(sl),bkf_src_block(x));sum=gpu_direct_add(sum,bkf_sum_high_preimages(dc,bkf_cross_depth(x),sb.he,bkf_src_block(x),sl));}}return sum;
#endif
}
__device__ __forceinline__ Count bkoc_r_high_extra(uint32_t rid,const BucketPhysicalBlock&db,uint32_t lr){
    if(rid==BKOC_NONE)return 0;BucketFusedDst rec=D_RBF_HIGH_DST[rid];uint32_t lc=rec.counts&0xffffu,cc=rec.counts>>16;
#if GPU_DIRECT_PM_ACCUM
    uint64_t sum=0;for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){uint32_t x=D_RBF_HIGH_LOCAL_SRC[e],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);BucketPhysicalBlock sb=bkf_high_main(ss,bkf_src_block(x));sum+=uint64_t(bkf_ptr(ss,sb.off+Code(bkf_loc_rank(sl))*sb.cols+lr)[0]);}
    if(cc){uint32_t dc=D_BKF_LOW_CODES[D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.hs]+lr];for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){uint32_t x=D_RBF_HIGH_CROSS_OP[e],sl=bkf_src_locator(x);BucketPhysicalBlock sb=bkf_high_main(bkf_loc_owner(sl),bkf_src_block(x));sum+=bkf_sum_low_preimages_u64(dc,bkf_cross_depth(x),sb.hs,bkf_src_block(x),sl);}}return gpu_direct_pm_reduce_u64(sum);
#else
    Count sum=0;for(uint32_t e=rec.local_begin;e<rec.local_begin+lc;++e){uint32_t x=D_RBF_HIGH_LOCAL_SRC[e],sl=bkf_src_locator(x),ss=bkf_loc_owner(sl);BucketPhysicalBlock sb=bkf_high_main(ss,bkf_src_block(x));sum=gpu_direct_add(sum,bkf_ptr(ss,sb.off+Code(bkf_loc_rank(sl))*sb.cols+lr)[0]);}
    if(cc){uint32_t dc=D_BKF_LOW_CODES[D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+db.hs]+lr];for(uint32_t e=rec.cross_begin;e<rec.cross_begin+cc;++e){uint32_t x=D_RBF_HIGH_CROSS_OP[e],sl=bkf_src_locator(x);BucketPhysicalBlock sb=bkf_high_main(bkf_loc_owner(sl),bkf_src_block(x));sum=gpu_direct_add(sum,bkf_sum_low_preimages(dc,bkf_cross_depth(x),sb.hs,bkf_src_block(x),sl));}}return sum;
#endif
}

__global__ void bucket_low_orbit_closure_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(LOW_LUT_K-p);size_t oi=size_t(pi)*D_BKF_LOW_PITCH+bid;
    uint32_t na=D_BKF_LOW_NN_OFF[oi],nb=D_BKF_LOW_NN_OFF[oi+1],ra=D_BKF_LOW_NR_OFF[oi],rb=D_BKF_LOW_NR_OFF[oi+1],la=D_BKF_LOW_NL_OFF[oi],lb=D_BKF_LOW_NL_OFF[oi+1];uint32_t n0=nb-na,n1=rb-ra,total=n0+n1+(lb-la);if(!total)return;
    for(uint32_t k=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;k<total;k+=uint32_t(gridDim.x)*blockDim.x){uint32_t kind,rid;BucketOrbitOp op;if(k<n0){kind=CPU_ORBIT_NN;op=D_BKF_LOW_NN[na+k];rid=D_BKOC_F_LOW_NN[na+k];}else if(k<n0+n1){kind=CPU_ORBIT_NR;op=D_BKF_LOW_NR[ra+k-n0];rid=D_BKOC_F_LOW_NR[ra+k-n0];}else{kind=CPU_ORBIT_NL;op=D_BKF_LOW_NL[la+k-n0-n1];rid=D_BKOC_F_LOW_NL[la+k-n0-n1];}
        uint32_t sl=bkf_orbit_src(op),jl=bkf_orbit_partner(op),dl=bkf_orbit_drop(op),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl);BucketPhysicalBlock xb=bkf_low_main(ss,bid);if(!xb.valid||!xb.rows||!xb.cols)continue;uint32_t jbid=bid;if(p==LOW_LUT_K){uint32_t center=kind==CPU_ORBIT_NR?uint32_t(R):uint32_t(::L);jbid=3u*uint32_t(xb.he)+center;}BucketPhysicalBlock jb=bkf_low_main(js,jbid),db=bkf_low_block(ds,uint32_t(xb.he));uint32_t sr=bkf_loc_rank(sl),jr=bkf_loc_rank(jl),dr=bkf_loc_rank(dl);
        for(uint32_t hr=blockIdx.y;hr<xb.rows;hr+=gridDim.y){Count*ip=bkf_ptr(ss,xb.off+Code(hr)*xb.cols+sr),*jp=bkf_ptr(js,jb.off+Code(hr)*jb.cols+jr),*dp=bkf_ptr(ds,db.off+Code(hr)*db.cols+dr);Count c=*ip,old=*dp,extra=bkoc_f_low_extra(rid,p==1?xb:db,hr);if(kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(gpu_direct_add(c,old),p==1?extra:0);*dp=p==1?0:extra;}else{Count cc=*jp,all=gpu_direct_add(gpu_direct_add(c,cc),old);if(p==1){*ip=all;*jp=gpu_direct_add(c,cc);*dp=0;}else{*ip=all;*dp=gpu_direct_add(c,extra);}}}
    }
}
__global__ void bucket_high_orbit_closure_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t((TARGET_W-1)-p);size_t oi=size_t(pi)*D_BKF_HIGH_PITCH+bid;uint32_t na=D_BKF_HIGH_NN_OFF[oi],nb=D_BKF_HIGH_NN_OFF[oi+1],ra=D_BKF_HIGH_NRNL_OFF[oi],rb=D_BKF_HIGH_NRNL_OFF[oi+1],n0=nb-na,total=n0+(rb-ra);if(!total)return;
    for(uint32_t k=blockIdx.y;k<total;k+=gridDim.y){bool nn=k<n0;uint32_t qi=nn?na+k:ra+k-n0;BucketOrbitOp op=nn?D_BKF_HIGH_NN[qi]:D_BKF_HIGH_NRNL[qi];uint32_t rid=nn?D_BKOC_F_HIGH_NN[qi]:D_BKOC_F_HIGH_NRNL[qi];uint32_t sl=bkf_orbit_src(op),jl=bkf_orbit_partner(op),dl=bkf_orbit_drop(op),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl);BucketPhysicalBlock xb=bkf_high_main(ss,bid);if(!xb.valid||!xb.rows||!xb.cols)continue;uint32_t jbid=bid;if(p==LOW_LUT_K+1){uint32_t center=nn?uint32_t(R):uint32_t(N);int he=int(xb.hs)+(center==uint32_t(R)?1:0);jbid=uint32_t(3*he+int(center));}BucketPhysicalBlock jb=bkf_high_main(js,jbid),db=bkf_high_block(ds,uint32_t(xb.hs));uint32_t sr=bkf_loc_rank(sl),jr=bkf_loc_rank(jl),dr=bkf_loc_rank(dl);
        for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<xb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){Count*ip=bkf_ptr(ss,xb.off+Code(sr)*xb.cols+lr),*jp=bkf_ptr(js,jb.off+Code(jr)*jb.cols+lr),*dp=bkf_ptr(ds,db.off+Code(dr)*db.cols+lr);Count c=*ip,old=*dp,extra=bkoc_f_high_extra(rid,db,lr);if(nn){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(c,old);*dp=extra;}else{Count cc=*jp;*ip=gpu_direct_add(gpu_direct_add(c,cc),old);*dp=gpu_direct_add(c,extra);}}
    }
}
__global__ void bucket_reverse_low_orbit_closure_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(p-1);size_t oi=size_t(pi)*D_RB_PITCH+bid;uint32_t a=D_RB_LOW_ORBIT_OFF[oi],b=D_RB_LOW_ORBIT_OFF[oi+1];
    for(uint32_t q=a+uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;q<b;q+=uint32_t(gridDim.x)*blockDim.x){uint64_t w=D_RB_LOW_ORBIT[q];uint32_t rid=D_BKOC_R_LOW[q],sl=rb_orbit_src(w),jl=rb_orbit_partner(w),dl=rb_orbit_drop(w),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl),kind=rb_orbit_kind(w);BucketPhysicalBlock xb=bkf_low_main(ss,bid),jb=bkf_low_main(js,rb_orbit_jblock(w)),db=bkf_low_block(ds,uint32_t(xb.he));
        for(uint32_t hr=blockIdx.y;hr<xb.rows;hr+=gridDim.y){Count*ip=bkf_ptr(ss,xb.off+Code(hr)*xb.cols+bkf_loc_rank(sl)),*jp=bkf_ptr(js,jb.off+Code(hr)*jb.cols+bkf_loc_rank(jl)),*dp=bkf_ptr(ds,db.off+Code(hr)*db.cols+bkf_loc_rank(dl));Count c=*ip,old=*dp,extra=bkoc_r_low_extra(rid,db,hr);if(kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(c,old);*dp=extra;}else{Count cc=*jp;*ip=gpu_direct_add(gpu_direct_add(c,cc),old);*dp=gpu_direct_add(c,extra);}}
    }
}
__global__ void bucket_reverse_high_orbit_closure_kernel(int p){
    uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(p-(LOW_LUT_K+1));size_t oi=size_t(pi)*D_RB_PITCH+bid;uint32_t a=D_RB_HIGH_ORBIT_OFF[oi],b=D_RB_HIGH_ORBIT_OFF[oi+1];bool edge=p==TARGET_W-1;
    for(uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y){uint64_t w=D_RB_HIGH_ORBIT[q];uint32_t rid=D_BKOC_R_HIGH[q],sl=rb_orbit_src(w),jl=rb_orbit_partner(w),dl=rb_orbit_drop(w),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl),kind=rb_orbit_kind(w);BucketPhysicalBlock xb=bkf_high_main(ss,bid),jb=bkf_high_main(js,rb_orbit_jblock(w)),db=bkf_high_block(ds,uint32_t(xb.hs));
        for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<xb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){Count*ip=bkf_ptr(ss,xb.off+Code(bkf_loc_rank(sl))*xb.cols+lr),*jp=bkf_ptr(js,jb.off+Code(bkf_loc_rank(jl))*jb.cols+lr),*dp=bkf_ptr(ds,db.off+Code(bkf_loc_rank(dl))*db.cols+lr);Count c=*ip,old=*dp,extra=bkoc_r_high_extra(rid,edge?xb:db,lr);if(kind==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(gpu_direct_add(c,old),edge?extra:0);*dp=edge?0:extra;}else{Count cc=*jp;*ip=gpu_direct_add(gpu_direct_add(c,cc),old);if(edge){*jp=gpu_direct_add(c,cc);*dp=0;}else *dp=gpu_direct_add(c,extra);}}
    }
}

static void bucket_launch_low_orbit_closure_fused(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=LOW_LUT_K;p>=1;--p){bucket_low_orbit_closure_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket low orbit closure");}}
static void bucket_launch_high_orbit_closure_fused(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=TARGET_W-1;p>=LOW_LUT_K+1;--p){bucket_high_orbit_closure_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket high orbit closure");}}
static void bucket_launch_reverse_low_orbit_closure_fused(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=1;p<=LOW_LUT_K;++p){bucket_reverse_low_orbit_closure_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket reverse low orbit closure");}}
static void bucket_launch_reverse_high_orbit_closure_fused(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=LOW_LUT_K+1;p<TARGET_W;++p){bucket_reverse_high_orbit_closure_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"bucket reverse high orbit closure");}}
