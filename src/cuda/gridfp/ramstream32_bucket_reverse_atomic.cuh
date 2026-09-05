#pragma once

#include "ramstream32_reverse_orbit.hpp"
#include "ramstream32_bucket_direct.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

// Correctness-first bucket backend for snake/reverse rows. Reverse orbit
// updates are conflict-free in-place. Closure updates use modular CAS atomics;
// destination-gather fusion is the next optimization after the snake schedule
// is validated on real multi-GPU hardware.

using ReverseBucketOrbitOp=uint64_t;
using ReverseBucketClosureOp=uint64_t;
static constexpr uint64_t RB_LOC_MASK=(1ull<<BUCKET_LOCATOR_BITS)-1ull;
static constexpr int RB_PARTNER_SHIFT=BUCKET_LOCATOR_BITS;
static constexpr int RB_DROP_SHIFT=2*BUCKET_LOCATOR_BITS;
static constexpr int RB_KIND_SHIFT=3*BUCKET_LOCATOR_BITS;
static constexpr int RB_JBLOCK_SHIFT=RB_KIND_SHIFT+2;
static constexpr uint64_t RB_JBLOCK_MASK=0x3full;
static_assert(RB_JBLOCK_SHIFT+6<=64);

static ReverseBucketOrbitOp rb_orbit_pack(uint32_t src,uint32_t partner,uint32_t drop,uint32_t kind,uint32_t jblock){
    if((uint64_t(src)&~RB_LOC_MASK)||(uint64_t(partner)&~RB_LOC_MASK)||(uint64_t(drop)&~RB_LOC_MASK)
       ||kind<CPU_ORBIT_NN||kind>CPU_ORBIT_NL||jblock>RB_JBLOCK_MASK){std::cerr<<"reverse bucket orbit pack overflow\n";std::exit(290);}
    return uint64_t(src)|(uint64_t(partner)<<RB_PARTNER_SHIFT)|(uint64_t(drop)<<RB_DROP_SHIFT)
        |(uint64_t(kind-CPU_ORBIT_NN)<<RB_KIND_SHIFT)|(uint64_t(jblock)<<RB_JBLOCK_SHIFT);
}
#if defined(__CUDACC__)
#define RB_HD __host__ __device__ __forceinline__
#else
#define RB_HD inline
#endif
RB_HD uint32_t rb_orbit_src(uint64_t x){return uint32_t(x&RB_LOC_MASK);}
RB_HD uint32_t rb_orbit_partner(uint64_t x){return uint32_t((x>>RB_PARTNER_SHIFT)&RB_LOC_MASK);}
RB_HD uint32_t rb_orbit_drop(uint64_t x){return uint32_t((x>>RB_DROP_SHIFT)&RB_LOC_MASK);}
RB_HD uint32_t rb_orbit_kind(uint64_t x){return uint32_t((x>>RB_KIND_SHIFT)&3u)+CPU_ORBIT_NN;}
RB_HD uint32_t rb_orbit_jblock(uint64_t x){return uint32_t((x>>RB_JBLOCK_SHIFT)&RB_JBLOCK_MASK);}
#undef RB_HD

static constexpr int RB_C_DST_SHIFT=BUCKET_LOCATOR_BITS;
static constexpr int RB_C_BLOCK_SHIFT=2*BUCKET_LOCATOR_BITS;
static constexpr int RB_C_DEPTH_SHIFT=RB_C_BLOCK_SHIFT+6;
static constexpr int RB_C_TARGET_BLOCK_SHIFT=RB_C_DEPTH_SHIFT+4;
static_assert(RB_C_TARGET_BLOCK_SHIFT+1<=64);
static ReverseBucketClosureOp rb_closure_pack(uint32_t src,uint32_t dst,uint32_t block,uint32_t depth,bool target_block){
    if((uint64_t(src)&~RB_LOC_MASK)||(uint64_t(dst)&~RB_LOC_MASK)||block>0x3fu||depth>0xfu)std::exit(291);
    return uint64_t(src)|(uint64_t(dst)<<RB_C_DST_SHIFT)|(uint64_t(block)<<RB_C_BLOCK_SHIFT)
        |(uint64_t(depth)<<RB_C_DEPTH_SHIFT)|(uint64_t(target_block)<<RB_C_TARGET_BLOCK_SHIFT);
}
#if defined(__CUDACC__)
#define RB_HD __host__ __device__ __forceinline__
#else
#define RB_HD inline
#endif
RB_HD uint32_t rb_closure_src(uint64_t x){return uint32_t(x&RB_LOC_MASK);}
RB_HD uint32_t rb_closure_dst(uint64_t x){return uint32_t((x>>RB_C_DST_SHIFT)&RB_LOC_MASK);}
RB_HD uint32_t rb_closure_block(uint64_t x){return uint32_t((x>>RB_C_BLOCK_SHIFT)&0x3fu);}
RB_HD uint32_t rb_closure_depth(uint64_t x){return uint32_t((x>>RB_C_DEPTH_SHIFT)&0xfu);}
RB_HD bool rb_closure_target_block(uint64_t x){return ((x>>RB_C_TARGET_BLOCK_SHIFT)&1u)!=0;}
#undef RB_HD

struct ReverseBucketAtomicHost{
    std::vector<ReverseBucketOrbitOp> low_orbit,high_orbit;
    std::vector<ReverseBucketClosureOp> low_closure,high_closure;
    std::vector<uint32_t> low_orbit_off,high_orbit_off,low_closure_off,high_closure_off;
    uint32_t nblocks=0;
    size_t bytes()const{return (low_orbit.size()+high_orbit.size()+low_closure.size()+high_closure.size())*sizeof(uint64_t)
        +(low_orbit_off.size()+high_orbit_off.size()+low_closure_off.size()+high_closure_off.size())*sizeof(uint32_t);}
};

static ReverseBucketAtomicHost build_reverse_bucket_atomic(
    const StorageFactorHost&storage,const StorageLayout&layout,const BucketOwnerHost&owner,
    const ReverseLowDescHost&ld,const ReverseHighDescHost&hd,const ReverseOrbitHost&lo,const ReverseOrbitHost&ho
){
    ReverseBucketAtomicHost out;out.nblocks=uint32_t(layout.main_blocks.size());size_t pitch=size_t(out.nblocks)+1;
    out.low_orbit_off.resize(size_t(LOW_LUT_K)*pitch);out.low_closure_off.resize(size_t(LOW_LUT_K)*pitch);
    out.high_orbit_off.resize(size_t(HIGH_LUT_K)*pitch);out.high_closure_off.resize(size_t(HIGH_LUT_K)*pitch);
    auto build_side=[&](bool low,const ReverseOrbitHost&o){
        int p0=low?1:LOW_LUT_K+1,p1=low?LOW_LUT_K:TARGET_W-1;
        auto&ov=low?out.low_orbit:out.high_orbit;auto&cv=low?out.low_closure:out.high_closure;
        auto&oo=low?out.low_orbit_off:out.high_orbit_off;auto&co=low?out.low_closure_off:out.high_closure_off;
        for(int p=p0;p<=p1;++p){uint32_t pi=uint32_t(p-p0);
            for(uint32_t bid=0;bid<out.nblocks;++bid){size_t oi=size_t(pi)*pitch+bid;oo[oi]=uint32_t(ov.size());co[oi]=uint32_t(cv.size());const StorageBlock&sb=layout.main_blocks[bid];if(!sb.valid)continue;uint32_t n=low?sb.cols:sb.rows;
                for(uint32_t ar=0;ar<n;++ar){uint64_t w=o.rec[size_t(pi)*o.main_total+o.main_base[bid]+ar];uint32_t k=cpu_orbit_kind(w);
                    if(k>=CPU_ORBIT_NN&&k<=CPU_ORBIT_NL){uint32_t jb=cpu_orbit_jblock(w),db=cpu_orbit_dblock(w);uint32_t sl,jl,dl;if(low){sl=bucket_low_locator(storage,owner,sb.hs,ar);jl=bucket_low_locator(storage,owner,layout.main_blocks[jb].hs,cpu_orbit_jlr(w));dl=bucket_low_locator(storage,owner,layout.block_blocks[db].hs,cpu_orbit_dlr(w));}else{sl=bucket_high_locator(storage,owner,sb.he,ar);jl=bucket_high_locator(storage,owner,layout.main_blocks[jb].he,cpu_orbit_jlr(w));dl=bucket_high_locator(storage,owner,layout.block_blocks[db].he,cpu_orbit_dlr(w));}ov.push_back(rb_orbit_pack(sl,jl,dl,k,jb));}
                    else if(k==CPU_ORBIT_CLOSURE){const ReverseDesc&x=low?ld.main_desc[size_t(pi)*ld.main_total+ld.main_base[bid]+ar]:hd.main_desc[size_t(pi)*hd.main_total+hd.main_base[bid]+ar];if(x.kind==REVDESC_INVALID)continue;bool tb=x.kind==REVDESC_BLOCK||x.kind==REVDESC_CROSS_BLOCK;uint32_t sl,dloc;if(low){sl=bucket_low_locator(storage,owner,sb.hs,ar);int dh=tb?layout.block_blocks[x.block].hs:layout.main_blocks[x.block].hs;dloc=bucket_low_locator(storage,owner,dh,x.rank);}else{sl=bucket_high_locator(storage,owner,sb.he,ar);int dh=tb?layout.block_blocks[x.block].he:layout.main_blocks[x.block].he;dloc=bucket_high_locator(storage,owner,dh,x.rank);}cv.push_back(rb_closure_pack(sl,dloc,x.block,x.depth,tb));}
                }
            }size_t end=size_t(pi)*pitch+out.nblocks;oo[end]=uint32_t(ov.size());co[end]=uint32_t(cv.size());
        }
    };
    build_side(true,lo);build_side(false,ho);
    std::cerr<<"reverse_bucket_atomic low_orbit="<<out.low_orbit.size()<<" low_closure="<<out.low_closure.size()
             <<" high_orbit="<<out.high_orbit.size()<<" high_closure="<<out.high_closure.size()
             <<" mib="<<double(out.bytes())/(1<<20)<<'\n';return out;
}

__constant__ ReverseBucketOrbitOp* D_RB_LOW_ORBIT;
__constant__ ReverseBucketOrbitOp* D_RB_HIGH_ORBIT;
__constant__ ReverseBucketClosureOp* D_RB_LOW_CLOSURE;
__constant__ ReverseBucketClosureOp* D_RB_HIGH_CLOSURE;
__constant__ uint32_t* D_RB_LOW_ORBIT_OFF;
__constant__ uint32_t* D_RB_HIGH_ORBIT_OFF;
__constant__ uint32_t* D_RB_LOW_CLOSURE_OFF;
__constant__ uint32_t* D_RB_HIGH_CLOSURE_OFF;
__constant__ uint32_t D_RB_PITCH;

__device__ __forceinline__ void rb_atomic_add(Count*p,Count v){
    if(!v)return;Count old=*p,assumed;
    do{assumed=old;Count nv=gpu_direct_add(assumed,v);old=atomicCAS(reinterpret_cast<unsigned int*>(p),assumed,nv);}while(old!=assumed);
}
__device__ __forceinline__ uint32_t rb_flip_high(uint32_t hc,uint32_t depth){int s=int(depth);for(int pos=0;pos<HIGH_LUT_K;++pos){uint32_t v=(hc>>(2*pos))&3u;if(v==uint32_t(::L)){if(--s==0){uint32_t z=3u<<(2*pos);return (hc&~z)|(uint32_t(R)<<(2*pos));}}else if(v==uint32_t(R))++s;}return 0xffffffffu;}
__device__ __forceinline__ uint32_t rb_flip_low(uint32_t lc,uint32_t depth){int s=int(depth);for(int pos=LOW_LUT_K-1;pos>=0;--pos){uint32_t v=(lc>>(2*pos))&3u;if(v==uint32_t(::L))++s;else if(v==uint32_t(R)){if(--s==0){uint32_t z=3u<<(2*pos);return (lc&~z)|(uint32_t(::L)<<(2*pos));}}}return 0xffffffffu;}

__global__ void bucket_reverse_low_orbit_kernel(int p){uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(p-1);size_t oi=size_t(pi)*D_RB_PITCH+bid;uint32_t a=D_RB_LOW_ORBIT_OFF[oi],b=D_RB_LOW_ORBIT_OFF[oi+1];
    for(uint32_t q=a+uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;q<b;q+=uint32_t(gridDim.x)*blockDim.x){uint64_t w=D_RB_LOW_ORBIT[q];uint32_t sl=rb_orbit_src(w),jl=rb_orbit_partner(w),dl=rb_orbit_drop(w),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl),k=rb_orbit_kind(w);BucketPhysicalBlock xb=bkf_low_main(ss,bid),jb=bkf_low_main(js,rb_orbit_jblock(w)),db=bkf_low_block(ds,uint32_t(xb.he));
        for(uint32_t hr=blockIdx.y;hr<xb.rows;hr+=gridDim.y){Count*ip=bkf_ptr(ss,xb.off+Code(hr)*xb.cols+bkf_loc_rank(sl)),*jp=bkf_ptr(js,jb.off+Code(hr)*jb.cols+bkf_loc_rank(jl)),*dp=bkf_ptr(ds,db.off+Code(hr)*db.cols+bkf_loc_rank(dl));Count c=*ip,old=*dp;if(k==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(c,old);*dp=0;}else{Count cc=*jp;*ip=gpu_direct_add(gpu_direct_add(c,cc),old);*dp=c;}}
    }}
__global__ void bucket_reverse_high_orbit_kernel(int p){uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(p-(LOW_LUT_K+1));size_t oi=size_t(pi)*D_RB_PITCH+bid;uint32_t a=D_RB_HIGH_ORBIT_OFF[oi],b=D_RB_HIGH_ORBIT_OFF[oi+1];bool edge=p==TARGET_W-1;
    for(uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y){uint64_t w=D_RB_HIGH_ORBIT[q];uint32_t sl=rb_orbit_src(w),jl=rb_orbit_partner(w),dl=rb_orbit_drop(w),ss=bkf_loc_owner(sl),js=bkf_loc_owner(jl),ds=bkf_loc_owner(dl),k=rb_orbit_kind(w);BucketPhysicalBlock xb=bkf_high_main(ss,bid),jb=bkf_high_main(js,rb_orbit_jblock(w)),db=bkf_high_block(ds,uint32_t(xb.hs));
        for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<xb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){Count*ip=bkf_ptr(ss,xb.off+Code(bkf_loc_rank(sl))*xb.cols+lr),*jp=bkf_ptr(js,jb.off+Code(bkf_loc_rank(jl))*jb.cols+lr),*dp=bkf_ptr(ds,db.off+Code(bkf_loc_rank(dl))*db.cols+lr);Count c=*ip,old=*dp;if(k==CPU_ORBIT_NN){*jp=gpu_direct_add(*jp,c);*ip=gpu_direct_add(c,old);*dp=0;}else{Count cc=*jp;*ip=gpu_direct_add(gpu_direct_add(c,cc),old);if(edge){*jp=gpu_direct_add(c,cc);*dp=0;}else *dp=c;}}
    }}

__global__ void bucket_reverse_low_closure_atomic_kernel(int p){uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(p-1);size_t oi=size_t(pi)*D_RB_PITCH+bid;uint32_t a=D_RB_LOW_CLOSURE_OFF[oi],b=D_RB_LOW_CLOSURE_OFF[oi+1];
    for(uint32_t q=a+uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;q<b;q+=uint32_t(gridDim.x)*blockDim.x){uint64_t w=D_RB_LOW_CLOSURE[q];uint32_t sl=rb_closure_src(w),dl=rb_closure_dst(w),ss=bkf_loc_owner(sl),ds=bkf_loc_owner(dl);BucketPhysicalBlock sb=bkf_low_main(ss,bid);bool tb=rb_closure_target_block(w);BucketPhysicalBlock db=tb?bkf_low_block(ds,rb_closure_block(w)):bkf_low_main(ds,rb_closure_block(w));uint32_t depth=rb_closure_depth(w);
        for(uint32_t hr=blockIdx.y;hr<sb.rows;hr+=gridDim.y){Count c=bkf_ptr(ss,sb.off+Code(hr)*sb.cols+bkf_loc_rank(sl))[0];if(!c)continue;uint32_t hr2=hr;if(depth){uint32_t hc=D_BKF_HIGH_CODES[D_BKF_HIGH_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+sb.he]+hr],hc2=rb_flip_high(hc,depth);uint32_t z=D_BKF_HIGH_DIRECT[gdx_ternary_key<HIGH_LUT_K>(hc2)];if(z==BKF_DIRECT_INVALID)continue;uint32_t hl=bkf_direct_locator(z);if(bkf_loc_owner(hl)!=D_BKF_FIXED_OWNER)continue;hr2=bkf_loc_rank(hl);}rb_atomic_add(bkf_ptr(ds,db.off+Code(hr2)*db.cols+bkf_loc_rank(dl)),c);}
    }}
__global__ void bucket_reverse_high_closure_atomic_kernel(int p){uint32_t bid=blockIdx.z;if(bid>=D_BKF_MAIN_NBLOCKS)return;uint32_t pi=uint32_t(p-(LOW_LUT_K+1));size_t oi=size_t(pi)*D_RB_PITCH+bid;uint32_t a=D_RB_HIGH_CLOSURE_OFF[oi],b=D_RB_HIGH_CLOSURE_OFF[oi+1];
    for(uint32_t q=a+blockIdx.y;q<b;q+=gridDim.y){uint64_t w=D_RB_HIGH_CLOSURE[q];uint32_t sl=rb_closure_src(w),dl=rb_closure_dst(w),ss=bkf_loc_owner(sl),ds=bkf_loc_owner(dl);BucketPhysicalBlock sb=bkf_high_main(ss,bid);bool tb=rb_closure_target_block(w);BucketPhysicalBlock db=tb?bkf_high_block(ds,rb_closure_block(w)):bkf_high_main(ds,rb_closure_block(w));uint32_t depth=rb_closure_depth(w);
        for(uint32_t lr=uint32_t(blockIdx.x)*blockDim.x+threadIdx.x;lr<sb.cols;lr+=uint32_t(gridDim.x)*blockDim.x){Count c=bkf_ptr(ss,sb.off+Code(bkf_loc_rank(sl))*sb.cols+lr)[0];if(!c)continue;uint32_t lr2=lr;if(depth){uint32_t lc=D_BKF_LOW_CODES[D_BKF_LOW_CODE_OFF[size_t(D_BKF_FIXED_OWNER)*D_BKF_CODE_PITCH+sb.hs]+lr],lc2=rb_flip_low(lc,depth);uint32_t z=D_BKF_LOW_DIRECT[gdx_ternary_key<LOW_LUT_K>(lc2)];if(z==BKF_DIRECT_INVALID)continue;uint32_t ll=bkf_direct_locator(z);if(bkf_loc_owner(ll)!=D_BKF_FIXED_OWNER)continue;lr2=bkf_loc_rank(ll);}rb_atomic_add(bkf_ptr(ds,db.off+Code(bkf_loc_rank(dl))*db.cols+lr2),c);}
    }}

struct ReverseBucketAtomicDeviceTables{
    ReverseBucketOrbitOp *lo=nullptr,*ho=nullptr;ReverseBucketClosureOp *lc=nullptr,*hc=nullptr;uint32_t *loo=nullptr,*hoo=nullptr,*lco=nullptr,*hco=nullptr;uint32_t pitch=0;
    template<class T>static void cp(T*&d,const std::vector<T>&v,const char*w){if(v.empty())return;ck(cudaMalloc(&d,v.size()*sizeof(T)),w);ck(cudaMemcpy(d,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice),w);}
    void install(const ReverseBucketAtomicHost&h){pitch=h.nblocks+1;cp(lo,h.low_orbit,"rb low orbit");cp(ho,h.high_orbit,"rb high orbit");cp(lc,h.low_closure,"rb low closure");cp(hc,h.high_closure,"rb high closure");cp(loo,h.low_orbit_off,"rb low orbit off");cp(hoo,h.high_orbit_off,"rb high orbit off");cp(lco,h.low_closure_off,"rb low closure off");cp(hco,h.high_closure_off,"rb high closure off");ck(cudaMemcpyToSymbol(D_RB_LOW_ORBIT,&lo,sizeof(lo)),"rb lo ptr");ck(cudaMemcpyToSymbol(D_RB_HIGH_ORBIT,&ho,sizeof(ho)),"rb ho ptr");ck(cudaMemcpyToSymbol(D_RB_LOW_CLOSURE,&lc,sizeof(lc)),"rb lc ptr");ck(cudaMemcpyToSymbol(D_RB_HIGH_CLOSURE,&hc,sizeof(hc)),"rb hc ptr");ck(cudaMemcpyToSymbol(D_RB_LOW_ORBIT_OFF,&loo,sizeof(loo)),"rb loo ptr");ck(cudaMemcpyToSymbol(D_RB_HIGH_ORBIT_OFF,&hoo,sizeof(hoo)),"rb hoo ptr");ck(cudaMemcpyToSymbol(D_RB_LOW_CLOSURE_OFF,&lco,sizeof(lco)),"rb lco ptr");ck(cudaMemcpyToSymbol(D_RB_HIGH_CLOSURE_OFF,&hco,sizeof(hco)),"rb hco ptr");ck(cudaMemcpyToSymbol(D_RB_PITCH,&pitch,sizeof(pitch)),"rb pitch");}
    void release(){auto f=[](auto*&p){if(p)cudaFree(p);p=nullptr;};f(lo);f(ho);f(lc);f(hc);f(loo);f(hoo);f(lco);f(hco);}
};

static void bucket_launch_reverse_low_atomic(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=1;p<=LOW_LUT_K;++p){bucket_reverse_low_orbit_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"rb low orbit");bucket_reverse_low_closure_atomic_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"rb low closure");}}
static void bucket_launch_reverse_high_atomic(const StorageLayout&layout,int threads=256,int gx=16,int gy=8){dim3 block(threads),grid(gx,gy,unsigned(layout.main_blocks.size()));for(int p=LOW_LUT_K+1;p<TARGET_W;++p){bucket_reverse_high_orbit_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"rb high orbit");bucket_reverse_high_closure_atomic_kernel<<<grid,block>>>(p);ck(cudaGetLastError(),"rb high closure");}}
