#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

using Count = uint32_t;
using MateID = unsigned long long;
using Code = unsigned long long;
static constexpr int MAXW=28, MAXGPU=8;
#ifndef TARGET_W
#define TARGET_W 28
#endif

enum MateValue:uint8_t{N=0,R=1,L=2,X=3};
enum MateValuePair:uint8_t{NN=0x0,NR=0x1,NL=0x2,NX=0x3,RN=0x4,RR=0x5,RL=0x6,RX=0x7,LN=0x8,LR=0x9,LL=0xa,LX=0xb,XN=0xc,XR=0xd,XL=0xe,XX=0xf};

static Code H_DP[MAXW+1][MAXW+2];
__constant__ Code D_FULL_DP[MAXW+1][MAXW+2];
__constant__ Code D_MAIN_DP[MAXW+1][MAXW+2];
__constant__ Code D_BLOCK_DP[MAXW+1][MAXW+2];
__constant__ uint32_t D_MAIN_FIXED,D_MAIN_OCC,D_BLOCK_FIXED,D_BLOCK_OCC;
__constant__ int D_MAIN_W,D_BLOCK_W,D_NGPU;
__constant__ Count D_MOD;
__constant__ Count* D_MAIN_PTR[MAXGPU];
__constant__ Count* D_BLOCK_PTR[MAXGPU];
__constant__ Code D_MAIN_CHUNK,D_BLOCK_CHUNK;

__host__ __device__ static inline MateValue mget(MateID m,int k){return MateValue((m>>(2*k))&3ULL);}
__host__ __device__ static inline MateValuePair mpair(MateID m,int p){return MateValuePair((m>>(2*(p-1)))&15ULL);}
__host__ __device__ static inline MateID mset(MateID m,int k,MateValue v){MateID z=3ULL<<(2*k);return (m&~z)|(MateID(v)<<(2*k));}
__host__ __device__ static inline MateID msetpair(MateID m,int p,MateValuePair v){MateID z=15ULL<<(2*(p-1));return (m&~z)|(MateID(v)<<(2*(p-1)));}
__host__ __device__ static inline MateID mshrink(MateID m,int k){MateID mask=(1ULL<<(2*k))-1ULL;return((m&~mask)>>2)|(m&mask);}
__host__ __device__ static inline MateID minsert(MateID m,int k,MateValue v){MateID lowmask=k?((1ULL<<(2*k))-1ULL):0ULL;MateID lo=m&lowmask,hi=m&~lowmask;return lo|(MateID(v)<<(2*k))|(hi<<2);}

static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){std::cerr<<w<<": "<<cudaGetErrorString(e)<<"\n";std::exit(1);}}
static void build_full_dp(){for(int h=0;h<=MAXW+1;++h)H_DP[0][h]=(h==0);for(int w=1;w<=MAXW;++w)for(int h=0;h<=MAXW;++h){Code x=H_DP[w-1][h];if(h>0)x+=H_DP[w-1][h-1];if(h<MAXW+1)x+=H_DP[w-1][h+1];H_DP[w][h]=x;}}

struct GroupSpec{int width=0;uint32_t fixed=0,occ=0;Code dp[MAXW+1][MAXW+2]{};Code size=0;};
static GroupSpec make_spec(int width,uint32_t fixed,uint32_t occ){GroupSpec s;s.width=width;s.fixed=fixed;s.occ=occ;for(int h=0;h<=MAXW+1;++h)s.dp[0][h]=(h==0);for(int w=1;w<=width;++w){int pos=w-1;bool f=(fixed>>pos)&1u,o=(occ>>pos)&1u;for(int h=0;h<=MAXW;++h){Code x=0;if(!f||!o)x+=s.dp[w-1][h];if(!f||o){if(h>0)x+=s.dp[w-1][h-1];if(h<MAXW+1)x+=s.dp[w-1][h+1];}s.dp[w][h]=x;}}s.size=s.dp[width][1];return s;}
static std::vector<int> window_candidates(int W,int hi,int lo){std::vector<int>v;for(int q=W-1;q>=0;--q)if(q<lo-1||q>hi)v.push_back(q);return v;}
static void window_masks(int W,int hi,int lo,const std::vector<int>&fp,uint32_t group,uint32_t&mf,uint32_t&mo,uint32_t&bf,uint32_t&bo){mf=mo=bf=bo=0;for(size_t i=0;i<fp.size();++i){int q=fp[i];bool one=(group>>i)&1u;mf|=1u<<q;if(one)mo|=1u<<q;int bq=(q<lo-1)?q:q-1;bf|=1u<<bq;if(one)bo|=1u<<bq;}}
struct WindowPlan{int p_hi=0,p_lo=0;std::vector<int>fixed_pos;size_t max_bytes=0;Code max_main=0,max_block=0;};
static WindowPlan plan_window(int W,int hi,int lo,size_t target,int maxbits=20){WindowPlan best;best.p_hi=hi;best.p_lo=lo;auto cand=window_candidates(W,hi,lo);int klim=std::min<int>(cand.size(),maxbits);for(int k=0;k<=klim;++k){std::vector<int>fp(cand.begin(),cand.begin()+k);uint64_t ng=1ull<<k;size_t mx=0;Code mm=0,md=0;for(uint64_t g=0;g<ng;++g){uint32_t mf,mo,bf,bo;window_masks(W,hi,lo,fp,(uint32_t)g,mf,mo,bf,bo);auto ms=make_spec(W,mf,mo);auto ds=make_spec(W-1,bf,bo);size_t b=size_t(2*ms.size+ds.size)*sizeof(Count);if(b>mx){mx=b;mm=ms.size;md=ds.size;}if(mx>target&&k<klim)break;}if(mx<=target||k==klim){best.fixed_pos=std::move(fp);best.max_bytes=mx;best.max_main=mm;best.max_block=md;return best;}}return best;}

__device__ __forceinline__ bool allowed(uint32_t fixed,uint32_t occ,int pos,MateValue v){if(!((fixed>>pos)&1u))return v!=X;bool o=(occ>>pos)&1u;return o?(v==R||v==L):(v==N);}
__device__ __forceinline__ MateID unrank_group(Code rank,int width,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){MateID m=0;int h=1;
#pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){if(pos>=width)continue;if(allowed(fixed,occ,pos,N)){Code z=dp[pos][h];if(rank<z)continue;rank-=z;}if(h>0&&allowed(fixed,occ,pos,R)){Code z=dp[pos][h-1];if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}m|=MateID(L)<<(2*pos);++h;}return m;}
__device__ __forceinline__ Code rank_group(MateID m,int width,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){Code rank=0;int h=1;
#pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){if(pos>=width)continue;MateValue s=mget(m,pos);if(s>N&&allowed(fixed,occ,pos,N))rank+=dp[pos][h];if(s>R&&h>0&&allowed(fixed,occ,pos,R))rank+=dp[pos][h-1];if(s==R)--h;else if(s==L)++h;}return rank;}
__device__ __forceinline__ Code rank_full_dev(MateID m,int width){Code rank=0;int h=1;
#pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){if(pos>=width)continue;MateValue s=mget(m,pos);if(s>N)rank+=D_FULL_DP[pos][h];if(s>R&&h>0)rank+=D_FULL_DP[pos][h-1];if(s==R)--h;else if(s==L)++h;}return rank;}

template<int WIDTH>
__device__ __forceinline__ MateID unrank_group_t(Code rank,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    MateID m=0;int h=1;
#pragma unroll
    for(int pos=WIDTH-1;pos>=0;--pos){
        if(allowed(fixed,occ,pos,N)){Code z=dp[pos][h];if(rank<z)continue;rank-=z;}
        if(h>0&&allowed(fixed,occ,pos,R)){Code z=dp[pos][h-1];if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;}rank-=z;}
        m|=MateID(L)<<(2*pos);++h;
    }
    return m;
}
template<int WIDTH>
__device__ __forceinline__ Code rank_group_t(MateID m,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    Code rank=0;int h=1;
#pragma unroll
    for(int pos=WIDTH-1;pos>=0;--pos){MateValue v=mget(m,pos);if(v>N&&allowed(fixed,occ,pos,N))rank+=dp[pos][h];if(v>R&&h>0&&allowed(fixed,occ,pos,R))rank+=dp[pos][h-1];if(v==R)--h;else if(v==L)++h;}
    return rank;
}
template<int WIDTH>
__device__ __forceinline__ Code rank_full_t(MateID m){
    Code rank=0;int h=1;
#pragma unroll
    for(int pos=WIDTH-1;pos>=0;--pos){MateValue v=mget(m,pos);if(v>N)rank+=D_FULL_DP[pos][h];if(v>R&&h>0)rank+=D_FULL_DP[pos][h-1];if(v==R)--h;else if(v==L)++h;}
    return rank;
}
__device__ __forceinline__ Count global_load_main(Code g){int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_MAIN_PTR[o][g-Code(o)*D_MAIN_CHUNK];}
__device__ __forceinline__ Count global_load_block(Code g){int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;return D_BLOCK_PTR[o][g-Code(o)*D_BLOCK_CHUNK];}
__device__ __forceinline__ void global_store_main(Code g,Count v){int o=int(g/D_MAIN_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;D_MAIN_PTR[o][g-Code(o)*D_MAIN_CHUNK]=v;}
__device__ __forceinline__ void global_store_block(Code g,Count v){int o=int(g/D_BLOCK_CHUNK);if(o>=D_NGPU)o=D_NGPU-1;D_BLOCK_PTR[o][g-Code(o)*D_BLOCK_CHUNK]=v;}

__global__ void gather_main_kernel(Count*out,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){MateID m=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);out[i]=global_load_main(rank_full_t<TARGET_W>(m));}}
__global__ void gather_block_kernel(Count*out,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){MateID m=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);out[i]=global_load_block(rank_full_t<TARGET_W-1>(m));}}
__global__ void scatter_main_kernel(const Count*in,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){MateID m=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);global_store_main(rank_full_t<TARGET_W>(m),in[i]);}}
__global__ void scatter_block_kernel(const Count*in,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){MateID m=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);global_store_block(rank_full_t<TARGET_W-1>(m),in[i]);}}

__device__ __forceinline__ void atomic_add_mod(Count*p,Count v){if(!v)return;Count mod=D_MOD;Count old=atomicCAS(p,0u,0u);for(;;){Count neu=(old>=mod-v)?old-(mod-v):old+v;Count seen=atomicCAS(p,old,neu);if(seen==old)return;old=seen;}}
__global__ void blocked_group_kernel(const Count*in,Code n,Count*out_main,int p){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Count c=in[i];if(!c)continue;MateID sm=unrank_group_t<TARGET_W-1>(i,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);MateID t=minsert(sm,p,N);Code j=rank_group_t<TARGET_W>(t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);}}
__global__ void main_group_kernel(const Count*in,Code n,Count*out_main,Count*out_block,int p){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride){Count c=in[i];if(!c)continue;MateID m=unrank_group_t<TARGET_W>(i,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);MateValuePair w=mpair(m,p);switch(w){case NN:{MateID t=msetpair(m,p,LR);atomic_add_mod(out_main+rank_group_t<TARGET_W>(t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP),c);break;}case NR:case NL:{if(p==1){MateID t=msetpair(m,p,w==NR?RN:LN);atomic_add_mod(out_main+rank_group_t<TARGET_W>(t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP),c);}else{MateID t=mshrink(m,p);atomic_add_mod(out_block+rank_group_t<TARGET_W-1>(t,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP),c);}break;}case RN:{MateID t=msetpair(m,p,NR);atomic_add_mod(out_main+rank_group_t<TARGET_W>(t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP),c);break;}case LN:{MateID t=msetpair(m,p,NL);atomic_add_mod(out_main+rank_group_t<TARGET_W>(t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP),c);break;}case LL:{MateID t=msetpair(m,p,NN);int q=p-1,s=1;while(s){--q;auto v=mget(t,q);if(v==L)++s;else if(v==R)--s;}t=mset(t,q,L);if(p==1)atomic_add_mod(out_main+rank_group_t<TARGET_W>(t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP),c);else{t=mshrink(t,p-1);atomic_add_mod(out_block+rank_group_t<TARGET_W-1>(t,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP),c);}break;}case RR:{MateID t=msetpair(m,p,NN);int q=p,s=1;while(s){++q;auto v=mget(t,q);if(v==L)--s;else if(v==R)++s;}t=mset(t,q,R);if(p==1)atomic_add_mod(out_main+rank_group_t<TARGET_W>(t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP),c);else{t=mshrink(t,p-1);atomic_add_mod(out_block+rank_group_t<TARGET_W-1>(t,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP),c);}break;}case RL:{MateID t=msetpair(m,p,NN);if(p==1)atomic_add_mod(out_main+rank_group_t<TARGET_W>(t,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP),c);else{t=mshrink(t,p-1);atomic_add_mod(out_block+rank_group_t<TARGET_W-1>(t,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP),c);}break;}default:break;}}}

static Code rank_full(MateID m,int width){Code r=0;int h=1;for(int pos=width-1;pos>=0;--pos){auto s=mget(m,pos);if(s>N)r+=H_DP[pos][h];if(s>R&&h>0)r+=H_DP[pos][h-1];if(s==R)--h;else if(s==L)++h;}return r;}

struct DeviceCtx{int dev=-1;Count*dA=nullptr,*dB=nullptr,*dD=nullptr;Code capM=0,capD=0;double active=0;uint64_t groups=0;void init(int d,Count mod,Count**mp,Count**bp,Code mc,Code bc,int ng){dev=d;ck(cudaSetDevice(dev),"set init");ck(cudaMemcpyToSymbol(D_FULL_DP,H_DP,sizeof(H_DP)),"full dp");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"mod");ck(cudaMemcpyToSymbol(D_MAIN_PTR,mp,sizeof(Count*)*MAXGPU),"main ptrs");ck(cudaMemcpyToSymbol(D_BLOCK_PTR,bp,sizeof(Count*)*MAXGPU),"block ptrs");ck(cudaMemcpyToSymbol(D_MAIN_CHUNK,&mc,sizeof(mc)),"main chunk");ck(cudaMemcpyToSymbol(D_BLOCK_CHUNK,&bc,sizeof(bc)),"block chunk");ck(cudaMemcpyToSymbol(D_NGPU,&ng,sizeof(ng)),"ngpu");}void ensure(Code m,Code b){ck(cudaSetDevice(dev),"set ensure");if(m>capM){if(dA){cudaFree(dA);cudaFree(dB);}capM=m;ck(cudaMalloc(&dA,size_t(m)*sizeof(Count)),"scratch A");ck(cudaMalloc(&dB,size_t(m)*sizeof(Count)),"scratch B");}if(b>capD){if(dD)cudaFree(dD);capD=b;ck(cudaMalloc(&dD,size_t(b)*sizeof(Count)),"scratch D");}}void destroy(){if(dev<0)return;cudaSetDevice(dev);if(dA){cudaFree(dA);cudaFree(dB);}if(dD)cudaFree(dD);}};

static void process_group(DeviceCtx&c,int W,const WindowPlan&wp,int g,int threads){auto t0=std::chrono::steady_clock::now();ck(cudaSetDevice(c.dev),"set worker");uint32_t mf,mo,bf,bo;window_masks(W,wp.p_hi,wp.p_lo,wp.fixed_pos,(uint32_t)g,mf,mo,bf,bo);auto ms=make_spec(W,mf,mo);auto ds=make_spec(W-1,bf,bo);if(!ms.size&&!ds.size)return;c.ensure(ms.size,ds.size);ck(cudaMemcpyToSymbol(D_MAIN_DP,ms.dp,sizeof(ms.dp)),"main dp");ck(cudaMemcpyToSymbol(D_BLOCK_DP,ds.dp,sizeof(ds.dp)),"block dp");ck(cudaMemcpyToSymbol(D_MAIN_FIXED,&mf,sizeof(mf)),"mf");ck(cudaMemcpyToSymbol(D_MAIN_OCC,&mo,sizeof(mo)),"mo");ck(cudaMemcpyToSymbol(D_BLOCK_FIXED,&bf,sizeof(bf)),"bf");ck(cudaMemcpyToSymbol(D_BLOCK_OCC,&bo,sizeof(bo)),"bo");int mw=W,bw=W-1;ck(cudaMemcpyToSymbol(D_MAIN_W,&mw,sizeof(mw)),"mw");ck(cudaMemcpyToSymbol(D_BLOCK_W,&bw,sizeof(bw)),"bw");int bm=int(std::min<Code>(65535,(ms.size+threads-1)/threads)),bd=int(std::min<Code>(65535,(ds.size+threads-1)/threads));if(ms.size)gather_main_kernel<<<bm,threads>>>(c.dA,ms.size);if(ds.size)gather_block_kernel<<<bd,threads>>>(c.dD,ds.size);Count*cur=c.dA,*nxt=c.dB;for(int p=wp.p_hi;p>=wp.p_lo;--p){if(ms.size)ck(cudaMemcpy(nxt,cur,size_t(ms.size)*sizeof(Count),cudaMemcpyDeviceToDevice),"identity");if(ds.size)blocked_group_kernel<<<bd,threads>>>(c.dD,ds.size,nxt,p);if(ds.size)ck(cudaMemset(c.dD,0,size_t(ds.size)*sizeof(Count)),"clear D");if(ms.size)main_group_kernel<<<bm,threads>>>(cur,ms.size,nxt,c.dD,p);ck(cudaGetLastError(),"transition");std::swap(cur,nxt);}if(ms.size)scatter_main_kernel<<<bm,threads>>>(cur,ms.size);if(ds.size)scatter_block_kernel<<<bd,threads>>>(c.dD,ds.size);ck(cudaGetLastError(),"scatter");ck(cudaDeviceSynchronize(),"group sync");c.groups++;c.active+=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();}

int main(int argc,char**argv){int n=argc>1?std::atoi(argv[1]):16;Count mod=argc>2?(Count)std::strtoul(argv[2],nullptr,10):2147483647u;int target_mib=argc>3?std::atoi(argv[3]):16384;int max_window=argc>4?std::atoi(argv[4]):14;int requested=argc>5?std::atoi(argv[5]):0;int W=n+1;if(n<2||W>MAXW){std::cerr<<"n=2..27\n";return 1;}if(W!=TARGET_W){std::cerr<<"specialized for width "<<TARGET_W<<" (n="<<(TARGET_W-1)<<")\n";return 1;}build_full_dp();int visible=0;ck(cudaGetDeviceCount(&visible),"count");int ng=requested<=0?visible:std::min(requested,visible);if(ng<1||ng>MAXGPU){std::cerr<<"need 1..8 GPUs\n";return 2;}
    int peers=0;for(int a=0;a<ng;++a)for(int b=0;b<ng;++b)if(a!=b){int can=0;ck(cudaDeviceCanAccessPeer(&can,a,b),"can peer");if(can){cudaSetDevice(a);auto e=cudaDeviceEnablePeerAccess(b,0);if(e==cudaErrorPeerAccessAlreadyEnabled)cudaGetLastError();else ck(e,"enable peer");peers++;}}
    if(ng>1&&peers!=ng*(ng-1)){std::cerr<<"HBM mode requires full P2P: "<<peers<<"/"<<ng*(ng-1)<<"\n";return 3;}
    Code mainN=H_DP[W][1],blockN=H_DP[W-1][1];Code mc=(mainN+ng-1)/ng,bc=(blockN+ng-1)/ng;Count*mp[MAXGPU]{},*bp[MAXGPU]{};std::vector<Code>ml(ng),bl(ng);
    for(int d=0;d<ng;++d){ml[d]=std::min<Code>(mc,mainN-std::min<Code>(mainN,Code(d)*mc));bl[d]=std::min<Code>(bc,blockN-std::min<Code>(blockN,Code(d)*bc));cudaSetDevice(d);if(ml[d]){ck(cudaMalloc(&mp[d],size_t(ml[d])*sizeof(Count)),"auth main");ck(cudaMemset(mp[d],0,size_t(ml[d])*sizeof(Count)),"zero main");}if(bl[d]){ck(cudaMalloc(&bp[d],size_t(bl[d])*sizeof(Count)),"auth block");ck(cudaMemset(bp[d],0,size_t(bl[d])*sizeof(Count)),"zero block");}}
    std::vector<DeviceCtx>ctx(ng);for(int d=0;d<ng;++d)ctx[d].init(d,mod,mp,bp,mc,bc,ng);
    MateID init=MateID(R)<<(2*(W-1));Code ig=rank_full(init,W);int io=int(ig/mc);Count one=1;cudaSetDevice(io);ck(cudaMemcpy(mp[io]+(ig-Code(io)*mc),&one,sizeof(one),cudaMemcpyHostToDevice),"init one");
    size_t target=size_t(target_mib)<<20;int threads=256,total_windows=0,maxgroups=0;auto wall0=std::chrono::steady_clock::now();
    for(int row=0;row<W;++row){int hi=W-1;while(hi>=1){WindowPlan wp;bool found=false;for(int lo=std::max(1,hi-max_window+1);lo<=hi;++lo){auto t=plan_window(W,hi,lo,target);if(t.max_bytes&&t.max_bytes<=target){wp=std::move(t);found=true;break;}}if(!found){std::cerr<<"cannot fit window hi="<<hi<<"\n";return 4;}int k=wp.fixed_pos.size();int nj=1<<k;maxgroups=std::max(maxgroups,nj);total_windows++;struct J{int g;Code w;};std::vector<J>jobs;jobs.reserve(nj);for(int g=0;g<nj;++g){uint32_t mf,mo,bf,bo;window_masks(W,wp.p_hi,wp.p_lo,wp.fixed_pos,g,mf,mo,bf,bo);auto ms=make_spec(W,mf,mo);auto ds=make_spec(W-1,bf,bo);jobs.push_back({g,2*ms.size+ds.size});}std::sort(jobs.begin(),jobs.end(),[](auto&a,auto&b){return a.w>b.w;});std::atomic<int>next{0};std::vector<std::thread>ths;for(int d=0;d<ng;++d)ths.emplace_back([&,d]{for(;;){int q=next.fetch_add(1);if(q>=nj)break;process_group(ctx[d],W,wp,jobs[q].g,threads);}});for(auto&t:ths)t.join();hi=wp.p_lo-1;}std::cerr<<"row "<<row+1<<"/"<<W<<" windows="<<total_windows<<"\n";}
    double wall=std::chrono::duration<double>(std::chrono::steady_clock::now()-wall0).count();Code fg=rank_full(MateID(R),W);int fo=int(fg/mc);Count ans=0;cudaSetDevice(fo);ck(cudaMemcpy(&ans,mp[fo]+(fg-Code(fo)*mc),sizeof(ans),cudaMemcpyDeviceToHost),"answer");double mx=0,sum=0;for(auto&c:ctx){mx=std::max(mx,c.active);sum+=c.active;std::cerr<<"gpu "<<c.dev<<" groups="<<c.groups<<" active_s="<<c.active<<"\n";}std::cout<<"backend=gridfp-b300-hbm32 n="<<n<<" residue="<<ans<<" modulus="<<mod<<" gpus="<<ng<<" peers="<<peers<<" main_states="<<mainN<<" blocked_states="<<blockN<<" auth_gib="<<double(mainN+blockN)*sizeof(Count)/(1ull<<30)<<" auth_per_gpu_gib="<<double(mainN+blockN)*sizeof(Count)/ng/(1ull<<30)<<" scratch_target_mib="<<target_mib<<" windows="<<total_windows<<" max_groups="<<maxgroups<<" active_max_s="<<mx<<" active_sum_s="<<sum<<" wall_s="<<wall<<"\n";
    for(auto&c:ctx)c.destroy();for(int d=0;d<ng;++d){cudaSetDevice(d);if(mp[d])cudaFree(mp[d]);if(bp[d])cudaFree(bp[d]);}
}
