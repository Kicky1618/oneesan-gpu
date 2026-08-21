#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>

using Count = unsigned long long;
using MateID = unsigned long long;
using Code = unsigned long long;
static constexpr int MAXW = 28;

enum MateValue : uint8_t { N=0, R=1, L=2, X=3 };
enum MateValuePair : uint8_t {
    NN=0x0, NR=0x1, NL=0x2, NX=0x3,
    RN=0x4, RR=0x5, RL=0x6, RX=0x7,
    LN=0x8, LR=0x9, LL=0xa, LX=0xb,
    XN=0xc, XR=0xd, XL=0xe, XX=0xf
};

__constant__ unsigned long long D_DP[MAXW+1][MAXW+1];
__constant__ Count D_MOD;

__host__ __device__ static inline MateValue mget(MateID m,int k){return MateValue((m>>(2*k))&3ULL);}
__host__ __device__ static inline MateValuePair mpair(MateID m,int p){return MateValuePair((m>>(2*(p-1)))&15ULL);}
__host__ __device__ static inline MateID mset(MateID m,int k,MateValue v){MateID z=3ULL<<(2*k);return (m&~z)|(MateID(v)<<(2*k));}
__host__ __device__ static inline MateID msetpair(MateID m,int p,MateValuePair v){MateID z=15ULL<<(2*(p-1));return (m&~z)|(MateID(v)<<(2*(p-1)));}
// Exact equivalent of GGCount Mate::shrink(k): collapse the slot just
// below k.  Note that this is deliberately NOT a normal remove(k).
__host__ __device__ static inline MateID mremove(MateID m,int k){
    MateID mask = (1ULL << (2*k)) - 1ULL;
    MateID hi = m & ~mask;
    MateID lo = m & mask;
    return (hi >> 2) | lo;
}
__host__ __device__ static inline MateID minsert(MateID m,int k,MateValue v){
    MateID lowmask = k?((1ULL<<(2*k))-1ULL):0ULL;
    MateID lo=m&lowmask, hi=m&~lowmask;
    return lo | (MateID(v)<<(2*k)) | (hi<<2);
}

__device__ __forceinline__ Code rank_mate(MateID m,int width){
    Code rank=0; int h=1;
    #pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){
        if(pos>=width) continue;
        MateValue s=mget(m,pos);
        if(s>N) rank+=D_DP[pos][h];
        if(s>R && h>0) rank+=D_DP[pos][h-1];
        if(s==R)--h; else if(s==L)++h;
    }
    return rank;
}

__device__ __forceinline__ MateID unrank_mate(Code rank,int width){
    MateID m=0; int h=1;
    #pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){
        if(pos>=width) continue;
        Code z=D_DP[pos][h];
        if(rank<z){ /* N */ }
        else {
            rank-=z;
            Code r=(h>0)?D_DP[pos][h-1]:0;
            if(rank<r){ m|=MateID(R)<<(2*pos); --h; }
            else { rank-=r; m|=MateID(L)<<(2*pos); ++h; }
        }
    }
    return m;
}

__device__ __forceinline__ void atomic_add_mod(Count* p,Count v){
    if(v==0) return;
    Count mod=D_MOD;
    Count old=atomicAdd(p,0ULL);
    for(;;){
        Count neu=(old>=mod-v)?(old-(mod-v)):(old+v);
        Count seen=atomicCAS(p,old,neu);
        if(seen==old) return;
        old=seen;
    }
}

__global__ void blocked_excluded_kernel(const Count* blocked, Code blocked_n,
                                        Count* out_main,int p,int width){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x;
    Code stride=Code(gridDim.x)*blockDim.x;
    for(;i<blocked_n;i+=stride){
        Count c=blocked[i]; if(!c) continue;
        MateID sm=unrank_mate(i,width-1);
        // Blocked state has X at current position p. Excluding the horizontal edge
        // changes X to N, i.e. inserts N into the compressed blocked state.
        MateID t=minsert(sm,p,N);
        atomic_add_mod(&out_main[rank_mate(t,width)],c);
    }
}

__global__ void main_included_kernel(const Count* in_main, Code main_n,
                                     Count* out_main,Count* out_blocked,
                                     int p,int width){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x;
    Code stride=Code(gridDim.x)*blockDim.x;
    for(;i<main_n;i+=stride){
        Count c=in_main[i]; if(!c) continue;
        MateID m=unrank_mate(i,width);
        MateValuePair w=mpair(m,p);
        switch(w){
        case NN: {
            MateID t=msetpair(m,p,LR);
            atomic_add_mod(&out_main[rank_mate(t,width)],c);
            break;
        }
        case NR: case NL: {
            // At the last horizontal position (p==1), GGCount folds the blocked
            // state immediately back into the main array.  Thus inclusion swaps
            // NR<->RN or NL<->LN instead of creating a deferred state.
            if (p == 1) {
                MateID t=msetpair(m,p,(w==NR)?RN:LN);
                atomic_add_mod(&out_main[rank_mate(t,width)],c);
            } else {
                MateID t=mremove(m,p);
                atomic_add_mod(&out_blocked[rank_mate(t,width-1)],c);
            }
            break;
        }
        case RN: {
            MateID t=msetpair(m,p,NR);
            atomic_add_mod(&out_main[rank_mate(t,width)],c);
            break;
        }
        case LN: {
            MateID t=msetpair(m,p,NL);
            atomic_add_mod(&out_main[rank_mate(t,width)],c);
            break;
        }
        case LL: {
            MateID t=msetpair(m,p,NN);
            int q=p-1,s=1;
            while(s>0){--q; MateValue v=mget(t,q); if(v==L)++s; else if(v==R)--s;}
            t=mset(t,q,L);
            if(p==1) atomic_add_mod(&out_main[rank_mate(t,width)],c);
            else { t=mremove(t,p-1); atomic_add_mod(&out_blocked[rank_mate(t,width-1)],c); }
            break;
        }
        case RR: {
            MateID t=msetpair(m,p,NN);
            int q=p,s=1;
            while(s>0){++q; MateValue v=mget(t,q); if(v==L)--s; else if(v==R)++s;}
            t=mset(t,q,R);
            if(p==1) atomic_add_mod(&out_main[rank_mate(t,width)],c);
            else { t=mremove(t,p-1); atomic_add_mod(&out_blocked[rank_mate(t,width-1)],c); }
            break;
        }
        case RL: {
            MateID t=msetpair(m,p,NN);
            if(p==1) atomic_add_mod(&out_main[rank_mate(t,width)],c);
            else { t=mremove(t,p-1); atomic_add_mod(&out_blocked[rank_mate(t,width-1)],c); }
            break;
        }
        case LR: // would close a cycle
        default:
            break;
        }
    }
}

static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){std::cerr<<w<<": "<<cudaGetErrorString(e)<<"\n";std::exit(1);}}
static unsigned long long H_DP[MAXW+1][MAXW+1];
static void build_dp(){
    for(int h=0;h<=MAXW;++h)H_DP[0][h]=(h==0);
    for(int w=1;w<=MAXW;++w)for(int h=0;h<=MAXW;++h){
        unsigned long long x=H_DP[w-1][h];
        if(h>0)x+=H_DP[w-1][h-1];
        if(h<MAXW)x+=H_DP[w-1][h+1];
        H_DP[w][h]=x;
    }
}
static Code hrank(MateID m,int width){Code rank=0;int h=1;for(int pos=width-1;pos>=0;--pos){auto s=mget(m,pos);if(s>N)rank+=H_DP[pos][h];if(s>R&&h>0)rank+=H_DP[pos][h-1];if(s==R)--h;else if(s==L)++h;}return rank;}

int main(int argc,char**argv){
    int n=argc>1?std::atoi(argv[1]):10;
    Count mod=argc>2?std::strtoull(argv[2],nullptr,10):2305843009213693951ULL;
    int width=n+1;
    if(n<1||width>MAXW){std::cerr<<"supported n=1..27\n";return 1;}
    build_dp(); ck(cudaMemcpyToSymbol(D_DP,H_DP,sizeof(H_DP)),"dp"); ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"mod");
    Code main_n=H_DP[width][1], blocked_n=H_DP[width-1][1];
    size_t value_bytes=size_t(main_n)*sizeof(Count), blocked_bytes=size_t(blocked_n)*sizeof(Count);
    Count *a=nullptr,*b=nullptr,*blocked=nullptr;
    ck(cudaMalloc(&a,value_bytes),"malloc main a"); ck(cudaMalloc(&b,value_bytes),"malloc main b"); ck(cudaMalloc(&blocked,blocked_bytes),"malloc blocked");
    ck(cudaMemset(a,0,value_bytes),"clear a"); ck(cudaMemset(b,0,value_bytes),"clear b"); ck(cudaMemset(blocked,0,blocked_bytes),"clear blocked");
    MateID init=MateID(R)<<(2*(width-1)); Count one=1; Code initc=hrank(init,width); ck(cudaMemcpy(a+initc,&one,sizeof(one),cudaMemcpyHostToDevice),"init");
    cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventRecord(e0);
    int threads=256;
    int main_blocks=int(std::min<Code>(65535,(main_n+threads-1)/threads));
    int blocked_blocks=int(std::min<Code>(65535,(blocked_n+threads-1)/threads));
    for(int row=0;row<width;++row){
        for(int j=0;j<width-1;++j){
            int p=width-j-1;
            // Exclusion from main states is identity.
            ck(cudaMemcpy(b,a,value_bytes,cudaMemcpyDeviceToDevice),"copy excluded main");
            // Old blocked states are consumed by exclusion into main.
            blocked_excluded_kernel<<<blocked_blocks,threads>>>(blocked,blocked_n,b,p,width);
            ck(cudaGetLastError(),"blocked kernel");
            ck(cudaMemset(blocked,0,blocked_bytes),"clear new blocked");
            // Inclusion from main states adds to either main or new blocked states.
            main_included_kernel<<<main_blocks,threads>>>(a,main_n,b,blocked,p,width);
            ck(cudaGetLastError(),"main kernel");
            ck(cudaDeviceSynchronize(),"sync update");
            std::swap(a,b);
        }
        std::cerr<<"row "<<(row+1)<<"/"<<width<<"\n";
    }
    cudaEventRecord(e1);cudaEventSynchronize(e1);float ms=0;cudaEventElapsedTime(&ms,e0,e1);
    MateID fin=MateID(R);Code fc=hrank(fin,width);Count ans=0;ck(cudaMemcpy(&ans,a+fc,sizeof(ans),cudaMemcpyDeviceToHost),"answer");
    size_t alloc=2*value_bytes+blocked_bytes;
    std::cout<<"backend=gridfp-parallel n="<<n<<" residue="<<ans<<" modulus="<<mod
             <<" main_states="<<main_n<<" blocked_states="<<blocked_n
             <<" alloc_bytes="<<alloc<<" gpu_ms="<<ms<<"\n";
    cudaFree(a);cudaFree(b);cudaFree(blocked);
}
