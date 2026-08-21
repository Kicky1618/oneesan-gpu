#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

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

static Code H_DP[MAXW+1][MAXW+2];
__constant__ Code D_FULL_DP[MAXW+1][MAXW+2];
__constant__ Code D_MAIN_DP[MAXW+1][MAXW+2];
__constant__ Code D_BLOCK_DP[MAXW+1][MAXW+2];
__constant__ uint32_t D_MAIN_FIXED, D_MAIN_OCC, D_BLOCK_FIXED, D_BLOCK_OCC;
__constant__ int D_MAIN_W, D_BLOCK_W;
__constant__ Count D_MOD;

__host__ __device__ static inline MateValue mget(MateID m,int k){return MateValue((m>>(2*k))&3ULL);}
__host__ __device__ static inline MateValuePair mpair(MateID m,int p){return MateValuePair((m>>(2*(p-1)))&15ULL);}
__host__ __device__ static inline MateID mset(MateID m,int k,MateValue v){MateID z=3ULL<<(2*k);return (m&~z)|(MateID(v)<<(2*k));}
__host__ __device__ static inline MateID msetpair(MateID m,int p,MateValuePair v){MateID z=15ULL<<(2*(p-1));return (m&~z)|(MateID(v)<<(2*(p-1)));}
// GGCount Mate::shrink(k), not an ordinary erase(k).
__host__ __device__ static inline MateID mshrink(MateID m,int k){
    MateID mask=(1ULL<<(2*k))-1ULL;
    return ((m&~mask)>>2)|(m&mask);
}
__host__ __device__ static inline MateID minsert(MateID m,int k,MateValue v){
    MateID lowmask=k?((1ULL<<(2*k))-1ULL):0ULL;
    MateID lo=m&lowmask, hi=m&~lowmask;
    return lo|(MateID(v)<<(2*k))|(hi<<2);
}

static void ck(cudaError_t e,const char* what){if(e!=cudaSuccess){std::cerr<<what<<": "<<cudaGetErrorString(e)<<"\n";std::exit(1);}}

static void build_full_dp(){
    for(int h=0;h<=MAXW+1;++h) H_DP[0][h]=(h==0);
    for(int w=1;w<=MAXW;++w) for(int h=0;h<=MAXW;++h){
        Code x=H_DP[w-1][h];
        if(h>0)x+=H_DP[w-1][h-1];
        if(h<MAXW+1)x+=H_DP[w-1][h+1];
        H_DP[w][h]=x;
    }
}

struct GroupSpec {
    int width=0;
    uint32_t fixed=0, occ=0;
    Code dp[MAXW+1][MAXW+2]{};
    Code size=0;
};

static GroupSpec make_spec(int width,uint32_t fixed,uint32_t occ){
    GroupSpec s; s.width=width; s.fixed=fixed; s.occ=occ;
    for(int h=0;h<=MAXW+1;++h)s.dp[0][h]=(h==0);
    for(int w=1;w<=width;++w){
        int pos=w-1; bool f=(fixed>>pos)&1u, o=(occ>>pos)&1u;
        for(int h=0;h<=MAXW;++h){
            Code x=0;
            if(!f||!o) x+=s.dp[w-1][h];
            if(!f||o){ if(h>0)x+=s.dp[w-1][h-1]; if(h<MAXW+1)x+=s.dp[w-1][h+1]; }
            s.dp[w][h]=x;
        }
    }
    s.size=s.dp[width][1];
    return s;
}

struct Interval { Code global, local, len; };

static void add_interval(std::vector<Interval>& out,Code g,Code l,Code n){
    if(!n)return;
    if(!out.empty() && out.back().global+out.back().len==g && out.back().local+out.back().len==l) out.back().len+=n;
    else out.push_back({g,l,n});
}

static void intervals_rec(const GroupSpec& s,int pos,int h,Code gbase,Code lbase,std::vector<Interval>& out){
    if(pos<0){ if(h==0)add_interval(out,gbase,lbase,1); return; }
    uint32_t lowerMask = (pos==31)?0xffffffffu:((1u<<(pos+1))-1u);
    if((s.fixed & lowerMask)==0){
        add_interval(out,gbase,lbase,H_DP[pos+1][h]);
        return;
    }
    bool f=(s.fixed>>pos)&1u, o=(s.occ>>pos)&1u;
    // N branch
    Code gsz=H_DP[pos][h];
    if(!f||!o){ Code lsz=s.dp[pos][h]; intervals_rec(s,pos-1,h,gbase,lbase,out); lbase+=lsz; }
    gbase+=gsz;
    // R branch
    if(h>0){
        gsz=H_DP[pos][h-1];
        if(!f||o){ Code lsz=s.dp[pos][h-1]; intervals_rec(s,pos-1,h-1,gbase,lbase,out); lbase+=lsz; }
        gbase+=gsz;
    }
    // L branch
    if(h<MAXW+1){
        gsz=H_DP[pos][h+1];
        if(!f||o){ intervals_rec(s,pos-1,h+1,gbase,lbase,out); }
    }
}

static std::vector<Interval> make_intervals(const GroupSpec& s){
    std::vector<Interval> v; v.reserve(1024);
    intervals_rec(s,s.width-1,1,0,0,v);
    Code sum=0;for(auto const&i:v)sum+=i.len;
    if(sum!=s.size){std::cerr<<"interval size mismatch "<<sum<<" != "<<s.size<<"\n";std::exit(2);}
    return v;
}

static void group_masks(int W,int p,uint32_t group,uint32_t& mf,uint32_t& mo,uint32_t& bf,uint32_t& bo,int& gw){
    int wR=(W+1)/2, wL=W-wR;
    int u=(p-wR>1)?p-wR:1;
    gw=std::max(0,wL-2); mf=mo=bf=bo=0;
    for(int w=0;w<wL;++w){
        if(w==u||w==u-1)continue;
        int k=(w>u)?w-2:w;
        int q=wR+w;
        bool one=(group>>k)&1u;
        mf|=1u<<q; if(one)mo|=1u<<q;
        // deferred ownership is defined by final D->M mapping minsert(sm,p,N)
        if(q==p){std::cerr<<"internal grouped q==p\n";std::exit(2);}
        int bq=(q<p)?q:q-1;
        bf|=1u<<bq; if(one)bo|=1u<<bq;
    }
}

__device__ __forceinline__ bool allowed(uint32_t fixed,uint32_t occ,int pos,MateValue v){
    if(!((fixed>>pos)&1u))return v!=X;
    bool o=(occ>>pos)&1u;
    return o?(v==R||v==L):(v==N);
}

__device__ __forceinline__ MateID unrank_group(Code rank,int width,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    MateID m=0; int h=1;
    #pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){
        if(pos>=width)continue;
        if(allowed(fixed,occ,pos,N)){
            Code z=dp[pos][h]; if(rank<z)continue; rank-=z;
        }
        if(h>0 && allowed(fixed,occ,pos,R)){
            Code z=dp[pos][h-1]; if(rank<z){m|=MateID(R)<<(2*pos);--h;continue;} rank-=z;
        }
        // remaining valid branch is L
        m|=MateID(L)<<(2*pos); ++h;
    }
    return m;
}

__device__ __forceinline__ Code rank_group(MateID m,int width,uint32_t fixed,uint32_t occ,const Code dp[MAXW+1][MAXW+2]){
    Code rank=0;int h=1;
    #pragma unroll
    for(int pos=MAXW-1;pos>=0;--pos){
        if(pos>=width)continue;
        MateValue s=mget(m,pos);
        if(s>N && allowed(fixed,occ,pos,N)) rank+=dp[pos][h];
        if(s>R && h>0 && allowed(fixed,occ,pos,R)) rank+=dp[pos][h-1];
        if(s==R)--h; else if(s==L)++h;
    }
    return rank;
}

__device__ __forceinline__ void atomic_add_mod(Count* p,Count v){
    if(!v)return; Count mod=D_MOD; Count old=atomicAdd(p,0ULL);
    for(;;){Count neu=(old>=mod-v)?old-(mod-v):old+v;Count seen=atomicCAS(p,old,neu);if(seen==old)return;old=seen;}
}

__global__ void blocked_group_kernel(const Count* in,Code n,Count* out_main,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x, stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){Count c=in[i];if(!c)continue;MateID sm=unrank_group(i,D_BLOCK_W,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);MateID t=minsert(sm,p,N);Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);}
}

__global__ void main_group_kernel(const Count* in,Code n,Count* out_main,Count* out_block,int p){
    Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x, stride=Code(gridDim.x)*blockDim.x;
    for(;i<n;i+=stride){
        Count c=in[i];if(!c)continue;MateID m=unrank_group(i,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);MateValuePair w=mpair(m,p);
        switch(w){
        case NN:{MateID t=msetpair(m,p,LR);Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);break;}
        case NR:case NL:{if(p==1){MateID t=msetpair(m,p,w==NR?RN:LN);Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);}else{MateID t=mshrink(m,p);Code j=rank_group(t,D_BLOCK_W,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);atomic_add_mod(out_block+j,c);}break;}
        case RN:{MateID t=msetpair(m,p,NR);Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);break;}
        case LN:{MateID t=msetpair(m,p,NL);Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);break;}
        case LL:{MateID t=msetpair(m,p,NN);int q=p-1,s=1;while(s){--q;auto v=mget(t,q);if(v==L)++s;else if(v==R)--s;}t=mset(t,q,L);if(p==1){Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);}else{t=mshrink(t,p-1);Code j=rank_group(t,D_BLOCK_W,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);atomic_add_mod(out_block+j,c);}break;}
        case RR:{MateID t=msetpair(m,p,NN);int q=p,s=1;while(s){++q;auto v=mget(t,q);if(v==L)--s;else if(v==R)++s;}t=mset(t,q,R);if(p==1){Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);}else{t=mshrink(t,p-1);Code j=rank_group(t,D_BLOCK_W,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);atomic_add_mod(out_block+j,c);}break;}
        case RL:{MateID t=msetpair(m,p,NN);if(p==1){Code j=rank_group(t,D_MAIN_W,D_MAIN_FIXED,D_MAIN_OCC,D_MAIN_DP);atomic_add_mod(out_main+j,c);}else{t=mshrink(t,p-1);Code j=rank_group(t,D_BLOCK_W,D_BLOCK_FIXED,D_BLOCK_OCC,D_BLOCK_DP);atomic_add_mod(out_block+j,c);}break;}
        default:break;
        }
    }
}

static void gather(const std::vector<Count>& global,const std::vector<Interval>& iv,std::vector<Count>& local){
    for(auto const& x:iv) std::memcpy(local.data()+x.local,global.data()+x.global,size_t(x.len)*sizeof(Count));
}
static void scatter(std::vector<Count>& global,const std::vector<Interval>& iv,const std::vector<Count>& local){
    for(auto const& x:iv) std::memcpy(global.data()+x.global,local.data()+x.local,size_t(x.len)*sizeof(Count));
}

static Code rank_full(MateID m,int width){Code r=0;int h=1;for(int pos=width-1;pos>=0;--pos){auto s=mget(m,pos);if(s>N)r+=H_DP[pos][h];if(s>R&&h>0)r+=H_DP[pos][h-1];if(s==R)--h;else if(s==L)++h;}return r;}

int main(int argc,char**argv){
    int n=argc>1?std::atoi(argv[1]):10; Count mod=argc>2?std::strtoull(argv[2],nullptr,10):2305843009213693951ULL;
    int W=n+1;if(n<2||W>MAXW){std::cerr<<"stream prototype supports n=2..27\n";return 1;}
    build_full_dp();ck(cudaMemcpyToSymbol(D_FULL_DP,H_DP,sizeof(H_DP)),"full dp");ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"mod");
    Code mainN=H_DP[W][1], blockN=H_DP[W-1][1];
    // External store: one main array + one deferred array, exactly the paper memory model.
    std::vector<Count> mainv(mainN), blockv(blockN);
    MateID init=MateID(R)<<(2*(W-1));mainv[rank_full(init,W)]=1;
    Count *dA=nullptr,*dB=nullptr,*dD=nullptr;Code capM=0,capD=0;
    std::vector<Count> hA,hB,hD;
    int threads=256; double transferred=0; Code maxGM=0,maxGD=0;size_t maxIntervals=0;
    cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventRecord(e0);
    for(int row=0;row<W;++row){
        for(int jj=0;jj<W-1;++jj){
            int p=W-jj-1;int dummyGw;uint32_t mf,mo,bf,bo;group_masks(W,p,0,mf,mo,bf,bo,dummyGw);int ng=1<<dummyGw;
            std::vector<int> order(ng);std::iota(order.begin(),order.end(),0);std::sort(order.begin(),order.end(),[](int a,int b){return __builtin_popcount((unsigned)a)>__builtin_popcount((unsigned)b);});
            for(int g:order){
                int gw;group_masks(W,p,g,mf,mo,bf,bo,gw);GroupSpec ms=make_spec(W,mf,mo), ds=make_spec(W-1,bf,bo);if(!ms.size&&!ds.size)continue;
                auto mi=make_intervals(ms),di=make_intervals(ds);maxGM=std::max(maxGM,ms.size);maxGD=std::max(maxGD,ds.size);maxIntervals=std::max({maxIntervals,mi.size(),di.size()});
                if(ms.size>capM){if(dA){cudaFree(dA);cudaFree(dB);}capM=ms.size;ck(cudaMalloc(&dA,size_t(capM)*8),"dA");ck(cudaMalloc(&dB,size_t(capM)*8),"dB");hA.resize(capM);hB.resize(capM);}if(ds.size>capD){if(dD)cudaFree(dD);capD=ds.size;ck(cudaMalloc(&dD,size_t(capD)*8),"dD");hD.resize(capD);}
                gather(mainv,mi,hA);gather(blockv,di,hD);transferred+=double(ms.size+ds.size)*8.0;
                if(ms.size)ck(cudaMemcpy(dA,hA.data(),size_t(ms.size)*8,cudaMemcpyHostToDevice),"H2D main");
                if(ds.size)ck(cudaMemcpy(dD,hD.data(),size_t(ds.size)*8,cudaMemcpyHostToDevice),"H2D block");
                ck(cudaMemcpy(dB,dA,size_t(ms.size)*8,cudaMemcpyDeviceToDevice),"identity");
                ck(cudaMemcpyToSymbol(D_MAIN_DP,ms.dp,sizeof(ms.dp)),"main gdp");ck(cudaMemcpyToSymbol(D_BLOCK_DP,ds.dp,sizeof(ds.dp)),"block gdp");ck(cudaMemcpyToSymbol(D_MAIN_FIXED,&mf,sizeof(mf)),"mf");ck(cudaMemcpyToSymbol(D_MAIN_OCC,&mo,sizeof(mo)),"mo");ck(cudaMemcpyToSymbol(D_BLOCK_FIXED,&bf,sizeof(bf)),"bf");ck(cudaMemcpyToSymbol(D_BLOCK_OCC,&bo,sizeof(bo)),"bo");int mw=W,bw=W-1;ck(cudaMemcpyToSymbol(D_MAIN_W,&mw,sizeof(mw)),"mw");ck(cudaMemcpyToSymbol(D_BLOCK_W,&bw,sizeof(bw)),"bw");
                int bm=int(std::min<Code>(65535,(ms.size+threads-1)/threads)),bd=int(std::min<Code>(65535,(ds.size+threads-1)/threads));
                if(ds.size)blocked_group_kernel<<<bd,threads>>>(dD,ds.size,dB,p);
                if(ds.size)ck(cudaMemset(dD,0,size_t(ds.size)*8),"clear new block");
                if(ms.size)main_group_kernel<<<bm,threads>>>(dA,ms.size,dB,dD,p);
                ck(cudaGetLastError(),"group kernels");ck(cudaDeviceSynchronize(),"group sync");
                if(ms.size)ck(cudaMemcpy(hB.data(),dB,size_t(ms.size)*8,cudaMemcpyDeviceToHost),"D2H main");
                if(ds.size)ck(cudaMemcpy(hD.data(),dD,size_t(ds.size)*8,cudaMemcpyDeviceToHost),"D2H block");transferred+=double(ms.size+ds.size)*8.0;
                scatter(mainv,mi,hB);scatter(blockv,di,hD);
            }
        }
        std::cerr<<"row "<<row+1<<"/"<<W<<"\n";
    }
    cudaEventRecord(e1);cudaEventSynchronize(e1);float ms=0;cudaEventElapsedTime(&ms,e0,e1);
    Count ans=mainv[rank_full(MateID(R),W)];
    std::cout<<"backend=gridfp-stream n="<<n<<" residue="<<ans<<" modulus="<<mod<<" main_states="<<mainN<<" blocked_states="<<blockN<<" external_bytes="<<size_t(mainN+blockN)*8ULL<<" max_group_main="<<maxGM<<" max_group_blocked="<<maxGD<<" max_vram_bytes="<<size_t(2*maxGM+maxGD)*8ULL<<" max_intervals="<<maxIntervals<<" transfer_gib="<<transferred/(1ull<<30)<<" gpu_ms="<<ms<<"\n";
    if(dA){cudaFree(dA);cudaFree(dB);}if(dD)cudaFree(dD);
}
