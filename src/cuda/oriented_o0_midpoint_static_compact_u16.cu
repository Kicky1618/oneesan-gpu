#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "o0_midpoint_tables.hpp"

namespace {
constexpr std::uint32_t MOD=65521, ZETA=61640, ZETA_INV=19685;
constexpr int MAXH=10, MAXQ=2*MAXH+1;
constexpr int KQ_HOST[4]={-1,0,0,1};
constexpr int DQ_HOST[3]={0,1,-1};
#define CUDA_CHECK(x) do{auto e=(x);if(e!=cudaSuccess)throw std::runtime_error(std::string(#x)+": "+cudaGetErrorString(e));}while(0)

__constant__ std::uint64_t D_BASE[4];
__constant__ std::uint64_t D_OFF[4][MAXQ];
__constant__ std::uint64_t D_CNT_R[MAXQ];
__constant__ int D_L,D_R;

__device__ __forceinline__ int dq(int d){return d==1?1:d==2?-1:0;}
__device__ __forceinline__ int kq(int k){return k==0?-1:k==3?1:0;}
__device__ __forceinline__ std::uint32_t add2(std::uint32_t a,std::uint32_t b){auto x=a+b;return x>=MOD?x-MOD:x;}
__device__ __forceinline__ std::uint32_t reduce_mod(std::uint32_t x){std::uint32_t r=(x&0xffffu)+15u*(x>>16);r=(r&0xffffu)+15u*(r>>16);if(r>=MOD)r-=MOD;return r;}
__device__ __forceinline__ std::uint32_t mulm(std::uint32_t a,std::uint32_t b){return reduce_mod(a*b);}

__device__ __forceinline__ std::uint64_t state_index(int k,std::uint32_t lraw,std::uint32_t rraw,
                                                     const std::uint32_t* __restrict__ rL,
                                                     const std::uint32_t* __restrict__ rR,
                                                     const std::int8_t* __restrict__ qL,
                                                     const std::int8_t* __restrict__ qR){
    int a=qL[lraw], b=qR[rraw];
    if(a+b!=1-kq(k)) return ~std::uint64_t{0};
    return D_BASE[k]+D_OFF[k][a+D_L]+std::uint64_t(rL[lraw])*D_CNT_R[b+D_R]+rR[rraw];
}

__device__ __forceinline__ void transform12(bool rev,const std::uint32_t x[12],std::uint32_t o[12]){
    if(!rev){
        o[0]=add2(add2(x[0],x[9]),mulm(ZETA,x[10]));
        o[1]=add2(x[1],mulm(ZETA,x[2]));
        o[2]=add2(mulm(53583,x[4]),mulm(ZETA,x[11]));
        o[3]=add2(add2(x[5],mulm(ZETA,x[6])),x[3]);
        o[4]=add2(mulm(57852,x[1]),x[11]); o[5]=x[3]; o[6]=mulm(ZETA_INV,x[5]);
        o[7]=x[7]; o[8]=x[8]; o[9]=x[0]; o[10]=mulm(ZETA_INV,x[9]); o[11]=add2(x[4],mulm(8031,x[1]));
    }else{
        o[0]=add2(add2(x[0],x[9]),mulm(ZETA_INV,x[10]));
        o[1]=add2(x[1],mulm(ZETA_INV,x[2]));
        o[2]=add2(mulm(16855,x[4]),mulm(ZETA_INV,x[11]));
        o[3]=add2(add2(x[5],mulm(ZETA_INV,x[6])),x[3]);
        o[4]=add2(mulm(8031,x[1]),x[11]); o[5]=x[3]; o[6]=mulm(ZETA,x[5]);
        o[7]=x[7]; o[8]=x[8]; o[9]=x[0]; o[10]=mulm(ZETA,x[9]); o[11]=add2(x[4],mulm(57852,x[1]));
    }
}

__global__ void local_left(std::uint16_t* v,int p,bool rev,bool bottom,std::uint64_t threads_total,
                           std::uint32_t powp,std::uint32_t powp1,std::uint32_t rest_count,std::uint32_t right_count,
                           const std::uint32_t* rL,const std::uint32_t* rR,const std::int8_t* qL,const std::int8_t* qR,
                           const std::int8_t* qRest){
    std::uint64_t t=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=threads_total)return;
    std::uint32_t rest=t/right_count, rr=t%right_count;
    int qrest=qRest[rest], qr=qR[rr], T=1-qrest-qr;
    if(T<-2||T>2)return;
    std::uint32_t low=rest%powp, high=rest/powp;
    std::uint32_t x[12]{}; std::uint64_t ids[12];
#pragma unroll
    for(int k=0;k<4;++k)for(int d=0;d<3;++d){int s=k+4*d;ids[s]=~std::uint64_t{0};if(kq(k)+dq(d)!=T)continue;std::uint32_t lr=low+d*powp+high*powp1;auto id=state_index(k,lr,rr,rL,rR,qL,qR);ids[s]=id;if(id!=~std::uint64_t{0})x[s]=v[id];}
    std::uint32_t o[12];transform12(rev,x,o);
#pragma unroll
    for(int k=0;k<4;++k)for(int d=0;d<3;++d){int s=k+4*d;auto id=ids[s];if(id==~std::uint64_t{0})continue;v[id]=(bottom&&d!=0)?0:static_cast<std::uint16_t>(o[s]);}
}

__global__ void local_right(std::uint16_t* v,int p,bool rev,bool bottom,std::uint64_t threads_total,
                            std::uint32_t powp,std::uint32_t powp1,std::uint32_t left_count,std::uint32_t rest_count,
                            const std::uint32_t* rL,const std::uint32_t* rR,const std::int8_t* qL,const std::int8_t* qR,
                            const std::int8_t* qRest){
    std::uint64_t t=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=threads_total)return;
    std::uint32_t lr=t/rest_count, rest=t%rest_count;
    int ql=qL[lr], qrest=qRest[rest], T=1-ql-qrest;
    if(T<-2||T>2)return;
    std::uint32_t low=rest%powp, high=rest/powp;
    std::uint32_t x[12]{}; std::uint64_t ids[12];
#pragma unroll
    for(int k=0;k<4;++k)for(int d=0;d<3;++d){int s=k+4*d;ids[s]=~std::uint64_t{0};if(kq(k)+dq(d)!=T)continue;std::uint32_t rr=low+d*powp+high*powp1;auto id=state_index(k,lr,rr,rL,rR,qL,qR);ids[s]=id;if(id!=~std::uint64_t{0})x[s]=v[id];}
    std::uint32_t o[12];transform12(rev,x,o);
#pragma unroll
    for(int k=0;k<4;++k)for(int d=0;d<3;++d){int s=k+4*d;auto id=ids[s];if(id==~std::uint64_t{0})continue;v[id]=(bottom&&d!=0)?0:static_cast<std::uint16_t>(o[s]);}
}

__global__ void turn_kernel(std::uint16_t* v,std::uint64_t raw_total,std::uint32_t right_count,
                            const std::uint32_t* rL,const std::uint32_t* rR,const std::int8_t* qL,const std::int8_t* qR,bool rev){
    std::uint64_t t=std::uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(t>=raw_total)return;
    std::uint32_t lr=t/right_count, rr=t%right_count;if(qL[lr]+qR[rr]!=1)return;
    auto i1=state_index(1,lr,rr,rL,rR,qL,qR), i2=state_index(2,lr,rr,rL,rR,qL,qR);
    std::uint32_t x1=v[i1],x2=v[i2];v[i1]=static_cast<std::uint16_t>(add2(x1,mulm(rev?ZETA_INV:ZETA,x2)));v[i2]=0;
}

std::uint64_t pow3i(int e){std::uint64_t x=1;while(e--)x*=3;return x;}
int charge_raw(std::uint32_t raw,int h){int q=0;for(int i=0;i<h;++i){int d=raw%3;raw/=3;q+=DQ_HOST[d];}return q;}
struct HalfTable{int h=0;std::uint32_t n=0;std::vector<std::uint32_t> rank;std::vector<std::int8_t> charge;std::array<std::uint64_t,MAXQ> count{};};
HalfTable make_half(int h){HalfTable t;t.h=h;t.n=pow3i(h);t.rank.resize(t.n);t.charge.resize(t.n);std::array<std::uint32_t,MAXQ> next{};for(std::uint32_t raw=0;raw<t.n;++raw){int q=charge_raw(raw,h);t.charge[raw]=q;t.rank[raw]=next[q+h]++;}for(int q=-h;q<=h;++q)t.count[q+h]=next[q+h];return t;}

struct Layout{int m,L,R;HalfTable left,right;std::array<std::uint64_t,4> base{};std::array<std::array<std::uint64_t,MAXQ>,4> off{};std::uint64_t total=0;};
Layout make_layout(int m){Layout z;z.m=m;z.L=m/2;z.R=m-z.L;if(z.L>MAXH||z.R>MAXH)throw std::runtime_error("prototype half too large");z.left=make_half(z.L);z.right=make_half(z.R);for(int k=0;k<4;++k){z.base[k]=z.total;int need=1-KQ_HOST[k];std::uint64_t cur=0;for(int a=-z.L;a<=z.L;++a){z.off[k][a+z.L]=cur;int b=need-a;if(b>=-z.R&&b<=z.R)cur+=z.left.count[a+z.L]*z.right.count[b+z.R];}z.total+=cur;}return z;}

struct DevTables{std::uint32_t *rL=nullptr,*rR=nullptr;std::int8_t *qL=nullptr,*qR=nullptr;};
void upload_layout(const Layout& z,DevTables& d){CUDA_CHECK(cudaMalloc(&d.rL,z.left.rank.size()*4));CUDA_CHECK(cudaMalloc(&d.rR,z.right.rank.size()*4));CUDA_CHECK(cudaMalloc(&d.qL,z.left.charge.size()));CUDA_CHECK(cudaMalloc(&d.qR,z.right.charge.size()));CUDA_CHECK(cudaMemcpy(d.rL,z.left.rank.data(),z.left.rank.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(d.rR,z.right.rank.data(),z.right.rank.size()*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(d.qL,z.left.charge.data(),z.left.charge.size(),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(d.qR,z.right.charge.data(),z.right.charge.size(),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpyToSymbol(D_BASE,z.base.data(),sizeof(z.base)));CUDA_CHECK(cudaMemcpyToSymbol(D_OFF,z.off.data(),sizeof(z.off)));CUDA_CHECK(cudaMemcpyToSymbol(D_CNT_R,z.right.count.data(),sizeof(z.right.count)));CUDA_CHECK(cudaMemcpyToSymbol(D_L,&z.L,sizeof(int)));CUDA_CHECK(cudaMemcpyToSymbol(D_R,&z.R,sizeof(int)));}
void free_tables(DevTables& d){cudaFree(d.rL);cudaFree(d.rR);cudaFree(d.qL);cudaFree(d.qR);}

std::uint64_t host_index(const Layout& z,int k,std::uint32_t lr,std::uint32_t rr){int a=z.left.charge[lr],b=z.right.charge[rr];if(a+b!=1-KQ_HOST[k])return ~std::uint64_t{0};return z.base[k]+z.off[k][a+z.L]+std::uint64_t(z.left.rank[lr])*z.right.count[b+z.R]+z.right.rank[rr];}

struct Result{std::uint32_t residue;std::uint64_t states;float ms;};
Result solve(int n){if(n<3||!(n&1)||n>19)throw std::runtime_error("odd n 3..19");int m=n-1;auto z=make_layout(m);DevTables d;upload_layout(z,d);std::uint16_t* v=nullptr;CUDA_CHECK(cudaMalloc(&v,z.total*2));CUDA_CHECK(cudaMemset(v,0,z.total*2));constexpr int TPB=256;auto blocks=[](std::uint64_t n){return unsigned((n+TPB-1)/TPB);};
    // Source.
    for(int d0=0;d0<2;++d0){int r=d0==0?1:0;std::uint32_t sw=d0==0?1:ZETA_INV;std::uint32_t lraw=(z.L?d0:0),rraw=0;for(int k=0;k<4;++k){auto a=o0mid::A_F[k*9+r];if(!a)continue;auto id=host_index(z,k,lraw,rraw);if(id==~std::uint64_t{0})continue;std::uint16_t w=std::uint64_t(sw)*a%MOD;CUDA_CHECK(cudaMemcpy(v+id,&w,2,cudaMemcpyHostToDevice));}}
    auto dump=[&](const char* tag){ if(n!=3)return; std::vector<std::uint16_t> h(z.total); CUDA_CHECK(cudaMemcpy(h.data(),v,z.total*2,cudaMemcpyDeviceToHost)); std::cerr<<tag<<":"; for(auto x:h)std::cerr<<" "<<x; std::cerr<<"\n"; }; dump("source");
    // Rest-charge tables per half-minus-one.
    auto qLm1=make_half(z.L-1),qRm1=make_half(z.R-1);std::int8_t *dqLm1=nullptr,*dqRm1=nullptr;CUDA_CHECK(cudaMalloc(&dqLm1,qLm1.charge.size()));CUDA_CHECK(cudaMalloc(&dqRm1,qRm1.charge.size()));CUDA_CHECK(cudaMemcpy(dqLm1,qLm1.charge.data(),qLm1.charge.size(),cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(dqRm1,qRm1.charge.data(),qRm1.charge.size(),cudaMemcpyHostToDevice));
    cudaEvent_t a{},b{};cudaEventCreate(&a);cudaEventCreate(&b);cudaEventRecord(a);
    auto local=[&](int p,bool rev,bool bottom=false){if(p<z.L){int hp=p;std::uint32_t pp=pow3i(hp),pp1=pp*3,rest=pow3i(z.L-1),right=z.right.n;std::uint64_t nt=std::uint64_t(rest)*right;local_left<<<blocks(nt),TPB>>>(v,p,rev,bottom,nt,pp,pp1,rest,right,d.rL,d.rR,d.qL,d.qR,dqLm1);}else{int hp=p-z.L;std::uint32_t pp=pow3i(hp),pp1=pp*3,left=z.left.n,rest=pow3i(z.R-1);std::uint64_t nt=std::uint64_t(left)*rest;local_right<<<blocks(nt),TPB>>>(v,hp,rev,bottom,nt,pp,pp1,left,rest,d.rL,d.rR,d.qL,d.qR,dqRm1);}CUDA_CHECK(cudaGetLastError());};
    auto turn=[&](bool rev){std::uint64_t nt=std::uint64_t(z.left.n)*z.right.n;turn_kernel<<<blocks(nt),TPB>>>(v,nt,z.right.n,d.rL,d.rR,d.qL,d.qR,rev);CUDA_CHECK(cudaGetLastError());};
    for(int p=1;p<m;++p)local(p,false); dump("top");
    bool forward=true;for(int y=1;y<n-1;++y){if(forward){turn(false);forward=false;for(int p=m-1;p>=0;--p)local(p,true);}else{turn(true);forward=true;for(int p=0;p<m;++p)local(p,false);}}
    dump("interior"); if(forward)throw std::runtime_error("orientation");turn(true);for(int p=0;p<m-1;++p)local(p,false,true); dump("bottom");
    cudaEventRecord(b);cudaEventSynchronize(b);float ms=0;cudaEventElapsedTime(&ms,a,b);
    // Target: q[0..m-2]=0, q[m-1]=u.
    std::uint32_t ans=0;for(int k=0;k<4;++k)for(int u=0;u<3;++u){std::uint32_t raw=std::uint32_t(u*pow3i(m-1));std::uint32_t lr=raw%z.left.n,rr=raw/z.left.n;auto id=host_index(z,k,lr,rr);if(id==~std::uint64_t{0})continue;std::uint32_t tw=0;for(int r=0;r<3;++r){std::uint32_t ew=(r==1&&u==0)?1:((r==0&&u==1)?ZETA:0);if(ew)tw=(tw+std::uint64_t(o0mid::B_F[(3*r)*4+k])*ew)%MOD;}if(!tw)continue;std::uint16_t val=0;CUDA_CHECK(cudaMemcpy(&val,v+id,2,cudaMemcpyDeviceToHost));ans=(ans+std::uint64_t(val)*tw)%MOD;}
    cudaEventDestroy(a);cudaEventDestroy(b);cudaFree(dqLm1);cudaFree(dqRm1);cudaFree(v);free_tables(d);return{ans,z.total,ms};}
}
int main(int argc,char**argv){CUDA_CHECK(cudaSetDevice(0));cudaDeviceProp p{};CUDA_CHECK(cudaGetDeviceProperties(&p,0));std::cerr<<"gpu="<<p.name<<"\n";int first=argc>1?std::atoi(argv[1]):3,last=argc>2?std::atoi(argv[2]):first;for(int n=first;n<=last;n+=2){auto r=solve(n);std::cout<<"n="<<n<<" residue="<<r.residue<<" states="<<r.states<<" mib="<<std::fixed<<std::setprecision(1)<<(r.states*2.0/1048576.0)<<" ms="<<std::setprecision(3)<<r.ms<<"\n";}}
