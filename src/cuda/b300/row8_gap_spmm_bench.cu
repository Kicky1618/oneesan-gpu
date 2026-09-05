#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static inline void ck(cudaError_t e,const char*w){if(e!=cudaSuccess)throw std::runtime_error(std::string(w)+": "+cudaGetErrorString(e));}
struct SHdr {char magic[8];uint32_t version,r;uint32_t dims[9];uint64_t total_nz,fnv_hash;};
struct BHdr {uint32_t sym,h,h2,rows,cols,nnz;};
struct VHdr {uint32_t tag,sym,h,nnz;};
struct Block {int rows=0,cols=0;std::vector<uint32_t>rp,cp;std::vector<uint16_t>ci,ri;std::vector<int8_t>cv,cv_csc;};
template<class T>static T rd(std::ifstream&in){T x{};in.read((char*)&x,sizeof(x));if(!in)throw std::runtime_error("short read");return x;}
static Block load_block(const std::string&path,bool gap,int wantA,int wantH){
    std::ifstream in(path,std::ios::binary);if(!in)throw std::runtime_error("open "+path);auto sh=rd<SHdr>(in);
    for(int q=0;q<5;++q){auto v=rd<VHdr>(in);in.seekg((std::streamoff)v.nnz*(gap?2:3),std::ios::cur);}
    for(int q=0;q<25;++q){auto b=rd<BHdr>(in);Block z;z.rows=b.rows;z.cols=b.cols;z.rp.resize((size_t)b.rows+1);z.ci.resize(b.nnz);in.read((char*)z.rp.data(),z.rp.size()*4);in.read((char*)z.ci.data(),z.ci.size()*2);if(!gap){z.cv.resize(b.nnz);in.read((char*)z.cv.data(),z.cv.size());}if(!in)throw std::runtime_error("truncated block");if((int)b.sym==wantA&&(int)b.h==wantH){
            z.cp.assign((size_t)z.cols+1,0);for(auto j:z.ci)++z.cp[(size_t)j+1];for(int j=0;j<z.cols;++j)z.cp[j+1]+=z.cp[j];auto cur=z.cp;z.ri.resize(z.ci.size());if(!gap)z.cv_csc.resize(z.cv.size());for(int i=0;i<z.rows;++i)for(uint32_t e=z.rp[i];e<z.rp[i+1];++e){auto j=z.ci[e];auto p=cur[j]++;z.ri[p]=(uint16_t)i;if(!gap)z.cv_csc[p]=z.cv[e];}return z;}}
    throw std::runtime_error("block missing");
}
__device__ inline unsigned long long term(uint32_t x,int c,uint32_t mod){if(c==1)return x;if(c==2)return 2ULL*x;if(c==-1)return x?uint64_t(mod)-x:0;if(c==-2)return x?2ULL*(uint64_t(mod)-x):0;long long z=(long long)x*c,m=(long long)mod;z%=m;if(z<0)z+=m;return (uint64_t)z;}
__global__ void structural_forward(const uint32_t*x,int m,uint32_t*y,const uint32_t*cp,const uint16_t*ri,const int8_t*cv,int n,uint32_t mod){unsigned long long total=(unsigned long long)m*n;for(unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,st=(unsigned long long)gridDim.x*blockDim.x;q<total;q+=st){int r=q%m,j=q/m;unsigned long long s=0;for(uint32_t e=cp[j];e<cp[j+1];++e)s+=term(x[(size_t)ri[e]*m+r],cv[e],mod);y[q]=(uint32_t)(s%mod);}}
__global__ void gap_forward(const uint32_t*x,int m,uint32_t*y,const uint32_t*cp,const uint16_t*ri,int n,uint32_t mod){unsigned long long total=(unsigned long long)m*n;for(unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,st=(unsigned long long)gridDim.x*blockDim.x;q<total;q+=st){int r=q%m,j=q/m;unsigned long long s=0;for(uint32_t e=cp[j];e<cp[j+1];++e)s+=x[(size_t)ri[e]*m+r];y[q]=(uint32_t)(s%mod);}}
static float ms(cudaEvent_t a,cudaEvent_t b){float x=0;ck(cudaEventElapsedTime(&x,a,b),"elapsed");return x;}
int main(int ac,char**av){
    int m=ac>1?std::atoi(av[1]):4096,reps=ac>2?std::atoi(av[2]):50,a=ac>3?std::atoi(av[3]):0,h=ac>4?std::atoi(av[4]):1;uint32_t mod=1000000007u;
    auto S=load_block("work/row8_structural_cache/row8_structural_int_v1.bin",false,a,h);auto G=load_block("work/row8_gap_cache/row8_gap01.bin",true,a,h);if(S.rows!=G.rows||S.cols!=G.cols)throw std::runtime_error("shape mismatch");int k=S.rows,n=S.cols;
    std::vector<uint32_t>hx((size_t)k*m);std::mt19937 rng(7);for(auto&x:hx)x=rng()%mod;
    uint32_t *dx=nullptr,*ys=nullptr,*yg=nullptr,*scp=nullptr,*gcp=nullptr;uint16_t *sri=nullptr,*gri=nullptr;int8_t*scv=nullptr;
    ck(cudaMalloc(&dx,hx.size()*4),"dx");ck(cudaMalloc(&ys,(size_t)n*m*4),"ys");ck(cudaMalloc(&yg,(size_t)n*m*4),"yg");ck(cudaMalloc(&scp,S.cp.size()*4),"scp");ck(cudaMalloc(&gcp,G.cp.size()*4),"gcp");ck(cudaMalloc(&sri,S.ri.size()*2),"sri");ck(cudaMalloc(&gri,G.ri.size()*2),"gri");ck(cudaMalloc(&scv,S.cv_csc.size()),"scv");
    ck(cudaMemcpy(dx,hx.data(),hx.size()*4,cudaMemcpyHostToDevice),"x");ck(cudaMemcpy(scp,S.cp.data(),S.cp.size()*4,cudaMemcpyHostToDevice),"scp");ck(cudaMemcpy(gcp,G.cp.data(),G.cp.size()*4,cudaMemcpyHostToDevice),"gcp");ck(cudaMemcpy(sri,S.ri.data(),S.ri.size()*2,cudaMemcpyHostToDevice),"sri");ck(cudaMemcpy(gri,G.ri.data(),G.ri.size()*2,cudaMemcpyHostToDevice),"gri");ck(cudaMemcpy(scv,S.cv_csc.data(),S.cv_csc.size(),cudaMemcpyHostToDevice),"scv");
    int blocks=(int)std::max<unsigned long long>(1,std::min<unsigned long long>(65535,((unsigned long long)m*n+255)/256));for(int i=0;i<5;++i){structural_forward<<<blocks,256>>>(dx,m,ys,scp,sri,scv,n,mod);gap_forward<<<blocks,256>>>(dx,m,yg,gcp,gri,n,mod);}ck(cudaDeviceSynchronize(),"warm");
    cudaEvent_t e0,e1;ck(cudaEventCreate(&e0),"e0");ck(cudaEventCreate(&e1),"e1");
    ck(cudaEventRecord(e0),"s0");for(int i=0;i<reps;++i)structural_forward<<<blocks,256>>>(dx,m,ys,scp,sri,scv,n,mod);ck(cudaEventRecord(e1),"s1");ck(cudaEventSynchronize(e1),"ss");float sm=ms(e0,e1)/reps;
    ck(cudaEventRecord(e0),"g0");for(int i=0;i<reps;++i)gap_forward<<<blocks,256>>>(dx,m,yg,gcp,gri,n,mod);ck(cudaEventRecord(e1),"g1");ck(cudaEventSynchronize(e1),"gs");float gm=ms(e0,e1)/reps;
    std::cout<<"m="<<m<<" a="<<a<<" h="<<h<<" shape="<<k<<"x"<<n<<" structural_nz="<<S.ci.size()<<" gap_nz="<<G.ci.size()<<" structural_ms="<<sm<<" gap_ms="<<gm<<" speedup="<<sm/gm<<"\n";
    cudaEventDestroy(e0);cudaEventDestroy(e1);cudaFree(dx);cudaFree(ys);cudaFree(yg);cudaFree(scp);cudaFree(gcp);cudaFree(sri);cudaFree(gri);cudaFree(scv);return 0;
}
