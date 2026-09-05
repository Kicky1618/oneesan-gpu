#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
static inline void ck(cudaError_t e,const char*w){if(e!=cudaSuccess)throw std::runtime_error(std::string(w)+": "+cudaGetErrorString(e));}
struct SHdr{char magic[8];uint32_t version,r,dims[9];uint64_t total_nz,fnv_hash;};
struct BHdr{uint32_t sym,h,h2,rows,cols,nnz;};struct VHdr{uint32_t tag,sym,h,nnz;};
struct Block{int k=0,n=0;std::vector<uint32_t>rp;std::vector<uint16_t>ci;std::vector<int8_t>cv;};
template<class T>static T rd(std::ifstream&in){T x{};in.read((char*)&x,sizeof(x));if(!in)throw std::runtime_error("short");return x;}
static Block loadB(int wa,int wh){std::ifstream in("work/row8_structural_cache/row8_structural_int_v1.bin",std::ios::binary);auto sh=rd<SHdr>(in);for(int q=0;q<5;++q){auto v=rd<VHdr>(in);in.seekg((std::streamoff)v.nnz*3,std::ios::cur);}for(int q=0;q<25;++q){auto b=rd<BHdr>(in);Block z;z.k=b.rows;z.n=b.cols;z.rp.resize(b.rows+1);z.ci.resize(b.nnz);z.cv.resize(b.nnz);in.read((char*)z.rp.data(),z.rp.size()*4);in.read((char*)z.ci.data(),z.ci.size()*2);in.read((char*)z.cv.data(),z.cv.size());if((int)b.sym==wa&&(int)b.h==wh)return z;}throw std::runtime_error("block missing");}
struct Csc{std::vector<uint32_t>cp;std::vector<uint16_t>ri;std::vector<int8_t>cv;};
static Csc toc(Block const&b){Csc z;z.cp.assign(b.n+1,0);for(auto j:b.ci)++z.cp[j+1];for(int j=0;j<b.n;++j)z.cp[j+1]+=z.cp[j];auto cur=z.cp;z.ri.resize(b.ci.size());z.cv.resize(b.cv.size());for(int i=0;i<b.k;++i)for(uint32_t e=b.rp[i];e<b.rp[i+1];++e){uint16_t j=b.ci[e]; uint32_t pp=cur[j]++;z.ri[pp]=i;z.cv[pp]=b.cv[e];}return z;}
__device__ __forceinline__ uint32_t red(uint64_t x,uint32_t p,uint64_t mu){uint64_t q=__umul64hi(x,mu),r=x-q*(uint64_t)p;if(r>=p)r-=p;if(r>=p)r-=p;return r;}
__global__ void kern(const uint32_t* __restrict__ x,int m,const uint32_t* __restrict__ cp,const uint16_t* __restrict__ ri,const int8_t* __restrict__ cv,int n,uint32_t* __restrict__ y,uint32_t p,uint64_t mu){int r=(int)blockIdx.x*blockDim.x+threadIdx.x,j=blockIdx.y;if(r>=m||j>=n)return;uint64_t pos=0,neg=0;for(uint32_t e=cp[j];e<cp[j+1];++e){int c=cv[e];uint64_t v=x[(size_t)ri[e]*m+r];if(c>0)pos+=v*(unsigned)c;else neg+=v*(unsigned)(-c);}uint32_t a=red(pos,p,mu),b=red(neg,p,mu);y[(size_t)j*m+r]=a>=b?a-b:a+p-b;}
static float em(cudaEvent_t a,cudaEvent_t b){float x;ck(cudaEventElapsedTime(&x,a,b),"el");return x;}
int main(int ac,char**av){int m=ac>1?atoi(av[1]):4096,reps=ac>2?atoi(av[2]):10,a=ac>3?atoi(av[3]):0,h=ac>4?atoi(av[4]):1;uint32_t p=ac>5?(uint32_t)std::stoull(av[5]):4294967291u;auto B=loadB(a,h);auto C=toc(B);std::map<int,size_t>hist;for(auto c:C.cv)++hist[c];std::cout<<"m="<<m<<" a="<<a<<" h="<<h<<" k="<<B.k<<" n="<<B.n<<" nnz="<<C.ri.size()<<" coeff";for(auto[x,c]:hist)std::cout<<' '<<x<<':'<<c;std::cout<<'\n';std::vector<uint32_t>hx((size_t)B.k*m);std::mt19937 rng(7);for(auto&v:hx)v=(uint64_t(rng())<<32|rng())%p;uint32_t*dx,*dy,*dcp;uint16_t*dri;int8_t*dcv;ck(cudaMalloc(&dx,hx.size()*4),"x");ck(cudaMalloc(&dy,(size_t)B.n*m*4),"y");ck(cudaMalloc(&dcp,C.cp.size()*4),"cp");ck(cudaMalloc(&dri,C.ri.size()*2),"ri");ck(cudaMalloc(&dcv,C.cv.size()),"cv");ck(cudaMemcpy(dx,hx.data(),hx.size()*4,cudaMemcpyHostToDevice),"xc");ck(cudaMemcpy(dcp,C.cp.data(),C.cp.size()*4,cudaMemcpyHostToDevice),"cpc");ck(cudaMemcpy(dri,C.ri.data(),C.ri.size()*2,cudaMemcpyHostToDevice),"ric");ck(cudaMemcpy(dcv,C.cv.data(),C.cv.size(),cudaMemcpyHostToDevice),"cvc");dim3 bl(256),gr((m+255)/256,B.n);uint64_t mu=(~uint64_t(0))/p;for(int q=0;q<3;++q)kern<<<gr,bl>>>(dx,m,dcp,dri,dcv,B.n,dy,p,mu);ck(cudaDeviceSynchronize(),"warm");cudaEvent_t e0,e1;cudaEventCreate(&e0);cudaEventCreate(&e1);cudaEventRecord(e0);for(int q=0;q<reps;++q)kern<<<gr,bl>>>(dx,m,dcp,dri,dcv,B.n,dy,p,mu);cudaEventRecord(e1);cudaEventSynchronize(e1);std::cout<<"structural2d_ms="<<em(e0,e1)/reps<<'\n';cudaFree(dx);cudaFree(dy);cudaFree(dcp);cudaFree(dri);cudaFree(dcv);cudaEventDestroy(e0);cudaEventDestroy(e1);}
