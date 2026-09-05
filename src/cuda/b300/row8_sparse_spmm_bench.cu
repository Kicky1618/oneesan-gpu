#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Count=uint32_t; using Code=unsigned long long;
static unsigned long long H_DP[64][64]{};
static constexpr int TARGET_W=19;
static inline void ck(cudaError_t e,const char*w){if(e!=cudaSuccess)throw std::runtime_error(std::string(w)+": "+cudaGetErrorString(e));}
#include "row8_tensor_init.cuh"

struct SHdr {char magic[8];uint32_t version,r;uint32_t dims[9];uint64_t total_nz,fnv_hash;};
struct BHdr {uint32_t sym,h,h2,rows,cols,nnz;}; struct VHdr {uint32_t tag,sym,h,nnz;};
struct Block {int rows=0,cols=0;std::vector<uint32_t>rp;std::vector<uint16_t>ci;std::vector<int8_t>cv;};
template<class T>static T rd(std::ifstream&in){T x{};in.read((char*)&x,sizeof(x));if(!in)throw std::runtime_error("short");return x;}
static Block loadB(int wantA,int wantH){std::ifstream in("work/row8_structural_cache/row8_structural_int_v1.bin",std::ios::binary);auto sh=rd<SHdr>(in);for(int q=0;q<5;++q){auto v=rd<VHdr>(in);in.seekg((std::streamoff)v.nnz*3,std::ios::cur);}for(int q=0;q<25;++q){auto b=rd<BHdr>(in);Block z;z.rows=b.rows;z.cols=b.cols;z.rp.resize(b.rows+1);z.ci.resize(b.nnz);z.cv.resize(b.nnz);in.read((char*)z.rp.data(),z.rp.size()*4);in.read((char*)z.ci.data(),z.ci.size()*2);in.read((char*)z.cv.data(),z.cv.size());if((int)b.sym==wantA&&(int)b.h==wantH)return z;}throw std::runtime_error("block missing");}

__global__ void spmm_csc(const uint32_t*x,int m,int k,const uint32_t*cp,const uint16_t*ri,const int8_t*cv,int n,uint32_t*y,uint32_t mod){
  for(unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,st=(unsigned long long)gridDim.x*blockDim.x;q<(unsigned long long)m*n;q+=st){int j=q%n,r=q/n;long long z=0;for(uint32_t e=cp[j];e<cp[j+1];++e)z+=(long long)x[(size_t)r*k+ri[e]]*(long long)cv[e];long long t=z%(long long)mod;if(t<0)t+=mod;y[q]=(uint32_t)t;}
}
__global__ void spmm_csc_soa(const uint32_t*x,int m,const uint32_t*cp,const uint16_t*ri,const int8_t*cv,int n,uint32_t*y,uint32_t mod){
  for(unsigned long long q=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x,st=(unsigned long long)gridDim.x*blockDim.x;q<(unsigned long long)m*n;q+=st){int j=q/m,r=q-j*(unsigned long long)m;long long z=0;for(uint32_t e=cp[j];e<cp[j+1];++e)z+=(long long)x[(size_t)ri[e]*m+r]*(long long)cv[e];long long t=z%(long long)mod;if(t<0)t+=mod;y[(size_t)j*m+r]=(uint32_t)t;}
}
static float elapsed(cudaEvent_t a,cudaEvent_t b){float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");return ms;}
int main(int ac,char**av){int m=ac>1?std::atoi(av[1]):4096,reps=ac>2?std::atoi(av[2]):20,a=ac>3?std::atoi(av[3]):0,hh=ac>4?std::atoi(av[4]):1;uint32_t mod=1000000007u;auto B=loadB(a,hh);int k=B.rows,n=B.cols;std::cout<<"m="<<m<<" a="<<a<<" h="<<hh<<" k="<<k<<" n="<<n<<" nz="<<B.ci.size()<<" density="<<double(B.ci.size())/(k*n)<<"\n";
  std::vector<uint32_t>dense((size_t)k*n),hX((size_t)m*k);for(int i=0;i<k;++i)for(uint32_t e=B.rp[i];e<B.rp[i+1];++e){int v=B.cv[e];dense[(size_t)i*n+B.ci[e]]=v>=0?(uint32_t)v:mod-(uint32_t)(-v);}std::mt19937 rng(7);for(auto&x:hX)x=rng()%mod;
  // CSR -> CSC
  std::vector<uint32_t>cp(n+1);for(auto j:B.ci)++cp[j+1];for(int j=0;j<n;++j)cp[j+1]+=cp[j];std::vector<uint32_t>cur=cp;std::vector<uint16_t>ri(B.ci.size());std::vector<int8_t>cv(B.cv.size());for(int i=0;i<k;++i)for(uint32_t e=B.rp[i];e<B.rp[i+1];++e){auto j=B.ci[e];auto p=cur[j]++;ri[p]=i;cv[p]=B.cv[e];}
  std::vector<uint32_t>hXT((size_t)k*m);for(int r=0;r<m;++r)for(int i=0;i<k;++i)hXT[(size_t)i*m+r]=hX[(size_t)r*k+i];uint32_t *dX=nullptr,*dXT=nullptr,*dYS=nullptr,*dYST=nullptr,*dYD=nullptr,*dcp=nullptr;uint16_t*dri=nullptr;int8_t*dcv=nullptr;ck(cudaMalloc(&dX,hX.size()*4),"x");ck(cudaMalloc(&dXT,hXT.size()*4),"xt");ck(cudaMalloc(&dYS,(size_t)m*n*4),"ys");ck(cudaMalloc(&dYST,(size_t)m*n*4),"yst");ck(cudaMalloc(&dYD,(size_t)m*n*4),"yd");ck(cudaMalloc(&dcp,cp.size()*4),"cp");ck(cudaMalloc(&dri,ri.size()*2),"ri");ck(cudaMalloc(&dcv,cv.size()),"cv");ck(cudaMemcpy(dX,hX.data(),hX.size()*4,cudaMemcpyHostToDevice),"xcopy");ck(cudaMemcpy(dXT,hXT.data(),hXT.size()*4,cudaMemcpyHostToDevice),"xtcopy");ck(cudaMemcpy(dcp,cp.data(),cp.size()*4,cudaMemcpyHostToDevice),"cpcopy");ck(cudaMemcpy(dri,ri.data(),ri.size()*2,cudaMemcpyHostToDevice),"ricopy");ck(cudaMemcpy(dcv,cv.data(),cv.size(),cudaMemcpyHostToDevice),"cvcopy");
  int blocks=std::min<unsigned long long>(65535,((unsigned long long)m*n+255)/256);for(int q=0;q<3;++q)spmm_csc<<<blocks,256>>>(dX,m,k,dcp,dri,dcv,n,dYS,mod);ck(cudaDeviceSynchronize(),"sp warm");cudaEvent_t e0,e1;ck(cudaEventCreate(&e0),"e0");ck(cudaEventCreate(&e1),"e1");ck(cudaEventRecord(e0),"r0");for(int q=0;q<reps;++q)spmm_csc<<<blocks,256>>>(dX,m,k,dcp,dri,dcv,n,dYS,mod);ck(cudaEventRecord(e1),"r1");ck(cudaEventSynchronize(e1),"sync");float sp=elapsed(e0,e1)/reps;for(int q=0;q<3;++q)spmm_csc_soa<<<blocks,256>>>(dXT,m,dcp,dri,dcv,n,dYST,mod);ck(cudaDeviceSynchronize(),"soa warm");ck(cudaEventRecord(e0),"s0");for(int q=0;q<reps;++q)spmm_csc_soa<<<blocks,256>>>(dXT,m,dcp,dri,dcv,n,dYST,mod);ck(cudaEventRecord(e1),"s1");ck(cudaEventSynchronize(e1),"ss");float soa=elapsed(e0,e1)/reps;
  oneesan::row8tensor::HostMat hm;hm.k=k;hm.n=n;hm.a=std::move(dense);auto dm=oneesan::row8tensor::upload_mat(hm);oneesan::row8tensor::Group g;g.h=hh;g.rows=m;g.dim=k;g.d=dX;oneesan::row8tensor::Scratch sc;cublasHandle_t ch=nullptr;oneesan::row8tensor::cublas_ck(cublasCreate(&ch),"create");oneesan::row8tensor::cublas_ck(cublasSetMathMode(ch,CUBLAS_TENSOR_OP_MATH),"math");
  auto timePack=[&](){ck(cudaEventRecord(e0),"p0");oneesan::row8tensor::prepare_a(sc,g,dm.kp);ck(cudaEventRecord(e1),"p1");ck(cudaEventSynchronize(e1),"ps");return elapsed(e0,e1);};float packms=0;for(int q=0;q<5;++q)packms+=timePack();packms/=5;
  oneesan::row8tensor::prepare_a(sc,g,dm.kp);for(int q=0;q<2;++q)oneesan::row8tensor::gemm_level_rows(ch,sc,m,dm,dYD,0,mod);ck(cudaDeviceSynchronize(),"dense warm");ck(cudaEventRecord(e0),"d0");for(int q=0;q<reps;++q)oneesan::row8tensor::gemm_level_rows(ch,sc,m,dm,dYD,0,mod);ck(cudaEventRecord(e1),"d1");ck(cudaEventSynchronize(e1),"ds");float den=elapsed(e0,e1)/reps;
  std::vector<uint32_t>ys((size_t)m*n),yst((size_t)m*n),yd((size_t)m*n);ck(cudaMemcpy(ys.data(),dYS,ys.size()*4,cudaMemcpyDeviceToHost),"yscopy");ck(cudaMemcpy(yst.data(),dYST,yst.size()*4,cudaMemcpyDeviceToHost),"ystcopy");ck(cudaMemcpy(yd.data(),dYD,yd.size()*4,cudaMemcpyDeviceToHost),"ydcopy");size_t bad=0;for(int r=0;r<m;++r)for(int j=0;j<n;++j)if(yst[(size_t)j*m+r]!=yd[(size_t)r*n+j]){if(bad<4)std::cerr<<"bad soa r="<<r<<" j="<<j<<" s="<<yst[(size_t)j*m+r]<<" d="<<yd[(size_t)r*n+j]<<"\n";++bad;}for(size_t i=0;i<ys.size();++i)if(ys[i]!=yd[i]){if(bad<4)std::cerr<<"bad i="<<i<<" s="<<ys[i]<<" d="<<yd[i]<<"\n";++bad;}
  std::cout<<"sparse_ms="<<sp<<" sparse_soa_ms="<<soa<<" dense_gemm_ms="<<den<<" pack_ms="<<packms<<" dense_plus_pack_ms="<<den+packms<<" speedup_vs_gemm="<<den/sp<<" speedup_vs_pack="<<(den+packms)/sp<<" bad="<<bad<<"\n";
  cublasDestroy(ch);oneesan::row8tensor::free_mat(dm);cudaFree(dX);cudaFree(dXT);cudaFree(dYS);cudaFree(dYST);cudaFree(dYD);cudaFree(dcp);cudaFree(dri);cudaFree(dcv);cudaEventDestroy(e0);cudaEventDestroy(e1);return bad?1:0;
}
