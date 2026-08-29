#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));std::exit(2);}}

__device__ __forceinline__ void cpasync_u32(uint32_t* dst,const uint32_t* src,bool valid){
#if __CUDA_ARCH__ >= 800
    const uint32_t smem=static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    const unsigned long long gmem=reinterpret_cast<unsigned long long>(src);
    const uint32_t bytes=valid?4u:0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 4, %2;" :: "r"(smem),"l"(gmem),"r"(bytes));
#else
    *dst=valid?*src:0u;
#endif
}

__global__ void probe(const uint32_t* in,const uint32_t* idx,const uint8_t* valid,uint32_t* out,int n){
    extern __shared__ uint32_t sm[];
    const int t=threadIdx.x;
    const int i=blockIdx.x*blockDim.x+t;
    if(i>=n)return;
    uint32_t* a=&sm[t];
    uint32_t* b=&sm[blockDim.x+t];
    const uint32_t ia=idx[2*i],ib=idx[2*i+1];
    const bool va=valid[2*i]!=0,vb=valid[2*i+1]!=0;
    cpasync_u32(a,in+ia,va);
    cpasync_u32(b,in+ib,vb);
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;");
#endif
    uint32_t independent=uint32_t(i)*2654435761u+17u;
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
    const uint32_t x=*a,y=*b;
    out[i]=x+y+independent;
}

int main(){
    constexpr int N=1<<20, M=1<<18, T=256;
    std::vector<uint32_t> h(M),idx(2*N),ref(N),got(N);
    std::vector<uint8_t> valid(2*N);
    uint64_t s=0x123456789abcdef0ULL;
    auto rng=[&](){s^=s<<7;s^=s>>9;s^=s<<8;return s;};
    for(int i=0;i<M;++i)h[i]=uint32_t(rng());
    uint64_t zeros=0,loads=0;
    for(int i=0;i<N;++i){
        uint64_t sum=uint32_t(i)*2654435761u+17u;
        for(int q=0;q<2;++q){int k=2*i+q;idx[k]=uint32_t(rng()%M);valid[k]=uint8_t((rng()&7)!=0);if(valid[k]){sum+=h[idx[k]];++loads;}else ++zeros;}
        ref[i]=uint32_t(sum);
    }
    uint32_t *di=nullptr,*dx=nullptr,*dout=nullptr;uint8_t*dv=nullptr;
    ck(cudaMalloc(&di,size_t(M)*4),"malloc in");ck(cudaMalloc(&dx,size_t(2*N)*4),"malloc idx");
    ck(cudaMalloc(&dv,size_t(2*N)),"malloc valid");ck(cudaMalloc(&dout,size_t(N)*4),"malloc out");
    ck(cudaMemcpy(di,h.data(),size_t(M)*4,cudaMemcpyHostToDevice),"copy in");
    ck(cudaMemcpy(dx,idx.data(),size_t(2*N)*4,cudaMemcpyHostToDevice),"copy idx");
    ck(cudaMemcpy(dv,valid.data(),size_t(2*N),cudaMemcpyHostToDevice),"copy valid");
    probe<<<(N+T-1)/T,T,size_t(T)*2*4>>>(di,dx,dv,dout,N);ck(cudaGetLastError(),"launch");ck(cudaDeviceSynchronize(),"sync");
    ck(cudaMemcpy(got.data(),dout,size_t(N)*4,cudaMemcpyDeviceToHost),"copy out");
    for(int i=0;i<N;++i)if(got[i]!=ref[i]){std::fprintf(stderr,"mismatch i=%d got=%u ref=%u\n",i,got[i],ref[i]);return 3;}
    cudaDeviceProp p{};ck(cudaGetDeviceProperties(&p,0),"props");
    std::printf("b300-cpasync-gather-proof OK device=%s cc=%d.%d cases=%d loads=%llu zero_fill=%llu cp_bytes=4 exact=1\n",p.name,p.major,p.minor,N,(unsigned long long)loads,(unsigned long long)zeros);
    return 0;
}
