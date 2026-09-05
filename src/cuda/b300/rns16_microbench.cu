#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <random>
#include <algorithm>

static inline void ck(cudaError_t e,const char*w){if(e!=cudaSuccess)throw std::runtime_error(std::string(w)+": "+cudaGetErrorString(e));}

__device__ __forceinline__ uint16_t add16(uint16_t a,uint16_t b,uint16_t p){
    uint32_t s=(uint32_t)a+(uint32_t)b; if(s>=p)s-=p; return (uint16_t)s;
}
__device__ __forceinline__ uint32_t add32(uint32_t a,uint32_t b,uint32_t p){return a>=p-b?a-(p-b):a+b;}

__device__ __forceinline__ uint32_t vadd_u16x2(uint32_t a,uint32_t b){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t r; asm("add.u16x2 %0, %1, %2;" : "=r"(r) : "r"(a),"r"(b)); return r;
#else
    return __vadd2(a,b);
#endif
}
__device__ __forceinline__ uint32_t vmin_u16x2(uint32_t a,uint32_t b){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    uint32_t r; asm("min.u16x2 %0, %1, %2;" : "=r"(r) : "r"(a),"r"(b)); return r;
#else
    return __vminu2(a,b);
#endif
}
__device__ __forceinline__ uint32_t add15x2(uint32_t a,uint32_t b,uint32_t dpack){
    uint32_t s=vadd_u16x2(a,b); uint32_t t=vadd_u16x2(s,dpack); return vmin_u16x2(s,t);
}

__device__ __forceinline__ void atomic_add16(uint16_t*p,uint16_t v,uint16_t mod){
    if(!v)return; uint16_t old=*p;
    for(;;){uint16_t neu=add16(old,v,mod);uint16_t got=atomicCAS((unsigned short*)p,old,neu);if(got==old)return;old=got;}
}
__device__ __forceinline__ void atomic_add32(uint32_t*p,uint32_t v,uint32_t mod){
    if(!v)return; cuda::atomic_ref<uint32_t,cuda::thread_scope_device> a(*p);
    uint32_t old=a.load(cuda::memory_order_relaxed);
    for(;;){uint32_t neu=add32(old,v,mod);if(a.compare_exchange_weak(old,neu,cuda::memory_order_relaxed,cuda::memory_order_relaxed))return;}
}

__global__ void pull3_16(const uint16_t*in,uint16_t*out,size_t n,uint16_t mod){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,st=(size_t)gridDim.x*blockDim.x;i<n;i+=st){
        size_t j1=(i*17+13)%n,j2=(i*257+31)%n;out[i]=add16(add16(in[i],in[j1],mod),in[j2],mod);
    }
}
__global__ void pull3_32(const uint32_t*in,uint32_t*out,size_t n,uint32_t mod){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,st=(size_t)gridDim.x*blockDim.x;i<n;i+=st){
        size_t j1=(i*17+13)%n,j2=(i*257+31)%n;out[i]=add32(add32(in[i],in[j1],mod),in[j2],mod);
    }
}
__global__ void pull3_15x2(const uint32_t*in,uint32_t*out,size_t n,uint32_t dpack){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,st=(size_t)gridDim.x*blockDim.x;i<n;i+=st){
        size_t j1=(i*17+13)%n,j2=(i*257+31)%n;out[i]=add15x2(add15x2(in[i],in[j1],dpack),in[j2],dpack);
    }
}
__global__ void atomic16_kernel(uint16_t*out,size_t mask,size_t ops,uint16_t mod){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,st=(size_t)gridDim.x*blockDim.x;i<ops;i+=st){size_t j=(i*2654435761u)&mask;atomic_add16(out+j,1,mod);}
}
__global__ void atomic32_kernel(uint32_t*out,size_t mask,size_t ops,uint32_t mod){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,st=(size_t)gridDim.x*blockDim.x;i<ops;i+=st){size_t j=(i*2654435761u)&mask;atomic_add32(out+j,1,mod);}
}

static float elapsed(cudaEvent_t a,cudaEvent_t b){float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");return ms;}
template<class F> static float bench(F f,int reps){cudaEvent_t a,b;ck(cudaEventCreate(&a),"ev");ck(cudaEventCreate(&b),"ev");f();ck(cudaDeviceSynchronize(),"warm");ck(cudaEventRecord(a),"rec");for(int i=0;i<reps;++i)f();ck(cudaEventRecord(b),"rec");ck(cudaEventSynchronize(b),"sync");float ms=elapsed(a,b)/reps;cudaEventDestroy(a);cudaEventDestroy(b);return ms;}

int main(int argc,char**argv){
    size_t n=argc>1?std::strtoull(argv[1],nullptr,10):(1ull<<24);int reps=argc>2?std::atoi(argv[2]):10;
    constexpr uint16_t P16=65521, P15A=32749, P15B=32719;constexpr uint32_t P32=4294967291u;
    constexpr uint32_t D15=(uint32_t(uint16_t(65536u-P15A)))|(uint32_t(uint16_t(65536u-P15B))<<16);
    std::vector<uint16_t> h16(n);std::vector<uint32_t> h32(n),hp(n);std::mt19937_64 rg(12345);
    for(size_t i=0;i<n;++i){uint64_t z=rg();h16[i]=z%P16;h32[i]=z%P32;uint16_t a=z%P15A,b=(z>>32)%P15B;hp[i]=uint32_t(a)|(uint32_t(b)<<16);}
    uint16_t *d16i=nullptr,*d16o=nullptr;uint32_t *d32i=nullptr,*d32o=nullptr,*dpi=nullptr,*dpo=nullptr;
    ck(cudaMalloc(&d16i,n*2),"d16i");ck(cudaMalloc(&d16o,n*2),"d16o");ck(cudaMalloc(&d32i,n*4),"d32i");ck(cudaMalloc(&d32o,n*4),"d32o");ck(cudaMalloc(&dpi,n*4),"dpi");ck(cudaMalloc(&dpo,n*4),"dpo");
    ck(cudaMemcpy(d16i,h16.data(),n*2,cudaMemcpyHostToDevice),"copy16");ck(cudaMemcpy(d32i,h32.data(),n*4,cudaMemcpyHostToDevice),"copy32");ck(cudaMemcpy(dpi,hp.data(),n*4,cudaMemcpyHostToDevice),"copyp");
    int th=256,bl=std::min<size_t>(65535,(n+th-1)/th);
    float m16=bench([&]{pull3_16<<<bl,th>>>(d16i,d16o,n,P16);},reps);
    float m32=bench([&]{pull3_32<<<bl,th>>>(d32i,d32o,n,P32);},reps);
    float mp=bench([&]{pull3_15x2<<<bl,th>>>(dpi,dpo,n,D15);},reps);
    ck(cudaGetLastError(),"pull launches");
    std::vector<uint16_t> o16(std::min<size_t>(n,4096));std::vector<uint32_t> o32(o16.size()),op(o16.size());
    ck(cudaMemcpy(o16.data(),d16o,o16.size()*2,cudaMemcpyDeviceToHost),"o16");ck(cudaMemcpy(o32.data(),d32o,o32.size()*4,cudaMemcpyDeviceToHost),"o32");ck(cudaMemcpy(op.data(),dpo,op.size()*4,cudaMemcpyDeviceToHost),"op");
    size_t bad=0;for(size_t i=0;i<o16.size();++i){size_t j1=(i*17+13)%n,j2=(i*257+31)%n;uint16_t e16=(uint32_t(h16[i])+h16[j1]+h16[j2])%P16;if(o16[i]!=e16)++bad;uint64_t e32=(uint64_t(h32[i])+h32[j1]+h32[j2])%P32;if(o32[i]!=e32)++bad;uint16_t a0=hp[i],a1=hp[j1],a2=hp[j2];uint16_t b0=hp[i]>>16,b1=hp[j1]>>16,b2=hp[j2]>>16;uint32_t ep=uint32_t((uint32_t(a0)+a1+a2)%P15A)|(uint32_t((uint32_t(b0)+b1+b2)%P15B)<<16);if(op[i]!=ep)++bad;}
    size_t an=1ull<<20,ops=std::min<size_t>(n,1ull<<24);uint16_t*da16=nullptr;uint32_t*da32=nullptr;ck(cudaMalloc(&da16,an*2),"a16");ck(cudaMalloc(&da32,an*4),"a32");
    auto f16=[&]{ck(cudaMemset(da16,0,an*2),"z16");atomic16_kernel<<<std::min<size_t>(65535,(ops+255)/256),256>>>(da16,an-1,ops,P16);};
    auto f32=[&]{ck(cudaMemset(da32,0,an*4),"z32");atomic32_kernel<<<std::min<size_t>(65535,(ops+255)/256),256>>>(da32,an-1,ops,P32);};
    float a16ms=bench(f16,std::max(2,reps/2)),a32ms=bench(f32,std::max(2,reps/2));
    std::cout<<"n="<<n<<" reps="<<reps<<" bad="<<bad<<"\n";
    std::cout<<"pull3_u16_ms="<<m16<<" states_Gs="<<(double(n)/m16/1e6)<<" logical_GBps="<<(double(n)*8/m16/1e6)<<"\n";
    std::cout<<"pull3_u32_ms="<<m32<<" states_Gs="<<(double(n)/m32/1e6)<<" logical_GBps="<<(double(n)*16/m32/1e6)<<" speedup_states_u16="<<m32/m16<<"\n";
    std::cout<<"pull3_15x2_ms="<<mp<<" states_Gs="<<(double(n)/mp/1e6)<<" pair_residue_Gs="<<(2.0*n/mp/1e6)<<" speedup_vs_u32_state="<<m32/mp<<"\n";
    std::cout<<"atomic_u16_ms="<<a16ms<<" atomic_u32_ms="<<a32ms<<" speedup_u16="<<a32ms/a16ms<<" ops_Ms16="<<(double(ops)/a16ms/1e3)<<"\n";
    cudaFree(d16i);cudaFree(d16o);cudaFree(d32i);cudaFree(d32o);cudaFree(dpi);cudaFree(dpo);cudaFree(da16);cudaFree(da32);return bad?1:0;
}
