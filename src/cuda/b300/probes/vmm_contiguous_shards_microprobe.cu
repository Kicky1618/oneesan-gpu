#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

using Count=std::uint32_t;
using Code=unsigned long long;
static constexpr int MAXGPU=8;

__constant__ Count* D_VMM_PTR[MAXGPU];
__constant__ Count* D_VMM_BASE;
__constant__ Code D_VMM_CHUNK;

static void ck(cudaError_t e,const char* w){if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));std::exit(1);}}
static void cuck(CUresult e,const char* w){if(e!=CUDA_SUCCESS){const char* s=nullptr;cuGetErrorString(e,&s);std::fprintf(stderr,"%s: %s (%d)\n",w,s?s:"CUerror",int(e));std::exit(1);}}

__host__ __device__ __forceinline__ Count value_for(Code g){return Count((g*2654435761ULL+17ULL)&0xffffffffu);}

__global__ void fill_segment(Count* base,Code begin,Code n){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride)base[begin+i]=value_for(begin+i);}

__device__ __forceinline__ Count load_old(Code g){int o=0;Code chunk=D_VMM_CHUNK,c4=chunk<<2;if(g>=c4){g-=c4;o|=4;}Code c2=chunk<<1;if(g>=c2){g-=c2;o|=2;}if(g>=chunk){g-=chunk;o|=1;}return D_VMM_PTR[o][g];}
__device__ __forceinline__ Count load_vmm_symbol(Code g){return D_VMM_BASE[g];}
__device__ __forceinline__ Count load_vmm_arg(const Count* base,Code g){return base[g];}

template<int MODE>
__global__ void perf_kernel(Code* out,int n,int iters,Code total,Code stride,Code step,const Count* base){int tid=int(blockIdx.x*blockDim.x+threadIdx.x);if(tid>=n)return;Code g=Code(tid)*stride;if(g>=total)g=total-1;Code acc=0;for(int i=0;i<iters;++i){Count v;if constexpr(MODE==0)v=load_old(g);else if constexpr(MODE==1)v=load_vmm_symbol(g);else v=load_vmm_arg(base,g);acc+=Code(v)^(g<<7);g+=step;if(g>=total)g-=total;}out[tid]=acc;}

__global__ void exact_kernel(const Code* idx,Count* oldv,Count* symbolv,Count* argv,int n,const Count* base){int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i<n){Code g=idx[i];oldv[i]=load_old(g);symbolv[i]=load_vmm_symbol(g);argv[i]=load_vmm_arg(base,g);}}

template<int MODE>
float run(Code* out,int n,int blocks,int threads,int iters,Code total,Code stride,Code step,const Count* base){cudaEvent_t a{},b{};ck(cudaEventCreate(&a),"event a");ck(cudaEventCreate(&b),"event b");ck(cudaEventRecord(a),"record a");perf_kernel<MODE><<<blocks,threads>>>(out,n,iters,total,stride,step,base);ck(cudaGetLastError(),"perf launch");ck(cudaEventRecord(b),"record b");ck(cudaEventSynchronize(b),"event sync");float ms=0;ck(cudaEventElapsedTime(&ms,a,b),"elapsed");cudaEventDestroy(a);cudaEventDestroy(b);return ms;}

static double median(std::vector<float> x){std::sort(x.begin(),x.end());size_t n=x.size();return n&1?x[n/2]:0.5*(x[n/2-1]+x[n/2]);}
static size_t round_up(size_t x,size_t a){return (x+a-1)/a*a;}

int main(int argc,char**argv){
    const int requested=argc>1?std::atoi(argv[1]):8;
    const size_t requested_chunk_elems=argc>2?std::strtoull(argv[2],nullptr,10):(1u<<20);
    const int blocks=argc>3?std::atoi(argv[3]):256,threads=argc>4?std::atoi(argv[4]):256,iters=argc>5?std::atoi(argv[5]):4096,repeats=argc>6?std::atoi(argv[6]):9,srcdev=argc>7?std::atoi(argv[7]):0;
    if(requested<1||requested>MAXGPU||requested_chunk_elems<1024||blocks<1||threads<1||threads>1024||iters<1||repeats<1)return 2;
    cuck(cuInit(0),"cuInit");int visible=0;ck(cudaGetDeviceCount(&visible),"device count");int ng=std::min(requested,visible);if(srcdev<0||srcdev>=ng)return 2;
    for(int d=0;d<ng;++d){int vmm=0;CUdevice dev{};cuck(cuDeviceGet(&dev,d),"cuDeviceGet");cuck(cuDeviceGetAttribute(&vmm,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,dev),"vmm attr");if(!vmm){std::fprintf(stderr,"device %d lacks VMM\n",d);return 3;}ck(cudaSetDevice(d),"set init device");ck(cudaFree(nullptr),"init runtime context");}

    auto props=std::vector<CUmemAllocationProp>(static_cast<size_t>(ng));
    auto grans=std::vector<size_t>(static_cast<size_t>(ng));size_t gran=1;
    for(int d=0;d<ng;++d){auto&p=props[static_cast<size_t>(d)];p.type=CU_MEM_ALLOCATION_TYPE_PINNED;p.location.type=CU_MEM_LOCATION_TYPE_DEVICE;p.location.id=d;p.requestedHandleTypes=CU_MEM_HANDLE_TYPE_NONE;cuck(cuMemGetAllocationGranularity(&grans[static_cast<size_t>(d)],&p,CU_MEM_ALLOC_GRANULARITY_MINIMUM),"granularity");gran=std::lcm(gran,grans[static_cast<size_t>(d)]);}
    const size_t chunk_bytes=round_up(requested_chunk_elems*sizeof(Count),gran);const Code chunk=Code(chunk_bytes/sizeof(Count));const size_t total_bytes=chunk_bytes*static_cast<size_t>(ng);const Code total=chunk*Code(ng);
    CUdeviceptr va=0;cuck(cuMemAddressReserve(&va,total_bytes,gran,0,0),"address reserve");
    auto handles=std::vector<CUmemGenericAllocationHandle>(static_cast<size_t>(ng));
    for(int d=0;d<ng;++d){cuck(cuMemCreate(&handles[static_cast<size_t>(d)],chunk_bytes,&props[static_cast<size_t>(d)],0),"mem create");cuck(cuMemMap(va+CUdeviceptr(static_cast<size_t>(d)*chunk_bytes),chunk_bytes,0,handles[static_cast<size_t>(d)],0),"mem map");}
    auto access=std::vector<CUmemAccessDesc>(static_cast<size_t>(ng));for(int d=0;d<ng;++d){access[static_cast<size_t>(d)].location.type=CU_MEM_LOCATION_TYPE_DEVICE;access[static_cast<size_t>(d)].location.id=d;access[static_cast<size_t>(d)].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;}cuck(cuMemSetAccess(va,total_bytes,access.data(),access.size()),"set access");
    Count* base=reinterpret_cast<Count*>(va);Count* ptrs[MAXGPU]{};for(int d=0;d<ng;++d)ptrs[d]=base+Code(d)*chunk;
    for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"set fill");ck(cudaMemcpyToSymbol(D_VMM_PTR,ptrs,sizeof(ptrs)),"copy ptrs");ck(cudaMemcpyToSymbol(D_VMM_BASE,&base,sizeof(base)),"copy base");ck(cudaMemcpyToSymbol(D_VMM_CHUNK,&chunk,sizeof(chunk)),"copy chunk");fill_segment<<<256,256>>>(base,Code(d)*chunk,chunk);ck(cudaGetLastError(),"fill launch");}
    for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"set fill sync");ck(cudaDeviceSynchronize(),"fill sync");}

    ck(cudaSetDevice(srcdev),"set src");const int n=blocks*threads;Code *didx=nullptr,*dout=nullptr;Count *dold=nullptr,*dsymbol=nullptr,*darg=nullptr;ck(cudaMalloc(&didx,static_cast<size_t>(n)*sizeof(Code)),"idx alloc");ck(cudaMalloc(&dout,static_cast<size_t>(n)*sizeof(Code)),"out alloc");ck(cudaMalloc(&dold,static_cast<size_t>(n)*sizeof(Count)),"old alloc");ck(cudaMalloc(&dsymbol,static_cast<size_t>(n)*sizeof(Count)),"symbol alloc");ck(cudaMalloc(&darg,static_cast<size_t>(n)*sizeof(Count)),"arg alloc");
    auto hidx=std::vector<Code>(static_cast<size_t>(n));const Code stride=std::max<Code>(1,total/Code(n)),step=std::max<Code>(1,chunk/7+1);for(int i=0;i<n;++i)hidx[static_cast<size_t>(i)]=(Code(i)*stride)%total;ck(cudaMemcpy(didx,hidx.data(),static_cast<size_t>(n)*sizeof(Code),cudaMemcpyHostToDevice),"idx copy");exact_kernel<<<blocks,threads>>>(didx,dold,dsymbol,darg,n,base);ck(cudaGetLastError(),"exact launch");ck(cudaDeviceSynchronize(),"exact sync");
    auto hold=std::vector<Count>(static_cast<size_t>(n));auto hsymbol=std::vector<Count>(static_cast<size_t>(n));auto harg=std::vector<Count>(static_cast<size_t>(n));ck(cudaMemcpy(hold.data(),dold,static_cast<size_t>(n)*sizeof(Count),cudaMemcpyDeviceToHost),"old copy");ck(cudaMemcpy(hsymbol.data(),dsymbol,static_cast<size_t>(n)*sizeof(Count),cudaMemcpyDeviceToHost),"symbol copy");ck(cudaMemcpy(harg.data(),darg,static_cast<size_t>(n)*sizeof(Count),cudaMemcpyDeviceToHost),"arg copy");for(int i=0;i<n;++i){Count ex=value_for(hidx[static_cast<size_t>(i)]);if(hold[static_cast<size_t>(i)]!=ex||hsymbol[static_cast<size_t>(i)]!=ex||harg[static_cast<size_t>(i)]!=ex){std::fprintf(stderr,"exact mismatch i=%d\n",i);return 4;}}

    run<0>(dout,n,blocks,threads,iters,total,stride,step,base);run<1>(dout,n,blocks,threads,iters,total,stride,step,base);run<2>(dout,n,blocks,threads,iters,total,stride,step,base);
    std::vector<float>a,b,c;a.reserve(repeats);b.reserve(repeats);c.reserve(repeats);
    for(int r=0;r<repeats;++r){
        switch(r%3){
        case 0:a.push_back(run<0>(dout,n,blocks,threads,iters,total,stride,step,base));b.push_back(run<1>(dout,n,blocks,threads,iters,total,stride,step,base));c.push_back(run<2>(dout,n,blocks,threads,iters,total,stride,step,base));break;
        case 1:b.push_back(run<1>(dout,n,blocks,threads,iters,total,stride,step,base));c.push_back(run<2>(dout,n,blocks,threads,iters,total,stride,step,base));a.push_back(run<0>(dout,n,blocks,threads,iters,total,stride,step,base));break;
        default:c.push_back(run<2>(dout,n,blocks,threads,iters,total,stride,step,base));a.push_back(run<0>(dout,n,blocks,threads,iters,total,stride,step,base));b.push_back(run<1>(dout,n,blocks,threads,iters,total,stride,step,base));break;
        }
    }
    double oldms=median(a),symbolms=median(b),argms=median(c),ops=double(n)*iters;
    std::printf("gridfp-b300-vmm-contiguous-shards-microprobe OK gpus=%d src_gpu=%d chunk_elems=%llu chunk_bytes=%zu granularity=%zu total_mib=%.3f blocks=%d threads=%d iters=%d repeats=%d old_ms=%.6f vmm_ms=%.6f symbol_ms=%.6f arg_ms=%.6f speedup=%.6f arg_speedup_vs_symbol=%.6f old_Gload_s=%.6f vmm_Gload_s=%.6f symbol_Gload_s=%.6f arg_Gload_s=%.6f exact=OK owner_ops_vmm=0 dynamic_ptr_index_vmm=0 owner_ops_arg=0 dynamic_ptr_index_arg=0 base_source_symbol=constant base_source_arg=kernel_param\n",ng,srcdev,(unsigned long long)chunk,chunk_bytes,gran,double(total_bytes)/(1<<20),blocks,threads,iters,repeats,oldms,symbolms,symbolms,argms,oldms/symbolms,symbolms/argms,ops/oldms/1e6,ops/symbolms/1e6,ops/symbolms/1e6,ops/argms/1e6);
    cudaFree(darg);cudaFree(dsymbol);cudaFree(dold);cudaFree(dout);cudaFree(didx);for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"set final sync");ck(cudaDeviceSynchronize(),"final sync");}for(int d=0;d<ng;++d)cuck(cuMemUnmap(va+CUdeviceptr(static_cast<size_t>(d)*chunk_bytes),chunk_bytes),"unmap");for(auto h:handles)cuck(cuMemRelease(h),"release");cuck(cuMemAddressFree(va,total_bytes),"address free");return 0;
}
