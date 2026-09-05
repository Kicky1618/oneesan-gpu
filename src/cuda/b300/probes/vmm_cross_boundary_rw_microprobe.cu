#include "../b300_vmm_contiguous_storage.cuh"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

using Count=std::uint32_t;
using Code=unsigned long long;

__device__ __forceinline__ Count value_for(Code g){return Count((g*2654435761ULL+0x85ebca6bULL)&0xffffffffu);}
__global__ void fill_range(Count*base,Code begin,Code end){Code i=begin+Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<end;i+=stride)base[i]=value_for(i);}
__global__ void xor_range(Count*base,Code begin,Code n,Count mask){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;for(;i<n;i+=stride)base[begin+i]^=mask;}
__global__ void verify_range(const Count*base,Code begin,Code n,Count mask,unsigned long long*err){Code i=Code(blockIdx.x)*blockDim.x+threadIdx.x,stride=Code(gridDim.x)*blockDim.x;unsigned long long e=0;for(;i<n;i+=stride)e+=base[begin+i]!=(value_for(begin+i)^mask);if(e)atomicAdd(err,e);}

static void ck(cudaError_t e,const char*w){if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",w,cudaGetErrorString(e));std::exit(1);}}

int main(int argc,char**argv){
    const int ng=argc>1?std::atoi(argv[1]):8;
    const Code elems=argc>2?std::strtoull(argv[2],nullptr,10):8388731ULL;
    const Code span=argc>3?std::strtoull(argv[3],nullptr,10):65536ULL;
    if(ng<2||ng>8||elems<Code(ng)*span*2||span<256)return 2;
    int visible=0;ck(cudaGetDeviceCount(&visible),"device count");if(visible<ng)return 2;

    b300_vmm::ContiguousStorage store;store.create(elems,ng,0,"cross-rw");store.zero_local_segments();Count*base=store.base_as<Count>();
    for(int d=0;d<ng;++d){const Code lo=Code(store.offsets[size_t(d)]/sizeof(Count));const Code hi=std::min<Code>(elems,Code(store.offsets[size_t(d)+1]/sizeof(Count)));if(lo>=hi)continue;ck(cudaSetDevice(d),"fill set");fill_range<<<256,256>>>(base,lo,hi);ck(cudaGetLastError(),"fill launch");}
    for(int d=0;d<ng;++d){ck(cudaSetDevice(d),"fill sync set");ck(cudaDeviceSynchronize(),"fill sync");}

    const int boundary_index=ng/2;const Code boundary=Code(store.offsets[size_t(boundary_index)]/sizeof(Count));
    if(boundary<span/2||boundary+span/2>=elems){std::fprintf(stderr,"boundary range unavailable\n");return 3;}
    const Code begin=boundary-span/2;
    for(int d=0;d<ng;++d){
        ck(cudaSetDevice(d),"rw set device");
        const Count mask=Count(0x9e3779b9u^unsigned(d*0x10203u+1));
        xor_range<<<256,256>>>(base,begin,span,mask);ck(cudaGetLastError(),"cross-boundary xor launch");ck(cudaDeviceSynchronize(),"cross-boundary xor sync");
        unsigned long long*derr=nullptr;ck(cudaMalloc(&derr,sizeof(*derr)),"err alloc");ck(cudaMemset(derr,0,sizeof(*derr)),"err zero");verify_range<<<256,256>>>(base,begin,span,mask,derr);ck(cudaGetLastError(),"cross-boundary verify launch");ck(cudaDeviceSynchronize(),"cross-boundary verify sync");unsigned long long e=0;ck(cudaMemcpy(&e,derr,sizeof(e),cudaMemcpyDeviceToHost),"err copy");cudaFree(derr);if(e){std::fprintf(stderr,"device %d cross-boundary verify errors=%llu\n",d,e);return 4;}
        xor_range<<<256,256>>>(base,begin,span,mask);ck(cudaGetLastError(),"restore xor launch");ck(cudaDeviceSynchronize(),"restore xor sync");
    }

    ck(cudaSetDevice(0),"final verify set");unsigned long long*derr=nullptr;ck(cudaMalloc(&derr,sizeof(*derr)),"final err alloc");ck(cudaMemset(derr,0,sizeof(*derr)),"final err zero");verify_range<<<256,256>>>(base,begin,span,0,derr);ck(cudaGetLastError(),"final verify launch");ck(cudaDeviceSynchronize(),"final verify sync");unsigned long long e=0;ck(cudaMemcpy(&e,derr,sizeof(e),cudaMemcpyDeviceToHost),"final err copy");cudaFree(derr);if(e)return 5;

    std::printf("gridfp-b300-vmm-cross-boundary-rw-microprobe OK gpus=%d elems=%llu granularity=%zu boundary_index=%d boundary_elem=%llu span=%llu all_gpu_write=OK all_gpu_read=OK restore=OK shard_owner_ops=0 cross_physical_boundary_rw_all_gpu=OK exact=OK\n",ng,(unsigned long long)elems,store.granularity,boundary_index,(unsigned long long)boundary,(unsigned long long)span);
    store.destroy();return 0;
}
