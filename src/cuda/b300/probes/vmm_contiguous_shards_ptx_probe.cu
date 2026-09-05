#include <cuda_runtime.h>
#include <cstdint>

using Count=std::uint32_t;
using Code=unsigned long long;
static constexpr int MAXGPU=8;
__constant__ Count* D_VMM_PTX_PTR[MAXGPU];
__constant__ Count* D_VMM_PTX_BASE;
__constant__ Code D_VMM_PTX_CHUNK;

__device__ __forceinline__ Count old_shard_load(Code g){int o=0;Code chunk=D_VMM_PTX_CHUNK,c4=chunk<<2;if(g>=c4){g-=c4;o|=4;}Code c2=chunk<<1;if(g>=c2){g-=c2;o|=2;}if(g>=chunk){g-=chunk;o|=1;}return D_VMM_PTX_PTR[o][g];}
__device__ __forceinline__ Count vmm_shard_load(Code g){return D_VMM_PTX_BASE[g];}

extern "C" __global__ void b300_vmm_old_address_probe(const Code* g,Count* out){int i=int(blockIdx.x*blockDim.x+threadIdx.x);out[i]=old_shard_load(g[i]);}
extern "C" __global__ void b300_vmm_direct_address_probe(const Code* g,Count* out){int i=int(blockIdx.x*blockDim.x+threadIdx.x);out[i]=vmm_shard_load(g[i]);}
