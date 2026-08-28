#include <cuda_runtime.h>
#include <cstdint>

using Count=std::uint32_t;
using Code=unsigned long long;

__constant__ Count* D_VMM_BASE_SOURCE_PROBE;

extern "C" __global__ void b300_vmm_symbol_base_probe(Count* out,Code g){
    if(threadIdx.x==0&&blockIdx.x==0)out[0]=D_VMM_BASE_SOURCE_PROBE[g];
}

extern "C" __global__ void b300_vmm_arg_base_probe(Count* out,const Count* base,Code g){
    if(threadIdx.x==0&&blockIdx.x==0)out[0]=base[g];
}
