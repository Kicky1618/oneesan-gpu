#define RP_RUNTIME_OWNER_U32LIMB 1
#define RP_RUNTIME_OWNER_W28_NGPU8_DIRECT 1

#include "gridfp_reduced_production_group_context_device.cuh"

extern "C" __global__ void gridfp_runtime_owner_w28_ngpu8_direct_integration_probe(
    const oneesan::gridfp::reducedprod::Rank64* midpoint,
    int* owner,
    int n
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n) return;
    using namespace oneesan::gridfp::reducedprod;
    owner[tid] = runtime_owner_from_group_base_device(
        midpoint[tid], 0, 28, 13, 8, nullptr);
}
