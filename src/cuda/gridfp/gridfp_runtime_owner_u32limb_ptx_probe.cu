#define RP_RUNTIME_OWNER_U32LIMB 1
#define RP_RUNTIME_OWNER_FIXED52 0
#define RP_RUNTIME_OWNER_FIXED54 0
#define RP_RUNTIME_OWNER_RECIPROCAL 0
#define RP_RUNTIME_OWNER_FROM_BOUNDARIES 0

#include "gridfp_reduced_production_group_context_device.cuh"

using namespace oneesan::gridfp::reducedprod;

extern "C" __global__ void gridfp_runtime_owner_u32limb_ptx_probe(
    Rank64* out,
    Rank64 group_base,
    Rank64 group,
    int W,
    int K,
    int ngpu
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        out[0] = static_cast<Rank64>(runtime_owner_from_group_base_device(
            group_base, group, W, K, ngpu, nullptr));
    }
}
