#pragma push_macro("main")
#undef main
#define main gridfp_b300_fullmate_dropn_main_unused
#include "../oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
#pragma pop_macro("main")

extern "C" __global__ void gridfp_b300_shard_address8_helper_probe(
    const Code* global_index,
    Code chunk,
    int* owner,
    Code* local
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    const ShardAddress8 a = shard_address8(global_index[tid], chunk);
    owner[tid] = a.owner;
    local[tid] = a.local;
}
