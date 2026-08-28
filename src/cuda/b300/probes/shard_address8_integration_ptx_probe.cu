#pragma push_macro("main")
#undef main
#define main gridfp_b300_fullmate_dropn_main_unused
#include "../oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
#pragma pop_macro("main")

extern "C" __global__ void gridfp_b300_shard_address8_integration_probe(
    const Code* global_index,
    Count* out,
    int n
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n) return;
    out[tid] = global_load_main(global_index[tid]);
}
