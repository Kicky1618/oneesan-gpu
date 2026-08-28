#pragma push_macro("main")
#undef main
#define main gridfp_b300_fullmate_dropn_main_unused
#include "../oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
#pragma pop_macro("main")

__device__ __forceinline__ ShardAddress8 shard_address8_select(Code g, Code chunk) {
    int owner = 0;
    const Code c4 = chunk << 2;
    const bool p4 = g >= c4;
    const Code s4 = g - c4;
    g = p4 ? s4 : g;
    owner |= int(p4) << 2;

    const Code c2 = chunk << 1;
    const bool p2 = g >= c2;
    const Code s2 = g - c2;
    g = p2 ? s2 : g;
    owner |= int(p2) << 1;

    const bool p1 = g >= chunk;
    const Code s1 = g - chunk;
    g = p1 ? s1 : g;
    owner |= int(p1);
    return {owner, g};
}

extern "C" __global__ void gridfp_b300_shard_address8_branchy_probe(
    const Code* global_index, Code chunk, int* owner, Code* local
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    const ShardAddress8 a = shard_address8(global_index[tid], chunk);
    owner[tid] = a.owner;
    local[tid] = a.local;
}

extern "C" __global__ void gridfp_b300_shard_address8_select_probe(
    const Code* global_index, Code chunk, int* owner, Code* local
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    const ShardAddress8 a = shard_address8_select(global_index[tid], chunk);
    owner[tid] = a.owner;
    local[tid] = a.local;
}
