#pragma push_macro("main")
#undef main
#define main gridfp_b300_fullmate_dropn_main_unused
#include "../oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
#pragma pop_macro("main")

static constexpr Code B300_MAIN_TOTAL_W28 = 385719506620ULL;
static constexpr Code B300_MAIN_CHUNK_W28_G8 = 48214938328ULL;
static constexpr Code B300_MAIN_MAGIC_W28_G8 = 195888106327ULL;
static constexpr unsigned B300_MAIN_HIGH_SHIFT_W28_G8 = 9;

__device__ __constant__ Code B300_MAIN_SHARD_BASE_W28_G8[8] = {
    0ULL,
    48214938328ULL,
    96429876656ULL,
    144644814984ULL,
    192859753312ULL,
    241074691640ULL,
    289289629968ULL,
    337504568296ULL,
};

__device__ __forceinline__ ShardAddress8 shard_address8_mulhi_mul(Code g) {
    const int owner = int(__umul64hi(g, B300_MAIN_MAGIC_W28_G8) >>
                          B300_MAIN_HIGH_SHIFT_W28_G8);
    return {owner, g - Code(owner) * B300_MAIN_CHUNK_W28_G8};
}

__device__ __forceinline__ ShardAddress8 shard_address8_mulhi_table(Code g) {
    const int owner = int(__umul64hi(g, B300_MAIN_MAGIC_W28_G8) >>
                          B300_MAIN_HIGH_SHIFT_W28_G8);
    return {owner, g - B300_MAIN_SHARD_BASE_W28_G8[owner]};
}

extern "C" __global__ void gridfp_b300_shard_owner_branchy_probe(
    const Code* global_index, int* owner, Code* local
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    const ShardAddress8 a = shard_address8(
        global_index[tid], B300_MAIN_CHUNK_W28_G8);
    owner[tid] = a.owner;
    local[tid] = a.local;
}

extern "C" __global__ void gridfp_b300_shard_owner_mulhi_mul_probe(
    const Code* global_index, int* owner, Code* local
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    const ShardAddress8 a = shard_address8_mulhi_mul(global_index[tid]);
    owner[tid] = a.owner;
    local[tid] = a.local;
}

extern "C" __global__ void gridfp_b300_shard_owner_mulhi_table_probe(
    const Code* global_index, int* owner, Code* local
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    const ShardAddress8 a = shard_address8_mulhi_table(global_index[tid]);
    owner[tid] = a.owner;
    local[tid] = a.local;
}
