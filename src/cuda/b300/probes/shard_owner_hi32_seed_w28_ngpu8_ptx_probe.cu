#pragma push_macro("main")
#undef main
#define main gridfp_b300_fullmate_dropn_main_unused
#include "../oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
#pragma pop_macro("main")

static constexpr Code B300_MAIN_CHUNK_W28_G8 = 48214938328ULL;

__device__ __forceinline__ Code hi32_seed_masked_base(int owner) {
    const Code u = Code(unsigned(owner));
    return ((Code(0) - (u & 1ULL)) & B300_MAIN_CHUNK_W28_G8) +
           ((Code(0) - ((u >> 1) & 1ULL)) & (B300_MAIN_CHUNK_W28_G8 << 1)) +
           ((Code(0) - ((u >> 2) & 1ULL)) & (B300_MAIN_CHUNK_W28_G8 << 2));
}

__device__ __forceinline__ std::uint32_t hi32_seed_mul(Code g) {
    const std::uint32_t h = std::uint32_t(g >> 32);
    return (h * 365u) >> 12;
}

__device__ __forceinline__ std::uint32_t hi32_seed_shiftadd(Code g) {
    const std::uint32_t h = std::uint32_t(g >> 32);
    return ((h << 8) + (h << 6) + (h << 5) +
            (h << 3) + (h << 2) + h) >> 12;
}

template<bool SHIFTADD>
__device__ __forceinline__ ShardAddress8 hi32_seed_address(Code g) {
    std::uint32_t owner = SHIFTADD ? hi32_seed_shiftadd(g) : hi32_seed_mul(g);
    Code local = g - hi32_seed_masked_base(int(owner));
    const std::uint32_t correction = std::uint32_t(local >= B300_MAIN_CHUNK_W28_G8);
    owner += correction;
    local -= (Code(0) - Code(correction)) & B300_MAIN_CHUNK_W28_G8;
    return {int(owner), local};
}

extern "C" __global__ void gridfp_b300_shard_owner_hi32_compare_probe(
    const Code* global_index, int* owner, Code* local
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    const ShardAddress8 a = shard_address8(global_index[tid], B300_MAIN_CHUNK_W28_G8);
    owner[tid] = a.owner;
    local[tid] = a.local;
}

extern "C" __global__ void gridfp_b300_shard_owner_hi32_mul_probe(
    const Code* global_index, int* owner, Code* local
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    const ShardAddress8 a = hi32_seed_address<false>(global_index[tid]);
    owner[tid] = a.owner;
    local[tid] = a.local;
}

extern "C" __global__ void gridfp_b300_shard_owner_hi32_shiftadd_probe(
    const Code* global_index, int* owner, Code* local
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    const ShardAddress8 a = hi32_seed_address<true>(global_index[tid]);
    owner[tid] = a.owner;
    local[tid] = a.local;
}
