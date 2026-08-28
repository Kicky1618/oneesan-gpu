#pragma push_macro("main")
#undef main
#define main gridfp_b300_fullmate_dropn_main_unused
#include "../oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
#pragma pop_macro("main")

static constexpr Code B300_MAIN_TOTAL_W28 = 385719506620ULL;
static constexpr Code B300_MAIN_CHUNK_W28_G8 = 48214938328ULL;
static constexpr Code B300_MAIN_MAGIC_W28_G8 = 195888106327ULL;
static constexpr unsigned B300_MAIN_HIGH_SHIFT_W28_G8 = 9;
static constexpr std::uint32_t B300_MAIN_MAGIC_LO_W28_G8 = 2614578007u;
static constexpr std::uint32_t B300_MAIN_MAGIC_HI_W28_G8 = 45u;

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

__device__ __forceinline__ int shard_owner8_mulhi(Code g) {
    return int(__umul64hi(g, B300_MAIN_MAGIC_W28_G8) >>
               B300_MAIN_HIGH_SHIFT_W28_G8);
}

__device__ __forceinline__ int shard_owner8_u32limb(Code g) {
    const std::uint32_t a0 = std::uint32_t(g);
    const std::uint32_t a1 = std::uint32_t(g >> 32);
    const std::uint32_t p00_hi = __umulhi(a0, B300_MAIN_MAGIC_LO_W28_G8);
    const std::uint32_t p01_lo = a0 * B300_MAIN_MAGIC_HI_W28_G8;
    const std::uint32_t p01_hi = __umulhi(a0, B300_MAIN_MAGIC_HI_W28_G8);
    const std::uint32_t p10_lo = a1 * B300_MAIN_MAGIC_LO_W28_G8;
    const std::uint32_t p10_hi = __umulhi(a1, B300_MAIN_MAGIC_LO_W28_G8);
    const std::uint32_t p11 = a1 * B300_MAIN_MAGIC_HI_W28_G8;
    const std::uint32_t s0 = p00_hi + p01_lo;
    std::uint32_t carry = std::uint32_t(s0 < p00_hi);
    const std::uint32_t s1 = s0 + p10_lo;
    carry += std::uint32_t(s1 < s0);
    const std::uint32_t high64 = p01_hi + p10_hi + p11 + carry;
    return int(high64 >> B300_MAIN_HIGH_SHIFT_W28_G8);
}

__device__ __forceinline__ Code shard_base8_masked_probe(int owner) {
    const Code u = Code(unsigned(owner));
    const Code b0 = (Code(0) - (u & 1ULL)) & B300_MAIN_CHUNK_W28_G8;
    const Code b1 = (Code(0) - ((u >> 1) & 1ULL)) & (B300_MAIN_CHUNK_W28_G8 << 1);
    const Code b2 = (Code(0) - ((u >> 2) & 1ULL)) & (B300_MAIN_CHUNK_W28_G8 << 2);
    return b0 + b1 + b2;
}

__device__ __forceinline__ ShardAddress8 shard_address8_mulhi_mul(Code g) {
    const int owner = shard_owner8_mulhi(g);
    return {owner, g - Code(owner) * B300_MAIN_CHUNK_W28_G8};
}

__device__ __forceinline__ ShardAddress8 shard_address8_mulhi_table(Code g) {
    const int owner = shard_owner8_mulhi(g);
    return {owner, g - B300_MAIN_SHARD_BASE_W28_G8[owner]};
}

__device__ __forceinline__ ShardAddress8 shard_address8_mulhi_mask(Code g) {
    const int owner = shard_owner8_mulhi(g);
    return {owner, g - shard_base8_masked_probe(owner)};
}

__device__ __forceinline__ ShardAddress8 shard_address8_u32limb_mask(Code g) {
    const int owner = shard_owner8_u32limb(g);
    return {owner, g - shard_base8_masked_probe(owner)};
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

extern "C" __global__ void gridfp_b300_shard_owner_mulhi_mask_probe(
    const Code* global_index, int* owner, Code* local
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    const ShardAddress8 a = shard_address8_mulhi_mask(global_index[tid]);
    owner[tid] = a.owner;
    local[tid] = a.local;
}

extern "C" __global__ void gridfp_b300_shard_owner_u32limb_mask_probe(
    const Code* global_index, int* owner, Code* local
) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    const ShardAddress8 a = shard_address8_u32limb_mask(global_index[tid]);
    owner[tid] = a.owner;
    local[tid] = a.local;
}