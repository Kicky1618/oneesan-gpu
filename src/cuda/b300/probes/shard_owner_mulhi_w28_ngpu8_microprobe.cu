#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#pragma push_macro("main")
#undef main
#define main gridfp_b300_fullmate_dropn_main_unused
#include "../oneesan_cuda_gridfp_b300_hbm32_fullmate_dropN.cu"
#pragma pop_macro("main")

#ifndef B300_SHARD_OWNER_MODE
#define B300_SHARD_OWNER_MODE 0
#endif
static_assert(B300_SHARD_OWNER_MODE >= 0 && B300_SHARD_OWNER_MODE <= 5,
              "B300_SHARD_OWNER_MODE must be 0..5");

namespace {

static constexpr Code MAIN_TOTAL = 385719506620ULL;
static constexpr Code MAIN_CHUNK = 48214938328ULL;
static constexpr Code MAIN_MAGIC = 195888106327ULL;
static constexpr unsigned MAIN_HIGH_SHIFT = 9;
static constexpr std::uint32_t MAIN_MAGIC_LO = 2614578007u;
static constexpr std::uint32_t MAIN_MAGIC_HI = 45u;

__device__ __constant__ Code MAIN_BASE[8] = {
    0ULL,48214938328ULL,96429876656ULL,144644814984ULL,
    192859753312ULL,241074691640ULL,289289629968ULL,337504568296ULL,
};

__device__ __forceinline__ int mulhi_owner(Code g) {
    return int(__umul64hi(g, MAIN_MAGIC) >> MAIN_HIGH_SHIFT);
}

__device__ __forceinline__ Code masked_base(int owner) {
    const Code u = Code(unsigned(owner));
    const Code b0 = (Code(0) - (u & 1ULL)) & MAIN_CHUNK;
    const Code b1 = (Code(0) - ((u >> 1) & 1ULL)) & (MAIN_CHUNK << 1);
    const Code b2 = (Code(0) - ((u >> 2) & 1ULL)) & (MAIN_CHUNK << 2);
    return b0 + b1 + b2;
}

__device__ __forceinline__ int u32limb_owner(Code g) {
    const std::uint32_t a0 = std::uint32_t(g);
    const std::uint32_t a1 = std::uint32_t(g >> 32);
    const std::uint32_t p00_hi = __umulhi(a0, MAIN_MAGIC_LO);
    const std::uint32_t p01_lo = a0 * MAIN_MAGIC_HI;
    const std::uint32_t p01_hi = __umulhi(a0, MAIN_MAGIC_HI);
    const std::uint32_t p10_lo = a1 * MAIN_MAGIC_LO;
    const std::uint32_t p10_hi = __umulhi(a1, MAIN_MAGIC_LO);
    const std::uint32_t p11 = a1 * MAIN_MAGIC_HI;
    const std::uint32_t s0 = p00_hi + p01_lo;
    std::uint32_t carry = std::uint32_t(s0 < p00_hi);
    const std::uint32_t s1 = s0 + p10_lo;
    carry += std::uint32_t(s1 < s0);
    const std::uint32_t high64 = p01_hi + p10_hi + p11 + carry;
    return int(high64 >> MAIN_HIGH_SHIFT);
}

struct U32Product { std::uint32_t lo, hi; };

__device__ __forceinline__ U32Product mul45_shiftadd(std::uint32_t x) {
    std::uint32_t lo = x << 5;
    std::uint32_t hi = x >> 27;
    std::uint32_t add = x << 3;
    std::uint32_t old = lo;
    lo += add;
    hi += (x >> 29) + std::uint32_t(lo < old);
    add = x << 2;
    old = lo;
    lo += add;
    hi += (x >> 30) + std::uint32_t(lo < old);
    old = lo;
    lo += x;
    hi += std::uint32_t(lo < old);
    return {lo, hi};
}

__device__ __forceinline__ int u32shift_owner(Code g) {
    const std::uint32_t a0 = std::uint32_t(g);
    const std::uint32_t a1 = std::uint32_t(g >> 32);
    const std::uint32_t p00_hi = __umulhi(a0, MAIN_MAGIC_LO);
    const U32Product p01 = mul45_shiftadd(a0);
    const std::uint32_t p10_lo = a1 * MAIN_MAGIC_LO;
    const std::uint32_t p10_hi = __umulhi(a1, MAIN_MAGIC_LO);
    const std::uint32_t p11 = (a1 << 5) + (a1 << 3) + (a1 << 2) + a1;
    const std::uint32_t s0 = p00_hi + p01.lo;
    std::uint32_t carry = std::uint32_t(s0 < p00_hi);
    const std::uint32_t s1 = s0 + p10_lo;
    carry += std::uint32_t(s1 < s0);
    const std::uint32_t high64 = p01.hi + p10_hi + p11 + carry;
    return int(high64 >> MAIN_HIGH_SHIFT);
}

__device__ __forceinline__ ShardAddress8 mulhi_mul_address(Code g) {
    const int owner = mulhi_owner(g); return {owner, g - Code(owner) * MAIN_CHUNK};
}
__device__ __forceinline__ ShardAddress8 mulhi_table_address(Code g) {
    const int owner = mulhi_owner(g); return {owner, g - MAIN_BASE[owner]};
}
__device__ __forceinline__ ShardAddress8 mulhi_mask_address(Code g) {
    const int owner = mulhi_owner(g); return {owner, g - masked_base(owner)};
}
__device__ __forceinline__ ShardAddress8 u32limb_mask_address(Code g) {
    const int owner = u32limb_owner(g); return {owner, g - masked_base(owner)};
}
__device__ __forceinline__ ShardAddress8 u32shift_mask_address(Code g) {
    const int owner = u32shift_owner(g); return {owner, g - masked_base(owner)};
}

__device__ __forceinline__ ShardAddress8 candidate_address(Code g) {
#if B300_SHARD_OWNER_MODE == 0
    return shard_address8(g, MAIN_CHUNK);
#elif B300_SHARD_OWNER_MODE == 1
    return mulhi_mul_address(g);
#elif B300_SHARD_OWNER_MODE == 2
    return mulhi_table_address(g);
#elif B300_SHARD_OWNER_MODE == 3
    return mulhi_mask_address(g);
#elif B300_SHARD_OWNER_MODE == 4
    return u32limb_mask_address(g);
#else
    return u32shift_mask_address(g);
#endif
}

__global__ void exact_kernel(const Code* global_index, int* owner, Code* local, int n) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n) return;
    const ShardAddress8 a = candidate_address(global_index[tid]);
    owner[tid] = a.owner; local[tid] = a.local;
}

__global__ void perf_kernel(Code* out, int n, int iters, Code stride, Code step) {
    const int tid = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (tid >= n) return;
    Code g = Code(tid) * stride;
    if (g >= MAIN_TOTAL) g = MAIN_TOTAL - 1;
    Code acc = 0;
    for (int i = 0; i < iters; ++i) {
        const ShardAddress8 a = candidate_address(g);
        acc += a.local ^ (Code(a.owner) << 56);
        g += step;
        if (g >= MAIN_TOTAL) g -= MAIN_TOTAL;
    }
    out[tid] = acc;
}

float run_once(Code* out, int n, int blocks, int threads, int iters, Code stride, Code step) {
    cudaEvent_t a{}, b{}; ck(cudaEventCreate(&a), "event a"); ck(cudaEventCreate(&b), "event b");
    ck(cudaEventRecord(a), "record a"); perf_kernel<<<blocks, threads>>>(out, n, iters, stride, step);
    ck(cudaGetLastError(), "perf launch"); ck(cudaEventRecord(b), "record b"); ck(cudaEventSynchronize(b), "sync b");
    float ms = 0; ck(cudaEventElapsedTime(&ms, a, b), "elapsed"); cudaEventDestroy(a); cudaEventDestroy(b); return ms;
}

float median(std::vector<float> x) { std::sort(x.begin(), x.end()); const size_t n=x.size(); return n&1?x[n/2]:0.5f*(x[n/2-1]+x[n/2]); }

} // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::atoi(argv[1]) : 256;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 8192;
    const int repeats = argc > 4 ? std::atoi(argv[4]) : 9;
    if (blocks < 1 || threads < 1 || threads > 1024 || iters < 1 || repeats < 1) return 2;

    const int n = blocks * threads;
    const Code stride = std::max<Code>(1, MAIN_TOTAL / Code(n));
    const Code step = MAIN_CHUNK / 7 + 1;
    auto h_index = std::vector<Code>(static_cast<size_t>(n));
    for (int i=0;i<n;++i) h_index[static_cast<size_t>(i)] = Code(i) * stride;

    Code *d_index=nullptr,*d_local=nullptr,*d_out=nullptr; int* d_owner=nullptr;
    ck(cudaMalloc(&d_index,static_cast<size_t>(n)*sizeof(Code)),"alloc index");
    ck(cudaMalloc(&d_owner,static_cast<size_t>(n)*sizeof(int)),"alloc owner");
    ck(cudaMalloc(&d_local,static_cast<size_t>(n)*sizeof(Code)),"alloc local");
    ck(cudaMalloc(&d_out,static_cast<size_t>(n)*sizeof(Code)),"alloc out");
    ck(cudaMemcpy(d_index,h_index.data(),static_cast<size_t>(n)*sizeof(Code),cudaMemcpyHostToDevice),"copy index");

    exact_kernel<<<blocks,threads>>>(d_index,d_owner,d_local,n); ck(cudaGetLastError(),"exact launch"); ck(cudaDeviceSynchronize(),"exact sync");
    auto h_owner = std::vector<int>(static_cast<size_t>(n));
    auto h_local = std::vector<Code>(static_cast<size_t>(n));
    ck(cudaMemcpy(h_owner.data(),d_owner,static_cast<size_t>(n)*sizeof(int),cudaMemcpyDeviceToHost),"copy owner");
    ck(cudaMemcpy(h_local.data(),d_local,static_cast<size_t>(n)*sizeof(Code),cudaMemcpyDeviceToHost),"copy local");
    for(int i=0;i<n;++i){
        const Code g=h_index[static_cast<size_t>(i)]; const int owner=int(g/MAIN_CHUNK); const Code local=g-Code(owner)*MAIN_CHUNK;
        if(h_owner[static_cast<size_t>(i)]!=owner||h_local[static_cast<size_t>(i)]!=local){
            std::fprintf(stderr,"mismatch i=%d g=%llu got=(%d,%llu) exact=(%d,%llu)\n",i,(unsigned long long)g,h_owner[static_cast<size_t>(i)],(unsigned long long)h_local[static_cast<size_t>(i)],owner,(unsigned long long)local); return 3;
        }
    }

    run_once(d_out,n,blocks,threads,iters,stride,step);
    std::vector<float> times; times.reserve(static_cast<size_t>(repeats));
    for(int r=0;r<repeats;++r) times.push_back(run_once(d_out,n,blocks,threads,iters,stride,step));
    auto h_out=std::vector<Code>(static_cast<size_t>(n));
    ck(cudaMemcpy(h_out.data(),d_out,static_cast<size_t>(n)*sizeof(Code),cudaMemcpyDeviceToHost),"copy out");
    std::uint64_t checksum=0; for(Code x:h_out) checksum^=x+0x9e3779b97f4a7c15ULL+(checksum<<6)+(checksum>>2);
    const double ops=double(n)*double(iters),ms=median(times);
    std::printf("gridfp-b300-shard-owner-mulhi-w28-ngpu8-microprobe OK mode=%d blocks=%d threads=%d iters=%d repeats=%d addresses=%.0f median_ms=%.6f Gaddr_s=%.6f checksum=%llu exact=OK\n",int(B300_SHARD_OWNER_MODE),blocks,threads,iters,repeats,ops,ms,ops/ms/1.0e6,(unsigned long long)checksum);
    cudaFree(d_out);cudaFree(d_local);cudaFree(d_owner);cudaFree(d_index);return 0;
}
