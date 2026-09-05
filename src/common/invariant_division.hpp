#pragma once
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define ONEESAN_DIV_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_DIV_HD inline
#endif

namespace oneesan {
// Host setup only. Zero denotes an empty factor block, never a valid divisor.
inline uint64_t division_reciprocal(uint32_t divisor) {
    return divisor ? std::numeric_limits<uint64_t>::max() / divisor : 0;
}

struct Divmod64By32 { uint64_t quotient; uint32_t remainder; };

// Precondition: divisor > 0, reciprocal = floor((2^64 - 1) / divisor).
// The multiply-high estimate never exceeds floor(n / divisor), and is at
// most one below it: 0 < 2^64/divisor - reciprocal <= 1 and n < 2^64.
// Consequently one exact remainder correction suffices, including divisor=1.
ONEESAN_DIV_HD Divmod64By32 invariant_divmod(
    uint64_t n, uint32_t divisor, uint64_t reciprocal) {
#if defined(__CUDA_ARCH__)
    uint64_t q = __umul64hi(n, reciprocal);
#else
    uint64_t q = uint64_t((__uint128_t(n) * reciprocal) >> 64);
#endif
    uint64_t r = n - q * divisor;
    const bool correct = r >= divisor;
    return {q + correct, uint32_t(r - (correct ? divisor : 0))};
}
} // namespace oneesan

#undef ONEESAN_DIV_HD
