#include <cstdint>

using Rank64 = std::uint64_t;

extern "C" __global__ void owner_generic_w28_probe(
    std::uint32_t* out, Rank64 midpoint, int ngpu
) {
    constexpr unsigned shift = 52;
    constexpr std::uint32_t magic = 9513u;
    const std::uint32_t scale = magic * static_cast<std::uint32_t>(ngpu);
    const std::uint32_t lo = static_cast<std::uint32_t>(midpoint);
    const std::uint32_t hi = static_cast<std::uint32_t>(midpoint >> 32);
    const std::uint32_t product_hi = __umulhi(lo, scale);
    out[0] = (hi * scale + product_hi) >> (shift - 32);
}

extern "C" __global__ void owner_ngpu8_w28_probe(
    std::uint32_t* out, Rank64 midpoint
) {
    constexpr unsigned shift = 49; // 52 - log2(8)
    constexpr std::uint32_t magic = 9513u;
    const std::uint32_t lo = static_cast<std::uint32_t>(midpoint);
    const std::uint32_t hi = static_cast<std::uint32_t>(midpoint >> 32);
    const std::uint32_t product_hi = __umulhi(lo, magic);
    out[0] = (hi * magic + product_hi) >> (shift - 32);
}
