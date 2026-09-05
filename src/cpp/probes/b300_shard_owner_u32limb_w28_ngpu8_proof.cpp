#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

namespace {

struct Case {
    const char* name;
    u64 total;
    u64 chunk;
    u64 magic;
    unsigned shift;
    u32 expected_g_hi_max;
    u32 expected_m_hi;
    u32 expected_m_lo;
    u32 expected_high_bound;
};

constexpr std::array<Case, 2> CASES{{
    {"main", 385719506620ULL, 48214938328ULL, 195888106327ULL, 73, 89, 45, 2614578007u, 4139},
    {"block", 135015505407ULL, 16876938176ULL, 139905900989ULL, 71, 31, 32, 2466947517u, 1055},
}};

u32 mul_hi_u32_model(u32 a, u32 b) {
    return u32((u64(a) * u64(b)) >> 32);
}

u32 high64_u32limb(u64 g, u64 magic, u32& carry_out) {
    const u32 a0 = u32(g);
    const u32 a1 = u32(g >> 32);
    const u32 b0 = u32(magic);
    const u32 b1 = u32(magic >> 32);

    const u32 p00_hi = mul_hi_u32_model(a0, b0);
    const u32 p01_lo = a0 * b1;
    const u32 p01_hi = mul_hi_u32_model(a0, b1);
    const u32 p10_lo = a1 * b0;
    const u32 p10_hi = mul_hi_u32_model(a1, b0);
    const u32 p11 = a1 * b1;

    const u32 s0 = p00_hi + p01_lo;
    u32 carry = u32(s0 < p00_hi);
    const u32 s1 = s0 + p10_lo;
    carry += u32(s1 < s0);
    carry_out = carry;

    // For W28x8 the upper operand limbs are tiny.  The proven bound below keeps
    // this sum below 2^32, so the high 64 bits of the 128-bit product fit in u32.
    return p01_hi + p10_hi + p11 + carry;
}

u64 owner_u32limb(u64 g, const Case& c) {
    u32 carry = 0;
    const u32 hi64 = high64_u32limb(g, c.magic, carry);
    return u64(hi64 >> (c.shift - 64));
}

u64 masked_base(u64 owner, u64 chunk) {
    const u64 b0 = (u64(0) - (owner & 1ULL)) & chunk;
    const u64 b1 = (u64(0) - ((owner >> 1) & 1ULL)) & (chunk << 1);
    const u64 b2 = (u64(0) - ((owner >> 2) & 1ULL)) & (chunk << 2);
    return b0 + b1 + b2;
}

bool check_case(const Case& c) {
    const u32 g_hi_max = u32((c.total - 1) >> 32);
    const u32 m_hi = u32(c.magic >> 32);
    const u32 m_lo = u32(c.magic);
    const u32 high_bound = (m_hi - 1) + (g_hi_max ? g_hi_max - 1 : 0) +
                           g_hi_max * m_hi + 2;
    if (g_hi_max != c.expected_g_hi_max || m_hi != c.expected_m_hi ||
        m_lo != c.expected_m_lo || high_bound != c.expected_high_bound) {
        std::fprintf(stderr, "%s bound/constants mismatch\n", c.name);
        return false;
    }

    u64 cases = 0;
    u32 max_carry = 0;
    u32 max_high = 0;
    for (u64 q = 0; q < 8; ++q) {
        const u64 lo = q * c.chunk;
        if (lo >= c.total) break;
        const u64 hi = std::min((q + 1) * c.chunk - 1, c.total - 1);
        const std::array<u64, 6> xs{{
            lo,
            std::min(lo + 1, hi),
            std::min(lo + 2, hi),
            hi > lo + 1 ? hi - 2 : lo,
            hi > lo ? hi - 1 : hi,
            hi,
        }};
        for (u64 g : xs) {
            u32 carry = 0;
            const u32 limb_hi = high64_u32limb(g, c.magic, carry);
            const u64 exact_hi = u64((u128(g) * c.magic) >> 64);
            const u64 owner = u64(limb_hi >> (c.shift - 64));
            const u64 exact_owner = g / c.chunk;
            if (limb_hi != exact_hi || owner != exact_owner ||
                g - masked_base(owner, c.chunk) != g - exact_owner * c.chunk) {
                std::fprintf(stderr,
                    "%s endpoint failure g=%llu limb_hi=%u exact_hi=%llu owner=%llu exact_owner=%llu\n",
                    c.name, (unsigned long long)g, limb_hi,
                    (unsigned long long)exact_hi, (unsigned long long)owner,
                    (unsigned long long)exact_owner);
                return false;
            }
            max_carry = std::max(max_carry, carry);
            max_high = std::max(max_high, limb_hi);
            ++cases;
        }
    }

    // Dense deterministic coverage across the full production interval checks
    // the limb/carry implementation independently of the boundary proof above.
    constexpr u64 SAMPLES = 1000000;
    for (u64 i = 0; i <= SAMPLES; ++i) {
        const u64 g = u64((u128(i) * (c.total - 1)) / SAMPLES);
        u32 carry = 0;
        const u32 limb_hi = high64_u32limb(g, c.magic, carry);
        const u64 exact_hi = u64((u128(g) * c.magic) >> 64);
        if (limb_hi != exact_hi || owner_u32limb(g, c) != g / c.chunk) {
            std::fprintf(stderr, "%s dense failure i=%llu g=%llu\n",
                c.name, (unsigned long long)i, (unsigned long long)g);
            return false;
        }
        max_carry = std::max(max_carry, carry);
        max_high = std::max(max_high, limb_hi);
        ++cases;
    }

    std::printf(
        "%s total=%llu chunk=%llu magic=%llu shift=%u high_shift=%u "
        "g_hi_max=%u magic_hi=%u magic_lo=%u high_bound=%u high_bound_bits=%d "
        "max_carry=%u max_high=%u cases=%llu exact=1\n",
        c.name, (unsigned long long)c.total, (unsigned long long)c.chunk,
        (unsigned long long)c.magic, c.shift, c.shift - 64, g_hi_max, m_hi,
        m_lo, high_bound, 32 - __builtin_clz(high_bound), max_carry, max_high,
        (unsigned long long)cases);
    return true;
}

} // namespace

int main() {
    for (const Case& c : CASES) if (!check_case(c)) return 1;
    std::puts(
        "b300-shard-owner-u32limb-w28-ngpu8-proof OK cases=2 ngpu=8 "
        "device_mul64=0 device_div64=0 device_table_load=0 "
        "umulhi_u32_per_owner=3 mullo_u32_per_owner=3 masked_base_exact=1 "
        "high64_fits_u32=1 dense_samples_per_case=1000001 exact=1");
    return 0;
}
