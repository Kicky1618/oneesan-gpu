#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>

using u64 = std::uint64_t;
using u128 = unsigned __int128;

namespace {

constexpr int W = 28;
constexpr int NGPU = 8;

u64 dp[W + 1][W + 2]{};

void build_dp() {
    for (int h = 0; h <= W + 1; ++h) dp[0][h] = (h == 0);
    for (int w = 1; w <= W; ++w) {
        for (int h = 0; h <= W; ++h) {
            u64 x = dp[w - 1][h];
            if (h > 0) x += dp[w - 1][h - 1];
            if (h < W + 1) x += dp[w - 1][h + 1];
            dp[w][h] = x;
        }
    }
}

u128 ceil_div(u128 a, u128 b) { return (a + b - 1) / b; }

struct MagicProof {
    int shift = -1;
    u64 magic = 0;
    u64 lo_magic = 0;
    u64 hi_magic = 0;
};

MagicProof find_min_magic(u64 total, u64 chunk) {
    for (int shift = 64; shift <= 96; ++shift) {
        const u128 unit = u128(1) << shift;
        u128 lo_magic = 1;
        u128 hi_magic = std::numeric_limits<u64>::max();
        for (int q = 0; q < NGPU; ++q) {
            const u64 lo = u64(q) * chunk;
            if (lo >= total) break;
            const u64 hi = std::min<u64>(u64(q + 1) * chunk - 1, total - 1);
            if (q > 0) {
                lo_magic = std::max(lo_magic, ceil_div(u128(q) * unit, lo));
            }
            hi_magic = std::min(hi_magic,
                (u128(q + 1) * unit - 1) / hi);
        }
        if (lo_magic <= hi_magic && hi_magic <= std::numeric_limits<u64>::max()) {
            return MagicProof{
                shift,
                static_cast<u64>(lo_magic),
                static_cast<u64>(lo_magic),
                static_cast<u64>(hi_magic)};
        }
    }
    return {};
}

u64 owner_mulhi_model(u64 g, u64 magic, int shift) {
    return static_cast<u64>((u128(g) * magic) >> shift);
}

u64 masked_base(u64 owner, u64 chunk) {
    const u64 b0 = (u64(0) - (owner & 1ULL)) & chunk;
    const u64 b1 = (u64(0) - ((owner >> 1) & 1ULL)) & (chunk << 1);
    const u64 b2 = (u64(0) - ((owner >> 2) & 1ULL)) & (chunk << 2);
    return b0 + b1 + b2;
}

bool prove_case(const char* name, u64 total, u64 expected_total,
                int expected_shift, u64 expected_magic) {
    if (total != expected_total) {
        std::fprintf(stderr, "%s total mismatch got=%llu expected=%llu\n",
            name, (unsigned long long)total, (unsigned long long)expected_total);
        return false;
    }
    const u64 chunk = (total + NGPU - 1) / NGPU;
    const MagicProof p = find_min_magic(total, chunk);
    if (p.shift != expected_shift || p.magic != expected_magic ||
        p.lo_magic != p.hi_magic) {
        std::fprintf(stderr,
            "%s magic mismatch shift=%d magic=%llu range=[%llu,%llu]\n",
            name, p.shift, (unsigned long long)p.magic,
            (unsigned long long)p.lo_magic, (unsigned long long)p.hi_magic);
        return false;
    }

    for (u64 owner = 0; owner < NGPU; ++owner) {
        if (masked_base(owner, chunk) != owner * chunk) {
            std::fprintf(stderr, "%s masked-base failure owner=%llu\n",
                name, (unsigned long long)owner);
            return false;
        }
    }

    u64 endpoint_cases = 0;
    u64 max_owner = 0;
    u64 max_local = 0;
    for (int q = 0; q < NGPU; ++q) {
        const u64 lo = u64(q) * chunk;
        if (lo >= total) break;
        const u64 hi = std::min<u64>(u64(q + 1) * chunk - 1, total - 1);
        const std::array<u64, 4> xs = {
            lo,
            std::min<u64>(lo + 1, hi),
            hi > lo ? hi - 1 : hi,
            hi,
        };
        for (u64 g : xs) {
            const u64 owner = owner_mulhi_model(g, p.magic, p.shift);
            const u64 exact = g / chunk;
            if (owner != exact || owner >= NGPU) {
                std::fprintf(stderr,
                    "%s endpoint failure g=%llu got=%llu exact=%llu\n",
                    name, (unsigned long long)g,
                    (unsigned long long)owner,
                    (unsigned long long)exact);
                return false;
            }
            const u64 local = g - owner * chunk;
            const u64 masked_local = g - masked_base(owner, chunk);
            if (local >= chunk || masked_local != local) {
                std::fprintf(stderr, "%s local/masked-base failure\n", name);
                return false;
            }
            max_owner = std::max(max_owner, owner);
            max_local = std::max(max_local, local);
            ++endpoint_cases;
        }
    }

    // The transformed quotient is monotone in g. Exactness at every interval's
    // first and last point therefore proves the entire integer interval.
    for (int q = 0; q < NGPU; ++q) {
        const u64 lo = u64(q) * chunk;
        if (lo >= total) break;
        const u64 hi = std::min<u64>(u64(q + 1) * chunk - 1, total - 1);
        if (owner_mulhi_model(lo, p.magic, p.shift) != u64(q) ||
            owner_mulhi_model(hi, p.magic, p.shift) != u64(q)) {
            std::fprintf(stderr, "%s interval proof failure q=%d\n", name, q);
            return false;
        }
    }

    std::printf(
        "%s total=%llu chunk=%llu shift=%d high_shift=%d magic=%llu "
        "magic_bits=%d endpoint_cases=%llu max_owner=%llu max_local=%llu "
        "masked_base_exact=1 exact=1\n",
        name,
        (unsigned long long)total,
        (unsigned long long)chunk,
        p.shift,
        p.shift - 64,
        (unsigned long long)p.magic,
        64 - __builtin_clzll(p.magic),
        (unsigned long long)endpoint_cases,
        (unsigned long long)max_owner,
        (unsigned long long)max_local);
    return true;
}

} // namespace

int main() {
    build_dp();
    const u64 main_total = dp[28][1];
    const u64 block_total = dp[27][1];
    const bool ok_main = prove_case(
        "main", main_total, 385719506620ULL, 73, 195888106327ULL);
    const bool ok_block = prove_case(
        "block", block_total, 135015505407ULL, 71, 139905900989ULL);
    if (!ok_main || !ok_block) return 1;
    std::puts(
        "b300-shard-owner-mulhi-w28-ngpu8-proof OK cases=2 ngpu=8 "
        "main_high_shift=9 block_high_shift=7 magic_unique=1 "
        "masked_base_exact=1 masked_base_mul64=0 masked_base_table=0 "
        "monotone_interval_proof=1 exact=1");
    return 0;
}