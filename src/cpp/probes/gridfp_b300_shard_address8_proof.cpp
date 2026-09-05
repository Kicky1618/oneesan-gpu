#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <limits>

namespace {
using Code = std::uint64_t;

struct ShardAddress8 {
    int owner = 0;
    Code local = 0;
};

ShardAddress8 shard_address8(Code g, Code chunk) {
    int owner = 0;
    const Code c4 = chunk << 2;
    if (g >= c4) {
        g -= c4;
        owner |= 4;
    }
    const Code c2 = chunk << 1;
    if (g >= c2) {
        g -= c2;
        owner |= 2;
    }
    if (g >= chunk) {
        g -= chunk;
        owner |= 1;
    }
    return {owner, g};
}

std::array<std::array<Code, 30>, 29> full_dp() {
    std::array<std::array<Code, 30>, 29> dp{};
    dp[0][0] = 1;
    for (int w = 1; w <= 28; ++w) {
        for (int h = 0; h <= 28; ++h) {
            Code x = dp[w - 1][h];
            if (h > 0) x += dp[w - 1][h - 1];
            if (h < 29) x += dp[w - 1][h + 1];
            dp[w][h] = x;
        }
    }
    return dp;
}

bool check(Code g, Code chunk, int ngpu, std::uint64_t& cases) {
    const auto fast = shard_address8(g, chunk);
    const int exact_owner = int(g / chunk);
    const Code exact_local = g % chunk;
    ++cases;
    return exact_owner < ngpu && fast.owner == exact_owner &&
           fast.local == exact_local && fast.local < chunk;
}
}  // namespace

int main() {
    std::uint64_t cases = 0;

    // Exhaust the complete legal domain for many small chunks and every 1..8 GPU count.
    for (Code chunk = 1; chunk <= 4096; ++chunk) {
        for (int ngpu = 1; ngpu <= 8; ++ngpu) {
            for (Code g = 0; g < Code(ngpu) * chunk; ++g)
                if (!check(g, chunk, ngpu, cases)) return 2;
        }
    }

    // Exercise every decision boundary at large 64-bit chunks as well.
    constexpr std::array<Code, 8> large_chunks = {
        (Code(1) << 32) - 1,
        Code(1) << 32,
        (Code(1) << 40) + 12345,
        (Code(1) << 48) - 17,
        Code(1) << 52,
        (Code(1) << 56) + 3,
        Code(1) << 60,
        (Code(1) << 61) - 1
    };
    for (Code chunk : large_chunks) {
        if (chunk > (std::numeric_limits<Code>::max() >> 3)) return 3;
        for (int ngpu = 1; ngpu <= 8; ++ngpu) {
            for (int q = 0; q < ngpu; ++q) {
                const std::array<Code, 5> offsets = {
                    0,
                    std::min<Code>(1, chunk - 1),
                    chunk / 2,
                    chunk > 1 ? chunk - 2 : 0,
                    chunk - 1
                };
                for (Code r : offsets) {
                    const Code g = Code(q) * chunk + r;
                    if (!check(g, chunk, ngpu, cases)) return 4;
                }
            }
        }
    }

    const auto dp = full_dp();
    constexpr int ngpu = 8;
    const Code main_n = dp[28][1];
    const Code block_n = dp[27][1];
    const Code main_chunk = (main_n + ngpu - 1) / ngpu;
    const Code block_chunk = (block_n + ngpu - 1) / ngpu;
    if (main_n != 385719506620ULL || block_n != 135015505407ULL ||
        main_chunk != 48214938328ULL || block_chunk != 16876938176ULL)
        return 5;

    // Check all shard starts/ends plus the actual final global index.
    for (Code chunk : {main_chunk, block_chunk}) {
        for (int q = 0; q < ngpu; ++q) {
            for (Code r : {Code(0), Code(1), chunk / 2, chunk - 2, chunk - 1}) {
                if (!check(Code(q) * chunk + r, chunk, ngpu, cases)) return 6;
            }
        }
    }
    if (!check(main_n - 1, main_chunk, ngpu, cases) ||
        !check(block_n - 1, block_chunk, ngpu, cases)) return 7;

    std::cout << "gridfp-b300-shard-address8-proof OK"
              << " ngpu_min=1 ngpu_max=8"
              << " cases=" << cases
              << " main_states=" << main_n
              << " block_states=" << block_n
              << " main_chunk=" << main_chunk
              << " block_chunk=" << block_chunk
              << " max_chunk_bits=61"
              << " compare_stages=3 max_subtractions=3"
              << " owner_mul=0 div64=0 mod64=0 exact=1\n";
    return 0;
}
