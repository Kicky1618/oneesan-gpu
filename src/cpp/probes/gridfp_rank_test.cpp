#include <cstdint>
#define main ggcount_original_main
#include "../../../third_party/ggcount/src/ggcount.cpp"
#undef main

#include <iostream>
#include <vector>

static unsigned long long dp[32][32];

static void build_dp(int maxw) {
    for (int h = 0; h < 32; ++h) dp[0][h] = (h == 0);
    for (int w = 1; w <= maxw; ++w) {
        for (int h = 0; h < 31; ++h) {
            unsigned long long x = dp[w-1][h]; // N
            if (h > 0) x += dp[w-1][h-1];      // R: height --
            if (h + 1 < 32) x += dp[w-1][h+1]; // L: height ++
            dp[w][h] = x;
        }
    }
}

static unsigned long long rank_mate(Mate m, int width, int start_h=1) {
    unsigned long long rank = 0;
    int h = start_h;
    for (int pos = width - 1; pos >= 0; --pos) {
        MateValue s = m.get(pos);
        int rem = pos;
        // Symbol order in GGCount fillCode*: N, R, L.
        if (s > N) {
            rank += dp[rem][h];
        }
        if (s > R && h > 0) {
            rank += dp[rem][h-1];
        }
        if (s == R) --h;
        else if (s == L) ++h;
        if (h < 0) return ~0ULL;
    }
    return h == 0 ? rank : ~0ULL;
}

static Mate unrank_mate(unsigned long long rank, int width) {
    unsigned long long id = 0;
    int h = 1;
    for (int pos = width - 1; pos >= 0; --pos) {
        auto z = dp[pos][h];
        if (rank < z) {
            // N
        } else {
            rank -= z;
            auto r = h > 0 ? dp[pos][h-1] : 0;
            if (rank < r) { id |= 1ULL << (2*pos); --h; }
            else { rank -= r; id |= 2ULL << (2*pos); ++h; }
        }
    }
    return Mate(id);
}

int main() {
    msg = NONE;
    build_dp(30);
    for (int width = 2; width <= 18; ++width) {
        MateCodec mc(width, (width + 1) / 2, 1, 0);
        unsigned long long seen = 0;
        for (Code bi = 0; bi < mc.codeSizeL(); ++bi) {
            auto const& b = mc.codeTable(bi);
            for (Code i = 0; i < b.size; ++i) {
                Mate m = b.mateL | b.mateR[i];
                auto r = rank_mate(m, width, 1);
                auto e = mc.encode(m);
                if (r != e) {
                    std::cerr << "mismatch width=" << width << " mate=" << m
                              << " rank=" << r << " encode=" << e << "\n";
                    return 1;
                }
                ++seen;
            }
        }
        std::cout << "width=" << width << " states=" << seen << " rank_ok\n";
    }
    return 0;
}
