#include "../../common/gridfp_transition.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {
using namespace oneesan::gridfp;

bool valid_mate(MateID m, int len) {
    int h = 1;
    for (int pos = 0; pos < len; ++pos) {
        const MateValue v = mget(m, len - 1 - pos);
        if (v == X) return false;
        if (v == R) --h;
        else if (v == L) ++h;
        if (h < 0) return false;
    }
    return h == 0;
}

bool rl_prefix_valid(MateID x, int W, int p) {
    const int first_high = p + 1;
    const MateID prefix = first_high >= W ? 0 : (x >> (2 * first_high));
    constexpr MateID EVEN = 0x5555555555555555ULL;
    const int r = __builtin_popcountll(prefix & EVEN);
    const int l = __builtin_popcountll((prefix >> 1) & EVEN);
    return 1 + l - r > 0;
}

void enumerate_valid_rec(int len, int bit, MateID m, std::vector<MateID>& out) {
    if (bit == len) {
        if (valid_mate(m, len)) out.push_back(m);
        return;
    }
    enumerate_valid_rec(len, bit + 1, m, out);
    enumerate_valid_rec(len, bit + 1, m | (MateID(R) << (2 * bit)), out);
    enumerate_valid_rec(len, bit + 1, m | (MateID(L) << (2 * bit)), out);
}

std::vector<MateID> enumerate_valid(int len) {
    std::vector<MateID> out;
    enumerate_valid_rec(len, 0, 0, out);
    return out;
}

struct ValidGenerator {
    static constexpr int MAX_W = 28;
    std::array<std::array<std::uint64_t, MAX_W + 2>, MAX_W + 1> dp{};
    ValidGenerator() {
        dp[0][0] = 1;
        for (int rem = 1; rem <= MAX_W; ++rem)
            for (int h = 0; h <= MAX_W; ++h)
                dp[rem][h] = dp[rem - 1][h] +
                    (h ? dp[rem - 1][h - 1] : 0) + dp[rem - 1][h + 1];
    }
    MateID sample(int W, std::mt19937_64& rng) const {
        MateID m = 0;
        int h = 1;
        for (int pos = 0; pos < W; ++pos) {
            const int rem = W - pos - 1;
            const std::uint64_t cn = dp[rem][h];
            const std::uint64_t cr = h ? dp[rem][h - 1] : 0;
            const std::uint64_t cl = dp[rem][h + 1];
            std::uint64_t pick = rng() % (cn + cr + cl);
            MateValue v = N;
            if (pick < cn) v = N;
            else if ((pick -= cn) < cr) { v = R; --h; }
            else { v = L; ++h; }
            m |= MateID(v) << (2 * (W - 1 - pos));
        }
        return m;
    }
};

bool check_main(MateID d, int W, std::uint64_t& simple) {
    if (!valid_mate(d, W)) return false;
    for (int p = 1; p < W; ++p) {
        const MateValuePair w = mpair(d, p);
        MateID x = 0;
        bool candidate = true;
        if (w == LR) x = msetpair(d, p, NN);
        else if (w == NR) x = msetpair(d, p, RN);
        else if (w == NL) x = msetpair(d, p, LN);
        else candidate = false;
        if (!candidate) continue;
        ++simple;
        if (!valid_mate(x, W)) return false;
        const IncludeResult z = include_horizontal(x, W, p);
        if (!z.valid || z.blocked || z.mate != d) {
            std::cerr << "simple roundtrip mismatch W=" << W << " p=" << p
                      << " pair=" << int(w) << '\n';
            return false;
        }
    }
    return true;
}

bool check_blocked(
    MateID b, int W, std::uint64_t& inserted, std::uint64_t& rl
) {
    if (!valid_mate(b, W - 1)) return false;
    for (int p = 1; p < W; ++p) {
        if (is_endpoint(mget(b, p - 1))) {
            const MateID x = minsert(b, p, N);
            ++inserted;
            if (!valid_mate(x, W)) return false;
            const IncludeResult z = include_horizontal(x, W, p);
            if (!z.valid || !z.blocked || z.mate != b) {
                std::cerr << "insert-N roundtrip mismatch W=" << W
                          << " p=" << p << '\n';
                return false;
            }
        }

        const MateID d = minsert(b, p - 1, N);
        if (mpair(d, p) != NN) continue;
        const MateID x = msetpair(d, p, RL);
        const bool full = valid_mate(x, W);
        const bool fast = rl_prefix_valid(x, W, p);
        if (full != fast) return false;
        if (!fast) continue;
        ++rl;
        const IncludeResult z = include_horizontal(x, W, p);
        if (!z.valid || !z.blocked || z.mate != b) {
            std::cerr << "RL roundtrip mismatch W=" << W << " p=" << p << '\n';
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    std::uint64_t states = 0, blocked = 0;
    std::uint64_t simple = 0, inserted = 0, rl = 0;
    for (int W = 2; W <= 10; ++W) {
        for (MateID d : enumerate_valid(W)) {
            ++states;
            if (!check_main(d, W, simple)) return 2;
        }
        for (MateID b : enumerate_valid(W - 1)) {
            ++blocked;
            if (!check_blocked(b, W, inserted, rl)) return 3;
        }
    }

    ValidGenerator gen;
    std::mt19937_64 rng(0x726f756e64747269ULL);
    constexpr std::uint64_t RANDOM_CASES = 500000;
    for (std::uint64_t i = 0; i < RANDOM_CASES; ++i) {
        const int W = 8 + 2 * int(rng() % 11);
        if (!check_main(gen.sample(W, rng), W, simple)) return 4;
        if (!check_blocked(gen.sample(W - 1, rng), W, inserted, rl)) return 5;
    }

    if (!simple || !inserted || !rl) return 6;
    std::cout << "gridfp-runtime-discovery-roundtrip-proof OK"
              << " exhaustive_W_max=10"
              << " exhaustive_states=" << states
              << " exhaustive_blocked=" << blocked
              << " random_cases=" << RANDOM_CASES
              << " simple_roundtrips=" << simple
              << " insert_n_roundtrips=" << inserted
              << " rl_valid_roundtrips=" << rl
              << " rl_prefix_exact=1"
              << " include_recheck_required=0"
              << " production_W_max=28 exact=1\n";
    return 0;
}
