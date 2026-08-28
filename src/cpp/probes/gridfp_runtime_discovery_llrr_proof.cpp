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
        for (int rem = 1; rem <= MAX_W; ++rem) {
            for (int h = 0; h <= MAX_W; ++h) {
                dp[rem][h] = dp[rem - 1][h];
                if (h) dp[rem][h] += dp[rem - 1][h - 1];
                dp[rem][h] += dp[rem - 1][h + 1];
            }
        }
    }

    MateID sample(int W, std::mt19937_64& rng) const {
        MateID m = 0;
        int h = 1;
        for (int pos = 0; pos < W; ++pos) {
            const int rem = W - pos - 1;
            const std::uint64_t cn = dp[rem][h];
            const std::uint64_t cr = h ? dp[rem][h - 1] : 0;
            const std::uint64_t cl = dp[rem][h + 1];
            const std::uint64_t total = cn + cr + cl;
            std::uint64_t pick = rng() % total;
            MateValue v = N;
            if (pick < cn) {
                v = N;
            } else if ((pick -= cn) < cr) {
                v = R;
                --h;
            } else {
                v = L;
                ++h;
            }
            m |= MateID(v) << (2 * (W - 1 - pos));
        }
        return m;
    }
};

bool verify_candidate(
    MateID x, MateID blocked_dest, int W, int p,
    const char* kind, std::uint64_t& count
) {
    ++count;
    if (!valid_mate(x, W)) {
        std::cerr << kind << " candidate invalid W=" << W << " p=" << p << '\n';
        return false;
    }
    const IncludeResult z = include_horizontal(x, W, p);
    if (!z.valid || !z.blocked || z.mate != blocked_dest) {
        std::cerr << kind << " candidate does not return blocked dest W="
                  << W << " p=" << p << '\n';
        return false;
    }
    return true;
}

bool verify_blocked(
    MateID b, int W, std::uint64_t& ll, std::uint64_t& rr
) {
    if (!valid_mate(b, W - 1)) return false;
    for (int p = 1; p < W; ++p) {
        const MateID d = minsert(b, p - 1, N);
        if (mpair(d, p) != NN) continue;

        int bal = 0;
        for (int q = p - 2; q >= 0; --q) {
            const MateValue v = mget(d, q);
            if (bal == 0 && v == L) {
                MateID x = msetpair(d, p, LL);
                x = mset(x, q, R);
                if (!verify_candidate(x, b, W, p, "LL", ll)) return false;
            }
            if (v == L) ++bal;
            else if (v == R) --bal;
            if (bal < 0) break;
        }

        bal = 0;
        for (int q = p + 1; q < W; ++q) {
            const MateValue v = mget(d, q);
            if (bal == 0 && v == R) {
                MateID x = msetpair(d, p, RR);
                x = mset(x, q, L);
                if (!verify_candidate(x, b, W, p, "RR", rr)) return false;
            }
            if (v == R) ++bal;
            else if (v == L) --bal;
            if (bal < 0) break;
        }
    }
    return true;
}

} // namespace

int main() {
    std::uint64_t exhaustive_blocked = 0;
    std::uint64_t ll = 0, rr = 0;
    for (int W = 3; W <= 10; ++W) {
        for (MateID b : enumerate_valid(W - 1)) {
            ++exhaustive_blocked;
            if (!verify_blocked(b, W, ll, rr)) return 2;
        }
    }

    ValidGenerator gen;
    std::mt19937_64 rng(0x4c4c525250524f4fULL);
    constexpr std::uint64_t RANDOM_CASES = 500000;
    for (std::uint64_t i = 0; i < RANDOM_CASES; ++i) {
        const int W = 8 + 2 * int(rng() % 11);
        if (!verify_blocked(gen.sample(W - 1, rng), W, ll, rr)) return 3;
    }

    if (!ll || !rr) return 4;
    std::cout << "gridfp-runtime-discovery-llrr-proof OK"
              << " exhaustive_W_max=10"
              << " exhaustive_blocked=" << exhaustive_blocked
              << " random_cases=" << RANDOM_CASES
              << " ll_candidates=" << ll
              << " rr_candidates=" << rr
              << " candidate_validity=structural"
              << " include_returns_original_blocked=1"
              << " production_W_max=28 exact=1\n";
    return 0;
}
