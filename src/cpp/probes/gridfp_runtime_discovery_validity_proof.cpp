#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

using MateID = std::uint64_t;
enum MateValue : std::uint8_t { N=0, R=1, L=2, X=3 };
enum MateValuePair : std::uint8_t {
    NN=0x0, NR=0x1, NL=0x2, NX=0x3,
    RN=0x4, RR=0x5, RL=0x6, RX=0x7,
    LN=0x8, LR=0x9, LL=0xa, LX=0xb,
    XN=0xc, XR=0xd, XL=0xe, XX=0xf
};

MateValue mget(MateID m,int k){return MateValue((m>>(2*k))&3ULL);}
MateValuePair mpair(MateID m,int p){return MateValuePair((m>>(2*(p-1)))&15ULL);}
MateID mset(MateID m,int k,MateValue v){MateID z=3ULL<<(2*k);return (m&~z)|(MateID(v)<<(2*k));}
MateID msetpair(MateID m,int p,MateValuePair v){MateID z=15ULL<<(2*(p-1));return (m&~z)|(MateID(v)<<(2*(p-1)));}
MateID minsert(MateID m,int k,MateValue v){MateID lowmask=k?((MateID(1)<<(2*k))-1ULL):0ULL;MateID lo=m&lowmask,hi=m&~lowmask;return lo|(MateID(v)<<(2*k))|(hi<<2);}

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
            const int bit = W - 1 - pos;
            m |= MateID(v) << (2 * bit);
        }
        return m;
    }
};

bool check_state(MateID d, int W, std::uint64_t& simple, std::uint64_t& rl) {
    if (!valid_mate(d, W)) return false;
    for (int p = 1; p < W; ++p) {
        const MateValuePair w = mpair(d, p);
        MateID x = 0;
        bool has_simple = true;
        if (w == LR) x = msetpair(d, p, NN);
        else if (w == NR) x = msetpair(d, p, RN);
        else if (w == NL) x = msetpair(d, p, LN);
        else has_simple = false;
        if (has_simple) {
            ++simple;
            if (!valid_mate(x, W)) {
                std::cerr << "simple inverse candidate invalid W=" << W
                          << " p=" << p << " pair=" << int(w) << '\n';
                return false;
            }
        }
    }
    return true;
}

bool check_blocked_base(MateID b, int W, std::uint64_t& inserted,
                        std::uint64_t& rl) {
    if (!valid_mate(b, W - 1)) return false;
    for (int p = 1; p < W; ++p) {
        if (mget(b, p - 1) == L || mget(b, p - 1) == R) {
            ++inserted;
            if (!valid_mate(minsert(b, p, N), W)) {
                std::cerr << "insert-N candidate invalid W=" << W
                          << " p=" << p << '\n';
                return false;
            }
        }
        const MateID d = minsert(b, p - 1, N);
        if (mpair(d, p) != NN) continue;
        const MateID x = msetpair(d, p, RL);
        ++rl;
        const bool full = valid_mate(x, W);
        const bool fast = rl_prefix_valid(x, W, p);
        if (full != fast) {
            std::cerr << "RL prefix mismatch W=" << W << " p=" << p
                      << " full=" << full << " fast=" << fast << '\n';
            return false;
        }
    }
    return true;
}

} // namespace

int main() {
    std::uint64_t exhaustive_states = 0;
    std::uint64_t exhaustive_blocked = 0;
    std::uint64_t simple_candidates = 0;
    std::uint64_t inserted_candidates = 0;
    std::uint64_t rl_candidates = 0;

    for (int W = 2; W <= 10; ++W) {
        const auto states = enumerate_valid(W);
        for (MateID d : states) {
            ++exhaustive_states;
            if (!check_state(d, W, simple_candidates, rl_candidates)) return 2;
        }
        const auto blocked = enumerate_valid(W - 1);
        for (MateID b : blocked) {
            ++exhaustive_blocked;
            if (!check_blocked_base(
                    b, W, inserted_candidates, rl_candidates)) return 3;
        }
    }

    ValidGenerator gen;
    std::mt19937_64 rng(0x6d61746576616cULL);
    constexpr std::uint64_t RANDOM_CASES = 500000;
    for (std::uint64_t i = 0; i < RANDOM_CASES; ++i) {
        const int W = 8 + 2 * int(rng() % 11);
        const MateID d = gen.sample(W, rng);
        if (!check_state(d, W, simple_candidates, rl_candidates)) return 4;
        const MateID b = gen.sample(W - 1, rng);
        if (!check_blocked_base(b, W, inserted_candidates, rl_candidates)) return 5;
    }

    std::cout << "gridfp-runtime-discovery-validity-proof OK"
              << " exhaustive_W_max=10"
              << " exhaustive_states=" << exhaustive_states
              << " exhaustive_blocked=" << exhaustive_blocked
              << " random_cases=" << RANDOM_CASES
              << " simple_known_valid=" << simple_candidates
              << " insert_n_known_valid=" << inserted_candidates
              << " rl_prefix_exact=" << rl_candidates
              << " production_W_max=28 exact=1\n";
    return 0;
}
