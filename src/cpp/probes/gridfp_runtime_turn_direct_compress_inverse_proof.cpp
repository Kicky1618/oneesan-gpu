#include "../../common/gridfp_transition.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>
#include <vector>

namespace {
using namespace oneesan::gridfp;

struct Key {
    MateID mate = 0;
    bool blocked = false;
    bool operator<(const Key& o) const {
        return blocked != o.blocked ? blocked < o.blocked : mate < o.mate;
    }
};

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

void old_try(std::set<Key>& out, MateID x, MateID dest, int W) {
    if (!valid_mate(x, W)) return;
    const IncludeResult z = include_horizontal(x, W, 1);
    if (z.valid && !z.blocked && z.mate == dest)
        out.insert(Key{x, false});
}

std::set<Key> old_inverse(MateID d, int W) {
    std::set<Key> out;
    out.insert(Key{d, false});
    const MateValuePair pair = mpair(d, 1);
    if (pair == LR) old_try(out, msetpair(d, 1, NN), d, W);
    if (pair == RN) old_try(out, msetpair(d, 1, NR), d, W);
    if (pair == LN) old_try(out, msetpair(d, 1, NL), d, W);
    if (pair == NR) old_try(out, msetpair(d, 1, RN), d, W);
    if (pair == NL) old_try(out, msetpair(d, 1, LN), d, W);

    if (pair == NN) {
        old_try(out, msetpair(d, 1, RL), d, W);
        int bal = 0;
        for (int q = 2; q < W; ++q) {
            const MateValue v = mget(d, q);
            if (bal == 0 && v == R) {
                MateID x = msetpair(d, 1, RR);
                x = mset(x, q, L);
                old_try(out, x, d, W);
            }
            if (v == R) ++bal;
            else if (v == L) --bal;
            if (bal < 0) break;
        }
    }

    if (mget(d, 1) == N && is_endpoint(mget(d, 0))) {
        const MateID b = mshrink(d, 1);
        if (valid_mate(b, W - 1) && mget(b, 0) != N &&
            blocked_exclude(b, 1) == d)
            out.insert(Key{b, true});
    }
    return out;
}

std::set<Key> direct_inverse(MateID d, int W) {
    std::set<Key> out;
    out.insert(Key{d, false});
    const MateValuePair pair = mpair(d, 1);

    // For a valid production destination, only NN/NR/RN/LR/RR can occur at
    // the final two positions. The three local invertible rewrites below are
    // automatically valid and map back to d.
    if (pair == LR) out.insert(Key{msetpair(d, 1, NN), false});
    if (pair == RN) out.insert(Key{msetpair(d, 1, NR), false});
    if (pair == NR) out.insert(Key{msetpair(d, 1, RN), false});

    if (pair == NN) {
        // RL at the physical low boundary would take height 0 to -1, so it is
        // never a valid source. RR closure candidates selected by this balance
        // scan are valid by construction and need no include recheck.
        int bal = 0;
        for (int q = 2; q < W; ++q) {
            const MateValue v = mget(d, q);
            if (bal == 0 && v == R) {
                MateID x = msetpair(d, 1, RR);
                x = mset(x, q, L);
                out.insert(Key{x, false});
            }
            if (v == R) ++bal;
            else if (v == L) --bal;
            if (bal < 0) break;
        }
    }

    // The only valid destination with bit1=N and bit0 occupied is NR. Removing
    // that N therefore directly recovers the retained blocked predecessor.
    if (pair == NR)
        out.insert(Key{mshrink(d, 1), true});
    return out;
}

std::vector<MateID> generate_valid(int W) {
    std::vector<MateID> out;
    auto rec = [&](auto&& self, int pos, int h, MateID m) -> void {
        const int rem = W - pos;
        if (h < 0 || h > rem) return;
        if (pos == W) {
            if (h == 0) out.push_back(m);
            return;
        }
        const int bit = W - 1 - pos;
        self(self, pos + 1, h, m);
        if (h > 0)
            self(self, pos + 1, h - 1, m | (MateID(R) << (2 * bit)));
        self(self, pos + 1, h + 1, m | (MateID(L) << (2 * bit)));
    };
    rec(rec, 0, 1, 0);
    return out;
}

using CountTable = std::array<std::array<std::uint64_t, 31>, 29>;
CountTable make_counts() {
    CountTable f{};
    f[0][0] = 1;
    for (int rem = 1; rem <= 28; ++rem) {
        for (int h = 0; h <= 29; ++h) {
            std::uint64_t z = f[rem - 1][h];
            if (h > 0) z += f[rem - 1][h - 1];
            if (h + 1 <= 30) z += f[rem - 1][h + 1];
            f[rem][h] = z;
        }
    }
    return f;
}

MateID unrank_valid(int W, std::uint64_t rank, const CountTable& f) {
    MateID m = 0;
    int h = 1;
    for (int pos = 0; pos < W; ++pos) {
        const int rem = W - pos - 1;
        const int bit = W - 1 - pos;
        const std::uint64_t ncount = f[rem][h];
        if (rank < ncount) continue;
        rank -= ncount;
        const std::uint64_t rcount = h > 0 ? f[rem][h - 1] : 0;
        if (rank < rcount) {
            m |= MateID(R) << (2 * bit);
            --h;
            continue;
        }
        rank -= rcount;
        m |= MateID(L) << (2 * bit);
        ++h;
    }
    return m;
}

void check_one(MateID d, int W,
               std::uint64_t& nn_cases,
               std::uint64_t& rr_candidates,
               std::uint64_t& nr_cases) {
    if (!valid_mate(d, W)) std::exit(10);
    const auto old = old_inverse(d, W);
    const auto fast = direct_inverse(d, W);
    if (old != fast) {
        std::cerr << "mismatch W=" << W << " mate=" << d
                  << " old=" << old.size() << " fast=" << fast.size() << '\n';
        std::exit(2);
    }
    const MateValuePair pair = mpair(d, 1);
    if (pair == NN) {
        ++nn_cases;
        if (valid_mate(msetpair(d, 1, RL), W)) std::exit(3);
        int bal = 0;
        for (int q = 2; q < W; ++q) {
            const MateValue v = mget(d, q);
            if (bal == 0 && v == R) {
                MateID x = msetpair(d, 1, RR);
                x = mset(x, q, L);
                if (!valid_mate(x, W)) std::exit(4);
                const IncludeResult z = include_horizontal(x, W, 1);
                if (!z.valid || z.blocked || z.mate != d) std::exit(5);
                ++rr_candidates;
            }
            if (v == R) ++bal;
            else if (v == L) --bal;
            if (bal < 0) break;
        }
    }
    if (pair == NR) {
        ++nr_cases;
        const MateID b = mshrink(d, 1);
        if (!valid_mate(b, W - 1) || mget(b, 0) == N ||
            blocked_exclude(b, 1) != d) std::exit(6);
    }
    // LN and NL cannot be suffixes of a valid one-defect Motzkin word.
    if (pair == LN || pair == NL || pair == LL || pair == RL) std::exit(7);
}

} // namespace

int main() {
    std::uint64_t exhaustive_cases = 0;
    std::uint64_t random_cases = 0;
    std::uint64_t nn_cases = 0;
    std::uint64_t rr_candidates = 0;
    std::uint64_t nr_cases = 0;

    for (int W = 2; W <= 12; ++W) {
        const auto words = generate_valid(W);
        for (MateID d : words) {
            check_one(d, W, nn_cases, rr_candidates, nr_cases);
            ++exhaustive_cases;
        }
    }

    const CountTable f = make_counts();
    const std::uint64_t total28 = f[28][1];
    if (total28 != 385719506620ULL) return 8;
    std::mt19937_64 rng(0x7475726e64697265ULL);
    constexpr std::uint64_t RANDOM = 1000000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const MateID d = unrank_valid(28, rng() % total28, f);
        check_one(d, 28, nn_cases, rr_candidates, nr_cases);
        ++random_cases;
    }

    if (!nn_cases || !rr_candidates || !nr_cases) return 9;
    std::cout << "gridfp-runtime-turn-direct-compress-inverse-proof OK"
              << " exhaustive_width_max=12 exhaustive_cases=" << exhaustive_cases
              << " random_width=28 random_cases=" << random_cases
              << " total_W28=" << total28
              << " nn_cases=" << nn_cases
              << " rr_candidates=" << rr_candidates
              << " nr_cases=" << nr_cases
              << " local_direct=LR_NN,RN_NR,NR_RN"
              << " nn_rl_always_invalid=1"
              << " nn_rr_scan_candidates_direct=1"
              << " blocked_NR_direct=1"
              << " full_validity_scans_per_candidate=0"
              << " include_rechecks_per_candidate=0"
              << " inverse_set_exact=1\n";
    return 0;
}
