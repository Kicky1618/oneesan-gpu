#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_inverse_probe_main_unused
#include "gridfp_reduced_production_inverse_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <set>

namespace {
using KeySet = std::set<Key>;

void assert_preimage(Key src, Key dest, int W) {
    if (src.blocked) {
        if (!valid_mate(src.mate, W - 1)) std::exit(10);
    } else if (!valid_mate(src.mate, W)) {
        std::exit(10);
    }
    const Vec got = reduced_step_basis(src, W, W - 1, false);
    if (got.find(dest) == got.end()) std::exit(11);
}

KeySet direct_blocked_preimages_high(MateID b, int W,
                                     std::uint64_t& structural_candidates) {
    KeySet out;
    const int p = W - 1;
    if (is_endpoint(mget(b, p - 1))) {
        const Key x{false, minsert(b, p, N)};
        assert_preimage(x, Key{true, b}, W);
        out.insert(x);
        ++structural_candidates;
    }

    const MateID d = minsert(b, p - 1, N);
    if (mpair(d, p) != NN) return out;

    const Key rl{false, msetpair(d, p, RL)};
    assert_preimage(rl, Key{true, b}, W);
    out.insert(rl);
    ++structural_candidates;

    int bal = 0;
    for (int q = p - 2; q >= 0; --q) {
        const MateValue v = mget(d, q);
        if (bal == 0 && v == L) {
            MateID x = msetpair(d, p, LL);
            x = mset(x, q, R);
            const Key k{false, x};
            assert_preimage(k, Key{true, b}, W);
            out.insert(k);
            ++structural_candidates;
        }
        if (v == L) ++bal;
        else if (v == R) --bal;
        if (bal < 0) break;
    }
    return out;
}

KeySet direct_inverse_high(Key dest, int W,
                           std::uint64_t& structural_candidates,
                           std::uint64_t& projected_reconstructions,
                           std::uint64_t& blocked_source_candidates) {
    if (dest.blocked)
        return direct_blocked_preimages_high(
            dest.mate, W, structural_candidates);

    KeySet out;
    const MateID d = dest.mate;
    const int p = W - 1;
    out.insert(Key{false, d});

    const MateValuePair pair = mpair(d, p);
    if (pair == LR) {
        const Key x{false, msetpair(d, p, NN)};
        assert_preimage(x, dest, W); out.insert(x); ++structural_candidates;
    }
    if (pair == NR) {
        const Key x{false, msetpair(d, p, RN)};
        assert_preimage(x, dest, W); out.insert(x); ++structural_candidates;
    }
    if (pair == NL) {
        const Key x{false, msetpair(d, p, LN)};
        assert_preimage(x, dest, W); out.insert(x); ++structural_candidates;
    }

    if (mget(d, p) == N && is_endpoint(mget(d, p - 1))) {
        const Key b{true, mshrink(d, p)};
        assert_preimage(b, dest, W);
        out.insert(b);
        ++blocked_source_candidates;
    }

    const int q = p - 1;
    const MateValuePair qp = mpair(d, q);
    if (qp == NN || qp == LR) {
        const MateID nn = qp == NN ? d : msetpair(d, q, NN);
        const MateID b = mshrink(nn, q);
        if (!valid_mate(b, W - 1) || mget(b, q - 1) != N) std::exit(12);
        ++projected_reconstructions;
        const KeySet extra = direct_blocked_preimages_high(
            b, W, structural_candidates);
        out.insert(extra.begin(), extra.end());
    }
    return out;
}

KeySet old_inverse_high(Key dest, int W) {
    const Vec old = inverse_reduced(dest, W, W - 1, false);
    KeySet out;
    for (const auto& [k, c] : old)
        if (c) out.insert(k);
    return out;
}

void check_dest(Key d, int W,
                std::uint64_t& main_cases,
                std::uint64_t& blocked_cases,
                std::uint64_t& structural_candidates,
                std::uint64_t& projected_reconstructions,
                std::uint64_t& blocked_source_candidates) {
    const KeySet want = old_inverse_high(d, W);
    const KeySet got = direct_inverse_high(
        d, W, structural_candidates, projected_reconstructions,
        blocked_source_candidates);
    if (want != got) {
        std::cerr << "mismatch W=" << W << " blocked=" << d.blocked
                  << " mate=" << d.mate << " want=" << want.size()
                  << " got=" << got.size() << '\n';
        std::exit(2);
    }
    if (d.blocked) ++blocked_cases;
    else ++main_cases;
}

using CountTable = std::array<std::array<std::uint64_t, 31>, 29>;
CountTable make_counts() {
    CountTable f{}; f[0][0] = 1;
    for (int rem = 1; rem <= 28; ++rem) {
        for (int h = 0; h <= 29; ++h) {
            std::uint64_t z = f[rem - 1][h];
            if (h > 0) z += f[rem - 1][h - 1];
            if (h < 30) z += f[rem - 1][h + 1];
            f[rem][h] = z;
        }
    }
    return f;
}

MateID unrank_valid(int W, std::uint64_t rank, const CountTable& f) {
    MateID m = 0; int h = 1;
    for (int pos = 0; pos < W; ++pos) {
        const int rem = W - pos - 1, bit = W - 1 - pos;
        const std::uint64_t n = f[rem][h];
        if (rank < n) continue;
        rank -= n;
        const std::uint64_t r = h > 0 ? f[rem][h - 1] : 0;
        if (rank < r) { m |= MateID(R) << (2 * bit); --h; continue; }
        rank -= r; m |= MateID(L) << (2 * bit); ++h;
    }
    return m;
}
} // namespace

int main() {
    std::uint64_t exhaustive_cases = 0, random_cases = 0;
    std::uint64_t main_cases = 0, blocked_cases = 0;
    std::uint64_t structural_candidates = 0, projected_reconstructions = 0;
    std::uint64_t blocked_source_candidates = 0;

    for (int W = 4; W <= 12; ++W) {
        const auto main = gen_words(W);
        const auto block = gen_words(W - 1);
        const auto dst = layout(main, block, W - 2);
        for (Key d : dst) {
            check_dest(d, W, main_cases, blocked_cases,
                       structural_candidates, projected_reconstructions,
                       blocked_source_candidates);
            ++exhaustive_cases;
        }
    }

    const CountTable f = make_counts();
    if (f[28][1] != 385719506620ULL || f[27][1] != 135015505407ULL) return 3;
    std::mt19937_64 rng(0x6869676865787069ULL);
    constexpr std::uint64_t RANDOM = 500000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        check_dest(Key{false, unrank_valid(28, rng() % f[28][1], f)}, 28,
                   main_cases, blocked_cases,
                   structural_candidates, projected_reconstructions,
                   blocked_source_candidates);
        ++random_cases;

        MateID b;
        do {
            b = unrank_valid(27, rng() % f[27][1], f);
        } while (mget(b, 25) == N);
        check_dest(Key{true, b}, 28,
                   main_cases, blocked_cases,
                   structural_candidates, projected_reconstructions,
                   blocked_source_candidates);
        ++random_cases;
    }

    if (!main_cases || !blocked_cases || !structural_candidates ||
        !projected_reconstructions || !blocked_source_candidates) return 4;
    std::cout << "gridfp-runtime-turn-direct-high-expand-inverse-proof OK"
              << " exhaustive_width_max=12 exhaustive_cases=" << exhaustive_cases
              << " random_width=28 random_cases=" << random_cases
              << " main_cases=" << main_cases
              << " blocked_cases=" << blocked_cases
              << " structural_candidates=" << structural_candidates
              << " projected_reconstructions=" << projected_reconstructions
              << " blocked_source_candidates=" << blocked_source_candidates
              << " source_scope=full forward_p=Wm1"
              << " turn_main_only_filter=exact"
              << " right_closure_candidates=0"
              << " rl_validity_checks=0"
              << " full_validity_scans_per_candidate=0"
              << " include_rechecks_per_candidate=0"
              << " inverse_set_exact=1\n";
    return 0;
}
