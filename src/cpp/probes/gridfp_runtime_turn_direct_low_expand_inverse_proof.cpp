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

bool valid_reverse_mate(MateID m, int W) {
    int h = 1;
    for (int q = 0; q < W; ++q) {
        const MateValue v = mget(m, q);
        if (v == X) return false;
        if (v == L) --h;
        else if (v == R) ++h;
        if (h < 0) return false;
    }
    return h == 0;
}

void emit_if_reverse_blocked_preimage(
    KeySet& out, MateID x, MateID blocked_dest, int W
) {
    if (!valid_reverse_mate(x, W)) return;
    const IncludeResult z = include_horizontal_reverse(x, W, 1);
    if (z.valid && z.blocked && z.mate == blocked_dest)
        out.insert(Key{false, x});
}

KeySet direct_blocked_preimages_reverse_low(MateID b, int W) {
    KeySet out;
    if (is_endpoint(mget(b, 0)))
        emit_if_reverse_blocked_preimage(out, minsert(b, 0, N), b, W);

    const MateID d = minsert(b, 1, N);
    if (mpair(d, 1) != NN) return out;

    emit_if_reverse_blocked_preimage(out, msetpair(d, 1, RL), b, W);
    int bal = 0;
    for (int q = 2; q < W; ++q) {
        const MateValue v = mget(d, q);
        if (bal == 0 && v == R) {
            MateID x = msetpair(d, 1, RR);
            x = mset(x, q, L);
            emit_if_reverse_blocked_preimage(out, x, b, W);
        }
        if (v == R) ++bal;
        else if (v == L) --bal;
        if (bal < 0) break;
    }
    return out;
}

void try_main_reverse_low(KeySet& out, MateID x, MateID dest, int W) {
    if (!valid_reverse_mate(x, W)) return;
    const Vec got = reduced_step_basis(Key{false, x}, W, 1, true);
    if (got.find(Key{false, dest}) != got.end()) out.insert(Key{false, x});
}

KeySet direct_main_only_inverse_reverse_low(Key dest, int W) {
    if (dest.blocked)
        return direct_blocked_preimages_reverse_low(dest.mate, W);

    KeySet out;
    const MateID d = dest.mate;
    out.insert(Key{false, d});

    const MateValuePair pair = mpair(d, 1);
    if (pair == LR) try_main_reverse_low(out, msetpair(d, 1, NN), d, W);
    if (pair == LN) try_main_reverse_low(out, msetpair(d, 1, NL), d, W);
    if (pair == RN) try_main_reverse_low(out, msetpair(d, 1, NR), d, W);

    if (W > 2) {
        const MateValuePair qp = mpair(d, 2);
        if (qp == NN || qp == LR) {
            const MateID nn = qp == NN ? d : msetpair(d, 2, NN);
            const MateID b = mshrink(nn, 1);
            if (valid_reverse_mate(b, W - 1) && mget(b, 1) == N) {
                const KeySet extra = direct_blocked_preimages_reverse_low(b, W);
                out.insert(extra.begin(), extra.end());
            }
        }
    }
    return out;
}

KeySet old_main_only_inverse_reverse_low(Key dest, int W) {
    const Vec old = inverse_reduced(dest, W, 1, true);
    KeySet out;
    for (const auto& [k, c] : old)
        if (c && !k.blocked) out.insert(k);
    return out;
}

void check_dest(Key d, int W,
                std::uint64_t& main_cases,
                std::uint64_t& blocked_cases,
                std::uint64_t& projected_cases) {
    const KeySet want = old_main_only_inverse_reverse_low(d, W);
    const KeySet got = direct_main_only_inverse_reverse_low(d, W);
    if (want != got) {
        std::cerr << "mismatch W=" << W << " blocked=" << d.blocked
                  << " mate=" << d.mate << " want=" << want.size()
                  << " got=" << got.size() << '\n';
        std::exit(2);
    }
    if (d.blocked) ++blocked_cases;
    else {
        ++main_cases;
        if (W > 2) {
            const MateValuePair qp = mpair(d.mate, 2);
            if (qp == NN || qp == LR) ++projected_cases;
        }
    }
}

using CountTable = std::array<std::array<std::uint64_t, 31>, 29>;
CountTable make_counts() {
    CountTable f{};
    f[0][0] = 1;
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
    MateID m = 0;
    int h = 1;
    for (int pos = 0; pos < W; ++pos) {
        const int rem = W - pos - 1;
        const int bit = W - 1 - pos;
        const std::uint64_t n = f[rem][h];
        if (rank < n) continue;
        rank -= n;
        const std::uint64_t r = h > 0 ? f[rem][h - 1] : 0;
        if (rank < r) {
            m |= MateID(R) << (2 * bit);
            --h;
            continue;
        }
        rank -= r;
        m |= MateID(L) << (2 * bit);
        ++h;
    }
    return m;
}

} // namespace

int main() {
    std::uint64_t exhaustive_cases = 0;
    std::uint64_t random_cases = 0;
    std::uint64_t main_cases = 0;
    std::uint64_t blocked_cases = 0;
    std::uint64_t projected_cases = 0;

    for (int W = 4; W <= 12; ++W) {
        const auto main = gen_words(W);
        const auto block = gen_words(W - 1);
        const auto forward_dst = layout(main, block, W - 2);
        for (Key d : forward_dst) {
            check_dest(mirror_key(d, W), W,
                       main_cases, blocked_cases, projected_cases);
            ++exhaustive_cases;
        }
    }

    const CountTable f = make_counts();
    if (f[28][1] != 385719506620ULL || f[27][1] != 135015505407ULL) return 3;
    std::mt19937_64 rng(0x6c6f77657870696eULL);
    constexpr std::uint64_t RANDOM = 500000;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const MateID fm = unrank_valid(28, rng() % f[28][1], f);
        check_dest(mirror_key(Key{false, fm}, 28), 28,
                   main_cases, blocked_cases, projected_cases);
        ++random_cases;

        MateID fb;
        do {
            fb = unrank_valid(27, rng() % f[27][1], f);
        } while (mget(fb, 25) == N);
        check_dest(mirror_key(Key{true, fb}, 28), 28,
                   main_cases, blocked_cases, projected_cases);
        ++random_cases;
    }

    if (!main_cases || !blocked_cases || !projected_cases) return 4;
    std::cout << "gridfp-runtime-turn-direct-low-expand-inverse-proof OK"
              << " exhaustive_width_max=12 exhaustive_cases=" << exhaustive_cases
              << " random_width=28 random_cases=" << random_cases
              << " main_cases=" << main_cases
              << " blocked_cases=" << blocked_cases
              << " projected_cases=" << projected_cases
              << " source_scope=main_only"
              << " reverse_basis=mirrored_Q_Wm2"
              << " direct_reverse_validity=low_to_high"
              << " direct_reverse_p=1"
              << " destination_mirror_passes=0 source_mirror_passes=0"
              << " candidate_validation=direct_reverse_step"
              << " inverse_set_exact=1\n";
    return 0;
}
