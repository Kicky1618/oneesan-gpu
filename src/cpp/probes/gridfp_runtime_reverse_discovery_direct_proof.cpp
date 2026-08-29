#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_channel_probe_main_unused
#include "gridfp_reduced_production_channel_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <set>

namespace {
using KeySet = std::set<Key>;

bool valid_main(MateID m, int W) {
    int h = 1;
    for (int pos = 0; pos < W; ++pos) {
        const MateValue v = mget(m, W - 1 - pos);
        if (v == X) return false;
        if (v == R) --h;
        else if (v == L) ++h;
        if (h < 0) return false;
    }
    return h == 0;
}

bool rl_valid_direct(MateID x, int W, int p) {
    int h = 1;
    for (int q = W - 1; q > p; --q) {
        const MateValue v = mget(x, q);
        if (v == R) --h;
        else if (v == L) ++h;
    }
    return h > 0;
}

KeySet direct_blocked_reverse(MateID b, int W, int p) {
    KeySet out;
    if (is_endpoint(mget(b, p - 1)))
        out.insert(Key{false, minsert(b, p - 1, N)});

    const MateID d = minsert(b, p, N);
    if (mpair(d, p) != NN) return out;

    const MateID rl = msetpair(d, p, RL);
    if (rl_valid_direct(rl, W, p)) out.insert(Key{false, rl});

    int bal = 0;
    for (int q = p - 2; q >= 0; --q) {
        const MateValue v = mget(d, q);
        if (bal == 0 && v == L) {
            MateID x = msetpair(d, p, LL);
            x = mset(x, q, R);
            out.insert(Key{false, x});
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
            out.insert(Key{false, x});
        }
        if (v == R) ++bal;
        else if (v == L) --bal;
        if (bal < 0) break;
    }
    return out;
}

KeySet direct_reverse_inverse(Key dest, int W, int p) {
    if (dest.blocked) return direct_blocked_reverse(dest.mate, W, p);

    KeySet out;
    const MateID d = dest.mate;
    out.insert(Key{false, d});
    const MateValuePair w = mpair(d, p);
    if (w == LR) out.insert(Key{false, msetpair(d, p, NN)});
    if (w == LN) out.insert(Key{false, msetpair(d, p, NL)});
    if (w == RN) out.insert(Key{false, msetpair(d, p, NR)});

    if (mget(d, p - 1) == N && is_endpoint(mget(d, p)))
        out.insert(Key{true, mshrink(d, p - 1)});

    const int q = p + 1;
    if (q < W) {
        const MateValuePair qp = mpair(d, q);
        if (qp == NN || qp == LR) {
            const MateID nn = qp == NN ? d : msetpair(d, q, NN);
            const MateID b = mshrink(nn, p);
            const KeySet extra = direct_blocked_reverse(b, W, p);
            out.insert(extra.begin(), extra.end());
        }
    }
    return out;
}

bool forward_rl_fast_valid(MateID x, int W, int p) {
    int h = 1;
    for (int q = W - 1; q > p; --q) {
        const MateValue v = mget(x, q);
        if (v == R) --h;
        else if (v == L) ++h;
    }
    return h > 0;
}

KeySet forward_fast_blocked(MateID b, int W, int p) {
    KeySet out;
    if (is_endpoint(mget(b, p - 1)))
        out.insert(Key{false, minsert(b, p, N)});
    const MateID d = minsert(b, p - 1, N);
    if (mpair(d, p) != NN) return out;
    const MateID rl = msetpair(d, p, RL);
    if (forward_rl_fast_valid(rl, W, p)) out.insert(Key{false, rl});
    int bal = 0;
    for (int q = p - 2; q >= 0; --q) {
        const MateValue v = mget(d, q);
        if (bal == 0 && v == L) {
            MateID x = msetpair(d, p, LL); x = mset(x, q, R);
            out.insert(Key{false, x});
        }
        if (v == L) ++bal; else if (v == R) --bal;
        if (bal < 0) break;
    }
    bal = 0;
    for (int q = p + 1; q < W; ++q) {
        const MateValue v = mget(d, q);
        if (bal == 0 && v == R) {
            MateID x = msetpair(d, p, RR); x = mset(x, q, L);
            out.insert(Key{false, x});
        }
        if (v == R) ++bal; else if (v == L) --bal;
        if (bal < 0) break;
    }
    return out;
}

KeySet forward_fast_inverse(Key dest, int W, int p) {
    if (dest.blocked) return forward_fast_blocked(dest.mate, W, p);
    KeySet out;
    const MateID d = dest.mate;
    out.insert(Key{false, d});
    const MateValuePair w = mpair(d, p);
    if (w == LR) out.insert(Key{false, msetpair(d, p, NN)});
    if (w == NR) out.insert(Key{false, msetpair(d, p, RN)});
    if (w == NL) out.insert(Key{false, msetpair(d, p, LN)});
    if (mget(d, p) == N && is_endpoint(mget(d, p - 1)))
        out.insert(Key{true, mshrink(d, p)});
    const int q = p - 1;
    if (q > 0) {
        const MateValuePair qp = mpair(d, q);
        if (qp == NN || qp == LR) {
            const MateID nn = qp == NN ? d : msetpair(d, q, NN);
            const MateID b = mshrink(nn, q);
            const KeySet extra = forward_fast_blocked(b, W, p);
            out.insert(extra.begin(), extra.end());
        }
    }
    return out;
}

Key mirror_local(Key k, int W) {
    return Key{k.blocked, mirror_mate(k.mate, k.blocked ? W - 1 : W)};
}

KeySet old_mirror_fast_reverse(Key dest, int W, int p) {
    const Key md = mirror_local(dest, W);
    const KeySet f = forward_fast_inverse(md, W, W - p);
    KeySet out;
    for (Key k : f) out.insert(mirror_local(k, W));
    return out;
}

void check_candidate(Key src, Key dest, int W, int p) {
    if (src.blocked) {
        if (!valid_main(src.mate, W - 1)) std::exit(20);
    } else if (!valid_main(src.mate, W)) {
        std::exit(21);
    }
    const Vec edge = reduced_step_basis(src, W, p, true);
    const auto it = edge.find(dest);
    if (it == edge.end() || !it->second) std::exit(22);
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
    std::uint64_t exhaustive_cases = 0;
    std::uint64_t old_mismatch_w8 = 0, old_extra_w8 = 0;
    std::uint64_t old_mismatch_w10 = 0, old_extra_w10 = 0;

    for (int W = 5; W <= 12; ++W) {
        const auto main = gen_words(W);
        const auto block = gen_words(W - 1);
        for (int p = 1; p <= W - 3; ++p) {
            const auto src = layout(main, block, p);
            const auto dst = layout(main, block, p + 1);
            std::map<Key, KeySet> incoming;
            for (Key s : src) {
                const Vec edge = reduced_step_basis(s, W, p, true);
                for (const auto& [d, c] : edge)
                    if (c) incoming[d].insert(s);
            }
            for (Key d : dst) {
                const KeySet empty;
                const auto it = incoming.find(d);
                const KeySet& want = it == incoming.end() ? empty : it->second;
                const KeySet got = direct_reverse_inverse(d, W, p);
                if (got != want) {
                    std::cerr << "direct mismatch W=" << W << " p=" << p
                              << " blocked=" << d.blocked << " mate=" << d.mate
                              << " want=" << want.size() << " got=" << got.size() << '\n';
                    return 2;
                }
                if (W == 8 || W == 10) {
                    const KeySet old = old_mirror_fast_reverse(d, W, p);
                    if (old != want) {
                        const std::uint64_t extra = [&] {
                            std::uint64_t n = 0;
                            for (Key k : old) if (!want.count(k)) ++n;
                            return n;
                        }();
                        if (W == 8) { ++old_mismatch_w8; old_extra_w8 += extra; }
                        else { ++old_mismatch_w10; old_extra_w10 += extra; }
                    }
                }
                ++exhaustive_cases;
            }
        }
    }

    if (old_mismatch_w8 != 237 || old_extra_w8 != 237 ||
        old_mismatch_w10 != 1813 || old_extra_w10 != 1813) {
        std::cerr << "old mirror-fast regression counts changed"
                  << " w8=" << old_mismatch_w8 << '/' << old_extra_w8
                  << " w10=" << old_mismatch_w10 << '/' << old_extra_w10 << '\n';
        return 3;
    }

    const CountTable f = make_counts();
    if (f[28][1] != 385719506620ULL || f[27][1] != 135015505407ULL) return 4;
    std::mt19937_64 rng(0x7265766469736332ULL);
    constexpr std::uint64_t RANDOM = 500000;
    std::uint64_t random_candidates = 0;
    for (std::uint64_t i = 0; i < RANDOM; ++i) {
        const int p = 1 + int(rng() % 25); // production reverse interior 1..25
        Key d{false, unrank_valid(28, rng() % f[28][1], f)};
        for (Key s : direct_reverse_inverse(d, 28, p)) {
            check_candidate(s, d, 28, p);
            ++random_candidates;
        }
    }
    if (!random_candidates) return 5;

    std::cout << "gridfp-runtime-reverse-discovery-direct-proof OK"
              << " exhaustive_width_max=12 exhaustive_cases=" << exhaustive_cases
              << " random_width=28 random_cases=" << RANDOM
              << " random_candidates=" << random_candidates
              << " old_mirror_fast_w8_mismatches=" << old_mismatch_w8
              << " old_mirror_fast_w8_extra_sources=" << old_extra_w8
              << " old_mirror_fast_w10_mismatches=" << old_mismatch_w10
              << " old_mirror_fast_w10_extra_sources=" << old_extra_w10
              << " direct_destination_mirror_passes=0"
              << " direct_source_mirror_passes=0"
              << " rl_validity=high_prefix"
              << " inverse_set_exact=1\n";
    return 0;
}
