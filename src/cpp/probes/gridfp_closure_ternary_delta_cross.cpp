#define main gridfp_closure_ternary_delta_base_main_unused
#include "gridfp_closure_ternary_delta.cpp"
#undef main
#include "../../common/gridfp_closure_inverse.hpp"

static void check_low_cross_delta(Stats& s, MateID d, int p) {
    MateID source = 0;
    int depth = low_cross_preimage_partial(d, LOW_LUT_K + 1, p, source);
    if (depth <= 0) return;
    MateID expected = msetpair(d, p, RR);
    if (source != expected) {
        std::cerr << "LOW cross source mismatch p=" << p << " depth=" << depth
                  << " got=" << source << " expected=" << expected << '\n';
        std::exit(3);
    }
    ++s.states;
    std::uint64_t before = s.sources;
    check_low_source(s, d, p, RR, -1, N, "cross-rr");
    s.max_sources = std::max(s.max_sources, std::uint32_t(s.sources - before));
}

static void check_high_cross_delta(Stats& s, MateID d, int p) {
    MateID source = 0;
    int depth = high_cross_preimage_partial(d, HIGH_LUT_K + 1, p, source);
    if (depth <= 0) return;
    MateID expected = msetpair(d, p, LL);
    if (source != expected) {
        std::cerr << "HIGH cross source mismatch p=" << p << " depth=" << depth
                  << " got=" << source << " expected=" << expected << '\n';
        std::exit(4);
    }
    ++s.states;
    std::uint64_t before = s.sources;
    check_high_source(s, d, p, LL, -1, N, "cross-ll");
    s.max_sources = std::max(s.max_sources, std::uint32_t(s.sources - before));
}

int main() {
    auto low = enumerate_low_codes();
    auto high = enumerate_high_codes();
    Stats fl, rl, fh, rh;

    for (int he = 0; he <= HIGH_LUT_K + 1; ++he) {
        for (int c = int(N); c <= int(L); ++c) {
            int h = he + (c == int(L) ? 1 : c == int(R) ? -1 : 0);
            if (h < 0 || h >= int(low.size())) continue;
            for (MateID code : low[h])
                check_low_cross_delta(fl, code | (MateID(c) << (2 * LOW_LUT_K)), 1);
        }
    }
    for (int p = 2; p <= LOW_LUT_K; ++p)
        for (const auto& bucket : low) for (MateID code : bucket)
            check_low_cross_delta(fl, minsert(code, p, N), p);
    for (int p = 1; p <= LOW_LUT_K; ++p)
        for (const auto& bucket : low) for (MateID code : bucket)
            check_low_cross_delta(rl, blocked_exclude_reverse(code, LOW_LUT_K + 1, p), p);

    for (int rel = 1; rel <= HIGH_LUT_K; ++rel)
        for (const auto& bucket : high) for (MateID code : bucket)
            check_high_cross_delta(fh, minsert(code, rel, N), rel);
    for (int rel = 1; rel < HIGH_LUT_K; ++rel)
        for (const auto& bucket : high) for (MateID code : bucket)
            check_high_cross_delta(rh, blocked_exclude_reverse(code, HIGH_LUT_K + 1, rel), rel);
    for (const auto& bucket : high) for (MateID code : bucket)
        for (int c = int(N); c <= int(L); ++c)
            check_high_cross_delta(rh, MateID(c) | (code << 2), HIGH_LUT_K);

    std::cout << "gridfp-closure-ternary-delta-cross OK W=" << TARGET_W
              << " forward_low_cross=" << fl.sources
              << " reverse_low_cross=" << rl.sources
              << " forward_high_cross=" << fh.sources
              << " reverse_high_cross=" << rh.sources
              << " cross_source_pair_exact=1 cross_delta_exact=1\n";
    return 0;
}
