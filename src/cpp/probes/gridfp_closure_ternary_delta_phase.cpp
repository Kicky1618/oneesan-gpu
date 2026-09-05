#define main gridfp_closure_ternary_delta_base_main_unused
#include "gridfp_closure_ternary_delta.cpp"
#undef main

int main() {
    static_assert(LOW_LUT_K > 0 && HIGH_LUT_K > 0);
    static_assert(LOW_LUT_K <= 14 && HIGH_LUT_K <= 14);
    static_assert(TARGET_W == LOW_LUT_K + HIGH_LUT_K + 1);

    auto low = enumerate_low_codes();
    auto high = enumerate_high_codes();
    Stats fl, rl, fh, rh;

    // Forward LOW p=1 keeps the center in the main-state word.
    for (int he = 0; he <= HIGH_LUT_K + 1; ++he) {
        for (int c = int(N); c <= int(L); ++c) {
            int h = he + (c == int(L) ? 1 : c == int(R) ? -1 : 0);
            if (h < 0 || h >= int(low.size())) continue;
            for (MateID code : low[h]) check_low(fl, code | (MateID(c) << (2 * LOW_LUT_K)), 1);
        }
    }
    // Forward LOW blocked phases insert N directly at p.
    for (int p = 2; p <= LOW_LUT_K; ++p)
        for (const auto& bucket : low) for (MateID code : bucket)
            check_low(fl, minsert(code, p, N), p);

    // Reverse LOW uses the mirrored blocked expansion.
    for (int p = 1; p <= LOW_LUT_K; ++p)
        for (const auto& bucket : low) for (MateID code : bucket)
            check_low(rl, blocked_exclude_reverse(code, LOW_LUT_K + 1, p), p);

    // Forward HIGH blocked phases insert N directly at rel.
    for (int rel = 1; rel <= HIGH_LUT_K; ++rel)
        for (const auto& bucket : high) for (MateID code : bucket)
            check_high(fh, minsert(code, rel, N), rel);

    // Reverse HIGH non-edge uses mirrored blocked expansion.
    for (int rel = 1; rel < HIGH_LUT_K; ++rel)
        for (const auto& bucket : high) for (MateID code : bucket)
            check_high(rh, blocked_exclude_reverse(code, HIGH_LUT_K + 1, rel), rel);

    // Reverse HIGH edge restores an explicit center at position 0.
    for (const auto& bucket : high) for (MateID code : bucket)
        for (int c = int(N); c <= int(L); ++c)
            check_high(rh, MateID(c) | (code << 2), HIGH_LUT_K);

    auto print = [](const char* tag, const Stats& s) {
        std::cout << ' ' << tag << "_states=" << s.states
                  << ' ' << tag << "_sources=" << s.sources
                  << ' ' << tag << "_max_sources=" << s.max_sources;
    };
    std::cout << "gridfp-closure-ternary-delta-phase OK W=" << TARGET_W;
    print("forward_low", fl);
    print("reverse_low", rl);
    print("forward_high", fh);
    print("reverse_high", rh);
    std::cout << " phase_complete=1 payload_only=1 mateid_source_rebuild_required=0\n";
    return 0;
}
