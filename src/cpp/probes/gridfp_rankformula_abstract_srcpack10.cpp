#define main gridfp_rankformula_abstract_lut_main_unused
#include "gridfp_rankformula_abstract_lut.cpp"
#undef main

static uint64_t choose_u64_srcpack(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    uint64_t z = 1;
    for (int i = 1; i <= k; ++i) z = z * uint64_t(n - k + i) / uint64_t(i);
    return z;
}

static uint8_t select_mask_srcpack(uint32_t lp, int n, uint32_t depth) {
    uint32_t state = depth, li = 0;
    uint8_t out = 0;
    for (int ord = 0; ord < n; ++ord) {
        if ((lp >> ord) & 1u) {
            if (state == 1u) out |= uint8_t(1u << li);
            ++li; ++state;
        } else {
            if (state == 1u) break;
            --state;
        }
    }
    return out;
}

int main() {
    constexpr uint32_t STATES = 7060u;
    constexpr uint32_t OVERFLOW_N = 429u;
    std::array<uint32_t, STATES> src03{}, src36{};
    std::array<uint16_t, OVERFLOW_N> overflow{};
    uint32_t di = 0, overflow_states = 0, transitions = 0, max_rank = 0;
    std::array<uint64_t, 8> lcount_hist{};

    for (int n = 0; n <= L; ++n) {
        for (int h = 0; h < 16; ++h) {
            const uint32_t cnt = ballot_suffix(n, h);
            for (uint32_t local = 0; local < cnt; ++local, ++di) {
                const uint32_t lp = abstract_lpattern(n, h, local);
                if (lp == INVALID || di >= STATES) return 2;
                const uint32_t lc = uint32_t(__builtin_popcount(lp));
                if (lc > 7u) return 3;
                ++lcount_hist[lc];
                uint32_t li = 0;
                for (int ord = 0; ord < n; ++ord) {
                    if (((lp >> ord) & 1u) == 0u) continue;
                    const uint32_t sr = abstract_rank(n, h + 2, lp & ~(1u << ord));
                    if (sr == INVALID || sr > 1000u) return 4;
                    max_rank = std::max(max_rank, sr);
                    if (li < 3u) {
                        src03[di] |= sr << (10u * li);
                    } else if (li < 6u) {
                        src36[di] |= sr << (10u * (li - 3u));
                    } else {
                        if (li != 6u || n != 14 || h != 0 || local >= OVERFLOW_N) return 5;
                        overflow[local] = uint16_t(sr);
                    }
                    ++li;
                    ++transitions;
                }
                if (li != lc) return 6;
                if (lc == 7u) ++overflow_states;
            }
        }
    }
    if (di != STATES || transitions != 32743u || overflow_states != OVERFLOW_N || max_rank != 1000u)
        return 7;

    // Decode every source rank from the split packed representation and compare
    // against an independent ballot-rank computation.
    di = 0;
    uint32_t exact = 0;
    uint64_t production_nonzero = 0, production_selected = 0;
    uint64_t word0_loads = 0, word1_loads = 0, overflow_loads = 0;
    for (int n = 0; n <= L; ++n) {
        const uint64_t weight = choose_u64_srcpack(L, n);
        for (int h = 0; h < 16; ++h) {
            const uint32_t cnt = ballot_suffix(n, h);
            for (uint32_t local = 0; local < cnt; ++local, ++di) {
                const uint32_t lp = abstract_lpattern(n, h, local);
                uint32_t li = 0;
                for (int ord = 0; ord < n; ++ord) {
                    if (((lp >> ord) & 1u) == 0u) continue;
                    const uint32_t want = abstract_rank(n, h + 2, lp & ~(1u << ord));
                    const uint32_t got = li < 3u
                        ? ((src03[di] >> (10u * li)) & 1023u)
                        : li < 6u
                            ? ((src36[di] >> (10u * (li - 3u))) & 1023u)
                            : uint32_t(overflow[local]);
                    if (got != want) return 8;
                    ++li;
                    ++exact;
                }
                for (uint32_t depth = 1; depth <= 13u; ++depth) {
                    const uint8_t sm = select_mask_srcpack(lp, n, depth);
                    if (!sm) continue;
                    production_nonzero += weight;
                    production_selected += uint64_t(__builtin_popcount(uint32_t(sm))) * weight;
                    if (sm & 0x07u) word0_loads += weight;
                    if (sm & 0x38u) word1_loads += weight;
                    if (sm & 0x40u) overflow_loads += weight;
                }
            }
        }
    }

    const std::array<uint64_t,8> want_hist{15,91,352,915,1652,2044,1562,429};
    if (exact != 32743u || lcount_hist != want_hist ||
        production_nonzero != 1757173ull || production_selected != 2492769ull ||
        word0_loads != 1675973ull || word1_loads != 241173ull || overflow_loads != 132ull)
        return 9;
    const uint64_t word_bytes = uint64_t(STATES) * 2ull * sizeof(uint32_t);
    const uint64_t overflow_bytes = uint64_t(OVERFLOW_N) * sizeof(uint16_t);
    const uint64_t dynamic_split_bytes = 4ull * (word0_loads + word1_loads) + 2ull * overflow_loads;
    const uint64_t dynamic_fixed64_bytes = 8ull * production_nonzero;
    const uint64_t dynamic_roffsrc_bytes = 2ull * production_nonzero + 2ull * production_selected;
    if (word_bytes != 56480ull || overflow_bytes != 858ull ||
        dynamic_split_bytes != 7668848ull || dynamic_fixed64_bytes != 14057384ull ||
        dynamic_roffsrc_bytes != 8499884ull) return 10;

    std::cout << "gridfp-rankformula-abstract-srcpack10 OK"
              << " states=" << STATES
              << " transitions=" << transitions
              << " exact=" << exact
              << " max_source_local_rank=" << max_rank
              << " packed_word_bytes=" << word_bytes
              << " overflow_states=" << overflow_states
              << " overflow_bytes=" << overflow_bytes
              << " total_bytes=" << (word_bytes + overflow_bytes)
              << " word0_loads=" << word0_loads
              << " word1_loads=" << word1_loads
              << " overflow_loads=" << overflow_loads
              << " dynamic_split_bytes=" << dynamic_split_bytes
              << " dynamic_fixed64_bytes=" << dynamic_fixed64_bytes
              << " dynamic_roffsrc_bytes=" << dynamic_roffsrc_bytes
              << " ranks_per_word=3"
              << " source_bits=10"
              << " overflow_only_n14_h0=1\n";
    return 0;
}
