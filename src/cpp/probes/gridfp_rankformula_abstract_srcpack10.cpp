#define main gridfp_rankformula_abstract_lut_main_unused
#include "gridfp_rankformula_abstract_lut.cpp"
#undef main

int main() {
    constexpr uint32_t STATES = 7060u;
    constexpr uint32_t OVERFLOW_N = 429u;
    std::array<uint64_t, STATES> packed{};
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
                    if (li < 6u) {
                        packed[di] |= uint64_t(sr) << (10u * li);
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

    // Decode every source rank from the packed representation and compare with
    // an independent ballot-rank computation.
    di = 0;
    uint32_t exact = 0;
    for (int n = 0; n <= L; ++n) {
        for (int h = 0; h < 16; ++h) {
            const uint32_t cnt = ballot_suffix(n, h);
            for (uint32_t local = 0; local < cnt; ++local, ++di) {
                const uint32_t lp = abstract_lpattern(n, h, local);
                uint32_t li = 0;
                for (int ord = 0; ord < n; ++ord) {
                    if (((lp >> ord) & 1u) == 0u) continue;
                    const uint32_t want = abstract_rank(n, h + 2, lp & ~(1u << ord));
                    const uint32_t got = li < 6u
                        ? uint32_t((packed[di] >> (10u * li)) & 1023u)
                        : uint32_t(overflow[local]);
                    if (got != want) return 8;
                    ++li;
                    ++exact;
                }
            }
        }
    }

    const std::array<uint64_t,8> want_hist{15,91,352,915,1652,2044,1562,429};
    if (exact != 32743u || lcount_hist != want_hist) return 9;
    const uint64_t packed_bytes = uint64_t(STATES) * sizeof(uint64_t);
    const uint64_t overflow_bytes = uint64_t(OVERFLOW_N) * sizeof(uint16_t);
    std::cout << "gridfp-rankformula-abstract-srcpack10 OK"
              << " states=" << STATES
              << " transitions=" << transitions
              << " exact=" << exact
              << " max_source_local_rank=" << max_rank
              << " packed_bytes=" << packed_bytes
              << " overflow_states=" << overflow_states
              << " overflow_bytes=" << overflow_bytes
              << " total_bytes=" << (packed_bytes + overflow_bytes)
              << " first6_bits=60"
              << " source_bits=10"
              << " overflow_only_n14_h0=1\n";
    return 0;
}
