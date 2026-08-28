#define main gridfp_rankformula_abstract_lut_main_unused
#include "gridfp_rankformula_abstract_lut.cpp"
#undef main

#include <array>

static uint64_t choose_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    uint64_t z = 1;
    for (int i = 1; i <= k; ++i) z = z * uint64_t(n - k + i) / uint64_t(i);
    return z;
}

static uint8_t select_mask(uint32_t lp, int n, uint32_t depth) {
    uint32_t state = depth;
    uint32_t li = 0;
    uint8_t out = 0;
    for (int ord = 0; ord < n; ++ord) {
        if ((lp >> ord) & 1u) {
            if (state == 1u) {
                if (li >= 7u) std::exit(30);
                out |= uint8_t(1u << li);
            }
            ++li;
            ++state;
        } else {
            if (state == 1u) break;
            --state;
        }
    }
    return out;
}

int main() {
    constexpr uint32_t DEPTH_N = 13;
    std::array<uint64_t, 8> universal_hist{};
    std::array<uint64_t, 8> production_hist{};
    uint64_t states = 0, entries = 0, nonzero = 0, selected = 0;
    uint64_t production_entries = 0, production_nonzero = 0, production_selected = 0;
    uint32_t max_lcount = 0, depth14_selected = 0, depth15_selected = 0;

    for (int n = 0; n <= L; ++n) {
        const uint64_t weight = choose_u64(L, n);
        for (int h = 0; h < 16; ++h) {
            const uint32_t cnt = ballot_suffix(n, h);
            for (uint32_t local = 0; local < cnt; ++local) {
                const uint32_t lp = abstract_lpattern(n, h, local);
                if (lp == INVALID) return 2;
                const uint32_t lc = uint32_t(__builtin_popcount(lp));
                max_lcount = std::max(max_lcount, lc);
                if (lc > 7u) return 3;
                ++states;
                for (uint32_t depth = 1; depth <= DEPTH_N; ++depth) {
                    const uint8_t sm = select_mask(lp, n, depth);
                    if (uint32_t(sm) >= (1u << lc)) return 4;
                    const uint32_t pc = uint32_t(__builtin_popcount(uint32_t(sm)));
                    ++entries;
                    ++universal_hist[pc];
                    nonzero += sm != 0;
                    selected += pc;
                    production_entries += weight;
                    production_hist[pc] += weight;
                    production_nonzero += (sm != 0) ? weight : 0;
                    production_selected += uint64_t(pc) * weight;

                    uint32_t state = depth, li = 0;
                    uint8_t want = 0;
                    for (int ord = 0; ord < n; ++ord) {
                        if ((lp >> ord) & 1u) {
                            if (state == 1u) want |= uint8_t(1u << li);
                            ++li; ++state;
                        } else {
                            if (state == 1u) break;
                            --state;
                        }
                    }
                    if (want != sm) return 5;
                }
                depth14_selected += __builtin_popcount(uint32_t(select_mask(lp, n, 14)));
                depth15_selected += __builtin_popcount(uint32_t(select_mask(lp, n, 15)));
            }
        }
    }

    const uint64_t table_bytes = entries;
    if (states != 7060ull || entries != 91780ull || table_bytes != 91780ull ||
        nonzero != 12755ull || selected != 19273ull || max_lcount != 7u ||
        production_entries != 15624921ull || production_nonzero != 1757173ull ||
        production_selected != 2492769ull || depth14_selected != 0u || depth15_selected != 0u)
        return 6;
    const std::array<uint64_t,8> want_universal{79025,8191,3107,1060,311,73,12,1};
    const std::array<uint64_t,8> want_production{13867748,1195735,416811,118082,23689,2727,128,1};
    if (universal_hist != want_universal || production_hist != want_production) return 7;

    std::cout << "gridfp-rankformula-abstract-select8 OK"
              << " abstract_states=" << states
              << " select_depths=" << DEPTH_N
              << " table_entries=" << entries
              << " table_bytes=" << table_bytes
              << " universal_nonzero=" << nonzero
              << " universal_selected=" << selected
              << " production_entries=" << production_entries
              << " production_nonzero=" << production_nonzero
              << " production_selected=" << production_selected
              << " max_lcount=" << max_lcount
              << " depth14_selected=" << depth14_selected
              << " depth15_selected=" << depth15_selected
              << " exact_rankmask=1"
              << " select_bits=7\n";
    return 0;
}
