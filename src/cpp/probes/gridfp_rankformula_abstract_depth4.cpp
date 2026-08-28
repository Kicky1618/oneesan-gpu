#define main gridfp_rankformula_abstract_lut_main_unused
#include "gridfp_rankformula_abstract_lut.cpp"
#undef main

static uint64_t choose_u64_depth4(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    uint64_t z = 1;
    for (int i = 1; i <= k; ++i) z = z * uint64_t(n - k + i) / uint64_t(i);
    return z;
}

static uint8_t select_mask_depth4(uint32_t lp, int n, uint32_t depth) {
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

static uint8_t depth4_swar(uint32_t dpack, uint32_t depth) {
    uint32_t x = dpack ^ (depth * 0x01111111u);
    uint32_t y = x | (x >> 1) | (x >> 2) | (x >> 3);
    uint32_t z = (~y) & 0x01111111u;
    z = (z | (z >> 3)) & 0x03030303u;
    z = (z | (z >> 6)) & 0x000f000fu;
    z = (z | (z >> 12)) & 0x000000ffu;
    return uint8_t(z & 0x7fu);
}

int main() {
    constexpr uint32_t STATES = 7060u;
    std::array<uint16_t, STATES> depth03{}, depth46{};
    uint32_t di = 0, max_depth = 0, exact_entries = 0, nonzero_slots = 0;
    uint64_t no_l_calls = 0, word0_loads = 0, word1_loads = 0;

    for (int n = 0; n <= L; ++n) {
        const uint64_t weight = choose_u64_depth4(L, n);
        for (int h = 0; h < 16; ++h) {
            const uint32_t cnt = ballot_suffix(n, h);
            for (uint32_t local = 0; local < cnt; ++local, ++di) {
                if (di >= STATES) return 2;
                const uint32_t lp = abstract_lpattern(n, h, local);
                if (lp == INVALID) return 3;
                const uint32_t lc = uint32_t(__builtin_popcount(lp));
                if (lc > 7u || lc != uint32_t((n - h) / 2)) return 4;
                uint32_t dpack = 0;
                for (uint32_t depth = 1; depth <= 13u; ++depth) {
                    const uint8_t sm = select_mask_depth4(lp, n, depth);
                    for (uint32_t li = 0; li < lc; ++li) {
                        if (((sm >> li) & 1u) == 0u) continue;
                        const uint32_t old = (dpack >> (4u * li)) & 15u;
                        if (old != 0u && old != depth) return 5;
                        dpack |= depth << (4u * li);
                        max_depth = std::max(max_depth, depth);
                    }
                }
                depth03[di] = uint16_t(dpack & 0xffffu);
                depth46[di] = uint16_t((dpack >> 16) & 0x0fffu);
                const uint32_t rebuilt = uint32_t(depth03[di]) | (uint32_t(depth46[di]) << 16);
                if (rebuilt != dpack) return 6;
                for (uint32_t li = 0; li < lc; ++li)
                    nonzero_slots += ((dpack >> (4u * li)) & 15u) != 0u;
                for (uint32_t depth = 1; depth <= 13u; ++depth) {
                    const uint8_t want = select_mask_depth4(lp, n, depth);
                    const uint8_t got = depth4_swar(rebuilt, depth);
                    if (got != want) return 7;
                    ++exact_entries;
                }
                if (depth4_swar(rebuilt, 14u) != 0u || depth4_swar(rebuilt, 15u) != 0u)
                    return 8;
                if (lc == 0u) no_l_calls += 13ull * weight;
                else {
                    word0_loads += 13ull * weight;
                    if (lc > 4u) word1_loads += 13ull * weight;
                }
            }
        }
    }

    const uint64_t bytes = uint64_t(STATES) * 2ull * sizeof(uint16_t);
    const uint64_t dynamic_split_bytes = 2ull * (word0_loads + word1_loads);
    if (di != STATES || exact_entries != 91780u || nonzero_slots != 19273u ||
        max_depth != 13u || bytes != 28240ull ||
        no_l_calls != 212992ull || word0_loads != 15411929ull || word1_loads != 1892462ull ||
        dynamic_split_bytes != 34608782ull) return 9;
    std::cout << "gridfp-rankformula-abstract-depth4 OK"
              << " states=" << STATES
              << " exact_depth_entries=" << exact_entries
              << " table_bytes=" << bytes
              << " nonzero_depth_slots=" << nonzero_slots
              << " max_selected_depth=" << max_depth
              << " no_l_calls=" << no_l_calls
              << " word0_loads=" << word0_loads
              << " word1_loads=" << word1_loads
              << " dynamic_split_bytes=" << dynamic_split_bytes
              << " bits_per_l_ordinal=4"
              << " packed_bits_per_state=28"
              << " split16=1"
              << " swar_exact=1"
              << " depth14_15_zero=1\n";
    return 0;
}
