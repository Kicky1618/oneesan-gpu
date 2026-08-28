#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

#include <limits>

int main() {
    Factors f = build_factors();
    const auto mask_owner = low_mask_owners(f);
    const uint32_t LM = 1u << L;

    std::vector<int32_t> base(size_t(NG) * S * LM, -1);
    auto base_ref = [&](int g, int h, uint32_t m) -> int32_t& {
        return base[(size_t(g) * S + size_t(h)) * LM + m];
    };
    std::vector<uint32_t> direct(pow3(L), INVALID);

    uint64_t total_codes = 0;
    for (int h = 0; h <= L + 1; ++h) {
        std::array<uint32_t, NG> next{};
        for (uint32_t m = 0; m < LM; ++m) {
            const int g = int(mask_owner[m]);
            base_ref(g, h, m) = int32_t(next[g]);
            const auto& v = f.low_mask_h[size_t(m) * S + size_t(h)];
            for (uint32_t code : v) {
                const uint32_t r = next[g]++;
                if (r >= RANK16_INVALID) return 2;
                const uint32_t key = ternary_key(code, L);
                if (direct[key] != INVALID) return 3;
                direct[key] = (uint32_t(g) << 16) | r;
                ++total_codes;
            }
        }
    }

    int min_delta = std::numeric_limits<int>::max();
    int max_delta = std::numeric_limits<int>::min();
    uint64_t delta_rows = 0, transitions = 0, mismatches = 0;

    for (int h = 0; h <= L + 1; ++h) {
        for (uint32_t m = 0; m < LM; ++m) {
            const int g = int(mask_owner[m]);
            const auto& v = f.low_mask_h[size_t(m) * S + size_t(h)];
            if (v.empty()) continue;

            bool needs_source = false;
            for (uint32_t code : v) {
                for (int pos = 0; pos < L; ++pos)
                    if (((code >> (2 * pos)) & 3u) == LL) { needs_source = true; break; }
                if (needs_source) break;
            }
            if (!needs_source) continue;
            if (h + 2 > L + 1) return 4;
            const int dest_base = base_ref(g, h, m);
            const int source_base = base_ref(g, h + 2, m);
            if (dest_base < 0 || source_base < 0) return 5;
            const int base_delta = source_base - dest_base;
            min_delta = std::min(min_delta, base_delta);
            max_delta = std::max(max_delta, base_delta);
            ++delta_rows;

            for (size_t j = 0; j < v.size(); ++j) {
                const uint32_t code = v[j];
                const uint32_t key = ternary_key(code, L);
                const uint32_t dest_rank = uint32_t(dest_base) + uint32_t(j);
                int s = h, rem = __builtin_popcount(m), prefix_corr = 0;
                for (int pos = L - 1; pos >= 0; --pos) {
                    if (((m >> pos) & 1u) == 0u) continue;
                    const uint32_t sym = (code >> (2 * pos)) & 3u;
                    if (sym == LL) {
                        const uint32_t dest_contrib = s > 0
                            ? ballot_suffix(rem - 1, s - 1) : 0u;
                        const uint32_t raised_contrib = ballot_suffix(rem - 1, s + 1);
                        const int formula_rank = int(dest_rank) + base_delta
                            + prefix_corr - int(dest_contrib);
                        const uint32_t x = direct[key - pow3(pos)];
                        if (x == INVALID || int(x >> 16) != g) return 6;
                        const int expected_rank = int(x & 0xffffu);
                        ++transitions;
                        if (formula_rank != expected_rank) {
                            if (++mismatches <= 8) {
                                std::cerr << "rankformula base-delta mismatch owner=" << g
                                          << " h=" << h << " mask=" << m
                                          << " pos=" << pos << " rank=" << dest_rank
                                          << " base_delta=" << base_delta
                                          << " corr=" << prefix_corr
                                          << " contrib=" << dest_contrib
                                          << " formula=" << formula_rank
                                          << " expected=" << expected_rank << '\n';
                            }
                        }
                        prefix_corr += int(raised_contrib) - int(dest_contrib);
                        ++s;
                    } else if (sym == R) {
                        --s;
                    } else {
                        return 7;
                    }
                    --rem;
                }
            }
        }
    }

    if (total_codes != 1201917ull || transitions != 3720805ull || mismatches != 0) return 8;
    if (min_delta < std::numeric_limits<int16_t>::min() ||
        max_delta > std::numeric_limits<int16_t>::max()) return 9;

    std::cout << "gridfp-rankformula-base-delta OK"
              << " codes=" << total_codes
              << " transitions=" << transitions
              << " delta_rows=" << delta_rows
              << " mismatches=" << mismatches
              << " min_base_delta=" << min_delta
              << " max_base_delta=" << max_delta
              << " int16_exact=1"
              << " source_formula=rank+base_delta+prefix_corr-dest_contrib"
              << " base_values_per_lookup=1\n";
    return 0;
}
