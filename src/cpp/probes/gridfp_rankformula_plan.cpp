#define main gridfp_low_rank16_plan_main_unused
#include "gridfp_low_rank16_plan.cpp"
#undef main

#include <limits>

static uint32_t choose_small(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n - k) k = n - k;
    uint64_t z = 1;
    for (int i = 1; i <= k; ++i) z = z * uint64_t(n - k + i) / uint64_t(i);
    return uint32_t(z);
}

// Number of R/L suffixes of length n that start at height h, finish at zero,
// and never cross below zero. R is the down step and is enumerated before L.
static uint32_t ballot_suffix(int n, int h) {
    if (n < 0 || h < 0 || h > n || ((n - h) & 1)) return 0;
    const int ups = (n - h) / 2;
    return choose_small(n, ups) - choose_small(n, ups - 1);
}

static uint32_t local_path_rank(uint32_t code, uint32_t mask, int h0) {
    int s = h0;
    int rem = __builtin_popcount(mask);
    uint32_t rank = 0;
    for (int pos = L - 1; pos >= 0; --pos) {
        if (((mask >> pos) & 1u) == 0u) continue;
        const uint32_t v = (code >> (2 * pos)) & 3u;
        if (v == uint32_t(LL)) {
            if (s > 0) rank += ballot_suffix(rem - 1, s - 1);
            ++s;
        } else if (v == uint32_t(R)) {
            if (s <= 0) return INVALID;
            --s;
        } else {
            return INVALID;
        }
        --rem;
    }
    return s == 0 && rem == 0 ? rank : INVALID;
}

int main() {
    Factors f = build_factors();
    const auto mask_owner = low_mask_owners(f);
    const uint32_t LM = 1u << L;

    std::vector<uint16_t> base(size_t(NG) * S * LM, RANK16_INVALID);
    auto base_ref = [&](int g, int h, uint32_t m) -> uint16_t& {
        return base[(size_t(g) * S + size_t(h)) * LM + m];
    };
    std::vector<uint32_t> direct(pow3(L), INVALID);
    uint64_t total_codes = 0;
    uint32_t max_rank = 0;

    for (int h = 0; h <= L + 1; ++h) {
        std::array<uint32_t, NG> next{};
        for (uint32_t m = 0; m < LM; ++m) {
            const int g = int(mask_owner[m]);
            if (next[g] >= RANK16_INVALID) return 2;
            base_ref(g, h, m) = uint16_t(next[g]);
            const auto& v = f.low_mask_h[size_t(m) * S + size_t(h)];
            for (size_t j = 0; j < v.size(); ++j) {
                const uint32_t code = v[j];
                const uint32_t local = local_path_rank(code, m, h);
                if (local != j) {
                    std::cerr << "rankformula local rank mismatch h=" << h << " mask=" << m
                              << " index=" << j << " got=" << local << '\n';
                    return 3;
                }
                const uint32_t r = next[g]++;
                const uint32_t key = ternary_key(code, L);
                if (direct[key] != INVALID || r >= RANK16_INVALID) return 4;
                direct[key] = (uint32_t(g) << 16) | r;
                max_rank = std::max(max_rank, r);
                ++total_codes;
            }
        }
    }

    uint64_t transitions = 0;
    uint64_t mismatches = 0;
    int min_offset = std::numeric_limits<int>::max();
    int max_offset = std::numeric_limits<int>::min();
    int min_prefix_corr = std::numeric_limits<int>::max();
    int max_prefix_corr = std::numeric_limits<int>::min();
    uint32_t max_dest_contrib = 0, max_source_local = 0;

    for (int h = 0; h <= L + 1; ++h) {
        for (uint32_t m = 0; m < LM; ++m) {
            const int g = int(mask_owner[m]);
            const auto& v = f.low_mask_h[size_t(m) * S + size_t(h)];
            const uint32_t dest_base = base_ref(g, h, m);
            for (size_t j = 0; j < v.size(); ++j) {
                const uint32_t code = v[j];
                const uint32_t key = ternary_key(code, L);
                const uint32_t dest_local = uint32_t(j);
                if (dest_base + dest_local >= RANK16_INVALID) return 5;

                int s = h;
                int rem = __builtin_popcount(m);
                int prefix_corr = 0;
                min_prefix_corr = std::min(min_prefix_corr, prefix_corr);
                max_prefix_corr = std::max(max_prefix_corr, prefix_corr);
                for (int pos = L - 1; pos >= 0; --pos) {
                    if (((m >> pos) & 1u) == 0u) continue;
                    const uint32_t sym = (code >> (2 * pos)) & 3u;
                    if (sym == uint32_t(LL)) {
                        if (h + 2 >= S) return 6;
                        const uint32_t dest_contrib = s > 0
                            ? ballot_suffix(rem - 1, s - 1) : 0u;
                        const uint32_t raised_contrib = ballot_suffix(rem - 1, s + 1);
                        const int formula_local = int(dest_local) + prefix_corr - int(dest_contrib);

                        const uint32_t weight = pow3(pos);
                        const uint32_t x = direct[key - weight];
                        if (x == INVALID || int(x >> 16) != g) return 7;
                        const uint32_t source_rank = x & 0xffffu;
                        const uint32_t source_base = base_ref(g, h + 2, m);
                        if (source_base == RANK16_INVALID || source_rank < source_base) return 8;
                        const uint32_t expected_local = source_rank - source_base;

                        ++transitions;
                        if (formula_local < 0 || uint32_t(formula_local) != expected_local) {
                            if (++mismatches <= 8) {
                                std::cerr << "rankformula transition mismatch owner=" << g
                                          << " h=" << h << " mask=" << m << " pos=" << pos
                                          << " dest_local=" << dest_local
                                          << " corr=" << prefix_corr
                                          << " contrib=" << dest_contrib
                                          << " formula=" << formula_local
                                          << " expected=" << expected_local << '\n';
                            }
                        }
                        const int off = formula_local - int(dest_local);
                        min_offset = std::min(min_offset, off);
                        max_offset = std::max(max_offset, off);
                        max_dest_contrib = std::max(max_dest_contrib, dest_contrib);
                        max_source_local = std::max(max_source_local, expected_local);

                        prefix_corr += int(raised_contrib) - int(dest_contrib);
                        min_prefix_corr = std::min(min_prefix_corr, prefix_corr);
                        max_prefix_corr = std::max(max_prefix_corr, prefix_corr);
                        ++s;
                    } else if (sym == uint32_t(R)) {
                        if (s <= 0) return 9;
                        --s;
                    } else {
                        return 10;
                    }
                    --rem;
                }
                if (s != 0 || rem != 0) return 11;
            }
        }
    }

    if (mismatches != 0 || transitions != 3720805ull || total_codes != 1201917ull) return 12;
    const uint64_t dense_base_bytes = uint64_t(S) * LM * sizeof(uint16_t);
    const uint64_t chunk_meta_bytes = total_codes * sizeof(uint32_t);
    std::cout << "gridfp-rankformula-plan OK"
              << " W=" << W << " low_k=" << L
              << " codes=" << total_codes
              << " transitions=" << transitions
              << " mismatches=" << mismatches
              << " max_rank=" << max_rank
              << " min_offset=" << min_offset
              << " max_offset=" << max_offset
              << " max_dest_contrib=" << max_dest_contrib
              << " min_prefix_corr=" << min_prefix_corr
              << " max_prefix_corr=" << max_prefix_corr
              << " max_source_local=" << max_source_local
              << " dense_base_bytes_per_owner=" << dense_base_bytes
              << " chunk_meta_bytes_all_owners=" << chunk_meta_bytes
              << " rankstream_bytes=0"
              << " source_height_delta=2"
              << " formula=dest_local+prefix_corr-current_contrib\n";
    return 0;
}
