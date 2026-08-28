#define main gridfp_low_rank16_plan_main_unused
#include "gridfp_low_rank16_plan.cpp"
#undef main

#include <limits>

static uint32_t lcount_code(uint32_t code) {
    uint32_t n = 0;
    for (int p = 0; p < L; ++p) if (((code >> (2 * p)) & 3u) == uint32_t(LL)) ++n;
    return n;
}

int main() {
    Factors f = build_factors();
    auto mask_owner = low_mask_owners(f);
    std::array<std::vector<LocalCode>, NG> codes;
    std::vector<uint32_t> direct(pow3(L), INVALID);
    uint32_t max_local_rank = 0;

    for (int h = 0; h <= L + 1; ++h) {
        std::array<uint32_t, NG> next{};
        for (uint32_t m = 0; m < (1u << L); ++m) {
            uint32_t g = mask_owner[m];
            const auto& v = f.low_mask_h[size_t(m) * S + size_t(h)];
            for (uint32_t code : v) {
                uint32_t r = next[g]++;
                if (r >= RANK16_INVALID) return 2;
                codes[g].push_back(LocalCode{code, uint16_t(r), uint8_t(h), uint8_t(g)});
                uint32_t key = ternary_key(code, L);
                if (direct[key] != INVALID) return 3;
                direct[key] = (uint32_t(g) << 16) | r;
                max_local_rank = std::max(max_local_rank, r);
            }
        }
    }

    uint64_t total_ranks = 0, total_bytes = 0, total_slow_delta = 0, total_delta = 0;
    uint32_t max_delta = 0, max_owner_ranks = 0, max_owner_bytes = 0;
    uint32_t max_prefix_packed = 0, max_prefix_align32 = 0;

    for (int g = 0; g < NG; ++g) {
        std::array<std::vector<uint32_t>, L + 2> row_bytes{};
        uint64_t owner_ranks = 0, owner_bytes = 0;
        for (const LocalCode& z : codes[g]) {
            uint32_t key = ternary_key(z.code, L), weight = pow3(L - 1);
            std::array<uint16_t, L> ranks{};
            uint32_t nr = 0;
            for (int pos = L - 1; pos >= 0; --pos) {
                if (((z.code >> (2 * pos)) & 3u) == uint32_t(LL)) {
                    uint32_t x = direct[key - weight];
                    if (x == INVALID || (x >> 16) != uint32_t(g)) return 4;
                    ranks[nr++] = uint16_t(x & 0xffffu);
                }
                if (pos) weight /= 3u;
            }
            if (nr != lcount_code(z.code)) return 5;
            owner_ranks += nr;
            uint32_t bytes = 0;
            if (nr) {
                bytes = 2;
                for (uint32_t i = 1; i < nr; ++i) {
                    if (ranks[i] <= ranks[i - 1]) {
                        std::cerr << "rankdelta8 non-monotone owner=" << g
                                  << " h=" << unsigned(z.h) << " i=" << i
                                  << " prev=" << ranks[i - 1] << " cur=" << ranks[i] << '\n';
                        return 6;
                    }
                    uint32_t d = uint32_t(ranks[i] - ranks[i - 1]);
                    max_delta = std::max(max_delta, d);
                    ++total_delta;
                    if (d >= 128u) ++total_slow_delta;
                    if (d >= (1u << 14)) return 7;
                    bytes += d < 128u ? 1u : 2u;
                }
            }
            row_bytes[z.h].push_back(bytes);
            owner_bytes += bytes;
        }
        total_ranks += owner_ranks;
        total_bytes += owner_bytes;
        max_owner_ranks = std::max(max_owner_ranks, uint32_t(owner_ranks));
        max_owner_bytes = std::max(max_owner_bytes, uint32_t(owner_bytes));

        for (int aligned = 0; aligned <= 1; ++aligned) {
            std::vector<uint32_t> sizes;
            for (int h = 0; h <= L + 1; ++h) {
                if (aligned) while (sizes.size() & 31u) sizes.push_back(0u);
                sizes.insert(sizes.end(), row_bytes[h].begin(), row_bytes[h].end());
            }
            uint32_t local_max = 0;
            for (size_t a = 0; a < sizes.size(); a += 32u) {
                uint32_t prefix = 0;
                for (size_t j = a; j < std::min(a + 32u, sizes.size()); ++j) {
                    local_max = std::max(local_max, prefix);
                    prefix += sizes[j];
                }
            }
            if (aligned) max_prefix_align32 = std::max(max_prefix_align32, local_max);
            else max_prefix_packed = std::max(max_prefix_packed, local_max);
        }
    }

    const uint64_t original_bytes = total_ranks * sizeof(uint16_t);
    if (max_local_rank >= (1u << 15) || max_owner_ranks >= (1u << 19) ||
        max_owner_bytes >= (1u << 20) || max_prefix_packed >= (1u << 9) ||
        max_prefix_align32 >= (1u << 9)) return 8;

    std::cout << "gridfp-rankdelta8-plan OK"
              << " W=" << W << " low_k=" << L
              << " max_local_rank=" << max_local_rank
              << " max_owner_ranks=" << max_owner_ranks
              << " max_owner_bytes=" << max_owner_bytes
              << " max_delta=" << max_delta
              << " slow_delta=" << total_slow_delta << '/' << total_delta
              << " original_rankstream_bytes=" << original_bytes
              << " rankdelta8_bytes=" << total_bytes
              << " ratio=" << (original_bytes ? double(total_bytes) / double(original_bytes) : 0.0)
              << " max_prefix_packed=" << max_prefix_packed
              << " max_prefix_align32=" << max_prefix_align32
              << " local_rank_bits=15 owner_rank_bits=19 owner_byte_bits=20"
              << " delta_bits=14 prefix_bits=9 block=32\n";
    return 0;
}
