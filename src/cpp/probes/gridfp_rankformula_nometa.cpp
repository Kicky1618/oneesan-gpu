#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

#include <limits>

struct Group { uint16_t mask; uint16_t start; uint16_t count; };

static uint32_t unrank_code(uint32_t mask, int h, uint32_t local) {
    uint32_t code = 0;
    int s = h;
    int rem = __builtin_popcount(mask);
    for (int pos = L - 1; pos >= 0; --pos) {
        if (((mask >> pos) & 1u) == 0u) continue;
        const uint32_t rcount = s > 0 ? ballot_suffix(rem - 1, s - 1) : 0u;
        if (s > 0 && local < rcount) {
            code |= uint32_t(R) << (2 * pos);
            --s;
        } else {
            if (local < rcount) return INVALID;
            local -= rcount;
            code |= uint32_t(LL) << (2 * pos);
            ++s;
        }
        --rem;
    }
    return (s == 0 && local == 0) ? code : INVALID;
}

int main() {
    Factors f = build_factors();
    const auto owner = low_mask_owners(f);
    const uint32_t LM = 1u << L;

    uint64_t codes = 0, unrank_exact = 0;
    uint64_t groups_total = 0, block32_total = 0, locator_steps_total = 0;
    uint64_t direct_blocks = 0, boundary_blocks = 0;
    uint32_t max_groups_owner_height = 0, max_locator_steps = 0, max_groups_per_block = 0;
    uint64_t group_table_bytes = 0, block_table_bytes = 0;

    for (int g = 0; g < NG; ++g) {
        for (int h = 0; h <= L + 1; ++h) {
            std::vector<Group> groups;
            uint32_t rank = 0;
            for (uint32_t m = 0; m < LM; ++m) {
                if (owner[m] != g) continue;
                const auto& v = f.low_mask_h[size_t(m) * S + size_t(h)];
                if (v.empty()) continue;
                if (rank >= RANK16_INVALID || v.size() >= RANK16_INVALID ||
                    rank + v.size() >= RANK16_INVALID) return 2;
                groups.push_back(Group{uint16_t(m), uint16_t(rank), uint16_t(v.size())});
                for (uint32_t j = 0; j < v.size(); ++j) {
                    const uint32_t got = unrank_code(m, h, j);
                    ++codes;
                    if (got != v[j]) {
                        std::cerr << "rankformula nometa unrank mismatch owner=" << g
                                  << " h=" << h << " mask=" << m
                                  << " local=" << j << " got=" << got
                                  << " expected=" << v[j] << '\n';
                        return 3;
                    }
                    ++unrank_exact;
                }
                rank += uint32_t(v.size());
            }
            if (groups.empty()) continue;
            groups_total += groups.size();
            max_groups_owner_height = std::max(max_groups_owner_height, uint32_t(groups.size()));
            // One uint32 per group: support14 | start16. One uint16 group index per 32 ranks.
            group_table_bytes += groups.size() * sizeof(uint32_t);
            const uint32_t blocks = (rank + 31u) / 32u;
            block32_total += blocks;
            block_table_bytes += uint64_t(blocks) * sizeof(uint16_t);

            uint32_t gi = 0;
            for (uint32_t b = 0; b < blocks; ++b) {
                const uint32_t lo = b * 32u;
                const uint32_t hi = std::min(rank, lo + 32u);
                while (gi + 1u < groups.size() &&
                       lo >= uint32_t(groups[gi].start) + groups[gi].count) ++gi;
                const uint32_t first = gi;
                uint32_t last = first;
                while (last + 1u < groups.size() && groups[last + 1u].start < hi) ++last;
                const uint32_t in_block = last - first + 1u;
                max_groups_per_block = std::max(max_groups_per_block, in_block);
                if (in_block == 1u) ++direct_blocks; else ++boundary_blocks;

                for (uint32_t r = lo; r < hi; ++r) {
                    uint32_t q = first, steps = 0;
                    while (q + 1u < groups.size() &&
                           r >= uint32_t(groups[q].start) + groups[q].count) {
                        ++q; ++steps;
                    }
                    if (r < groups[q].start ||
                        r >= uint32_t(groups[q].start) + groups[q].count) return 4;
                    max_locator_steps = std::max(max_locator_steps, steps);
                    locator_steps_total += steps;
                }
            }
        }
    }

    if (codes != 1201917ull || unrank_exact != codes || block32_total == 0 ||
        groups_total == 0) return 5;
    const double avg_steps = double(locator_steps_total) / double(codes);
    const double direct_fraction = double(direct_blocks) / double(block32_total);
    std::cout << "gridfp-rankformula-nometa OK"
              << " codes=" << codes
              << " unrank_exact=" << unrank_exact
              << " groups=" << groups_total
              << " block32=" << block32_total
              << " group_table_bytes_all=" << group_table_bytes
              << " block_table_bytes_all=" << block_table_bytes
              << " aux_bytes_all=" << (group_table_bytes + block_table_bytes)
              << " old_meta_bytes_all=" << (codes * sizeof(uint32_t))
              << " max_groups_owner_height=" << max_groups_owner_height
              << " max_groups_per_block=" << max_groups_per_block
              << " max_locator_steps=" << max_locator_steps
              << " avg_locator_steps=" << avg_steps
              << " direct_blocks=" << direct_blocks
              << " boundary_blocks=" << boundary_blocks
              << " direct_block_fraction=" << direct_fraction
              << " per_code_metadata_bytes=0"
              << " ballot_unrank_exact=1\n";
    return 0;
}
