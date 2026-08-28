#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

struct GroupB { uint16_t start; uint16_t count; };

static void measure(const Factors& f, const std::vector<uint8_t>& owner, int B) {
    uint64_t codes = 0, groups_total = 0, blocks_total = 0, steps_total = 0;
    uint64_t direct = 0, boundary = 0;
    uint32_t max_steps = 0, max_groups_block = 0;
    for (int g = 0; g < NG; ++g) {
        for (int h = 0; h <= L + 1; ++h) {
            std::vector<GroupB> groups;
            uint32_t rank = 0;
            for (uint32_t m = 0; m < (1u << L); ++m) {
                if (owner[m] != g) continue;
                const uint32_t n = uint32_t(f.low_mask_h[size_t(m) * S + size_t(h)].size());
                if (!n) continue;
                groups.push_back(GroupB{uint16_t(rank), uint16_t(n)});
                rank += n;
            }
            if (groups.empty()) continue;
            codes += rank;
            groups_total += groups.size();
            const uint32_t nb = (rank + uint32_t(B) - 1u) / uint32_t(B);
            blocks_total += nb;
            uint32_t gi = 0;
            for (uint32_t b = 0; b < nb; ++b) {
                const uint32_t lo = b * uint32_t(B);
                const uint32_t hi = std::min(rank, lo + uint32_t(B));
                while (gi + 1u < groups.size() &&
                       lo >= uint32_t(groups[gi].start) + groups[gi].count) ++gi;
                const uint32_t first = gi;
                uint32_t last = first;
                while (last + 1u < groups.size() && groups[last + 1u].start < hi) ++last;
                const uint32_t gb = last - first + 1u;
                max_groups_block = std::max(max_groups_block, gb);
                if (gb == 1u) ++direct; else ++boundary;
                for (uint32_t r = lo; r < hi; ++r) {
                    uint32_t q = first, steps = 0;
                    while (q + 1u < groups.size() &&
                           r >= uint32_t(groups[q].start) + groups[q].count) {
                        ++q; ++steps;
                    }
                    if (r < groups[q].start ||
                        r >= uint32_t(groups[q].start) + groups[q].count) std::exit(2);
                    steps_total += steps;
                    max_steps = std::max(max_steps, steps);
                }
            }
        }
    }
    const uint64_t group_bytes = groups_total * sizeof(uint32_t);
    const uint64_t block_bytes = blocks_total * sizeof(uint16_t);
    std::cout << "block=" << B
              << " codes=" << codes
              << " groups=" << groups_total
              << " blocks=" << blocks_total
              << " group_bytes=" << group_bytes
              << " block_bytes=" << block_bytes
              << " aux_bytes=" << (group_bytes + block_bytes)
              << " max_groups_per_block=" << max_groups_block
              << " max_locator_steps=" << max_steps
              << " avg_locator_steps=" << double(steps_total) / double(codes)
              << " direct_blocks=" << direct
              << " boundary_blocks=" << boundary
              << " direct_fraction=" << double(direct) / double(blocks_total) << '\n';
}

int main() {
    Factors f = build_factors();
    const auto owner = low_mask_owners(f);
    for (int B : {4, 8, 16, 32}) measure(f, owner, B);
    std::cout << "gridfp-rankformula-nometa-blocks OK old_meta_bytes=4807668"
              << " candidate_block=4 candidate_max_steps=3"
              << " candidate_aux_bytes=879576\n";
    return 0;
}
