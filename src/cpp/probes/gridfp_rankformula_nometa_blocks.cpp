#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

struct GroupB { uint16_t start; uint16_t count; };

struct MeasureResult {
    uint64_t codes = 0;
    uint64_t groups = 0;
    uint64_t blocks = 0;
    uint64_t group_bytes = 0;
    uint64_t block_bytes = 0;
    uint64_t aux_bytes = 0;
    uint64_t locator_steps = 0;
    uint64_t direct = 0;
    uint64_t boundary = 0;
    uint32_t max_steps = 0;
    uint32_t max_groups_block = 0;
};

static MeasureResult measure(const Factors& f, const std::vector<uint8_t>& owner, int B) {
    MeasureResult z;
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
            z.codes += rank;
            z.groups += groups.size();
            const uint32_t nb = (rank + uint32_t(B) - 1u) / uint32_t(B);
            z.blocks += nb;
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
                z.max_groups_block = std::max(z.max_groups_block, gb);
                if (gb == 1u) ++z.direct; else ++z.boundary;
                for (uint32_t r = lo; r < hi; ++r) {
                    uint32_t q = first, steps = 0;
                    while (q + 1u < groups.size() &&
                           r >= uint32_t(groups[q].start) + groups[q].count) {
                        ++q;
                        ++steps;
                    }
                    if (r < groups[q].start ||
                        r >= uint32_t(groups[q].start) + groups[q].count) std::exit(2);
                    z.locator_steps += steps;
                    z.max_steps = std::max(z.max_steps, steps);
                }
            }
        }
    }
    // Production nometa now stores start16+n4+delta16+count16 in group64.
    z.group_bytes = z.groups * sizeof(uint64_t);
    z.block_bytes = z.blocks * sizeof(uint16_t);
    z.aux_bytes = z.group_bytes + z.block_bytes;
    std::cout << "block=" << B
              << " codes=" << z.codes
              << " groups=" << z.groups
              << " blocks=" << z.blocks
              << " group_bytes=" << z.group_bytes
              << " block_bytes=" << z.block_bytes
              << " aux_bytes=" << z.aux_bytes
              << " max_groups_per_block=" << z.max_groups_block
              << " max_locator_steps=" << z.max_steps
              << " avg_locator_steps=" << double(z.locator_steps) / double(z.codes)
              << " avg_group_loads_model=" << (1.0 + double(z.locator_steps) / double(z.codes))
              << " direct_blocks=" << z.direct
              << " boundary_blocks=" << z.boundary
              << " direct_fraction=" << double(z.direct) / double(z.blocks) << '\n';
    return z;
}

int main() {
    Factors f = build_factors();
    const auto owner = low_mask_owners(f);
    const auto b4 = measure(f, owner, 4);
    const auto b8 = measure(f, owner, 8);
    const auto b16 = measure(f, owner, 16);
    const auto b32 = measure(f, owner, 32);
    if (b4.codes != 1201917ull || b8.codes != b4.codes ||
        b16.codes != b4.codes || b32.codes != b4.codes ||
        b4.groups != 69632ull || b8.groups != b4.groups ||
        b4.aux_bytes != 1158104ull || b8.aux_bytes != 857642ull ||
        b16.aux_bytes != 707406ull || b32.aux_bytes != 632312ull ||
        b4.max_steps != 3u || b8.max_steps != 7u) return 3;
    std::cout << "gridfp-rankformula-nometa-blocks OK"
              << " old_meta_bytes=4807668"
              << " group_entry_bytes=8"
              << " b4_aux_bytes=" << b4.aux_bytes
              << " b8_aux_bytes=" << b8.aux_bytes
              << " b4_max_steps=" << b4.max_steps
              << " b8_max_steps=" << b8.max_steps
              << " ab_blocks=4,8\n";
    return 0;
}
