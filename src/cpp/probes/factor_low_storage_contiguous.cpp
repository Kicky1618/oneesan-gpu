#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

using U32 = std::uint32_t;
static constexpr int R = 1;
static constexpr int L = 2;

static U32 occupancy(U32 code, int n) {
    U32 z = 0;
    for (int p = 0; p < n; ++p)
        if ((code >> (2 * p)) & 3u) z |= 1u << p;
    return z;
}

int main(int argc, char** argv) {
    const int low = argc > 1 ? std::atoi(argv[1]) : 14;
    if (low < 1 || low >= 16) {
        std::cerr << "usage: factor_low_storage_contiguous [LOW<16]\n";
        return 1;
    }

    std::uint64_t total = 0, groups = 0, checks = 0;
    U32 max_group = 0;
    for (int h0 = 0; h0 <= low + 1; ++h0) {
        std::vector<U32> all;
        auto rec = [&](auto&& self, int pos, int h, U32 code) -> void {
            if (pos < 0) {
                if (h == 0) all.push_back(code);
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, code);
            if (h > 0) self(self, pos - 1, h - 1, code | (U32(R) << (2 * pos)));
            self(self, pos - 1, h + 1, code | (U32(L) << (2 * pos)));
        };
        rec(rec, low - 1, h0, 0);

        std::vector<std::vector<U32>> by_mask(1u << low);
        for (U32 code : all) by_mask[occupancy(code, low)].push_back(code);

        // Independently reconstruct the occupancy-major storage order, then
        // prove that each base factor mask list is exactly one contiguous run.
        std::unordered_map<U32, U32> storage_rank;
        storage_rank.reserve(all.size() * 2 + 1);
        U32 sr = 0;
        for (U32 mask = 0; mask < (1u << low); ++mask) {
            const U32 base = sr;
            max_group = std::max<U32>(max_group, U32(by_mask[mask].size()));
            for (U32 r = 0; r < by_mask[mask].size(); ++r)
                storage_rank.emplace(by_mask[mask][r], sr++);
            for (U32 r = 0; r < by_mask[mask].size(); ++r) {
                const auto it = storage_rank.find(by_mask[mask][r]);
                if (it == storage_rank.end() || it->second != base + r) {
                    std::cerr << "LOW storage contiguous mismatch h=" << h0
                              << " mask=" << mask << " rank=" << r << '\n';
                    return 2;
                }
                ++checks;
            }
            ++groups;
        }
        if (sr != all.size()) {
            std::cerr << "LOW storage rank total mismatch h=" << h0 << '\n';
            return 3;
        }
        total += all.size();
    }

    std::cout << "factor-low-storage-contiguous OK low=" << low
              << " codes=" << total
              << " groups=" << groups
              << " checks=" << checks
              << " max_group=" << max_group << '\n';
    return 0;
}
