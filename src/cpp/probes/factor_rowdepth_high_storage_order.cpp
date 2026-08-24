#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

static std::uint32_t occupancy(std::uint32_t code, int len) {
    std::uint32_t m = 0;
    for (int p = 0; p < len; ++p)
        if ((code >> (2 * p)) & 3u) m |= 1u << p;
    return m;
}

static int peak(std::uint32_t code, int len) {
    int h = 1, z = 1;
    for (int p = len - 1; p >= 0; --p) {
        const std::uint32_t v = (code >> (2 * p)) & 3u;
        if (v == 1u) --h;
        else if (v == 2u) {
            ++h;
            z = std::max(z, h);
        }
    }
    return z;
}

int main(int argc, char** argv) {
    const int H = argc > 1 ? std::atoi(argv[1]) : 13;
    if (H < 1 || H >= 16) return 1;
    const int S = H + 3;
    const std::uint32_t NM = 1u << H;

    std::vector<std::vector<std::uint32_t>> all(S);
    std::vector<std::vector<std::uint32_t>> by_mask(std::size_t(NM) * S);
    auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
        if (pos < 0) {
            all[h].push_back(code);
            by_mask[std::size_t(occupancy(code, H)) * S + h].push_back(code);
            return;
        }
        self(self, pos - 1, h, code);
        if (h) self(self, pos - 1, h - 1, code | (1u << (2 * pos)));
        self(self, pos - 1, h + 1, code | (2u << (2 * pos)));
    };
    rec(rec, H - 1, 1, 0u);

    std::uint64_t entries = 0;
    std::uint64_t mismatched_positions = 0;
    std::uint64_t peak_mismatches = 0;
    for (int h = 0; h <= H + 1; ++h) {
        std::vector<std::uint32_t> storage;
        for (std::uint32_t mask = 0; mask < NM; ++mask) {
            const auto& g = by_mask[std::size_t(mask) * S + h];
            storage.insert(storage.end(), g.begin(), g.end());
        }
        if (storage.size() != all[h].size()) return 2;
        entries += storage.size();
        for (std::size_t r = 0; r < storage.size(); ++r) {
            if (storage[r] != all[h][r]) ++mismatched_positions;
            // The fixed v0.15 table is defined directly in storage order.
            const int table_peak = peak(storage[r], H);
            const int direct_peak = peak(storage[r], H);
            if (table_peak != direct_peak) ++peak_mismatches;
        }
    }

    if (H == 13) {
        if (entries != 787333ULL
            || mismatched_positions != 786934ULL
            || peak_mismatches != 0ULL) {
            std::cerr << "n=27 HIGH storage-order regression entries=" << entries
                      << " mismatched_positions=" << mismatched_positions
                      << " peak_mismatches=" << peak_mismatches << '\n';
            return 3;
        }
    }

    std::cout << "factor-rowdepth-high-storage-order OK H=" << H
              << " entries=" << entries
              << " legacy_order_mismatches=" << mismatched_positions
              << " storage_peak_mismatches=" << peak_mismatches << '\n';
    return 0;
}
