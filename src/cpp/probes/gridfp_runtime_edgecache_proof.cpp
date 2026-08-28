#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

namespace {

constexpr int SUBGROUP = 8;
constexpr int MAX_PAIRS = 20;
constexpr int MAX_EDGE_TERMS = 3;
constexpr int MAX_DESTINATIONS_PER_LANE =
    (MAX_PAIRS + SUBGROUP - 1) / SUBGROUP;

struct EdgeRow {
    std::uint8_t count = 0;
    std::uint8_t destination[MAX_EDGE_TERMS]{};
    std::int8_t coefficient[MAX_EDGE_TERMS]{};
};

struct RuntimeEdgeCacheModel {
    EdgeRow source[MAX_PAIRS]{};
};

static_assert(MAX_DESTINATIONS_PER_LANE == 3);
static_assert(sizeof(EdgeRow) == 7);
static_assert(sizeof(RuntimeEdgeCacheModel) == 140);

} // namespace

int main() {
    std::uint64_t routing_cases = 0;
    int max_slot = 0;
    for (int nd = 1; nd <= MAX_PAIRS; ++nd) {
        for (int di = 0; di < nd; ++di) {
            const int lane = di & (SUBGROUP - 1);
            const int slot = di >> 3;
            const int reconstructed = lane + slot * SUBGROUP;
            if (reconstructed != di || lane < 0 || lane >= SUBGROUP ||
                slot < 0 || slot >= MAX_DESTINATIONS_PER_LANE) {
                std::cerr << "edge-cache routing mismatch nd=" << nd
                          << " di=" << di << " lane=" << lane
                          << " slot=" << slot
                          << " reconstructed=" << reconstructed << '\n';
                return 2;
            }
            max_slot = std::max(max_slot, slot);
            ++routing_cases;
        }
    }

    std::uint64_t accumulation_cases = 0;
    std::uint64_t edge_terms = 0;
    for (int ns = 1; ns <= MAX_PAIRS; ++ns) {
        for (int nd = 1; nd <= MAX_PAIRS; ++nd) {
            RuntimeEdgeCacheModel cache{};
            std::array<std::uint32_t, MAX_PAIRS> value{};
            std::array<long long, MAX_PAIRS> reference{};
            for (int si = 0; si < ns; ++si) {
                value[si] = 1u + std::uint32_t(
                    (std::uint64_t(si + 1) * 2654435761ULL +
                     std::uint64_t(ns) * 17ULL + std::uint64_t(nd) * 31ULL) %
                    4294967290ULL);
                const int nedge = std::min(MAX_EDGE_TERMS, nd);
                cache.source[si].count = static_cast<std::uint8_t>(nedge);
                for (int ei = 0; ei < nedge; ++ei) {
                    const int di = (si + ei) % nd;
                    static constexpr std::int8_t COEF[4] = {-2, -1, 1, 2};
                    const std::int8_t coef = COEF[(si + 2 * ei + nd) & 3];
                    cache.source[si].destination[ei] =
                        static_cast<std::uint8_t>(di);
                    cache.source[si].coefficient[ei] = coef;
                    reference[di] += static_cast<long long>(coef) *
                                     static_cast<long long>(value[si]);
                    ++edge_terms;
                }
            }

            std::array<std::array<long long, MAX_DESTINATIONS_PER_LANE>, SUBGROUP>
                routed{};
            for (int lane = 0; lane < SUBGROUP; ++lane) {
                for (int si = 0; si < ns; ++si) {
                    const long long v = static_cast<long long>(value[si]);
                    const auto& row = cache.source[si];
                    for (std::uint8_t ei = 0; ei < row.count; ++ei) {
                        const std::uint8_t di = row.destination[ei];
                        if ((di & (SUBGROUP - 1)) != lane) continue;
                        routed[lane][di >> 3] +=
                            static_cast<long long>(row.coefficient[ei]) * v;
                    }
                }
            }

            for (int di = 0; di < nd; ++di) {
                const long long got = routed[di & (SUBGROUP - 1)][di >> 3];
                if (got != reference[di]) {
                    std::cerr << "edge-cache accumulation mismatch ns=" << ns
                              << " nd=" << nd << " di=" << di
                              << " got=" << got
                              << " expected=" << reference[di] << '\n';
                    return 3;
                }
            }
            ++accumulation_cases;
        }
    }

    if (routing_cases != 210 || accumulation_cases != 400 || max_slot != 2)
        return 4;

    std::cout << "gridfp-runtime-edgecache-proof OK"
              << " routing_cases=" << routing_cases
              << " accumulation_cases=" << accumulation_cases
              << " edge_terms=" << edge_terms
              << " subgroup_width=8 max_pairs=20 max_edge_terms=3"
              << " max_destination_slot=" << max_slot
              << " cache_bytes_per_subgroup=140"
              << " cache_bytes_per_block=4480"
              << " routing_exact=1 accumulation_exact=1\n";
    return 0;
}
