#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

namespace {

constexpr int SUBGROUP = 8;
constexpr int MAX_PAIRS = 20;
constexpr int MAX_EDGE_TERMS = 3;
constexpr int DEST_BITS = 5;
constexpr std::uint8_t DEST_MASK = (std::uint8_t(1u) << DEST_BITS) - 1u;
constexpr int COEF_BIAS = 1;
constexpr int COEF_MIN = -1;
constexpr int COEF_MAX = 2;
constexpr int MAX_DESTINATIONS_PER_LANE =
    (MAX_PAIRS + SUBGROUP - 1) / SUBGROUP;

constexpr std::uint8_t pack_edge(int destination, int coefficient) {
    return std::uint8_t(
        std::uint8_t(destination) |
        (std::uint8_t(coefficient + COEF_BIAS) << DEST_BITS));
}

constexpr int unpack_destination(std::uint8_t packed) {
    return int(packed & DEST_MASK);
}

constexpr int unpack_coefficient(std::uint8_t packed) {
    return int(packed >> DEST_BITS) - COEF_BIAS;
}

struct EdgeRow {
    std::uint8_t count = 0;
    std::uint8_t packed[MAX_EDGE_TERMS]{};
};

struct RuntimeEdgeCacheModel {
    EdgeRow source[MAX_PAIRS]{};
};

static_assert(MAX_DESTINATIONS_PER_LANE == 3);
static_assert(sizeof(EdgeRow) == 4);
static_assert(sizeof(RuntimeEdgeCacheModel) == 80);

template <std::size_t N>
bool prove_merged_coefficient_bound(
    const std::array<int, N>& primitive,
    int labels,
    std::uint64_t& cases,
    int& observed_min,
    int& observed_max
) {
    std::uint64_t assignments = 1;
    for (std::size_t i = 0; i < N; ++i) assignments *= std::uint64_t(labels);

    for (std::uint64_t code = 0; code < assignments; ++code) {
        std::array<int, MAX_EDGE_TERMS> sum{};
        std::uint64_t x = code;
        for (std::size_t i = 0; i < N; ++i) {
            const int label = int(x % std::uint64_t(labels));
            x /= std::uint64_t(labels);
            sum[std::size_t(label)] += primitive[i];
        }
        for (int label = 0; label < labels; ++label) {
            const int coefficient = sum[std::size_t(label)];
            if (!coefficient) continue;
            observed_min = std::min(observed_min, coefficient);
            observed_max = std::max(observed_max, coefficient);
            if (coefficient < COEF_MIN || coefficient > COEF_MAX) {
                std::cerr << "edge-cache coefficient bound mismatch assignment="
                          << code << " coefficient=" << coefficient << '\n';
                return false;
            }
        }
        ++cases;
    }
    return true;
}

} // namespace

int main() {
    // Interior/runtime_small_step has at most the primitive contributions
    // {+1,+1,-1}; turn compression has {+1,+1}. Enumerating every possible
    // equality partition of those keys proves the post-merge nonzero range
    // used by the two-bit coefficient encoding.
    std::uint64_t coefficient_merge_cases = 0;
    int observed_coef_min = 99;
    int observed_coef_max = -99;
    if (!prove_merged_coefficient_bound(
            std::array<int, 3>{1, 1, -1}, 3,
            coefficient_merge_cases, observed_coef_min, observed_coef_max) ||
        !prove_merged_coefficient_bound(
            std::array<int, 2>{1, 1}, 2,
            coefficient_merge_cases, observed_coef_min, observed_coef_max))
        return 2;
    if (coefficient_merge_cases != 31 ||
        observed_coef_min != COEF_MIN || observed_coef_max != COEF_MAX) {
        std::cerr << "unexpected coefficient proof coverage cases="
                  << coefficient_merge_cases << " observed="
                  << observed_coef_min << ".." << observed_coef_max << '\n';
        return 3;
    }

    std::uint64_t packing_cases = 0;
    for (int destination = 0; destination < MAX_PAIRS; ++destination) {
        for (int coefficient = COEF_MIN; coefficient <= COEF_MAX; ++coefficient) {
            const std::uint8_t packed = pack_edge(destination, coefficient);
            if (unpack_destination(packed) != destination ||
                unpack_coefficient(packed) != coefficient || (packed & 0x80u)) {
                std::cerr << "edge-cache packing mismatch destination=" << destination
                          << " coefficient=" << coefficient
                          << " packed=" << unsigned(packed) << '\n';
                return 4;
            }
            ++packing_cases;
        }
    }

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
                return 5;
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
                    static constexpr std::int8_t COEF[3] = {-1, 1, 2};
                    const int coef = COEF[(si + 2 * ei + nd) % 3];
                    cache.source[si].packed[ei] = pack_edge(di, coef);
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
                        const std::uint8_t packed = row.packed[ei];
                        const int di = unpack_destination(packed);
                        if ((di & (SUBGROUP - 1)) != lane) continue;
                        routed[lane][di >> 3] +=
                            static_cast<long long>(unpack_coefficient(packed)) * v;
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
                    return 6;
                }
            }
            ++accumulation_cases;
        }
    }

    if (packing_cases != 80 || routing_cases != 210 ||
        accumulation_cases != 400 || max_slot != 2)
        return 7;

    std::cout << "gridfp-runtime-edgecache-proof OK"
              << " coefficient_merge_cases=" << coefficient_merge_cases
              << " packing_cases=" << packing_cases
              << " routing_cases=" << routing_cases
              << " accumulation_cases=" << accumulation_cases
              << " edge_terms=" << edge_terms
              << " subgroup_width=8 max_pairs=20 max_edge_terms=3"
              << " coefficient_range=" << observed_coef_min
              << ".." << observed_coef_max
              << " max_destination_slot=" << max_slot
              << " cache_bytes_per_subgroup=80"
              << " cache_bytes_per_block=2560"
              << " coefficient_bound_exact=1 packing_exact=1"
              << " routing_exact=1 accumulation_exact=1\n";
    return 0;
}
