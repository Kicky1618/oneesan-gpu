#include <array>
#include <cstdint>
#include <iostream>
#include <set>
#include <utility>
#include <vector>

namespace {
using Rank64 = std::uint64_t;
static constexpr int MAX = 28;
std::array<std::array<Rank64, MAX + 2>, MAX + 1> P{};

void build() {
    P[0][0] = 1;
    for (int rem = 1; rem <= MAX; ++rem) {
        for (int h = 0; h <= MAX; ++h) {
            Rank64 z = P[rem - 1][h + 1];
            if (h > 0) z += P[rem - 1][h - 1];
            P[rem][h] = z;
        }
    }
}

int packed_base(int rem) {
    if (rem <= 13) {
        const int m = rem >> 1;
        return (rem & 1) ? m * (m + 2) : m * m + m - 1;
    }
    const int s = 26 - rem;
    const int m = s >> 1;
    const int tail = (s & 1)
        ? (m + 1) * (m + 2)
        : (m + 1) * (m + 1);
    return 104 - tail;
}

int packed_index(int rem, int h) {
    const int j = h - 1;
    return packed_base(rem) + (j >> 1);
}

std::array<std::uint32_t, 104> make_packed() {
    std::array<std::uint32_t, 104> out{};
    std::array<bool, 104> used{};
    for (int rem = 1; rem <= 26; ++rem) {
        for (int j = rem & 1; j <= std::min(rem, 26 - rem); j += 2) {
            const int ix = packed_base(rem) + (j >> 1);
            if (ix < 0 || ix >= 104 || used[std::size_t(ix)]) {
                std::cerr << "bad packed index rem=" << rem << " j=" << j
                          << " ix=" << ix << '\n';
                std::exit(2);
            }
            const Rank64 v = P[rem][j];
            if (v > 0xffffffffULL) return {};
            out[std::size_t(ix)] = static_cast<std::uint32_t>(v);
            used[std::size_t(ix)] = true;
        }
    }
    for (bool x : used) if (!x) std::exit(3);
    return out;
}

Rank64 packed_threshold(
    const std::array<std::uint32_t, 104>& packed, int rem, int h
) {
    if (h <= 0) return 0;
    return packed[std::size_t(packed_index(rem, h))];
}

// Decode rank to the primitive L/R path. 0=R, 1=L.
std::vector<int> unrank(int occupied, Rank64 rank) {
    std::vector<int> path;
    path.reserve(std::size_t(occupied));
    int h = 1;
    for (int seen = 0; seen < occupied; ++seen) {
        const int rem = occupied - seen - 1;
        const Rank64 r_count = h > 0 ? P[rem][h - 1] : 0;
        if (rank < r_count) {
            path.push_back(0);
            --h;
        } else {
            rank -= r_count;
            path.push_back(1);
            ++h;
        }
    }
    if (h != 0 || rank != 0) std::exit(4);
    return path;
}

Rank64 rank_ref(const std::vector<int>& path) {
    int h = 1;
    Rank64 rank = 0;
    for (std::size_t i = 0; i < path.size(); ++i) {
        const int rem = int(path.size() - i - 1);
        if (path[i]) {
            if (h > 0) rank += P[rem][h - 1];
            ++h;
        } else {
            --h;
        }
    }
    return rank;
}

Rank64 rank_packed(
    const std::vector<int>& path,
    const std::array<std::uint32_t, 104>& packed,
    std::set<std::pair<int,int>>& cells,
    std::uint64_t& loads
) {
    int h = 1;
    Rank64 rank = 0;
    for (std::size_t i = 0; i < path.size(); ++i) {
        const int rem = int(path.size() - i - 1);
        if (path[i]) {
            if (h > 0) {
                if (rem == 0) std::exit(5); // final symbol of a valid primitive is R.
                const Rank64 v = packed_threshold(packed, rem, h);
                if (v != P[rem][h - 1]) std::exit(6);
                rank += v;
                cells.emplace(rem, h - 1);
                ++loads;
            }
            ++h;
        } else {
            --h;
        }
    }
    return rank;
}
}

int main() {
    build();
    const auto packed = make_packed();
    std::set<std::pair<int,int>> cells;
    std::uint64_t cases = 0;
    std::uint64_t loads = 0;
    for (int occupied = 1; occupied <= 27; occupied += 2) {
        const Rank64 count = P[occupied][1];
        for (Rank64 r = 0; r < count; ++r) {
            const auto path = unrank(occupied, r);
            if (path.empty() || path.back() != 0) return 7;
            const Rank64 a = rank_ref(path);
            const Rank64 b = rank_packed(path, packed, cells, loads);
            if (a != r || b != r || a != b) {
                std::cerr << "rank mismatch occupied=" << occupied
                          << " rank=" << r << " ref=" << a
                          << " packed=" << b << '\n';
                return 8;
            }
            ++cases;
        }
    }
    if (cases != 3707851ULL) return 9;
    if (cells.size() != 91) return 10;
    std::uint32_t max_value = 0;
    for (auto v : packed) if (v > max_value) max_value = v;
    if (max_value != 742900u) return 11;
    std::cout << "gridfp-runtime-primitive-rank-packed-proof OK"
              << " primitive_cases=" << cases
              << " rank_threshold_cells=" << cells.size()
              << " shared_packed_entries=104"
              << " shared_packed_bytes=416"
              << " max_value=" << max_value
              << " threshold_loads=" << loads
              << " exact=1\n";
    return 0;
}
