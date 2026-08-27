#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_set>
#include <vector>

#define main odd_tl_factorization_unused_main
#include "odd_tl_gram_factorization_probe.cpp"
#undef main

static constexpr int HLEN = 13;
static constexpr int LLEN = 14;

// Packed coordinate used by the future midpoint CUDA kernel:
//   bits  0.. 9 : LOW mask-local topology rank
//   bits 10..19 : HIGH mask-local topology rank
//   bits 20..21 : center MateValue (N=0,R=1,L=2)
//   bits 22..25 : HIGH ending height he
// The LOW starting height hs is determined by (he,center).
static std::uint32_t pack_coord(int he, int center, int hr, int lr) {
    if (he < 0 || he >= 16 || center < 0 || center >= 4
        || hr < 0 || hr >= 1024 || lr < 0 || lr >= 1024) {
        std::cerr << "coordinate overflow he=" << he << " center=" << center
                  << " hr=" << hr << " lr=" << lr << "\n";
        std::exit(90);
    }
    return std::uint32_t(lr)
        | (std::uint32_t(hr) << 10)
        | (std::uint32_t(center) << 20)
        | (std::uint32_t(he) << 22);
}

struct Ways {
    // ways[len][start][end] for nonnegative +/-1 walks.
    std::uint32_t a[15][17][17]{};
    Ways() {
        for (int h = 0; h < 17; ++h) a[0][h][h] = 1;
        for (int len = 1; len <= 14; ++len) {
            for (int s = 0; s < 17; ++s) {
                for (int e = 0; e < 17; ++e) {
                    std::uint32_t z = 0;
                    if (s > 0) z += a[len - 1][s - 1][e]; // R first
                    if (s + 1 < 17) z += a[len - 1][s + 1][e]; // then L
                    a[len][s][e] = z;
                }
            }
        }
    }
} G_WAYS;

// Rank a compressed occupied segment scanned high->low.  dense_positions are
// supplied as a contiguous range of the odd-TL word, but visited in reverse.
// odd-TL U corresponds to frontier R, and D to frontier L.
static int segment_rank(
    std::uint32_t word,
    int dense_lo,
    int k,
    int start_h,
    int end_h
) {
    int rank = 0;
    int h = start_h;
    for (int t = 0; t < k; ++t) {
        int p = dense_lo + (k - 1 - t);
        bool is_R = ((word >> p) & 1u) != 0; // U -> R
        int rem = k - 1 - t;
        if (!is_R) {
            // L is second in the factorized recursion; count the R branch first.
            if (h > 0) rank += int(G_WAYS.a[rem][h - 1][end_h]);
            ++h;
        } else {
            if (h <= 0) return -1;
            --h;
        }
    }
    return h == end_h ? rank : -1;
}

static std::uint32_t coord_for_word(
    std::uint32_t word,
    int kh,
    int center_occ,
    int kl
) {
    int m = kh + center_occ + kl;
    assert(m >= 1 && (m & 1));

    int h = 1;
    // Determine he by actually scanning the HIGH occupied terminals high->low.
    for (int p = m - 1; p >= kl + center_occ; --p) {
        if ((word >> p) & 1u) --h; // U -> R
        else ++h;                  // D -> L
        if (h < 0) std::abort();
    }
    int he = h;
    int hr = segment_rank(word, kl + center_occ, kh, 1, he);
    if (hr < 0) std::abort();

    int center = 0; // N
    if (center_occ) {
        bool is_R = ((word >> kl) & 1u) != 0;
        center = is_R ? 1 : 2;
        h += is_R ? -1 : 1;
        if (h < 0) std::abort();
    }
    int hs = h;
    int lr = segment_rank(word, 0, kl, hs, 0);
    if (lr < 0) std::abort();
    return pack_coord(he, center, hr, lr);
}

static std::uint32_t reflected_word(std::uint32_t word, int m) {
    auto mate = mates(word, m);
    std::vector<int> rm(m, -1);
    for (int i = 0; i < m; ++i) {
        int ni = m - 1 - i;
        if (mate[i] < 0) {
            rm[ni] = -1;
        } else {
            rm[ni] = m - 1 - mate[i];
        }
    }

    std::uint32_t out = 0;
    for (int i = 0; i < m; ++i) {
        // U iff this terminal is the defect or opens an arc to a later terminal.
        if (rm[i] < 0 || rm[i] > i) out |= 1u << i;
    }
    return out;
}

int main() {
    std::uint64_t total_entries = 0;
    int max_hr = 0, max_lr = 0;
    std::array<std::uint64_t, 28> by_m{};

    // Reflection permutations are independent of the HIGH/LOW split.
    std::uint64_t reflection_entries = 0;
    for (int m = 1; m <= 27; m += 2) {
        auto const& B = basis(m, 1);
        std::vector<int> perm(B.words.size(), -1);
        for (int i = 0; i < int(B.words.size()); ++i) {
            auto rw = reflected_word(B.words[i], m);
            auto it = B.rank.find(rw);
            if (it == B.rank.end()) {
                std::cerr << "reflection left basis m=" << m << " i=" << i << "\n";
                return 1;
            }
            perm[i] = it->second;
        }
        for (int i = 0; i < int(perm.size()); ++i) {
            if (perm[perm[i]] != i) {
                std::cerr << "reflection is not involutive m=" << m << " i=" << i << "\n";
                return 2;
            }
        }
        reflection_entries += perm.size();
    }

    // Compile every possible segment-count tuple.  Exact physical occupancy
    // masks with the same counts use the same mask-local topology ranks.
    int tuples = 0;
    for (int kh = 0; kh <= HLEN; ++kh) {
        for (int center = 0; center <= 1; ++center) {
            for (int kl = 0; kl <= LLEN; ++kl) {
                int m = kh + center + kl;
                if (m < 1 || !(m & 1) || m > 27) continue;
                auto const& B = basis(m, 1);
                std::unordered_set<std::uint32_t> seen;
                seen.reserve(B.words.size() * 2 + 1);
                for (auto w : B.words) {
                    auto packed = coord_for_word(w, kh, center, kl);
                    if (!seen.insert(packed).second) {
                        std::cerr << "duplicate factor coordinate kh=" << kh
                                  << " c=" << center << " kl=" << kl
                                  << " m=" << m << "\n";
                        return 3;
                    }
                    int hr = int((packed >> 10) & 1023u);
                    int lr = int(packed & 1023u);
                    max_hr = std::max(max_hr, hr);
                    max_lr = std::max(max_lr, lr);
                }
                if (seen.size() != B.words.size()) return 4;
                total_entries += B.words.size();
                by_m[m] += B.words.size();
                ++tuples;
            }
        }
    }

    std::cout << "midpoint_factor_map"
              << " tuples=" << tuples
              << " entries=" << total_entries
              << " bytes_u32=" << total_entries * sizeof(std::uint32_t)
              << " mib_u32=" << double(total_entries * sizeof(std::uint32_t)) / (1 << 20)
              << " reflection_entries=" << reflection_entries
              << " reflection_mib=" << double(reflection_entries * sizeof(std::uint32_t)) / (1 << 20)
              << " max_high_local_rank=" << max_hr
              << " max_low_local_rank=" << max_lr << "\n";

    for (int m = 1; m <= 27; m += 2) {
        std::cout << "m=" << m << " tuple_entries=" << by_m[m] << "\n";
    }

    if (tuples != 210 || total_entries != 16878801ull
        || reflection_entries != 3707851ull) {
        std::cerr << "unexpected table cardinality\n";
        return 5;
    }
    if (max_hr >= 1024 || max_lr >= 1024) return 6;

    std::cout << "PASS midpoint factor-map compiler\n";
    return 0;
}
