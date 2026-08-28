#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Rank = std::uint64_t;
using Mask = std::uint32_t;

struct Seed {
    Mask support = 0;
    bool blocked = false;
};

struct Tables {
    int W;
    Rank choose[32][32]{};
    Rank primitive[32][32]{};

    explicit Tables(int w) : W(w) {
        for (int n = 0; n < 32; ++n) {
            choose[n][0] = choose[n][n] = 1;
            for (int k = 1; k < n; ++k)
                choose[n][k] = choose[n - 1][k - 1] + choose[n - 1][k];
        }
        primitive[0][0] = 1;
        for (int rem = 1; rem < 32; ++rem) {
            for (int h = 0; h < 31; ++h) {
                primitive[rem][h] = primitive[rem - 1][h + 1] +
                    (h ? primitive[rem - 1][h - 1] : 0);
            }
        }
    }

    Rank binom(int n, int k) const {
        if (n < 0 || k < 0 || k > n) return 0;
        return choose[n][k];
    }
};

Mask rotate_bits(Mask x, int len, int shift) {
    shift %= len;
    if (shift < 0) shift += len;
    if (!shift) return x;
    const Mask mask = (Mask(1) << len) - 1u;
    return ((x << shift) | (x >> (len - shift))) & mask;
}

Rank support_rank(Mask mask, int len, int ones, const Tables& t) {
    Rank rank = 0;
    int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        if (((mask >> pos) & 1u) == 0) continue;
        rank += t.binom(len - pos - 1, left);
        --left;
    }
    return rank;
}

Rank group_size(const Tables& t, int L, int outer_ones) {
    Rank total = 0;
    for (int local = 0; local <= L; ++local) {
        const int occupied = outer_ones + local;
        if (!(occupied & 1)) continue;
        total += (t.binom(L, local) + t.binom(L - 2, local - 1)) *
                 t.primitive[occupied][1];
    }
    return total;
}

int support_owner(
    Mask support,
    int W,
    int K,
    bool reverse,
    int ngpu,
    const Tables& t
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = reverse ? 0 : W - L;
    const int hi = reverse ? L - 1 : W - 1;

    Mask outer = 0;
    int q = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit >= lo && bit <= hi) continue;
        if ((support >> bit) & 1u) outer |= Mask(1) << q;
        ++q;
    }

    const int r = __builtin_popcount(outer);
    const Rank group = group_size(t, L, r);
    const Rank sr = support_rank(outer, O, r, t);
    Rank prefix = 0;
    Rank total = 0;
    for (int x = 0; x <= O; ++x) {
        const Rank g = group_size(t, L, x);
        total += t.binom(O, x) * g;
        if (x < r) prefix += t.binom(O, x) * g;
    }

    const __uint128_t midpoint =
        __uint128_t(prefix) + __uint128_t(sr) * group + group / 2;
    int owner = static_cast<int>(midpoint * ngpu / total);
    if (owner >= ngpu) owner = ngpu - 1;
    return owner;
}

Mask shift_next_support(
    Mask support,
    bool blocked,
    int W,
    int q,
    int K,
    int S,
    bool reverse
) {
    const int span = K + S + 2;
    const int lo = reverse ? 0 : W - span;
    if (!blocked) {
        const Mask mask = (Mask(1) << span) - 1u;
        const Mask x = (support >> lo) & mask;
        const int shift = reverse ? span - S : S;
        return (support & ~(mask << lo)) |
               (rotate_bits(x, span, shift) << lo);
    }

    const int compact_len = span - 2;
    Mask compact = 0;
    int cp = 0;
    for (int bit = lo; bit < lo + span; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        if ((support >> bit) & 1u) compact |= Mask(1) << cp;
        ++cp;
    }
    const int shift = reverse ? compact_len - S : S;
    const Mask rotated = rotate_bits(compact, compact_len, shift);
    Mask out = support;
    cp = 0;
    for (int bit = lo; bit < lo + span; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        out &= ~(Mask(1) << bit);
        if ((rotated >> cp) & 1u) out |= Mask(1) << bit;
        ++cp;
    }
    return out;
}

Mask shift_prev_support(
    Mask support,
    bool blocked,
    int W,
    int q,
    int K,
    int S,
    bool reverse
) {
    const int span = K + S + 2;
    const int lo = reverse ? 0 : W - span;
    if (!blocked) {
        const Mask mask = (Mask(1) << span) - 1u;
        const Mask x = (support >> lo) & mask;
        const int shift = reverse ? S : span - S;
        return (support & ~(mask << lo)) |
               (rotate_bits(x, span, shift) << lo);
    }

    const int compact_len = span - 2;
    Mask compact = 0;
    int cp = 0;
    for (int bit = lo; bit < lo + span; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        if ((support >> bit) & 1u) compact |= Mask(1) << cp;
        ++cp;
    }
    const int shift = reverse ? S : compact_len - S;
    const Mask rotated = rotate_bits(compact, compact_len, shift);
    Mask out = support;
    cp = 0;
    for (int bit = lo; bit < lo + span; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        out &= ~(Mask(1) << bit);
        if ((rotated >> cp) & 1u) out |= Mask(1) << bit;
        ++cp;
    }
    return out;
}

int gcd_int(int a, int b) {
    while (b) {
        const int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

int cycle_leader_length(
    Mask support,
    bool blocked,
    int W,
    int q,
    int K,
    int S,
    bool reverse
) {
    const int len = blocked ? K + S : K + S + 2;
    const int order = len / gcd_int(len, S);
    Mask cur = shift_next_support(support, blocked, W, q, K, S, reverse);
    if (cur == support) return 1;
    Mask minimum = support;
    int count = 1;
    while (cur != support) {
        minimum = std::min(minimum, cur);
        cur = shift_next_support(cur, blocked, W, q, K, S, reverse);
        if (++count > order) return -1;
    }
    return minimum == support ? count : 0;
}

Mask mix32(Mask x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

int batch_id(Mask support, bool blocked, int W, int q, int K, int batches) {
    Mask h = 0;
    if (!blocked) {
        h = Mask(__builtin_popcount(support)) * 0x9e3779b1u;
        constexpr std::pair<int, Mask> terms[] = {
            {1, 0x85ebca6bu},
            {3, 0xc2b2ae35u},
            {5, 0x27d4eb2fu},
            {7, 0x165667b1u},
        };
        for (const auto [distance, coefficient] : terms) {
            h ^= Mask(__builtin_popcount(
                     support & rotate_bits(support, W, distance))) *
                 coefficient;
        }
    } else {
        Mask compact = 0;
        int cp = 0;
        for (int bit = 0; bit < W; ++bit) {
            if (bit == q - 1 || bit == q) continue;
            if ((support >> bit) & 1u) compact |= Mask(1) << cp;
            ++cp;
        }
        const Mask half_mask = (Mask(1) << K) - 1u;
        const Mask a = compact & half_mask;
        const Mask b = (compact >> K) & half_mask;
        const Mask lo = std::min(a, b);
        const Mask hi = std::max(a, b);
        h = lo * 0x9e3779b1u;
        h ^= hi * 0x85ebca6bu;
        h ^= Mask(__builtin_popcount(support)) * 0xc2b2ae35u;
    }
    return int(mix32(h) & Mask(batches - 1));
}

std::vector<Seed> run_seeds(Rank compact, int W, int q, bool reverse) {
    Mask base = 0;
    int cp = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        if ((compact >> cp) & 1ULL) base |= Mask(1) << bit;
        ++cp;
    }
    const int fixed = reverse ? q : q - 1;
    const int missing = reverse ? q - 1 : q;
    const bool odd = (__builtin_popcountll(compact) & 1) != 0;
    if (odd) {
        return {
            {base, false},
            {base | (Mask(1) << (q - 1)) | (Mask(1) << q), false},
        };
    }
    const Mask fixed_support = base | (Mask(1) << fixed);
    return {
        {fixed_support, false},
        {fixed_support, true},
        {base | (Mask(1) << missing), false},
    };
}

bool run_case(int W, int K, int ngpu, int batches, bool reverse) {
    const int S = K;
    if (W != 2 * K + 2) return false;
    const Tables tables(W);
    const int q = (reverse ? 1 : W - 1) + (reverse ? S : -S);

    const auto key = [](Mask support, bool blocked) {
        return (Rank(blocked) << 32) | support;
    };
    std::unordered_map<Rank, int> coverage;
    coverage.reserve(std::size_t(5) * (std::size_t(1) << (W - 3)) * 2);

    Rank total_slabs = 0;
    Rank cross_entries = 0;
    Rank local_entries = 0;
    Rank peer_words = 0;

    for (Rank compact = 0; compact < (Rank(1) << (W - 2)); ++compact) {
        for (const Seed seed : run_seeds(compact, W, q, reverse)) {
            ++total_slabs;
            const int owner = support_owner(
                seed.support, W, K, reverse, ngpu, tables);
            const Mask prev = shift_prev_support(
                seed.support, seed.blocked, W, q, K, S, reverse);
            const int prev_owner = support_owner(
                prev, W, K, reverse, ngpu, tables);
            const Rank primitive_count =
                tables.primitive[__builtin_popcount(seed.support)][1];
            if (!primitive_count) return false;

            if (prev_owner != owner) {
                const int batch = batch_id(
                    seed.support, seed.blocked, W, q, K, batches);
                if (batch < 0 || batch >= batches) return false;
                ++cross_entries;
                peer_words += primitive_count;

                Mask cur = seed.support;
                int guard = 0;
                for (;;) {
                    ++coverage[key(cur, seed.blocked)];
                    const Mask next = shift_next_support(
                        cur, seed.blocked, W, q, K, S, reverse);
                    if (support_owner(next, W, K, reverse, ngpu, tables) != owner)
                        break;
                    cur = next;
                    if (++guard > W) return false;
                }
                continue;
            }

            const int cycle_len = cycle_leader_length(
                seed.support, seed.blocked, W, q, K, S, reverse);
            if (cycle_len < 0) return false;
            if (cycle_len <= 1) continue;

            bool all_local = true;
            Mask cur = seed.support;
            for (int hop = 0; hop < cycle_len; ++hop) {
                if (support_owner(cur, W, K, reverse, ngpu, tables) != owner) {
                    all_local = false;
                    break;
                }
                cur = shift_next_support(
                    cur, seed.blocked, W, q, K, S, reverse);
            }
            if (!all_local) continue;

            ++local_entries;
            cur = seed.support;
            for (int hop = 0; hop < cycle_len; ++hop) {
                ++coverage[key(cur, seed.blocked)];
                cur = shift_next_support(
                    cur, seed.blocked, W, q, K, S, reverse);
            }
        }
    }

    Rank nonfixed = 0;
    Rank fixed = 0;
    Rank bad = 0;
    for (Rank compact = 0; compact < (Rank(1) << (W - 2)); ++compact) {
        for (const Seed seed : run_seeds(compact, W, q, reverse)) {
            const Mask next = shift_next_support(
                seed.support, seed.blocked, W, q, K, S, reverse);
            const int seen = coverage[key(seed.support, seed.blocked)];
            if (next == seed.support) {
                ++fixed;
                if (seen != 0) ++bad;
            } else {
                ++nonfixed;
                if (seen != 1) ++bad;
            }
        }
    }

    if (total_slabs != Rank(5) * (Rank(1) << (W - 3))) return false;

    std::cout << "host-persistent-list-proof"
              << " W=" << W
              << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " support_slabs=" << total_slabs
              << " nonfixed_slabs=" << nonfixed
              << " fixed_slabs=" << fixed
              << " cross_entries=" << cross_entries
              << " local_entries=" << local_entries
              << " list_entries=" << (cross_entries + local_entries)
              << " logical_peer_values=" << peer_words
              << " coverage_bad=" << bad
              << " startup_gpu_support_scan_required=0"
              << " exact=" << (bad ? "FAIL" : "OK") << '\n';
    return bad == 0;
}

} // namespace

int main() {
    for (const int W : {8, 10, 12, 14, 16, 18}) {
        const int K = (W - 2) / 2;
        const int ngpu = std::min(8, 1 << std::min(3, K));
        for (const bool reverse : {false, true}) {
            if (!run_case(W, K, ngpu, 8, reverse)) return 1;
        }
    }
    std::cout << "ALL_OK host_persistent_list_coverage=1\n";
    return 0;
}
