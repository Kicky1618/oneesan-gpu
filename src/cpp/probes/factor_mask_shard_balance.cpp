#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using U64 = std::uint64_t;

static U64 choose_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    k = std::min(k, n - k);
    U64 r = 1;
    for (int i = 1; i <= k; ++i) r = r * U64(n - k + i) / U64(i);
    return r;
}

// With an occupancy mask fixed, zero bits are forced N and therefore do not
// affect the height walk.  The number of legal L/R assignments depends only on
// popcount(mask), not on the positions of its set bits.
static std::vector<std::vector<U64>> build_high_by_popcount(int width) {
    std::vector<std::vector<U64>> out(width + 1,
                                      std::vector<U64>(width + 3));
    for (int k = 0; k <= width; ++k) {
        std::vector<U64> cur(width + 3), next(width + 3);
        cur[1] = 1;
        for (int step = 0; step < k; ++step) {
            std::fill(next.begin(), next.end(), 0);
            for (int h = 0; h <= width + 1; ++h) {
                U64 c = cur[h];
                if (!c) continue;
                next[h + 1] += c;       // L
                if (h > 0) next[h - 1] += c; // R
            }
            cur.swap(next);
        }
        out[k] = cur;
    }
    return out;
}

static std::vector<std::vector<U64>> build_low_by_popcount(int width) {
    std::vector<std::vector<U64>> out(width + 1,
                                      std::vector<U64>(width + 3));
    for (int k = 0; k <= width; ++k) {
        for (int start = 0; start <= width + 1; ++start) {
            std::vector<U64> cur(width + 3), next(width + 3);
            cur[start] = 1;
            for (int step = 0; step < k; ++step) {
                std::fill(next.begin(), next.end(), 0);
                for (int h = 0; h <= width + 1; ++h) {
                    U64 c = cur[h];
                    if (!c) continue;
                    if (h + 1 < int(next.size())) next[h + 1] += c; // L
                    if (h > 0) next[h - 1] += c;                    // R
                }
                cur.swap(next);
            }
            out[k][start] = cur[0];
        }
    }
    return out;
}

struct Balance {
    std::vector<U64> bins;
    std::vector<int> masks;
};

static Balance greedy_balance(int bits, const std::vector<U64>& weight_by_popcount,
                              int ngpu) {
    struct Item { U64 w; std::uint32_t mask; };
    std::vector<Item> items;
    items.reserve(std::size_t(1) << bits);
    for (std::uint32_t m = 0; m < (std::uint32_t(1) << bits); ++m)
        items.push_back({weight_by_popcount[__builtin_popcount(m)], m});
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        if (a.w != b.w) return a.w > b.w;
        return a.mask < b.mask;
    });

    Balance b;
    b.bins.assign(ngpu, 0);
    b.masks.assign(ngpu, 0);
    for (const Item& x : items) {
        int d = int(std::min_element(b.bins.begin(), b.bins.end()) - b.bins.begin());
        b.bins[d] += x.w;
        ++b.masks[d];
    }
    return b;
}

static long double gib(U64 counts) {
    return (long double(counts) * 4.0L) / (long double(U64(1) << 30));
}
static long double tib_bytes(long double bytes) {
    return bytes / long double(U64(1) << 40);
}

int main(int argc, char** argv) {
    int W = argc > 1 ? std::atoi(argv[1]) : 28;
    int low = argc > 2 ? std::atoi(argv[2]) : 14;
    int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    int high = W - 1 - low;
    if (W < 3 || W > 28 || low < 1 || high < 1 || ngpu < 1 || ngpu > 8) {
        std::cerr << "usage: factor_mask_shard_balance [W<=28] [LOW] [1..8 GPUs]\n";
        return 1;
    }

    auto hc = build_high_by_popcount(high);
    auto lc = build_low_by_popcount(low);

    // pair[kh][kl] is the exact number of main+blocked states for one specific
    // HIGH mask of popcount kh and one specific LOW mask of popcount kl.
    std::vector<std::vector<U64>> pair(high + 1,
                                       std::vector<U64>(low + 1));
    for (int kh = 0; kh <= high; ++kh) {
        for (int kl = 0; kl <= low; ++kl) {
            U64 z = 0;
            for (int he = 0; he <= high + 1; ++he) {
                U64 a = hc[kh][he];
                if (!a) continue;
                // blocked + main(center=N)
                z += a * 2 * lc[kl][he];
                // main(center=L/R)
                if (he + 1 < int(lc[kl].size())) z += a * lc[kl][he + 1];
                if (he > 0) z += a * lc[kl][he - 1];
            }
            pair[kh][kl] = z;
        }
    }

    std::vector<U64> high_mask_weight(high + 1), low_mask_weight(low + 1);
    U64 total = 0;
    for (int kh = 0; kh <= high; ++kh) {
        for (int kl = 0; kl <= low; ++kl) {
            U64 w = pair[kh][kl];
            high_mask_weight[kh] += choose_u64(low, kl) * w;
            low_mask_weight[kl] += choose_u64(high, kh) * w;
            total += choose_u64(high, kh) * choose_u64(low, kl) * w;
        }
    }

    auto hb = greedy_balance(high, high_mask_weight, ngpu);
    auto lb = greedy_balance(low, low_mask_weight, ngpu);
    U64 hsum = std::accumulate(hb.bins.begin(), hb.bins.end(), U64(0));
    U64 lsum = std::accumulate(lb.bins.begin(), lb.bins.end(), U64(0));
    if (hsum != total || lsum != total) {
        std::cerr << "balance total mismatch\n";
        return 2;
    }

    // Mutual information between HIGH/LOW occupancy popcounts, weighted by the
    // actual authoritative state count.  A tiny value means that trying to
    // align the two mask-sharded layouts by popcount cannot make a transpose
    // substantially more local than chance.
    std::vector<long double> ph(high + 1), pl(low + 1);
    std::vector<std::vector<long double>> joint(high + 1,
                                                 std::vector<long double>(low + 1));
    for (int kh = 0; kh <= high; ++kh) {
        for (int kl = 0; kl <= low; ++kl) {
            long double x = long double(choose_u64(high, kh))
                          * long double(choose_u64(low, kl))
                          * long double(pair[kh][kl]) / long double(total);
            joint[kh][kl] = x;
            ph[kh] += x;
            pl[kl] += x;
        }
    }
    long double mi = 0;
    for (int kh = 0; kh <= high; ++kh)
        for (int kl = 0; kl <= low; ++kl) {
            long double p = joint[kh][kl];
            if (p > 0 && ph[kh] > 0 && pl[kl] > 0)
                mi += p * std::log2(p / (ph[kh] * pl[kl]));
        }

    U64 max_high_group = *std::max_element(high_mask_weight.begin(), high_mask_weight.end());
    U64 max_low_group = *std::max_element(low_mask_weight.begin(), low_mask_weight.end());
    U64 hmax = *std::max_element(hb.bins.begin(), hb.bins.end());
    U64 hmin = *std::min_element(hb.bins.begin(), hb.bins.end());
    U64 lmax = *std::max_element(lb.bins.begin(), lb.bins.end());
    U64 lmin = *std::min_element(lb.bins.begin(), lb.bins.end());
    long double avg = long double(total) / ngpu;

    long double auth_bytes = long double(total) * 4.0L;
    long double one_window_roundtrip = 2.0L * W * auth_bytes;
    long double two_window_roundtrip = 2.0L * one_window_roundtrip;
    long double uniform_remote = ngpu == 1 ? 0.0L : long double(ngpu - 1) / ngpu;

    std::cout << std::fixed << std::setprecision(6)
              << "factor-mask-shard W=" << W << " low=" << low << " high=" << high
              << " ngpu=" << ngpu << '\n'
              << "total_states=" << total
              << " authoritative_gib=" << double(gib(total)) << '\n'
              << "high_mask_max_group_gib=" << double(gib(max_high_group))
              << " low_mask_max_group_gib=" << double(gib(max_low_group)) << '\n'
              << "high_shard_min_gib=" << double(gib(hmin))
              << " high_shard_max_gib=" << double(gib(hmax))
              << " high_imbalance_frac=" << double((long double(hmax - hmin)) / avg) << '\n'
              << "low_shard_min_gib=" << double(gib(lmin))
              << " low_shard_max_gib=" << double(gib(lmax))
              << " low_imbalance_frac=" << double((long double(lmax - lmin)) / avg) << '\n'
              << "popcount_mutual_information_bits=" << double(mi) << '\n'
              << "two_window_group_roundtrip_tib_per_residue="
              << double(tib_bytes(two_window_roundtrip)) << '\n'
              << "one_window_group_roundtrip_tib_per_residue="
              << double(tib_bytes(one_window_roundtrip)) << '\n'
              << "uniform_1_over_ngpu_local_remote_tib_two_windows="
              << double(tib_bytes(two_window_roundtrip * uniform_remote)) << '\n'
              << "mask_shard_one_window_local_remote_tib_model="
              << double(tib_bytes(one_window_roundtrip * uniform_remote)) << '\n';

    std::cout << "high_shard_gib=";
    for (int d = 0; d < ngpu; ++d) {
        if (d) std::cout << ',';
        std::cout << double(gib(hb.bins[d]));
    }
    std::cout << '\n';
    std::cout << "low_shard_gib=";
    for (int d = 0; d < ngpu; ++d) {
        if (d) std::cout << ',';
        std::cout << double(gib(lb.bins[d]));
    }
    std::cout << '\n';
    return 0;
}
