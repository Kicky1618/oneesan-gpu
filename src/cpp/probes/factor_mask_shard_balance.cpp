#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using U64 = std::uint64_t;
using LD = long double;

static U64 choose_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    k = std::min(k, n - k);
    U64 r = 1;
    for (int i = 1; i <= k; ++i) r = r * U64(n - k + i) / U64(i);
    return r;
}

// For a fixed occupancy mask, zero bits are forced N and do not change height.
// Therefore the topology count depends only on popcount(mask), not bit positions.
static std::vector<std::vector<U64>> high_counts(int width) {
    std::vector<std::vector<U64>> out(width + 1, std::vector<U64>(width + 3));
    for (int k = 0; k <= width; ++k) {
        std::vector<U64> cur(width + 3), nxt(width + 3);
        cur[1] = 1;
        for (int step = 0; step < k; ++step) {
            std::fill(nxt.begin(), nxt.end(), 0);
            for (int h = 0; h <= width + 1; ++h) {
                if (!cur[h]) continue;
                nxt[h + 1] += cur[h];
                if (h > 0) nxt[h - 1] += cur[h];
            }
            cur.swap(nxt);
        }
        out[k] = std::move(cur);
    }
    return out;
}

static std::vector<std::vector<U64>> low_counts(int width) {
    std::vector<std::vector<U64>> out(width + 1, std::vector<U64>(width + 3));
    for (int k = 0; k <= width; ++k) {
        for (int start = 0; start <= width + 1; ++start) {
            std::vector<U64> cur(width + 3), nxt(width + 3);
            cur[start] = 1;
            for (int step = 0; step < k; ++step) {
                std::fill(nxt.begin(), nxt.end(), 0);
                for (int h = 0; h <= width + 1; ++h) {
                    if (!cur[h]) continue;
                    if (h + 1 < int(nxt.size())) nxt[h + 1] += cur[h];
                    if (h > 0) nxt[h - 1] += cur[h];
                }
                cur.swap(nxt);
            }
            out[k][start] = cur[0];
        }
    }
    return out;
}

struct Balance {
    std::vector<U64> bins;
    std::vector<std::uint8_t> owner;
};

static Balance greedy_balance(int bits, const std::vector<U64>& weight, int ngpu) {
    struct Item { U64 w; std::uint32_t mask; };
    const std::uint32_t nm = std::uint32_t(1) << bits;
    std::vector<Item> items;
    items.reserve(nm);
    for (std::uint32_t m = 0; m < nm; ++m)
        items.push_back({weight[__builtin_popcount(m)], m});
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return a.w != b.w ? a.w > b.w : a.mask < b.mask;
    });

    Balance out{std::vector<U64>(ngpu), std::vector<std::uint8_t>(nm)};
    for (const Item& x : items) {
        const int d = int(std::min_element(out.bins.begin(), out.bins.end()) - out.bins.begin());
        out.bins[d] += x.w;
        out.owner[x.mask] = std::uint8_t(d);
    }
    return out;
}

// If authoritative data is sharded by HIGH mask, a HIGH-active window fixes a
// LOW mask and touches pieces owned by all GPUs.  Schedule that LOW-mask group
// on the GPU owning the largest piece; only the remainder crosses peer links.
static U64 remote_states_high_active(
    int high, int low, int ngpu,
    const std::vector<std::vector<U64>>& pair,
    const Balance& high_shard
) {
    const std::uint32_t hm_count = std::uint32_t(1) << high;
    const std::uint32_t lm_count = std::uint32_t(1) << low;
    U64 remote = 0;
    for (std::uint32_t lm = 0; lm < lm_count; ++lm) {
        const int kl = __builtin_popcount(lm);
        std::vector<U64> by_gpu(ngpu);
        U64 group = 0;
        for (std::uint32_t hm = 0; hm < hm_count; ++hm) {
            const U64 w = pair[__builtin_popcount(hm)][kl];
            by_gpu[high_shard.owner[hm]] += w;
            group += w;
        }
        remote += group - *std::max_element(by_gpu.begin(), by_gpu.end());
    }
    return remote;
}

// Symmetric model for authoritative data sharded by LOW mask.  LOW-active
// groups then span all LOW-mask owners, while the HIGH-active window is local.
static U64 remote_states_low_active(
    int high, int low, int ngpu,
    const std::vector<std::vector<U64>>& pair,
    const Balance& low_shard
) {
    const std::uint32_t hm_count = std::uint32_t(1) << high;
    const std::uint32_t lm_count = std::uint32_t(1) << low;
    U64 remote = 0;
    for (std::uint32_t hm = 0; hm < hm_count; ++hm) {
        const int kh = __builtin_popcount(hm);
        std::vector<U64> by_gpu(ngpu);
        U64 group = 0;
        for (std::uint32_t lm = 0; lm < lm_count; ++lm) {
            const U64 w = pair[kh][__builtin_popcount(lm)];
            by_gpu[low_shard.owner[lm]] += w;
            group += w;
        }
        remote += group - *std::max_element(by_gpu.begin(), by_gpu.end());
    }
    return remote;
}

static LD gib(U64 states) {
    return LD(states) * 4.0L / LD(U64(1) << 30);
}
static LD tib(LD bytes) {
    return bytes / LD(U64(1) << 40);
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    const int high = W - 1 - low;
    if (W < 3 || W > 28 || low < 1 || high < 1 || ngpu < 1 || ngpu > 8) {
        std::cerr << "usage: factor_mask_shard_balance [W<=28] [LOW] [1..8 GPUs]\n";
        return 1;
    }

    const auto hc = high_counts(high);
    const auto lc = low_counts(low);

    // Exact main+blocked state count for one specific pair of masks having
    // popcounts (kh, kl). Main center N and blocked share the same height;
    // center L/R shift the LOW starting height by +/-1.
    std::vector<std::vector<U64>> pair(high + 1, std::vector<U64>(low + 1));
    for (int kh = 0; kh <= high; ++kh) {
        for (int kl = 0; kl <= low; ++kl) {
            U64 z = 0;
            for (int he = 0; he <= high + 1; ++he) {
                const U64 a = hc[kh][he];
                if (!a) continue;
                z += 2 * a * lc[kl][he];
                if (he + 1 < int(lc[kl].size())) z += a * lc[kl][he + 1];
                if (he > 0) z += a * lc[kl][he - 1];
            }
            pair[kh][kl] = z;
        }
    }

    std::vector<U64> high_weight(high + 1), low_weight(low + 1);
    U64 total = 0;
    for (int kh = 0; kh <= high; ++kh) {
        for (int kl = 0; kl <= low; ++kl) {
            const U64 w = pair[kh][kl];
            high_weight[kh] += choose_u64(low, kl) * w;
            low_weight[kl] += choose_u64(high, kh) * w;
            total += choose_u64(high, kh) * choose_u64(low, kl) * w;
        }
    }

    const Balance hb = greedy_balance(high, high_weight, ngpu);
    const Balance lb = greedy_balance(low, low_weight, ngpu);
    if (std::accumulate(hb.bins.begin(), hb.bins.end(), U64(0)) != total ||
        std::accumulate(lb.bins.begin(), lb.bins.end(), U64(0)) != total) {
        std::cerr << "balance total mismatch\n";
        return 2;
    }

    // Mutual information of HIGH/LOW mask popcounts in the authoritative-state
    // distribution. A tiny value predicts little benefit from trying to align
    // two independent mask-sharded layouts by popcount class alone.
    std::vector<LD> ph(high + 1), pl(low + 1);
    std::vector<std::vector<LD>> joint(high + 1, std::vector<LD>(low + 1));
    for (int kh = 0; kh <= high; ++kh) {
        for (int kl = 0; kl <= low; ++kl) {
            const LD p = LD(choose_u64(high, kh)) * LD(choose_u64(low, kl)) *
                         LD(pair[kh][kl]) / LD(total);
            joint[kh][kl] = p;
            ph[kh] += p;
            pl[kl] += p;
        }
    }
    LD mi = 0;
    for (int kh = 0; kh <= high; ++kh)
        for (int kl = 0; kl <= low; ++kl) {
            const LD p = joint[kh][kl];
            if (p > 0 && ph[kh] > 0 && pl[kl] > 0)
                mi += p * std::log2(p / (ph[kh] * pl[kl]));
        }

    const U64 hmin = *std::min_element(hb.bins.begin(), hb.bins.end());
    const U64 hmax = *std::max_element(hb.bins.begin(), hb.bins.end());
    const U64 lmin = *std::min_element(lb.bins.begin(), lb.bins.end());
    const U64 lmax = *std::max_element(lb.bins.begin(), lb.bins.end());
    const U64 max_high_group = *std::max_element(high_weight.begin(), high_weight.end());
    const U64 max_low_group = *std::max_element(low_weight.begin(), low_weight.end());
    const LD avg = LD(total) / ngpu;

    const U64 high_shard_remote_states = remote_states_high_active(high, low, ngpu, pair, hb);
    const U64 low_shard_remote_states = remote_states_low_active(high, low, ngpu, pair, lb);
    const LD auth_bytes = LD(total) * 4.0L;
    const LD one_window_roundtrip = 2.0L * W * auth_bytes;
    const LD two_window_roundtrip = 2.0L * one_window_roundtrip;
    const LD uniform_remote = ngpu == 1 ? 0.0L : LD(ngpu - 1) / ngpu;
    const LD high_shard_peer_bytes = 2.0L * W * LD(high_shard_remote_states) * 4.0L;
    const LD low_shard_peer_bytes = 2.0L * W * LD(low_shard_remote_states) * 4.0L;

    std::cout << std::fixed << std::setprecision(6)
              << "factor-mask-shard W=" << W << " low=" << low << " high=" << high
              << " ngpu=" << ngpu << '\n'
              << "total_states=" << total << " authoritative_gib=" << double(gib(total)) << '\n'
              << "high_mask_max_group_gib=" << double(gib(max_high_group))
              << " low_mask_max_group_gib=" << double(gib(max_low_group)) << '\n'
              << "high_shard_min_gib=" << double(gib(hmin))
              << " high_shard_max_gib=" << double(gib(hmax))
              << " high_imbalance_frac=" << double(LD(hmax - hmin) / avg) << '\n'
              << "low_shard_min_gib=" << double(gib(lmin))
              << " low_shard_max_gib=" << double(gib(lmax))
              << " low_imbalance_frac=" << double(LD(lmax - lmin) / avg) << '\n'
              << "popcount_mutual_information_bits=" << double(mi) << '\n'
              << "two_window_group_roundtrip_tib_per_residue=" << double(tib(two_window_roundtrip)) << '\n'
              << "uniform_flat_remote_tib_two_windows=" << double(tib(two_window_roundtrip * uniform_remote)) << '\n'
              << "high_mask_shard_remote_fraction=" << double(LD(high_shard_remote_states) / total)
              << " high_mask_shard_peer_tib=" << double(tib(high_shard_peer_bytes)) << '\n'
              << "low_mask_shard_remote_fraction=" << double(LD(low_shard_remote_states) / total)
              << " low_mask_shard_peer_tib=" << double(tib(low_shard_peer_bytes)) << '\n';

    std::cout << "high_shard_gib=";
    for (int d = 0; d < ngpu; ++d) std::cout << (d ? "," : "") << double(gib(hb.bins[d]));
    std::cout << '\n' << "low_shard_gib=";
    for (int d = 0; d < ngpu; ++d) std::cout << (d ? "," : "") << double(gib(lb.bins[d]));
    std::cout << '\n';
    return 0;
}
