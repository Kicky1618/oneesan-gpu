#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using U64 = std::uint64_t;
using LD = long double;
static constexpr int MAXW = 28;
static constexpr int S = MAXW + 2;

static U64 choose_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    k = std::min(k, n - k);
    U64 r = 1;
    for (int i = 1; i <= k; ++i) r = r * U64(n - k + i) / U64(i);
    return r;
}

static std::vector<std::vector<U64>> high_fixed_counts(int width) {
    std::vector<std::vector<U64>> out(width + 1, std::vector<U64>(MAXW + 3));
    for (int k = 0; k <= width; ++k) {
        std::vector<U64> cur(MAXW + 3), nxt(MAXW + 3);
        cur[1] = 1;
        for (int step = 0; step < k; ++step) {
            std::fill(nxt.begin(), nxt.end(), 0);
            for (int h = 0; h <= MAXW + 1; ++h) if (cur[h]) {
                if (h + 1 < int(nxt.size())) nxt[h + 1] += cur[h];
                if (h > 0) nxt[h - 1] += cur[h];
            }
            cur.swap(nxt);
        }
        out[k] = std::move(cur);
    }
    return out;
}

static std::vector<std::vector<U64>> low_fixed_counts(int width) {
    std::vector<std::vector<U64>> out(width + 1, std::vector<U64>(MAXW + 3));
    for (int k = 0; k <= width; ++k) {
        for (int start = 0; start <= MAXW + 1; ++start) {
            std::vector<U64> cur(MAXW + 3), nxt(MAXW + 3);
            cur[start] = 1;
            for (int step = 0; step < k; ++step) {
                std::fill(nxt.begin(), nxt.end(), 0);
                for (int h = 0; h <= MAXW + 1; ++h) if (cur[h]) {
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

static std::vector<U64> high_all_counts(int width) {
    std::vector<U64> cur(MAXW + 3), nxt(MAXW + 3);
    cur[1] = 1;
    for (int step = 0; step < width; ++step) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= MAXW + 1; ++h) if (cur[h]) {
            nxt[h] += cur[h];
            if (h + 1 < int(nxt.size())) nxt[h + 1] += cur[h];
            if (h > 0) nxt[h - 1] += cur[h];
        }
        cur.swap(nxt);
    }
    return cur;
}

static std::vector<U64> low_all_counts(int width) {
    std::vector<U64> out(MAXW + 3);
    for (int start = 0; start <= MAXW + 1; ++start) {
        std::vector<U64> cur(MAXW + 3), nxt(MAXW + 3);
        cur[start] = 1;
        for (int step = 0; step < width; ++step) {
            std::fill(nxt.begin(), nxt.end(), 0);
            for (int h = 0; h <= MAXW + 1; ++h) if (cur[h]) {
                nxt[h] += cur[h];
                if (h + 1 < int(nxt.size())) nxt[h + 1] += cur[h];
                if (h > 0) nxt[h - 1] += cur[h];
            }
            cur.swap(nxt);
        }
        out[start] = cur[0];
    }
    return out;
}

static std::vector<U64> greedy_bins(
    int bits, const std::vector<U64>& by_popcount, int ngpu
) {
    struct Item { U64 w; std::uint32_t mask; };
    std::vector<Item> items;
    items.reserve(std::size_t(1) << bits);
    for (std::uint32_t m = 0; m < (std::uint32_t(1) << bits); ++m)
        items.push_back({by_popcount[__builtin_popcount(m)], m});
    std::sort(items.begin(), items.end(), [](const Item& a, const Item& b) {
        return a.w != b.w ? a.w > b.w : a.mask < b.mask;
    });
    std::vector<U64> bins(ngpu);
    for (const Item& x : items)
        *std::min_element(bins.begin(), bins.end()) += x.w;
    return bins;
}

static LD gib(LD bytes) { return bytes / LD(U64(1) << 30); }
static LD mib(LD bytes) { return bytes / LD(U64(1) << 20); }

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    const LD usable_gib = argc > 4 ? std::strtold(argv[4], nullptr) : 268.59L;
    const int high = W - 1 - low;
    if (W < 4 || W > MAXW || low < 1 || high < 1 || low >= 16 || high >= 16 ||
        ngpu < 1 || ngpu > 8) {
        std::cerr << "usage: factor_maskshard_memory [W<=28] [LOW] [GPUs] [usable GiB]\n";
        return 1;
    }

    const auto hc_by_k = high_fixed_counts(high);
    const auto lc_by_k = low_fixed_counts(low);
    std::vector<std::vector<U64>> pair(high + 1, std::vector<U64>(low + 1));
    for (int kh = 0; kh <= high; ++kh) {
        for (int kl = 0; kl <= low; ++kl) {
            U64 z = 0;
            for (int he = 0; he <= high + 1; ++he) {
                const U64 a = hc_by_k[kh][he];
                if (!a) continue;
                z += 2 * a * lc_by_k[kl][he];
                if (he + 1 <= MAXW + 1) z += a * lc_by_k[kl][he + 1];
                if (he > 0) z += a * lc_by_k[kl][he - 1];
            }
            pair[kh][kl] = z;
        }
    }

    std::vector<U64> high_mask_weight(high + 1), low_mask_weight(low + 1);
    U64 total = 0;
    for (int kh = 0; kh <= high; ++kh) {
        for (int kl = 0; kl <= low; ++kl) {
            const U64 z = pair[kh][kl];
            high_mask_weight[kh] += choose_u64(low, kl) * z;
            low_mask_weight[kl] += choose_u64(high, kh) * z;
            total += choose_u64(high, kh) * choose_u64(low, kl) * z;
        }
    }

    const auto bins = greedy_bins(high, high_mask_weight, ngpu);
    const U64 max_auth = *std::max_element(bins.begin(), bins.end());
    const U64 max_high_mask_group = *std::max_element(
        high_mask_weight.begin(), high_mask_weight.end());
    const U64 max_low_mask_group = *std::max_element(
        low_mask_weight.begin(), low_mask_weight.end());

    const auto hc_all = high_all_counts(high);
    const auto lc_all = low_all_counts(low);
    const U64 high_all = std::accumulate(hc_all.begin(), hc_all.end(), U64(0));
    const U64 low_all = std::accumulate(lc_all.begin(), lc_all.end(), U64(0));
    const U64 low_dense = U64(1) << (2 * low);
    const U64 high_dense = U64(1) << (2 * high);
    const U64 low_masks = U64(1) << low;
    const U64 high_masks = U64(1) << high;

    const LD factor_bytes = LD(4) * LD(
        2 * low_all + low_masks * S + low_dense +
        2 * high_all + high_masks * S + high_dense);
    const U64 main_blocks = U64(3) * U64(high + 2);
    const U64 block_blocks = U64(high + 2);
    const LD low_begin_bytes = LD(4) * LD(low_masks * S);
    const LD meta_bytes = LD(high_masks)
        + LD(8) * LD(2 * high_masks + high_masks * main_blocks + high_masks * block_blocks)
        + LD(4) * LD(high_all)
        + low_begin_bytes;

    U64 highdesc_main_active = 0, lowdesc_main_active = 0;
    U64 highdesc_block_active = 0, lowdesc_block_active = 0;
    for (int he = 0; he <= high + 1; ++he) {
        for (int cv = 0; cv < 3; ++cv) {
            const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
            if (hs < 0 || hs > low + 1 || !hc_all[he] || !lc_all[hs]) continue;
            highdesc_main_active += hc_all[he];
            lowdesc_main_active += lc_all[hs];
        }
        if (he <= low + 1 && hc_all[he] && lc_all[he]) {
            highdesc_block_active += hc_all[he];
            lowdesc_block_active += lc_all[he];
        }
    }

    const LD highdesc_bytes = LD(4) * LD(high)
        * LD(highdesc_main_active + highdesc_block_active);
    const LD lowdesc_bytes = LD(4) * LD(low)
        * LD(lowdesc_main_active + lowdesc_block_active);

    // v0.5/v0.6: one aux word per active MAIN coordinate.
    const LD high_orbit_aux_bytes = LD(4) * LD(high) * LD(highdesc_main_active);
    const LD low_orbit_aux_bytes = LD(4) * LD(low) * LD(lowdesc_main_active);
    const LD orbit_aux_bytes = high_orbit_aux_bytes + low_orbit_aux_bytes;

    // v0.7: block_desc already gives representative main coordinates, so the
    // extra word follows active BLOCKED coordinates and only carries orbit kind
    // plus the companion target where needed.
    const LD high_block_orbit_aux_bytes = LD(4) * LD(high) * LD(highdesc_block_active);
    const LD low_block_orbit_aux_bytes = LD(4) * LD(low) * LD(lowdesc_block_active);
    const LD block_orbit_aux_bytes = high_block_orbit_aux_bytes + low_block_orbit_aux_bytes;

    const LD auth_bytes = LD(max_auth) * 4;
    const LD low_md_bytes = LD(max_high_mask_group) * 4;
    const LD high_md_bytes = LD(max_low_mask_group) * 4;
    const LD v01_scratch = std::max(low_md_bytes, 2 * high_md_bytes);
    const LD v03_scratch = std::max(low_md_bytes, high_md_bytes);
    const LD v04_scratch = high_md_bytes;
    const LD common = auth_bytes + factor_bytes + meta_bytes;
    const LD v01 = common + v01_scratch;
    const LD v02 = v01 + highdesc_bytes;
    const LD v03 = common + v03_scratch + highdesc_bytes;
    const LD v04 = common + v04_scratch + highdesc_bytes + lowdesc_bytes;
    const LD v05 = v04 + orbit_aux_bytes;
    const LD v07 = v04 + block_orbit_aux_bytes;

    std::cout << std::fixed << std::setprecision(6)
              << "maskshard-memory W=" << W << " low=" << low << " high=" << high
              << " gpus=" << ngpu << '\n'
              << "total_states=" << total
              << " max_auth_gpu_gib=" << double(gib(auth_bytes)) << '\n'
              << "low_md_gib=" << double(gib(low_md_bytes))
              << " high_md_gib=" << double(gib(high_md_bytes)) << '\n'
              << "factor_tables_mib=" << double(mib(factor_bytes))
              << " maskshard_meta_mib=" << double(mib(meta_bytes))
              << " low_begin_mib=" << double(mib(low_begin_bytes)) << '\n'
              << "highdesc_main_active=" << highdesc_main_active
              << " highdesc_block_active=" << highdesc_block_active
              << " highdesc_mib=" << double(mib(highdesc_bytes)) << '\n'
              << "lowdesc_main_active=" << lowdesc_main_active
              << " lowdesc_block_active=" << lowdesc_block_active
              << " lowdesc_mib=" << double(mib(lowdesc_bytes)) << '\n'
              << "high_orbit_aux_mib=" << double(mib(high_orbit_aux_bytes))
              << " low_orbit_aux_mib=" << double(mib(low_orbit_aux_bytes))
              << " orbit_aux_mib=" << double(mib(orbit_aux_bytes)) << '\n'
              << "high_block_orbit_aux_mib=" << double(mib(high_block_orbit_aux_bytes))
              << " low_block_orbit_aux_mib=" << double(mib(low_block_orbit_aux_bytes))
              << " block_orbit_aux_mib=" << double(mib(block_orbit_aux_bytes)) << '\n'
              << "v01_peak_gib=" << double(gib(v01)) << '\n'
              << "v02_highdesc_peak_gib=" << double(gib(v02)) << '\n'
              << "v03_highorbit_peak_gib=" << double(gib(v03)) << '\n'
              << "v04_fullorbit_peak_gib=" << double(gib(v04)) << '\n'
              << "v05_orbitaux_peak_gib=" << double(gib(v05)) << '\n'
              << "v06_blockorbit_peak_gib=" << double(gib(v05)) << '\n'
              << "v07_compact_blockorbit_peak_gib=" << double(gib(v07)) << '\n'
              << "usable_gib=" << double(usable_gib)
              << " v07_headroom_gib=" << double(usable_gib - gib(v07)) << '\n';

    if (gib(v07) >= usable_gib) {
        std::cerr << "v0.7 exceeds requested usable HBM\n";
        return 2;
    }
    return 0;
}
