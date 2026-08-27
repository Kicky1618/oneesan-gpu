#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

using U64 = std::uint64_t;
static constexpr int W = 28;
static constexpr int L = 14;
static constexpr int H = 13;
static constexpr int F = W / 2;
static constexpr int NG = 8;
static constexpr int THRESHOLD = 29;

static U64 choose_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    k = std::min(k, n - k);
    U64 r = 1;
    for (int i = 1; i <= k; ++i) r = r * U64(n - k + i) / U64(i);
    return r;
}

static int pc(std::uint32_t x) {
    int n = 0;
    while (x) { x &= x - 1; ++n; }
    return n;
}

static U64 low_fixed(int occupied, int start, int cap) {
    if (start < 0 || start > cap) return 0;
    std::array<U64, F + 2> cur{}, nxt{};
    cur[std::size_t(start)] = 1;
    for (int s = 0; s < occupied; ++s) {
        nxt.fill(0);
        for (int h = 0; h <= cap; ++h) if (cur[std::size_t(h)]) {
            if (h) nxt[std::size_t(h - 1)] += cur[std::size_t(h)];
            if (h + 1 <= cap) nxt[std::size_t(h + 1)] += cur[std::size_t(h)];
        }
        cur = nxt;
    }
    return cur[0];
}

static U64 high_fixed(int occupied, int end, int cap) {
    if (end < 0 || end > cap || cap < 1) return 0;
    std::array<U64, F + 2> cur{}, nxt{};
    cur[1] = 1;
    for (int s = 0; s < occupied; ++s) {
        nxt.fill(0);
        for (int h = 0; h <= cap; ++h) if (cur[std::size_t(h)]) {
            if (h) nxt[std::size_t(h - 1)] += cur[std::size_t(h)];
            if (h + 1 <= cap) nxt[std::size_t(h + 1)] += cur[std::size_t(h)];
        }
        cur = nxt;
    }
    return cur[std::size_t(end)];
}

static U64 high_all(int end, int cap) {
    U64 z = 0;
    for (int k = 0; k <= H; ++k)
        z += choose_u64(H, k) * high_fixed(k, end, cap);
    return z;
}

static U64 low_all(int start) {
    U64 z = 0;
    for (int k = 0; k <= L; ++k)
        z += choose_u64(L, k) * low_fixed(k, start, F);
    return z;
}

using PeakHist = std::array<U64, F + 1>;
using CenterHist = std::array<PeakHist, 3>;
using HeightHist = std::array<CenterHist, H + 2>;
using ClosureHist = std::array<HeightHist, H>;

static void enum_high_rec(
    int pos, int h, int peak, std::array<std::uint8_t, H>& sym,
    ClosureHist& hist
) {
    if (pos < 0) {
        for (int qi = 0; qi < H; ++qi) {
            for (int cv = 0; cv < 3; ++cv) {
                const int pair = qi == 0
                    ? int(cv) | (int(sym[0]) << 2)
                    : int(sym[std::size_t(qi - 1)])
                        | (int(sym[std::size_t(qi)]) << 2);
                if (pair == 0xa || pair == 0x5 || pair == 0x6)
                    ++hist[std::size_t(qi)][std::size_t(h)]
                           [std::size_t(cv)][std::size_t(peak)];
            }
        }
        return;
    }
    sym[std::size_t(pos)] = 0;
    enum_high_rec(pos - 1, h, peak, sym, hist);
    if (h) {
        sym[std::size_t(pos)] = 1;
        enum_high_rec(pos - 1, h - 1, peak, sym, hist);
    }
    sym[std::size_t(pos)] = 2;
    enum_high_rec(pos - 1, h + 1, std::max(peak, h + 1), sym, hist);
}

static ClosureHist build_closure_hist() {
    ClosureHist hist{};
    std::array<std::uint8_t, H> sym{};
    enum_high_rec(H - 1, 1, 1, sym, hist);
    for (int qi = 0; qi < H; ++qi)
        for (int he = 0; he <= H + 1; ++he)
            for (int cv = 0; cv < 3; ++cv)
                for (int cap = 2; cap <= F; ++cap)
                    hist[std::size_t(qi)][std::size_t(he)][std::size_t(cv)]
                        [std::size_t(cap)] +=
                    hist[std::size_t(qi)][std::size_t(he)][std::size_t(cv)]
                        [std::size_t(cap - 1)];
    return hist;
}

static U64 closure_lanes(const ClosureHist& hist, int k, int cap) {
    U64 lanes = 0;
    for (int qi = 0; qi < H; ++qi) {
        for (int he = 0; he <= H + 1; ++he) {
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs < 0 || hs > L) continue;
                const U64 rows = hist[std::size_t(qi)][std::size_t(he)]
                    [std::size_t(cv)][std::size_t(cap)];
                const U64 cols = low_fixed(k, hs, cap);
                const U64 dense = low_fixed(k, hs, F);
                if (!rows || !cols || !dense) continue;
                if (dense < THRESHOLD)
                    lanes += ((rows * cols + 31) / 32) * 32;
                else
                    lanes += rows * ((cols + 31) / 32) * 32;
            }
        }
    }
    return lanes;
}

static U64 main_n(int k) {
    U64 z = 0;
    for (int he = 0; he <= H + 1; ++he) {
        const U64 hc = high_all(he, F);
        for (int cv = 0; cv < 3; ++cv) {
            const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
            if (0 <= hs && hs <= L + 1) z += hc * low_fixed(k, hs, F);
        }
    }
    return z;
}

static U64 block_n(int k) {
    U64 z = 0;
    for (int he = 0; he <= H + 1; ++he)
        z += high_all(he, F) * low_fixed(k, he, F);
    return z;
}

static U64 auth_main(int j) {
    U64 z = 0;
    for (int he = 0; he <= H + 1; ++he) {
        const U64 hc = high_fixed(j, he, F);
        for (int cv = 0; cv < 3; ++cv) {
            const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
            if (0 <= hs && hs <= L + 1) z += hc * low_all(hs);
        }
    }
    return z;
}

static U64 auth_block(int j) {
    U64 z = 0;
    for (int he = 0; he <= H + 1; ++he)
        z += high_fixed(j, he, F) * low_all(he);
    return z;
}

struct Plan {
    std::array<U64, NG> load{};
    std::array<std::vector<int>, NG> jobs;
};

enum class Mode { PLAIN, AFFINITY, LOCALITY };

static Plan build_plan(
    const std::vector<int>& job_class,
    const std::array<U64, L + 1>& weight,
    const std::array<std::array<U64, NG>, L + 1>& local,
    Mode mode
) {
    std::vector<std::size_t> order(job_class.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        const U64 wa = weight[std::size_t(job_class[a])];
        const U64 wb = weight[std::size_t(job_class[b])];
        return wa != wb ? wa > wb : a < b;
    });
    Plan p;
    bool seen[NG][L + 1]{};
    for (std::size_t q : order) {
        const int k = job_class[q];
        const U64 mn = *std::min_element(p.load.begin(), p.load.end());
        int best = -1;
        bool br = false;
        U64 bl = 0;
        for (int d = 0; d < NG; ++d) {
            if (p.load[std::size_t(d)] != mn) continue;
            if (mode == PLAIN) { best = d; break; }
            const bool reuse = seen[d][k];
            const U64 loc = mode == LOCALITY ? local[std::size_t(k)][std::size_t(d)] : 0;
            if (best < 0 || reuse > br || (reuse == br && loc > bl)) {
                best = d; br = reuse; bl = loc;
            }
        }
        p.jobs[std::size_t(best)].push_back(k);
        p.load[std::size_t(best)] += weight[std::size_t(k)];
        seen[best][k] = true;
    }
    return p;
}

static U64 graph_classes(const Plan& p) {
    U64 z = 0;
    for (int d = 0; d < NG; ++d) {
        bool seen[L + 1]{};
        for (int k : p.jobs[std::size_t(d)]) if (!seen[k]) {
            seen[k] = true;
            ++z;
        }
    }
    return z;
}

static U64 peer_io(
    const Plan& p,
    const std::array<U64, L + 1>& total,
    const std::array<std::array<U64, NG>, L + 1>& local
) {
    U64 z = 0;
    for (int d = 0; d < NG; ++d)
        for (int k : p.jobs[std::size_t(d)])
            z += total[std::size_t(k)] - local[std::size_t(k)][std::size_t(d)];
    return z;
}

static bool same_loads(const Plan& a, const Plan& b) {
    auto x = a.load, y = b.load;
    std::sort(x.begin(), x.end());
    std::sort(y.begin(), y.end());
    return x == y;
}

int main() {
    const ClosureHist closure = build_closure_hist();
    std::array<U64, L + 1> mn{}, dn{};
    for (int k = 0; k <= L; ++k) { mn[k] = main_n(k); dn[k] = block_n(k); }
    U64 main_total = 0, block_total = 0;
    for (int k = 0; k <= L; ++k) {
        main_total += choose_u64(L, k) * mn[k];
        block_total += choose_u64(L, k) * dn[k];
    }
    if (main_total != 385719506620ULL || block_total != 135015505407ULL) {
        std::cerr << "n27 authoritative regression mismatch main=" << main_total
                  << " block=" << block_total << '\n';
        return 1;
    }

    U64 closure_aggregate = 0;
    for (int row = 1; row <= W; ++row) {
        const int cap = std::min(row, F);
        for (int k = 0; k <= L; ++k)
            closure_aggregate += choose_u64(L, k) * closure_lanes(closure, k, cap);
    }
    if (closure_aggregate != 42734081059456ULL) {
        std::cerr << "n27 closure lane regression mismatch " << closure_aggregate << '\n';
        return 2;
    }

    std::array<U64, H + 1> aw{};
    for (int j = 0; j <= H; ++j) aw[j] = auth_main(j) + auth_block(j);
    std::vector<std::uint32_t> masks(1u << H);
    std::iota(masks.begin(), masks.end(), 0u);
    std::sort(masks.begin(), masks.end(), [&](std::uint32_t a, std::uint32_t b) {
        const U64 wa = aw[std::size_t(pc(a))], wb = aw[std::size_t(pc(b))];
        return wa != wb ? wa > wb : a < b;
    });
    std::array<U64, NG> shard_load{};
    std::array<std::uint8_t, 1u << H> owner{};
    for (std::uint32_t mask : masks) {
        int d = int(std::min_element(shard_load.begin(), shard_load.end())
                    - shard_load.begin());
        owner[mask] = std::uint8_t(d);
        shard_load[std::size_t(d)] += aw[std::size_t(pc(mask))];
    }
    if (*std::max_element(shard_load.begin(), shard_load.end()) != 65092277859ULL) {
        std::cerr << "n27 shard max regression mismatch\n";
        return 3;
    }

    std::array<std::array<U64, H + 1>, NG> owner_class{};
    for (std::uint32_t mask = 0; mask < owner.size(); ++mask)
        ++owner_class[owner[mask]][std::size_t(pc(mask))];
    U64 high_active[NG][H + 2][F + 1]{};
    for (int d = 0; d < NG; ++d)
        for (int he = 0; he <= H + 1; ++he)
            for (int cap = 1; cap <= F; ++cap)
                for (int j = 0; j <= H; ++j)
                    high_active[d][he][cap] += owner_class[d][j]
                        * high_fixed(j, he, cap);

    std::array<std::array<std::array<U64, NG>, F + 1>, L + 1> local{};
    std::array<std::array<U64, F + 1>, L + 1> total_io{};
    for (int k = 0; k <= L; ++k) for (int cap = 1; cap <= F; ++cap) {
        const int gc = cap < F ? std::max(cap - 1, 1) : F - 1;
        for (int d = 0; d < NG; ++d) {
            auto main_active = [&](int c) {
                U64 z = 0;
                for (int he = 0; he <= H + 1; ++he)
                    for (int cv = 0; cv < 3; ++cv) {
                        const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                        if (0 <= hs && hs <= L + 1)
                            z += high_active[d][he][c] * low_fixed(k, hs, c);
                    }
                return z;
            };
            auto block_active = [&](int c) {
                U64 z = 0;
                for (int he = 0; he <= H + 1; ++he)
                    z += high_active[d][he][c] * low_fixed(k, he, c);
                return z;
            };
            const U64 mg = main_active(gc), ms = main_active(cap), bs = block_active(cap);
            U64 z = mg + ms + bs;
            if (cap == F) z += U64(W - F) * (2 * ms + bs);
            local[k][cap][d] = z;
            total_io[k][cap] += z;
        }
    }

    std::vector<int> job_class;
    for (int k = L; k >= 0; --k)
        for (U64 q = 0; q < choose_u64(L, k); ++q) job_class.push_back(k);
    if (job_class.size() != (1u << L)) return 4;

    U64 graph_v077 = 0, graph_v078 = 0;
    U64 peer_v077 = 0, peer_v078 = 0;
    int selected_caps = 0;
    for (int cap = 1; cap <= F; ++cap) {
        std::array<U64, L + 1> weight{};
        std::array<std::array<U64, NG>, L + 1> loc{};
        std::array<U64, L + 1> tot{};
        for (int k = 0; k <= L; ++k) {
            U64 orbit = 0;
            for (int he = 0; he <= H + 1; ++he)
                orbit += high_all(he, cap) * low_fixed(k, he, cap);
            weight[k] = 2 * mn[k] + dn[k] + U64(H) * orbit
                + closure_lanes(closure, k, cap);
            loc[k] = local[k][cap];
            tot[k] = total_io[k][cap];
        }
        const Plan plain = build_plan(job_class, weight, loc, PLAIN);
        const Plan affinity = build_plan(job_class, weight, loc, AFFINITY);
        const Plan v077 = graph_classes(affinity) < graph_classes(plain)
            ? affinity : plain;
        const Plan candidate = build_plan(job_class, weight, loc, LOCALITY);
        if (!same_loads(v077, candidate)) {
            std::cerr << "n27 locality changed load multiset cap=" << cap << '\n';
            return 5;
        }
        const U64 gc77 = graph_classes(v077), gcc = graph_classes(candidate);
        const U64 p77 = peer_io(v077, tot, loc), pcand = peer_io(candidate, tot, loc);
        const bool adopt = gcc <= gc77 && pcand <= p77 && (gcc < gc77 || pcand < p77);
        const Plan& final = adopt ? candidate : v077;
        selected_caps += adopt ? 1 : 0;
        graph_v077 += gc77;
        graph_v078 += graph_classes(final);
        peer_v077 += p77;
        peer_v078 += peer_io(final, tot, loc);
    }

    if (graph_v077 != 1470ULL || graph_v078 != 1470ULL
        || peer_v077 != 19315327806382ULL
        || peer_v078 != 19315151844499ULL
        || selected_caps != 10) {
        std::cerr << "n27 locality schedule regression graph77=" << graph_v077
                  << " graph78=" << graph_v078
                  << " peer77=" << peer_v077
                  << " peer78=" << peer_v078
                  << " selected=" << selected_caps << '\n';
        return 6;
    }

    const U64 peer_saved = peer_v077 - peer_v078;
    std::cout << "factor-high-cap-lpt-locality-n27 OK"
              << " main=" << main_total
              << " block=" << block_total
              << " closure_lanes=" << closure_aggregate
              << " shard_max=" << *std::max_element(shard_load.begin(), shard_load.end())
              << " graph_v077=" << graph_v077
              << " graph_v078=" << graph_v078
              << " peer_v077=" << peer_v077
              << " peer_v078=" << peer_v078
              << " peer_saved=" << peer_saved
              << " peer_saved_gib=" << double(peer_saved * 4ULL) / double(1ULL << 30)
              << " selected_caps=" << selected_caps << '\n';
    return 0;
}
