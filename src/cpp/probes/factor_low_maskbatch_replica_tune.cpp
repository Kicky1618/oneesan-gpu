#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

using U64 = std::uint64_t;

struct LowCode {
    std::uint32_t code = 0;
    std::uint8_t peak = 0;
};

static bool closure_pair(std::uint32_t lc, int cv, int low, int p) {
    const std::uint32_t active = lc | (std::uint32_t(cv) << (2 * low));
    const std::uint32_t w = (active >> (2 * (p - 1))) & 15u;
    return w == 0xau || w == 0x5u || w == 0x6u;
}

struct ReplicaMetrics {
    U64 total_desc = 0;
    std::vector<U64> device_desc;
    U64 worst_tasks_per_cta = 0;
    U64 max_replicas_used = 0;
};

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    const U64 target = argc > 4 ? std::strtoull(argv[4], nullptr, 10) : 16384ULL;
    const int high = W - 1 - low;
    const int full_cap = (W + 1) / 2;
    const int orbit_full_cap = W / 2;
    if (W < 4 || W > 30 || low < 1 || high < 1 || low >= 16 || high >= 16
        || full_cap >= 16 || ngpu < 1 || ngpu > 8 || target < 1)
        return 1;

    const std::uint32_t nm = 1u << high;

    // Exact HIGH mask-local row counts by ending height and row-depth cap.
    std::vector<std::array<U64, 16>> high_cum(
        std::size_t(nm) * (high + 2));
    auto hrec = [&](auto&& self, int pos, int h, int peak,
                    std::uint32_t mask) -> void {
        if (pos < 0) {
            auto& a = high_cum[std::size_t(mask) * (high + 2) + h];
            for (int cap = peak; cap <= full_cap; ++cap) ++a[size_t(cap)];
            return;
        }
        self(self, pos - 1, h, peak, mask);
        if (h > 0)
            self(self, pos - 1, h - 1, peak, mask | (1u << pos));
        self(self, pos - 1, h + 1, std::max(peak, h + 1),
             mask | (1u << pos));
    };
    hrec(hrec, high - 1, 1, 1, 0u);

    // Exact LOW-all code lists grouped by starting height.  The code encoding
    // matches the closure descriptor probes: R=1, L=2, N=0.
    std::vector<std::vector<LowCode>> low_codes(low + 2);
    for (int h0 = 0; h0 <= low + 1; ++h0) {
        auto lrec = [&](auto&& self, int pos, int h, int peak,
                        std::uint32_t code) -> void {
            if (pos < 0) {
                if (h == 0)
                    low_codes[size_t(h0)].push_back(
                        {code, std::uint8_t(peak)});
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, peak, code);
            if (h > 0)
                self(self, pos - 1, h - 1, peak,
                     code | (1u << (2 * pos)));
            self(self, pos - 1, h + 1, std::max(peak, h + 1),
                 code | (2u << (2 * pos)));
        };
        lrec(lrec, low - 1, h0, h0, 0u);
    }

    std::vector<std::array<U64, 16>> low_cum(low + 2);
    for (int h = 0; h <= low + 1; ++h)
        for (const LowCode& x : low_codes[size_t(h)])
            for (int cap = int(x.peak); cap <= full_cap; ++cap)
                ++low_cum[size_t(h)][size_t(cap)];

    // Exact selected closure columns for each LOW position, MAIN FBlock kind,
    // and cap.  Flatten [pi][he][cv][cap].
    const int PSTRIDE = (high + 2) * 3 * 16;
    std::vector<U64> selected(std::size_t(low) * PSTRIDE, 0);
    auto six = [&](int pi, int he, int cv, int cap) -> std::size_t {
        return std::size_t(pi) * PSTRIDE
             + (std::size_t(he) * 3 + std::size_t(cv)) * 16
             + std::size_t(cap);
    };
    for (int p = low; p >= 1; --p) {
        const int pi = low - p;
        for (int he = 0; he <= high + 1; ++he) {
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs < 0 || hs > low + 1) continue;
                for (const LowCode& x : low_codes[size_t(hs)]) {
                    if (!closure_pair(x.code, cv, low, p)) continue;
                    for (int cap = int(x.peak); cap <= full_cap; ++cap)
                        ++selected[six(pi, he, cv, cap)];
                }
            }
        }
    }

    // Exact batched warp-task counts, indexed by [mask][cap] and
    // [mask][cap][LOW position].
    std::vector<U64> orbit(std::size_t(nm) * (full_cap + 1), 0);
    std::vector<U64> closure(
        std::size_t(nm) * (full_cap + 1) * low, 0);
    auto oix = [&](std::uint32_t mask, int cap) -> std::size_t {
        return std::size_t(mask) * (full_cap + 1) + std::size_t(cap);
    };
    auto cix = [&](std::uint32_t mask, int cap, int pi) -> std::size_t {
        return (std::size_t(mask) * (full_cap + 1) + std::size_t(cap))
             * low + std::size_t(pi);
    };

    U64 max_orbit_group = 0, max_closure_group = 0;
    for (std::uint32_t mask = 0; mask < nm; ++mask) {
        for (int cap = 1; cap <= full_cap; ++cap) {
            const int ocap = std::min(cap, orbit_full_cap);
            U64 ot = 0;
            for (int h = 0; h <= high + 1; ++h) {
                const U64 hc = high_cum[
                    std::size_t(mask) * (high + 2) + h][size_t(ocap)];
                const U64 lc = low_cum[size_t(h)][size_t(ocap)];
                ot += hc * ((lc + 31ULL) >> 5);
            }
            orbit[oix(mask, cap)] = ot;
            max_orbit_group = std::max(max_orbit_group, ot);

            for (int pi = 0; pi < low; ++pi) {
                U64 ct = 0;
                for (int he = 0; he <= high + 1; ++he) {
                    const U64 rows = high_cum[
                        std::size_t(mask) * (high + 2) + he][size_t(cap)];
                    if (!rows) continue;
                    for (int cv = 0; cv < 3; ++cv) {
                        const U64 cols = selected[six(pi, he, cv, cap)];
                        ct += rows * ((cols + 31ULL) >> 5);
                    }
                }
                closure[cix(mask, cap, pi)] = ct;
                max_closure_group = std::max(max_closure_group, ct);
            }
        }
    }

    // Reconstruct the exact authoritative state weight used by the HIGH-mask
    // LPT shard planner: all MAIN center-value blocks plus BLOCKED blocks.
    std::vector<U64> weight(nm, 0);
    U64 authoritative_total = 0;
    for (std::uint32_t mask = 0; mask < nm; ++mask) {
        U64 main = 0, block = 0;
        for (int he = 0; he <= high + 1; ++he) {
            const U64 rows = high_cum[
                std::size_t(mask) * (high + 2) + he][size_t(full_cap)];
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == 2 ? 1 : cv == 1 ? -1 : 0);
                if (hs >= 0 && hs <= low + 1)
                    main += rows * U64(low_codes[size_t(hs)].size());
            }
            if (he <= low + 1)
                block += rows * U64(low_codes[size_t(he)].size());
        }
        weight[mask] = main + block;
        authoritative_total += weight[mask];
    }

    std::vector<std::uint32_t> order(nm);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](std::uint32_t a,
                                              std::uint32_t b) {
        return weight[a] != weight[b] ? weight[a] > weight[b] : a < b;
    });
    std::vector<U64> load(size_t(ngpu), 0);
    std::vector<std::uint8_t> owner(nm, 0);
    for (std::uint32_t mask : order) {
        int d = 0;
        for (int q = 1; q < ngpu; ++q)
            if (load[size_t(q)] < load[size_t(d)]) d = q;
        owner[mask] = std::uint8_t(d);
        load[size_t(d)] += weight[mask];
    }

    auto measure = [&](int max_replicas) -> ReplicaMetrics {
        ReplicaMetrics m;
        m.device_desc.assign(size_t(ngpu), 0);
        auto add = [&](std::uint32_t mask, U64 tasks) {
            if (!tasks) return;
            const U64 wanted = (tasks + target - 1) / target;
            const U64 replicas = std::min<U64>(
                U64(max_replicas), std::max<U64>(1, wanted));
            m.total_desc += replicas;
            m.device_desc[size_t(owner[mask])] += replicas;
            m.max_replicas_used = std::max(m.max_replicas_used, replicas);
            m.worst_tasks_per_cta = std::max(
                m.worst_tasks_per_cta,
                (tasks + replicas - 1) / replicas);
        };
        for (int cap = 1; cap <= full_cap; ++cap) {
            for (std::uint32_t mask = 0; mask < nm; ++mask) {
                add(mask, orbit[oix(mask, cap)]);
                for (int pi = 0; pi < low; ++pi)
                    add(mask, closure[cix(mask, cap, pi)]);
            }
        }
        return m;
    };

    const std::array<int, 7> caps{{16, 32, 64, 128, 256, 512, 1024}};
    std::array<ReplicaMetrics, caps.size()> results;
    for (std::size_t i = 0; i < caps.size(); ++i)
        results[i] = measure(caps[i]);

    if (W == 28 && low == 14 && ngpu == 8 && target == 16384ULL) {
        if (authoritative_total != 520735012027ULL
            || max_orbit_group != 15954186ULL
            || max_closure_group != 14097070ULL
            || results[0].total_desc != 16722484ULL
            || *std::max_element(results[0].device_desc.begin(),
                                 results[0].device_desc.end()) != 2095862ULL
            || results[0].worst_tasks_per_cta != 997137ULL
            || results[2].total_desc != 30228725ULL
            || results[2].worst_tasks_per_cta != 249285ULL
            || results[4].total_desc != 35892777ULL
            || results[4].worst_tasks_per_cta != 62322ULL
            || results[5].total_desc != 36303193ULL
            || results[5].worst_tasks_per_cta != 31161ULL
            || results[6].total_desc != 36350637ULL
            || *std::max_element(results[6].device_desc.begin(),
                                 results[6].device_desc.end()) != 4545074ULL
            || results[6].worst_tasks_per_cta != 16384ULL
            || results[6].max_replicas_used != 974ULL) {
            std::cerr << "n=27 LOW mask-batch replica tuning regression\n";
            return 2;
        }
    }

    std::cout << "low-maskbatch-replica-tune W=" << W
              << " low=" << low << " high=" << high
              << " gpus=" << ngpu << " target=" << target << '\n'
              << "authoritative_states=" << authoritative_total
              << " max_orbit_group_tasks=" << max_orbit_group
              << " max_closure_group_tasks=" << max_closure_group << '\n';
    std::cout << std::fixed << std::setprecision(6);
    for (std::size_t i = 0; i < caps.size(); ++i) {
        const ReplicaMetrics& m = results[i];
        const U64 max_dev = *std::max_element(
            m.device_desc.begin(), m.device_desc.end());
        std::cout << "max_replicas=" << caps[i]
                  << " total_descriptors=" << m.total_desc
                  << " max_device_descriptors=" << max_dev
                  << " max_device_descriptor_mib="
                  << double(max_dev * 8ULL) / double(1ULL << 20)
                  << " worst_tasks_per_cta=" << m.worst_tasks_per_cta
                  << " max_replicas_used=" << m.max_replicas_used
                  << '\n';
    }
    return 0;
}
