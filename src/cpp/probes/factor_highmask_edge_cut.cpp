#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using oneesan::gridfp::MateID;
using oneesan::gridfp::MateValue;
using oneesan::gridfp::N;
using oneesan::gridfp::R;
using oneesan::gridfp::L;
using U64 = std::uint64_t;
using LD = long double;

static std::uint32_t occ(std::uint32_t code, int len) {
    std::uint32_t z = 0;
    for (int p = 0; p < len; ++p)
        if ((code >> (2 * p)) & 3u) z |= 1u << p;
    return z;
}

static std::vector<std::vector<std::uint32_t>> enumerate_low(int len) {
    std::vector<std::vector<std::uint32_t>> by_h(len + 2);
    for (int h0 = 0; h0 <= len + 1; ++h0) {
        auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
            if (pos < 0) {
                if (h == 0) by_h[h0].push_back(code);
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, code);
            if (h > 0) self(self, pos - 1, h - 1,
                            code | (std::uint32_t(R) << (2 * pos)));
            self(self, pos - 1, h + 1,
                 code | (std::uint32_t(L) << (2 * pos)));
        };
        rec(rec, len - 1, h0, 0);
    }
    return by_h;
}

static std::vector<std::vector<std::uint32_t>> enumerate_high(int len) {
    std::vector<std::vector<std::uint32_t>> by_h(len + 2);
    auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
        if (pos < 0) {
            if (h >= 0 && h < int(by_h.size())) by_h[h].push_back(code);
            return;
        }
        self(self, pos - 1, h, code);
        if (h > 0) self(self, pos - 1, h - 1,
                        code | (std::uint32_t(R) << (2 * pos)));
        self(self, pos - 1, h + 1,
             code | (std::uint32_t(L) << (2 * pos)));
    };
    rec(rec, len - 1, 1, 0);
    return by_h;
}

static int gf2_rank3(std::array<std::uint32_t, 3> x) {
    int rank = 0;
    for (int bit = 31; bit >= 0 && rank < 3; --bit) {
        int pivot = -1;
        for (int r = rank; r < 3; ++r)
            if ((x[r] >> bit) & 1u) { pivot = r; break; }
        if (pivot < 0) continue;
        std::swap(x[rank], x[pivot]);
        for (int r = 0; r < 3; ++r)
            if (r != rank && ((x[r] >> bit) & 1u)) x[r] ^= x[rank];
        ++rank;
    }
    return rank;
}

static int owner_linear(std::uint32_t mask, const std::array<std::uint32_t, 3>& row) {
    return (__builtin_parity(mask & row[0]) << 0)
         | (__builtin_parity(mask & row[1]) << 1)
         | (__builtin_parity(mask & row[2]) << 2);
}

static LD tib(LD bytes) { return bytes / LD(U64(1) << 40); }
static LD gib_states(U64 states) { return LD(states) * 4.0L / LD(U64(1) << 30); }

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const int trials = argc > 3 ? std::atoi(argv[3]) : 10000;
    const LD max_load_ratio = argc > 4 ? std::strtold(argv[4], nullptr) : 1.025L;
    const int high = W - 1 - low;
    if (W < 3 || W > 28 || low < 1 || high < 3 || high > 15) {
        std::cerr << "usage: factor_highmask_edge_cut [W<=28] [LOW] [trials] [max_load_ratio]\n";
        return 1;
    }

    const auto lows = enumerate_low(low);
    const auto highs = enumerate_high(high);
    const std::uint32_t nm = 1u << high;
    const MateID high_code_mask = (MateID(1) << (2 * high)) - 1;

    std::vector<U64> node_weight(nm);
    std::vector<U64> delta_weight(nm);
    U64 total_transition_weight = 0;
    U64 total_states = 0;

    // Main states. One representative LOW topology is enough for HIGH-side
    // topology; multiply by the complete LOW count for the matching height.
    for (int he = 0; he < int(highs.size()); ++he) {
        for (std::uint32_t hc : highs[he]) {
            const std::uint32_t sm = occ(hc, high);
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == int(L) ? 1 : cv == int(R) ? -1 : 0);
                if (hs < 0 || hs >= int(lows.size()) || lows[hs].empty()) continue;
                const U64 weight = lows[hs].size();
                node_weight[sm] += weight;
                total_states += weight;
                const std::uint32_t lc = lows[hs][0];
                const MateID m = MateID(lc)
                    | (MateID(cv) << (2 * low))
                    | (MateID(hc) << (2 * (low + 1)));
                for (int p = W - 1; p >= low + 1; --p) {
                    const auto z = oneesan::gridfp::include_horizontal(m, W, p);
                    if (!z.valid) continue;
                    const int shift = z.blocked ? low : low + 1;
                    const std::uint32_t hc2 = std::uint32_t((z.mate >> (2 * shift)) & high_code_mask);
                    const std::uint32_t tm = occ(hc2, high);
                    delta_weight[sm ^ tm] += weight;
                    total_transition_weight += weight;
                }
            }
        }
    }

    // Blocked -> main exclusion branch.
    for (int he = 0; he < int(highs.size()) && he < int(lows.size()); ++he) {
        if (lows[he].empty()) continue;
        const U64 weight = lows[he].size();
        const std::uint32_t lc = lows[he][0];
        for (std::uint32_t hc : highs[he]) {
            const std::uint32_t sm = occ(hc, high);
            node_weight[sm] += weight;
            total_states += weight;
            const MateID m = MateID(lc) | (MateID(hc) << (2 * low));
            for (int p = W - 1; p >= low + 1; --p) {
                const MateID z = oneesan::gridfp::blocked_exclude(m, p);
                const std::uint32_t hc2 = std::uint32_t((z >> (2 * (low + 1))) & high_code_mask);
                const std::uint32_t tm = occ(hc2, high);
                delta_weight[sm ^ tm] += weight;
                total_transition_weight += weight;
            }
        }
    }

    const U64 expected_n27 = 520735012027ULL;
    if (W == 28 && low == 14 && total_states != expected_n27) {
        std::cerr << "state total mismatch: " << total_states << " != " << expected_n27 << '\n';
        return 2;
    }

    const LD average = LD(total_states) / 8.0L;
    const U64 max_allowed = U64(average * max_load_ratio);
    std::mt19937 rng(1618);
    std::uniform_int_distribution<std::uint32_t> bits(1, nm - 1);

    std::array<std::uint32_t, 3> best{};
    U64 best_cut = std::numeric_limits<U64>::max();
    U64 best_max_load = 0;
    int accepted = 0;

    for (int trial = 0; trial < trials; ++trial) {
        std::array<std::uint32_t, 3> row{bits(rng), bits(rng), bits(rng)};
        if (gf2_rank3(row) != 3) continue;

        std::array<U64, 8> load{};
        for (std::uint32_t m = 0; m < nm; ++m)
            load[owner_linear(m, row)] += node_weight[m];
        const U64 mx = *std::max_element(load.begin(), load.end());
        if (mx > max_allowed) continue;
        ++accepted;

        U64 cut = 0;
        for (std::uint32_t delta = 1; delta < nm; ++delta)
            if (owner_linear(delta, row) != 0) cut += delta_weight[delta];
        if (cut < best_cut || (cut == best_cut && mx < best_max_load)) {
            best_cut = cut;
            best = row;
            best_max_load = mx;
        }
    }

    if (best_cut == std::numeric_limits<U64>::max()) {
        std::cerr << "no balanced linear shard found in " << trials << " trials\n";
        return 3;
    }

    std::array<U64, 8> best_load{};
    for (std::uint32_t m = 0; m < nm; ++m)
        best_load[owner_linear(m, best)] += node_weight[m];
    const U64 local_transition_weight = total_transition_weight - best_cut;
    const LD direct_peer_bytes_per_residue = LD(best_cut) * 4.0L * W;

    std::cout << std::fixed << std::setprecision(6)
              << "factor-highmask-edge-cut W=" << W << " low=" << low << " high=" << high
              << " trials=" << trials << " accepted=" << accepted << '\n'
              << "states=" << total_states
              << " state_gib=" << double(gib_states(total_states)) << '\n'
              << "transition_updates_per_high_window=" << total_transition_weight
              << " nonzero_delta_updates=" << (total_transition_weight - delta_weight[0]) << '\n'
              << "best_rows=0x" << std::hex << best[0]
              << ",0x" << best[1] << ",0x" << best[2] << std::dec << '\n'
              << "best_max_authoritative_gib=" << double(gib_states(best_max_load))
              << " max_load_ratio=" << double(LD(best_max_load) / average) << '\n'
              << "remote_update_fraction=" << double(LD(best_cut) / total_transition_weight)
              << " local_update_fraction=" << double(LD(local_transition_weight) / total_transition_weight) << '\n'
              << "direct_peer_tib_per_residue=" << double(tib(direct_peer_bytes_per_residue)) << '\n';

    std::cout << "shard_gib=";
    for (int d = 0; d < 8; ++d)
        std::cout << (d ? "," : "") << double(gib_states(best_load[d]));
    std::cout << '\n';
    return 0;
}
