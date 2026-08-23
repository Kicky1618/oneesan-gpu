#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using oneesan::gridfp::MateID;
using oneesan::gridfp::MateValue;
using oneesan::gridfp::N;
using oneesan::gridfp::R;
using oneesan::gridfp::L;

static void enum_rec(int pos, int h, MateID m, std::vector<MateID>& out) {
    if (pos < 0) {
        if (h == 0) out.push_back(m);
        return;
    }
    enum_rec(pos - 1, h, m, out);
    if (h > 0)
        enum_rec(pos - 1, h - 1, m | (MateID(R) << (2 * pos)), out);
    enum_rec(pos - 1, h + 1, m | (MateID(L) << (2 * pos)), out);
}

static std::vector<MateID> states(int width) {
    std::vector<MateID> out;
    enum_rec(width - 1, 1, 0, out);
    return out;
}

static std::uint32_t occupancy(MateID m, int lo, int len) {
    std::uint32_t z = 0;
    for (int q = 0; q < len; ++q)
        if (((m >> (2 * (lo + q))) & 3u) != 0) z |= 1u << q;
    return z;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 12;
    const int low = argc > 2 ? std::atoi(argv[2]) : (W - 1) / 2;
    const int high = W - 1 - low;
    if (W < 3 || W > 13 || low < 1 || high < 1) {
        std::cerr << "usage: factor_highmask_closure [3<=W<=13] [LOW]\n";
        return 1;
    }

    const auto main_states = states(W);
    const auto block_states = states(W - 1);
    std::uint64_t main_edges = 0, block_edges = 0, cross_topology_edges = 0;

    // Main layout: LOW positions 0..low-1, center=low, HIGH starts low+1.
    for (MateID m : main_states) {
        const std::uint32_t hm = occupancy(m, low + 1, high);
        for (int p = low; p >= 1; --p) {
            auto z = oneesan::gridfp::include_horizontal(m, W, p);
            if (!z.valid) continue;
            ++main_edges;
            const int z_high_lo = z.blocked ? low : low + 1;
            const std::uint32_t hm2 = occupancy(z.mate, z_high_lo, high);
            if (hm2 != hm) {
                std::cerr << "HIGH occupancy changed in LOW main transition"
                          << " W=" << W << " p=" << p
                          << " blocked=" << z.blocked
                          << " before=" << hm << " after=" << hm2 << '\n';
                return 2;
            }

            // Count topology changes that are nevertheless mask-preserving.
            const MateID a = (m >> (2 * (low + 1))) & ((MateID(1) << (2 * high)) - 1);
            const MateID b = (z.mate >> (2 * z_high_lo)) & ((MateID(1) << (2 * high)) - 1);
            if (a != b) ++cross_topology_edges;
        }
    }

    // Blocked layout has no center, so HIGH starts at position low.  Excluding
    // a blocked state inserts N at p; for p<=low this shifts HIGH right by one
    // but cannot alter HIGH occupancy.
    for (MateID m : block_states) {
        const std::uint32_t hm = occupancy(m, low, high);
        for (int p = low; p >= 1; --p) {
            MateID z = oneesan::gridfp::blocked_exclude(m, p);
            ++block_edges;
            const std::uint32_t hm2 = occupancy(z, low + 1, high);
            if (hm2 != hm) {
                std::cerr << "HIGH occupancy changed in LOW blocked transition"
                          << " W=" << W << " p=" << p
                          << " before=" << hm << " after=" << hm2 << '\n';
                return 3;
            }
        }
    }

    std::cout << "factor-highmask-closure OK"
              << " W=" << W << " low=" << low << " high=" << high
              << " main_states=" << main_states.size()
              << " block_states=" << block_states.size()
              << " main_edges=" << main_edges
              << " block_edges=" << block_edges
              << " high_topology_changes=" << cross_topology_edges
              << '\n';
    return 0;
}
