#include "../../common/gridfp_transition.hpp"

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

using oneesan::gridfp::MateID;
using oneesan::gridfp::N;
using oneesan::gridfp::R;
using oneesan::gridfp::L;
using oneesan::gridfp::mget;
using oneesan::gridfp::mpair;
using oneesan::gridfp::include_horizontal;

using U64 = std::uint64_t;

static std::vector<std::vector<std::uint32_t>> enumerate_high(int len) {
    std::vector<std::vector<std::uint32_t>> out(size_t(len + 2));
    auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
        if (pos < 0) {
            if (h < int(out.size())) out[size_t(h)].push_back(code);
            return;
        }
        self(self, pos - 1, h, code);
        if (h > 0)
            self(self, pos - 1, h - 1,
                 code | (std::uint32_t(R) << (2 * pos)));
        self(self, pos - 1, h + 1,
             code | (std::uint32_t(L) << (2 * pos)));
    };
    rec(rec, len - 1, 1, 0);
    return out;
}

static std::vector<std::uint32_t> enumerate_low_from(int len, int start_h) {
    std::vector<std::uint32_t> out;
    auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
        if (pos < 0) {
            if (h == 0) out.push_back(code);
            return;
        }
        if (h > pos + 1) return;
        self(self, pos - 1, h, code);
        if (h > 0)
            self(self, pos - 1, h - 1,
                 code | (std::uint32_t(R) << (2 * pos)));
        self(self, pos - 1, h + 1,
             code | (std::uint32_t(L) << (2 * pos)));
    };
    rec(rec, len - 1, start_h, 0);
    return out;
}

static std::vector<std::vector<std::uint32_t>> enumerate_low(int len) {
    std::vector<std::vector<std::uint32_t>> out(size_t(len + 2));
    for (int h = 0; h <= len + 1; ++h)
        out[size_t(h)] = enumerate_low_from(len, h);
    return out;
}

struct Work {
    U64 main_states = 0;
    U64 blocked_states = 0;
    U64 baseline_state_steps = 0;
    U64 low_orbit_cells = 0;
    U64 low_closure_cells = 0;
    U64 high_nn_cells = 0;
    U64 high_nrnl_cells = 0;
    U64 high_block_closure_cells = 0;
    U64 high_cross_closure_cells = 0;

    U64 direct_cells() const {
        return low_orbit_cells + low_closure_cells
            + high_nn_cells + high_nrnl_cells
            + high_block_closure_cells + high_cross_closure_cells;
    }
};

static Work model(int high_len, int low_len) {
    const int W = high_len + low_len + 1;
    auto high = enumerate_high(high_len);
    auto low = enumerate_low(low_len);
    const MateID low_mask = (MateID(1) << (2 * low_len)) - 1;

    Work out;
    for (int he = 0; he <= high_len + 1; ++he) {
        for (int cv = 0; cv < 3; ++cv) {
            int hs = he + (cv == int(L) ? 1 : cv == int(R) ? -1 : 0);
            if (hs < 0 || hs > low_len + 1) continue;
            U64 rows = high[size_t(he)].size();
            U64 cols = low[size_t(hs)].size();
            out.main_states += rows * cols;
            if (!rows || !cols) continue;

            MateID high_part = MateID(high[size_t(he)][0]) << (2 * (low_len + 1));
            MateID center_part = MateID(cv) << (2 * low_len);

            // LOW direct: metadata is indexed by LOW all-rank and expanded over
            // all HIGH rows in the factor block.
            for (int p = low_len; p >= 1; --p) {
                for (std::uint32_t lc : low[size_t(hs)]) {
                    MateID m = MateID(lc) | center_part | high_part;
                    auto pair = mpair(m, p);
                    if (pair == oneesan::gridfp::NN
                        || pair == oneesan::gridfp::NR
                        || pair == oneesan::gridfp::NL) {
                        out.low_orbit_cells += rows;
                    } else if (pair == oneesan::gridfp::LL
                               || pair == oneesan::gridfp::RR
                               || pair == oneesan::gridfp::RL) {
                        if (include_horizontal(m, W, p).valid)
                            out.low_closure_cells += rows;
                    }
                }
            }

            // HIGH direct: metadata is indexed by HIGH all-rank and expanded
            // over all LOW columns in the factor block.
            MateID low_part = MateID(low[size_t(hs)][0]);
            for (int p = W - 1; p >= low_len + 1; --p) {
                for (std::uint32_t hc : high[size_t(he)]) {
                    MateID m = low_part | center_part
                        | (MateID(hc) << (2 * (low_len + 1)));
                    auto pair = mpair(m, p);
                    if (pair == oneesan::gridfp::NN) {
                        out.high_nn_cells += cols;
                    } else if (pair == oneesan::gridfp::NR
                               || pair == oneesan::gridfp::NL) {
                        out.high_nrnl_cells += cols;
                    } else if (pair == oneesan::gridfp::LL
                               || pair == oneesan::gridfp::RR
                               || pair == oneesan::gridfp::RL) {
                        auto z = include_horizontal(m, W, p);
                        if (!z.valid) continue;
                        if ((z.mate & low_mask) != low_part)
                            out.high_cross_closure_cells += cols;
                        else
                            out.high_block_closure_cells += cols;
                    }
                }
            }
        }
    }

    for (int h = 0; h <= high_len + 1 && h < int(low.size()); ++h)
        out.blocked_states += U64(high[size_t(h)].size()) * low[size_t(h)].size();

    out.baseline_state_steps = U64(high_len + low_len)
        * (out.main_states + out.blocked_states);
    return out;
}

int main(int argc, char** argv) {
    int high_len = argc > 1 ? std::atoi(argv[1]) : 13;
    int low_len = argc > 2 ? std::atoi(argv[2]) : 14;
    if (high_len <= 0 || low_len <= 0 || high_len > 15 || low_len > 15) {
        std::cerr << "usage: gpu_direct_work_model [HIGH_LEN LOW_LEN]\n";
        return 2;
    }

    Work w = model(high_len, low_len);
    double fraction = w.baseline_state_steps
        ? double(w.direct_cells()) / double(w.baseline_state_steps) : 0.0;
    std::cout << std::setprecision(12)
              << "high=" << high_len
              << " low=" << low_len
              << " n=" << (high_len + low_len)
              << " main_states=" << w.main_states
              << " blocked_states=" << w.blocked_states
              << " baseline_state_steps=" << w.baseline_state_steps
              << " low_orbit_cells=" << w.low_orbit_cells
              << " low_closure_cells=" << w.low_closure_cells
              << " high_nn_cells=" << w.high_nn_cells
              << " high_nrnl_cells=" << w.high_nrnl_cells
              << " high_block_closure_cells=" << w.high_block_closure_cells
              << " high_cross_closure_cells=" << w.high_cross_closure_cells
              << " direct_cells=" << w.direct_cells()
              << " direct_iteration_fraction=" << fraction
              << " direct_scan_reduction=" << (1.0 - fraction)
              << '\n';
    return 0;
}
