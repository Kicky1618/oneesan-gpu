#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using namespace oneesan::gridfp;

static std::vector<MateID> enumerate_states(int width) {
    std::vector<MateID> out;
    auto rec = [&](auto&& self, int pos, int h, MateID m) -> void {
        if (pos < 0) {
            if (h == 0) out.push_back(m);
            return;
        }
        if (h < 0 || h > pos + 1) return;
        self(self, pos - 1, h, m);
        if (h > 0) self(self, pos - 1, h - 1, m | (MateID(R) << (2 * pos)));
        self(self, pos - 1, h + 1, m | (MateID(L) << (2 * pos)));
    };
    rec(rec, width - 1, 1, 0);
    return out;
}

static std::uint32_t occupancy(MateID m, int begin, int len) {
    std::uint32_t z = 0;
    for (int i = 0; i < len; ++i)
        if (mget(m, begin + i) != N) z |= 1u << i;
    return z;
}

static bool is_orbit_source(MateValuePair w) {
    return w == NN || w == NR || w == NL;
}

static bool is_closure_source(MateValuePair w) {
    return w == LL || w == RR || w == RL;
}

static int verify_group(
    const std::vector<MateID>& main_states,
    const std::vector<MateID>& block_states,
    int W,
    int p,
    const char* mode,
    std::uint32_t mask,
    std::uint64_t& orbit_reps,
    std::uint64_t& closure_reps
) {
    std::unordered_map<MateID, std::size_t> mi, bi;
    mi.reserve(main_states.size() * 2 + 1);
    bi.reserve(block_states.size() * 2 + 1);
    for (std::size_t i = 0; i < main_states.size(); ++i) mi.emplace(main_states[i], i);
    for (std::size_t i = 0; i < block_states.size(); ++i) bi.emplace(block_states[i], i);

    std::vector<std::uint8_t> main_touch(main_states.size());
    std::vector<std::uint8_t> block_touch(block_states.size());

    for (std::size_t i = 0; i < main_states.size(); ++i) {
        const MateID m = main_states[i];
        const MateValuePair w = mpair(m, p);
        if (!is_orbit_source(w)) continue;
        ++orbit_reps;

        MateValuePair cw = LR;
        if (w == NR) cw = RN;
        else if (w == NL) cw = LN;
        const MateID companion = msetpair(m, p, cw);
        const MateID dropped = mshrink(m, p);
        const auto jt = mi.find(companion);
        const auto dt = bi.find(dropped);
        if (jt == mi.end() || dt == bi.end()) {
            std::cerr << "racefree missing orbit target mode=" << mode
                      << " mask=" << mask << " p=" << p << '\n';
            return 10;
        }
        const std::size_t j = jt->second;
        const std::size_t d = dt->second;
        if (is_orbit_source(mpair(companion, p))) {
            std::cerr << "racefree companion is another representative mode=" << mode
                      << " mask=" << mask << " p=" << p << '\n';
            return 11;
        }
        if (++main_touch[i] != 1 || ++main_touch[j] != 1 || ++block_touch[d] != 1) {
            std::cerr << "racefree overlapping non-atomic footprint mode=" << mode
                      << " mask=" << mask << " p=" << p
                      << " i=" << i << " j=" << j << " d=" << d << '\n';
            return 12;
        }
    }

    // Closure uses atomics for destination collisions, but a destination must
    // not be another closure source read later by the same kernel: otherwise
    // the in-place pass could cascade newly-added mass within one DP step.
    for (MateID m : main_states) {
        const MateValuePair w = mpair(m, p);
        if (!is_closure_source(w)) continue;
        ++closure_reps;
        const IncludeResult z = include_horizontal(m, W, p);
        if (!z.valid || z.blocked) continue;
        const auto it = mi.find(z.mate);
        if (it == mi.end()) {
            std::cerr << "racefree closure main target left group mode=" << mode
                      << " mask=" << mask << " p=" << p << '\n';
            return 13;
        }
        if (is_closure_source(mpair(z.mate, p))) {
            std::cerr << "racefree closure target is closure source mode=" << mode
                      << " mask=" << mask << " p=" << p << '\n';
            return 14;
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int low = argc > 2 ? std::atoi(argv[2]) : 5;
    const int high = W - 1 - low;
    if (W < 4 || W > 13 || low < 1 || high < 1) {
        std::cerr << "usage: factor_orbit_racefree [W<=13] [LOW]\n";
        return 1;
    }

    const auto all_main = enumerate_states(W);
    const auto all_block = enumerate_states(W - 1);
    std::uint64_t groups = 0, orbit_reps = 0, closure_reps = 0;

    // HIGH-active window: LOW occupancy mask is fixed.
    for (std::uint32_t mask = 0; mask < (1u << low); ++mask) {
        std::vector<MateID> ms, bs;
        for (MateID m : all_main) if (occupancy(m, 0, low) == mask) ms.push_back(m);
        for (MateID m : all_block) if (occupancy(m, 0, low) == mask) bs.push_back(m);
        if (ms.empty() && bs.empty()) continue;
        for (int p = W - 1; p >= low + 1; --p) {
            ++groups;
            const int rc = verify_group(ms, bs, W, p, "HIGH", mask, orbit_reps, closure_reps);
            if (rc) return rc;
        }
    }

    // LOW-active window: HIGH occupancy mask is fixed. Main HIGH starts after
    // center; blocked HIGH starts one position earlier after deleting p.
    for (std::uint32_t mask = 0; mask < (1u << high); ++mask) {
        std::vector<MateID> ms, bs;
        for (MateID m : all_main)
            if (occupancy(m, low + 1, high) == mask) ms.push_back(m);
        for (MateID m : all_block)
            if (occupancy(m, low, high) == mask) bs.push_back(m);
        if (ms.empty() && bs.empty()) continue;
        for (int p = low; p >= 1; --p) {
            ++groups;
            const int rc = verify_group(ms, bs, W, p, "LOW", mask, orbit_reps, closure_reps);
            if (rc) return rc;
        }
    }

    std::cout << "factor-orbit-racefree OK W=" << W
              << " low=" << low << " high=" << high
              << " groups=" << groups
              << " orbit_reps=" << orbit_reps
              << " closure_reps=" << closure_reps << '\n';
    return 0;
}
