#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using namespace oneesan::gridfp;
using Count = std::uint32_t;
static constexpr Count MOD = 4294967291u;

static Count addmod(Count a, Count b) {
    const std::uint64_t z = std::uint64_t(a) + b;
    return Count(z >= MOD ? z - MOD : z);
}

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

static Count seed_value(MateID m, std::uint32_t mask, int salt) {
    std::uint64_t x = m ^ (std::uint64_t(mask) << 37)
        ^ std::uint64_t(0x9e3779b9u * (salt + 1));
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    if ((x & 15u) == 0) return 0;
    return Count(x % MOD);
}

static std::uint32_t compact_low_center(MateID m, int low) {
    const MateID mask = (MateID(1) << (2 * (low + 1))) - 1;
    return std::uint32_t(m & mask);
}

static std::uint32_t compact_set_pair(
    std::uint32_t active, int p, MateValuePair v
) {
    const int shift = 2 * (p - 1);
    const std::uint32_t z = 15u << shift;
    return (active & ~z) | (std::uint32_t(v) << shift);
}

static std::uint32_t compact_drop_n(std::uint32_t active, int p) {
    const int shift = 2 * p;
    const std::uint32_t lowmask = (std::uint32_t(1) << shift) - 1u;
    return (active & lowmask) | ((active & ~lowmask) >> 2);
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int low = argc > 2 ? std::atoi(argv[2]) : 5;
    const int high = W - 1 - low;
    if (W < 4 || W > 13 || low < 1 || high < 1 || low >= 15) {
        std::cerr << "usage: factor_loworbit_semantics [W<=13] [LOW]\n";
        return 1;
    }

    const auto all_main = enumerate_states(W);
    const auto all_block = enumerate_states(W - 1);
    const MateID low_bits = (MateID(1) << (2 * low)) - 1;
    const MateID active_bits = (MateID(1) << (2 * (low + 1))) - 1;
    std::uint64_t groups = 0, state_steps = 0, orbit_sources = 0, closure_sources = 0;

    for (std::uint32_t high_mask = 0; high_mask < (1u << high); ++high_mask) {
        std::vector<MateID> main_states, block_states;
        for (MateID m : all_main)
            if (occupancy(m, low + 1, high) == high_mask) main_states.push_back(m);
        for (MateID m : all_block)
            if (occupancy(m, low, high) == high_mask) block_states.push_back(m);
        if (main_states.empty() && block_states.empty()) continue;

        std::unordered_map<MateID, std::size_t> mi, bi;
        mi.reserve(main_states.size() * 2 + 1);
        bi.reserve(block_states.size() * 2 + 1);
        for (std::size_t i = 0; i < main_states.size(); ++i) mi.emplace(main_states[i], i);
        for (std::size_t i = 0; i < block_states.size(); ++i) bi.emplace(block_states[i], i);

        for (int p = low; p >= 1; --p) {
            ++groups;
            std::vector<Count> in_main(main_states.size()), in_block(block_states.size());
            for (std::size_t i = 0; i < main_states.size(); ++i)
                in_main[i] = seed_value(main_states[i], high_mask, p + 200);
            for (std::size_t i = 0; i < block_states.size(); ++i)
                in_block[i] = seed_value(block_states[i], high_mask, p + 301);

            std::vector<Count> ref_main = in_main;
            std::vector<Count> ref_block(block_states.size(), 0);
            for (std::size_t i = 0; i < main_states.size(); ++i) {
                const Count c = in_main[i];
                if (!c) continue;
                const IncludeResult z = include_horizontal(main_states[i], W, p);
                if (!z.valid) continue;
                if (z.blocked) {
                    const auto it = bi.find(z.mate);
                    if (it == bi.end()) return 10;
                    ref_block[it->second] = addmod(ref_block[it->second], c);
                } else {
                    const auto it = mi.find(z.mate);
                    if (it == mi.end()) return 11;
                    ref_main[it->second] = addmod(ref_main[it->second], c);
                }
            }
            for (std::size_t i = 0; i < block_states.size(); ++i) {
                const Count c = in_block[i];
                if (!c) continue;
                const MateID z = blocked_exclude(block_states[i], p);
                const auto it = mi.find(z);
                if (it == mi.end()) return 12;
                ref_main[it->second] = addmod(ref_main[it->second], c);
            }

            std::vector<Count> got_main = in_main;
            std::vector<Count> got_block = in_block;
            for (std::size_t i = 0; i < main_states.size(); ++i) {
                const MateID m = main_states[i];
                const MateValuePair w = mpair(m, p);
                if (w != NN && w != NR && w != NL) continue;
                ++orbit_sources;

                const std::uint32_t active = compact_low_center(m, low);
                MateValuePair cw = LR;
                if (w == NR) cw = RN;
                else if (w == NL) cw = LN;
                const MateID companion = (m & ~active_bits)
                    | MateID(compact_set_pair(active, p, cw));
                if (companion != msetpair(m, p, cw)) {
                    std::cerr << "LOW compact companion mismatch\n";
                    return 13;
                }

                const std::uint32_t dropped_low = compact_drop_n(active, p);
                const MateID high_code = m >> (2 * (low + 1));
                const MateID dropped = MateID(dropped_low & std::uint32_t(low_bits))
                    | (high_code << (2 * low));
                if (dropped != mshrink(m, p)) {
                    std::cerr << "LOW compact drop-N mismatch\n";
                    return 14;
                }

                const auto jt = mi.find(companion);
                const auto dt = bi.find(dropped);
                if (jt == mi.end() || dt == bi.end()) return 15;
                const std::size_t j = jt->second, d = dt->second;
                const Count c = got_main[i];
                const Count old_d = got_block[d];
                if (w == NN) {
                    got_main[j] = addmod(got_main[j], c);
                    got_main[i] = addmod(c, old_d);
                    got_block[d] = 0;
                } else {
                    const Count cc = got_main[j];
                    const Count all = addmod(addmod(c, cc), old_d);
                    got_main[i] = all;
                    if (p == 1) {
                        got_main[j] = addmod(c, cc);
                        got_block[d] = 0;
                    } else {
                        got_block[d] = c;
                    }
                }
            }

            for (std::size_t i = 0; i < main_states.size(); ++i) {
                const MateID m = main_states[i];
                const MateValuePair w = mpair(m, p);
                if (w != LL && w != RR && w != RL) continue;
                ++closure_sources;
                const Count c = got_main[i];
                if (!c) continue;
                const IncludeResult z = include_horizontal(m, W, p);
                if (!z.valid) continue;
                if (z.blocked) {
                    const auto it = bi.find(z.mate);
                    if (it == bi.end()) return 16;
                    got_block[it->second] = addmod(got_block[it->second], c);
                } else {
                    const auto it = mi.find(z.mate);
                    if (it == mi.end()) return 17;
                    got_main[it->second] = addmod(got_main[it->second], c);
                }
            }

            if (got_main != ref_main || got_block != ref_block) {
                std::size_t bm = 0, bb = 0;
                while (bm < got_main.size() && got_main[bm] == ref_main[bm]) ++bm;
                while (bb < got_block.size() && got_block[bb] == ref_block[bb]) ++bb;
                std::cerr << "LOW orbit semantic mismatch W=" << W << " low=" << low
                          << " mask=" << high_mask << " p=" << p
                          << " bad_main=" << bm << '/' << got_main.size()
                          << " bad_block=" << bb << '/' << got_block.size() << '\n';
                return 18;
            }
            state_steps += main_states.size() + block_states.size();
        }
    }

    std::cout << "factor-loworbit-semantics OK W=" << W
              << " low=" << low << " high=" << high
              << " groups=" << groups
              << " state_steps=" << state_steps
              << " orbit_sources=" << orbit_sources
              << " closure_sources=" << closure_sources << '\n';
    return 0;
}
