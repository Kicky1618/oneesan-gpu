#include <algorithm>
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

static std::uint32_t occ(MateID m, int begin, int len) {
    std::uint32_t z = 0;
    for (int p = 0; p < len; ++p)
        if (mget(m, begin + p) != N) z |= 1u << p;
    return z;
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

static Count seed_value(MateID m, std::uint32_t mask, int salt) {
    std::uint64_t x = m ^ (std::uint64_t(mask) << 37) ^ std::uint64_t(0x9e3779b9u * (salt + 1));
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    // Keep some exact zeros to exercise sparse branches too.
    if ((x & 15u) == 0) return 0;
    return Count(x % MOD);
}

static std::uint32_t compact_active(MateID m, int low, int high) {
    const std::uint64_t mask = (std::uint64_t(1) << (2 * (high + 1))) - 1;
    return std::uint32_t((m >> (2 * low)) & mask);
}

static std::uint32_t compact_set_pair(
    std::uint32_t active, int p, int low, MateValuePair v
) {
    const int q = p - low;
    const int shift = 2 * (q - 1);
    const std::uint32_t z = 15u << shift;
    return (active & ~z) | (std::uint32_t(v) << shift);
}

static std::uint32_t compact_drop_n(std::uint32_t active, int p, int low) {
    const int q = p - low;
    const int shift = 2 * q;
    const std::uint32_t lowmask = (std::uint32_t(1) << shift) - 1u;
    return (active & lowmask) | ((active & ~lowmask) >> 2);
}

static MateID compact_main_to_full(std::uint32_t active, MateID low_part, int low) {
    return low_part | (MateID(active) << (2 * low));
}

static MateID compact_block_to_full(std::uint32_t blocked_high, MateID low_part, int low) {
    return low_part | (MateID(blocked_high) << (2 * low));
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int low = argc > 2 ? std::atoi(argv[2]) : 5;
    const int high = W - 1 - low;
    if (W < 4 || W > 13 || low < 1 || high < 1 || high >= 15) {
        std::cerr << "usage: factor_highorbit_semantics [W<=13] [LOW]\n";
        return 1;
    }

    const auto all_main = enumerate_states(W);
    const auto all_block = enumerate_states(W - 1);
    const MateID low_bits = (MateID(1) << (2 * low)) - 1;
    std::uint64_t tested_groups = 0;
    std::uint64_t tested_steps = 0;
    std::uint64_t orbit_sources = 0;
    std::uint64_t closure_sources = 0;

    for (std::uint32_t low_mask = 0; low_mask < (1u << low); ++low_mask) {
        std::vector<MateID> main_states, block_states;
        for (MateID m : all_main)
            if (occ(m, 0, low) == low_mask) main_states.push_back(m);
        for (MateID m : all_block)
            if (occ(m, 0, low) == low_mask) block_states.push_back(m);
        if (main_states.empty() && block_states.empty()) continue;

        std::unordered_map<MateID, std::size_t> mi, bi;
        mi.reserve(main_states.size() * 2 + 1);
        bi.reserve(block_states.size() * 2 + 1);
        for (std::size_t i = 0; i < main_states.size(); ++i) mi.emplace(main_states[i], i);
        for (std::size_t i = 0; i < block_states.size(); ++i) bi.emplace(block_states[i], i);

        for (int p = W - 1; p >= low + 1; --p) {
            ++tested_groups;
            std::vector<Count> in_main(main_states.size()), in_block(block_states.size());
            for (std::size_t i = 0; i < main_states.size(); ++i)
                in_main[i] = seed_value(main_states[i], low_mask, p);
            for (std::size_t i = 0; i < block_states.size(); ++i)
                in_block[i] = seed_value(block_states[i], low_mask, p + 101);

            std::vector<Count> ref_main = in_main;
            std::vector<Count> ref_block(block_states.size(), 0);
            for (std::size_t i = 0; i < main_states.size(); ++i) {
                const Count c = in_main[i];
                if (!c) continue;
                const IncludeResult z = include_horizontal(main_states[i], W, p);
                if (!z.valid) continue;
                if (z.blocked) {
                    auto it = bi.find(z.mate);
                    if (it == bi.end()) {
                        std::cerr << "reference blocked destination escaped low-mask group\n";
                        return 10;
                    }
                    ref_block[it->second] = addmod(ref_block[it->second], c);
                } else {
                    auto it = mi.find(z.mate);
                    if (it == mi.end()) {
                        std::cerr << "reference main destination escaped low-mask group\n";
                        return 11;
                    }
                    ref_main[it->second] = addmod(ref_main[it->second], c);
                }
            }
            for (std::size_t i = 0; i < block_states.size(); ++i) {
                const Count c = in_block[i];
                if (!c) continue;
                const MateID z = blocked_exclude(block_states[i], p);
                auto it = mi.find(z);
                if (it == mi.end()) {
                    std::cerr << "blocked exclude escaped low-mask group\n";
                    return 12;
                }
                ref_main[it->second] = addmod(ref_main[it->second], c);
            }

            std::vector<Count> got_main = in_main;
            std::vector<Count> got_block = in_block;
            for (std::size_t i = 0; i < main_states.size(); ++i) {
                const MateID m = main_states[i];
                const MateValuePair w = mpair(m, p);
                if (w != NN && w != NR && w != NL) continue;
                ++orbit_sources;

                const MateID low_part = m & low_bits;
                const std::uint32_t active = compact_active(m, low, high);
                MateValuePair cw = LR;
                if (w == NR) cw = RN;
                else if (w == NL) cw = LN;
                const MateID companion = compact_main_to_full(
                    compact_set_pair(active, p, low, cw), low_part, low);
                const MateID direct_companion = msetpair(m, p, cw);
                if (companion != direct_companion) {
                    std::cerr << "compact companion construction mismatch\n";
                    return 13;
                }
                const MateID dropped = compact_block_to_full(
                    compact_drop_n(active, p, low), low_part, low);
                const MateID direct_dropped = mshrink(m, p);
                if (dropped != direct_dropped) {
                    std::cerr << "compact drop-N construction mismatch\n";
                    return 14;
                }

                auto jt = mi.find(companion);
                auto dt = bi.find(dropped);
                if (jt == mi.end() || dt == bi.end()) {
                    std::cerr << "orbit target missing from group\n";
                    return 15;
                }
                const std::size_t j = jt->second;
                const std::size_t d = dt->second;
                const Count c = got_main[i];
                const Count old_d = got_block[d];
                if (w == NN) {
                    got_main[j] = addmod(got_main[j], c);
                    got_main[i] = addmod(c, old_d);
                    got_block[d] = 0;
                } else {
                    const Count cc = got_main[j];
                    got_main[i] = addmod(addmod(c, cc), old_d);
                    got_block[d] = c;
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
                if (!z.blocked) {
                    std::cerr << "HIGH closure unexpectedly stayed main p=" << p << '\n';
                    return 16;
                }
                auto it = bi.find(z.mate);
                if (it == bi.end()) {
                    std::cerr << "closure target escaped group\n";
                    return 17;
                }
                got_block[it->second] = addmod(got_block[it->second], c);
            }

            if (got_main != ref_main || got_block != ref_block) {
                std::size_t bad_m = 0, bad_b = 0;
                while (bad_m < got_main.size() && got_main[bad_m] == ref_main[bad_m]) ++bad_m;
                while (bad_b < got_block.size() && got_block[bad_b] == ref_block[bad_b]) ++bad_b;
                std::cerr << "orbit semantic mismatch W=" << W << " low=" << low
                          << " low_mask=" << low_mask << " p=" << p
                          << " bad_main=" << bad_m << '/' << got_main.size()
                          << " bad_block=" << bad_b << '/' << got_block.size() << '\n';
                return 18;
            }
            tested_steps += main_states.size() + block_states.size();
        }
    }

    std::cout << "factor-highorbit-semantics OK W=" << W
              << " low=" << low << " high=" << high
              << " groups=" << tested_groups
              << " state_steps=" << tested_steps
              << " orbit_sources=" << orbit_sources
              << " closure_sources=" << closure_sources << '\n';
    return 0;
}
