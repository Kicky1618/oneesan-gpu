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

static Count seed_value(MateID m, int p, int salt) {
    std::uint64_t x = m ^ (std::uint64_t(p) << 48)
        ^ std::uint64_t(0x9e3779b9u * (salt + 1));
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    if ((x & 15u) == 0) return 0;
    return Count(x % MOD);
}

static bool orbit_rep(MateValuePair w) {
    return w == NN || w == NR || w == NL;
}
static bool closure_source(MateValuePair w) {
    return w == LL || w == RR || w == RL;
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    if (W < 4 || W > 13) {
        std::cerr << "usage: factor_blockorbit_semantics [W<=13]\n";
        return 1;
    }

    const auto main_states = enumerate_states(W);
    const auto block_states = enumerate_states(W - 1);
    std::unordered_map<MateID, std::size_t> mi, bi;
    mi.reserve(main_states.size() * 2 + 1);
    bi.reserve(block_states.size() * 2 + 1);
    for (std::size_t i = 0; i < main_states.size(); ++i) mi.emplace(main_states[i], i);
    for (std::size_t i = 0; i < block_states.size(); ++i) bi.emplace(block_states[i], i);

    std::uint64_t blocked_steps = 0, nn = 0, pair = 0, p1_pair = 0, closure = 0;
    for (int p = W - 1; p >= 1; --p) {
        std::vector<Count> in_main(main_states.size()), in_block(block_states.size());
        for (std::size_t i = 0; i < main_states.size(); ++i)
            in_main[i] = seed_value(main_states[i], p, 17);
        for (std::size_t i = 0; i < block_states.size(); ++i)
            in_block[i] = seed_value(block_states[i], p, 29);

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
        for (std::size_t d = 0; d < block_states.size(); ++d) {
            const Count c = in_block[d];
            if (!c) continue;
            const MateID z = blocked_exclude(block_states[d], p);
            const auto it = mi.find(z);
            if (it == mi.end()) return 12;
            ref_main[it->second] = addmod(ref_main[it->second], c);
        }

        std::vector<Count> got_main = in_main;
        std::vector<Count> got_block = in_block;
        std::vector<std::uint8_t> seen_main(main_states.size());

        // Blocked-domain orbit pass. Inserting N at p maps each blocked state
        // bijectively to one NN/NR/NL representative main state.
        for (std::size_t d = 0; d < block_states.size(); ++d) {
            ++blocked_steps;
            const MateID source = blocked_exclude(block_states[d], p);
            const auto it = mi.find(source);
            if (it == mi.end()) return 20;
            const std::size_t i = it->second;
            if (++seen_main[i] != 1) {
                std::cerr << "blocked->representative is not injective p=" << p << '\n';
                return 21;
            }
            const MateValuePair w = mpair(source, p);
            if (!orbit_rep(w)) {
                std::cerr << "blocked exclusion did not create orbit representative p=" << p << '\n';
                return 22;
            }

            MateValuePair cw = LR;
            if (w == NR) cw = RN;
            else if (w == NL) cw = LN;
            const MateID companion = msetpair(source, p, cw);
            const auto jt = mi.find(companion);
            if (jt == mi.end()) return 23;
            const std::size_t j = jt->second;

            const Count c = got_main[i];
            const Count old_d = got_block[d];
            if (w == NN) {
                ++nn;
                got_main[j] = addmod(got_main[j], c);
                got_main[i] = addmod(c, old_d);
                got_block[d] = 0;
            } else {
                ++pair;
                const Count cc = got_main[j];
                got_main[i] = addmod(addmod(c, cc), old_d);
                if (p == 1) {
                    ++p1_pair;
                    got_main[j] = addmod(c, cc);
                    got_block[d] = 0;
                } else {
                    got_block[d] = c;
                }
            }
        }

        std::size_t reps = 0;
        for (std::size_t i = 0; i < main_states.size(); ++i)
            if (orbit_rep(mpair(main_states[i], p))) {
                ++reps;
                if (seen_main[i] != 1) return 24;
            } else if (seen_main[i]) return 25;
        if (reps != block_states.size()) return 26;

        // Same closure pass used by v0.4/v0.5/v0.6 after orbit updates.
        for (std::size_t i = 0; i < main_states.size(); ++i) {
            const MateValuePair w = mpair(main_states[i], p);
            if (!closure_source(w)) continue;
            ++closure;
            const Count c = got_main[i];
            if (!c) continue;
            const IncludeResult z = include_horizontal(main_states[i], W, p);
            if (!z.valid) continue;
            if (z.blocked) {
                const auto it = bi.find(z.mate);
                if (it == bi.end()) return 27;
                got_block[it->second] = addmod(got_block[it->second], c);
            } else {
                const auto it = mi.find(z.mate);
                if (it == mi.end()) return 28;
                got_main[it->second] = addmod(got_main[it->second], c);
            }
        }

        if (got_main != ref_main || got_block != ref_block) {
            std::size_t bm = 0, bb = 0;
            while (bm < got_main.size() && got_main[bm] == ref_main[bm]) ++bm;
            while (bb < got_block.size() && got_block[bb] == ref_block[bb]) ++bb;
            std::cerr << "blockorbit semantic mismatch W=" << W << " p=" << p
                      << " bad_main=" << bm << '/' << got_main.size()
                      << " bad_block=" << bb << '/' << got_block.size() << '\n';
            return 29;
        }
    }

    std::cout << "factor-blockorbit-semantics OK W=" << W
              << " main_states=" << main_states.size()
              << " block_states=" << block_states.size()
              << " blocked_steps=" << blocked_steps
              << " nn=" << nn << " pair=" << pair
              << " p1_pair=" << p1_pair << " closure=" << closure << '\n';
    return 0;
}
