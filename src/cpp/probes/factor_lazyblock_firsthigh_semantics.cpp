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
        if (h > 0) self(self, pos - 1, h - 1,
                        m | (MateID(R) << (2 * pos)));
        self(self, pos - 1, h + 1,
             m | (MateID(L) << (2 * pos)));
    };
    rec(rec, width - 1, 1, 0);
    return out;
}

static Count seed_value(MateID m, int salt) {
    std::uint64_t x = m ^ std::uint64_t(0x9e3779b9u * (salt + 1));
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    if ((x & 31u) == 0) return 0;
    return Count(x % MOD);
}

static bool orbit_rep(MateValuePair w) {
    return w == NN || w == NR || w == NL;
}
static bool closure_source(MateValuePair w) {
    return w == LL || w == RR || w == RL;
}

static void verify_width(int W, int salt) {
    const int p = W - 1;
    const auto main_states = enumerate_states(W);
    const auto block_states = enumerate_states(W - 1);
    std::unordered_map<MateID, std::size_t> mi, bi;
    mi.reserve(main_states.size() * 2 + 1);
    bi.reserve(block_states.size() * 2 + 1);
    for (std::size_t i = 0; i < main_states.size(); ++i) mi.emplace(main_states[i], i);
    for (std::size_t i = 0; i < block_states.size(); ++i) bi.emplace(block_states[i], i);

    std::vector<Count> in_main(main_states.size());
    for (std::size_t i = 0; i < main_states.size(); ++i)
        in_main[i] = seed_value(main_states[i], salt);

    // Canonical first-HIGH-position transition with known row-boundary BLOCKED=0.
    std::vector<Count> ref_main = in_main;
    std::vector<Count> ref_block(block_states.size(), 0);
    for (std::size_t i = 0; i < main_states.size(); ++i) {
        const Count c = in_main[i];
        if (!c) continue;
        const IncludeResult z = include_horizontal(main_states[i], W, p);
        if (!z.valid) continue;
        if (z.blocked) {
            const auto it = bi.find(z.mate);
            if (it == bi.end()) std::exit(10);
            ref_block[it->second] = addmod(ref_block[it->second], c);
        } else {
            const auto it = mi.find(z.mate);
            if (it == mi.end()) std::exit(11);
            ref_main[it->second] = addmod(ref_main[it->second], c);
        }
    }
    // No BLOCKED input contribution: row-boundary BLOCKED is exactly zero.

    // Lazy blocked-domain orbit. Scratch starts poisoned to ensure the first
    // HIGH orbit does not accidentally depend on old contents.
    std::vector<Count> got_main = in_main;
    std::vector<Count> got_block(block_states.size());
    for (std::size_t d = 0; d < block_states.size(); ++d)
        got_block[d] = seed_value(block_states[d], salt + 1000) | 1u;
    std::vector<std::uint8_t> written(block_states.size(), 0);

    for (std::size_t d = 0; d < block_states.size(); ++d) {
        const MateID source = blocked_exclude(block_states[d], p);
        const auto it = mi.find(source);
        if (it == mi.end()) std::exit(20);
        const std::size_t i = it->second;
        const MateValuePair w = mpair(source, p);
        if (!orbit_rep(w)) {
            std::cerr << "first-HIGH blocked exclusion is not orbit rep W=" << W << '\n';
            std::exit(21);
        }

        MateValuePair cw = LR;
        if (w == NR) cw = RN;
        else if (w == NL) cw = LN;
        const MateID companion = msetpair(source, p, cw);
        const auto jt = mi.find(companion);
        if (jt == mi.end()) std::exit(22);
        const std::size_t j = jt->second;

        const Count c = got_main[i];
        const Count d0 = 0; // v0.13 first-HIGH specialization
        if (w == NN) {
            got_main[j] = addmod(got_main[j], c);
            got_main[i] = addmod(c, d0);
            got_block[d] = 0;
        } else {
            const Count cc = got_main[j];
            got_main[i] = addmod(addmod(c, cc), d0);
            got_block[d] = c;
        }
        if (++written[d] != 1) std::exit(23);
    }

    for (std::uint8_t x : written) if (x != 1) {
        std::cerr << "first-HIGH orbit did not overwrite every BLOCKED coordinate W="
                  << W << '\n';
        std::exit(24);
    }

    // Same closure pass used after the blocked-domain orbit.
    for (std::size_t i = 0; i < main_states.size(); ++i) {
        if (!closure_source(mpair(main_states[i], p))) continue;
        const Count c = got_main[i];
        if (!c) continue;
        const IncludeResult z = include_horizontal(main_states[i], W, p);
        if (!z.valid) continue;
        if (z.blocked) {
            const auto it = bi.find(z.mate);
            if (it == bi.end()) std::exit(25);
            got_block[it->second] = addmod(got_block[it->second], c);
        } else {
            const auto it = mi.find(z.mate);
            if (it == mi.end()) std::exit(26);
            got_main[it->second] = addmod(got_main[it->second], c);
        }
    }

    if (got_main != ref_main || got_block != ref_block) {
        std::cerr << "lazy first-HIGH semantic mismatch W=" << W
                  << " salt=" << salt << '\n';
        std::exit(27);
    }

    std::cout << "lazy-firsthigh OK W=" << W
              << " salt=" << salt
              << " main=" << main_states.size()
              << " blocked=" << block_states.size() << '\n';
}

int main(int argc, char** argv) {
    const int max_w = argc > 1 ? std::atoi(argv[1]) : 12;
    if (max_w < 4 || max_w > 13) return 1;
    for (int W = 4; W <= max_w; ++W)
        for (int salt = 0; salt < 3; ++salt)
            verify_width(W, salt);
    return 0;
}
