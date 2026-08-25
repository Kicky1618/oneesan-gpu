#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using namespace oneesan::gridfp;
using Count = std::uint32_t;
static constexpr Count MOD = 4294967291u;

enum AuxKind : std::uint32_t { AUX_INVALID = 0, AUX_NN = 1, AUX_PAIR = 2 };
struct Aux {
    AuxKind kind = AUX_INVALID;
    MateID companion = 0; // needed only for PAIR and p>1
};

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

static bool closure_source(MateValuePair w) {
    return w == LL || w == RR || w == RL;
}

static Aux build_compact_aux(MateID representative, int p) {
    const MateValuePair w = mpair(representative, p);
    if (w == NN) return {AUX_NN, 0};
    if (w != NR && w != NL) return {};
    if (p == 1) return {AUX_PAIR, 0};
    const MateValuePair cw = w == NR ? RN : LN;
    return {AUX_PAIR, msetpair(representative, p, cw)};
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    if (W < 4 || W > 13) return 1;

    const auto ms = enumerate_states(W);
    const auto ds = enumerate_states(W - 1);
    std::unordered_map<MateID, std::size_t> mi, di;
    mi.reserve(ms.size() * 2 + 1);
    di.reserve(ds.size() * 2 + 1);
    for (std::size_t i = 0; i < ms.size(); ++i) mi.emplace(ms[i], i);
    for (std::size_t i = 0; i < ds.size(); ++i) di.emplace(ds[i], i);

    std::uint64_t aux_words = 0, nn = 0, pair = 0, p1_pair = 0;
    for (int p = W - 1; p >= 1; --p) {
        std::vector<Count> in_m(ms.size()), in_d(ds.size());
        for (std::size_t i = 0; i < ms.size(); ++i) in_m[i] = seed_value(ms[i], p, 3);
        for (std::size_t i = 0; i < ds.size(); ++i) in_d[i] = seed_value(ds[i], p, 7);

        std::vector<Count> ref_m = in_m;
        std::vector<Count> ref_d(ds.size(), 0);
        for (std::size_t i = 0; i < ms.size(); ++i) {
            const IncludeResult z = include_horizontal(ms[i], W, p);
            if (!z.valid || !in_m[i]) continue;
            if (z.blocked) ref_d[di.at(z.mate)] = addmod(ref_d[di.at(z.mate)], in_m[i]);
            else ref_m[mi.at(z.mate)] = addmod(ref_m[mi.at(z.mate)], in_m[i]);
        }
        for (std::size_t d = 0; d < ds.size(); ++d) if (in_d[d]) {
            const MateID r = blocked_exclude(ds[d], p);
            ref_m[mi.at(r)] = addmod(ref_m[mi.at(r)], in_d[d]);
        }

        std::vector<Count> got_m = in_m;
        std::vector<Count> got_d = in_d;
        for (std::size_t d = 0; d < ds.size(); ++d) {
            ++aux_words;
            const MateID r = blocked_exclude(ds[d], p); // block_desc target
            const Aux aux = build_compact_aux(r, p);
            if (aux.kind == AUX_INVALID) return 10;
            const std::size_t i = mi.at(r);
            const Count c = got_m[i], old_d = got_d[d];

            if (aux.kind == AUX_NN) {
                ++nn;
                const IncludeResult z = include_horizontal(r, W, p); // main_desc target
                if (!z.valid || z.blocked) return 11;
                const std::size_t j = mi.at(z.mate);
                got_m[j] = addmod(got_m[j], c);
                got_m[i] = addmod(c, old_d);
                got_d[d] = 0;
            } else if (p == 1) {
                ++pair;
                ++p1_pair;
                const IncludeResult z = include_horizontal(r, W, p); // companion in main_desc
                if (!z.valid || z.blocked) return 12;
                const std::size_t j = mi.at(z.mate);
                const Count cc = got_m[j];
                got_m[i] = addmod(addmod(c, cc), old_d);
                got_m[j] = addmod(c, cc);
                got_d[d] = 0;
            } else {
                ++pair;
                const IncludeResult z = include_horizontal(r, W, p);
                if (!z.valid || !z.blocked || z.mate != ds[d]) return 13;
                const std::size_t j = mi.at(aux.companion);
                const Count cc = got_m[j];
                got_m[i] = addmod(addmod(c, cc), old_d);
                got_d[d] = c;
            }
        }

        // unchanged closure pass
        for (std::size_t i = 0; i < ms.size(); ++i) {
            if (!closure_source(mpair(ms[i], p))) continue;
            const Count c = got_m[i];
            if (!c) continue;
            const IncludeResult z = include_horizontal(ms[i], W, p);
            if (!z.valid) continue;
            if (z.blocked) got_d[di.at(z.mate)] = addmod(got_d[di.at(z.mate)], c);
            else got_m[mi.at(z.mate)] = addmod(got_m[mi.at(z.mate)], c);
        }

        if (got_m != ref_m || got_d != ref_d) {
            std::cerr << "compact block aux mismatch W=" << W << " p=" << p << '\n';
            return 14;
        }
    }

    std::cout << "factor-blockorbit-compactaux-semantics OK W=" << W
              << " main_states=" << ms.size() << " block_states=" << ds.size()
              << " aux_words=" << aux_words << " nn=" << nn << " pair=" << pair
              << " p1_pair=" << p1_pair << '\n';
    return 0;
}
