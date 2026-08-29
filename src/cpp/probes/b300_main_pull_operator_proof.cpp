#include "../../common/gridfp_transition.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <vector>

using namespace oneesan::gridfp;
using Count = std::uint32_t;
using Wide = std::uint64_t;
static constexpr Count MOD = 4294967291u;

namespace {

std::vector<MateID> gen_valid(int W) {
    std::vector<MateID> out;
    auto rec = [&](auto&& self, int pos, int h, MateID m) -> void {
        const int rem = W - pos;
        if (h < 0 || h > rem) return;
        if (pos == W) { if (h == 0) out.push_back(m); return; }
        const int bit = W - 1 - pos;
        self(self, pos + 1, h, m);
        if (h > 0) self(self, pos + 1, h - 1, m | (MateID(R) << (2 * bit)));
        self(self, pos + 1, h + 1, m | (MateID(L) << (2 * bit)));
    };
    rec(rec, 0, 1, 0);
    std::sort(out.begin(), out.end());
    return out;
}

Count add_mod(Count a, Count b) {
    Wide z = Wide(a) + b;
    if (z >= MOD) z -= MOD;
    return Count(z);
}

Count add3_mod(Count a, Count b, Count c) {
    Wide z = Wide(a) + b + c;
    if (z >= MOD) z -= MOD;
    if (z >= MOD) z -= MOD;
    return Count(z);
}

void add(std::map<MateID, Count>& v, MateID k, Count x) {
    auto it = v.find(k);
    if (it == v.end()) v.emplace(k, x);
    else it->second = add_mod(it->second, x);
}

std::map<MateID, Count> push_main(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    const std::vector<Count>& mv,
    const std::vector<Count>& bv,
    int W, int p
) {
    std::map<MateID, Count> out;
    for (std::size_t i = 0; i < main.size(); ++i) {
        const MateID m = main[i];
        add(out, m, mv[i]); // identity
        const IncludeResult z = include_horizontal(m, W, p);
        if (z.valid && !z.blocked) add(out, z.mate, mv[i]);
    }
    for (std::size_t i = 0; i < block.size(); ++i)
        add(out, minsert(block[i], p, N), bv[i]);
    return out;
}

std::map<MateID, Count> pull_main(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    const std::vector<Count>& mv,
    const std::vector<Count>& bv,
    int W, int p
) {
    std::map<MateID, std::size_t> mi, bi;
    for (std::size_t i = 0; i < main.size(); ++i) mi.emplace(main[i], i);
    for (std::size_t i = 0; i < block.size(); ++i) bi.emplace(block[i], i);
    std::map<MateID, Count> out;
    for (std::size_t i = 0; i < main.size(); ++i) {
        const MateID d = main[i];
        Count a = mv[i], b = 0, c = 0;
        MateID x = 0;
        switch (mpair(d, p)) {
        case LR: x = msetpair(d, p, NN); break;
        case NR: x = msetpair(d, p, RN); break;
        case NL: x = msetpair(d, p, LN); break;
        default: break;
        }
        if (x) {
            auto it = mi.find(x);
            if (it == mi.end()) std::exit(10);
            b = mv[it->second];
        }
        if (mget(d, p) == N) {
            const MateID q = mshrink(d, p);
            auto it = bi.find(q);
            if (it == bi.end()) std::exit(11);
            c = bv[it->second];
        }
        out[d] = add3_mod(a, b, c);
    }
    return out;
}

} // namespace

int main() {
    std::mt19937_64 rng(0x6233303070756c6cULL);
    std::uint64_t positions = 0, main_states = 0, blocked_states = 0;
    for (int W = 4; W <= 11; ++W) {
        const auto main = gen_valid(W);
        const auto block = gen_valid(W - 1);
        std::vector<Count> mv(main.size()), bv(block.size());
        for (auto& x : mv) x = Count(rng() % MOD);
        for (auto& x : bv) x = Count(rng() % MOD);
        for (int p = 2; p < W; ++p) {
            const auto push = push_main(main, block, mv, bv, W, p);
            const auto pull = pull_main(main, block, mv, bv, W, p);
            if (push != pull) {
                std::cerr << "mismatch W=" << W << " p=" << p << '\n';
                return 2;
            }
            ++positions;
            main_states += main.size();
            blocked_states += block.size();
        }
    }
    std::cout << "b300-main-pull-operator-proof OK"
              << " exhaustive_width_max=11 positions=" << positions
              << " main_destination_cases=" << main_states
              << " blocked_source_cases=" << blocked_states
              << " p_scope=2..Wm1"
              << " identity_copy_required=0"
              << " blocked_to_main_scatter_required=0"
              << " main_atomic_updates_required=0"
              << " pull_terms_max=3 exact=1\n";
    return 0;
}
