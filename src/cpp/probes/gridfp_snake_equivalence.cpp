#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../common/gridfp_transition_reverse.hpp"

using namespace oneesan::gridfp;
using Count = std::uint32_t;
using Vec = std::unordered_map<MateID, Count>;

static constexpr Count MOD = 4294967291u;

static void add(Vec& v, MateID k, Count x) {
    if (!x) return;
    auto it = v.find(k);
    if (it == v.end()) {
        v.emplace(k, x);
        return;
    }
    std::uint64_t s = std::uint64_t(it->second) + x;
    Count z = Count(s % MOD);
    if (z) it->second = z;
    else v.erase(it);
}

static std::pair<Vec, Vec> step_forward(const Vec& main, const Vec& block, int W, int p) {
    Vec nm, nb;
    nm.reserve(main.size() * 2 + block.size());
    nb.reserve(main.size() / 2 + 1);
    for (auto [m, c] : main) {
        add(nm, m, c); // excluded main branch
        IncludeResult z = include_horizontal(m, W, p);
        if (!z.valid) continue;
        add(z.blocked ? nb : nm, z.mate, c);
    }
    for (auto [b, c] : block) add(nm, blocked_exclude(b, p), c);
    return {std::move(nm), std::move(nb)};
}

static std::pair<Vec, Vec> step_reverse(const Vec& main, const Vec& block, int W, int p) {
    Vec nm, nb;
    nm.reserve(main.size() * 2 + block.size());
    nb.reserve(main.size() / 2 + 1);
    for (auto [m, c] : main) {
        add(nm, m, c); // excluded main branch
        IncludeResult z = include_horizontal_reverse(m, W, p);
        if (!z.valid) continue;
        add(z.blocked ? nb : nm, z.mate, c);
    }
    for (auto [b, c] : block) add(nm, blocked_exclude_reverse(b, W, p), c);
    return {std::move(nm), std::move(nb)};
}

static void row_forward(Vec& main, Vec& block, int W) {
    for (int p = W - 1; p >= 1; --p) {
        auto z = step_forward(main, block, W, p);
        main = std::move(z.first); block = std::move(z.second);
    }
}

static void row_reverse(Vec& main, Vec& block, int W) {
    for (int p = 1; p < W; ++p) {
        auto z = step_reverse(main, block, W, p);
        main = std::move(z.first); block = std::move(z.second);
    }
}

static bool equal_vec(const Vec& a, const Vec& b, const char* name, int n, int row) {
    if (a == b) return true;
    std::cerr << "FAIL snake " << name << " n=" << n << " row=" << row
              << " sizes=" << a.size() << '/' << b.size() << '\n';
    for (auto [k, v] : a) {
        auto it = b.find(k);
        Count w = it == b.end() ? 0 : it->second;
        if (v != w) {
            std::cerr << " first mismatch mate=" << k << " forward=" << v
                      << " snake=" << w << '\n';
            break;
        }
    }
    return false;
}

static Count solve_and_compare(int n) {
    const int W = n + 1;
    Vec fm{{MateID(R) << (2 * (W - 1)), 1}}, fb;
    Vec sm = fm, sb;

    for (int row = 0; row < W; ++row) {
        row_forward(fm, fb, W);
        if ((row & 1) == 0) row_forward(sm, sb, W);
        else row_reverse(sm, sb, W);
        if (!equal_vec(fm, sm, "main", n, row + 1)
            || !equal_vec(fb, sb, "blocked", n, row + 1)) {
            std::exit(2);
        }
        for (auto [m, c] : sm) {
            (void)c;
            if (mirror_mate(mirror_mate(m, W), W) != m) {
                std::cerr << "FAIL mirror involution n=" << n << " row=" << row + 1 << '\n';
                std::exit(3);
            }
        }
    }
    auto it = fm.find(MateID(R));
    return it == fm.end() ? 0 : it->second;
}

int main(int argc, char** argv) {
    int max_n = argc > 1 ? std::atoi(argv[1]) : 10;
    if (max_n < 1 || max_n > 11) return 1;
    const std::vector<Count> known = {
        0u, 2u, 12u, 184u, 8512u, 1262816u, 575780564u, 3381038999u
    };
    for (int n = 1; n <= max_n; ++n) {
        Count answer = solve_and_compare(n);
        if (n < int(known.size()) && answer != known[n]) {
            std::cerr << "FAIL known residue n=" << n << " got=" << answer
                      << " expected=" << known[n] << '\n';
            return 4;
        }
        std::cout << "snake-row-equivalence n=" << n << " residue=" << answer
                  << " rows=" << (n + 1) << " OK\n";
    }
    std::cout << "gridfp-snake-equivalence OK max_n=" << max_n << '\n';
    return 0;
}
