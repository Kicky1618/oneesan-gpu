#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../../common/gridfp_transition.hpp"
#include "../../common/gridfp_transition_reverse.hpp"

using namespace oneesan::gridfp;

namespace {

using Coef = std::int64_t;
using Rank = std::uint64_t;

struct Key {
    bool blocked = false;
    MateID mate = 0;
    bool operator<(const Key& o) const {
        return blocked != o.blocked ? blocked < o.blocked : mate < o.mate;
    }
    bool operator==(const Key& o) const {
        return blocked == o.blocked && mate == o.mate;
    }
};
using Vec = std::map<Key, Coef>;

[[noreturn]] void fail(const std::string& s) {
    std::cerr << "FAIL " << s << '\n';
    std::exit(1);
}

void add(Vec& v, Key k, Coef c) {
    if (!c) return;
    v[k] += c;
    if (v[k] == 0) v.erase(k);
}

std::vector<MateID> gen_words(int W) {
    std::vector<MateID> out;
    auto rec = [&](auto&& self, int pos, int h, MateID m) -> void {
        const int rem = W - pos;
        if (h < 0 || h > rem) return;
        if (pos == W) {
            if (h == 0) out.push_back(m);
            return;
        }
        const int bit = W - 1 - pos;
        self(self, pos + 1, h, m);
        if (h > 0) self(self, pos + 1, h - 1, m | (MateID(R) << (2 * bit)));
        self(self, pos + 1, h + 1, m | (MateID(L) << (2 * bit)));
    };
    rec(rec, 0, 1, 0);
    std::sort(out.begin(), out.end());
    return out;
}

Vec step_basis(Key src, int W, int p, bool reverse) {
    Vec out;
    if (!src.blocked) {
        add(out, src, 1); // production excluded branch
        IncludeResult z = reverse
            ? include_horizontal_reverse(src.mate, W, p)
            : include_horizontal(src.mate, W, p);
        if (z.valid) add(out, Key{z.blocked, z.mate}, 1);
    } else {
        const MateID m = reverse
            ? blocked_exclude_reverse(src.mate, W, p)
            : blocked_exclude(src.mate, p);
        add(out, Key{false, m}, 1); // blocked states have no included branch
    }
    return out;
}

Vec step_vec(const Vec& in, int W, int p, bool reverse) {
    Vec out;
    for (const auto& [k, c] : in)
        for (const auto& [z, a] : step_basis(k, W, p, reverse)) add(out, z, a * c);
    return out;
}

// For every interior production step S_p, ker(S_p) contains exactly the local
// three-term directions
//
//   B_p(N q) + M_p(LR q) - M_p(NN q).
//
// We use these directions to eliminate blocked coordinates whose lookahead
// symbol is N.  The canonical quotient Q_p therefore consists of all main
// states plus blocked states with bit p-1 occupied.
Vec project_term(Key k, Coef c, int W, int p, bool reverse) {
    Vec out;
    if (!k.blocked || mget(k.mate, p - 1) != N) {
        add(out, k, c);
        return out;
    }

    const MateID nn = reverse
        ? blocked_exclude_reverse(k.mate, W, p)
        : blocked_exclude(k.mate, p);
    const MateID lr = msetpair(nn, p, LR);
    add(out, Key{false, nn}, c);
    add(out, Key{false, lr}, -c);
    return out;
}

Vec project_vec(const Vec& in, int W, int p, bool reverse) {
    Vec out;
    for (const auto& [k, c] : in)
        for (const auto& [z, a] : project_term(k, c, W, p, reverse)) add(out, z, a);
    return out;
}

bool canonical(Key k, int p) {
    return !k.blocked || mget(k.mate, p - 1) != N;
}

std::vector<Key> layout(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int p
) {
    std::vector<Key> out;
    out.reserve(main.size() + block.size());
    for (MateID m : main) out.push_back(Key{false, m});
    for (MateID b : block)
        if (mget(b, p - 1) != N) out.push_back(Key{true, b});
    return out;
}

Vec reduced_step_basis(Key src, int W, int p, bool reverse) {
    const int next = reverse ? p + 1 : p - 1;
    return project_vec(step_basis(src, W, p, reverse), W, next, reverse);
}

Vec full_row(MateID m, int W, bool reverse) {
    Vec v;
    add(v, Key{false, m}, 1);
    if (!reverse) {
        for (int p = W - 1; p >= 1; --p) v = step_vec(v, W, p, false);
    } else {
        for (int p = 1; p < W; ++p) v = step_vec(v, W, p, true);
    }
    return v;
}

// Interior states use Q_p.  At a physical row edge the last two cells are
// fused, which consumes the deferred channel completely and returns to a
// main-only vector.  Thus no special row-boundary blocked buffer is needed.
Vec reduced_row(MateID m, int W, bool reverse) {
    Vec v;
    add(v, Key{false, m}, 1);
    if (!reverse) {
        for (int p = W - 1; p >= 3; --p)
            v = project_vec(step_vec(v, W, p, false), W, p - 1, false);
        v = step_vec(v, W, 2, false);
        v = step_vec(v, W, 1, false);
    } else {
        for (int p = 1; p <= W - 3; ++p)
            v = project_vec(step_vec(v, W, p, true), W, p + 1, true);
        v = step_vec(v, W, W - 2, true);
        v = step_vec(v, W, W - 1, true);
    }
    return v;
}

void verify_kernel_relation(
    const std::vector<MateID>& block,
    int W,
    int p,
    bool reverse
) {
    Rank count = 0;
    for (MateID b : block) {
        if (mget(b, p - 1) != N) continue;
        const MateID nn = reverse
            ? blocked_exclude_reverse(b, W, p)
            : blocked_exclude(b, p);
        const MateID lr = msetpair(nn, p, LR);
        Vec v;
        add(v, Key{true, b}, 1);
        add(v, Key{false, lr}, 1);
        add(v, Key{false, nn}, -1);
        if (!step_vec(v, W, p, reverse).empty())
            fail("three-term kernel relation W=" + std::to_string(W) +
                 " p=" + std::to_string(p));
        ++count;
    }
    const Rank want = gen_words(W - 2).size();
    if (count != want)
        fail("kernel direction count W=" + std::to_string(W));
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 9;
    if (maxW < 4 || maxW > 12) return 2;

    std::vector<Rank> M(static_cast<std::size_t>(maxW + 1));
    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) {
        words[W] = gen_words(W);
        M[W] = words[W].size();
    }

    for (int W = 4; W <= maxW; ++W) {
        const Rank dim = M[W] + M[W - 1] - M[W - 2];
        Rank max_fanout = 0;
        Rank neg_columns = 0;

        for (bool reverse : {false, true}) {
            const int first = reverse ? 1 : 2;
            const int last = reverse ? W - 2 : W - 1;
            for (int p = first; p <= last; ++p) {
                verify_kernel_relation(words[W - 1], W, p, reverse);
                const auto q = layout(words[W], words[W - 1], p);
                if (q.size() != dim)
                    fail("layout dimension W=" + std::to_string(W));
            }

            // Only transitions whose destination is another interior quotient.
            if (!reverse) {
                for (int p = W - 1; p >= 3; --p) {
                    const auto src = layout(words[W], words[W - 1], p);
                    for (Key k : src) {
                        const Vec col = reduced_step_basis(k, W, p, false);
                        max_fanout = std::max<Rank>(max_fanout, col.size());
                        bool neg = false;
                        for (const auto& [z, c] : col) {
                            if (!canonical(z, p - 1)) fail("noncanonical forward destination");
                            if (c != 1 && c != -1) fail("forward coefficient outside +/-1");
                            neg |= c < 0;
                        }
                        neg_columns += neg;
                    }
                }
            } else {
                for (int p = 1; p <= W - 3; ++p) {
                    const auto src = layout(words[W], words[W - 1], p);
                    for (Key k : src) {
                        const Vec col = reduced_step_basis(k, W, p, true);
                        max_fanout = std::max<Rank>(max_fanout, col.size());
                        bool neg = false;
                        for (const auto& [z, c] : col) {
                            if (!canonical(z, p + 1)) fail("noncanonical reverse destination");
                            if (c != 1 && c != -1) fail("reverse coefficient outside +/-1");
                            neg |= c < 0;
                        }
                        neg_columns += neg;
                    }
                }
            }

            // Strong row check: equality on every main basis column implies
            // equality of the complete row operator by linearity.
            for (MateID m : words[W]) {
                if (full_row(m, W, reverse) != reduced_row(m, W, reverse))
                    fail(std::string(reverse ? "reverse" : "forward") +
                         " row mismatch W=" + std::to_string(W));
            }
        }

        if (max_fanout > 3) fail("fanout bound W=" + std::to_string(W));
        std::cout << "W=" << W
                  << " main=" << M[W]
                  << " blocked=" << M[W - 1]
                  << " reduced=" << dim
                  << " eliminated=" << M[W - 2]
                  << " max_fanout=" << max_fanout
                  << " signed_projection_columns=" << neg_columns
                  << " forward_row=OK reverse_row=OK\n";
    }

    // Production-size constants used by n=27 / W=28 planning.
    if (maxW >= 12) {
        std::cout << "NOTE use closed-form Motzkin counts for W=28 planning\n";
    }
    std::cout << "W=28_theory main=385719506620 blocked=135015505407"
              << " eliminated=47337954326 reduced=473397057701"
              << " u32_GiB=1763.541466"
              << " per_8gpu_single_GiB=220.442683"
              << " row_boundary=main_only"
              << "\n";
    std::cout << "ALL_OK production_reduced_channel=1\n";
    return 0;
}
