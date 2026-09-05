#include "../../common/gridfp_transition.hpp"
#include <array>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace oneesan::gridfp;

static_assert(N == 0 && R == 1 && L == 2 && X == 3);
static_assert(NN == 0x0 && NR == 0x1 && NL == 0x2);
static_assert(RN == 0x4 && RR == 0x5 && RL == 0x6);
static_assert(LN == 0x8 && LR == 0x9 && LL == 0xa);

static MateID enc(const std::string& s) {
    MateID m = 0;
    const int W = static_cast<int>(s.size());
    for (int i = 0; i < W; ++i) {
        MateValue v = s[i] == 'N' ? N : s[i] == 'R' ? R : L;
        m = mset(m, W - 1 - i, v);
    }
    return m;
}

static std::string dec(MateID m, int W) {
    std::string s;
    s.reserve(W);
    for (int p = W - 1; p >= 0; --p) {
        auto v = mget(m, p);
        s += v == N ? 'N' : v == R ? 'R' : v == L ? 'L' : 'X';
    }
    return s;
}

static bool check(const std::string& src, int p, bool blocked,
                  const std::string& expected_effective) {
    const int W = static_cast<int>(src.size());
    auto z = include_horizontal(enc(src), W, p);
    if (!z.valid || z.blocked != blocked) return false;
    MateID out = z.mate;
    int outW = W - (z.blocked ? 1 : 0);
    if (z.blocked) {
        if (p <= 1) return false;
        out = blocked_exclude(out, p - 1);
        outW = W;
    }
    return dec(out, outW) == expected_effective;
}

int main() {
    struct T { const char* src; int p; bool blocked; const char* dst; };
    const std::vector<T> tests = {
        // Interior short cases; blocked cases are composed with forced N.
        {"NNNN", 3, false, "LRNN"},
        {"NRNN", 3, true,  "RNNN"},
        {"NLNN", 3, true,  "LNNN"},
        {"RNNN", 3, false, "NRNN"},
        {"LNNN", 3, false, "NLNN"},
        {"RLNN", 3, true,  "NNNN"},
        // Final-cell p=1 versions return directly to main.
        {"NN", 1, false, "LR"},
        {"NR", 1, false, "RN"},
        {"NL", 1, false, "LN"},
        {"RN", 1, false, "NR"},
        {"LN", 1, false, "NL"},
        {"RL", 1, false, "NN"},
        // LL matching-loop semantics: LL mid R -> NN mid L.
        {"LLRNN",   4, true, "NNLNN"},
        {"LLNRN",   4, true, "NNNLN"},
        {"LLLRRNN", 6, true, "NNLRLNN"},
        // RR matching-loop semantics: L mid RR -> R mid NN.
        {"LRRNN",  3, true, "RNNNN"},
        {"LNRRNN", 3, true, "RNNNNN"},
        {"LLRRNN", 3, true, "LRNNNN"},
    };
    std::size_t bad = 0;
    for (auto const& t : tests) {
        if (!check(t.src, t.p, t.blocked, t.dst)) {
            std::cerr << "bad src=" << t.src << " p=" << t.p
                      << " blocked=" << t.blocked << " expected=" << t.dst << "\n";
            ++bad;
        }
    }
    // LR closes a component cycle and is intentionally not an included branch
    // in the production table; invalid X-containing pairs are rejected too.
    auto lr = include_horizontal(enc("LR"), 2, 1);
    if (lr.valid) ++bad;
    MateID x = msetpair(0, 1, NX);
    if (include_horizontal(x, 2, 1).valid) ++bad;

    std::cout << "gridfp_transition_semantic_abi cases=" << tests.size() + 2
              << " bad=" << bad << " exact=" << (bad == 0) << "\n";
    return bad ? 1 : 0;
}
