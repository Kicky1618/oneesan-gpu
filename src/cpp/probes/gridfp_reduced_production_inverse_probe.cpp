#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_channel_probe_main_unused
#include "gridfp_reduced_production_channel_probe.cpp"
#pragma pop_macro("main")

#include "../../common/gridfp_closure_inverse.hpp"

#include <set>

namespace {

bool valid_mate(MateID m, int W) {
    int h = 1;
    for (int pos = 0; pos < W; ++pos) {
        const MateValue v = mget(m, W - 1 - pos);
        if (v == X) return false;
        if (v == R) --h;
        else if (v == L) ++h;
        if (h < 0) return false;
    }
    return h == 0;
}

std::vector<MateID> blocked_include_preimages_forward(MateID b, int W, int p) {
    std::set<MateID> out;

    // Deferred NR/NL shift: the removed high site was N.
    if (is_endpoint(mget(b, p - 1))) {
        const MateID x = minsert(b, p, N);
        const IncludeResult z = include_horizontal(x, W, p);
        if (valid_mate(x, W) && z.valid && z.blocked && z.mate == b) out.insert(x);
    }

    // Deferred LL/RR/RL closure: restore the removed low site as N, then use
    // the same algorithmic closure inverse as the production CUDA backend.
    const MateID closure_dest = minsert(b, p - 1, N);
    MateID cand[32]{};
    const int n = ordinary_closure_preimages_partial(closure_dest, W, p, cand);
    for (int i = 0; i < n; ++i) {
        const MateID x = cand[i];
        const IncludeResult z = include_horizontal(x, W, p);
        if (valid_mate(x, W) && z.valid && z.blocked && z.mate == b) out.insert(x);
    }

    return std::vector<MateID>(out.begin(), out.end());
}

Vec inverse_reduced_forward(Key dest, int W, int p) {
    Vec out;
    if (dest.blocked) {
        if (mget(dest.mate, p - 2) == N) fail("noncanonical blocked destination inverse");
        for (MateID x : blocked_include_preimages_forward(dest.mate, W, p))
            add(out, Key{false, x}, 1);
        return out;
    }

    const MateID d = dest.mate;
    add(out, Key{false, d}, 1); // excluded identity

    // Ordinary nonblocked included inverses at an interior position.
    const MateValuePair w = mpair(d, p);
    auto try_main = [&](MateID x) {
        if (!valid_mate(x, W)) return;
        const IncludeResult z = include_horizontal(x, W, p);
        if (z.valid && !z.blocked && z.mate == d) add(out, Key{false, x}, 1);
    };
    if (w == LR) try_main(msetpair(d, p, NN));
    if (w == NR) try_main(msetpair(d, p, RN));
    if (w == NL) try_main(msetpair(d, p, LN));

    // A retained blocked source has only its excluded branch.
    if (mget(d, p) == N && is_endpoint(mget(d, p - 1))) {
        const MateID b = mshrink(d, p);
        if (valid_mate(b, W - 1) && mget(b, p - 1) != N &&
            blocked_exclude(b, p) == d)
            add(out, Key{true, b}, 1);
    }

    // If S_p produced a blocked state that is noncanonical for Q_{p-1}, its
    // quotient projection contributes +M(NN) - M(LR). Reconstruct that blocked
    // result and then all of its ordinary included preimages.
    const int q = p - 1;
    const MateValuePair qpair = mpair(d, q);
    if (qpair == NN || qpair == LR) {
        const MateID nn = qpair == NN ? d : msetpair(d, q, NN);
        const MateID b = mshrink(nn, q);
        if (valid_mate(b, W - 1) && mget(b, q - 1) == N) {
            const Coef sign = qpair == NN ? 1 : -1;
            for (MateID x : blocked_include_preimages_forward(b, W, p))
                add(out, Key{false, x}, sign);
        }
    }

    return out;
}

Key mirror_key(Key k, int W) {
    return Key{k.blocked, mirror_mate(k.mate, k.blocked ? W - 1 : W)};
}

Vec inverse_reduced(Key dest, int W, int p, bool reverse) {
    if (!reverse) return inverse_reduced_forward(dest, W, p);
    const int fp = W - p;
    const Vec mirrored = inverse_reduced_forward(mirror_key(dest, W), W, fp);
    Vec out;
    for (const auto& [k, c] : mirrored) add(out, mirror_key(k, W), c);
    return out;
}

void verify_inverse_position(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    int p,
    bool reverse,
    Rank& max_indegree
) {
    const int next = reverse ? p + 1 : p - 1;
    const auto src = layout(main, block, p);
    const auto dst = layout(main, block, next);
    std::map<Key, Vec> incoming;
    for (Key s : src) {
        for (const auto& [d, c] : reduced_step_basis(s, W, p, reverse))
            add(incoming[d], s, c);
    }
    for (Key d : dst) {
        const Vec got = inverse_reduced(d, W, p, reverse);
        const auto it = incoming.find(d);
        const Vec empty;
        const Vec& want = it == incoming.end() ? empty : it->second;
        if (got != want)
            fail(std::string(reverse ? "reverse" : "forward") +
                 " inverse mismatch W=" + std::to_string(W) +
                 " p=" + std::to_string(p));
        max_indegree = std::max<Rank>(max_indegree, got.size());
    }
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    if (maxW < 5 || maxW > 12) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        Rank max_indegree = 0;
        for (int p = W - 1; p >= 3; --p)
            verify_inverse_position(words[W], words[W - 1], W, p, false, max_indegree);
        for (int p = 1; p <= W - 3; ++p)
            verify_inverse_position(words[W], words[W - 1], W, p, true, max_indegree);

        const Rank observed_bound = Rank(W / 2 + 2);
        if (max_indegree > observed_bound) fail("inverse indegree bound W=" + std::to_string(W));
        std::cout << "W=" << W
                  << " max_indegree=" << max_indegree
                  << " observed_bound=floor(W/2)+2"
                  << " inverse_table_bytes=0"
                  << " closure_inverse=reused"
                  << " forward=OK reverse=OK\n";
    }

    std::cout << "W=28_theory observed_max_indegree_bound=16"
              << " inverse_table_bytes=0\n";
    std::cout << "ALL_OK production_reduced_inverse=1\n";
    return 0;
}
