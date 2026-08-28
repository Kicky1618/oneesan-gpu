#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_channel_probe_main_unused
#include "gridfp_reduced_production_channel_probe.cpp"
#pragma pop_macro("main")

namespace {

static constexpr std::uint32_t MOD = 1000003u;

std::uint32_t mod_pow(std::uint32_t a, std::uint32_t e) {
    std::uint64_t r = 1, x = a;
    while (e) {
        if (e & 1u) r = r * x % MOD;
        x = x * x % MOD;
        e >>= 1;
    }
    return static_cast<std::uint32_t>(r);
}

Rank rank_mod(std::vector<std::uint32_t> a, Rank n) {
    Rank row = 0;
    for (Rank col = 0; col < n && row < n; ++col) {
        Rank pivot = row;
        while (pivot < n && a[static_cast<std::size_t>(pivot * n + col)] == 0) ++pivot;
        if (pivot == n) continue;
        if (pivot != row) {
            for (Rank j = col; j < n; ++j)
                std::swap(a[static_cast<std::size_t>(row * n + j)],
                          a[static_cast<std::size_t>(pivot * n + j)]);
        }
        const std::uint32_t pv = a[static_cast<std::size_t>(row * n + col)];
        const std::uint32_t inv = mod_pow(pv, MOD - 2);
        for (Rank j = col; j < n; ++j)
            a[static_cast<std::size_t>(row * n + j)] =
                std::uint64_t(a[static_cast<std::size_t>(row * n + j)]) * inv % MOD;

        for (Rank i = 0; i < n; ++i) {
            if (i == row) continue;
            const std::uint32_t f = a[static_cast<std::size_t>(i * n + col)];
            if (!f) continue;
            for (Rank j = col; j < n; ++j) {
                const std::uint32_t x = a[static_cast<std::size_t>(i * n + j)];
                const std::uint32_t y =
                    std::uint64_t(f) * a[static_cast<std::size_t>(row * n + j)] % MOD;
                a[static_cast<std::size_t>(i * n + j)] = x >= y ? x - y : x + MOD - y;
            }
        }
        ++row;
    }
    return row;
}

Rank full_step_rank(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    int W,
    int p,
    bool reverse
) {
    const Rank nm = main.size();
    const Rank nb = block.size();
    const Rank n = nm + nb;
    std::map<MateID, Rank> mr, br;
    for (Rank r = 0; r < nm; ++r) mr.emplace(main[static_cast<std::size_t>(r)], r);
    for (Rank r = 0; r < nb; ++r) br.emplace(block[static_cast<std::size_t>(r)], r);

    std::vector<std::uint32_t> a(static_cast<std::size_t>(n * n));
    for (Rank s = 0; s < nm; ++s) {
        const Key src{false, main[static_cast<std::size_t>(s)]};
        for (const auto& [d, c] : step_basis(src, W, p, reverse)) {
            if (c != 1) fail("rank main coefficient");
            const Rank r = d.blocked ? nm + br.at(d.mate) : mr.at(d.mate);
            ++a[static_cast<std::size_t>(r * n + s)];
        }
    }
    for (Rank s = 0; s < nb; ++s) {
        const Key src{true, block[static_cast<std::size_t>(s)]};
        for (const auto& [d, c] : step_basis(src, W, p, reverse)) {
            if (c != 1 || d.blocked) fail("rank blocked coefficient");
            const Rank r = mr.at(d.mate);
            ++a[static_cast<std::size_t>(r * n + nm + s)];
        }
    }
    return rank_mod(std::move(a), n);
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 7;
    if (maxW < 4 || maxW > 8) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    std::vector<Rank> M(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) {
        words[W] = gen_words(W);
        M[W] = words[W].size();
    }

    for (int W = 4; W <= maxW; ++W) {
        const Rank want = M[W] + M[W - 1] - M[W - 2];
        for (bool reverse : {false, true}) {
            const int first = reverse ? 1 : 2;
            const int last = reverse ? W - 2 : W - 1;
            for (int p = first; p <= last; ++p) {
                const Rank got = full_step_rank(words[W], words[W - 1], W, p, reverse);
                if (got != want)
                    fail(std::string(reverse ? "reverse" : "forward") +
                         " rank W=" + std::to_string(W) +
                         " p=" + std::to_string(p) +
                         " got=" + std::to_string(got) +
                         " want=" + std::to_string(want));
            }
        }
        std::cout << "W=" << W
                  << " full_dim=" << M[W] + M[W - 1]
                  << " rank=" << want
                  << " nullity=" << M[W - 2]
                  << " quotient_dim=" << want
                  << " forward=OK reverse=OK\n";
    }
    std::cout << "ALL_OK production_reduced_rank=1 prime=" << MOD << '\n';
    return 0;
}
