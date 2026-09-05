#pragma push_macro("main")
#undef main
#define main two_cell_packed_component_probe_main_unused
#include "two_cell_packed_component_probe.cpp"
#pragma pop_macro("main")

#include <optional>

namespace {

// Deep labels are exactly the three local patterns RN, LN and LR. Collapsing
// RN->R, LN->L and LR->N is a bijection from deep width-(W-2) labels to all
// width-(W-3) one-defect Motzkin words.
std::optional<Word> deep_collapse_label(const Word& u, int fixed) {
    if (fixed < 0 || fixed + 1 >= static_cast<int>(u.size())) return std::nullopt;
    char c = 0;
    if (u[fixed] == R && u[fixed + 1] == N) c = R;
    else if (u[fixed] == L && u[fixed + 1] == N) c = L;
    else if (u[fixed] == L && u[fixed + 1] == R) c = N;
    else return std::nullopt;
    const Word v = u.substr(0, fixed) + c + u.substr(fixed + 2);
    if (!valid_word(v)) fail("deep collapse invalid");
    return v;
}

// At the left boundary, r is the number of top-level strands encountered
// before and including the distinguished root. N is skipped; L contributes one
// strand and jumps to the symbol after its matching R; R is the root and ends
// the walk. This is the face degree seen by the reduced component.
int boundary_face_strands(const Word& v) {
    int pos = 0;
    int r = 0;
    while (pos < static_cast<int>(v.size())) {
        if (v[pos] == N) {
            ++pos;
            continue;
        }
        if (v[pos] == R) {
            ++r;
            return r;
        }
        const LinkState s = decode(v);
        const int q = s.mate[pos];
        if (q <= pos) fail("boundary face invalid L partner");
        ++r;
        pos = q + 1;
    }
    fail("boundary face without root");
}

Rank standard_motzkin(int n) {
    std::vector<Rank> cur(static_cast<std::size_t>(n + 2));
    std::vector<Rank> nxt(static_cast<std::size_t>(n + 2));
    cur[0] = 1;
    for (int pos = 0; pos < n; ++pos) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= n; ++h) {
            const Rank x = cur[h];
            if (!x) continue;
            nxt[h] += x;
            nxt[h + 1] += x;
            if (h > 0) nxt[h - 1] += x;
        }
        cur.swap(nxt);
    }
    return cur[0];
}

// D[n][r] counts width-n one-defect words by boundary_face_strands=r.
// The first-symbol decomposition is
//
//   R a       : r=1, a ordinary Motzkin
//   N b       : r(b)
//   L a R b   : 1+r(b), a ordinary Motzkin
//
// so the bivariate generating function is
//
//   D(x,y) = x y M(x) / (1 - x - x^2 y M(x)).
//
// The recurrence below is the coefficient form and uses exact uint64 counts
// through the widths relevant here.
std::vector<std::vector<Rank>> build_face_distribution(int max_n) {
    std::vector<Rank> motz(static_cast<std::size_t>(max_n + 1));
    for (int n = 0; n <= max_n; ++n) motz[n] = standard_motzkin(n);

    std::vector<std::vector<Rank>> D(
        static_cast<std::size_t>(max_n + 1),
        std::vector<Rank>(static_cast<std::size_t>(max_n + 2), 0));
    if (max_n >= 1) D[1][1] = 1;
    for (int n = 2; n <= max_n; ++n) {
        for (int r = 1; r <= max_n; ++r) D[n][r] += D[n - 1][r];
        D[n][1] += motz[n - 1];
        for (int a = 0; a <= n - 3; ++a) {
            const int b = n - 2 - a;
            for (int r = 1; r <= max_n; ++r)
                D[n][r + 1] += motz[a] * D[b][r];
        }
    }
    return D;
}

Rank one_defect_count_dp(int n) {
    std::vector<Rank> cur(static_cast<std::size_t>(n + 2));
    std::vector<Rank> nxt(static_cast<std::size_t>(n + 2));
    cur[1] = 1;
    for (int pos = 0; pos < n; ++pos) {
        std::fill(nxt.begin(), nxt.end(), 0);
        for (int h = 0; h <= n; ++h) {
            const Rank x = cur[h];
            if (!x) continue;
            nxt[h] += x;
            nxt[h + 1] += x;
            if (h > 0) nxt[h - 1] += x;
        }
        cur.swap(nxt);
    }
    return cur[0];
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 5 || maxW > 15) return 2;

    const auto D = build_face_distribution(25);
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        const int n = W - 3;
        std::map<Rank, Rank> reference_distribution;
        bool first_position = true;

        for (int i = 0; i <= W - 4; ++i) {
            std::map<Rank, Rank> distribution;
            std::set<Word> deep_collapsed;
            Rank shallow1 = 0;
            Rank shallow3 = 0;
            Rank deep = 0;

            for (const Word& u : words[W - 2]) {
                const Key seed = project_key(Key{'C', u}, i, W);
                const auto packed = packed_component_sources(pack_key(seed), W, i);
                const Rank pairs = packed.size;
                ++distribution[pairs];

                const auto collapsed = deep_collapse_label(u, i);
                if (u[i] == N) {
                    if (pairs != 1 || collapsed)
                        fail("size-1 classification W=" + std::to_string(W));
                    ++shallow1;
                } else if (!collapsed) {
                    if (pairs != 3)
                        fail("size-3 classification W=" + std::to_string(W));
                    ++shallow3;
                } else {
                    if (pairs < 5)
                        fail("deep component too small W=" + std::to_string(W));
                    if (!deep_collapsed.insert(*collapsed).second)
                        fail("deep collapse collision W=" + std::to_string(W));
                    ++deep;
                    if (i == 0) {
                        const int r = boundary_face_strands(*collapsed);
                        if (pairs != Rank(4 + r))
                            fail("boundary face size W=" + std::to_string(W));
                    }
                }
            }

            if (shallow1 != words[W - 3].size())
                fail("size-1 count W=" + std::to_string(W));
            if (deep != words[W - 3].size() || deep_collapsed.size() != words[W - 3].size())
                fail("deep bijection count W=" + std::to_string(W));
            if (deep_collapsed != std::set<Word>(words[W - 3].begin(), words[W - 3].end()))
                fail("deep collapse not onto W-3 basis");
            if (shallow3 != words[W - 2].size() - 2 * words[W - 3].size())
                fail("size-3 count W=" + std::to_string(W));

            if (first_position) {
                reference_distribution = distribution;
                first_position = false;
            } else if (distribution != reference_distribution) {
                fail("component size distribution depends on position W=" + std::to_string(W));
            }
        }

        std::map<Rank, Rank> predicted;
        predicted[1] = words[W - 3].size();
        predicted[3] = words[W - 2].size() - 2 * words[W - 3].size();
        for (int r = 1; r <= n; ++r)
            if (D[n][r]) predicted[Rank(4 + r)] += D[n][r];
        if (predicted != reference_distribution)
            fail("component size distribution recurrence W=" + std::to_string(W));

        Rank weighted = 0;
        Rank components = 0;
        Rank max_pairs = 0;
        for (const auto& [pairs, count] : reference_distribution) {
            weighted += pairs * count;
            components += count;
            max_pairs = std::max(max_pairs, pairs);
        }
        const Rank reduced = words[W - 1].size() + words[W - 2].size() - words[W - 3].size();
        if (weighted != reduced || components != words[W - 2].size())
            fail("component weighted dimension W=" + std::to_string(W));

        const Rank exact_bound = Rank(W / 2 + 3);
        if (max_pairs != exact_bound && W % 2 == 0)
            fail("even-width max component bound W=" + std::to_string(W));
        if (max_pairs > exact_bound)
            fail("component bound W=" + std::to_string(W));

        std::cout << "W=" << W
                  << " components=" << components
                  << " reduced=" << reduced
                  << " max_pairs=" << max_pairs
                  << " bound=floor(W/2)+3"
                  << " deep_bijection=M_Wm3"
                  << " position_independent_distribution=1"
                  << " generating_function=xyM/(1-x-x2yM)"
                  << " OK\n";
    }

    constexpr int W = 28;
    constexpr int n = W - 3;
    const Rank m27 = one_defect_count_dp(27);
    const Rank m26 = one_defect_count_dp(26);
    const Rank m25 = one_defect_count_dp(25);
    const Rank reduced = m27 + m26 - m25;
    std::map<Rank, Rank> dist28;
    dist28[1] = m25;
    dist28[3] = m26 - 2 * m25;
    for (int r = 1; r <= n; ++r)
        if (D[n][r]) dist28[Rank(4 + r)] += D[n][r];

    Rank component_sum = 0;
    Rank weighted = 0;
    for (const auto& [pairs, count] : dist28) {
        component_sum += count;
        weighted += pairs * count;
        std::cout << "W=28_bucket pairs=" << pairs << " components=" << count << '\n';
    }
    if (component_sum != m26 || weighted != reduced)
        fail("W=28 distribution totals");

    std::cout << "W=28_theory components=" << component_sum
              << " reduced=" << reduced
              << " max_pairs=17"
              << " max_component_values_u32_bytes=68"
              << " fixed_local_capacity_32=SAFE_BY_SIZE_BOUND"
              << "\n";
    std::cout << "ALL_OK component_size_distribution=1\n";
    return 0;
}
