#pragma push_macro("main")
#undef main
#define main two_cell_turn_component_probe_main_unused
#include "two_cell_turn_component_probe.cpp"
#pragma pop_macro("main")

namespace {

struct ClosedTurnComponent {
    bool singular = false;
    Key alpha;
    Key beta;
    std::vector<Key> passive;

    std::vector<Key> states() const {
        std::vector<Key> out;
        out.reserve(static_cast<std::size_t>(2 + passive.size()));
        out.push_back(alpha);
        out.push_back(beta);
        out.insert(out.end(), passive.begin(), passive.end());
        return out;
    }
};

ClosedTurnComponent closed_right_turn_component(const Word& u, int W) {
    if (static_cast<int>(u.size()) != W - 2) fail("closed turn label width");
    const int edge_pos = W - 3;
    ClosedTurnComponent out;

    if (u.back() == R) {
        // Retained C seed. The whole cyclic turn component is the three local
        // coordinates around the boundary vacancy.
        out.singular = true;
        out.alpha = Key{'A', u.substr(0, static_cast<std::size_t>(edge_pos)) + N + R};
        out.beta = Key{'C', u};
        out.passive.push_back(Key{'A', u + N});
        return out;
    }
    if (u.back() != N) fail("closed turn invalid last symbol");

    // Eliminated C seed. Write u=vN. The two distinguished A coordinates are
    // vLR and vNN. Every passive coordinate corresponds to one top-level
    // strand on the right outer face of v, i.e. an R step that lands at height
    // zero. Cutting that strand flips its R endpoint to L and appends RR.
    const Word v = u.substr(0, u.size() - 1);
    out.alpha = Key{'A', v + L + R};
    out.beta = Key{'A', v + N + N};
    int h = 1;
    for (int q = 0; q < static_cast<int>(v.size()); ++q) {
        if (v[q] == L) {
            ++h;
        } else if (v[q] == R) {
            --h;
            if (h == 0) {
                Word z = v;
                z[static_cast<std::size_t>(q)] = L;
                z += R;
                z += R;
                if (!valid_word(z)) fail("closed turn passive invalid");
                out.passive.push_back(Key{'A', z});
            }
        }
        if (h < 0) fail("closed turn prefix height");
    }
    if (h != 0 || out.passive.empty()) fail("closed turn final height");
    return out;
}

CVec closed_turn_column(const ClosedTurnComponent& c, const Key& src) {
    CVec out;
    if (c.singular) {
        if (src == c.alpha) {
            add(out, c.alpha, 2);
            return out;
        }
        if (src == c.beta) {
            add(out, c.beta, 1);
            add(out, c.passive[0], 1);
            return out;
        }
        if (src == c.passive[0]) {
            add(out, c.alpha, 2);
            add(out, c.beta, 1);
            add(out, c.passive[0], 1);
            return out;
        }
        fail("closed singular source outside component");
    }

    if (src == c.alpha) {
        add(out, c.alpha, 2);
        return out;
    }
    if (src == c.beta) {
        add(out, c.alpha, 1);
        add(out, c.beta, 1);
        return out;
    }
    for (const Key& p : c.passive) {
        if (!(src == p)) continue;
        add(out, c.alpha, 1);
        add(out, c.beta, 1);
        add(out, p, 2);
        return out;
    }
    fail("closed nonsingular source outside component");
}

std::vector<std::int64_t> closed_turn_apply(
    const ClosedTurnComponent& c,
    const std::vector<std::int64_t>& x
) {
    const std::size_t n = 2 + c.passive.size();
    if (x.size() != n) fail("closed turn apply size");
    std::vector<std::int64_t> y(n);

    if (c.singular) {
        if (n != 3) fail("closed singular size");
        // [alpha,beta,passive]:
        // y_alpha=2(alpha+passive), y_beta=y_passive=beta+passive.
        const std::int64_t t0 = x[0] + x[2];
        const std::int64_t t1 = x[1] + x[2];
        y[0] = 2 * t0;
        y[1] = t1;
        y[2] = t1;
        return y;
    }

    // [alpha,beta,passive...]. Let t=beta+sum(passive). Then
    // y_alpha=2*alpha+t, y_beta=t, every passive is doubled. This uses
    // exactly n-1 additions, independent of the cyclic sparse support graph.
    std::int64_t t = x[1];
    for (std::size_t q = 2; q < n; ++q) t += x[q];
    y[0] = 2 * x[0] + t;
    y[1] = t;
    for (std::size_t q = 2; q < n; ++q) y[q] = 2 * x[q];
    return y;
}

Rank count_return_paths(int len, int returns) {
    std::vector<std::vector<Rank>> cur(
        static_cast<std::size_t>(len + 2),
        std::vector<Rank>(static_cast<std::size_t>(returns + 2)));
    std::vector<std::vector<Rank>> nxt = cur;
    cur[1][0] = 1;
    for (int pos = 0; pos < len; ++pos) {
        for (auto& row : nxt) std::fill(row.begin(), row.end(), 0);
        for (int h = 0; h <= len; ++h) {
            for (int r = 0; r <= returns; ++r) {
                const Rank x = cur[static_cast<std::size_t>(h)][static_cast<std::size_t>(r)];
                if (!x) continue;
                nxt[static_cast<std::size_t>(h)][static_cast<std::size_t>(r)] += x; // N
                nxt[static_cast<std::size_t>(h + 1)][static_cast<std::size_t>(r)] += x; // L
                if (h > 0) {
                    const int nr = r + (h == 1);
                    if (nr <= returns)
                        nxt[static_cast<std::size_t>(h - 1)][static_cast<std::size_t>(nr)] += x;
                }
            }
        }
        cur.swap(nxt);
    }
    return cur[0][static_cast<std::size_t>(returns)];
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 13;
    if (maxW < 4 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        const auto basis = q_basis(W, W - 3, words);
        const std::set<Key> basis_set(basis.begin(), basis.end());
        std::set<Key> covered;
        Rank singular = 0;
        Rank nonsingular = 0;
        Rank max_size = 0;
        Rank total_adds = 0;
        Rank checked_columns = 0;

        for (const Word& u : words[W - 2]) {
            const ClosedTurnComponent c = closed_right_turn_component(u, W);
            const auto states = c.states();
            if (c.singular) ++singular;
            else ++nonsingular;
            max_size = std::max<Rank>(max_size, states.size());
            total_adds += states.size() - 1;

            for (const Key& s : states) {
                if (!basis_set.count(s) || !covered.insert(s).second)
                    fail("closed turn state partition W=" + std::to_string(W));
                const CVec exact = turn_right_basis(s, W);
                const CVec formula = closed_turn_column(c, s);
                if (exact != formula)
                    fail("closed turn column formula W=" + std::to_string(W));
                ++checked_columns;
            }

            std::vector<std::int64_t> x(states.size());
            for (std::size_t q = 0; q < x.size(); ++q)
                x[q] = static_cast<std::int64_t>(17 + 13 * q + u.size());
            const auto y = closed_turn_apply(c, x);
            CVec expected;
            for (std::size_t q = 0; q < states.size(); ++q) {
                for (const auto& [d, coef] : turn_right_basis(states[q], W))
                    expected[d] += coef * x[q];
            }
            for (std::size_t q = 0; q < states.size(); ++q) {
                if (expected[states[q]] != y[q])
                    fail("closed turn block apply W=" + std::to_string(W));
            }
        }

        if (covered != basis_set)
            fail("closed turn incomplete partition W=" + std::to_string(W));
        const Rank m2 = words[W - 2].size();
        const Rank m3 = words[W - 3].size();
        if (nonsingular != m3 || singular != m2 - m3)
            fail("closed turn terminal class counts W=" + std::to_string(W));
        if (total_adds != basis.size() - m2)
            fail("closed turn addition count W=" + std::to_string(W));

        std::cout << "W=" << W
                  << " states=" << basis.size()
                  << " components=" << m2
                  << " singular_R=" << singular
                  << " nonsingular_N=" << nonsingular
                  << " max_size=" << max_size
                  << " checked_columns=" << checked_columns
                  << " closed_block_adds=" << total_adds
                  << " expected_adds=states-components"
                  << " destination_rank=0 component_graph=0"
                  << " OK\n";
    }

    const Rank m27 = count_words(27);
    const Rank m26 = count_words(26);
    const Rank m25 = count_words(25);
    const Rank r28 = m27 + m26 - m25;
    const Rank singular28 = m26 - m25;
    const Rank adds28 = r28 - m26;
    std::cout << "W=28_theory states=" << r28
              << " components=" << m26
              << " singular_R=" << singular28
              << " nonsingular_N=" << m25
              << " closed_block_adds=" << adds28
              << " sparse_turn_nnz=" << 3 * (m27 - m25)
              << " max_component_size=15"
              << " one_u32_vector_GiB="
              << double(r28 * 4ULL) / double(1ULL << 30)
              << "\n";
    std::cout << "W=28_nonsingular_size_distribution";
    for (int r = 1; r <= 13; ++r) {
        const Rank cnt = count_return_paths(25, r);
        if (cnt) std::cout << " size" << (2 + r) << "=" << cnt;
    }
    std::cout << "\n";
    std::cout << "ALL_OK closed_form_physical_turn_blocks=1\n";
    return 0;
}
