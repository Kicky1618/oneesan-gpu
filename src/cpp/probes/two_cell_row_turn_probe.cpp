#pragma push_macro("main")
#undef main
#define main two_cell_reverse_channel_probe_main_unused
#include "two_cell_reverse_channel_probe.cpp"
#pragma pop_macro("main")

#include <array>

namespace {

// A physical snake turn sees the boundary pair twice: once as the final cell
// of the old row and once as the first cell of the new row.  Thus the direct
// reduced turn contains T_edge^2, not one T_edge.
CVec turn_right_basis(const Key& src, int W) {
    const int source_pair = W - 3;
    const int edge = W - 2;
    CVec out;
    for (const auto& [w0, a] : E_raw_basis(src, source_pair)) {
        for (const auto& [raw1, b] : R_raw_basis(w0, edge)) {
            for (const auto& [w1, c] : E_raw_basis(raw1, edge)) {
                for (const auto& [raw2, d] : R_raw_basis(w1, edge))
                    add(out, project_key_reverse(raw2, edge, W), a * b * c * d);
            }
        }
    }
    return out;
}

CVec turn_left_basis(const Key& src, int W) {
    // Symmetrically, a reverse row ends with T_0 and the following forward row
    // starts with another T_0.
    const int source_pair = 1;
    const int edge = 0;
    CVec out;
    for (const auto& [w0, a] : E_raw_basis(src, source_pair)) {
        for (const auto& [raw1, b] : R_raw_basis(w0, edge)) {
            for (const auto& [w1, c] : E_raw_basis(raw1, edge)) {
                for (const auto& [raw2, d] : R_raw_basis(w1, edge))
                    add(out, project_key(raw2, edge, W), a * b * c * d);
            }
        }
    }
    return out;
}

std::map<Word, int64_t> expand_channel(const CVec& v, int pair) {
    std::map<Word, int64_t> out;
    for (const auto& [k, c] : v) {
        for (const auto& [w, a] : E_raw_basis(k, pair)) {
            out[w] += c * a;
            if (!out[w]) out.erase(w);
        }
    }
    return out;
}

std::map<Word, int64_t> apply_full(
    const std::map<Word, int64_t>& v, int pair
) {
    std::map<Word, int64_t> out;
    for (const auto& [w, c] : v) {
        for (const Word& z : apply_T_basis(w, pair)) {
            out[z] += c;
            if (!out[z]) out.erase(z);
        }
    }
    return out;
}

std::map<Word, int64_t> expand_one(const Key& k, int pair) {
    std::map<Word, int64_t> out;
    for (const auto& [w, c] : E_raw_basis(k, pair)) out[w] += c;
    return out;
}

Rank count_words(int W) {
    std::vector<std::vector<Rank>> dp(
        static_cast<std::size_t>(W + 1),
        std::vector<Rank>(static_cast<std::size_t>(W + 2), 0));
    dp[W][0] = 1;
    for (int pos = W - 1; pos >= 0; --pos) {
        for (int h = 0; h <= W; ++h) {
            Rank z = dp[pos + 1][h];
            if (h > 0) z += dp[pos + 1][h - 1];
            z += dp[pos + 1][h + 1];
            dp[pos][h] = z;
        }
    }
    return dp[0][1];
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 4 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        const Rank m1 = words[W - 1].size();
        const Rank m2 = words[W - 2].size();
        const Rank m3 = words[W - 3].size();
        const Rank dim = m1 + m2 - m3;

        // T_edge = I + e at beta=0, hence T_edge^2 = I + 2e.  After the
        // quotient this gives the following exact source fanout/nnz counts.
        const Rank expect_n1 = m2;
        const Rank expect_n2 = m2;
        const Rank expect_n3 = m1 - m2 - m3;
        const Rank expect_nnz = 3 * (m1 - m3);
        const Rank expect_coeff2 = m1 - m3;
        const Rank expect_coeff1 = 2 * (m1 - m3);

        const auto right_src = q_basis(W, W - 3, words);
        const auto right_dst = reverse_q_basis(W, W - 2, words);
        const auto left_src = reverse_q_basis(W, 1, words);
        const auto left_dst = q_basis(W, 0, words);
        if (right_src.size() != dim || right_dst.size() != dim ||
            left_src.size() != dim || left_dst.size() != dim)
            fail("row-turn dimension W=" + std::to_string(W));

        const std::set<Key> right_dst_set(right_dst.begin(), right_dst.end());
        const std::set<Key> left_dst_set(left_dst.begin(), left_dst.end());
        std::array<Rank, 4> right_fan{}, left_fan{};
        std::array<Rank, 3> right_coeff{}, left_coeff{};
        Rank right_nnz = 0, left_nnz = 0;
        std::map<Key, Rank> right_indeg, left_indeg;

        for (const Key& src : right_src) {
            const CVec col = turn_right_basis(src, W);
            if (col.empty() || col.size() > 3)
                fail("right turn fanout W=" + std::to_string(W));
            ++right_fan[col.size()];
            right_nnz += col.size();
            for (const auto& [dst, c] : col) {
                if (c < 1 || c > 2 || !right_dst_set.count(dst))
                    fail("right turn destination/coefficient W=" + std::to_string(W));
                ++right_coeff[c];
                ++right_indeg[dst];
            }

            // Exactness is checked after the first adjacent transfer of the
            // new reverse row.  The full side executes T_edge twice.
            const auto full0 = expand_one(src, W - 3);
            const auto actual = apply_full(
                apply_full(apply_full(full0, W - 2), W - 2), W - 3);
            const auto reduced = apply_full(expand_channel(col, W - 2), W - 3);
            if (actual != reduced)
                fail("right turn delayed exactness W=" + std::to_string(W));
        }

        for (const Key& src : left_src) {
            const CVec col = turn_left_basis(src, W);
            if (col.empty() || col.size() > 3)
                fail("left turn fanout W=" + std::to_string(W));
            ++left_fan[col.size()];
            left_nnz += col.size();
            for (const auto& [dst, c] : col) {
                if (c < 1 || c > 2 || !left_dst_set.count(dst))
                    fail("left turn destination/coefficient W=" + std::to_string(W));
                ++left_coeff[c];
                ++left_indeg[dst];
            }

            const auto full0 = expand_one(src, 1);
            const auto actual = apply_full(
                apply_full(apply_full(full0, 0), 0), 1);
            const auto reduced = apply_full(expand_channel(col, 0), 1);
            if (actual != reduced)
                fail("left turn delayed exactness W=" + std::to_string(W));

            // Left and right physical turns are exact geometric reflections.
            const Key mirrored_src = reflect_key(src);
            const CVec mirrored_right = reflect_vec(turn_right_basis(mirrored_src, W));
            if (col != mirrored_right)
                fail("turn reflection W=" + std::to_string(W));
        }

        const std::array<Rank, 4> expected_fan{{0, expect_n1, expect_n2, expect_n3}};
        const std::array<Rank, 3> expected_coeff{{0, expect_coeff1, expect_coeff2}};
        if (right_fan != expected_fan || left_fan != expected_fan ||
            right_coeff != expected_coeff || left_coeff != expected_coeff ||
            right_nnz != expect_nnz || left_nnz != expect_nnz)
            fail("turn formula W=" + std::to_string(W));

        Rank max_indeg = 0;
        for (const auto& kv : right_indeg) max_indeg = std::max(max_indeg, kv.second);
        for (const auto& kv : left_indeg) max_indeg = std::max(max_indeg, kv.second);

        std::cout << "W=" << W
                  << " reduced=" << dim
                  << " turn_nnz=" << right_nnz
                  << " n1=" << right_fan[1]
                  << " n2=" << right_fan[2]
                  << " n3=" << right_fan[3]
                  << " coeff1=" << right_coeff[1]
                  << " coeff2=" << right_coeff[2]
                  << " max_indeg=" << max_indeg
                  << " one_cell_boundary_buffer=0"
                  << " OK\n";
    }

    const Rank m27 = count_words(27);
    const Rank m26 = count_words(26);
    const Rank m25 = count_words(25);
    const Rank r28 = m27 + m26 - m25;
    const Rank nnz28 = 3 * (m27 - m25);
    const Rank n3_28 = m27 - m26 - m25;
    const Rank coeff2_28 = m27 - m25;
    const Rank coeff1_28 = 2 * coeff2_28;
    const double removed_gib = double(m25) * 4.0 / double(1ull << 30);
    std::cout << "W=28_theory reduced=" << r28
              << " turn_nnz=" << nnz28
              << " n1=" << m26
              << " n2=" << m26
              << " n3=" << n3_28
              << " coeff1=" << coeff1_28
              << " coeff2=" << coeff2_28
              << " avoided_one_cell_values=" << m25
              << " avoided_one_cell_gib=" << removed_gib
              << "\n";
    std::cout << "ALL_OK direct_physical_row_turn=1 boundary_fallback=0\n";
    return 0;
}
