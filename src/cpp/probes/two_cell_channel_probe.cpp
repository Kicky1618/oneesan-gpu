#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

// Exact CPU probe for the two-cell minimal channel of the canonical N/R/L
// one-defect Motzkin basis.
//
// For an active pair i,i+1 the local beta=0 seven-tile operator factors as
//
//     T_i = E_i R_i,
//
// where the one-cell image channel is A_i(M_{W-1}) + C_i(M_{W-2}).
// A collapses the active pair to one site; C inserts the local LR cup.
// Before the next forward cell, the only redundant channel direction is
//
//     C_i(Nq) == A_i(LRq)  mod ker(T_{i+1}).
//
// The canonical rank layout is therefore
//
//     [ A_i(M_{W-1}) ][ C_i(M_{W-2}) with w[i] != N ],
//
// of size M_{W-1} + M_{W-2} - M_{W-3}. The second block is position
// dependent, but the total size is not.
//
// The reduced transition has source fan-out at most three, with
//
//     n1 = 2 M_{W-2} - M_{W-3},
//     n2 = M_{W-1} - 2 M_{W-2} + M_{W-3},
//     n3 = M_{W-2} - M_{W-3}.
//
// This probe exhaustively verifies:
//   1. T_i = E_i R_i on every full basis state and every position;
//   2. the quotient relation above;
//   3. the channel dimension/fan-out/nnz formulae and unit coefficients;
//   4. delayed exactness: after one reduced step, applying the following full
//      T gives exactly the same full vector as two consecutive full T steps;
//   5. the explicit A/C rank layout is a bijection;
//   6. a destination-oriented CSR preimage table reproduces the same reduced
//      step as the forward source scatter for every tested width/position.
//
// Default max width is 12. Width 14 is still a quick CPU check on a desktop.

namespace {

constexpr char N = 'N', R = 'R', L = 'L';
using Word = std::string;
using Vec = std::map<Word, int64_t>;
using Rank = std::uint64_t;

struct Key {
    char type; // 'A' or 'C'
    Word w;

    bool operator<(const Key& o) const {
        return type < o.type || (type == o.type && w < o.w);
    }
    bool operator==(const Key& o) const { return type == o.type && w == o.w; }
};
using CVec = std::map<Key, int64_t>;

std::vector<Word> gen_words(int W) {
    std::vector<Word> out;
    Word cur;
    auto rec = [&](auto&& self, int pos, int h) -> void {
        const int rem = W - pos;
        if (h < 0 || h > rem) return;
        if (pos == W) {
            if (h == 0) out.push_back(cur);
            return;
        }
        cur.push_back(N);
        self(self, pos + 1, h);
        cur.pop_back();
        if (h > 0) {
            cur.push_back(R);
            self(self, pos + 1, h - 1);
            cur.pop_back();
        }
        cur.push_back(L);
        self(self, pos + 1, h + 1);
        cur.pop_back();
    };
    rec(rec, 0, 1);
    return out;
}

bool valid_word(const Word& w) {
    int h = 1;
    for (char c : w) {
        if (c == R) --h;
        else if (c == L) ++h;
        if (h < 0) return false;
    }
    return h == 0;
}

struct LinkState {
    std::vector<int> mate; // -2 vacant, -1 root
    int root = -1;
};

LinkState decode(const Word& w) {
    LinkState s;
    s.mate.assign(w.size(), -2);
    std::vector<int> st;
    st.push_back(-1);
    for (int i = 0; i < static_cast<int>(w.size()); ++i) {
        if (w[i] == N) continue;
        if (w[i] == L) {
            st.push_back(i);
        } else {
            assert(w[i] == R && !st.empty());
            const int a = st.back();
            st.pop_back();
            if (a == -1) {
                s.root = i;
                s.mate[i] = -1;
            } else {
                s.mate[a] = i;
                s.mate[i] = a;
            }
        }
    }
    assert(st.empty() && s.root >= 0);
    return s;
}

Word encode(const LinkState& s) {
    const int W = static_cast<int>(s.mate.size());
    Word w(W, N);
    for (int i = 0; i < W; ++i) {
        if (i == s.root) {
            w[i] = R;
            continue;
        }
        const int j = s.mate[i];
        if (j >= 0) w[i] = (i < j) ? L : R;
    }
    assert(valid_word(w));
    return w;
}

std::vector<Word> apply_T_basis(const Word& w, int i) {
    const int W = static_cast<int>(w.size());
    assert(i >= 0 && i + 1 < W);
    const bool a = w[i] != N, b = w[i + 1] != N;
    const LinkState s = decode(w);
    std::vector<Word> out;
    auto partner = [&](int x) { return x == s.root ? -1 : s.mate[x]; };

    if (!a && !b) {
        out.push_back(w);
        LinkState t = s;
        t.mate[i] = i + 1;
        t.mate[i + 1] = i;
        out.push_back(encode(t));
    } else if (a && !b) {
        out.push_back(w);
        LinkState t = s;
        const int p = partner(i);
        if (i == s.root) {
            t.mate[i] = -2;
            t.root = i + 1;
            t.mate[i + 1] = -1;
        } else {
            t.mate[i] = -2;
            t.mate[p] = i + 1;
            t.mate[i + 1] = p;
        }
        out.push_back(encode(t));
    } else if (!a && b) {
        LinkState t = s;
        const int p = partner(i + 1);
        if (i + 1 == s.root) {
            t.mate[i + 1] = -2;
            t.root = i;
            t.mate[i] = -1;
        } else {
            t.mate[i + 1] = -2;
            t.mate[p] = i;
            t.mate[i] = p;
        }
        out.push_back(encode(t));
        out.push_back(w);
    } else {
        const int p = partner(i), q = partner(i + 1);
        if (p == i + 1 && q == i) return out; // beta=0 loop

        LinkState t = s;
        if (i == s.root) {
            assert(q >= 0);
            t.mate[i] = -2;
            t.mate[i + 1] = -2;
            t.mate[q] = -1;
            t.root = q;
        } else if (i + 1 == s.root) {
            assert(p >= 0);
            t.mate[i] = -2;
            t.mate[i + 1] = -2;
            t.mate[p] = -1;
            t.root = p;
        } else {
            assert(p >= 0 && q >= 0 && p != q);
            t.mate[i] = -2;
            t.mate[i + 1] = -2;
            t.mate[p] = q;
            t.mate[q] = p;
        }
        out.push_back(encode(t));
    }
    return out;
}

void add(Vec& v, const Word& w, int64_t c = 1) {
    if (!c) return;
    v[w] += c;
    if (v[w] == 0) v.erase(w);
}
void add(CVec& v, const Key& k, int64_t c = 1) {
    if (!c) return;
    v[k] += c;
    if (v[k] == 0) v.erase(k);
}

Vec apply_T(const Vec& v, int i) {
    Vec out;
    for (auto const& [w, c] : v)
        for (auto const& z : apply_T_basis(w, i)) add(out, z, c);
    return out;
}

Word collapse_A(const Word& w, int i) {
    const bool a = w[i] != N, b = w[i + 1] != N;
    assert(!(a && b));
    const char s = a ? w[i] : (b ? w[i + 1] : N);
    return w.substr(0, i) + s + w.substr(i + 2);
}
Word remove_pair(const Word& w, int i) { return w.substr(0, i) + w.substr(i + 2); }

CVec R_raw_basis(const Word& w, int i) {
    CVec out;
    const bool a = w[i] != N, b = w[i + 1] != N;
    if (!a && !b) {
        add(out, {'A', collapse_A(w, i)});
        add(out, {'C', remove_pair(w, i)});
    } else if (a != b) {
        add(out, {'A', collapse_A(w, i)});
    } else {
        const auto z = apply_T_basis(w, i);
        if (!z.empty()) {
            assert(z.size() == 1);
            add(out, {'A', collapse_A(z[0], i)});
        }
    }
    return out;
}

Vec E_raw_basis(const Key& k, int i) {
    Vec out;
    if (k.type == 'C') {
        const Word z = k.w.substr(0, i) + Word() + L + R + k.w.substr(i);
        assert(valid_word(z));
        add(out, z);
        return out;
    }

    assert(k.type == 'A' && i < static_cast<int>(k.w.size()));
    const char s = k.w[i];
    if (s == N) {
        const Word z = k.w.substr(0, i) + Word() + N + N + k.w.substr(i + 1);
        assert(valid_word(z));
        add(out, z);
    } else {
        const Word z1 = k.w.substr(0, i) + Word() + s + N + k.w.substr(i + 1);
        const Word z2 = k.w.substr(0, i) + Word() + N + s + k.w.substr(i + 1);
        assert(valid_word(z1) && valid_word(z2));
        add(out, z1);
        add(out, z2);
    }
    return out;
}

Vec E_raw(const CVec& c, int i) {
    Vec out;
    for (auto const& [k, a] : c)
        for (auto const& [w, b] : E_raw_basis(k, i)) add(out, w, a * b);
    return out;
}

Key project_key(const Key& k, int i, int W) {
    if (k.type == 'A') return k;
    assert(k.type == 'C');
    if (i <= W - 3 && k.w[i] == N) {
        const Word a = k.w.substr(0, i) + Word() + L + R + k.w.substr(i + 1);
        assert(static_cast<int>(a.size()) == W - 1 && valid_word(a));
        return {'A', a};
    }
    return k;
}

CVec project(const CVec& raw, int i, int W) {
    CVec out;
    for (auto const& [k, c] : raw) add(out, project_key(k, i, W), c);
    return out;
}

std::vector<Key> q_basis(int W, int i, const std::vector<std::vector<Word>>& words) {
    std::vector<Key> q;
    for (auto const& w : words[W - 1]) q.push_back({'A', w});
    for (auto const& w : words[W - 2])
        if (i > W - 3 || w[i] != N) q.push_back({'C', w});
    return q;
}

CVec K_basis(const Key& src, int W, int i) {
    CVec raw;
    for (auto const& [w, a] : E_raw_basis(src, i)) {
        const CVec r = R_raw_basis(w, i + 1);
        for (auto const& [k, b] : r) add(raw, k, a * b);
    }
    return project(raw, i + 1, W);
}

struct ReducedLayout {
    std::vector<Key> key;
    std::map<Key, Rank> rank;
    Rank a_size = 0;

    Rank size() const { return static_cast<Rank>(key.size()); }
};

ReducedLayout make_layout(int W, int i, const std::vector<std::vector<Word>>& words) {
    ReducedLayout out;
    out.key.reserve(words[W - 1].size() + words[W - 2].size());

    for (auto const& w : words[W - 1]) out.key.push_back({'A', w});
    out.a_size = out.size();
    for (auto const& w : words[W - 2])
        if (i > W - 3 || w[i] != N) out.key.push_back({'C', w});

    for (Rank r = 0; r < out.size(); ++r) {
        if (!out.rank.emplace(out.key[r], r).second) std::abort();
        if ((r < out.a_size) != (out.key[r].type == 'A')) std::abort();
    }
    return out;
}

struct PreimageCSR {
    std::vector<Rank> offset; // destination -> [offset[d], offset[d+1])
    std::vector<Rank> source;
};

PreimageCSR build_preimage_csr(
    const ReducedLayout& src,
    const ReducedLayout& dst,
    int W,
    int i
) {
    std::vector<std::vector<Rank>> incoming(static_cast<std::size_t>(dst.size()));
    for (Rank s = 0; s < src.size(); ++s) {
        const CVec col = K_basis(src.key[s], W, i);
        for (auto const& [k, c] : col) {
            if (c != 1) std::abort();
            const auto it = dst.rank.find(k);
            if (it == dst.rank.end()) std::abort();
            incoming[it->second].push_back(s);
        }
    }

    PreimageCSR out;
    out.offset.resize(static_cast<std::size_t>(dst.size()) + 1);
    for (Rank d = 0; d < dst.size(); ++d) {
        out.offset[d] = static_cast<Rank>(out.source.size());
        auto& v = incoming[d];
        if (!std::is_sorted(v.begin(), v.end())) std::abort();
        out.source.insert(out.source.end(), v.begin(), v.end());
    }
    out.offset[dst.size()] = static_cast<Rank>(out.source.size());
    return out;
}

std::vector<std::uint64_t> scatter_step(
    const ReducedLayout& src,
    const ReducedLayout& dst,
    int W,
    int i,
    const std::vector<std::uint64_t>& value
) {
    std::vector<std::uint64_t> out(static_cast<std::size_t>(dst.size()));
    for (Rank s = 0; s < src.size(); ++s) {
        for (auto const& [k, c] : K_basis(src.key[s], W, i)) {
            assert(c == 1);
            const auto it = dst.rank.find(k);
            assert(it != dst.rank.end());
            out[it->second] += value[s];
        }
    }
    return out;
}

std::vector<std::uint64_t> gather_step(
    const PreimageCSR& pre,
    const std::vector<std::uint64_t>& value
) {
    std::vector<std::uint64_t> out(pre.offset.size() - 1);
    for (Rank d = 0; d + 1 < pre.offset.size(); ++d)
        for (Rank j = pre.offset[d]; j < pre.offset[d + 1]; ++j)
            out[d] += value[pre.source[j]];
    return out;
}

[[noreturn]] void fail(const std::string& s) {
    std::cerr << "FAIL: " << s << '\n';
    std::exit(1);
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::stoi(argv[1]) : 12;
    if (maxW < 4 || maxW > 15) {
        std::cerr << "maxW must be 4..15\n";
        return 2;
    }

    std::vector<std::vector<Word>> words(maxW + 1);
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        const uint64_t MW = words[W].size();
        const uint64_t m1 = words[W - 1].size();
        const uint64_t m2 = words[W - 2].size();
        const uint64_t m3 = words[W - 3].size();
        const uint64_t expect_dim = m1 + m2 - m3;
        const uint64_t expect_n1 = 2 * m2 - m3;
        const uint64_t expect_n2 = m1 - 2 * m2 + m3;
        const uint64_t expect_n3 = m2 - m3;
        const uint64_t expect_nnz = 2 * m1 + m2 - 2 * m3;

        for (int i = 0; i < W - 1; ++i) {
            for (auto const& w : words[W]) {
                Vec direct;
                for (auto const& z : apply_T_basis(w, i)) add(direct, z);
                const Vec fact = E_raw(R_raw_basis(w, i), i);
                if (direct != fact)
                    fail("T=ER W=" + std::to_string(W) + " i=" + std::to_string(i) + " w=" + w);
            }
        }

        uint64_t worst_indeg = 0;
        uint64_t worst_preimage_bytes = 0;
        for (int i = 0; i <= W - 4; ++i) {
            const auto qb = q_basis(W, i, words);
            const auto qn = q_basis(W, i + 1, words);
            if (qb.size() != expect_dim || qn.size() != expect_dim)
                fail("dimension formula W=" + std::to_string(W));
            const std::set<Key> qnset(qn.begin(), qn.end());

            const ReducedLayout src_layout = make_layout(W, i, words);
            const ReducedLayout dst_layout = make_layout(W, i + 1, words);
            if (src_layout.size() != expect_dim || dst_layout.size() != expect_dim)
                fail("rank layout dimension W=" + std::to_string(W));
            if (src_layout.a_size != m1 || dst_layout.a_size != m1)
                fail("A span W=" + std::to_string(W));
            if (src_layout.key != qb || dst_layout.key != qn)
                fail("rank layout order W=" + std::to_string(W));
            for (Rank r = 0; r < src_layout.size(); ++r)
                if (src_layout.rank.at(src_layout.key[r]) != r)
                    fail("rank roundtrip W=" + std::to_string(W));

            for (auto const& u : words[W - 2]) {
                if (u[i] != N) continue;
                const Key c{'C', u};
                const Key a = project_key(c, i, W);
                const Vec lhs = apply_T(E_raw_basis(c, i), i + 1);
                const Vec rhs = apply_T(E_raw_basis(a, i), i + 1);
                if (lhs != rhs)
                    fail("quotient relation W=" + std::to_string(W) + " i=" + std::to_string(i) + " u=" + u);
            }

            std::array<uint64_t, 4> fan{};
            uint64_t nnz = 0;
            std::map<Key, uint64_t> indeg;
            for (auto const& src : qb) {
                const CVec col = K_basis(src, W, i);
                if (col.empty() || col.size() > 3)
                    fail("fanout W=" + std::to_string(W) + " i=" + std::to_string(i));
                ++fan[col.size()];
                nnz += col.size();
                for (auto const& [dst, c] : col) {
                    if (c != 1)
                        fail("non-unit coefficient W=" + std::to_string(W) + " i=" + std::to_string(i));
                    if (!qnset.count(dst)) fail("destination outside Q W=" + std::to_string(W));
                    ++indeg[dst];
                }

                const Vec actual = apply_T(apply_T(E_raw_basis(src, i), i + 1), i + 2);
                Vec rep;
                for (auto const& [dst, c] : col)
                    for (auto const& [w, d] : E_raw_basis(dst, i + 1)) add(rep, w, c * d);
                const Vec reduced = apply_T(rep, i + 2);
                if (actual != reduced)
                    fail("delayed exactness W=" + std::to_string(W) + " i=" + std::to_string(i));
            }

            if (fan[1] != expect_n1 || fan[2] != expect_n2 || fan[3] != expect_n3 || nnz != expect_nnz)
                fail("fanout/nnz formula W=" + std::to_string(W) + " i=" + std::to_string(i));
            for (auto const& [k, d] : indeg) worst_indeg = std::max(worst_indeg, d);

            const PreimageCSR pre = build_preimage_csr(src_layout, dst_layout, W, i);
            if (pre.source.size() != expect_nnz || pre.offset.size() != expect_dim + 1)
                fail("preimage CSR size W=" + std::to_string(W) + " i=" + std::to_string(i));
            for (Rank d = 0; d < dst_layout.size(); ++d)
                worst_indeg = std::max(worst_indeg, pre.offset[d + 1] - pre.offset[d]);

            std::vector<std::uint64_t> value(static_cast<std::size_t>(src_layout.size()));
            for (Rank s = 0; s < src_layout.size(); ++s)
                value[s] = 1 + ((s * 0x9e3779b97f4a7c15ULL) ^ (Rank(W) << 32) ^ Rank(i));
            if (scatter_step(src_layout, dst_layout, W, i, value) != gather_step(pre, value))
                fail("scatter/gather mismatch W=" + std::to_string(W) + " i=" + std::to_string(i));

            const uint64_t bytes =
                uint64_t(pre.offset.size() + pre.source.size()) * sizeof(Rank);
            worst_preimage_bytes = std::max(worst_preimage_bytes, bytes);
        }

        std::cout << "W=" << W
                  << " M=" << MW
                  << " reduced=" << expect_dim
                  << " A=" << m1
                  << " C=" << (m2 - m3)
                  << " n1=" << expect_n1
                  << " n2=" << expect_n2
                  << " n3=" << expect_n3
                  << " nnz=" << expect_nnz
                  << " avg=" << double(expect_nnz) / double(expect_dim)
                  << " max_indeg=" << worst_indeg
                  << " preimage_kib=" << double(worst_preimage_bytes) / 1024.0
                  << " OK\n";
    }

    std::cout << "ALL_OK maxW=" << maxW << '\n';
    return 0;
}
