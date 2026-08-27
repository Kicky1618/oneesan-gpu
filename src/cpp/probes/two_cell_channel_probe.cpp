#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
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
// The resulting two-cell channel has dimension
//
//     M_{W-1} + M_{W-2} - M_{W-3},
//
// and the reduced transition has source fan-out at most three, with
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
//      T gives exactly the same full vector as two consecutive full T steps.
//
// Default max width is 12. Width 14 is still a quick CPU check on a desktop.

namespace {

constexpr char N = 'N', R = 'R', L = 'L';
using Word = std::string;
using Vec = std::map<Word, int64_t>;

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
        // vacancy and cup
        out.push_back(w);
        LinkState t = s;
        t.mate[i] = i + 1;
        t.mate[i + 1] = i;
        out.push_back(encode(t));
    } else if (a && !b) {
        // straight and turn
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
        // turn and straight
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
        // cap; an already paired adjacent LR creates a beta=0 loop.
        const int p = partner(i), q = partner(i + 1);
        if (p == i + 1 && q == i) return out;

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

    // Quotient for the next forward cell (i+1,i+2):
    // C_i(Nq) and A_i(LRq) have the same image under T_{i+1}.
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
    // Q_i -> Q_{i+1}: P_{i+1} R_{i+1} E_i.
    CVec raw;
    for (auto const& [w, a] : E_raw_basis(src, i)) {
        const CVec r = R_raw_basis(w, i + 1);
        for (auto const& [k, b] : r) add(raw, k, a * b);
    }
    return project(raw, i + 1, W);
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

        // 1. Exact local rank factorization T_i = E_i R_i.
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
        for (int i = 0; i <= W - 4; ++i) {
            const auto qb = q_basis(W, i, words);
            const auto qn = q_basis(W, i + 1, words);
            if (qb.size() != expect_dim || qn.size() != expect_dim)
                fail("dimension formula W=" + std::to_string(W));
            const std::set<Key> qnset(qn.begin(), qn.end());

            // 2. Every eliminated C direction is killed by the next cell.
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

                // 3/4. One reduced step may change the representative by a
                // kernel vector of T_{i+2}; after applying T_{i+2}, the full
                // vectors must therefore agree exactly.
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
        }

        std::cout << "W=" << W
                  << " M=" << MW
                  << " reduced=" << expect_dim
                  << " n1=" << expect_n1
                  << " n2=" << expect_n2
                  << " n3=" << expect_n3
                  << " nnz=" << expect_nnz
                  << " avg=" << double(expect_nnz) / double(expect_dim)
                  << " max_indeg=" << worst_indeg
                  << " OK\n";
    }

    std::cout << "ALL_OK maxW=" << maxW << '\n';
    return 0;
}
