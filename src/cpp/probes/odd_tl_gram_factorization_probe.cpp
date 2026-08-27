#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <random>
#include <unordered_map>
#include <utility>
#include <vector>

using u64 = std::uint64_t;
static constexpr std::uint32_t MOD = 4294967291u;

static inline std::uint32_t addm(std::uint32_t a, std::uint32_t b) {
    u64 s = u64(a) + b;
    if (s >= MOD) s -= MOD;
    return std::uint32_t(s);
}
static inline std::uint32_t mulm(std::uint32_t a, std::uint32_t b) {
    return std::uint32_t((__uint128_t(a) * b) % MOD);
}
static std::uint32_t powm(std::uint32_t a, u64 e) {
    std::uint32_t r = 1;
    while (e) {
        if (e & 1) r = mulm(r, a);
        a = mulm(a, a);
        e >>= 1;
    }
    return r;
}
static std::uint32_t invm(std::uint32_t a) { return powm(a, MOD - 2u); }
static std::uint32_t negm(std::uint32_t a) { return a ? MOD - a : 0; }

struct Basis {
    int n = 0, d = 0;
    std::vector<std::uint32_t> words; // bit i = U, otherwise D
    std::unordered_map<std::uint32_t, int> rank;
};

struct Key {
    int n, d;
    bool operator==(Key const& o) const { return n == o.n && d == o.d; }
};
struct KeyHash {
    size_t operator()(Key const& k) const { return (size_t(k.n) << 8) ^ size_t(k.d); }
};
static std::unordered_map<Key, Basis, KeyHash> cache;

static Basis const& basis(int n, int d) {
    Key k{n, d};
    auto it = cache.find(k);
    if (it != cache.end()) return it->second;

    Basis b;
    b.n = n;
    b.d = d;
    if (n == 0) {
        if (d == 0) b.words.push_back(0);
    } else if (n >= 2 && d >= 0 && d <= n && ((n - d) & 1) == 0) {
        auto append = [&](int child_d, int s0, int s1) {
            if (child_d < 0 || child_d > n - 2) return;
            auto const& child = basis(n - 2, child_d);
            for (auto w : child.words) {
                if (s0) w |= 1u << (n - 2);
                if (s1) w |= 1u << (n - 1);
                b.words.push_back(w);
            }
        };
        // Two-step odd-TL branching order: [UU, UD, DU, DD].
        append(d - 2, 1, 1);
        append(d, 1, 0);
        append(d, 0, 1);
        append(d + 2, 0, 0);
    } else if (n == 1 && d == 1) {
        b.words.push_back(1);
    }

    for (int i = 0; i < int(b.words.size()); ++i) b.rank[b.words[i]] = i;
    auto [jt, ok] = cache.emplace(k, std::move(b));
    (void)ok;
    return jt->second;
}

static std::vector<int> defects(std::uint32_t w, int n) {
    std::vector<int> stack;
    for (int i = 0; i < n; ++i) {
        if ((w >> i) & 1u) {
            stack.push_back(i);
        } else {
            if (stack.empty()) std::abort();
            stack.pop_back();
        }
    }
    return stack;
}

static std::vector<int> mates(std::uint32_t w, int n) {
    std::vector<int> stack, mate(n, -1);
    for (int i = 0; i < n; ++i) {
        if ((w >> i) & 1u) {
            stack.push_back(i);
        } else {
            int a = stack.back();
            stack.pop_back();
            mate[a] = i;
            mate[i] = a;
        }
    }
    return mate;
}

// Gram form at beta=0. Each connected overlay component must be one propagating
// path with exactly one defect from each link state. A closed component is a loop
// and therefore has weight beta=0.
static int gram01(std::uint32_t a, std::uint32_t b, int n) {
    auto ma = mates(a, n);
    auto mb = mates(b, n);
    std::vector<char> seen(n, 0);
    for (int s = 0; s < n; ++s) {
        if (seen[s]) continue;
        std::vector<int> todo{s};
        seen[s] = 1;
        int a_defects = 0, b_defects = 0;
        while (!todo.empty()) {
            int u = todo.back();
            todo.pop_back();
            if (ma[u] < 0) {
                ++a_defects;
            } else if (!seen[ma[u]]) {
                seen[ma[u]] = 1;
                todo.push_back(ma[u]);
            }
            if (mb[u] < 0) {
                ++b_defects;
            } else if (!seen[mb[u]]) {
                seen[mb[u]] = 1;
                todo.push_back(mb[u]);
            }
        }
        if (a_defects != 1 || b_defects != 1) return 0;
    }
    return 1;
}

static int map_cup_word(std::uint32_t w, int n, int d, int ia, int ib) {
    auto z = defects(w, n);
    if (int(z.size()) != d || ia < 0 || ib < 0 || ia >= d || ib >= d || ia >= ib) return -1;
    w &= ~(1u << z[ib]);
    auto const& out = basis(n, d - 2);
    auto it = out.rank.find(w);
    return it == out.rank.end() ? -1 : it->second;
}

static int map_two_cups(std::uint32_t w, int n, int d,
                        std::pair<int, int> p1, std::pair<int, int> p2) {
    auto z = defects(w, n);
    if (int(z.size()) != d) return -1;
    w &= ~(1u << z[p1.second]);
    w &= ~(1u << z[p2.second]);
    auto const& out = basis(n, d - 4);
    auto it = out.rank.find(w);
    return it == out.rank.end() ? -1 : it->second;
}

static void add_scaled(std::vector<std::uint32_t>& dst, int j,
                       std::uint32_t c, std::uint32_t x) {
    dst[j] = addm(dst[j], mulm(c, x));
}

// Alternating adjacent-defect cup map partial_d: W_n^d -> W_n^{d-2}.
static void apply_partial(int n, int d,
                          std::vector<std::uint32_t> const& src,
                          std::vector<std::uint32_t>& dst) {
    if (d < 3) return;
    auto const& B = basis(n, d);
    int r = (d - 3) / 2;
    for (int i = 0; i < int(B.words.size()); ++i) {
        auto x = src[i];
        if (!x) continue;
        for (int j = 0; j <= (d - 3) / 2; ++j) {
            int to = map_cup_word(B.words[i], n, d, 2 * j, 2 * j + 1);
            assert(to >= 0);
            std::uint32_t c = ((r - j) & 1) ? MOD - 1 : 1;
            add_scaled(dst, to, c, x);
        }
    }
}

// Rightmost adjacent-defect cup map J_d: W_n^d -> W_n^{d-2}.
static void apply_J(int n, int d,
                    std::vector<std::uint32_t> const& src,
                    std::vector<std::uint32_t>& dst) {
    if (d < 2) return;
    auto const& B = basis(n, d);
    for (int i = 0; i < int(B.words.size()); ++i) {
        int to = map_cup_word(B.words[i], n, d, d - 2, d - 1);
        assert(to >= 0);
        add_scaled(dst, to, 1, src[i]);
    }
}

// Two-cup correction Q_d: W_n^{d+2} -> W_n^{d-2}.
static void apply_Q(int n, int d,
                    std::vector<std::uint32_t> const& src,
                    std::vector<std::uint32_t>& dst) {
    int din = d + 2;
    int r = (d - 1) / 2;
    if (r == 0) return;
    auto const& B = basis(n, din);
    std::uint32_t den_inv = invm(std::uint32_t(r + 1));

    for (int idx = 0; idx < int(B.words.size()); ++idx) {
        auto x = src[idx];
        if (!x) continue;

        // Nested terms: (z_s,z_{s+3}) and (z_{s+1},z_{s+2}).
        for (int s = 0; s < 2 * r; ++s) {
            int to = map_two_cups(B.words[idx], n, din,
                                  {s, s + 3}, {s + 1, s + 2});
            assert(to >= 0);
            int num = s / 2 + 1;
            std::uint32_t c = mulm(std::uint32_t(num), den_inv);
            if (s & 1) c = negm(c);
            add_scaled(dst, to, c, x);
        }

        // Separated terms: (z_{2i},z_{2i+1}) and (z_{2j+1},z_{2j+2}).
        for (int i = 0; i <= r; ++i) {
            for (int j = i + 1; j <= r; ++j) {
                int to = map_two_cups(B.words[idx], n, din,
                                      {2 * i, 2 * i + 1}, {2 * j + 1, 2 * j + 2});
                assert(to >= 0);
                int num = 1;
                int sign_exp = j - i;
                if (j == r) {
                    num = r;
                    sign_exp = r - 1 - i;
                }
                std::uint32_t c = mulm(std::uint32_t(num), den_inv);
                if (sign_exp & 1) c = negm(c);
                add_scaled(dst, to, c, x);
            }
        }
    }
}

static void transform(int n, int d, std::vector<std::uint32_t>& v) {
    if (n <= 1) return;
    int da = int(basis(n - 2, d - 2).words.size());
    int d0 = int(basis(n - 2, d).words.size());
    int dp = int(basis(n - 2, d + 2).words.size());
    assert(int(v.size()) == da + 2 * d0 + dp);

    std::vector<std::uint32_t> A(v.begin(), v.begin() + da);
    std::vector<std::uint32_t> B(v.begin() + da, v.begin() + da + d0);
    std::vector<std::uint32_t> C(v.begin() + da + d0, v.begin() + da + 2 * d0);
    std::vector<std::uint32_t> D(v.begin() + da + 2 * d0, v.end());

    if (da) {
        std::vector<std::uint32_t> t(da);
        apply_partial(n - 2, d, C, t);
        for (int i = 0; i < da; ++i) A[i] = addm(A[i], t[i]);
        std::fill(t.begin(), t.end(), 0);
        apply_Q(n - 2, d, D, t);
        for (int i = 0; i < da; ++i) A[i] = addm(A[i], t[i]);
    }
    if (d0 && dp) {
        std::vector<std::uint32_t> t(d0);
        apply_partial(n - 2, d + 2, D, t);
        for (int i = 0; i < d0; ++i) B[i] = addm(B[i], t[i]);
        std::fill(t.begin(), t.end(), 0);
        apply_J(n - 2, d + 2, D, t);
        for (int i = 0; i < d0; ++i) C[i] = addm(C[i], t[i]);
    }

    if (da) transform(n - 2, d - 2, A);
    if (d0) {
        transform(n - 2, d, B);
        transform(n - 2, d, C);
    }
    if (dp) transform(n - 2, d + 2, D);

    int off = 0;
    for (auto x : A) v[off++] = x;
    for (auto x : B) v[off++] = x;
    for (auto x : C) v[off++] = x;
    for (auto x : D) v[off++] = x;
}

static std::uint32_t canonical_bilinear(
    int n, int d,
    std::vector<std::uint32_t> const& x, int xo,
    std::vector<std::uint32_t> const& y, int yo
) {
    if (n == 0) return d == 0 ? mulm(x[xo], y[yo]) : 0;
    if (n == 1) return d == 1 ? mulm(x[xo], y[yo]) : 0;

    int da = int(basis(n - 2, d - 2).words.size());
    int d0 = int(basis(n - 2, d).words.size());
    int dp = int(basis(n - 2, d + 2).words.size());
    std::uint32_t z = 0;

    if (da) z = addm(z, canonical_bilinear(n - 2, d - 2, x, xo, y, yo));
    if (d0) {
        z = addm(z, canonical_bilinear(n - 2, d,
                                       x, xo + da,
                                       y, yo + da + d0));
        z = addm(z, canonical_bilinear(n - 2, d,
                                       x, xo + da + d0,
                                       y, yo + da));
    }
    if (dp) {
        auto q = canonical_bilinear(n - 2, d + 2,
                                    x, xo + da + 2 * d0,
                                    y, yo + da + 2 * d0);
        std::uint32_t c = negm(mulm(std::uint32_t(d + 3), invm(std::uint32_t(d + 1))));
        z = addm(z, mulm(c, q));
    }
    return z;
}

static std::uint32_t explicit_bilinear(
    int n, int d,
    std::vector<std::uint32_t> const& x,
    std::vector<std::uint32_t> const& y
) {
    auto const& B = basis(n, d);
    std::uint32_t z = 0;
    for (int i = 0; i < int(B.words.size()); ++i) {
        if (!x[i]) continue;
        for (int j = 0; j < int(B.words.size()); ++j) {
            if (!y[j] || !gram01(B.words[i], B.words[j], n)) continue;
            z = addm(z, mulm(x[i], y[j]));
        }
    }
    return z;
}

int main() {
    std::mt19937_64 rng(1234567);
    for (int n = 1; n <= 13; n += 2) {
        for (int d = 1; d <= n; d += 2) {
            int dim = int(basis(n, d).words.size());
            if (!dim) continue;
            for (int trial = 0; trial < 4; ++trial) {
                std::vector<std::uint32_t> x(dim), y(dim);
                for (auto& v : x) v = std::uint32_t(rng() % MOD);
                for (auto& v : y) v = std::uint32_t(rng() % MOD);

                auto explicit_value = explicit_bilinear(n, d, x, y);
                transform(n, d, x);
                transform(n, d, y);
                auto factorized_value = canonical_bilinear(n, d, x, 0, y, 0);
                if (explicit_value != factorized_value) {
                    std::cerr << "MISMATCH n=" << n << " d=" << d
                              << " dim=" << dim << " trial=" << trial
                              << " explicit=" << explicit_value
                              << " factorized=" << factorized_value << "\n";
                    return 1;
                }
            }
            std::cout << "OK n=" << n << " d=" << d << " dim=" << dim << "\n";
        }
    }
    return 0;
}
