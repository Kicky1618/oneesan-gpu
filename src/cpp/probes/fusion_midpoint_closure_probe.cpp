#define main odd_tl_probe_original_main
#include "odd_tl_gram_factorization_probe.cpp"
#undef main

#include <iomanip>

static std::uint32_t subm2(std::uint32_t a, std::uint32_t b) {
    return addm(a, negm(b));
}

// Inverse of transform().  Forward transform first applies the local
// upper-triangular [A|B|C|D] shear and then recursively transforms the four
// children.  Inversion therefore recursively undoes the children first and
// then reverses the local shear.
static void inverse_transform(int n, int d, std::vector<std::uint32_t>& v) {
    if (n <= 1) return;
    int da = int(basis(n - 2, d - 2).words.size());
    int d0 = int(basis(n - 2, d).words.size());
    int dp = int(basis(n - 2, d + 2).words.size());
    assert(int(v.size()) == da + 2 * d0 + dp);

    std::vector<std::uint32_t> A(v.begin(), v.begin() + da);
    std::vector<std::uint32_t> B(v.begin() + da, v.begin() + da + d0);
    std::vector<std::uint32_t> C(v.begin() + da + d0, v.begin() + da + 2 * d0);
    std::vector<std::uint32_t> D(v.begin() + da + 2 * d0, v.end());

    if (da) inverse_transform(n - 2, d - 2, A);
    if (d0) {
        inverse_transform(n - 2, d, B);
        inverse_transform(n - 2, d, C);
    }
    if (dp) inverse_transform(n - 2, d + 2, D);

    // C0 = C1 - J D, B0 = B1 - partial D.
    if (d0 && dp) {
        std::vector<std::uint32_t> t(d0);
        apply_J(n - 2, d + 2, D, t);
        for (int i = 0; i < d0; ++i) C[i] = subm2(C[i], t[i]);
        std::fill(t.begin(), t.end(), 0);
        apply_partial(n - 2, d + 2, D, t);
        for (int i = 0; i < d0; ++i) B[i] = subm2(B[i], t[i]);
    }

    // A0 = A1 - partial(C0) - Q(D).
    if (da) {
        std::vector<std::uint32_t> t(da);
        apply_partial(n - 2, d, C, t);
        for (int i = 0; i < da; ++i) A[i] = subm2(A[i], t[i]);
        std::fill(t.begin(), t.end(), 0);
        apply_Q(n - 2, d, D, t);
        for (int i = 0; i < da; ++i) A[i] = subm2(A[i], t[i]);
    }

    int off = 0;
    for (auto x : A) v[off++] = x;
    for (auto x : B) v[off++] = x;
    for (auto x : C) v[off++] = x;
    for (auto x : D) v[off++] = x;
}

// Apply the canonical Gram operator D in the transformed basis.  The B/C
// sectors form the hyperbolic block H \otimes D0; the D-sector receives the
// root-of-unity scalar -(d+3)/(d+1).
static void apply_canonical_D(int n, int d, std::vector<std::uint32_t>& v) {
    if (n <= 1) return;
    int da = int(basis(n - 2, d - 2).words.size());
    int d0 = int(basis(n - 2, d).words.size());
    int dp = int(basis(n - 2, d + 2).words.size());
    assert(int(v.size()) == da + 2 * d0 + dp);

    std::vector<std::uint32_t> A(v.begin(), v.begin() + da);
    std::vector<std::uint32_t> B(v.begin() + da, v.begin() + da + d0);
    std::vector<std::uint32_t> C(v.begin() + da + d0, v.begin() + da + 2 * d0);
    std::vector<std::uint32_t> D(v.begin() + da + 2 * d0, v.end());

    if (da) apply_canonical_D(n - 2, d - 2, A);
    if (d0) {
        // H swaps the two identical W^{d} copies before the child Gram.
        std::swap(B, C);
        apply_canonical_D(n - 2, d, B);
        apply_canonical_D(n - 2, d, C);
    }
    if (dp) {
        apply_canonical_D(n - 2, d + 2, D);
        std::uint32_t c = negm(mulm(std::uint32_t(d + 3), invm(std::uint32_t(d + 1))));
        for (auto& x : D) x = mulm(c, x);
    }

    int off = 0;
    for (auto x : A) v[off++] = x;
    for (auto x : B) v[off++] = x;
    for (auto x : C) v[off++] = x;
    for (auto x : D) v[off++] = x;
}

static std::uint32_t reflect_word(std::uint32_t w, int n) {
    auto m = mates(w, n);
    std::uint32_t z = 0;
    for (int i = 0; i < n; ++i) {
        int ri = n - 1 - i;
        if (m[i] < 0) {
            // A propagating defect remains a defect after horizontal reflection.
            z |= std::uint32_t(1) << ri;
        } else if (i < m[i]) {
            int a = n - 1 - m[i];
            int b = n - 1 - i;
            assert(a < b);
            // Ballot-word convention: U at the left end of every cup.
            z |= std::uint32_t(1) << a;
        }
    }
    return z;
}

static void apply_reflection(int n, int d, std::vector<std::uint32_t>& v) {
    auto const& B = basis(n, d);
    std::vector<std::uint32_t> out(v.size());
    for (int i = 0; i < int(B.words.size()); ++i) {
        if (!v[i]) continue;
        auto rw = reflect_word(B.words[i], n);
        auto it = B.rank.find(rw);
        assert(it != B.rank.end());
        out[it->second] = addm(out[it->second], v[i]);
    }
    v.swap(out);
}

// Apply K = D * T * R * T^{-1} to a vector already expressed in the fusion
// basis.  This is exactly the point-reflection midpoint closure operator for one
// dense occupied sector.
static void apply_K(int n, std::vector<std::uint32_t>& v) {
    inverse_transform(n, 1, v);
    apply_reflection(n, 1, v);
    transform(n, 1, v);
    apply_canonical_D(n, 1, v);
}

static std::vector<unsigned long long> secondary_structure_numbers(int nmax) {
    // OEIS A004148: a(n+1)=a(n)+sum_{k=1}^{n-1} a(k)a(n-1-k).
    std::vector<unsigned long long> a(nmax + 1, 0);
    a[0] = 1;
    if (nmax >= 1) a[1] = 1;
    for (int n = 1; n < nmax; ++n) {
        unsigned long long s = a[n];
        for (int k = 1; k <= n - 1; ++k) s += a[k] * a[n - 1 - k];
        a[n + 1] = s;
    }
    return a;
}

static unsigned long long binom_u64(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n-k) k = n-k;
    unsigned long long z = 1;
    for (int i = 1; i <= k; ++i) z = z * unsigned(n-k+i) / unsigned(i);
    return z;
}

int main(int argc, char** argv) {
    int maxn = argc > 1 ? std::atoi(argv[1]) : 15;
    if (maxn > 15) maxn = 15; // uint32 ballot words and probe run time are ample here.
    auto a = secondary_structure_numbers(28);

    std::cout << "fusion midpoint closure sparsity modulo " << MOD << "\n";
    for (int n = 1; n <= maxn; n += 2) {
        int dim = int(basis(n, 1).words.size());
        unsigned long long nnz = 0;
        int max_col = 0;
        for (int j = 0; j < dim; ++j) {
            std::vector<std::uint32_t> v(dim, 0);
            v[j] = 1;
            apply_K(n, v);
            int c = 0;
            for (auto x : v) if (x) ++c;
            nnz += unsigned(c);
            max_col = std::max(max_col, c);
        }
        std::cout << "n=" << std::setw(2) << n
                  << " dim=" << std::setw(8) << dim
                  << " nnz=" << std::setw(12) << nnz
                  << " A004148=" << std::setw(12) << a[n]
                  << " avg=" << std::fixed << std::setprecision(6)
                  << (double(nnz) / double(dim))
                  << " max_col=" << max_col
                  << (nnz == a[n] ? " OK" : " MISMATCH") << "\n";
        if (nnz != a[n]) return 1;
    }

    // Width-28 dilute midpoint work prediction if every odd occupancy sector is
    // contracted by the dense K_m support.  Point reflection has no fixed odd
    // occupancy mask, so an implementation can process mask pairs once and halve
    // this number.
    unsigned long long total = 0;
    for (int m = 1; m <= 27; m += 2) total += binom_u64(28, m) * a[m];
    std::cout << "W=28 directed_closure_edges=" << total << "\n";
    std::cout << "W=28 reflected_mask_pair_edges=" << total / 2 << "\n";
    assert(total == 23977709765604ULL);
    return 0;
}
