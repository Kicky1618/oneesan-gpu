#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <vector>

using u64 = std::uint64_t;

static u64 binom(int n, int k) {
    if (k < 0 || k > n) return 0;
    if (k > n-k) k = n-k;
    u64 r = 1;
    for (int i = 1; i <= k; ++i) r = r * u64(n-k+i) / u64(i);
    return r;
}

static u64 catalan(int n) {
    return binom(2*n,n) / u64(n+1);
}

struct PathDP {
    int r = 0;
    std::vector<std::vector<u64>> pref;   // pref[pos][height]
    std::vector<std::vector<u64>> suffix; // suffix[len][height] -> end at 0
};

static PathDP make_path_dp(int r) {
    PathDP z;
    z.r = r;
    z.pref.assign(r+1, std::vector<u64>(r+3));
    z.suffix.assign(r+1, std::vector<u64>(r+3));
    z.pref[0][0] = 1;
    for (int p = 0; p < r; ++p) {
        for (int h = 0; h <= r; ++h) {
            u64 x = z.pref[p][h];
            if (!x) continue;
            if (h) z.pref[p+1][h-1] += x; // A
            z.pref[p+1][h] += 2*x;         // B,C
            z.pref[p+1][h+1] += x;         // D
        }
    }
    z.suffix[0][0] = 1;
    for (int len = 1; len <= r; ++len) {
        for (int h = 0; h <= r+1; ++h) {
            u64 x = 2*z.suffix[len-1][h];
            if (h) x += z.suffix[len-1][h-1];
            x += z.suffix[len-1][h+1];
            z.suffix[len][h] = x;
        }
    }
    assert(z.pref[r][0] == catalan(r+1));
    return z;
}

static u64 insert_outputs_sum(int r, int dense_pos) {
    const u64 dim = catalan(r+1);
    // Boundary insertion and insertion between the two members of a fusion pair
    // are deterministic in the two-colour Motzkin fusion basis.
    if (dense_pos == 0 || (dense_pos & 1)) return dim;

    const int j = dense_pos / 2;
    const int k = r - j;
    assert(0 <= k && k < r);
    const auto dp = make_path_dp(r);
    const int rem = r-k-1;
    u64 out = 0;
    for (int h = 0; h <= r; ++h) {
        const u64 pre = dp.pref[k][h];
        if (!pre) continue;
        if (h) out += 2*pre*dp.suffix[rem][h-1]; // A -> AC+CA
        const u64 horizontal = pre*dp.suffix[rem][h];
        out += (h ? 4 : 3)*horizontal;            // B
        out += horizontal;                         // C -> CC
        out += 2*pre*dp.suffix[rem][h+1];         // D -> CD+DC
    }
    return out;
}

static int dh(char c) {
    return c == 'A' ? -1 : c == 'D' ? +1 : 0;
}

static u64 cap_survivors(int r, int dense_pos) {
    assert(r >= 1);
    const auto dp = make_path_dp(r);

    if (dense_pos == 0) {
        // Boundary cap requires the last symbol B.
        return dp.pref[r-1][0];
    }
    if (dense_pos & 1) {
        // Within a fusion pair the removed symbol must be C.
        const int j = (dense_pos-1)/2;
        const int k = (r-1)-j;
        const int rem = r-k-1;
        u64 out = 0;
        for (int h = 0; h <= r; ++h)
            out += dp.pref[k][h] * dp.suffix[rem][h];
        return out;
    }

    // Cross-pair merge.  These are exactly the nonzero local cap rules.
    static constexpr std::array<const char*,9> allowed = {
        "AB","BA","BB","AD","BC","CB","DA","BD","DB"
    };
    const int j = dense_pos/2;
    const int k = (r-1)-j;
    const int rem = r-k-2;
    assert(rem >= 0);
    u64 out = 0;
    for (int h = 0; h <= r; ++h) {
        const u64 pre = dp.pref[k][h];
        if (!pre) continue;
        for (auto pat : allowed) {
            int hh = h;
            bool ok = true;
            for (int q = 0; q < 2; ++q) {
                hh += dh(pat[q]);
                if (hh < 0) { ok = false; break; }
            }
            if (ok) out += pre * dp.suffix[rem][hh];
        }
    }
    return out;
}

static u64 dense_dim(int occupied) {
    if (occupied <= 0 || !(occupied & 1)) return 0;
    return catalan((occupied+1)/2);
}

struct Work {
    u64 states = 0;
    u64 nn_raw = 0;
    u64 nn_fusion = 0;
    u64 move = 0;
    u64 cap = 0;
    u64 raw = 0;
    u64 fusion = 0;
};

static Work work_at_p(int W, int p) {
    const int below = p-1;
    const int above = W-p-1;
    Work z;

    // 00: NN / adjacent arc insertion.
    for (int i = 0; i <= below; ++i) for (int j = 0; j <= above; ++j) {
        const u64 masks = binom(below,i) * binom(above,j);
        const int m = i+j;
        if (!(m & 1)) continue;
        const int r = (m-1)/2;
        const u64 d = dense_dim(m);
        z.states += masks*d;
        z.nn_raw += masks*d;
        z.nn_fusion += masks*insert_outputs_sum(r,i);
    }

    // 01 / 10: endpoint-vacancy motion.  Fusion topology is unchanged.
    for (int which = 0; which < 2; ++which) {
        (void)which;
        for (int i = 0; i <= below; ++i) for (int j = 0; j <= above; ++j) {
            const u64 masks = binom(below,i) * binom(above,j);
            const int m = i+j+1;
            if (!(m & 1)) continue;
            const u64 d = dense_dim(m);
            z.states += masks*d;
            z.move += masks*d;
        }
    }

    // 11: cap.  Raw diagram basis and fusion basis have the same number of
    // nonzero included branches; fusion merely makes the operation local.
    for (int i = 0; i <= below; ++i) for (int j = 0; j <= above; ++j) {
        const u64 masks = binom(below,i) * binom(above,j);
        const int m = i+j+2;
        if (!(m & 1)) continue;
        const int r = (m-1)/2;
        const u64 d = dense_dim(m);
        z.states += masks*d;
        z.cap += masks*cap_survivors(r,i);
    }

    z.raw = z.nn_raw + z.move + z.cap;
    z.fusion = z.nn_fusion + z.move + z.cap;
    return z;
}

int main(int argc, char** argv) {
    int W = argc > 1 ? std::atoi(argv[1]) : 28;
    if (W < 2 || W > 28) return 2;

    u64 raw = 0, fusion = 0, nn_raw = 0, nn_fusion = 0, move = 0, cap = 0;
    u64 full_states = 0;
    for (int m = 1; m <= W; m += 2)
        full_states += binom(W,m) * dense_dim(m);

    std::cout << "W=" << W << " main_states=" << full_states << "\n";
    std::cout << std::fixed << std::setprecision(6);
    for (int p = 1; p < W; ++p) {
        Work z = work_at_p(W,p);
        assert(z.states == full_states);
        raw += z.raw;
        fusion += z.fusion;
        nn_raw += z.nn_raw;
        nn_fusion += z.nn_fusion;
        move += z.move;
        cap += z.cap;
        std::cout << "p=" << p
                  << " raw_included=" << z.raw
                  << " fusion_included=" << z.fusion
                  << " ratio=" << (double(z.fusion)/double(z.raw))
                  << " extra=" << (z.fusion-z.raw) << "\n";
    }

    const u64 excluded = full_states * u64(W-1);
    std::cout << "SUMMARY\n"
              << "  nn_raw=" << nn_raw << "\n"
              << "  nn_fusion=" << nn_fusion << "\n"
              << "  move=" << move << "\n"
              << "  cap=" << cap << "\n"
              << "  raw_included=" << raw << "\n"
              << "  fusion_included=" << fusion << "\n"
              << "  included_ratio=" << (double(fusion)/double(raw)) << "\n"
              << "  excluded_identity=" << excluded << "\n"
              << "  raw_all_edges=" << (raw+excluded) << "\n"
              << "  fusion_all_edges=" << (fusion+excluded) << "\n"
              << "  all_edge_ratio=" << (double(fusion+excluded)/double(raw+excluded)) << "\n";

    if (W == 28) {
        assert(full_states == 385719506620ULL);
        assert(nn_raw == 1278124766802ULL);
        assert(nn_fusion == 1906681380330ULL);
        assert(move == 4734587758374ULL);
        assert(cap == 1906681380330ULL);
        assert(raw == 7919393905506ULL);
        assert(fusion == 8547950519034ULL);
        assert(excluded == 10414426678740ULL);
    }
    return 0;
}
