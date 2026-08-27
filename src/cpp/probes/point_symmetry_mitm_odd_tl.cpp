#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main

// Reuse the independently validated odd-TL factorization, but keep its modular
// helpers out of the namespace of this probe.
#define main odd_tl_factorization_unused_main
#define addm tl_addm
#define mulm tl_mulm
#define powm tl_powm
#define invm tl_invm
#define negm tl_negm
#include "odd_tl_gram_factorization_probe.cpp"
#undef negm
#undef invm
#undef powm
#undef mulm
#undef addm
#undef main

struct Rec {
    uint32_t mask = 0;
    uint32_t rmask = 0;
    int k = 0;
    std::string top;
    std::string bot;
    int top_rank = -1;
    int bot_rank = -1;
    uint32_t val = 0;
};

static std::vector<Mate> code_states(MateCodec const& mc) {
    std::vector<Mate> out(mc.codeSize());
    for (Code bi = 0; bi < mc.codeSizeL(); ++bi) {
        auto const& b = mc.codeTable(bi);
        for (Code i = 0; i < b.size; ++i) out[b.base + i] = b.mateL | b.mateR[i];
    }
    return out;
}

static uint32_t occ(Mate m, int W) {
    uint32_t z = 0;
    for (int p = 0; p < W; ++p) if (m.get(p) != N) z |= 1u << p;
    return z;
}

static uint32_t revbits(uint32_t x, int W) {
    uint32_t z = 0;
    for (int p = 0; p < W; ++p) if ((x >> p) & 1u) z |= 1u << (W - 1 - p);
    return z;
}

static std::vector<std::pair<int, int>> raw_pairing(Mate m, int W) {
    std::vector<int> st{W};
    std::vector<std::pair<int, int>> e;
    for (int p = W - 1; p >= 0; --p) {
        auto v = m.get(p);
        if (v == L) {
            st.push_back(p);
        } else if (v == R) {
            int q = st.back();
            st.pop_back();
            e.push_back({q, p});
        }
    }
    return e;
}

// Signature labels are 1..k in high-to-low frontier order; 0 is the defect.
static std::string sig_top(Mate m, int W) {
    uint32_t om = occ(m, W);
    std::vector<int> idx(W, -1);
    int n = 1;
    for (int p = W - 1; p >= 0; --p) if ((om >> p) & 1u) idx[p] = n++;
    std::string s(n, '\0');
    for (auto [u, v] : raw_pairing(m, W)) {
        int a = u == W ? 0 : idx[u];
        int b = v == W ? 0 : idx[v];
        s[a] = char(b);
        s[b] = char(a);
    }
    return s;
}

static std::string sig_bottom_reflected(Mate m, int W) {
    uint32_t om = revbits(occ(m, W), W);
    std::vector<int> idx(W, -1);
    int n = 1;
    for (int p = W - 1; p >= 0; --p) if ((om >> p) & 1u) idx[p] = n++;
    std::string s(n, '\0');
    for (auto [u, v] : raw_pairing(m, W)) {
        int ru = u == W ? -1 : W - 1 - u;
        int rv = v == W ? -1 : W - 1 - v;
        int a = ru < 0 ? 0 : idx[ru];
        int b = rv < 0 ? 0 : idx[rv];
        s[a] = char(b);
        s[b] = char(a);
    }
    return s;
}

static bool compat(std::string const& a, std::string const& b) {
    int n = int(a.size());
    if (int(b.size()) != n || n == 0) return false;
    std::vector<uint8_t> seen(n);
    std::vector<int> st{0};
    seen[0] = 1;
    int got = 0;
    while (!st.empty()) {
        int u = st.back();
        st.pop_back();
        ++got;
        int vs[2] = {(unsigned char)a[u], (unsigned char)b[u]};
        for (int v : vs) if (!seen[v]) {
            seen[v] = 1;
            st.push_back(v);
        }
    }
    return got == n;
}

// Convert a pairing signature to the odd-TL ballot word used by basis(k,1).
// basis() scans terminals low-to-high. Signature labels scan high-to-low, so
// terminal i corresponds to signature label k-i. It is U iff it is the defect
// or its partner appears later in low-to-high order.
static uint32_t signature_word(std::string const& s) {
    int k = int(s.size()) - 1;
    uint32_t w = 0;
    for (int i = 0; i < k; ++i) {
        int label = k - i;
        int partner = (unsigned char)s[label];
        bool up = partner == 0 || label > partner;
        if (up) w |= 1u << i;
    }
    return w;
}

static int signature_rank(std::string const& s) {
    int k = int(s.size()) - 1;
    auto const& B = basis(k, 1);
    auto it = B.rank.find(signature_word(s));
    return it == B.rank.end() ? -1 : it->second;
}

static void init_half(PathCounter<Modnum<uint64_t>>& pc) {
    for (Code i = 0; i < pc.mc.codeSize(); ++i) pc.value[i] = 0;
    for (Code i = 0; i < pc.wc.codeSize(); ++i) pc.deferred[i] = 0;
    pc.value[pc.mc.encode(Mate(pc.cols - 1, R))] = 1;
}

static void run_rows(PathCounter<Modnum<uint64_t>>& pc, int nr) {
    for (int i = 0; i < nr; ++i) {
        for (int j = 0; j < pc.cols - 2; ++j) pc.update(j, false);
        pc.update(pc.cols - 2, false);
    }
}

static uint32_t from64(uint64_t x) { return uint32_t(x % MOD); }

struct Group {
    int k = 0;
    std::vector<uint32_t> x_tl, y_tl;
    std::vector<uint32_t> x_sig, y_sig;
};

int main(int argc, char** argv) {
    msg = NONE;
    modulus = MOD;
    int maxW = argc > 1 ? std::atoi(argv[1]) : 12;

    for (int W = 2; W <= maxW; W += 2) {
        PathCounter<Modnum<uint64_t>> half(W, W, false, false);
        init_half(half);
        run_rows(half, W / 2);
        auto states = code_states(half.mc);

        std::vector<Rec> recs;
        recs.reserve(states.size());
        std::vector<std::unordered_map<std::string, int>> sig_ids(W + 1);
        std::vector<std::vector<std::string>> sigs(W + 1);

        for (Code i = 0; i < half.mc.codeSize(); ++i) {
            uint64_t raw = half.value[i];
            if (!raw) continue;
            Mate m = states[i];
            uint32_t mask = occ(m, W);
            int k = __builtin_popcount(mask);
            auto top = sig_top(m, W);
            auto bot = sig_bottom_reflected(m, W);
            int tr = signature_rank(top);
            int br = signature_rank(bot);
            if (tr < 0 || br < 0) {
                std::cerr << "signature->TL rank failed W=" << W << " k=" << k << "\n";
                return 2;
            }
            for (auto const& s : {top, bot}) {
                if (!sig_ids[k].count(s)) {
                    int id = int(sigs[k].size());
                    sig_ids[k][s] = id;
                    sigs[k].push_back(s);
                }
            }
            recs.push_back({mask, revbits(mask, W), k, top, bot, tr, br, from64(raw)});
        }

        // Structural cross-check: original meander compatibility and the beta=0
        // Gram form agree on every topology actually reached by the half-DP.
        for (int k = 1; k <= W; k += 2) {
            auto const& ss = sigs[k];
            for (int i = 0; i < int(ss.size()); ++i) {
                int ri = signature_rank(ss[i]);
                for (int j = 0; j < int(ss.size()); ++j) {
                    int rj = signature_rank(ss[j]);
                    bool a = compat(ss[i], ss[j]);
                    bool b = gram01(basis(k, 1).words[ri], basis(k, 1).words[rj], k) != 0;
                    if (a != b) {
                        std::cerr << "compat/Gram mismatch W=" << W << " k=" << k
                                  << " i=" << i << " j=" << j << "\n";
                        return 3;
                    }
                }
            }
        }

        std::unordered_map<uint32_t, Group> groups;
        for (auto const& r : recs) {
            int dim = int(basis(r.k, 1).words.size());
            auto& gx = groups[r.mask];
            if (gx.x_tl.empty()) {
                gx.k = r.k;
                gx.x_tl.assign(dim, 0);
                gx.y_tl.assign(dim, 0);
                gx.x_sig.assign(sigs[r.k].size(), 0);
                gx.y_sig.assign(sigs[r.k].size(), 0);
            }
            gx.x_tl[r.top_rank] = tl_addm(gx.x_tl[r.top_rank], r.val);
            gx.x_sig[sig_ids[r.k][r.top]] = tl_addm(gx.x_sig[sig_ids[r.k][r.top]], r.val);

            auto& gy = groups[r.rmask];
            if (gy.x_tl.empty()) {
                gy.k = r.k;
                gy.x_tl.assign(dim, 0);
                gy.y_tl.assign(dim, 0);
                gy.x_sig.assign(sigs[r.k].size(), 0);
                gy.y_sig.assign(sigs[r.k].size(), 0);
            }
            gy.y_tl[r.bot_rank] = tl_addm(gy.y_tl[r.bot_rank], r.val);
            gy.y_sig[sig_ids[r.k][r.bot]] = tl_addm(gy.y_sig[sig_ids[r.k][r.bot]], r.val);
        }

        uint32_t ans_tl = 0;
        uint32_t ans_meander = 0;
        uint64_t used_meander_edges = 0;

        for (auto& kv : groups) {
            Group& g = kv.second;
            if (!g.k) continue;

            auto tx = g.x_tl;
            auto ty = g.y_tl;
            transform(g.k, 1, tx);
            transform(g.k, 1, ty);
            ans_tl = tl_addm(ans_tl, canonical_bilinear(g.k, 1, tx, 0, ty, 0));

            auto const& ss = sigs[g.k];
            for (int i = 0; i < int(g.x_sig.size()); ++i) {
                if (!g.x_sig[i]) continue;
                for (int j = 0; j < int(g.y_sig.size()); ++j) {
                    if (!g.y_sig[j] || !compat(ss[i], ss[j])) continue;
                    ans_meander = tl_addm(
                        ans_meander,
                        tl_mulm(g.x_sig[i], g.y_sig[j]));
                    ++used_meander_edges;
                }
            }
        }

        PathCounter<Modnum<uint64_t>> full(W, W, false, false);
        uint32_t exact = from64(full.count());
        bool ok = ans_tl == ans_meander && ans_tl == exact;
        std::cout << "W=" << W
                  << " half_states=" << half.mc.codeSize()
                  << " reached=" << recs.size()
                  << " masks=" << groups.size()
                  << " used_meander_edges=" << used_meander_edges
                  << " tl=" << ans_tl
                  << " meander=" << ans_meander
                  << " full=" << exact
                  << " " << (ok ? "OK" : "MISMATCH") << "\n";
        if (!ok) return 1;
    }
    return 0;
}
