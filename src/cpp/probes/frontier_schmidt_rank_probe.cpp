#include <cstdint>
#define main ggcount_original_main
#include "ggcount_public.cpp"
#undef main

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

using U32 = std::uint32_t;
using U64 = std::uint64_t;
static constexpr U64 P = 4294967291ULL;

static U64 addp(U64 a, U64 b) {
    U64 z = a + b;
    if (z >= P) z -= P;
    return z;
}
static U64 subp(U64 a, U64 b) { return a >= b ? a - b : a + P - b; }
static U64 mulp(U64 a, U64 b) { return U64((__uint128_t(a) * b) % P); }
static U64 powp(U64 a, U64 e) {
    U64 r = 1;
    while (e) {
        if (e & 1) r = mulp(r, a);
        a = mulp(a, a);
        e >>= 1;
    }
    return r;
}
static U64 invp(U64 a) { return powp(a, P - 2); }

static std::vector<Mate> code_states(MateCodec const& mc) {
    std::vector<Mate> out(mc.codeSize());
    for (Code bi = 0; bi < mc.codeSizeL(); ++bi) {
        auto const& b = mc.codeTable(bi);
        for (Code i = 0; i < b.size; ++i) out[b.base + i] = b.mateL | b.mateR[i];
    }
    return out;
}

struct Entry {
    U32 hi = 0, lo = 0;
    U64 value = 0;
};
struct Block {
    int h = 0;
    std::vector<Entry> entries;
};

static U32 segment_code(Mate m, int first, int len) {
    U32 z = 0;
    for (int i = 0; i < len; ++i) z |= U32(m.get(first + i)) << (2 * i);
    return z;
}

// Balanced W/2 | W/2 cut.  This is the natural Schmidt cut: only one
// horizontal cell update per row crosses it.  The shared block label is the
// intermediate Motzkin height after the HIGH half has been consumed.
static int cut_height(Mate m, int W) {
    const int k = W / 2;
    int h = 1;
    for (int p = W - 1; p >= k; --p) {
        auto v = m.get(p);
        if (v == R) --h;
        else if (v == L) ++h;
    }
    return h;
}

static int rank_dense(std::vector<U64>& a, int rows, int cols) {
    int r = 0;
    for (int c = 0; c < cols && r < rows; ++c) {
        int piv = r;
        while (piv < rows && a[size_t(piv) * cols + c] == 0) ++piv;
        if (piv == rows) continue;
        if (piv != r) {
            for (int j = c; j < cols; ++j)
                std::swap(a[size_t(piv) * cols + j], a[size_t(r) * cols + j]);
        }
        U64 inv = invp(a[size_t(r) * cols + c]);
        for (int i = r + 1; i < rows; ++i) {
            U64 x = a[size_t(i) * cols + c];
            if (!x) continue;
            U64 f = mulp(x, inv);
            a[size_t(i) * cols + c] = 0;
            for (int j = c + 1; j < cols; ++j) {
                a[size_t(i) * cols + j] = subp(
                    a[size_t(i) * cols + j],
                    mulp(f, a[size_t(r) * cols + j]));
            }
        }
        ++r;
    }
    return r;
}

struct RankStats {
    U64 envelope = 0;
    U64 factor_elems = 0;
    U64 nnz = 0;
    int max_rank = 0;
    int blocks = 0;
};

static RankStats measure(PathCounter<Modnum<U64>>& pc,
                         std::vector<Mate> const& states,
                         int row) {
    const int W = pc.cols;
    const int k = W / 2;
    std::vector<Block> blocks(k + 2);
    for (int h = 0; h < int(blocks.size()); ++h) blocks[h].h = h;

    U64 active = 0;
    for (Code i = 0; i < pc.mc.codeSize(); ++i) {
        U64 v = U64(pc.value[i]);
        if (!v) continue;
        Mate m = states[i];
        int h = cut_height(m, W);
        if (h < 0 || h >= int(blocks.size())) continue;
        U32 lo = segment_code(m, 0, k);
        U32 hi = segment_code(m, k, k);
        blocks[h].entries.push_back({hi, lo, v % P});
        ++active;
    }

    RankStats total;
    std::cout << "W=" << W << " row=" << row << " active=" << active << "\n";
    for (auto& b : blocks) {
        if (b.entries.empty()) continue;
        std::unordered_map<U32, int> ri, ci;
        ri.reserve(b.entries.size());
        ci.reserve(b.entries.size());
        for (auto const& e : b.entries) {
            if (!ri.count(e.hi)) ri.emplace(e.hi, int(ri.size()));
            if (!ci.count(e.lo)) ci.emplace(e.lo, int(ci.size()));
        }
        int Rn = int(ri.size()), Cn = int(ci.size());
        bool transposed = false;
        if (Rn > Cn) {
            transposed = true;
            std::swap(Rn, Cn);
        }
        std::vector<U64> mat(size_t(Rn) * Cn, 0);
        for (auto const& e : b.entries) {
            int rr = ri[e.hi], cc = ci[e.lo];
            if (transposed) std::swap(rr, cc);
            U64& z = mat[size_t(rr) * Cn + cc];
            z = addp(z, e.value);
        }
        int rank = rank_dense(mat, Rn, Cn);
        U64 R0 = ri.size(), C0 = ci.size();
        U64 env = R0 * C0;
        U64 fac = U64(rank) * (R0 + C0);
        double be = (R0 + C0) ? double(env) / double(R0 + C0) : 0.0;
        double ratio = env ? double(fac) / double(env) : 0.0;
        std::cout << "  h=" << b.h
                  << " rows=" << R0
                  << " cols=" << C0
                  << " nnz=" << b.entries.size()
                  << " rank=" << rank
                  << " break_even_rank=" << be
                  << " factor/original=" << ratio
                  << (double(rank) < be ? " WIN" : " LOSE")
                  << "\n";
        total.envelope += env;
        total.factor_elems += fac;
        total.nnz += b.entries.size();
        total.max_rank = std::max(total.max_rank, rank);
        ++total.blocks;
    }
    std::cout << "SUMMARY W=" << W << " row=" << row
              << " blocks=" << total.blocks
              << " envelope=" << total.envelope
              << " factor_elems=" << total.factor_elems
              << " factor/original="
              << (total.envelope ? double(total.factor_elems) / double(total.envelope) : 0.0)
              << " max_rank=" << total.max_rank << "\n";
    return total;
}

static void init_pc(PathCounter<Modnum<U64>>& pc) {
    for (Code i = 0; i < pc.mc.codeSize(); ++i) pc.value[i] = 0;
    for (Code i = 0; i < pc.wc.codeSize(); ++i) pc.deferred[i] = 0;
    pc.value[pc.mc.encode(Mate(pc.cols - 1, R))] = 1;
}

static void run_one_row(PathCounter<Modnum<U64>>& pc) {
    for (int j = 0; j < pc.cols - 2; ++j) pc.update(j, false);
    pc.update(pc.cols - 2, false);
}

int main(int argc, char** argv) {
    msg = NONE;
    modulus = P;
    int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    int onlyW = argc > 2 ? std::atoi(argv[2]) : 0;
    if (maxW > 14) {
        std::cerr << "exact dense elimination is intentionally capped at W<=14\n";
        maxW = 14;
    }
    for (int W = 4; W <= maxW; W += 2) {
        if (onlyW && W != onlyW) continue;
        PathCounter<Modnum<U64>> pc(W, W, false, false);
        init_pc(pc);
        auto states = code_states(pc.mc);
        measure(pc, states, 0);
        for (int row = 1; row <= W / 2; ++row) {
            run_one_row(pc);
            measure(pc, states, row);
        }
    }
    return 0;
}
