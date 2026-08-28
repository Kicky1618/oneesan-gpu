#pragma once

#include "ramstream32_bucket_closure_cross5_rankformula_nometa4.cuh"
#include "ramstream32_bucket_low_rankformula_nometa_warpshare.cuh"

#ifndef P10DC_RANKFORMULA_ABSTRACT_SELECT8
#define P10DC_RANKFORMULA_ABSTRACT_SELECT8 0
#endif
static_assert(P10DC_RANKFORMULA_ABSTRACT_SELECT8 == 0 ||
              P10DC_RANKFORMULA_ABSTRACT_SELECT8 == 1,
              "P10DC_RANKFORMULA_ABSTRACT_SELECT8 must be 0 or 1");

static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_MAX_K = 14u;
static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_DESC_N = 7060u;
static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_RANK_N = 32743u;
static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_OFF_N = (P10DC_RANKFORMULA_ABSTRACT_MAX_K + 1u) * 16u;
static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_LP_BITS = P10DC_RANKFORMULA_ABSTRACT_MAX_K;
static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS = 13u;
static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_SELECT_N =
    P10DC_RANKFORMULA_ABSTRACT_DESC_N * P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS;
static_assert(LOW_LUT_K <= int(P10DC_RANKFORMULA_ABSTRACT_MAX_K));

#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
__device__ __align__(128) uint16_t
    D_P10DC_RANKFORMULA_ABSTRACT_ROFF[P10DC_RANKFORMULA_ABSTRACT_DESC_N];
#else
__device__ __align__(128) uint32_t
    D_P10DC_RANKFORMULA_ABSTRACT_DESC[P10DC_RANKFORMULA_ABSTRACT_DESC_N];
#endif
__device__ __align__(128) uint16_t
    D_P10DC_RANKFORMULA_ABSTRACT_SRC[P10DC_RANKFORMULA_ABSTRACT_RANK_N];
#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
// Depth-major: all lanes in a HIGH warp share cross_depth, so nearby abstract
// descriptor indices touch adjacent bytes instead of a 13-byte lane stride.
__device__ __align__(128) uint8_t
    D_P10DC_RANKFORMULA_ABSTRACT_SELECT[P10DC_RANKFORMULA_ABSTRACT_SELECT_N];
#endif
__constant__ uint16_t
    D_P10DC_RANKFORMULA_ABSTRACT_OFF[P10DC_RANKFORMULA_ABSTRACT_OFF_N];

static uint32_t p10dc_rankformula_abstract_lpattern_host(int n, int h, uint32_t local) {
    uint32_t lp = 0;
    int s = h, rem = n;
    for (int ord = 0; ord < n; ++ord) {
        const uint32_t rc = s > 0
            ? p10dc_rankformula_ballot_host(rem - 1, s - 1) : 0u;
        if (s > 0 && local < rc) {
            --s;
        } else {
            if (local < rc) return 0xffffffffu;
            local -= rc;
            lp |= 1u << ord;
            ++s;
        }
        --rem;
    }
    return (s == 0 && local == 0) ? lp : 0xffffffffu;
}

static uint32_t p10dc_rankformula_abstract_rank_host(int n, int h, uint32_t lp) {
    uint32_t rank = 0;
    int s = h, rem = n;
    for (int ord = 0; ord < n; ++ord) {
        const bool is_l = ((lp >> ord) & 1u) != 0u;
        const uint32_t rc = s > 0
            ? p10dc_rankformula_ballot_host(rem - 1, s - 1) : 0u;
        if (is_l) {
            rank += rc;
            ++s;
        } else {
            if (s <= 0) return 0xffffffffu;
            --s;
        }
        --rem;
    }
    return s == 0 ? rank : 0xffffffffu;
}

static uint8_t p10dc_rankformula_abstract_select_host(uint32_t lp, int n, uint32_t depth) {
    uint32_t state = depth, li = 0;
    uint8_t select = 0;
    for (int ord = 0; ord < n; ++ord) {
        if ((lp >> ord) & 1u) {
            if (state == 1u) {
                if (li >= 7u) std::exit(774);
                select |= uint8_t(1u << li);
            }
            ++li;
            ++state;
        } else {
            if (state == 1u) break;
            --state;
        }
    }
    return select;
}

struct P10DCRankFormulaAbstractHost {
    std::array<uint16_t, P10DC_RANKFORMULA_ABSTRACT_OFF_N> off{};
#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
    std::array<uint16_t, P10DC_RANKFORMULA_ABSTRACT_DESC_N> roff{};
#else
    std::array<uint32_t, P10DC_RANKFORMULA_ABSTRACT_DESC_N> desc{};
#endif
    std::array<uint16_t, P10DC_RANKFORMULA_ABSTRACT_RANK_N> src{};
#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
    std::array<uint8_t, P10DC_RANKFORMULA_ABSTRACT_SELECT_N> select{};
#endif
};

static P10DCRankFormulaAbstractHost p10dc_rankformula_abstract_build_host() {
    P10DCRankFormulaAbstractHost out{};
    uint32_t dn = 0, rn = 0;
    for (int n = 0; n <= int(P10DC_RANKFORMULA_ABSTRACT_MAX_K); ++n) {
        for (int h = 0; h < 16; ++h) {
            if (dn >= 65536u) std::exit(768);
            out.off[size_t(n) * 16u + size_t(h)] = uint16_t(dn);
            const uint32_t cnt = p10dc_rankformula_ballot_host(n, h);
            for (uint32_t j = 0; j < cnt; ++j) {
                if (dn >= P10DC_RANKFORMULA_ABSTRACT_DESC_N ||
                    rn >= (1u << 15)) std::exit(769);
                const uint32_t lp = p10dc_rankformula_abstract_lpattern_host(n, h, j);
                if (lp == 0xffffffffu || lp >= (1u << P10DC_RANKFORMULA_ABSTRACT_MAX_K)) std::exit(770);
                const uint32_t di = dn++;
                const uint32_t roff = rn;
#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
                out.roff[di] = uint16_t(roff);
                for (uint32_t depth = 1; depth <= P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS; ++depth) {
                    out.select[size_t(depth - 1u) * P10DC_RANKFORMULA_ABSTRACT_DESC_N + size_t(di)] =
                        p10dc_rankformula_abstract_select_host(lp, n, depth);
                }
                if (p10dc_rankformula_abstract_select_host(lp, n, 14u) != 0u ||
                    p10dc_rankformula_abstract_select_host(lp, n, 15u) != 0u)
                    std::exit(775);
#else
                out.desc[di] = lp | (roff << P10DC_RANKFORMULA_ABSTRACT_LP_BITS);
#endif
                for (int ord = 0; ord < n; ++ord) {
                    if (((lp >> ord) & 1u) == 0u) continue;
                    if (rn >= P10DC_RANKFORMULA_ABSTRACT_RANK_N) std::exit(771);
                    const uint32_t sr = p10dc_rankformula_abstract_rank_host(
                        n, h + 2, lp & ~(1u << ord));
                    if (sr == 0xffffffffu || sr > 1000u) std::exit(772);
                    out.src[rn++] = uint16_t(sr);
                }
            }
        }
    }
    if (dn != P10DC_RANKFORMULA_ABSTRACT_DESC_N ||
        rn != P10DC_RANKFORMULA_ABSTRACT_RANK_N) {
        std::cerr << "p10dc abstract LUT size mismatch desc=" << dn
                  << " src=" << rn << '\n';
        std::exit(773);
    }
    return out;
}

static void p10dc_install_rankformula_abstract_lut() {
    static const auto t = p10dc_rankformula_abstract_build_host();
    ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_ABSTRACT_OFF,
                          t.off.data(), t.off.size() * sizeof(uint16_t)),
       "p10dc abstract off LUT");
#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
    ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_ABSTRACT_ROFF,
                          t.roff.data(), t.roff.size() * sizeof(uint16_t)),
       "p10dc abstract roff16 LUT");
#else
    ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_ABSTRACT_DESC,
                          t.desc.data(), t.desc.size() * sizeof(uint32_t)),
       "p10dc abstract descriptor LUT");
#endif
    ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_ABSTRACT_SRC,
                          t.src.data(), t.src.size() * sizeof(uint16_t)),
       "p10dc abstract source-rank LUT");
#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
    ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_ABSTRACT_SELECT,
                          t.select.data(), t.select.size() * sizeof(uint8_t)),
       "p10dc abstract select8 LUT");
#endif
}

#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
__device__ __forceinline__ uint32_t p10dc_rankformula_abstract_roff_load(uint32_t ix) {
    return uint32_t(__ldg(D_P10DC_RANKFORMULA_ABSTRACT_ROFF + ix));
}
#else
__device__ __forceinline__ uint32_t p10dc_rankformula_abstract_desc_load(uint32_t ix) {
    return __ldg(D_P10DC_RANKFORMULA_ABSTRACT_DESC + ix);
}
#endif
__device__ __forceinline__ uint32_t p10dc_rankformula_abstract_src_load(uint32_t ix) {
    return uint32_t(__ldg(D_P10DC_RANKFORMULA_ABSTRACT_SRC + ix));
}
#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
__device__ __forceinline__ uint32_t p10dc_rankformula_abstract_select_load(uint32_t ix) {
    return uint32_t(__ldg(D_P10DC_RANKFORMULA_ABSTRACT_SELECT + ix));
}
#endif

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
    // Exhaustive K14 proof: depths 14 and 15 never select an L->R source.
    if (!depth || depth > P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS)
        return BkczCrossAccum(0);
    const P10DCRankFormulaNometa4Resolved z =
        p10dc_low_rankformula_nometa_resolve_active(h, rank);
    const uint32_t local = rank - z.start;
    const uint32_t di = uint32_t(D_P10DC_RANKFORMULA_ABSTRACT_OFF[z.n * 16u + h]) + local;

#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
    uint32_t select = p10dc_rankformula_abstract_select_load(
        (depth - 1u) * P10DC_RANKFORMULA_ABSTRACT_DESC_N + di);
    if (!select) return BkczCrossAccum(0);
    const uint32_t rp = p10dc_rankformula_abstract_roff_load(di);
    const uint32_t source_base = uint32_t(int(z.start) + z.base_delta);
    BkczCrossAccum sum = 0;
    while (select) {
        const uint32_t li = uint32_t(__ffs(int(select)) - 1);
        select &= select - 1u;
        const uint32_t source_local = p10dc_rankformula_abstract_src_load(rp + li);
        sum = bkcz_cross_add(sum, source_row[source_base + source_local]);
    }
    return sum;
#else
    const uint32_t source_base = uint32_t(int(z.start) + z.base_delta);
    BkczCrossAccum sum = 0;
    const uint32_t d = p10dc_rankformula_abstract_desc_load(di);
    const uint32_t lp = d & ((1u << P10DC_RANKFORMULA_ABSTRACT_LP_BITS) - 1u);
    uint32_t rp = d >> P10DC_RANKFORMULA_ABSTRACT_LP_BITS;
    uint32_t state = depth;
    // The abstract word is indexed by support ordinal, not physical position.
    // Group64 already stores n, so no runtime popcount or clz scan remains.
    for (uint32_t ordinal = 0; ordinal < z.n; ++ordinal) {
        if ((lp >> ordinal) & 1u) {
            if (state == 1u) {
                const uint32_t source_local = p10dc_rankformula_abstract_src_load(rp);
                sum = bkcz_cross_add(sum, source_row[source_base + source_local]);
            }
            ++rp;
            ++state;
        } else {
            if (state == 1u) return sum;
            --state;
        }
    }
    return sum;
#endif
}
