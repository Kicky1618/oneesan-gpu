#pragma once

#include "ramstream32_bucket_closure_cross5_rankformula_nometa4.cuh"
#include "ramstream32_bucket_low_rankformula_nometa_warpshare.cuh"

#ifndef P10DC_RANKFORMULA_ABSTRACT_SELECT8
#define P10DC_RANKFORMULA_ABSTRACT_SELECT8 0
#endif
#ifndef P10DC_RANKFORMULA_ABSTRACT_SRCPACK10
#define P10DC_RANKFORMULA_ABSTRACT_SRCPACK10 0
#endif
static_assert(P10DC_RANKFORMULA_ABSTRACT_SELECT8 == 0 ||
              P10DC_RANKFORMULA_ABSTRACT_SELECT8 == 1,
              "P10DC_RANKFORMULA_ABSTRACT_SELECT8 must be 0 or 1");
static_assert(P10DC_RANKFORMULA_ABSTRACT_SRCPACK10 == 0 ||
              P10DC_RANKFORMULA_ABSTRACT_SRCPACK10 == 1,
              "P10DC_RANKFORMULA_ABSTRACT_SRCPACK10 must be 0 or 1");
static_assert(!P10DC_RANKFORMULA_ABSTRACT_SRCPACK10 ||
              P10DC_RANKFORMULA_ABSTRACT_SELECT8,
              "abstract source pack10 requires select8");

static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_MAX_K = 14u;
static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_DESC_N = 7060u;
static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_RANK_N = 32743u;
static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_SRC7_N = 429u;
static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_OFF_N = (P10DC_RANKFORMULA_ABSTRACT_MAX_K + 1u) * 16u;
static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_LP_BITS = P10DC_RANKFORMULA_ABSTRACT_MAX_K;
static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS = 13u;
static constexpr uint32_t P10DC_RANKFORMULA_ABSTRACT_SELECT_N =
    P10DC_RANKFORMULA_ABSTRACT_DESC_N * P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS;
static_assert(LOW_LUT_K <= int(P10DC_RANKFORMULA_ABSTRACT_MAX_K));

#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
#if P10DC_RANKFORMULA_ABSTRACT_SRCPACK10
__device__ __align__(128) uint64_t
    D_P10DC_RANKFORMULA_ABSTRACT_SRC6[P10DC_RANKFORMULA_ABSTRACT_DESC_N];
__device__ __align__(128) uint16_t
    D_P10DC_RANKFORMULA_ABSTRACT_SRC7[P10DC_RANKFORMULA_ABSTRACT_SRC7_N];
#else
__device__ __align__(128) uint16_t
    D_P10DC_RANKFORMULA_ABSTRACT_ROFF[P10DC_RANKFORMULA_ABSTRACT_DESC_N];
__device__ __align__(128) uint16_t
    D_P10DC_RANKFORMULA_ABSTRACT_SRC[P10DC_RANKFORMULA_ABSTRACT_RANK_N];
#endif
#else
__device__ __align__(128) uint32_t
    D_P10DC_RANKFORMULA_ABSTRACT_DESC[P10DC_RANKFORMULA_ABSTRACT_DESC_N];
__device__ __align__(128) uint16_t
    D_P10DC_RANKFORMULA_ABSTRACT_SRC[P10DC_RANKFORMULA_ABSTRACT_RANK_N];
#endif
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
#if P10DC_RANKFORMULA_ABSTRACT_SRCPACK10
    std::array<uint64_t, P10DC_RANKFORMULA_ABSTRACT_DESC_N> src6{};
    std::array<uint16_t, P10DC_RANKFORMULA_ABSTRACT_SRC7_N> src7{};
#else
    std::array<uint16_t, P10DC_RANKFORMULA_ABSTRACT_DESC_N> roff{};
    std::array<uint16_t, P10DC_RANKFORMULA_ABSTRACT_RANK_N> src{};
#endif
    std::array<uint8_t, P10DC_RANKFORMULA_ABSTRACT_SELECT_N> select{};
#else
    std::array<uint32_t, P10DC_RANKFORMULA_ABSTRACT_DESC_N> desc{};
    std::array<uint16_t, P10DC_RANKFORMULA_ABSTRACT_RANK_N> src{};
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
                if (dn >= P10DC_RANKFORMULA_ABSTRACT_DESC_N || rn >= (1u << 15)) std::exit(769);
                const uint32_t lp = p10dc_rankformula_abstract_lpattern_host(n, h, j);
                if (lp == 0xffffffffu || lp >= (1u << P10DC_RANKFORMULA_ABSTRACT_MAX_K)) std::exit(770);
                const uint32_t di = dn++;
                const uint32_t roff = rn;
#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
#if !P10DC_RANKFORMULA_ABSTRACT_SRCPACK10
                out.roff[di] = uint16_t(roff);
#endif
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
#if P10DC_RANKFORMULA_ABSTRACT_SELECT8 && P10DC_RANKFORMULA_ABSTRACT_SRCPACK10
                uint64_t src6 = 0;
                uint32_t li = 0;
#endif
                for (int ord = 0; ord < n; ++ord) {
                    if (((lp >> ord) & 1u) == 0u) continue;
                    const uint32_t sr = p10dc_rankformula_abstract_rank_host(n, h + 2, lp & ~(1u << ord));
                    if (sr == 0xffffffffu || sr > 1000u) std::exit(772);
#if P10DC_RANKFORMULA_ABSTRACT_SELECT8 && P10DC_RANKFORMULA_ABSTRACT_SRCPACK10
                    if (li < 6u) {
                        src6 |= uint64_t(sr) << (10u * li);
                    } else {
                        if (li != 6u || n != 14 || h != 0 || j >= P10DC_RANKFORMULA_ABSTRACT_SRC7_N)
                            std::exit(776);
                        out.src7[j] = uint16_t(sr);
                    }
                    ++li;
                    ++rn;
#else
                    if (rn >= P10DC_RANKFORMULA_ABSTRACT_RANK_N) std::exit(771);
                    out.src[rn++] = uint16_t(sr);
#endif
                }
#if P10DC_RANKFORMULA_ABSTRACT_SELECT8 && P10DC_RANKFORMULA_ABSTRACT_SRCPACK10
                out.src6[di] = src6;
#endif
            }
        }
    }
    if (dn != P10DC_RANKFORMULA_ABSTRACT_DESC_N || rn != P10DC_RANKFORMULA_ABSTRACT_RANK_N) {
        std::cerr << "p10dc abstract LUT size mismatch desc=" << dn << " src=" << rn << '\n';
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
#if P10DC_RANKFORMULA_ABSTRACT_SRCPACK10
    ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_ABSTRACT_SRC6,
                          t.src6.data(), t.src6.size() * sizeof(uint64_t)),
       "p10dc abstract src6 pack10 LUT");
    ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_ABSTRACT_SRC7,
                          t.src7.data(), t.src7.size() * sizeof(uint16_t)),
       "p10dc abstract src7 overflow LUT");
#else
    ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_ABSTRACT_ROFF,
                          t.roff.data(), t.roff.size() * sizeof(uint16_t)),
       "p10dc abstract roff16 LUT");
    ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_ABSTRACT_SRC,
                          t.src.data(), t.src.size() * sizeof(uint16_t)),
       "p10dc abstract source-rank LUT");
#endif
#else
    ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_ABSTRACT_DESC,
                          t.desc.data(), t.desc.size() * sizeof(uint32_t)),
       "p10dc abstract descriptor LUT");
    ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_ABSTRACT_SRC,
                          t.src.data(), t.src.size() * sizeof(uint16_t)),
       "p10dc abstract source-rank LUT");
#endif
#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
    ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_ABSTRACT_SELECT,
                          t.select.data(), t.select.size() * sizeof(uint8_t)),
       "p10dc abstract select8 LUT");
#endif
}

#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
#if P10DC_RANKFORMULA_ABSTRACT_SRCPACK10
__device__ __forceinline__ uint64_t p10dc_rankformula_abstract_src6_load(uint32_t ix) {
    return __ldg(D_P10DC_RANKFORMULA_ABSTRACT_SRC6 + ix);
}
__device__ __forceinline__ uint32_t p10dc_rankformula_abstract_src7_load(uint32_t ix) {
    return uint32_t(__ldg(D_P10DC_RANKFORMULA_ABSTRACT_SRC7 + ix));
}
#else
__device__ __forceinline__ uint32_t p10dc_rankformula_abstract_roff_load(uint32_t ix) {
    return uint32_t(__ldg(D_P10DC_RANKFORMULA_ABSTRACT_ROFF + ix));
}
__device__ __forceinline__ uint32_t p10dc_rankformula_abstract_src_load(uint32_t ix) {
    return uint32_t(__ldg(D_P10DC_RANKFORMULA_ABSTRACT_SRC + ix));
}
#endif
#else
__device__ __forceinline__ uint32_t p10dc_rankformula_abstract_desc_load(uint32_t ix) {
    return __ldg(D_P10DC_RANKFORMULA_ABSTRACT_DESC + ix);
}
__device__ __forceinline__ uint32_t p10dc_rankformula_abstract_src_load(uint32_t ix) {
    return uint32_t(__ldg(D_P10DC_RANKFORMULA_ABSTRACT_SRC + ix));
}
#endif
#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
__device__ __forceinline__ uint32_t p10dc_rankformula_abstract_select_load(uint32_t ix) {
    return uint32_t(__ldg(D_P10DC_RANKFORMULA_ABSTRACT_SELECT + ix));
}
#endif

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankformula_nometa4_abstract_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
    if (!depth || depth > P10DC_RANKFORMULA_ABSTRACT_SELECT_DEPTHS)
        return BkczCrossAccum(0);
    const P10DCRankFormulaNometa4Resolved z = p10dc_low_rankformula_nometa_resolve_active(h, rank);
    const uint32_t local = rank - z.start;
    const uint32_t di = uint32_t(D_P10DC_RANKFORMULA_ABSTRACT_OFF[z.n * 16u + h]) + local;

#if P10DC_RANKFORMULA_ABSTRACT_SELECT8
    uint32_t select = p10dc_rankformula_abstract_select_load(
        (depth - 1u) * P10DC_RANKFORMULA_ABSTRACT_DESC_N + di);
    if (!select) return BkczCrossAccum(0);
    const uint32_t source_base = uint32_t(int(z.start) + z.base_delta);
    BkczCrossAccum sum = 0;
#if P10DC_RANKFORMULA_ABSTRACT_SRCPACK10
    const uint64_t src6 = p10dc_rankformula_abstract_src6_load(di);
    while (select) {
        const uint32_t li = uint32_t(__ffs(int(select)) - 1);
        select &= select - 1u;
        const uint32_t source_local = li < 6u
            ? uint32_t((src6 >> (10u * li)) & 1023u)
            : p10dc_rankformula_abstract_src7_load(local);
        sum = bkcz_cross_add(sum, source_row[source_base + source_local]);
    }
#else
    const uint32_t rp = p10dc_rankformula_abstract_roff_load(di);
    while (select) {
        const uint32_t li = uint32_t(__ffs(int(select)) - 1);
        select &= select - 1u;
        const uint32_t source_local = p10dc_rankformula_abstract_src_load(rp + li);
        sum = bkcz_cross_add(sum, source_row[source_base + source_local]);
    }
#endif
    return sum;
#else
    const uint32_t source_base = uint32_t(int(z.start) + z.base_delta);
    BkczCrossAccum sum = 0;
    const uint32_t d = p10dc_rankformula_abstract_desc_load(di);
    const uint32_t lp = d & ((1u << P10DC_RANKFORMULA_ABSTRACT_LP_BITS) - 1u);
    uint32_t rp = d >> P10DC_RANKFORMULA_ABSTRACT_LP_BITS;
    uint32_t state = depth;
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
