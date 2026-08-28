#pragma once

#include "ramstream32_bucket_closure_cross5_rankdelta8.cuh"
#include "ramstream32_bucket_low_rankformula.cuh"

static constexpr uint32_t P10DC_RANKFORMULA_CHUNKINFO_CODE_MASK = 0x03ffu;
static constexpr uint32_t P10DC_RANKFORMULA_CHUNKINFO_SUPPORT_SHIFT = 10u;
static constexpr uint32_t P10DC_RANKFORMULA_BALLOT_STRIDE = 16u;
static_assert(LOW_LUT_K <= 14);

static constexpr uint32_t p10dc_rankformula_choose_host(int n, int k) {
    if (k < 0 || k > n) return 0u;
    if (k > n - k) k = n - k;
    uint64_t z = 1;
    for (int i = 1; i <= k; ++i)
        z = z * uint64_t(n - k + i) / uint64_t(i);
    return uint32_t(z);
}

static constexpr uint32_t p10dc_rankformula_ballot_host(int n, int h) {
    if (n < 0 || h < 0 || h > n || ((n - h) & 1)) return 0u;
    const int ups = (n - h) / 2;
    return p10dc_rankformula_choose_host(n, ups) -
           p10dc_rankformula_choose_host(n, ups - 1);
}

static constexpr uint16_t p10dc_rankformula_chunkinfo_host(uint32_t key) {
    uint32_t code = 0, support = 0;
    for (int pos = 0; pos < P10DC_CROSS5_CHUNK; ++pos) {
        const uint32_t v = (key / p10dc_cross5_pow3_host(pos)) % 3u;
        code |= v << (2 * pos);
        if (v != 0u) support |= 1u << pos;
    }
    return uint16_t(code | (support << P10DC_RANKFORMULA_CHUNKINFO_SUPPORT_SHIFT));
}

static constexpr uint32_t p10dc_rankformula_ballot_pair_host(uint32_t rem, uint32_t s) {
    if (rem == 0u) return 0u;
    const uint32_t dest = s > 0u
        ? p10dc_rankformula_ballot_host(int(rem) - 1, int(s) - 1) : 0u;
    const uint32_t raised = p10dc_rankformula_ballot_host(int(rem) - 1, int(s) + 1);
    const int diff = int(raised) - int(dest);
    return (dest & 0xffffu) |
           (uint32_t(uint16_t(int16_t(diff))) << 16);
}

#if P10DC_RANKSTREAM_LUT_LDG
__device__ __align__(128) uint16_t D_P10DC_RANKFORMULA_CHUNKINFO[P10DC_CROSS5_KEYS];
__device__ __align__(128) uint32_t
    D_P10DC_RANKFORMULA_BALLOT[(LOW_LUT_K + 1) * P10DC_RANKFORMULA_BALLOT_STRIDE];
#else
__constant__ uint16_t D_P10DC_RANKFORMULA_CHUNKINFO[P10DC_CROSS5_KEYS];
__constant__ uint32_t
    D_P10DC_RANKFORMULA_BALLOT[(LOW_LUT_K + 1) * P10DC_RANKFORMULA_BALLOT_STRIDE];
#endif

static std::array<uint16_t, P10DC_CROSS5_KEYS> p10dc_rankformula_chunkinfo_table() {
    std::array<uint16_t, P10DC_CROSS5_KEYS> out{};
    for (uint32_t k = 0; k < P10DC_CROSS5_KEYS; ++k)
        out[k] = p10dc_rankformula_chunkinfo_host(k);
    return out;
}

static std::array<uint32_t,
                  (LOW_LUT_K + 1) * P10DC_RANKFORMULA_BALLOT_STRIDE>
p10dc_rankformula_ballot_table() {
    std::array<uint32_t,
               (LOW_LUT_K + 1) * P10DC_RANKFORMULA_BALLOT_STRIDE> out{};
    for (uint32_t rem = 0; rem <= uint32_t(LOW_LUT_K); ++rem)
        for (uint32_t s = 0; s < P10DC_RANKFORMULA_BALLOT_STRIDE; ++s)
            out[size_t(rem) * P10DC_RANKFORMULA_BALLOT_STRIDE + s] =
                p10dc_rankformula_ballot_pair_host(rem, s);
    return out;
}

static void p10dc_install_rankformula_lut() {
    p10dc_install_rankdelta8_lut();
    static const auto chunkinfo = p10dc_rankformula_chunkinfo_table();
    static const auto ballot = p10dc_rankformula_ballot_table();
    ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_CHUNKINFO, chunkinfo.data(),
                          chunkinfo.size() * sizeof(uint16_t)),
       "p10dc rankformula chunkinfo table");
    ck(cudaMemcpyToSymbol(D_P10DC_RANKFORMULA_BALLOT, ballot.data(),
                          ballot.size() * sizeof(uint32_t)),
       "p10dc rankformula ballot table");
}

__device__ __forceinline__ uint16_t p10dc_rankformula_chunkinfo_load(uint32_t chunk) {
#if P10DC_RANKSTREAM_LUT_LDG
    return __ldg(D_P10DC_RANKFORMULA_CHUNKINFO + chunk);
#else
    return D_P10DC_RANKFORMULA_CHUNKINFO[chunk];
#endif
}

__device__ __forceinline__ uint32_t p10dc_rankformula_ballot_load(
    uint32_t rem, uint32_t s
) {
    const size_t ix = size_t(rem) * P10DC_RANKFORMULA_BALLOT_STRIDE + s;
#if P10DC_RANKSTREAM_LUT_LDG
    return __ldg(D_P10DC_RANKFORMULA_BALLOT + ix);
#else
    return D_P10DC_RANKFORMULA_BALLOT[ix];
#endif
}

template<int LEN>
__device__ __forceinline__ uint32_t p10dc_cross5_apply_rankformula(
    uint32_t chunk, uint16_t chunkinfo, uint32_t& state,
    const Count* source_row, uint32_t source_base, uint32_t dest_local,
    int& factor_h, uint32_t& rem, int& prefix_corr, BkczCrossAccum& sum
) {
    static_assert(LEN >= 1 && LEN <= P10DC_CROSS5_CHUNK);
    uint32_t rankmask, consume, delta_bias;
    bool halt;
#if P10DC_RANKDELTA8_FUSED13
    const uint32_t pair = uint32_t(p10dc_rankdelta8_pair_load(state, chunk));
    rankmask = pair & P10DC_CROSS5_MASK_MASK;
    halt = ((pair >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0;
    consume = (pair >> P10DC_RANKDELTA8_PAIR_CONSUME_SHIFT) &
              P10DC_RANKDELTA8_PAIR_CONSUME_MASK;
    delta_bias = (pair >> P10DC_RANKDELTA8_PAIR_DELTA_SHIFT) &
                 P10DC_RANKDELTA8_PAIR_DELTA_MASK;
#else
    const uint8_t e = p10dc_rankstream_entry_load(
        size_t(state) * P10DC_RANKSTREAM_LUT_STRIDE + chunk);
    const uint8_t meta = p10dc_rankstream_meta_load(chunk);
    rankmask = uint32_t(e & P10DC_CROSS5_MASK_MASK);
    halt = ((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0;
    consume = halt
        ? (rankmask ? 32u - uint32_t(__clz(rankmask)) : 0u)
        : uint32_t(meta & P10DC_RANKSTREAM_META_LCOUNT_MASK);
    delta_bias = uint32_t(meta >> P10DC_RANKSTREAM_META_DELTA_SHIFT);
#endif

    const uint32_t code10 = uint32_t(chunkinfo) & P10DC_RANKFORMULA_CHUNKINFO_CODE_MASK;
    uint32_t lordinal = 0;
#pragma unroll
    for (int bit = LEN - 1; bit >= 0; --bit) {
        // Once the last rank needed by a halting chunk has been produced, the
        // remainder of the factor path cannot affect the returned sum.
        if (halt && lordinal >= consume) continue;
        const uint32_t sym = (code10 >> (2 * bit)) & 3u;
        if (sym == uint32_t(::L)) {
            const uint32_t bp = p10dc_rankformula_ballot_load(rem, uint32_t(factor_h));
            const uint32_t dest_contrib = bp & 0xffffu;
            const int diff = int(int16_t(bp >> 16));
            if (rankmask & (1u << lordinal)) {
                const int source_local = int(dest_local) + prefix_corr - int(dest_contrib);
                sum = bkcz_cross_add(
                    sum, source_row[source_base + uint32_t(source_local)]);
            }
            prefix_corr += diff;
            ++factor_h;
            --rem;
            ++lordinal;
        } else if (sym == uint32_t(R)) {
            --factor_h;
            --rem;
        }
    }
    if (halt) return 1u;
    state = uint32_t(
        int(state) + int(delta_bias) - P10DC_RANKSTREAM_META_DELTA_BIAS);
    return 0u;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankformula_fast(
    uint32_t packed_chunks, uint32_t h, uint32_t rank, uint32_t depth,
    const Count* source_row
) {
    if (!depth) return BkczCrossAccum(0);
    constexpr int L0 = LOW_LUT_K >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    const uint32_t c0 = packed_chunks & 0xffu;
    const uint16_t i0 = p10dc_rankformula_chunkinfo_load(c0);
    uint32_t mask = (uint32_t(i0) >> P10DC_RANKFORMULA_CHUNKINFO_SUPPORT_SHIFT) << S0;

    uint32_t c1 = 0, c2 = 0;
    uint16_t i1 = 0, i2 = 0;
    if constexpr (S0 > 0) {
        constexpr int L1 = S0 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S0;
        constexpr int S1 = S0 - L1;
        c1 = (packed_chunks >> 8) & 0xffu;
        i1 = p10dc_rankformula_chunkinfo_load(c1);
        mask |= (uint32_t(i1) >> P10DC_RANKFORMULA_CHUNKINFO_SUPPORT_SHIFT) << S1;
        if constexpr (S1 > 0) {
            c2 = (packed_chunks >> 16) & 0x7fu;
            i2 = p10dc_rankformula_chunkinfo_load(c2);
            mask |= uint32_t(i2) >> P10DC_RANKFORMULA_CHUNKINFO_SUPPORT_SHIFT;
        }
    }

    const uint32_t dest_base = p10dc_low_rankformula_base(h, mask);
    const uint32_t source_base = p10dc_low_rankformula_base(h + 2u, mask);
    const uint32_t dest_local = rank - dest_base;
    uint32_t state = depth;
    uint32_t rem = uint32_t(__popc(mask));
    int factor_h = int(h), prefix_corr = 0;
    BkczCrossAccum sum = 0;

    uint32_t st = p10dc_cross5_apply_rankformula<L0>(
        c0, i0, state, source_row, source_base, dest_local,
        factor_h, rem, prefix_corr, sum);
    if (st == 1u) return sum;

    if constexpr (S0 > 0) {
        constexpr int L1 = S0 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S0;
        constexpr int S1 = S0 - L1;
        st = p10dc_cross5_apply_rankformula<L1>(
            c1, i1, state, source_row, source_base, dest_local,
            factor_h, rem, prefix_corr, sum);
        if (st == 1u) return sum;
        if constexpr (S1 > 0) {
            constexpr int L2 = S1 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S1;
            constexpr int S2 = S1 - L2;
            static_assert(S2 == 0, "rankformula K<=14 must fit in three chunks");
            p10dc_cross5_apply_rankformula<L2>(
                c2, i2, state, source_row, source_base, dest_local,
                factor_h, rem, prefix_corr, sum);
        }
    }
    return sum;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankformula_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
    const uint32_t packed_chunks = p10dc_low_rankformula_chunks(h, rank);
    return p10dc_resolved_low_preimages_cross5_rankformula_fast(
        packed_chunks, h, rank, depth, source_row);
}
