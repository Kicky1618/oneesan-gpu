#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine.cuh"
#include "ramstream32_bucket_closure_cross5_rankstream.cuh"

// The full LOW ternary key for K<=14 fits in 23 bits. CROSS5 needs at most
// three base-243 chunks, each <243 and therefore byte-sized. Re-encode the same
// information as {high,mid,low} chunk bytes in the existing uint32 prekey slot.
// This costs no extra device memory and removes all runtime / and % used only to
// split the full ternary key back into CROSS5 chunks.
static constexpr uint32_t p10dc_cross5_pack_chunks_host(uint32_t key) {
    constexpr int L0 = LOW_LUT_K >= 5 ? 5 : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    const uint32_t c0 =
        (key / p10dc_cross5_pow3_host(S0)) % p10dc_cross5_pow3_host(L0);
    uint32_t c1 = 0, c2 = 0;
    if constexpr (S0 > 0) {
        constexpr int L1 = S0 >= 5 ? 5 : S0;
        constexpr int S1 = S0 - L1;
        c1 = (key / p10dc_cross5_pow3_host(S1)) % p10dc_cross5_pow3_host(L1);
        if constexpr (S1 > 0) {
            constexpr int L2 = S1 >= 5 ? 5 : S1;
            constexpr int S2 = S1 - L2;
            static_assert(S2 == 0, "K<=14 must fit in three CROSS5 chunks");
            c2 = key % p10dc_cross5_pow3_host(L2);
        }
    }
    return c0 | (c1 << 8) | (c2 << 16);
}

static constexpr uint32_t p10dc_cross5_unpack_chunks_host(uint32_t packed) {
    constexpr int L0 = LOW_LUT_K >= 5 ? 5 : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    const uint32_t c0 = packed & 0xffu;
    uint32_t key = c0 * p10dc_cross5_pow3_host(S0);
    if constexpr (S0 > 0) {
        constexpr int L1 = S0 >= 5 ? 5 : S0;
        constexpr int S1 = S0 - L1;
        const uint32_t c1 = (packed >> 8) & 0xffu;
        key += c1 * p10dc_cross5_pow3_host(S1);
        if constexpr (S1 > 0) {
            const uint32_t c2 = (packed >> 16) & 0xffu;
            key += c2;
        }
    }
    return key;
}

static_assert(LOW_LUT_K <= 14, "chunkkey backend assumes at most three CROSS5 chunks");
static_assert(p10dc_cross5_pow3_host(LOW_LUT_K) <= (1u << 23),
              "LOW ternary key no longer fits the proved compact range");

__device__ __forceinline__ uint32_t p10dc_low_prechunk_fixed(uint32_t h, uint32_t rank) {
    return D_P10DC_LOW_PREKEY[D_P10DC_LOW_PREKEY_HOFF[h] + rank];
}

// Rankstream metadata is unchanged. After the ordinary rankstream bind builds
// its sparse L->R rank lists, rewrite only the inherited 4-byte prekey buffer
// from full ternary keys to packed CROSS5 chunk keys.
struct BucketFusedDirectHighRowsPrekeyRankStreamChunkKeyTables
    : BucketFusedDirectHighRowsPrekeyRankStreamTables {
    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsPrekeyRankStreamTables::bind_owner(fixed, buckets, slot);
        if (!host_fused || fixed >= BUCKET_NGPU) {
            std::cerr << "p10dc LOW chunkkey invalid bind owner=" << fixed << '\n';
            std::exit(649);
        }

        const BucketFusedHost& f = *host_fused;
        constexpr size_t P = size_t(MAXW + 2);
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_begin = f.low_code_off[owner_base];
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());

        std::vector<uint32_t> packed;
        packed.reserve(low_prekey_count);
        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
            const uint32_t a = f.low_code_off[owner_base + h];
            const uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u]
                : owner_end;
            if (a < owner_begin || a > b || b > owner_end) {
                std::cerr << "p10dc LOW chunkkey height range invalid owner=" << fixed
                          << " h=" << h << " a=" << a << " b=" << b << '\n';
                std::exit(650);
            }
            for (uint32_t i = a; i < b; ++i) {
                const uint32_t key = gpu_direct_ternary_key_host(f.low_codes[i], LOW_LUT_K);
                const uint32_t x = p10dc_cross5_pack_chunks_host(key);
                if ((x >> 24) != 0u || p10dc_cross5_unpack_chunks_host(x) != key) {
                    std::cerr << "p10dc LOW chunkkey roundtrip failed owner=" << fixed
                              << " h=" << h << " key=" << key << " packed=" << x << '\n';
                    std::exit(651);
                }
                packed.push_back(x);
            }
        }
        if (packed.size() != low_prekey_count) {
            std::cerr << "p10dc LOW chunkkey compact size mismatch owner=" << fixed
                      << " got=" << packed.size() << " expected=" << low_prekey_count << '\n';
            std::exit(652);
        }
        if (!packed.empty())
            ck(cudaMemcpy(low_prekey, packed.data(), packed.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "p10dc compact LOW chunkkey H2D");

        std::cerr << "p10dc_low_chunkkey fixed_owner=" << fixed
                  << " entries=" << packed.size()
                  << " bytes_per_code=4 extra_device_bytes=0"
                  << " cross5_chunk_div_runtime=0 cross5_chunk_mod_runtime=0"
                  << " chunkkey_roundtrip_exact=1\n";
    }
};

__device__ __forceinline__ uint32_t p10dc_cross5_apply_chunk_rankstream_value(
    uint32_t chunk, uint32_t& state, const Count* source_row,
    const uint16_t* rank_row, uint32_t& lbase, BkczCrossAccum& sum
) {
    const uint8_t e = D_P10DC_RANKSTREAM_CROSS5[
        size_t(state) * P10DC_CROSS5_KEYS + chunk];
    uint8_t rankmask = uint8_t(e & P10DC_CROSS5_MASK_MASK);
    while (rankmask) {
        const int ordinal = __ffs(int(rankmask)) - 1;
        rankmask = uint8_t(rankmask & uint8_t(rankmask - 1));
        const uint16_t source_rank = rank_row[lbase + uint32_t(ordinal)];
        sum = bkcz_cross_add(sum, source_row[uint32_t(source_rank)]);
    }
    if (((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0) return 1u;
    lbase += uint32_t(D_P10DC_RANKSTREAM_LCOUNT[chunk]);
    state = uint32_t(int(state) + int(D_P10DC_CROSS5_DELTA[chunk]));
    return 0u;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankstream_chunkkey_fast(
    uint32_t packed, uint32_t depth, const Count* source_row,
    const uint16_t* rank_row
) {
    if (!depth) return BkczCrossAccum(0);
    uint32_t state = depth, lbase = 0;
    BkczCrossAccum sum = 0;

    uint32_t st = p10dc_cross5_apply_chunk_rankstream_value(
        packed & 0xffu, state, source_row, rank_row, lbase, sum);
    if (st == 1u) return sum;

    constexpr int L0 = LOW_LUT_K >= 5 ? 5 : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    if constexpr (S0 > 0) {
        st = p10dc_cross5_apply_chunk_rankstream_value(
            (packed >> 8) & 0xffu, state, source_row, rank_row, lbase, sum);
        if (st == 1u) return sum;
        constexpr int L1 = S0 >= 5 ? 5 : S0;
        constexpr int S1 = S0 - L1;
        if constexpr (S1 > 0) {
            p10dc_cross5_apply_chunk_rankstream_value(
                (packed >> 16) & 0xffu, state, source_row, rank_row, lbase, sum);
        }
    }
    return sum;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankstream_chunkkey_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
    const uint32_t packed = p10dc_low_prechunk_fixed(h, rank);
    const uint16_t* rank_row = p10dc_low_rankstream_row(h, rank);
    return p10dc_resolved_low_preimages_cross5_rankstream_chunkkey_fast(
        packed, depth, source_row, rank_row);
}

__device__ __forceinline__ Count
p10dc_direct_resolved_high_plan_sum_cross5_prekey_rankstream_chunkkey(
    const P10DCDirectHighResolvedCtx& c, const BucketPhysicalBlock& db, uint32_t lr
) {
#if GPU_DIRECT_PM_ACCUM
    uint64_t sum = 0;
#else
    Count sum = 0;
#endif
#pragma unroll
    for (uint32_t i = 0; i < BKCZ_MAX_LOCAL; ++i) {
        if (i < c.local_n) {
            const Count v = c.local_base[i][lr];
#if GPU_DIRECT_PM_ACCUM
            sum += uint64_t(v);
#else
            sum = gpu_direct_add(sum, v);
#endif
        }
    }
    if (c.cross_depth) {
#if GPU_DIRECT_PM_ACCUM
        sum += p10dc_resolved_low_preimages_cross5_rankstream_chunkkey_fixed(
            db.hs, lr, c.cross_depth, c.cross_base);
#else
        sum = gpu_direct_add(
            sum, p10dc_resolved_low_preimages_cross5_rankstream_chunkkey_fixed(
                     db.hs, lr, c.cross_depth, c.cross_base));
#endif
    }
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
}

#define P10DC_WARPSTRIPED_CTX P10DCDirectHighResolvedCtx
#define P10DC_WARPSTRIPED_PREPARE_FORWARD(c,payload,loc,p,ss,js,ds,sr,jr,dr) \
    p10dc_prepare_forward_high_delta_direct_affine((c),(payload),(loc),(p),(ss),(js),(ds),(sr),(jr),(dr))
#define P10DC_WARPSTRIPED_PREPARE_REVERSE(c,payload,loc,plan_db,p,edge,ss,js,ds,sr,jr,dr) \
    p10dc_prepare_reverse_high_delta_direct_affine((c),(payload),(loc),(plan_db),(p),(edge),(ss),(js),(ds),(sr),(jr),(dr))
#define p10dc_resolved_high_plan_sum \
    p10dc_direct_resolved_high_plan_sum_cross5_prekey_rankstream_chunkkey
#define bucket_high_orbit_closure_pattern10_depthcode_warpstriped_kernel \
    bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_prekey_rankstream_chunkkey_cross5_kernel
#define bucket_reverse_high_pattern10_depthcode_warpstriped_kernel \
    bucket_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_prekey_rankstream_chunkkey_cross5_kernel
#include "ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped.cuh"
#undef bucket_reverse_high_pattern10_depthcode_warpstriped_kernel
#undef bucket_high_orbit_closure_pattern10_depthcode_warpstriped_kernel
#undef p10dc_resolved_high_plan_sum
#undef P10DC_WARPSTRIPED_PREPARE_REVERSE
#undef P10DC_WARPSTRIPED_PREPARE_FORWARD
#undef P10DC_WARPSTRIPED_CTX
