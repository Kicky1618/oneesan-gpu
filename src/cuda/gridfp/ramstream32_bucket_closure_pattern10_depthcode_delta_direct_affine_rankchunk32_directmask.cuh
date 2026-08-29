#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankchunk32.cuh"

#ifndef P10DC_RANKCHUNK32_DIRECTMASK
#define P10DC_RANKCHUNK32_DIRECTMASK 0
#endif
static_assert(P10DC_RANKCHUNK32_DIRECTMASK == 0 || P10DC_RANKCHUNK32_DIRECTMASK == 1,
              "P10DC_RANKCHUNK32_DIRECTMASK must be 0 or 1");

#if P10DC_RANKCHUNK32_DIRECTMASK

// The CROSS5 automaton and rank-stream projection depend only on the LOW code
// and incoming cross depth. Precompute the final global rank-ordinal mask for
// every compact LOW code/depth. A second compact table stores the absolute
// rankstream offset for each LOW rank, eliminating rankchunk metadata decode,
// block-base loads, and warp shuffles from the directmask hot path.
__constant__ uint8_t* D_P10DC_LOW_RANKDIRECTMASK;
__constant__ uint32_t D_P10DC_LOW_RANKDIRECTMASK_STRIDE;
__constant__ uint32_t* D_P10DC_LOW_RANKDIRECTOFF;

static constexpr uint8_t p10dc_rankchunk32_directmask_host(
    uint32_t packed_chunks, uint32_t depth
) {
    if (depth == 0u) return 0u;
    uint32_t state = depth;
    uint32_t lbase = 0u;
    uint8_t out = 0u;

    constexpr int L0 = LOW_LUT_K >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    constexpr int L1 = S0 > 0 ? (S0 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S0) : 0;
    constexpr int S1 = S0 - L1;
    constexpr int NCHUNK = 1 + (S0 > 0 ? 1 : 0) + (S1 > 0 ? 1 : 0);
    static_assert(NCHUNK >= 1 && NCHUNK <= 3);

    for (int slot = 0; slot < NCHUNK; ++slot) {
        const uint32_t chunk = slot == 0 ? (packed_chunks & 0xffu)
            : (slot == 1 ? ((packed_chunks >> 8) & 0xffu)
                         : ((packed_chunks >> 16) & 0xffu));
        const uint8_t e = p10dc_rankstream_host_entry(chunk, state);
        const uint8_t local = uint8_t(e & P10DC_CROSS5_MASK_MASK);
        out = uint8_t(out | uint8_t(uint32_t(local) << lbase));
        if (((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0u) return out;
        const uint8_t meta = p10dc_rankstream_meta_host(chunk);
        lbase += uint32_t(meta & P10DC_RANKSTREAM_META_LCOUNT_MASK);
        state = uint32_t(
            int(state) + int(meta >> P10DC_RANKSTREAM_META_DELTA_SHIFT)
            - P10DC_RANKSTREAM_META_DELTA_BIAS);
    }
    return out;
}

__device__ __forceinline__ uint32_t p10dc_low_rankchunk32_directcompact(
    uint32_t h, uint32_t rank
) {
    return D_P10DC_LOW_RANKCHUNK_HOFF[h] + rank;
}

__device__ __forceinline__ uint8_t p10dc_low_rankchunk32_directmask_load(
    uint32_t h, uint32_t rank, uint32_t depth
) {
    const uint32_t compact = p10dc_low_rankchunk32_directcompact(h, rank);
    const size_t ix = size_t(depth) * D_P10DC_LOW_RANKDIRECTMASK_STRIDE + compact;
    return __ldg(D_P10DC_LOW_RANKDIRECTMASK + ix);
}

__device__ __forceinline__ const uint16_t* p10dc_low_rankchunk32_directoff_row(
    uint32_t h, uint32_t rank
) {
    const uint32_t compact = p10dc_low_rankchunk32_directcompact(h, rank);
    const uint32_t off = __ldg(D_P10DC_LOW_RANKDIRECTOFF + compact);
    return D_P10DC_LOW_RANKSTREAM + off;
}

struct BucketFusedDirectHighRowsRankChunk32DirectMaskTables
    : BucketFusedDirectHighRowsRankChunk32Tables {
    uint8_t* low_rankdirectmask = nullptr;
    uint32_t* low_rankdirectoff = nullptr;
    size_t low_rankdirectmask_count = 0, low_rankdirectoff_count = 0;
    size_t low_rankdirectmask_capacity = 0, low_rankdirectoff_capacity = 0;

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsRankChunk32Tables::bind_owner(fixed, buckets, slot);
        if (!host_fused) {
            std::cerr << "p10dc rankchunk32 directmask missing host fused metadata\n";
            std::exit(681);
        }
        if (low_rankchunkmeta32_count > uint64_t(0xffffffffu)) {
            std::cerr << "p10dc rankchunk32 directmask stride overflow entries="
                      << low_rankchunkmeta32_count << '\n';
            std::exit(682);
        }

        const BucketFusedHost& f = *host_fused;
        constexpr size_t P = size_t(MAXW + 2);
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());
        const size_t stride = low_rankchunkmeta32_count;
        std::vector<uint8_t> mask(size_t(16) * stride, uint8_t(0));
        std::vector<uint32_t> off(stride, uint32_t(0));

        size_t compact = 0;
        size_t actual_codes = 0;
        uint32_t stream_cursor = 0;
        uint8_t mask_or = 0;
        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
#if P10DC_RANKCHUNK32_ALIGN32
            compact = (compact + P10DC_RANKCHUNK32_BLOCK - 1u) &
                      ~(size_t(P10DC_RANKCHUNK32_BLOCK) - 1u);
#endif
            const uint32_t a = f.low_code_off[owner_base + h];
            const uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u] : owner_end;
            for (uint32_t i = a; i < b; ++i, ++compact, ++actual_codes) {
                if (compact >= stride) {
                    std::cerr << "p10dc rankchunk32 directmask compact overflow owner="
                              << fixed << " h=" << h << " compact=" << compact
                              << " stride=" << stride << '\n';
                    std::exit(683);
                }
                off[compact] = stream_cursor;
                const uint32_t code = f.low_codes[i];
                const uint32_t key = gpu_direct_ternary_key_host(code, LOW_LUT_K);
                const uint32_t packed = p10dc_rankchunk32_pack_host(key);
                for (uint32_t depth = 1; depth <= 15u; ++depth) {
                    const uint8_t m = p10dc_rankchunk32_directmask_host(packed, depth);
                    mask[size_t(depth) * stride + compact] = m;
                    mask_or = uint8_t(mask_or | m);
                }
                for (int pos = 0; pos < LOW_LUT_K; ++pos)
                    if (((code >> (2 * pos)) & 3u) == uint32_t(::L)) ++stream_cursor;
            }
        }
        if (compact != stride || actual_codes != low_prekey_count ||
            stream_cursor != low_rankstream_count || (mask_or & 0x80u) != 0u) {
            std::cerr << "p10dc rankchunk32 directmask shape mismatch owner=" << fixed
                      << " compact=" << compact << '/' << stride
                      << " codes=" << actual_codes << '/' << low_prekey_count
                      << " stream=" << stream_cursor << '/' << low_rankstream_count
                      << " mask_or=0x" << std::hex << unsigned(mask_or) << std::dec << '\n';
            std::exit(684);
        }

        low_rankdirectmask_count = mask.size();
        low_rankdirectoff_count = off.size();
        if (low_rankdirectmask_count > low_rankdirectmask_capacity) {
            if (low_rankdirectmask) cudaFree(low_rankdirectmask);
            low_rankdirectmask = nullptr;
            low_rankdirectmask_capacity = low_rankdirectmask_count;
            if (low_rankdirectmask_capacity)
                ck(cudaMalloc(&low_rankdirectmask, low_rankdirectmask_capacity * sizeof(uint8_t)),
                   "p10dc rankchunk32 directmask alloc");
        }
        if (low_rankdirectoff_count > low_rankdirectoff_capacity) {
            if (low_rankdirectoff) cudaFree(low_rankdirectoff);
            low_rankdirectoff = nullptr;
            low_rankdirectoff_capacity = low_rankdirectoff_count;
            if (low_rankdirectoff_capacity)
                ck(cudaMalloc(&low_rankdirectoff, low_rankdirectoff_capacity * sizeof(uint32_t)),
                   "p10dc rankchunk32 directoff alloc");
        }
        if (!mask.empty())
            ck(cudaMemcpy(low_rankdirectmask, mask.data(), mask.size() * sizeof(uint8_t),
                          cudaMemcpyHostToDevice),
               "p10dc rankchunk32 directmask H2D");
        if (!off.empty())
            ck(cudaMemcpy(low_rankdirectoff, off.data(), off.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "p10dc rankchunk32 directoff H2D");
        const uint32_t stride32 = uint32_t(stride);
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKDIRECTMASK, &low_rankdirectmask,
                              sizeof(low_rankdirectmask)),
           "p10dc rankchunk32 directmask ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKDIRECTMASK_STRIDE, &stride32,
                              sizeof(stride32)),
           "p10dc rankchunk32 directmask stride");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKDIRECTOFF, &low_rankdirectoff,
                              sizeof(low_rankdirectoff)),
           "p10dc rankchunk32 directoff ptr");

        std::cerr << "p10dc_low_rankchunk32_directmask fixed_owner=" << fixed
                  << " stride=" << stride
                  << " mask_entries=" << low_rankdirectmask_count
                  << " offset_entries=" << low_rankdirectoff_count
                  << " mask_mib=" << double(low_rankdirectmask_count) / double(1 << 20)
                  << " offset_mib=" << double(low_rankdirectoff_count * sizeof(uint32_t)) / double(1 << 20)
                  << " depth_major=1 coalesced_warp_byte_load=1 coalesced_offset32_load=1"
                  << " cross5_runtime_lut=0 cross5_runtime_state=0"
                  << " rankchunk_meta_runtime=0 blockbase_runtime=0 blockbase_shuffle_runtime=0"
                  << " zero_mask_skips_offset_and_rankstream=1 mask_or=0x"
                  << std::hex << unsigned(mask_or) << std::dec << '\n';
    }

    void release() {
        if (low_rankdirectmask) cudaFree(low_rankdirectmask);
        if (low_rankdirectoff) cudaFree(low_rankdirectoff);
        low_rankdirectmask = nullptr;
        low_rankdirectoff = nullptr;
        low_rankdirectmask_count = low_rankdirectoff_count = 0;
        low_rankdirectmask_capacity = low_rankdirectoff_capacity = 0;
        BucketFusedDirectHighRowsRankChunk32Tables::release();
    }
};

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_rankchunk32_directmask_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
    if (!depth) return BkczCrossAccum(0);
    uint8_t pending = p10dc_low_rankchunk32_directmask_load(h, rank, depth);
    if (!pending) return BkczCrossAccum(0);

    const uint16_t* rank_row = p10dc_low_rankchunk32_directoff_row(h, rank);
    BkczCrossAccum sum = 0;
#pragma unroll
    for (uint32_t ordinal = 0; ordinal < P10DC_RANKCHUNK32_MAX_L_PER_LEGAL_CODE; ++ordinal) {
        if (pending & uint8_t(1u << ordinal)) {
            const uint16_t source_rank = rank_row[ordinal];
            sum = bkcz_cross_add(sum, source_row[uint32_t(source_rank)]);
        }
    }
    return sum;
}

__device__ __forceinline__ Count
p10dc_direct_resolved_high_plan_sum_rankchunk32_directmask(
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
        sum += p10dc_resolved_low_preimages_rankchunk32_directmask_fixed(
            db.hs, lr, c.cross_depth, c.cross_base);
#else
        sum = gpu_direct_add(
            sum, p10dc_resolved_low_preimages_rankchunk32_directmask_fixed(
                     db.hs, lr, c.cross_depth, c.cross_base));
#endif
    }
#if GPU_DIRECT_PM_ACCUM
    return gpu_direct_pm_reduce_u64(sum);
#else
    return sum;
#endif
}

#endif  // P10DC_RANKCHUNK32_DIRECTMASK
