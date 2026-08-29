#pragma once

#include "ramstream32_bucket_closure_pattern10_depthcode_delta_direct_affine_rankchunk32.cuh"

#ifndef P10DC_RANKCHUNK32_DIRECTMASK
#define P10DC_RANKCHUNK32_DIRECTMASK 0
#endif
static_assert(P10DC_RANKCHUNK32_DIRECTMASK == 0 || P10DC_RANKCHUNK32_DIRECTMASK == 1,
              "P10DC_RANKCHUNK32_DIRECTMASK must be 0 or 1");

#ifndef P10DC_RANKCHUNK32_RANKPLANE
#define P10DC_RANKCHUNK32_RANKPLANE 0
#endif
static_assert(P10DC_RANKCHUNK32_RANKPLANE == 0 || P10DC_RANKCHUNK32_RANKPLANE == 1,
              "P10DC_RANKCHUNK32_RANKPLANE must be 0 or 1");
static_assert(!P10DC_RANKCHUNK32_RANKPLANE || P10DC_RANKCHUNK32_DIRECTMASK,
              "rankplane is only valid with directmask");

#if P10DC_RANKCHUNK32_DIRECTMASK

// CROSS5 and rank projection depend only on LOW code + incoming depth. Store a
// depth-major final ordinal mask. The optional ordinal-major rank plane stores
// the uint16 source rank for each L ordinal directly, so the hot chain becomes
//   mask8 -> rank16 plane -> source32
// instead of
//   mask8 -> offset32 -> rank16 -> source32.
// Pad every direct plane to 128 entries. With height ALIGN32, mask warps issue
// aligned 32-byte spans and uint16 rank planes issue aligned 64-byte spans;
// neither span crosses a 128-byte plane boundary.
static constexpr size_t P10DC_RANKDIRECT_STRIDE_ALIGN = 128u;
__constant__ uint8_t* D_P10DC_LOW_RANKDIRECTMASK;
__constant__ uint32_t D_P10DC_LOW_RANKDIRECTMASK_STRIDE;
__constant__ uint32_t* D_P10DC_LOW_RANKDIRECTOFF;
#if P10DC_RANKCHUNK32_RANKPLANE
__constant__ uint16_t* D_P10DC_LOW_RANKDIRECTPLANE;
#endif

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

__device__ __forceinline__ uint8_t p10dc_low_rankchunk32_directmask_load_compact(
    uint32_t compact, uint32_t depth
) {
    const size_t ix = size_t(depth) * D_P10DC_LOW_RANKDIRECTMASK_STRIDE + compact;
    return __ldg(D_P10DC_LOW_RANKDIRECTMASK + ix);
}

__device__ __forceinline__ uint8_t p10dc_low_rankchunk32_directmask_load(
    uint32_t h, uint32_t rank, uint32_t depth
) {
    return p10dc_low_rankchunk32_directmask_load_compact(
        p10dc_low_rankchunk32_directcompact(h, rank), depth);
}

__device__ __forceinline__ const uint16_t* p10dc_low_rankchunk32_directoff_row_compact(
    uint32_t compact
) {
    const uint32_t off = __ldg(D_P10DC_LOW_RANKDIRECTOFF + compact);
    return D_P10DC_LOW_RANKSTREAM + off;
}

__device__ __forceinline__ const uint16_t* p10dc_low_rankchunk32_directoff_row(
    uint32_t h, uint32_t rank
) {
    return p10dc_low_rankchunk32_directoff_row_compact(
        p10dc_low_rankchunk32_directcompact(h, rank));
}

#if P10DC_RANKCHUNK32_RANKPLANE
__device__ __forceinline__ uint16_t p10dc_low_rankchunk32_directrank_load_compact(
    uint32_t compact, uint32_t ordinal
) {
    const size_t ix = size_t(ordinal) * D_P10DC_LOW_RANKDIRECTMASK_STRIDE + compact;
    return __ldg(D_P10DC_LOW_RANKDIRECTPLANE + ix);
}
#endif

struct BucketFusedDirectHighRowsRankChunk32DirectMaskTables
    : BucketFusedDirectHighRowsRankChunk32Tables {
    uint8_t* low_rankdirectmask = nullptr;
    uint32_t* low_rankdirectoff = nullptr;
#if P10DC_RANKCHUNK32_RANKPLANE
    uint16_t* low_rankdirectplane = nullptr;
#endif
    size_t low_rankdirect_stride = 0;
    size_t low_rankdirectmask_count = 0, low_rankdirectoff_count = 0;
    size_t low_rankdirectmask_capacity = 0, low_rankdirectoff_capacity = 0;
#if P10DC_RANKCHUNK32_RANKPLANE
    size_t low_rankdirectplane_count = 0, low_rankdirectplane_capacity = 0;
#endif

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
        constexpr size_t NRANK = size_t(P10DC_RANKCHUNK32_MAX_L_PER_LEGAL_CODE);
        constexpr size_t RANK_STORAGE = NRANK ? NRANK : 1u;
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());
        const size_t compact_count = low_rankchunkmeta32_count;
        const size_t direct_stride =
            (compact_count + P10DC_RANKDIRECT_STRIDE_ALIGN - 1u) &
            ~(P10DC_RANKDIRECT_STRIDE_ALIGN - 1u);
        if (direct_stride > uint64_t(0xffffffffu)) {
            std::cerr << "p10dc rankchunk32 direct stride overflow entries="
                      << direct_stride << '\n';
            std::exit(688);
        }
        std::vector<uint8_t> mask(size_t(16) * direct_stride, uint8_t(0));
        std::vector<uint32_t> off(compact_count, uint32_t(0));
#if P10DC_RANKCHUNK32_RANKPLANE
        std::vector<uint16_t> plane(RANK_STORAGE * direct_stride, uint16_t(0));
#endif

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
                if (compact >= compact_count) {
                    std::cerr << "p10dc rankchunk32 directmask compact overflow owner="
                              << fixed << " h=" << h << " compact=" << compact
                              << " compact_count=" << compact_count << '\n';
                    std::exit(683);
                }
                off[compact] = stream_cursor;
                const uint32_t code = f.low_codes[i];
                const uint32_t key = gpu_direct_ternary_key_host(code, LOW_LUT_K);
                const uint32_t packed = p10dc_rankchunk32_pack_host(key);
                for (uint32_t depth = 1; depth <= 15u; ++depth) {
                    const uint8_t m = p10dc_rankchunk32_directmask_host(packed, depth);
                    mask[size_t(depth) * direct_stride + compact] = m;
                    mask_or = uint8_t(mask_or | m);
                }

                uint32_t ordinal = 0;
                uint32_t weight = bkcz_pow3_const(LOW_LUT_K - 1);
                for (int pos = LOW_LUT_K - 1; pos >= 0; --pos) {
                    if (((code >> (2 * pos)) & 3u) == uint32_t(::L)) {
#if P10DC_RANKCHUNK32_RANKPLANE
                        const uint32_t cand_key = key - weight;
                        if (cand_key >= f.low_direct.size()) {
                            std::cerr << "p10dc directrank key overflow owner=" << fixed
                                      << " h=" << h << " pos=" << pos << '\n';
                            std::exit(685);
                        }
                        const uint32_t x = f.low_direct[cand_key];
                        if (x == BKF_DIRECT_INVALID || bkf_loc_owner(x & BKF_LOC_MASK) != fixed ||
                            bkf_loc_rank(x & BKF_LOC_MASK) >= 0xffffu || ordinal >= NRANK) {
                            std::cerr << "p10dc directrank invariant failure owner=" << fixed
                                      << " h=" << h << " pos=" << pos
                                      << " ordinal=" << ordinal << '\n';
                            std::exit(686);
                        }
                        plane[size_t(ordinal) * direct_stride + compact] =
                            uint16_t(bkf_loc_rank(x & BKF_LOC_MASK));
#endif
                        ++ordinal;
                        ++stream_cursor;
                    }
                    if (pos) weight /= 3u;
                }
                if (ordinal > NRANK) {
                    std::cerr << "p10dc directrank ordinal overflow owner=" << fixed
                              << " h=" << h << " count=" << ordinal << '\n';
                    std::exit(687);
                }
            }
        }
        if (compact != compact_count || actual_codes != low_prekey_count ||
            stream_cursor != low_rankstream_count || (mask_or & 0x80u) != 0u) {
            std::cerr << "p10dc rankchunk32 directmask shape mismatch owner=" << fixed
                      << " compact=" << compact << '/' << compact_count
                      << " codes=" << actual_codes << '/' << low_prekey_count
                      << " stream=" << stream_cursor << '/' << low_rankstream_count
                      << " mask_or=0x" << std::hex << unsigned(mask_or) << std::dec << '\n';
            std::exit(684);
        }

        low_rankdirect_stride = direct_stride;
        low_rankdirectmask_count = mask.size();
        low_rankdirectoff_count = off.size();
#if P10DC_RANKCHUNK32_RANKPLANE
        low_rankdirectplane_count = plane.size();
#endif
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
#if P10DC_RANKCHUNK32_RANKPLANE
        if (low_rankdirectplane_count > low_rankdirectplane_capacity) {
            if (low_rankdirectplane) cudaFree(low_rankdirectplane);
            low_rankdirectplane = nullptr;
            low_rankdirectplane_capacity = low_rankdirectplane_count;
            if (low_rankdirectplane_capacity)
                ck(cudaMalloc(&low_rankdirectplane,
                              low_rankdirectplane_capacity * sizeof(uint16_t)),
                   "p10dc rankchunk32 directplane alloc");
        }
#endif
        if (!mask.empty())
            ck(cudaMemcpy(low_rankdirectmask, mask.data(), mask.size() * sizeof(uint8_t),
                          cudaMemcpyHostToDevice),
               "p10dc rankchunk32 directmask H2D");
        if (!off.empty())
            ck(cudaMemcpy(low_rankdirectoff, off.data(), off.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "p10dc rankchunk32 directoff H2D");
#if P10DC_RANKCHUNK32_RANKPLANE
        if (!plane.empty())
            ck(cudaMemcpy(low_rankdirectplane, plane.data(), plane.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice),
               "p10dc rankchunk32 directplane H2D");
#endif
        const uint32_t stride32 = uint32_t(direct_stride);
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKDIRECTMASK, &low_rankdirectmask,
                              sizeof(low_rankdirectmask)),
           "p10dc rankchunk32 directmask ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKDIRECTMASK_STRIDE, &stride32,
                              sizeof(stride32)),
           "p10dc rankchunk32 directmask stride");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKDIRECTOFF, &low_rankdirectoff,
                              sizeof(low_rankdirectoff)),
           "p10dc rankchunk32 directoff ptr");
#if P10DC_RANKCHUNK32_RANKPLANE
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKDIRECTPLANE, &low_rankdirectplane,
                              sizeof(low_rankdirectplane)),
           "p10dc rankchunk32 directplane ptr");
#endif

        std::cerr << "p10dc_low_rankchunk32_directmask fixed_owner=" << fixed
                  << " compact_count=" << compact_count
                  << " direct_stride=" << direct_stride
                  << " direct_stride_align=" << P10DC_RANKDIRECT_STRIDE_ALIGN
                  << " mask_entries=" << low_rankdirectmask_count
                  << " offset_entries=" << low_rankdirectoff_count
                  << " mask_mib=" << double(low_rankdirectmask_count) / double(1 << 20)
                  << " offset_mib=" << double(low_rankdirectoff_count * sizeof(uint32_t)) / double(1 << 20)
#if P10DC_RANKCHUNK32_RANKPLANE
                  << " rankplane_entries=" << low_rankdirectplane_count
                  << " rankplane_mib=" << double(low_rankdirectplane_count * sizeof(uint16_t)) / double(1 << 20)
#endif
                  << " depth_major_mask=1 ordinal_major_rank=" << P10DC_RANKCHUNK32_RANKPLANE
                  << " aligned_mask_warp32=1 aligned_rankplane_warp64="
                  << (P10DC_RANKCHUNK32_RANKPLANE && P10DC_RANKCHUNK32_ALIGN32)
                  << " cross5_runtime_lut=0 cross5_runtime_state=0"
                  << " rankchunk_meta_runtime=0 blockbase_runtime=0 blockbase_shuffle_runtime=0"
#if P10DC_RANKCHUNK32_RANKPLANE
                  << " offset_runtime=0 dependency_chain=mask8_rank16_source32"
#else
                  << " dependency_chain=mask8_offset32_rank16_source32"
#endif
                  << " mask_or=0x" << std::hex << unsigned(mask_or) << std::dec << '\n';
    }

    void release() {
        if (low_rankdirectmask) cudaFree(low_rankdirectmask);
        if (low_rankdirectoff) cudaFree(low_rankdirectoff);
#if P10DC_RANKCHUNK32_RANKPLANE
        if (low_rankdirectplane) cudaFree(low_rankdirectplane);
        low_rankdirectplane = nullptr;
        low_rankdirectplane_count = low_rankdirectplane_capacity = 0;
#endif
        low_rankdirectmask = nullptr;
        low_rankdirectoff = nullptr;
        low_rankdirect_stride = 0;
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
    const uint32_t compact = p10dc_low_rankchunk32_directcompact(h, rank);
    const uint8_t pending = p10dc_low_rankchunk32_directmask_load_compact(compact, depth);
    if (!pending) return BkczCrossAccum(0);

#if !P10DC_RANKCHUNK32_RANKPLANE
    const uint16_t* rank_row = p10dc_low_rankchunk32_directoff_row_compact(compact);
#endif
    BkczCrossAccum sum = 0;
#pragma unroll
    for (uint32_t ordinal = 0; ordinal < P10DC_RANKCHUNK32_MAX_L_PER_LEGAL_CODE; ++ordinal) {
        if (pending & uint8_t(1u << ordinal)) {
#if P10DC_RANKCHUNK32_RANKPLANE
            const uint16_t source_rank =
                p10dc_low_rankchunk32_directrank_load_compact(compact, ordinal);
#else
            const uint16_t source_rank = rank_row[ordinal];
#endif
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
