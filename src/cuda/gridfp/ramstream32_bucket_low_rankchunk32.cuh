#pragma once

#include "ramstream32_bucket_low_prekey_rankstream.cuh"
#include "ramstream32_bucket_closure_cross5_common.cuh"

#ifndef P10DC_RANKCHUNK32_ONESHFL
#define P10DC_RANKCHUNK32_ONESHFL 1
#endif
static_assert(P10DC_RANKCHUNK32_ONESHFL == 0 || P10DC_RANKCHUNK32_ONESHFL == 1,
              "P10DC_RANKCHUNK32_ONESHFL must be 0 or 1");

// Three CROSS5 chunks need only 23 bits for LOW_LUT_K<=14: the first two
// chunks use 8 bits each (0..242), while the final chunk has at most four
// ternary digits and therefore fits in 7 bits (0..80). The reclaimed bit
// grows the within-block rankstream prefix from 8 to 9 bits, allowing 32-code
// metadata blocks with no per-height padding.
static constexpr uint32_t P10DC_RANKCHUNK32_BLOCK_LOG2 = 5u;
static constexpr uint32_t P10DC_RANKCHUNK32_BLOCK = 1u << P10DC_RANKCHUNK32_BLOCK_LOG2;
static constexpr uint32_t P10DC_RANKCHUNK32_HEIGHT_ALIGN = 1u; // compatibility: no padding
static constexpr uint32_t P10DC_RANKCHUNK32_CHUNK_BITS = 23u;
static constexpr uint32_t P10DC_RANKCHUNK32_PREFIX_BITS = 9u;
static constexpr uint32_t P10DC_RANKCHUNK32_CHUNK_MASK = (1u << P10DC_RANKCHUNK32_CHUNK_BITS) - 1u;
static_assert(P10DC_RANKCHUNK32_CHUNK_BITS + P10DC_RANKCHUNK32_PREFIX_BITS == 32u);
static_assert(LOW_LUT_K <= 14, "rankchunk32 assumes at most three CROSS5 chunks");
static_assert((P10DC_RANKCHUNK32_BLOCK - 1u) * uint32_t(LOW_LUT_K) <
              (1u << P10DC_RANKCHUNK32_PREFIX_BITS),
              "rankchunk32 worst-case within-block prefix no longer fits 9 bits");
static_assert(p10dc_cross5_pow3_host(4) <= (1u << 7),
              "rankchunk32 four-digit tail no longer fits 7 bits");

static constexpr uint32_t p10dc_rankchunk32_pack_host(uint32_t key) {
    constexpr int L0 = LOW_LUT_K >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    const uint32_t c0 = (key / p10dc_cross5_pow3_host(S0)) % p10dc_cross5_pow3_host(L0);
    uint32_t c1 = 0, c2 = 0;
    if constexpr (S0 > 0) {
        constexpr int L1 = S0 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S0;
        constexpr int S1 = S0 - L1;
        c1 = (key / p10dc_cross5_pow3_host(S1)) % p10dc_cross5_pow3_host(L1);
        if constexpr (S1 > 0) {
            constexpr int L2 = S1 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S1;
            constexpr int S2 = S1 - L2;
            static_assert(S2 == 0, "rankchunk32 K<=14 must fit in three chunks");
            static_assert(L2 <= 4, "rankchunk32 tail must fit the 7-bit third chunk");
            c2 = key % p10dc_cross5_pow3_host(L2);
        }
    }
    return c0 | (c1 << 8) | (c2 << 16);
}

static constexpr uint32_t p10dc_rankchunk32_unpack_host(uint32_t packed) {
    constexpr int L0 = LOW_LUT_K >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    uint32_t key = (packed & 0xffu) * p10dc_cross5_pow3_host(S0);
    if constexpr (S0 > 0) {
        constexpr int L1 = S0 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S0;
        constexpr int S1 = S0 - L1;
        key += ((packed >> 8) & 0xffu) * p10dc_cross5_pow3_host(S1);
        if constexpr (S1 > 0) key += (packed >> 16) & 0x7fu;
    }
    return key;
}

__constant__ uint32_t* D_P10DC_LOW_RANKCHUNKMETA32;
// Historical symbol name retained to avoid touching callers; blocks are now 32 codes.
__constant__ uint32_t* D_P10DC_LOW_RANKCHUNKBLOCK16;
// Kept as a compatibility/verification table. With padding removed these are
// exactly the compact fixed-owner height offsets.
__constant__ uint32_t D_P10DC_LOW_RANKCHUNK_HOFF[MAXW + 2];

__device__ __forceinline__ void p10dc_low_rankchunk32_row(
    uint32_t h, uint32_t rank, uint32_t& packed_chunks, const uint16_t*& row
) {
    const uint32_t compact = D_P10DC_LOW_RANKCHUNK_HOFF[h] + rank;
    const uint32_t meta = D_P10DC_LOW_RANKCHUNKMETA32[compact];
    packed_chunks = meta & P10DC_RANKCHUNK32_CHUNK_MASK;
    const uint32_t prefix = meta >> P10DC_RANKCHUNK32_CHUNK_BITS;
    const uint32_t block_base = D_P10DC_LOW_RANKCHUNKBLOCK16[
        compact >> P10DC_RANKCHUNK32_BLOCK_LOG2];
    row = D_P10DC_LOW_RANKSTREAM + block_base + prefix;
}

// A contiguous 32-lane stripe can intersect at most two 32-code metadata
// blocks even when a height begins at an arbitrary compact index. At most two
// lanes load block bases. In the default path every lane chooses lane 0 or the
// boundary lane as its source in one variable-index shuffle. The two-shuffle
// form remains available as an A/B baseline.
__device__ __forceinline__ void p10dc_low_rankchunk32_row_warpstripe(
    uint32_t h, uint32_t rank, uint32_t& packed_chunks, const uint16_t*& row
) {
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const unsigned active = __activemask();
    const uint32_t compact = D_P10DC_LOW_RANKCHUNK_HOFF[h] + rank;
    const uint32_t meta = D_P10DC_LOW_RANKCHUNKMETA32[compact];
    packed_chunks = meta & P10DC_RANKCHUNK32_CHUNK_MASK;
    const uint32_t prefix = meta >> P10DC_RANKCHUNK32_CHUNK_BITS;

    const uint32_t first_compact = compact - lane;
    const uint32_t first_block = first_compact >> P10DC_RANKCHUNK32_BLOCK_LOG2;
    const uint32_t first_off = first_compact & (P10DC_RANKCHUNK32_BLOCK - 1u);
    const uint32_t split_lane = P10DC_RANKCHUNK32_BLOCK - first_off; // [1,32]

#if P10DC_RANKCHUNK32_ONESHFL
    uint32_t local_base = 0;
    if (lane == 0u)
        local_base = D_P10DC_LOW_RANKCHUNKBLOCK16[first_block];
    if (split_lane < 32u && lane == split_lane)
        local_base = D_P10DC_LOW_RANKCHUNKBLOCK16[first_block + 1u];
    const uint32_t source_lane =
        (split_lane < 32u && lane >= split_lane) ? split_lane : 0u;
    const uint32_t block_base = __shfl_sync(active, local_base, int(source_lane));
#else
    uint32_t b0_local = 0;
    if (lane == 0u) b0_local = D_P10DC_LOW_RANKCHUNKBLOCK16[first_block];
    uint32_t block_base = __shfl_sync(active, b0_local, 0);
    if (split_lane < 32u) {
        const unsigned split_bit = 1u << split_lane;
        if (active & split_bit) {
            uint32_t b1_local = 0;
            if (lane == split_lane)
                b1_local = D_P10DC_LOW_RANKCHUNKBLOCK16[first_block + 1u];
            const uint32_t b1 = __shfl_sync(active, b1_local, int(split_lane));
            if (lane >= split_lane) block_base = b1;
        }
    }
#endif
    row = D_P10DC_LOW_RANKSTREAM + block_base + prefix;
}

struct BucketFusedDirectHighRowsRankChunk32Tables
    : BucketFusedDirectHighRowsPrekeyRankStreamTables {
    uint32_t* low_rankchunkmeta32 = nullptr;
    uint32_t* low_rankchunkblock16 = nullptr;
    size_t low_rankchunkmeta32_count = 0, low_rankchunkblock16_count = 0;
    size_t low_rankchunkmeta32_capacity = 0, low_rankchunkblock16_capacity = 0;
    size_t low_rankchunk_padding_count = 0;

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsPrekeyRankStreamTables::bind_owner(fixed, buckets, slot);
        if (!host_fused) { std::cerr << "p10dc rankchunk32 missing host fused metadata\n"; std::exit(661); }
        const BucketFusedHost& f = *host_fused;
        constexpr size_t P = size_t(MAXW + 2);
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());

        std::array<uint32_t, MAXW + 2> hoff{};
        std::vector<uint32_t> meta, blocks;
        meta.reserve(low_prekey_count);
        blocks.reserve((low_prekey_count + P10DC_RANKCHUNK32_BLOCK - 1u) /
                       P10DC_RANKCHUNK32_BLOCK);
        uint32_t stream_cursor = 0, block_base = 0;

        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
            hoff[h] = uint32_t(meta.size());
            const uint32_t a = f.low_code_off[owner_base + h];
            const uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u] : owner_end;
            for (uint32_t i = a; i < b; ++i) {
                const uint32_t compact = uint32_t(meta.size());
                if ((compact & (P10DC_RANKCHUNK32_BLOCK - 1u)) == 0u) {
                    block_base = stream_cursor;
                    blocks.push_back(block_base);
                }
                const uint32_t prefix = stream_cursor - block_base;
                const uint32_t code = f.low_codes[i];
                const uint32_t key = gpu_direct_ternary_key_host(code, LOW_LUT_K);
                const uint32_t chunks = p10dc_rankchunk32_pack_host(key);
                if ((chunks >> P10DC_RANKCHUNK32_CHUNK_BITS) != 0u ||
                    p10dc_rankchunk32_unpack_host(chunks) != key ||
                    prefix >= (1u << P10DC_RANKCHUNK32_PREFIX_BITS)) {
                    std::cerr << "p10dc rankchunk32 packing failure key=" << key
                              << " chunks=" << chunks << " prefix=" << prefix << '\n';
                    std::exit(662);
                }
                meta.push_back(chunks | (prefix << P10DC_RANKCHUNK32_CHUNK_BITS));
                for (int pos = 0; pos < LOW_LUT_K; ++pos)
                    if (((code >> (2 * pos)) & 3u) == uint32_t(::L)) ++stream_cursor;
            }
        }
        if (meta.size() != low_prekey_count || stream_cursor != low_rankstream_count) {
            std::cerr << "p10dc rankchunk32 size mismatch meta=" << meta.size()
                      << '/' << low_prekey_count << " stream=" << stream_cursor
                      << '/' << low_rankstream_count << '\n';
            std::exit(663);
        }

        low_rankchunkmeta32_count = meta.size();
        low_rankchunkblock16_count = blocks.size();
        low_rankchunk_padding_count = 0;
        if (low_rankchunkmeta32_count > low_rankchunkmeta32_capacity) {
            if (low_rankchunkmeta32) cudaFree(low_rankchunkmeta32);
            low_rankchunkmeta32 = nullptr;
            low_rankchunkmeta32_capacity = low_rankchunkmeta32_count;
            if (low_rankchunkmeta32_capacity) ck(cudaMalloc(&low_rankchunkmeta32,
                low_rankchunkmeta32_capacity * sizeof(uint32_t)), "p10dc rankchunk32 meta alloc");
        }
        if (low_rankchunkblock16_count > low_rankchunkblock16_capacity) {
            if (low_rankchunkblock16) cudaFree(low_rankchunkblock16);
            low_rankchunkblock16 = nullptr;
            low_rankchunkblock16_capacity = low_rankchunkblock16_count;
            if (low_rankchunkblock16_capacity) ck(cudaMalloc(&low_rankchunkblock16,
                low_rankchunkblock16_capacity * sizeof(uint32_t)), "p10dc rankchunk32 block alloc");
        }
        if (!meta.empty()) ck(cudaMemcpy(low_rankchunkmeta32, meta.data(), meta.size() * sizeof(uint32_t),
                                         cudaMemcpyHostToDevice), "p10dc rankchunk32 meta H2D");
        if (!blocks.empty()) ck(cudaMemcpy(low_rankchunkblock16, blocks.data(), blocks.size() * sizeof(uint32_t),
                                           cudaMemcpyHostToDevice), "p10dc rankchunk32 blocks H2D");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKCHUNKMETA32, &low_rankchunkmeta32,
                              sizeof(low_rankchunkmeta32)), "p10dc rankchunk32 meta ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKCHUNKBLOCK16, &low_rankchunkblock16,
                              sizeof(low_rankchunkblock16)), "p10dc rankchunk32 block ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKCHUNK_HOFF, hoff.data(),
                              hoff.size() * sizeof(uint32_t)), "p10dc rankchunk32 height offsets");

        if (low_prekey) cudaFree(low_prekey);
        low_prekey = nullptr; low_prekey_capacity = 0;
        if (low_rankstream_off) cudaFree(low_rankstream_off);
        low_rankstream_off = nullptr; low_rankstream_off_capacity = 0;
        uint32_t* null32 = nullptr;
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_PREKEY, &null32, sizeof(null32)), "p10dc rankchunk32 null old prekey ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKSTREAM_OFF, &null32, sizeof(null32)), "p10dc rankchunk32 null old offset ptr");

        const size_t bytes = meta.size() * sizeof(uint32_t) + blocks.size() * sizeof(uint32_t) +
                             low_rankstream_count * sizeof(uint16_t);
        std::cerr << "p10dc_low_rankchunk32 fixed_owner=" << fixed
                  << " codes=" << meta.size() << " blocks=" << blocks.size()
                  << " l_ranks=" << low_rankstream_count << " bytes=" << bytes
                  << " meta_entries=" << meta.size() << " padding=0"
                  << " chunk_bits=23 prefix_bits=9 block=32 height_align=1"
                  << " block_base_loads_per_warp_max=2"
                  << " block_base_shuffles_per_warp=" << (P10DC_RANKCHUNK32_ONESHFL ? 1 : 2)
                  << " chunk_div_runtime=0 chunk_mod_runtime=0"
                  << " old_prekey_offset_arrays_freed=1 direct_lookup_runtime=0\n";
    }

    void release() {
        if (low_rankchunkmeta32) cudaFree(low_rankchunkmeta32);
        if (low_rankchunkblock16) cudaFree(low_rankchunkblock16);
        low_rankchunkmeta32 = low_rankchunkblock16 = nullptr;
        low_rankchunkmeta32_count = low_rankchunkblock16_count = 0;
        low_rankchunkmeta32_capacity = low_rankchunkblock16_capacity = 0;
        low_rankchunk_padding_count = 0;
        BucketFusedDirectHighRowsPrekeyRankStreamTables::release();
    }
};
