#pragma once

#include "ramstream32_bucket_low_rankchunk32.cuh"

static_assert(P10DC_RANKCHUNK32_BYTEPACK == 1,
              "rankchunk32 basepair64 requires bytepack24+8 metadata");
static_assert(P10DC_RANKCHUNK32_BLOCK64 == 0,
              "rankchunk32 basepair64 keeps 32-code prefix blocks");
static_assert(P10DC_RANKCHUNK32_BLOCK == 32u);
static_assert(P10DC_RANKCHUNK32_PREFIX_BITS == 8u);

static constexpr uint32_t P10DC_RANKCHUNK32_BASEPAIR_BASE_BITS = 22u;
static constexpr uint32_t P10DC_RANKCHUNK32_BASEPAIR_DELTA_BITS = 8u;
static constexpr uint32_t P10DC_RANKCHUNK32_BASEPAIR_BASE_MASK =
    (1u << P10DC_RANKCHUNK32_BASEPAIR_BASE_BITS) - 1u;
static constexpr uint32_t P10DC_RANKCHUNK32_BASEPAIR_DELTA_MASK =
    (1u << P10DC_RANKCHUNK32_BASEPAIR_DELTA_BITS) - 1u;
static_assert(P10DC_RANKCHUNK32_BASEPAIR_BASE_BITS +
              P10DC_RANKCHUNK32_BASEPAIR_DELTA_BITS <= 32u);
static_assert(32u * P10DC_RANKCHUNK32_MAX_L_PER_LEGAL_CODE <
              (1u << P10DC_RANKCHUNK32_BASEPAIR_DELTA_BITS));

__device__ __forceinline__ uint32_t p10dc_rankchunk32_basepair64_decode(
    uint32_t pair, uint32_t compact
) {
    const uint32_t base = pair & P10DC_RANKCHUNK32_BASEPAIR_BASE_MASK;
    const uint32_t delta =
        (pair >> P10DC_RANKCHUNK32_BASEPAIR_BASE_BITS) &
        P10DC_RANKCHUNK32_BASEPAIR_DELTA_MASK;
    return base + (((compact >> 5) & 1u) ? delta : 0u);
}

__device__ __forceinline__ void p10dc_low_rankchunk32_basepair64_row(
    uint32_t h, uint32_t rank, uint32_t& packed_chunks, const uint16_t*& row
) {
    const uint32_t compact = D_P10DC_LOW_RANKCHUNK_HOFF[h] + rank;
    const uint32_t meta = D_P10DC_LOW_RANKCHUNKMETA32[compact];
    packed_chunks = meta & P10DC_RANKCHUNK32_CHUNK_MASK;
    const uint32_t prefix = meta >> P10DC_RANKCHUNK32_CHUNK_BITS;
    const uint32_t pair = D_P10DC_LOW_RANKCHUNKBLOCK16[compact >> 6];
    const uint32_t block_base = p10dc_rankchunk32_basepair64_decode(pair, compact);
    row = D_P10DC_LOW_RANKSTREAM + block_base + prefix;
}

// A packed pair owns two adjacent 32-code prefix blocks. With aligned heights,
// lane0 loads the pair for the whole stripe. Without padding, a 32-lane stripe
// can cross one block32 boundary. If both block32 bases live in the same pair,
// one pair load still suffices; only odd->even block crossings need a second
// pair, loaded by the boundary lane and selected through one variable shuffle.
__device__ __forceinline__ void p10dc_low_rankchunk32_basepair64_row_warpstripe(
    uint32_t h, uint32_t rank, uint32_t& packed_chunks, const uint16_t*& row
) {
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const unsigned active = __activemask();
    const uint32_t compact = D_P10DC_LOW_RANKCHUNK_HOFF[h] + rank;
    const uint32_t meta = D_P10DC_LOW_RANKCHUNKMETA32[compact];
    packed_chunks = meta & P10DC_RANKCHUNK32_CHUNK_MASK;
    const uint32_t prefix = meta >> P10DC_RANKCHUNK32_CHUNK_BITS;
    const uint32_t first_compact = compact - lane;

#if P10DC_RANKCHUNK32_ALIGN32
    uint32_t local_pair = 0;
    if (lane == 0u)
        local_pair = D_P10DC_LOW_RANKCHUNKBLOCK16[first_compact >> 6];
    const uint32_t pair = __shfl_sync(active, local_pair, 0);
#else
    const uint32_t first_block32 = first_compact >> 5;
    const uint32_t first_off = first_compact & 31u;
    const uint32_t split_lane = 32u - first_off;
    const uint32_t first_pair = first_block32 >> 1;
    const uint32_t second_pair = (first_block32 + 1u) >> 1;
    const bool pair_changes = split_lane < 32u && second_pair != first_pair;
    uint32_t local_pair = 0;
    if (lane == 0u)
        local_pair = D_P10DC_LOW_RANKCHUNKBLOCK16[first_pair];
    if (pair_changes && lane == split_lane)
        local_pair = D_P10DC_LOW_RANKCHUNKBLOCK16[second_pair];
    const uint32_t source_lane =
        (pair_changes && lane >= split_lane) ? split_lane : 0u;
    const uint32_t pair = __shfl_sync(active, local_pair, int(source_lane));
#endif
    const uint32_t block_base = p10dc_rankchunk32_basepair64_decode(pair, compact);
    row = D_P10DC_LOW_RANKSTREAM + block_base + prefix;
}

struct BucketFusedDirectHighRowsRankChunk32BasePair64Tables
    : BucketFusedDirectHighRowsRankChunk32Tables {
    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsRankChunk32Tables::bind_owner(fixed, buckets, slot);
        const size_t base32_count = low_rankchunkblock16_count;
        std::vector<uint32_t> base32(base32_count);
        if (base32_count) {
            ck(cudaMemcpy(base32.data(), low_rankchunkblock16,
                          base32_count * sizeof(uint32_t), cudaMemcpyDeviceToHost),
               "p10dc rankchunk32 basepair64 base32 D2H");
        }
        std::vector<uint32_t> pairs((base32_count + 1u) >> 1);
        for (size_t i = 0; i < pairs.size(); ++i) {
            const uint32_t b0 = base32[2u * i];
            const uint32_t b1 = 2u * i + 1u < base32_count ? base32[2u * i + 1u] : b0;
            if (b1 < b0 ||
                b0 > P10DC_RANKCHUNK32_BASEPAIR_BASE_MASK ||
                b1 - b0 > P10DC_RANKCHUNK32_BASEPAIR_DELTA_MASK) {
                std::cerr << "p10dc rankchunk32 basepair64 packing failure owner=" << fixed
                          << " pair=" << i << " b0=" << b0 << " b1=" << b1
                          << " delta=" << (b1 >= b0 ? b1 - b0 : 0xffffffffu) << '\n';
                std::exit(665);
            }
            pairs[i] = b0 | ((b1 - b0) << P10DC_RANKCHUNK32_BASEPAIR_BASE_BITS);
        }

        if (low_rankchunkblock16) cudaFree(low_rankchunkblock16);
        low_rankchunkblock16 = nullptr;
        low_rankchunkblock16_count = pairs.size();
        low_rankchunkblock16_capacity = pairs.size();
        if (!pairs.empty()) {
            ck(cudaMalloc(&low_rankchunkblock16, pairs.size() * sizeof(uint32_t)),
               "p10dc rankchunk32 basepair64 alloc");
            ck(cudaMemcpy(low_rankchunkblock16, pairs.data(), pairs.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "p10dc rankchunk32 basepair64 H2D");
        }
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKCHUNKBLOCK16, &low_rankchunkblock16,
                              sizeof(low_rankchunkblock16)),
           "p10dc rankchunk32 basepair64 ptr");

        const size_t pair_bytes = pairs.size() * sizeof(uint32_t);
        std::cerr << "p10dc_low_rankchunk32_basepair64 fixed_owner=" << fixed
                  << " base32_entries=" << base32_count
                  << " pair_entries=" << pairs.size()
                  << " pair_bytes=" << pair_bytes
                  << " base_bits=22 delta_bits=8"
                  << " codes_per_pair=64 block_base_bytes_per_code=0.0625"
                  << " height_align=" << P10DC_RANKCHUNK32_HEIGHT_ALIGN
                  << " block_base_loads_per_warp_max="
                  << (P10DC_RANKCHUNK32_ALIGN32 ? 1 : 2)
                  << " byte_aligned_chunks=1\n";
    }
};
