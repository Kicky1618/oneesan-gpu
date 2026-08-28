#pragma once

#include "ramstream32_bucket_low_rankchunk32.cuh"

#ifndef P10DC_RANKDELTA8_ALIGN32
#define P10DC_RANKDELTA8_ALIGN32 1
#endif
static_assert(P10DC_RANKDELTA8_ALIGN32 == 0 || P10DC_RANKDELTA8_ALIGN32 == 1,
              "P10DC_RANKDELTA8_ALIGN32 must be 0 or 1");

static constexpr uint32_t P10DC_RANKDELTA8_BLOCK_LOG2 = 5u;
static constexpr uint32_t P10DC_RANKDELTA8_BLOCK = 1u << P10DC_RANKDELTA8_BLOCK_LOG2;
static constexpr uint32_t P10DC_RANKDELTA8_CHUNK_BITS = 23u;
static constexpr uint32_t P10DC_RANKDELTA8_PREFIX_BITS = 9u;
static constexpr uint32_t P10DC_RANKDELTA8_CHUNK_MASK = (1u << P10DC_RANKDELTA8_CHUNK_BITS) - 1u;
static constexpr uint32_t P10DC_RANKDELTA8_PREFIX_LIMIT = 1u << P10DC_RANKDELTA8_PREFIX_BITS;
static constexpr uint32_t P10DC_RANKDELTA8_DELTA_LIMIT = 1u << 14;
static constexpr uint32_t P10DC_RANKDELTA8_MAX_L = uint32_t(LOW_LUT_K / 2);
static constexpr uint32_t P10DC_RANKDELTA8_MAX_ROW_BYTES =
    P10DC_RANKDELTA8_MAX_L ? 2u + 2u * (P10DC_RANKDELTA8_MAX_L - 1u) : 0u;
static_assert(P10DC_RANKDELTA8_CHUNK_BITS + P10DC_RANKDELTA8_PREFIX_BITS == 32u);
static_assert(LOW_LUT_K <= 14, "rankdelta8 assumes LOW_LUT_K<=14");
static_assert((P10DC_RANKDELTA8_BLOCK - 1u) * P10DC_RANKDELTA8_MAX_ROW_BYTES <
              P10DC_RANKDELTA8_PREFIX_LIMIT,
              "rankdelta8 worst-case byte prefix no longer fits 9 bits");

__constant__ uint32_t* D_P10DC_LOW_RANKDELTA8_META32;
__constant__ uint32_t* D_P10DC_LOW_RANKDELTA8_BLOCK32;
__constant__ uint8_t* D_P10DC_LOW_RANKDELTA8_STREAM;
__constant__ uint32_t D_P10DC_LOW_RANKDELTA8_HOFF[MAXW + 2];

__device__ __forceinline__ void p10dc_low_rankdelta8_row_warpstripe(
    uint32_t h, uint32_t rank, uint32_t& packed_chunks, const uint8_t*& row
) {
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const unsigned active = __activemask();
    const uint32_t compact = D_P10DC_LOW_RANKDELTA8_HOFF[h] + rank;
    const uint32_t meta = D_P10DC_LOW_RANKDELTA8_META32[compact];
    packed_chunks = meta & P10DC_RANKDELTA8_CHUNK_MASK;
    const uint32_t prefix = meta >> P10DC_RANKDELTA8_CHUNK_BITS;
    const uint32_t first_compact = compact - lane;
    const uint32_t first_block = first_compact >> P10DC_RANKDELTA8_BLOCK_LOG2;
#if P10DC_RANKDELTA8_ALIGN32
    uint32_t local_base = 0;
    if (lane == 0u) local_base = D_P10DC_LOW_RANKDELTA8_BLOCK32[first_block];
    const uint32_t block_base = __shfl_sync(active, local_base, 0);
#else
    const uint32_t first_off = first_compact & 31u;
    const uint32_t split_lane = 32u - first_off;
    uint32_t local_base = 0;
    if (lane == 0u) local_base = D_P10DC_LOW_RANKDELTA8_BLOCK32[first_block];
    if (split_lane < 32u && lane == split_lane)
        local_base = D_P10DC_LOW_RANKDELTA8_BLOCK32[first_block + 1u];
    const uint32_t source_lane =
        (split_lane < 32u && lane >= split_lane) ? split_lane : 0u;
    const uint32_t block_base = __shfl_sync(active, local_base, int(source_lane));
#endif
    row = D_P10DC_LOW_RANKDELTA8_STREAM + block_base + prefix;
}

struct BucketFusedDirectHighRowsRankDelta8Tables
    : BucketFusedDirectHighRowsPrekeyTables {
    uint32_t* low_rankdelta8_meta32 = nullptr;
    uint32_t* low_rankdelta8_block32 = nullptr;
    uint8_t* low_rankdelta8_stream = nullptr;
    size_t low_rankdelta8_meta32_count = 0, low_rankdelta8_block32_count = 0;
    size_t low_rankdelta8_stream_count = 0, low_rankdelta8_padding_count = 0;
    size_t low_rankdelta8_meta32_capacity = 0, low_rankdelta8_block32_capacity = 0;
    size_t low_rankdelta8_stream_capacity = 0;

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsPrekeyTables::bind_owner(fixed, buckets, slot);
        if (!host_fused) { std::cerr << "p10dc rankdelta8 missing host fused metadata\n"; std::exit(690); }
        const BucketFusedHost& f = *host_fused;
        constexpr size_t P = size_t(MAXW + 2);
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());

        std::array<uint32_t, MAXW + 2> hoff{};
        std::vector<uint32_t> meta, blocks;
        std::vector<uint8_t> stream;
        constexpr size_t PAD_BOUND = size_t(MAXW + 2) * 31u;
        meta.reserve(low_prekey_count + (P10DC_RANKDELTA8_ALIGN32 ? PAD_BOUND : 0u));
        blocks.reserve((meta.capacity() + 31u) >> 5);
        stream.reserve(low_prekey_count * 5u);
        uint32_t block_base = 0;
        size_t actual_codes = 0, padding = 0, slow_delta = 0, delta_count = 0;
        uint32_t max_delta = 0, max_prefix = 0;

        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
#if P10DC_RANKDELTA8_ALIGN32
            while (meta.size() & 31u) { meta.push_back(0u); ++padding; }
#endif
            hoff[h] = uint32_t(meta.size());
            const uint32_t a = f.low_code_off[owner_base + h];
            const uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u] : owner_end;
            for (uint32_t i = a; i < b; ++i) {
                const uint32_t compact = uint32_t(meta.size());
                if ((compact & 31u) == 0u) {
                    block_base = uint32_t(stream.size());
                    blocks.push_back(block_base);
                }
                const uint32_t prefix = uint32_t(stream.size()) - block_base;
                max_prefix = std::max(max_prefix, prefix);
                const uint32_t code = f.low_codes[i];
                const uint32_t key = gpu_direct_ternary_key_host(code, LOW_LUT_K);
                const uint32_t chunks = p10dc_rankchunk32_pack_host(key);
                if ((chunks >> P10DC_RANKDELTA8_CHUNK_BITS) != 0u ||
                    prefix >= P10DC_RANKDELTA8_PREFIX_LIMIT) {
                    std::cerr << "p10dc rankdelta8 metadata overflow owner=" << fixed
                              << " h=" << h << " compact=" << compact
                              << " chunks=" << chunks << " prefix=" << prefix << '\n';
                    std::exit(691);
                }
                meta.push_back(chunks | (prefix << P10DC_RANKDELTA8_CHUNK_BITS));
                ++actual_codes;

                std::array<uint16_t, LOW_LUT_K> ranks{};
                uint32_t nr = 0;
                uint32_t weight = bkcz_pow3_const(LOW_LUT_K - 1);
                for (int pos = LOW_LUT_K - 1; pos >= 0; --pos) {
                    if (((code >> (2 * pos)) & 3u) == uint32_t(::L)) {
                        const uint32_t cand_key = key - weight;
                        if (cand_key >= f.low_direct.size()) std::exit(692);
                        const uint32_t x = f.low_direct[cand_key];
                        if (x == BKF_DIRECT_INVALID) std::exit(693);
                        const uint32_t loc = x & BKF_LOC_MASK;
                        if (bkf_loc_owner(loc) != fixed) std::exit(694);
                        const uint32_t source_rank = bkf_loc_rank(loc);
                        if (source_rank >= (1u << 15)) std::exit(695);
                        ranks[nr++] = uint16_t(source_rank);
                    }
                    if (pos) weight /= 3u;
                }
                if (nr > P10DC_RANKDELTA8_MAX_L) std::exit(696);
                if (nr) {
                    stream.push_back(uint8_t(ranks[0]));
                    stream.push_back(uint8_t(ranks[0] >> 8));
                    for (uint32_t j = 1; j < nr; ++j) {
                        if (ranks[j] <= ranks[j - 1]) std::exit(697);
                        const uint32_t d = uint32_t(ranks[j] - ranks[j - 1]);
                        if (d >= P10DC_RANKDELTA8_DELTA_LIMIT) std::exit(698);
                        max_delta = std::max(max_delta, d);
                        ++delta_count;
                        if (d < 128u) {
                            stream.push_back(uint8_t(d));
                        } else {
                            stream.push_back(uint8_t(0x80u | (d & 0x7fu)));
                            stream.push_back(uint8_t(d >> 7));
                            ++slow_delta;
                        }
                    }
                }
            }
        }
        if (actual_codes != low_prekey_count || meta.size() != actual_codes + padding) {
            std::cerr << "p10dc rankdelta8 size mismatch owner=" << fixed
                      << " actual=" << actual_codes << '/' << low_prekey_count
                      << " meta=" << meta.size() << " padding=" << padding << '\n';
            std::exit(699);
        }

        low_rankdelta8_meta32_count = meta.size();
        low_rankdelta8_block32_count = blocks.size();
        low_rankdelta8_stream_count = stream.size();
        low_rankdelta8_padding_count = padding;
        if (low_rankdelta8_meta32_count > low_rankdelta8_meta32_capacity) {
            if (low_rankdelta8_meta32) cudaFree(low_rankdelta8_meta32);
            low_rankdelta8_meta32 = nullptr; low_rankdelta8_meta32_capacity = low_rankdelta8_meta32_count;
            if (low_rankdelta8_meta32_capacity) ck(cudaMalloc(&low_rankdelta8_meta32,
                low_rankdelta8_meta32_capacity * sizeof(uint32_t)), "p10dc rankdelta8 meta alloc");
        }
        if (low_rankdelta8_block32_count > low_rankdelta8_block32_capacity) {
            if (low_rankdelta8_block32) cudaFree(low_rankdelta8_block32);
            low_rankdelta8_block32 = nullptr; low_rankdelta8_block32_capacity = low_rankdelta8_block32_count;
            if (low_rankdelta8_block32_capacity) ck(cudaMalloc(&low_rankdelta8_block32,
                low_rankdelta8_block32_capacity * sizeof(uint32_t)), "p10dc rankdelta8 block alloc");
        }
        if (low_rankdelta8_stream_count > low_rankdelta8_stream_capacity) {
            if (low_rankdelta8_stream) cudaFree(low_rankdelta8_stream);
            low_rankdelta8_stream = nullptr; low_rankdelta8_stream_capacity = low_rankdelta8_stream_count;
            if (low_rankdelta8_stream_capacity) ck(cudaMalloc(&low_rankdelta8_stream,
                low_rankdelta8_stream_capacity * sizeof(uint8_t)), "p10dc rankdelta8 stream alloc");
        }
        if (!meta.empty()) ck(cudaMemcpy(low_rankdelta8_meta32, meta.data(), meta.size()*sizeof(uint32_t), cudaMemcpyHostToDevice), "p10dc rankdelta8 meta H2D");
        if (!blocks.empty()) ck(cudaMemcpy(low_rankdelta8_block32, blocks.data(), blocks.size()*sizeof(uint32_t), cudaMemcpyHostToDevice), "p10dc rankdelta8 block H2D");
        if (!stream.empty()) ck(cudaMemcpy(low_rankdelta8_stream, stream.data(), stream.size(), cudaMemcpyHostToDevice), "p10dc rankdelta8 stream H2D");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKDELTA8_META32, &low_rankdelta8_meta32, sizeof(low_rankdelta8_meta32)), "p10dc rankdelta8 meta ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKDELTA8_BLOCK32, &low_rankdelta8_block32, sizeof(low_rankdelta8_block32)), "p10dc rankdelta8 block ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKDELTA8_STREAM, &low_rankdelta8_stream, sizeof(low_rankdelta8_stream)), "p10dc rankdelta8 stream ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKDELTA8_HOFF, hoff.data(), hoff.size()*sizeof(uint32_t)), "p10dc rankdelta8 hoff");

        if (low_prekey) cudaFree(low_prekey);
        low_prekey = nullptr; low_prekey_capacity = 0;
        uint32_t* null32 = nullptr;
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_PREKEY, &null32, sizeof(null32)), "p10dc rankdelta8 null old prekey ptr");

        const size_t bytes = meta.size()*sizeof(uint32_t) + blocks.size()*sizeof(uint32_t) + stream.size();
        std::cerr << "p10dc_low_rankdelta8 fixed_owner=" << fixed
                  << " codes=" << actual_codes << " meta_entries=" << meta.size()
                  << " blocks=" << blocks.size() << " stream_bytes=" << stream.size()
                  << " total_bytes=" << bytes << " padding=" << padding
                  << " max_prefix=" << max_prefix << " max_delta=" << max_delta
                  << " slow_delta=" << slow_delta << '/' << delta_count
                  << " chunk_bits=23 prefix_bits=9 block=32"
                  << " height_align=" << (P10DC_RANKDELTA8_ALIGN32 ? 32 : 1)
                  << " block_base_loads_per_warp_max=" << (P10DC_RANKDELTA8_ALIGN32 ? 1 : 2)
                  << " old_prekey_freed=1\n";
    }

    void release() {
        if (low_rankdelta8_meta32) cudaFree(low_rankdelta8_meta32);
        if (low_rankdelta8_block32) cudaFree(low_rankdelta8_block32);
        if (low_rankdelta8_stream) cudaFree(low_rankdelta8_stream);
        low_rankdelta8_meta32 = low_rankdelta8_block32 = nullptr;
        low_rankdelta8_stream = nullptr;
        low_rankdelta8_meta32_count = low_rankdelta8_block32_count = 0;
        low_rankdelta8_stream_count = low_rankdelta8_padding_count = 0;
        low_rankdelta8_meta32_capacity = low_rankdelta8_block32_capacity = 0;
        low_rankdelta8_stream_capacity = 0;
        BucketFusedDirectHighRowsPrekeyTables::release();
    }
};
