#pragma once

#include "ramstream32_bucket_low_prekey_rankstream.cuh"

static constexpr uint32_t P10DC_RANKSTREAM32_BLOCK_LOG2 = 5u;
static constexpr uint32_t P10DC_RANKSTREAM32_BLOCK = 1u << P10DC_RANKSTREAM32_BLOCK_LOG2;
static constexpr uint32_t P10DC_RANKSTREAM32_KEY_BITS = 23u;
static constexpr uint32_t P10DC_RANKSTREAM32_PREFIX_BITS = 9u;
static constexpr uint32_t P10DC_RANKSTREAM32_KEY_MASK = (1u << P10DC_RANKSTREAM32_KEY_BITS) - 1u;
static_assert(P10DC_RANKSTREAM32_KEY_BITS + P10DC_RANKSTREAM32_PREFIX_BITS == 32u);
static_assert(bkcz_pow3_const(LOW_LUT_K) <= (1u << P10DC_RANKSTREAM32_KEY_BITS),
              "LOW ternary key no longer fits rankstream32 key field");
static_assert((P10DC_RANKSTREAM32_BLOCK - 1u) * uint32_t(LOW_LUT_K) <
              (1u << P10DC_RANKSTREAM32_PREFIX_BITS),
              "rankstream32 worst-case within-block prefix no longer fits 9 bits");

__constant__ uint32_t* D_P10DC_LOW_RANKMETA32;
__constant__ uint32_t* D_P10DC_LOW_RANKBLOCK32;

__device__ __forceinline__ void p10dc_low_rankstream32_row(
    uint32_t h, uint32_t rank, uint32_t& key, const uint16_t*& row
) {
    uint32_t compact = D_P10DC_LOW_PREKEY_HOFF[h] + rank;
    uint32_t meta = D_P10DC_LOW_RANKMETA32[compact];
    key = meta & P10DC_RANKSTREAM32_KEY_MASK;
    uint32_t prefix = meta >> P10DC_RANKSTREAM32_KEY_BITS;
    uint32_t block_base = D_P10DC_LOW_RANKBLOCK32[compact >> P10DC_RANKSTREAM32_BLOCK_LOG2];
    row = D_P10DC_LOW_RANKSTREAM + block_base + prefix;
}

// Warp-striped HIGH kernels visit LOW ranks as
//   blockIdx.x*32 + lane + q*gridDim.x*32.
// Hence rank%32==lane and compact=height_offset+rank is a contiguous 32-entry
// interval across the active warp.  An arbitrary height_offset can make that
// interval straddle one 32-code metadata block boundary, but never more than
// one.  Load the first block base in lane 0 and, only when required by active
// lanes, the second in the lane where that block begins.  This reduces block-
// base traffic from one global load per lane to at most two per warp stripe.
__device__ __forceinline__ void p10dc_low_rankstream32_row_warpstripe(
    uint32_t h, uint32_t rank, uint32_t& key, const uint16_t*& row
) {
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    const unsigned active = __activemask();
    const uint32_t compact = D_P10DC_LOW_PREKEY_HOFF[h] + rank;
    const uint32_t meta = D_P10DC_LOW_RANKMETA32[compact];
    key = meta & P10DC_RANKSTREAM32_KEY_MASK;
    const uint32_t prefix = meta >> P10DC_RANKSTREAM32_KEY_BITS;

    // rank%32==lane in the only caller, so this is the compact index belonging
    // to lane 0 of the current stripe.  It cannot underflow because rank>=lane.
    const uint32_t first_compact = compact - lane;
    const uint32_t first_block = first_compact >> P10DC_RANKSTREAM32_BLOCK_LOG2;
    const uint32_t first_off = first_compact & (P10DC_RANKSTREAM32_BLOCK - 1u);
    const uint32_t split_lane = P10DC_RANKSTREAM32_BLOCK - first_off; // [1,32]

    uint32_t b0_local = 0;
    if (lane == 0u) b0_local = D_P10DC_LOW_RANKBLOCK32[first_block];
    const uint32_t b0 = __shfl_sync(active, b0_local, 0);
    uint32_t block_base = b0;

    if (split_lane < 32u) {
        const unsigned split_bit = 1u << split_lane;
        // Active lanes are a low contiguous prefix on the final partial stripe.
        // If split_lane is inactive, no active lane belongs to the second block.
        if (active & split_bit) {
            uint32_t b1_local = 0;
            if (lane == split_lane)
                b1_local = D_P10DC_LOW_RANKBLOCK32[first_block + 1u];
            const uint32_t b1 = __shfl_sync(active, b1_local, int(split_lane));
            if (lane >= split_lane) block_base = b1;
        }
    }
    row = D_P10DC_LOW_RANKSTREAM + block_base + prefix;
}

// Sparse rankstream with warp-sized offset compression.  The parent builder is
// used as a correctness reference and to construct the uint16 rank stream; once
// packed metadata is installed, the separate prekey and uint32-per-code offset
// arrays are freed and their device symbols are nulled so accidental hot-path
// use is caught immediately.
struct BucketFusedDirectHighRowsRankStream32Tables
    : BucketFusedDirectHighRowsPrekeyRankStreamTables {
    uint32_t* low_rankmeta32 = nullptr;
    uint32_t* low_rankblock32 = nullptr;
    size_t low_rankmeta32_count = 0, low_rankblock32_count = 0;
    size_t low_rankmeta32_capacity = 0, low_rankblock32_capacity = 0;

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsPrekeyRankStreamTables::bind_owner(fixed, buckets, slot);
        if (!host_fused) { std::cerr << "p10dc rankstream32 missing host fused metadata\n"; std::exit(658); }
        const BucketFusedHost& f = *host_fused;
        constexpr size_t P = size_t(MAXW + 2);
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());

        std::vector<uint32_t> meta;
        std::vector<uint32_t> blocks;
        meta.reserve(low_prekey_count);
        blocks.reserve((low_prekey_count + P10DC_RANKSTREAM32_BLOCK - 1u) /
                       P10DC_RANKSTREAM32_BLOCK);
        uint32_t stream_cursor = 0, block_base = 0;
        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
            uint32_t a = f.low_code_off[owner_base + h];
            uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u]
                : owner_end;
            for (uint32_t i = a; i < b; ++i) {
                uint32_t compact = uint32_t(meta.size());
                if ((compact & (P10DC_RANKSTREAM32_BLOCK - 1u)) == 0u) {
                    block_base = stream_cursor;
                    blocks.push_back(block_base);
                }
                uint32_t prefix = stream_cursor - block_base;
                uint32_t code = f.low_codes[i];
                uint32_t key = gpu_direct_ternary_key_host(code, LOW_LUT_K);
                if (key > P10DC_RANKSTREAM32_KEY_MASK ||
                    prefix >= (1u << P10DC_RANKSTREAM32_PREFIX_BITS)) {
                    std::cerr << "p10dc rankstream32 packing overflow key=" << key
                              << " prefix=" << prefix << '\n';
                    std::exit(659);
                }
                meta.push_back(key | (prefix << P10DC_RANKSTREAM32_KEY_BITS));
                for (int pos = 0; pos < LOW_LUT_K; ++pos)
                    if (((code >> (2 * pos)) & 3u) == uint32_t(::L)) ++stream_cursor;
            }
        }
        if (meta.size() != low_prekey_count || stream_cursor != low_rankstream_count) {
            std::cerr << "p10dc rankstream32 size mismatch meta=" << meta.size()
                      << '/' << low_prekey_count << " stream=" << stream_cursor
                      << '/' << low_rankstream_count << '\n';
            std::exit(660);
        }

        low_rankmeta32_count = meta.size();
        low_rankblock32_count = blocks.size();
        if (low_rankmeta32_count > low_rankmeta32_capacity) {
            if (low_rankmeta32) cudaFree(low_rankmeta32);
            low_rankmeta32 = nullptr; low_rankmeta32_capacity = low_rankmeta32_count;
            if (low_rankmeta32_capacity)
                ck(cudaMalloc(&low_rankmeta32, low_rankmeta32_capacity * sizeof(uint32_t)),
                   "p10dc rankstream32 meta alloc");
        }
        if (low_rankblock32_count > low_rankblock32_capacity) {
            if (low_rankblock32) cudaFree(low_rankblock32);
            low_rankblock32 = nullptr; low_rankblock32_capacity = low_rankblock32_count;
            if (low_rankblock32_capacity)
                ck(cudaMalloc(&low_rankblock32, low_rankblock32_capacity * sizeof(uint32_t)),
                   "p10dc rankstream32 block alloc");
        }
        if (!meta.empty()) ck(cudaMemcpy(low_rankmeta32, meta.data(), meta.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "p10dc rankstream32 meta H2D");
        if (!blocks.empty()) ck(cudaMemcpy(low_rankblock32, blocks.data(), blocks.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "p10dc rankstream32 blocks H2D");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKMETA32, &low_rankmeta32, sizeof(low_rankmeta32)), "p10dc rankstream32 meta ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKBLOCK32, &low_rankblock32, sizeof(low_rankblock32)), "p10dc rankstream32 block ptr");

        if (low_prekey) cudaFree(low_prekey);
        low_prekey = nullptr; low_prekey_capacity = 0;
        if (low_rankstream_off) cudaFree(low_rankstream_off);
        low_rankstream_off = nullptr; low_rankstream_off_capacity = 0;
        uint32_t* null32 = nullptr;
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_PREKEY, &null32, sizeof(null32)), "p10dc rankstream32 null old prekey ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKSTREAM_OFF, &null32, sizeof(null32)), "p10dc rankstream32 null old offset ptr");

        size_t bytes = meta.size() * sizeof(uint32_t) + blocks.size() * sizeof(uint32_t) +
                       low_rankstream_count * sizeof(uint16_t);
        std::cerr << "p10dc_low_rankstream32 fixed_owner=" << fixed
                  << " codes=" << meta.size() << " blocks=" << blocks.size()
                  << " l_ranks=" << low_rankstream_count
                  << " bytes=" << bytes
                  << " key_bits=23 prefix_bits=9 block=32"
                  << " block_base_loads_per_warp_max=2"
                  << " old_prekey_offset_arrays_freed=1 direct_lookup_runtime=0\n";
    }

    void release() {
        if (low_rankmeta32) cudaFree(low_rankmeta32);
        if (low_rankblock32) cudaFree(low_rankblock32);
        low_rankmeta32 = low_rankblock32 = nullptr;
        low_rankmeta32_count = low_rankblock32_count = 0;
        low_rankmeta32_capacity = low_rankblock32_capacity = 0;
        BucketFusedDirectHighRowsPrekeyRankStreamTables::release();
    }
};
