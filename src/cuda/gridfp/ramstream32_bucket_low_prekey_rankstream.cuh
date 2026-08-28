#pragma once

#include "ramstream32_bucket_low_prekey.cuh"

__constant__ uint32_t* D_P10DC_LOW_RANKSTREAM_OFF;
__constant__ uint16_t* D_P10DC_LOW_RANKSTREAM;

__device__ __forceinline__ const uint16_t* p10dc_low_rankstream_row(uint32_t h, uint32_t rank) {
    size_t compact = size_t(D_P10DC_LOW_PREKEY_HOFF[h]) + rank;
    return D_P10DC_LOW_RANKSTREAM + D_P10DC_LOW_RANKSTREAM_OFF[compact];
}

// Sparse alternative to dense rank16.  The W28 host proof establishes that
// every LOW L->R flip is a legal factor and remains on the same occupancy owner.
// Store only those ranks, in descending symbol-position order.  One uint32
// offset per compact LOW code locates its variable-length rank stream.
struct BucketFusedDirectHighRowsPrekeyRankStreamTables
    : BucketFusedDirectHighRowsPrekeyTables {
    uint32_t* low_rankstream_off = nullptr;
    uint16_t* low_rankstream = nullptr;
    size_t low_rankstream_off_count = 0;
    size_t low_rankstream_count = 0;
    size_t low_rankstream_off_capacity = 0;
    size_t low_rankstream_capacity = 0;

    void install_metadata(
        const StorageLayout& layout, const BucketOrbitStreamsHost& o, const BucketFusedHost& f
    ) {
        BucketFusedDirectHighRowsPrekeyTables::install_metadata(layout, o, f);
    }

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        BucketFusedDirectHighRowsPrekeyTables::bind_owner(fixed, buckets, slot);
        if (!host_fused) {
            std::cerr << "p10dc LOW rankstream missing host fused metadata\n";
            std::exit(642);
        }
        const BucketFusedHost& f = *host_fused;
        constexpr size_t P = size_t(MAXW + 2);
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());

        std::vector<uint32_t> off;
        std::vector<uint16_t> stream;
        off.reserve(low_prekey_count);
        stream.reserve(low_prekey_count * 4u);
        uint64_t l_digits = 0;
        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
            uint32_t a = f.low_code_off[owner_base + h];
            uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u]
                : owner_end;
            for (uint32_t i = a; i < b; ++i) {
                off.push_back(uint32_t(stream.size()));
                uint32_t code = f.low_codes[i];
                uint32_t key = gpu_direct_ternary_key_host(code, LOW_LUT_K);
                uint32_t weight = bkcz_pow3_const(LOW_LUT_K - 1);
                for (int pos = LOW_LUT_K - 1; pos >= 0; --pos) {
                    if (((code >> (2 * pos)) & 3u) == uint32_t(::L)) {
                        uint32_t cand_key = key - weight;
                        if (cand_key >= f.low_direct.size()) {
                            std::cerr << "p10dc LOW rankstream direct-key overflow key=" << cand_key
                                      << " size=" << f.low_direct.size() << '\n';
                            std::exit(643);
                        }
                        uint32_t x = f.low_direct[cand_key];
                        if (x == BKF_DIRECT_INVALID) {
                            std::cerr << "p10dc LOW rankstream L->R legality invariant failed owner="
                                      << fixed << " h=" << h << " pos=" << pos << '\n';
                            std::exit(644);
                        }
                        uint32_t loc = x & BKF_LOC_MASK;
                        uint32_t owner = bkf_loc_owner(loc);
                        uint32_t rank = bkf_loc_rank(loc);
                        if (owner != fixed) {
                            std::cerr << "p10dc LOW rankstream occupancy-owner invariant violated fixed="
                                      << fixed << " got=" << owner << " h=" << h
                                      << " pos=" << pos << '\n';
                            std::exit(645);
                        }
                        if (rank >= 0xffffu) {
                            std::cerr << "p10dc LOW rankstream rank overflow rank=" << rank << '\n';
                            std::exit(646);
                        }
                        stream.push_back(uint16_t(rank));
                        ++l_digits;
                    }
                    if (pos) weight /= 3u;
                }
            }
        }
        if (off.size() != low_prekey_count || stream.size() != l_digits) {
            std::cerr << "p10dc LOW rankstream size mismatch offsets=" << off.size()
                      << '/' << low_prekey_count << " stream=" << stream.size()
                      << '/' << l_digits << '\n';
            std::exit(647);
        }

        low_rankstream_off_count = off.size();
        low_rankstream_count = stream.size();
        if (low_rankstream_off_count > low_rankstream_off_capacity) {
            if (low_rankstream_off) cudaFree(low_rankstream_off);
            low_rankstream_off = nullptr;
            low_rankstream_off_capacity = low_rankstream_off_count;
            if (low_rankstream_off_capacity)
                ck(cudaMalloc(&low_rankstream_off,
                              low_rankstream_off_capacity * sizeof(uint32_t)),
                   "p10dc LOW rankstream offset alloc");
        }
        if (low_rankstream_count > low_rankstream_capacity) {
            if (low_rankstream) cudaFree(low_rankstream);
            low_rankstream = nullptr;
            low_rankstream_capacity = low_rankstream_count;
            if (low_rankstream_capacity)
                ck(cudaMalloc(&low_rankstream,
                              low_rankstream_capacity * sizeof(uint16_t)),
                   "p10dc LOW rankstream alloc");
        }
        if (!off.empty())
            ck(cudaMemcpy(low_rankstream_off, off.data(), off.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice),
               "p10dc LOW rankstream offsets H2D");
        if (!stream.empty())
            ck(cudaMemcpy(low_rankstream, stream.data(), stream.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice),
               "p10dc LOW rankstream H2D");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKSTREAM_OFF, &low_rankstream_off,
                              sizeof(low_rankstream_off)),
           "p10dc LOW rankstream offset ptr");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKSTREAM, &low_rankstream,
                              sizeof(low_rankstream)),
           "p10dc LOW rankstream ptr");

        size_t bytes = off.size() * sizeof(uint32_t) + stream.size() * sizeof(uint16_t);
        std::cerr << "p10dc_low_rankstream fixed_owner=" << fixed
                  << " codes=" << low_prekey_count
                  << " l_ranks=" << low_rankstream_count
                  << " l_per_code=" << (low_prekey_count
                        ? double(low_rankstream_count) / double(low_prekey_count) : 0.0)
                  << " mib=" << double(bytes) / double(1 << 20)
                  << " dense_rank16_mib="
                  << double(low_prekey_count * size_t(LOW_LUT_K) * sizeof(uint16_t))
                     / double(1 << 20)
                  << " all_L_flips_legal=1 direct_lookup_runtime=0\n";
    }

    void release() {
        if (low_rankstream_off) cudaFree(low_rankstream_off);
        if (low_rankstream) cudaFree(low_rankstream);
        low_rankstream_off = nullptr;
        low_rankstream = nullptr;
        low_rankstream_off_count = low_rankstream_count = 0;
        low_rankstream_off_capacity = low_rankstream_capacity = 0;
        BucketFusedDirectHighRowsPrekeyTables::release();
    }
};
