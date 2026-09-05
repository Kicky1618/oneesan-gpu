#pragma once

#include "ramstream32_bucket_low_prekey.cuh"

static constexpr uint16_t P10DC_LOW_RANK16_INVALID = 0xffffu;
__constant__ uint16_t* D_P10DC_LOW_RANK16;

__device__ __forceinline__ const uint16_t* p10dc_low_rank16_row(uint32_t h, uint32_t rank) {
    size_t compact = size_t(D_P10DC_LOW_PREKEY_HOFF[h]) + rank;
    return D_P10DC_LOW_RANK16 + compact * LOW_LUT_K;
}

// Optional CROSS acceleration layered on top of fixed-owner prekeys.  For each
// bound-owner LOW factor code and each symbol position, store the owner-local
// rank of the code obtained by L->R (the only CROSS candidate flip), or 0xffff
// when that flip is not a legal direct-table state.  Occupancy is unchanged by
// the flip, so every valid candidate must remain on the bound owner.
struct BucketFusedDirectHighRowsPrekeyRank16Tables
    : BucketFusedDirectHighRowsPrekeyTables {
    uint16_t* low_rank16 = nullptr;
    size_t low_rank16_count = 0;
    size_t low_rank16_capacity = 0;

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
            std::cerr << "p10dc LOW rank16 missing host fused metadata\n";
            std::exit(629);
        }
        const BucketFusedHost& f = *host_fused;
        constexpr size_t P = size_t(MAXW + 2);
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? f.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(f.low_codes.size());

        std::vector<uint16_t> rank16;
        rank16.reserve(low_prekey_count * LOW_LUT_K);
        uint64_t valid = 0;
        for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
            uint32_t a = f.low_code_off[owner_base + h];
            uint32_t b = h + 1u < uint32_t(MAXW + 2)
                ? f.low_code_off[owner_base + h + 1u]
                : owner_end;
            for (uint32_t i = a; i < b; ++i) {
                uint32_t code = f.low_codes[i];
                uint32_t key = gpu_direct_ternary_key_host(code, LOW_LUT_K);
                uint32_t weight = 1u;
                for (int pos = 0; pos < LOW_LUT_K; ++pos) {
                    uint16_t out = P10DC_LOW_RANK16_INVALID;
                    uint32_t v = (code >> (2 * pos)) & 3u;
                    if (v == uint32_t(::L)) {
                        uint32_t cand_key = key - weight;
                        if (cand_key >= f.low_direct.size()) {
                            std::cerr << "p10dc LOW rank16 direct-key overflow key=" << cand_key
                                      << " size=" << f.low_direct.size() << '\n';
                            std::exit(630);
                        }
                        uint32_t x = f.low_direct[cand_key];
                        if (x != BKF_DIRECT_INVALID) {
                            uint32_t loc = x & BKF_LOC_MASK;
                            uint32_t owner = bkf_loc_owner(loc);
                            uint32_t rank = bkf_loc_rank(loc);
                            if (owner != fixed) {
                                std::cerr << "p10dc LOW rank16 occupancy-owner invariant violated fixed="
                                          << fixed << " got=" << owner << " h=" << h
                                          << " pos=" << pos << '\n';
                                std::exit(631);
                            }
                            if (rank >= uint32_t(P10DC_LOW_RANK16_INVALID)) {
                                std::cerr << "p10dc LOW rank16 rank overflow rank=" << rank << '\n';
                                std::exit(632);
                            }
                            out = uint16_t(rank);
                            ++valid;
                        }
                    }
                    rank16.push_back(out);
                    weight *= 3u;
                }
            }
        }
        if (rank16.size() != low_prekey_count * size_t(LOW_LUT_K)) {
            std::cerr << "p10dc LOW rank16 size mismatch got=" << rank16.size()
                      << " expected=" << low_prekey_count * size_t(LOW_LUT_K) << '\n';
            std::exit(633);
        }

        low_rank16_count = rank16.size();
        if (low_rank16_count > low_rank16_capacity) {
            if (low_rank16) cudaFree(low_rank16);
            low_rank16 = nullptr;
            low_rank16_capacity = low_rank16_count;
            if (low_rank16_capacity)
                ck(cudaMalloc(&low_rank16, low_rank16_capacity * sizeof(uint16_t)),
                   "p10dc LOW rank16 alloc");
        }
        if (!rank16.empty())
            ck(cudaMemcpy(low_rank16, rank16.data(), rank16.size() * sizeof(uint16_t),
                          cudaMemcpyHostToDevice),
               "p10dc LOW rank16 H2D");
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANK16, &low_rank16, sizeof(low_rank16)),
           "p10dc LOW rank16 ptr");

        std::cerr << "p10dc_low_rank16 fixed_owner=" << fixed
                  << " entries=" << low_rank16_count
                  << " valid=" << valid
                  << " mib=" << double(low_rank16_count * sizeof(uint16_t)) / double(1 << 20)
                  << " bytes_per_low_code=" << (LOW_LUT_K * sizeof(uint16_t))
                  << " direct_lookup_runtime=0\n";
    }

    void release() {
        if (low_rank16) cudaFree(low_rank16);
        low_rank16 = nullptr;
        low_rank16_count = 0;
        low_rank16_capacity = 0;
        BucketFusedDirectHighRowsPrekeyTables::release();
    }
};
