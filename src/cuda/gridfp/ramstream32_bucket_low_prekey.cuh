#pragma once

#include "ramstream32_bucket_direct_high_row_affine.cuh"

// Experimental per-owner/per-height LOW ternary keys.  The indexing is exactly
// the same as D_BKF_LOW_CODES, so the HIGH closure hot path can load the packed
// factor code for CROSS5 symbol tests and its ternary direct-table key without
// recomputing the base-3 fold in every lane.
__constant__ uint32_t* D_P10DC_LOW_PREKEY;

__device__ __forceinline__ uint32_t p10dc_low_prekey(uint32_t owner, uint32_t h, uint32_t rank) {
    return D_P10DC_LOW_PREKEY[
        D_BKF_LOW_CODE_OFF[size_t(owner) * D_BKF_CODE_PITCH + h] + rank];
}

struct BucketFusedDirectHighRowsPrekeyTables {
    BucketFusedDirectHighRowsTables base;
    uint32_t* low_prekey = nullptr;
    size_t low_prekey_count = 0;

    void install_metadata(
        const StorageLayout& layout, const BucketOrbitStreamsHost& o, const BucketFusedHost& f
    ) {
        base.install_metadata(layout, o, f);
        low_prekey_count = f.low_codes.size();
        std::vector<uint32_t> key(low_prekey_count);
        uint32_t max_key = 0;
        for (size_t i = 0; i < f.low_codes.size(); ++i) {
            key[i] = gpu_direct_ternary_key_host(f.low_codes[i], LOW_LUT_K);
            max_key = std::max(max_key, key[i]);
        }
        if (!key.empty()) {
            ck(cudaMalloc(&low_prekey, key.size() * sizeof(uint32_t)),
               "p10dc LOW prekey alloc");
            ck(cudaMemcpy(low_prekey, key.data(), key.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
               "p10dc LOW prekey H2D");
        }
        ck(cudaMemcpyToSymbol(D_P10DC_LOW_PREKEY, &low_prekey, sizeof(low_prekey)),
           "p10dc LOW prekey ptr");
        std::cerr << "p10dc_low_prekey entries=" << low_prekey_count
                  << " mib=" << double(low_prekey_count * sizeof(uint32_t)) / double(1 << 20)
                  << " max_key=" << max_key
                  << " fold_runtime=0\n";
    }

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        base.bind_owner(fixed, buckets, slot);
    }

    void release() {
        if (low_prekey) cudaFree(low_prekey);
        low_prekey = nullptr;
        low_prekey_count = 0;
        base.release();
    }
};
