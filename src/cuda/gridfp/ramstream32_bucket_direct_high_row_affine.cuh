#pragma once

#include "ramstream32_bucket_onepass_zero_alias.cuh"

// HIGH-window closure source rows have the affine form
//   slot[owner] + block.off + rank * block.cols.
// The fixed LOW owner is known at bind time, so pre-bind the first two terms
// and the stride once per (HIGH owner, main block). Runtime closure resolution
// then avoids loading a full BucketPhysicalBlock for every source.
struct P10DCHighRowAffine {
    Count* base = nullptr;
    uint32_t stride = 0;
    uint32_t valid = 0;
};
static_assert(sizeof(P10DCHighRowAffine) == 16,
              "HIGH row affine descriptor should remain one 16-byte transaction");

__constant__ P10DCHighRowAffine* D_P10DC_HIGH_ROW_AFFINE;

__device__ __forceinline__ bool p10dc_high_row_affine_resolve(
    uint32_t owner, uint32_t bid, uint32_t rank, Count*& out
) {
    if (owner >= BUCKET_NGPU || bid >= D_BKF_MAIN_NBLOCKS) return false;
    P10DCHighRowAffine a = D_P10DC_HIGH_ROW_AFFINE[size_t(owner) * D_BKF_MAIN_NBLOCKS + bid];
    if (!a.valid || !a.base) return false;
    out = a.base + Code(rank) * a.stride;
    return true;
}

struct BucketFusedDirectHighRowsTables {
    BucketFusedZeroClosureTables base;
    P10DCHighRowAffine* high_row_affine = nullptr;
    uint32_t main_nblocks = 0;

    void install_metadata(
        const StorageLayout& layout, const BucketOrbitStreamsHost& o, const BucketFusedHost& f
    ) {
        base.install_metadata(layout, o, f);
        main_nblocks = uint32_t(layout.main_blocks.size());
        size_t n = size_t(BUCKET_NGPU) * main_nblocks;
        if (n) ck(cudaMalloc(&high_row_affine, n * sizeof(P10DCHighRowAffine)),
                  "p10dc high row affine alloc");
        ck(cudaMemcpyToSymbol(D_P10DC_HIGH_ROW_AFFINE, &high_row_affine, sizeof(high_row_affine)),
           "p10dc high row affine ptr");
    }

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        base.bind_owner(fixed, buckets, slot);
        std::vector<P10DCHighRowAffine> h(size_t(BUCKET_NGPU) * main_nblocks);
        for (uint32_t owner = 0; owner < BUCKET_NGPU; ++owner) {
            const auto& q = buckets.pair[owner][fixed];
            if (q.main_blocks.size() != main_nblocks) {
                std::cerr << "p10dc high row affine block-count mismatch owner=" << owner
                          << " got=" << q.main_blocks.size() << " expected=" << main_nblocks << '\n';
                std::exit(615);
            }
            for (uint32_t bid = 0; bid < main_nblocks; ++bid) {
                const BucketPhysicalBlock& b = q.main_blocks[bid];
                auto& a = h[size_t(owner) * main_nblocks + bid];
                if (b.valid && slot[owner]) {
                    a.base = slot[owner] + b.off;
                    a.stride = b.cols;
                    a.valid = 1;
                }
            }
        }
        if (!h.empty()) ck(cudaMemcpy(high_row_affine, h.data(), h.size() * sizeof(P10DCHighRowAffine),
                                      cudaMemcpyHostToDevice),
                           "p10dc high row affine H2D");
    }

    void release() {
        if (high_row_affine) cudaFree(high_row_affine);
        high_row_affine = nullptr;
        main_nblocks = 0;
        base.release();
    }
};
