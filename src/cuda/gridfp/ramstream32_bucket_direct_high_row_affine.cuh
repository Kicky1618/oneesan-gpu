#pragma once

#include "ramstream32_bucket_onepass_zero_alias.cuh"

// HIGH-window closure source rows have the affine form
//   slot[owner] + block.off + rank * block.cols.
// For pair[HIGH owner][fixed LOW owner], block.cols depends only on block.hs
// and the fixed LOW owner, never on the HIGH owner. Keep only row base in the
// (owner,bid) table and bind one tiny stride-by-hs table alongside it.
struct P10DCHighRowAffine {
    Count* base = nullptr;
};
static_assert(sizeof(P10DCHighRowAffine) == sizeof(Count*),
              "HIGH row affine entry should be pointer-only");
static_assert(sizeof(P10DCHighRowAffine) == 8,
              "64-bit CUDA pointers are required for the compact affine table");

__constant__ P10DCHighRowAffine* D_P10DC_HIGH_ROW_AFFINE;
__constant__ uint32_t D_P10DC_HIGH_ROW_STRIDE[MAXW + 2];

__device__ __forceinline__ bool p10dc_high_row_affine_resolve(
    uint32_t owner, uint32_t bid, uint32_t rank, uint32_t hs, Count*& out
) {
    if (owner >= BUCKET_NGPU || bid >= D_BKF_MAIN_NBLOCKS || hs >= uint32_t(MAXW + 2)) return false;
    P10DCHighRowAffine a = D_P10DC_HIGH_ROW_AFFINE[size_t(owner) * D_BKF_MAIN_NBLOCKS + bid];
    if (!a.base) return false;
    out = a.base + Code(rank) * D_P10DC_HIGH_ROW_STRIDE[hs];
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
        std::array<uint32_t, MAXW + 2> stride{};
        std::array<uint8_t, MAXW + 2> stride_seen{};
        for (uint32_t owner = 0; owner < BUCKET_NGPU; ++owner) {
            const auto& q = buckets.pair[owner][fixed];
            if (q.main_blocks.size() != main_nblocks) {
                std::cerr << "p10dc high row affine block-count mismatch owner=" << owner
                          << " got=" << q.main_blocks.size() << " expected=" << main_nblocks << '\n';
                std::exit(615);
            }
            for (uint32_t bid = 0; bid < main_nblocks; ++bid) {
                const BucketPhysicalBlock& b = q.main_blocks[bid];
                if (!b.valid) continue;
                if (b.hs >= MAXW + 2) {
                    std::cerr << "p10dc high row affine hs overflow hs=" << unsigned(b.hs) << '\n';
                    std::exit(617);
                }
                if (stride_seen[b.hs] && stride[b.hs] != b.cols) {
                    std::cerr << "p10dc high row affine stride mismatch hs=" << unsigned(b.hs)
                              << " old=" << stride[b.hs] << " new=" << b.cols << '\n';
                    std::exit(618);
                }
                stride_seen[b.hs] = 1;
                stride[b.hs] = b.cols;
                auto& a = h[size_t(owner) * main_nblocks + bid];
                if (slot[owner]) a.base = slot[owner] + b.off;
            }
        }
        if (!h.empty()) ck(cudaMemcpy(high_row_affine, h.data(), h.size() * sizeof(P10DCHighRowAffine),
                                      cudaMemcpyHostToDevice),
                           "p10dc high row affine H2D");
        ck(cudaMemcpyToSymbol(D_P10DC_HIGH_ROW_STRIDE, stride.data(), stride.size() * sizeof(uint32_t)),
           "p10dc high row affine stride by hs");
    }

    void release() {
        if (high_row_affine) cudaFree(high_row_affine);
        high_row_affine = nullptr;
        main_nblocks = 0;
        base.release();
    }
};
