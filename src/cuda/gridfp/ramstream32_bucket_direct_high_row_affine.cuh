#pragma once

#include "ramstream32_bucket_onepass_zero_alias.cuh"

// HIGH-window closure source rows have the affine form
//   slot[owner] + block.off + rank * block.cols.
// For pair[HIGH owner][fixed LOW owner], block.cols depends only on block.hs
// and the fixed LOW owner. main_nblocks is exactly 3*(HIGH_LUT_K+2), so the
// pointer-only row-base table is only a few KiB even at W28 and fits naturally
// in CUDA constant memory.
struct P10DCHighRowAffine {
    Count* base = nullptr;
};
static_assert(sizeof(P10DCHighRowAffine) == sizeof(Count*),
              "HIGH row affine entry should be pointer-only");
static_assert(sizeof(P10DCHighRowAffine) == 8,
              "64-bit CUDA pointers are required for the compact affine table");

static constexpr uint32_t P10DC_HIGH_ROW_AFFINE_BLOCKS = 3u * uint32_t(HIGH_LUT_K + 2);
static constexpr uint32_t P10DC_HIGH_ROW_AFFINE_CAP = BUCKET_NGPU * P10DC_HIGH_ROW_AFFINE_BLOCKS;
__constant__ P10DCHighRowAffine D_P10DC_HIGH_ROW_AFFINE[P10DC_HIGH_ROW_AFFINE_CAP];
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
    uint32_t main_nblocks = 0;

    void install_metadata(
        const StorageLayout& layout, const BucketOrbitStreamsHost& o, const BucketFusedHost& f
    ) {
        base.install_metadata(layout, o, f);
        main_nblocks = uint32_t(layout.main_blocks.size());
        if (main_nblocks != P10DC_HIGH_ROW_AFFINE_BLOCKS) {
            std::cerr << "p10dc high row affine main-block mismatch got=" << main_nblocks
                      << " expected=" << P10DC_HIGH_ROW_AFFINE_BLOCKS << '\n';
            std::exit(619);
        }
    }

    void bind_owner(
        uint32_t fixed, const BucketPhysicalLayoutHost& buckets,
        const std::array<Count*, BUCKET_NGPU>& slot
    ) {
        base.bind_owner(fixed, buckets, slot);
        std::array<P10DCHighRowAffine, P10DC_HIGH_ROW_AFFINE_CAP> h{};
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
        ck(cudaMemcpyToSymbol(D_P10DC_HIGH_ROW_AFFINE, h.data(), h.size() * sizeof(P10DCHighRowAffine)),
           "p10dc high row affine constant table");
        ck(cudaMemcpyToSymbol(D_P10DC_HIGH_ROW_STRIDE, stride.data(), stride.size() * sizeof(uint32_t)),
           "p10dc high row affine stride by hs");
    }

    void release() {
        main_nblocks = 0;
        base.release();
    }
};
