#pragma push_macro("main")
#undef main
#define main compact_flat_bid_reference_main_unused
#include "ramstream32_bucket_reverse_fused_selftest.cu"
#pragma pop_macro("main")

#define P10DC_RANKFORMULA_PRECTX_FORWARD 1
#define P10DC_RANKFORMULA_PRECTX_REVERSE 1
#define P10DC_RANKFORMULA_PRECTX_COMPACT 1
#define P10DC_RANKFORMULA_PRECTX_FLAT_BID 1
#define P10DC_RANKFORMULA_PRECTX_FLAT_BID_FUSED 0
#include "../ramstream32_bucket_precomputed_high_ctx_compact_flat_bid.cuh"

struct CompactFlatBidCheck {
    uint64_t checked = 0;
    uint32_t errors = 0;
};

__device__ __forceinline__ void compact_flat_bid_note(
    CompactFlatBidCheck* out, uint32_t got, uint32_t expected
) {
    atomicAdd(reinterpret_cast<unsigned long long*>(&out->checked), 1ull);
    if (got != expected) atomicOr(&out->errors, 1u);
}

__global__ void compact_flat_bid_forward_check_kernel(int p, CompactFlatBidCheck* out) {
    const uint32_t bid = uint32_t(blockIdx.x);
    const uint32_t nb = D_BKF_MAIN_NBLOCKS;
    if (bid >= nb) return;
    const uint32_t pi = uint32_t((TARGET_W - 1) - p);
    const uint32_t oi = uint32_t(size_t(pi) * D_BKF_HIGH_PITCH + bid);
    for (uint32_t q = D_BKF_HIGH_NN_OFF[oi] + threadIdx.x;
         q < D_BKF_HIGH_NN_OFF[oi + 1u]; q += blockDim.x)
        compact_flat_bid_note(out, uint32_t(D_P10DC_COMPACT_PRECTX_FWD_NN[q].pad), bid);
    for (uint32_t q = D_BKF_HIGH_NRNL_OFF[oi] + threadIdx.x;
         q < D_BKF_HIGH_NRNL_OFF[oi + 1u]; q += blockDim.x)
        compact_flat_bid_note(out, uint32_t(D_P10DC_COMPACT_PRECTX_FWD_NRNL[q].pad), bid);
}

__global__ void compact_flat_bid_reverse_check_kernel(int p, CompactFlatBidCheck* out) {
    const uint32_t bid = uint32_t(blockIdx.x);
    const uint32_t nb = D_BKF_MAIN_NBLOCKS;
    if (bid >= nb) return;
    const uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
    const uint32_t oi = uint32_t(size_t(pi) * D_RS54_PITCH + bid);
    for (uint32_t q = D_RS54_HIGH_NN_OFF[oi] + threadIdx.x;
         q < D_RS54_HIGH_NN_OFF[oi + 1u]; q += blockDim.x)
        compact_flat_bid_note(out, uint32_t(D_P10DC_COMPACT_PRECTX_REV_NN[q].pad), bid);
    for (uint32_t q = D_RS54_HIGH_NR_OFF[oi] + threadIdx.x;
         q < D_RS54_HIGH_NR_OFF[oi + 1u]; q += blockDim.x)
        compact_flat_bid_note(out, uint32_t(D_P10DC_COMPACT_PRECTX_REV_NR[q].pad), bid);
    for (uint32_t q = D_RS54_HIGH_NL_OFF[oi] + threadIdx.x;
         q < D_RS54_HIGH_NL_OFF[oi + 1u]; q += blockDim.x)
        compact_flat_bid_note(out, uint32_t(D_P10DC_COMPACT_PRECTX_REV_NL[q].pad), bid);
}

int main() {
    static_assert(TARGET_W <= 12,
                  "compact flat-bid selftest intentionally uses a small exhaustive width");
    static_assert(P10DC_HIGH_ROW_AFFINE_BLOCKS <= 255u);
    int visible = 0;
    cudaError_t ce = cudaGetDeviceCount(&visible);
    if (ce != cudaSuccess || visible < 1) {
        std::cout << "bucket-compact-flat-bid-selftest SKIP no CUDA device\n";
        return 0;
    }
    ck(cudaSetDevice(0), "compact flat bid set device");

    build_full_dp(); G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost loworbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuHighDirectHost highdirect = build_cpu_high_direct(storage, layout, highdesc);
    GpuDirectGatherHost ordinary = build_gpu_direct_gather(layout, lowdesc, loworbit, highdirect);
    GpuDirectCrossGatherHost cross = build_gpu_direct_cross_gather(storage, layout, lowdesc, loworbit, highdirect);
    GpuDirectFusedHost fused = build_gpu_direct_fused_checked(layout, ordinary, cross, fused);
    CpuLowSparseHost lowsparse = build_cpu_low_sparse(storage, layout, lowdesc, loworbit);
    BucketOwnerHost owner = build_bucket_owners(G_FACTOR, storage);
    BucketPhysicalLayoutHost phy = build_bucket_physical_layout(layout, owner);
    BucketOrbitStreamsHost bo = build_bucket_orbits(storage, layout, owner, lowsparse, highdirect);
    BucketFusedHost bf = build_bucket_fused(storage, layout, owner, ordinary, cross, fused);

    ReverseLowDescHost rlow = build_reverse_low_descriptors(storage, layout);
    ReverseHighDescHost rhigh = build_reverse_high_descriptors(storage, layout);
    ReverseOrbitHost rlo = build_reverse_orbit(storage, layout, true),
                     rhi = build_reverse_orbit(storage, layout, false);
    ReverseBucketAtomicHost rb = build_reverse_bucket_atomic(
        storage, layout, owner, rlow, rhigh, rlo, rhi);
    ReverseBucketFusedHost rf = build_reverse_bucket_fused_checked(layout, owner, rb);
    auto fh = build_bucket_forward_pattern10_depthcode_placeholder(layout, bo, bf);
    auto rh = build_bucket_reverse_pattern10_depthcode_zero_checked(layout, bo, bf, rb, rf);

    using Tables = BucketFusedCompactPrecomputedHighCtxFlatBidTables<BucketFusedDirectHighRowsTables>;
    Tables dt;
    dt.install_metadata(layout, bo, bf);
    BucketForwardPattern10DepthCodeDeviceTables fdt; fdt.install(fh);
    BucketReversePattern10DepthCodeDeviceTables rdt; rdt.install(rh);

    CompactFlatBidCheck* dcheck = nullptr;
    ck(cudaMalloc(&dcheck, sizeof(CompactFlatBidCheck)), "compact flat bid check alloc");
    uint64_t total_checked = 0;
    for (uint32_t fixed = 0; fixed < BUCKET_NGPU; ++fixed) {
        std::array<Count*, BUCKET_NGPU> d{};
        bkft_alloc_slots(fixed, phy, d);
        dt.bind_owner(fixed, phy, d);
        ck(cudaMemset(dcheck, 0, sizeof(CompactFlatBidCheck)), "compact flat bid check zero");
        const dim3 grid(unsigned(layout.main_blocks.size())), block(128);
        for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
            compact_flat_bid_forward_check_kernel<<<grid, block>>>(p, dcheck);
            ck(cudaGetLastError(), "compact flat bid forward check launch");
        }
        for (int p = LOW_LUT_K + 1; p < TARGET_W; ++p) {
            compact_flat_bid_reverse_check_kernel<<<grid, block>>>(p, dcheck);
            ck(cudaGetLastError(), "compact flat bid reverse check launch");
        }
        ck(cudaDeviceSynchronize(), "compact flat bid checks sync");
        CompactFlatBidCheck h{};
        ck(cudaMemcpy(&h, dcheck, sizeof(h), cudaMemcpyDeviceToHost),
           "compact flat bid check copy");
        if (h.errors || !h.checked) {
            std::cerr << "compact flat bid mismatch fixed=" << fixed
                      << " checked=" << h.checked << " errors=" << h.errors << '\n';
            return 70;
        }
        total_checked += h.checked;
        dt.release();
        bkft_free_slots(d);
        if (fixed + 1u < BUCKET_NGPU) {
            dt.install_metadata(layout, bo, bf);
        }
    }

    cudaFree(dcheck);
    rdt.release(); fdt.release();
    std::cout << "bucket-compact-flat-bid-selftest OK W=" << TARGET_W
              << " checked=" << total_checked
              << " main_blocks=" << layout.main_blocks.size()
              << " context_bytes=" << sizeof(P10DCHighClosureCompactPreCtx)
              << " bytes_added=0 forward_exact=1 reverse_exact=1\n";
    return 0;
}
