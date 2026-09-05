#pragma push_macro("main")
#undef main
#define main compact_prectx_reference_main_unused
#include "ramstream32_bucket_reverse_fused_selftest.cu"
#pragma pop_macro("main")

#define P10DC_RANKFORMULA_PRECTX_FORWARD 1
#define P10DC_RANKFORMULA_PRECTX_REVERSE 1
#define P10DC_RANKFORMULA_PRECTX_COMPACT 1
#include "../ramstream32_bucket_precomputed_high_ctx_compact.cuh"

struct CompactPrectxCheck {
    uint32_t checked = 0;
    uint32_t errors = 0;
};

__device__ __forceinline__ void compact_prectx_compare(
    const P10DCDirectHighResolvedCtx& ref,
    const P10DCHighClosureCompactPreCtx& z,
    uint32_t expected_hs,
    CompactPrectxCheck* out
) {
    uint32_t err = 0;
    if (z.fixed_hs != expected_hs) err |= 1u;
    if (uint32_t(z.local_n) != uint32_t(ref.local_n)) err |= 2u;
    if (uint32_t(z.cross_depth) != ref.cross_depth) err |= 4u;
#pragma unroll
    for (uint32_t i = 0; i < BKCZ_MAX_LOCAL; ++i) {
        if (i < uint32_t(ref.local_n)) {
            Count* p = p10dc_high_row_ref_resolve_unchecked(z.local_ref[i], z.fixed_hs);
            if (p != ref.local_base[i]) err |= 8u;
        }
    }
    Count* cross = z.cross_depth
        ? p10dc_high_row_ref_resolve_unchecked(z.cross_ref, uint32_t(z.fixed_hs) + 2u)
        : nullptr;
    if (cross != ref.cross_base) err |= 16u;
    atomicAdd(&out->checked, 1u);
    if (err) atomicOr(&out->errors, err);
}

__global__ void compact_prectx_forward_check_kernel(int p, CompactPrectxCheck* out) {
    const uint32_t bid = uint32_t(blockIdx.x);
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    const uint32_t pi = uint32_t((TARGET_W - 1) - p);
    const uint32_t oi = uint32_t(size_t(pi) * D_BKF_HIGH_PITCH + bid);

    const uint32_t na = D_BKF_HIGH_NN_OFF[oi], nb = D_BKF_HIGH_NN_OFF[oi + 1u];
    for (uint32_t qi = na + threadIdx.x; qi < nb; qi += blockDim.x) {
        const BucketOrbitOp op = D_BKF_HIGH_NN[qi];
        const uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
        const uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
        P10DCDirectHighResolvedCtx ref{};
        ref.xb = bkf_high_main(ss, bid);
        if (!(ref.xb.valid && ref.xb.rows && ref.xb.cols)) continue;
        uint32_t jbid = bid;
        if (p == LOW_LUT_K + 1) {
            const uint32_t center = uint32_t(R);
            const int he = int(ref.xb.hs) + 1;
            jbid = uint32_t(3 * he + int(center));
        }
        ref.jb = bkf_high_main(js, jbid);
        ref.db = bkf_high_block(ds, uint32_t(ref.xb.hs));
        const uint32_t payload = p10dc_payload(op, false, true, 0u, p, uint32_t(ref.xb.hs));
        p10dc_prepare_forward_high_delta_direct_affine(
            ref, payload, dl, p, ss, js, ds,
            bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
        const auto z = p10dc_make_forward_compact_prectx(bid, p, op, true);
        compact_prectx_compare(ref, z, uint32_t(ref.db.hs), out);
    }

    const uint32_t ra = D_BKF_HIGH_NRNL_OFF[oi], rb = D_BKF_HIGH_NRNL_OFF[oi + 1u];
    for (uint32_t qi = ra + threadIdx.x; qi < rb; qi += blockDim.x) {
        const BucketOrbitOp op = D_BKF_HIGH_NRNL[qi];
        const uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
        const uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
        P10DCDirectHighResolvedCtx ref{};
        ref.xb = bkf_high_main(ss, bid);
        if (!(ref.xb.valid && ref.xb.rows && ref.xb.cols)) continue;
        uint32_t jbid = bid;
        if (p == LOW_LUT_K + 1) {
            const uint32_t center = uint32_t(N);
            const int he = int(ref.xb.hs);
            jbid = uint32_t(3 * he + int(center));
        }
        ref.jb = bkf_high_main(js, jbid);
        ref.db = bkf_high_block(ds, uint32_t(ref.xb.hs));
        const uint32_t payload = p10dc_payload(op, false, true, 3u, p, uint32_t(ref.xb.hs));
        p10dc_prepare_forward_high_delta_direct_affine(
            ref, payload, dl, p, ss, js, ds,
            bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
        const auto z = p10dc_make_forward_compact_prectx(bid, p, op, false);
        compact_prectx_compare(ref, z, uint32_t(ref.db.hs), out);
    }
}

__global__ void compact_prectx_reverse_check_kernel(int p, CompactPrectxCheck* out) {
    const uint32_t bid = uint32_t(blockIdx.x);
    if (bid >= D_BKF_MAIN_NBLOCKS) return;
    const uint32_t pi = uint32_t(p - (LOW_LUT_K + 1));
    const uint32_t oi = uint32_t(size_t(pi) * D_RS54_PITCH + bid);
    const bool edge = p == TARGET_W - 1;

    auto check_one = [&](BucketOrbitOp op, uint32_t kind, uint32_t sid, uint32_t qi) {
        const uint32_t sl = bkf_orbit_src(op), jl = bkf_orbit_partner(op), dl = bkf_orbit_drop(op);
        const uint32_t ss = bkf_loc_owner(sl), js = bkf_loc_owner(jl), ds = bkf_loc_owner(dl);
        P10DCDirectHighResolvedCtx ref{};
        ref.xb = bkf_high_main(ss, bid);
        if (!(ref.xb.valid && ref.xb.rows && ref.xb.cols)) return;
        ref.jb = bkf_high_main(js, bkcp10_reverse_high_jblock(bid, ref.xb, p, kind));
        ref.db = bkf_high_block(ds, uint32_t(ref.xb.hs));
        const BucketPhysicalBlock plan_db = edge ? ref.xb : ref.db;
        const uint32_t payload = p10dc_payload(op, true, true, sid, p, uint32_t(ref.xb.hs));
        p10dc_prepare_reverse_high_delta_direct_affine(
            ref, payload, edge ? sl : dl, plan_db, p, edge,
            ss, js, ds, bkf_loc_rank(sl), bkf_loc_rank(jl), bkf_loc_rank(dl));
        const auto z = p10dc_make_reverse_compact_prectx(bid, p, op, kind, sid);
        compact_prectx_compare(ref, z, uint32_t(plan_db.hs), out);
        (void)qi;
    };

    const uint32_t na = D_RS54_HIGH_NN_OFF[oi], nb = D_RS54_HIGH_NN_OFF[oi + 1u];
    for (uint32_t qi = na + threadIdx.x; qi < nb; qi += blockDim.x)
        check_one(D_RS54_HIGH_NN[qi], CPU_ORBIT_NN, 0u, qi);
    const uint32_t ra = D_RS54_HIGH_NR_OFF[oi], rb = D_RS54_HIGH_NR_OFF[oi + 1u];
    for (uint32_t qi = ra + threadIdx.x; qi < rb; qi += blockDim.x)
        check_one(D_RS54_HIGH_NR[qi], CPU_ORBIT_NR, 1u, qi);
    const uint32_t la = D_RS54_HIGH_NL_OFF[oi], lb = D_RS54_HIGH_NL_OFF[oi + 1u];
    for (uint32_t qi = la + threadIdx.x; qi < lb; qi += blockDim.x)
        check_one(D_RS54_HIGH_NL[qi], CPU_ORBIT_NL, 2u, qi);
}

int main() {
    static_assert(TARGET_W <= 12,
                  "compact prectx selftest intentionally uses a small exhaustive width");
    int visible = 0;
    cudaError_t ce = cudaGetDeviceCount(&visible);
    if (ce != cudaSuccess || visible < 1) {
        std::cout << "bucket-compact-prectx-selftest SKIP no CUDA device\n";
        return 0;
    }
    ck(cudaSetDevice(0), "compact prectx set device");

    build_full_dp(); G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost loworbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuHighDirectHost highdirect = build_cpu_high_direct(storage, layout, highdesc);
    GpuDirectGatherHost ordinary = build_gpu_direct_gather(layout, lowdesc, loworbit, highdirect);
    GpuDirectCrossGatherHost cross = build_gpu_direct_cross_gather(storage, layout, lowdesc, loworbit, highdirect);
    GpuDirectFusedHost fused = build_gpu_direct_fused_checked(layout, ordinary, cross);
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

    BucketFusedDirectHighRowsTables dt;
    dt.install_metadata(layout, bo, bf);
    BucketForwardPattern10DepthCodeDeviceTables fdt; fdt.install(fh);
    BucketReversePattern10DepthCodeDeviceTables rdt; rdt.install(rh);

    CompactPrectxCheck* dcheck = nullptr;
    ck(cudaMalloc(&dcheck, sizeof(CompactPrectxCheck)), "compact prectx check alloc");
    uint64_t total_checked = 0;
    for (uint32_t fixed = 0; fixed < BUCKET_NGPU; ++fixed) {
        std::array<Count*, BUCKET_NGPU> d{};
        bkft_alloc_slots(fixed, phy, d);
        dt.bind_owner(fixed, phy, d);
        ck(cudaMemset(dcheck, 0, sizeof(CompactPrectxCheck)), "compact prectx check zero");
        const dim3 grid(unsigned(layout.main_blocks.size())), block(128);
        for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
            compact_prectx_forward_check_kernel<<<grid, block>>>(p, dcheck);
            ck(cudaGetLastError(), "compact prectx forward check launch");
        }
        for (int p = LOW_LUT_K + 1; p < TARGET_W; ++p) {
            compact_prectx_reverse_check_kernel<<<grid, block>>>(p, dcheck);
            ck(cudaGetLastError(), "compact prectx reverse check launch");
        }
        ck(cudaDeviceSynchronize(), "compact prectx checks sync");
        CompactPrectxCheck h{};
        ck(cudaMemcpy(&h, dcheck, sizeof(h), cudaMemcpyDeviceToHost),
           "compact prectx check copy");
        if (h.errors) {
            std::cerr << "compact prectx mismatch fixed=" << fixed
                      << " checked=" << h.checked << " error_mask=" << h.errors << '\n';
            return 70;
        }
        total_checked += h.checked;
        bkft_free_slots(d);
    }

    cudaFree(dcheck);
    rdt.release(); fdt.release(); dt.release();
    std::cout << "bucket-compact-prectx-selftest OK W=" << TARGET_W
              << " checked=" << total_checked
              << " locator_bits=" << BUCKET_LOCATOR_BITS
              << " block_bits=6 compact_context_bytes="
              << sizeof(P10DCHighClosureCompactPreCtx)
              << " runtime_context_bytes=" << sizeof(P10DCDirectHighResolvedCtx)
              << " forward_exact=1 reverse_exact=1\n";
    return 0;
}
