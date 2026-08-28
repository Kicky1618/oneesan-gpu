#pragma push_macro("main")
#undef main
#define main pattern10_depthcode_rankformula_reference_main_unused
#include "ramstream32_bucket_reverse_fused_selftest.cu"
#pragma pop_macro("main")

#include "../ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_cross5.cuh"

static void p10dc_rankformula_run_high(
    BucketHostGrid& g, const StorageLayout& layout, const BucketPhysicalLayoutHost& phy,
    BucketFusedDirectHighRowsRankFormulaTables& dt,
    bool rev, bool experiment
) {
    constexpr int threads = 256, gx = 4, gy = 4;
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    const size_t smem = p10dc_direct_warpctx_smem_bytes(threads);
    for (uint32_t fixed = 0; fixed < BUCKET_NGPU; ++fixed) {
        std::array<Count*, BUCKET_NGPU> d{};
        bkft_alloc_slots(fixed, phy, d);
        for (uint32_t s = 0; s < BUCKET_NGPU; ++s) {
            const auto& v = g[s][fixed];
            if (!v.empty()) ck(cudaMemcpy(d[s], v.data(), v.size()*sizeof(Count),
                                          cudaMemcpyHostToDevice),
                               "p10dc rankformula H2D");
        }
        dt.bind_owner(fixed, phy, d);
        if (!rev) {
            for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
                if (experiment)
                    bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_cross5_kernel<<<grid,block,smem>>>(p);
                else
                    bucket_high_orbit_closure_pattern10_depthcode_resolved_kernel<<<grid,block>>>(p);
                ck(cudaGetLastError(), experiment
                    ? "p10dc rankformula forward" : "p10dc rankformula forward control");
            }
        } else {
            for (int p = LOW_LUT_K + 1; p < TARGET_W; ++p) {
                if (experiment)
                    bucket_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_rankformula_cross5_kernel<<<grid,block,smem>>>(p);
                else
                    bucket_reverse_high_pattern10_depthcode_resolved_kernel<<<grid,block>>>(p);
                ck(cudaGetLastError(), experiment
                    ? "p10dc rankformula reverse" : "p10dc rankformula reverse control");
            }
        }
        ck(cudaDeviceSynchronize(), "p10dc rankformula high sync");
        for (uint32_t s = 0; s < BUCKET_NGPU; ++s) {
            auto& v = g[s][fixed];
            if (!v.empty()) ck(cudaMemcpy(v.data(), d[s], v.size()*sizeof(Count),
                                          cudaMemcpyDeviceToHost),
                               "p10dc rankformula D2H");
        }
        bkft_free_slots(d);
    }
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W, L = LOW_LUT_K;
    static_assert(W <= 12, "rankformula CROSS5 selftest intentionally uses small width");
    static_assert(P10DC_RANKCHUNK32_BYTEPACK == 0);

    int visible = 0;
    cudaError_t ce = cudaGetDeviceCount(&visible);
    if (ce != cudaSuccess || visible < 1) {
        std::cout << "bucket-closure-pattern10-depthcode-rankformula-cross5-selftest SKIP no CUDA device\n";
        return 0;
    }
    ck(cudaSetDevice(0), "p10dc rankformula set device");

    build_full_dp();
    G_FACTOR = build_factor_tables();
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
    ReverseOrbitHost rlo = build_reverse_orbit(storage, layout, true);
    ReverseOrbitHost rhi = build_reverse_orbit(storage, layout, false);
    ReverseBucketAtomicHost rb = build_reverse_bucket_atomic(storage, layout, owner, rlow, rhigh, rlo, rhi);
    ReverseBucketFusedHost rf = build_reverse_bucket_fused_checked(layout, owner, rb);
    auto fh = build_bucket_forward_pattern10_depthcode_placeholder(layout, bo, bf);
    auto rh = build_bucket_reverse_pattern10_depthcode_zero_checked(layout, bo, bf, rb, rf);

    auto ms = gdg_enum_states(W), bs = gdg_enum_states(W - 1);
    std::unordered_map<MateID,size_t> mi, di;
    for (size_t i = 0; i < ms.size(); ++i) mi.emplace(ms[i], i);
    for (size_t i = 0; i < bs.size(); ++i) di.emplace(bs[i], i);
    std::mt19937_64 rng(0x16181624ULL);
    std::vector<Count> im(ms.size()), ib(bs.size());
    for (auto& x : im) x = Count(rng() % mod);
    for (auto& x : ib) x = Count(rng() % mod);
    auto [fhm, fhb] = gdg_reference_window(W, W - 1, L + 1, mod, ms, bs, mi, di, im, ib);
    auto [rhm, rhb] = bra_reference(L + 1, W - 1, mod, ms, bs, mi, di, im, ib);

    ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "p10dc rankformula modulus");
    BucketFusedDirectHighRowsRankFormulaTables dt;
    dt.install_metadata(layout, bo, bf);
    BucketForwardPattern10DepthCodeDeviceTables fdt;
    fdt.install(fh);
    BucketReversePattern10DepthCodeDeviceTables rdt;
    rdt.install(rh);
    p10dc_install_rankformula_lut();

    auto g0 = bkft_make_grid(ms, bs, im, ib, storage, layout, owner, phy);
    p10dc_rankformula_run_high(g0, layout, phy, dt, false, false);
    if (!bkft_compare("rankformula-forward-control", g0, ms, bs, fhm, fhb,
                      storage, layout, owner, phy)) return 90;

    auto g1 = bkft_make_grid(ms, bs, im, ib, storage, layout, owner, phy);
    p10dc_rankformula_run_high(g1, layout, phy, dt, false, true);
    if (!bkft_compare("rankformula-forward", g1, ms, bs, fhm, fhb,
                      storage, layout, owner, phy)) return 91;

    auto g2 = bkft_make_grid(ms, bs, im, ib, storage, layout, owner, phy);
    p10dc_rankformula_run_high(g2, layout, phy, dt, true, false);
    if (!bkft_compare("rankformula-reverse-control", g2, ms, bs, rhm, rhb,
                      storage, layout, owner, phy)) return 92;

    auto g3 = bkft_make_grid(ms, bs, im, ib, storage, layout, owner, phy);
    p10dc_rankformula_run_high(g3, layout, phy, dt, true, true);
    if (!bkft_compare("rankformula-reverse", g3, ms, bs, rhm, rhb,
                      storage, layout, owner, phy)) return 93;

    rdt.release();
    fdt.release();
    dt.release();
    std::cout << "bucket-closure-pattern10-depthcode-rankformula-cross5-selftest OK W=" << W
              << " forward_exact=1 reverse_exact=1"
              << " rankstream_bytes=0 chunk_meta_bytes_per_code=4"
              << " base16=1 formula_ballot=1"
              << " sparse_base=" << P10DC_RANKFORMULA_SPARSE_BASE
              << " base_delta=" << P10DC_RANKFORMULA_BASE_DELTA
              << " base_values_per_lookup=" << (P10DC_RANKFORMULA_BASE_DELTA ? 1 : 2)
              << " rawcode=" << P10DC_RANKFORMULA_RAWCODE
              << " slotmeta=" << P10DC_RANKFORMULA_SLOTMETA
              << " chunkinfo_elided=" << (P10DC_RANKFORMULA_RAWCODE || P10DC_RANKFORMULA_SLOTMETA)
              << " broadword_support_elided=" << P10DC_RANKFORMULA_SLOTMETA
              << " direct_mask_slot_elided=" << P10DC_RANKFORMULA_SLOTMETA
              << " inline_cross=" << P10DC_RANKFORMULA_INLINE_CROSS
              << " cross_lut_elided=" << P10DC_RANKFORMULA_INLINE_CROSS
              << " fused13=" << P10DC_RANKDELTA8_FUSED13
              << " cross_runtime_div=0 cross_runtime_mod=0\n";
    return 0;
}
