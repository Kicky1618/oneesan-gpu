#pragma push_macro("main")
#undef main
#define main pattern10_depthcode_rank16_reference_main_unused
#include "ramstream32_bucket_reverse_fused_selftest.cu"
#pragma pop_macro("main")

#include "../ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_prekey_rank16_cross5.cuh"

static std::array<uint8_t, BUCKET_NGPU> P10DC_RANK16_TABLE_VERIFIED{};

static void p10dc_verify_rank16_tables(
    const BucketFusedHost& bf, const BucketFusedDirectHighRowsPrekeyRank16Tables& dt,
    uint32_t fixed
) {
    if (P10DC_RANK16_TABLE_VERIFIED[fixed]) return;
    constexpr size_t P = size_t(MAXW + 2);
    const size_t owner_base = size_t(fixed) * P;
    const uint32_t owner_begin = bf.low_code_off[owner_base];
    const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
        ? bf.low_code_off[size_t(fixed + 1u) * P]
        : uint32_t(bf.low_codes.size());
    const size_t code_count = size_t(owner_end - owner_begin);
    if (dt.low_prekey_count != code_count ||
        dt.low_rank16_count != code_count * size_t(LOW_LUT_K)) {
        std::cerr << "p10dc rank16 table count mismatch owner=" << fixed
                  << " prekey=" << dt.low_prekey_count << '/' << code_count
                  << " rank16=" << dt.low_rank16_count << '/'
                  << code_count * size_t(LOW_LUT_K) << '\n';
        std::exit(636);
    }

    std::vector<uint32_t> got_key(code_count);
    std::vector<uint16_t> got_rank(dt.low_rank16_count);
    if (!got_key.empty())
        ck(cudaMemcpy(got_key.data(), dt.low_prekey, got_key.size() * sizeof(uint32_t),
                      cudaMemcpyDeviceToHost),
           "p10dc rank16 prekey verify D2H");
    if (!got_rank.empty())
        ck(cudaMemcpy(got_rank.data(), dt.low_rank16, got_rank.size() * sizeof(uint16_t),
                      cudaMemcpyDeviceToHost),
           "p10dc rank16 table verify D2H");
    std::array<uint32_t, MAXW + 2> got_hoff{};
    ck(cudaMemcpyFromSymbol(got_hoff.data(), D_P10DC_LOW_PREKEY_HOFF,
                            got_hoff.size() * sizeof(uint32_t)),
       "p10dc rank16 height offsets verify");

    size_t compact = 0;
    for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
        if (got_hoff[h] != compact) {
            std::cerr << "p10dc rank16 height offset mismatch owner=" << fixed
                      << " h=" << h << " got=" << got_hoff[h]
                      << " expected=" << compact << '\n';
            std::exit(637);
        }
        uint32_t a = bf.low_code_off[owner_base + h];
        uint32_t b = h + 1u < uint32_t(MAXW + 2)
            ? bf.low_code_off[owner_base + h + 1u]
            : owner_end;
        for (uint32_t i = a; i < b; ++i, ++compact) {
            uint32_t code = bf.low_codes[i];
            uint32_t key = gpu_direct_ternary_key_host(code, LOW_LUT_K);
            if (compact >= got_key.size() || got_key[compact] != key) {
                std::cerr << "p10dc rank16 prekey mismatch owner=" << fixed
                          << " h=" << h << " rank=" << (i - a) << '\n';
                std::exit(638);
            }
            uint32_t weight = 1u;
            for (int pos = 0; pos < LOW_LUT_K; ++pos) {
                uint16_t expected = P10DC_LOW_RANK16_INVALID;
                uint32_t v = (code >> (2 * pos)) & 3u;
                if (v == uint32_t(::L)) {
                    uint32_t x = bf.low_direct[key - weight];
                    if (x != BKF_DIRECT_INVALID) {
                        uint32_t loc = x & BKF_LOC_MASK;
                        if (bkf_loc_owner(loc) != fixed) {
                            std::cerr << "p10dc rank16 verify owner mismatch fixed=" << fixed
                                      << " got=" << bkf_loc_owner(loc) << '\n';
                            std::exit(639);
                        }
                        expected = uint16_t(bkf_loc_rank(loc));
                    }
                }
                size_t ix = compact * size_t(LOW_LUT_K) + size_t(pos);
                if (ix >= got_rank.size() || got_rank[ix] != expected) {
                    std::cerr << "p10dc rank16 value mismatch owner=" << fixed
                              << " h=" << h << " rank=" << (i - a)
                              << " pos=" << pos
                              << " got=" << (ix < got_rank.size() ? got_rank[ix] : 0xffffu)
                              << " expected=" << expected << '\n';
                    std::exit(640);
                }
                weight *= 3u;
            }
        }
    }
    if (compact != code_count) {
        std::cerr << "p10dc rank16 compact walk mismatch owner=" << fixed
                  << " got=" << compact << " expected=" << code_count << '\n';
        std::exit(641);
    }
    P10DC_RANK16_TABLE_VERIFIED[fixed] = 1;
}

static void p10dc_rank16_run_high(
    BucketHostGrid& g, const StorageLayout& layout, const BucketPhysicalLayoutHost& phy,
    const BucketFusedHost& bf, BucketFusedDirectHighRowsPrekeyRank16Tables& dt,
    bool rev, bool rank16
) {
    constexpr int threads = 256, gx = 4, gy = 4;
    dim3 block(threads), grid(gx, gy, unsigned(layout.main_blocks.size()));
    const size_t smem = p10dc_direct_warpctx_smem_bytes(threads);
    for (uint32_t fixed = 0; fixed < BUCKET_NGPU; ++fixed) {
        std::array<Count*, BUCKET_NGPU> d{};
        bkft_alloc_slots(fixed, phy, d);
        for (uint32_t s = 0; s < BUCKET_NGPU; ++s) {
            const auto& v = g[s][fixed];
            if (!v.empty()) ck(cudaMemcpy(d[s], v.data(), v.size() * sizeof(Count),
                                          cudaMemcpyHostToDevice),
                               "p10dc rank16 high H2D");
        }
        dt.bind_owner(fixed, phy, d);
        p10dc_verify_rank16_tables(bf, dt, fixed);
        if (!rev) {
            for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
                if (rank16)
                    bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_prekey_rank16_cross5_kernel<<<grid, block, smem>>>(p);
                else
                    bucket_high_orbit_closure_pattern10_depthcode_resolved_kernel<<<grid, block>>>(p);
                ck(cudaGetLastError(), rank16 ? "p10dc rank16 forward high" :
                                                "p10dc resolved forward high rank16 control");
            }
        } else {
            for (int p = LOW_LUT_K + 1; p < TARGET_W; ++p) {
                if (rank16)
                    bucket_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_prekey_rank16_cross5_kernel<<<grid, block, smem>>>(p);
                else
                    bucket_reverse_high_pattern10_depthcode_resolved_kernel<<<grid, block>>>(p);
                ck(cudaGetLastError(), rank16 ? "p10dc rank16 reverse high" :
                                                "p10dc resolved reverse high rank16 control");
            }
        }
        ck(cudaDeviceSynchronize(), "p10dc rank16 high sync");
        for (uint32_t s = 0; s < BUCKET_NGPU; ++s) {
            auto& v = g[s][fixed];
            if (!v.empty()) ck(cudaMemcpy(v.data(), d[s], v.size() * sizeof(Count),
                                          cudaMemcpyDeviceToHost),
                               "p10dc rank16 high D2H");
        }
        bkft_free_slots(d);
    }
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W, L = LOW_LUT_K;
    static_assert(W <= 12, "rank16 CROSS5 selftest intentionally uses small width");
    int visible = 0;
    cudaError_t ce = cudaGetDeviceCount(&visible);
    if (ce != cudaSuccess || visible < 1) {
        std::cout << "bucket-closure-pattern10-depthcode-rank16-cross5-selftest SKIP no CUDA device\n";
        return 0;
    }
    ck(cudaSetDevice(0), "p10dc rank16 set device");

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

    auto ms = gdg_enum_states(W), bs = gdg_enum_states(W - 1);
    std::unordered_map<MateID, size_t> mi, di;
    for (size_t i = 0; i < ms.size(); ++i) mi.emplace(ms[i], i);
    for (size_t i = 0; i < bs.size(); ++i) di.emplace(bs[i], i);
    std::mt19937_64 rng(0x1618a116ULL);
    std::vector<Count> im(ms.size()), ib(bs.size());
    for (auto& x : im) x = Count(rng() % mod);
    for (auto& x : ib) x = Count(rng() % mod);
    auto [fhm, fhb] = gdg_reference_window(W, W - 1, L + 1, mod, ms, bs, mi, di, im, ib);
    auto [rhm, rhb] = bra_reference(L + 1, W - 1, mod, ms, bs, mi, di, im, ib);

    ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "p10dc rank16 modulus");
    BucketFusedDirectHighRowsPrekeyRank16Tables dt; dt.install_metadata(layout, bo, bf);
    BucketForwardPattern10DepthCodeDeviceTables fdt; fdt.install(fh);
    BucketReversePattern10DepthCodeDeviceTables rdt; rdt.install(rh);
    p10dc_install_cross5_lut();

    auto g0 = bkft_make_grid(ms, bs, im, ib, storage, layout, owner, phy);
    p10dc_rank16_run_high(g0, layout, phy, bf, dt, false, false);
    if (!bkft_compare("pattern10-depthcode-forward-high-resolved-rank16-control",
                      g0, ms, bs, fhm, fhb, storage, layout, owner, phy)) return 60;
    auto g1 = bkft_make_grid(ms, bs, im, ib, storage, layout, owner, phy);
    p10dc_rank16_run_high(g1, layout, phy, bf, dt, false, true);
    if (!bkft_compare("pattern10-depthcode-forward-high-rank16-cross5",
                      g1, ms, bs, fhm, fhb, storage, layout, owner, phy)) return 61;

    auto g2 = bkft_make_grid(ms, bs, im, ib, storage, layout, owner, phy);
    p10dc_rank16_run_high(g2, layout, phy, bf, dt, true, false);
    if (!bkft_compare("pattern10-depthcode-reverse-high-resolved-rank16-control",
                      g2, ms, bs, rhm, rhb, storage, layout, owner, phy)) return 62;
    auto g3 = bkft_make_grid(ms, bs, im, ib, storage, layout, owner, phy);
    p10dc_rank16_run_high(g3, layout, phy, bf, dt, true, true);
    if (!bkft_compare("pattern10-depthcode-reverse-high-rank16-cross5",
                      g3, ms, bs, rhm, rhb, storage, layout, owner, phy)) return 63;

    for (uint32_t g = 0; g < BUCKET_NGPU; ++g)
        if (!P10DC_RANK16_TABLE_VERIFIED[g]) return 64;
    rdt.release(); fdt.release(); dt.release();
    std::cout << "bucket-closure-pattern10-depthcode-rank16-cross5-selftest OK W=" << W
              << " control=resolved experiment=warpstriped_delta_direct_affine_prekey_rank16_cross5"
              << " forward_exact=1 reverse_exact=1 prekey_scope=fixed_owner"
              << " prekey_table_exact=1 rank16_table_exact=1"
              << " rank16_bytes_per_low_code=" << (LOW_LUT_K * sizeof(uint16_t))
              << " cross_runtime_ternary_fold=0 cross_runtime_direct_lookup=0"
              << " fallback_structurally_unreachable=1 cross5_table_bytes=6561"
              << " pm_accum=" << GPU_DIRECT_PM_ACCUM << '\n';
    return 0;
}
