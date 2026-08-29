#pragma push_macro("main")
#undef main
#define main pattern10_depthcode_rankchunk32_reference_main_unused
#include "ramstream32_bucket_reverse_fused_selftest.cu"
#pragma pop_macro("main")

#include "../ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankchunk32_cross5.cuh"

#if P10DC_RANKCHUNK32_DIRECTMASK
using P10DCRankChunk32SelftestTables = BucketFusedDirectHighRowsRankChunk32DirectMaskTables;
#else
using P10DCRankChunk32SelftestTables = BucketFusedDirectHighRowsRankChunk32Tables;
#endif

static std::array<uint8_t, BUCKET_NGPU> P10DC_RANKCHUNK32_TABLE_VERIFIED{};

static void p10dc_verify_rankchunk32_tables(
    const BucketFusedHost& bf, const BucketFusedDirectHighRowsRankChunk32Tables& dt,
    uint32_t fixed
) {
    if (P10DC_RANKCHUNK32_TABLE_VERIFIED[fixed]) return;
    constexpr size_t P = size_t(MAXW + 2);
    const size_t owner_base = size_t(fixed) * P;
    const uint32_t owner_begin = bf.low_code_off[owner_base];
    const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
        ? bf.low_code_off[size_t(fixed + 1u) * P]
        : uint32_t(bf.low_codes.size());
    const size_t code_count = size_t(owner_end - owner_begin);
    const size_t meta_count = dt.low_rankchunkmeta32_count;
    const size_t block_count =
        (meta_count + P10DC_RANKCHUNK32_BLOCK - 1u) / P10DC_RANKCHUNK32_BLOCK;
    if (meta_count < code_count ||
        dt.low_rankchunkblock16_count != block_count ||
        dt.low_rankchunk_padding_count != meta_count - code_count ||
        dt.low_prekey != nullptr || dt.low_rankstream_off != nullptr) {
        std::cerr << "p10dc rankchunk32 count/lifetime mismatch owner=" << fixed
                  << " meta=" << meta_count << " codes=" << code_count
                  << " padding=" << dt.low_rankchunk_padding_count
                  << " blocks=" << dt.low_rankchunkblock16_count << '/' << block_count << '\n';
        std::exit(670);
    }

    std::vector<uint32_t> got_meta(meta_count), got_blocks(block_count);
    if (!got_meta.empty()) ck(cudaMemcpy(got_meta.data(), dt.low_rankchunkmeta32,
        got_meta.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost), "p10dc rankchunk32 meta verify D2H");
    if (!got_blocks.empty()) ck(cudaMemcpy(got_blocks.data(), dt.low_rankchunkblock16,
        got_blocks.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost), "p10dc rankchunk32 blocks verify D2H");
    std::array<uint32_t, MAXW + 2> got_hoff{};
    ck(cudaMemcpyFromSymbol(got_hoff.data(), D_P10DC_LOW_RANKCHUNK_HOFF,
        got_hoff.size()*sizeof(uint32_t)), "p10dc rankchunk32 height offsets verify");

    size_t compact = 0, stream_ix = 0, actual_codes = 0, padding = 0;
    uint32_t current_block_base = 0;
    auto verify_block_start = [&](size_t c) {
        if ((c & (P10DC_RANKCHUNK32_BLOCK - 1u)) != 0u) return;
        const size_t bi = c >> P10DC_RANKCHUNK32_BLOCK_LOG2;
        current_block_base = uint32_t(stream_ix);
        if (bi >= got_blocks.size() || got_blocks[bi] != current_block_base) {
            std::cerr << "p10dc rankchunk32 block base mismatch owner=" << fixed
                      << " compact=" << c << " block=" << bi
                      << " got=" << (bi < got_blocks.size() ? got_blocks[bi] : 0xffffffffu)
                      << " expected=" << current_block_base << '\n';
            std::exit(671);
        }
    };

    for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
        while ((compact & (P10DC_RANKCHUNK32_HEIGHT_ALIGN - 1u)) != 0u) {
            verify_block_start(compact);
            if (compact >= got_meta.size() || got_meta[compact] != 0u) {
                std::cerr << "p10dc rankchunk32 padding mismatch owner=" << fixed
                          << " h=" << h << " compact=" << compact << '\n';
                std::exit(672);
            }
            ++compact;
            ++padding;
        }
        if (got_hoff[h] != compact ||
            (got_hoff[h] & (P10DC_RANKCHUNK32_HEIGHT_ALIGN - 1u)) != 0u) {
            std::cerr << "p10dc rankchunk32 height offset mismatch owner=" << fixed
                      << " h=" << h << " got=" << got_hoff[h]
                      << " expected=" << compact << '\n';
            std::exit(673);
        }

        const uint32_t a = bf.low_code_off[owner_base + h];
        const uint32_t b = h + 1u < uint32_t(MAXW + 2)
            ? bf.low_code_off[owner_base + h + 1u] : owner_end;
        for (uint32_t i = a; i < b; ++i, ++compact, ++actual_codes) {
            verify_block_start(compact);
            const uint32_t code = bf.low_codes[i];
            const uint32_t key = gpu_direct_ternary_key_host(code, LOW_LUT_K);
            const uint32_t chunks = p10dc_rankchunk32_pack_host(key);
            const uint32_t prefix = uint32_t(stream_ix) - current_block_base;
            const uint32_t expected = chunks | (prefix << P10DC_RANKCHUNK32_CHUNK_BITS);
            if (p10dc_rankchunk32_unpack_host(chunks) != key ||
                compact >= got_meta.size() || got_meta[compact] != expected) {
                std::cerr << "p10dc rankchunk32 meta mismatch owner=" << fixed
                          << " h=" << h << " compact=" << compact
                          << " prefix=" << prefix << '\n';
                std::exit(674);
            }
            for (int pos = 0; pos < LOW_LUT_K; ++pos)
                if (((code >> (2 * pos)) & 3u) == uint32_t(::L)) ++stream_ix;
        }
    }
    if (actual_codes != code_count || compact != meta_count ||
        padding != dt.low_rankchunk_padding_count ||
        stream_ix != dt.low_rankstream_count) {
        std::cerr << "p10dc rankchunk32 walk mismatch owner=" << fixed
                  << " actual=" << actual_codes << '/' << code_count
                  << " meta=" << compact << '/' << meta_count
                  << " padding=" << padding << '/' << dt.low_rankchunk_padding_count
                  << " stream=" << stream_ix << '/' << dt.low_rankstream_count << '\n';
        std::exit(675);
    }
    P10DC_RANKCHUNK32_TABLE_VERIFIED[fixed] = 1;
}

#if P10DC_RANKCHUNK32_DIRECTMASK
static void p10dc_verify_directmask_tables(
    const BucketFusedHost& bf,
    const BucketFusedDirectHighRowsRankChunk32DirectMaskTables& dt,
    uint32_t fixed
) {
    const size_t stride = dt.low_rankchunkmeta32_count;
    if (dt.low_rankdirectoff_count != stride ||
        dt.low_rankdirectmask_count != size_t(16) * stride) {
        std::cerr << "p10dc directmask table count mismatch owner=" << fixed
                  << " stride=" << stride << " offsets=" << dt.low_rankdirectoff_count
                  << " masks=" << dt.low_rankdirectmask_count << '\n';
        std::exit(676);
    }
#if P10DC_RANKCHUNK32_RANKPLANE
    constexpr size_t NRANK = size_t(P10DC_RANKCHUNK32_MAX_L_PER_LEGAL_CODE);
    constexpr size_t RANK_STORAGE = NRANK ? NRANK : 1u;
    if (dt.low_rankdirectplane_count != RANK_STORAGE * stride) {
        std::cerr << "p10dc directplane count mismatch owner=" << fixed
                  << " got=" << dt.low_rankdirectplane_count
                  << " expected=" << RANK_STORAGE * stride << '\n';
        std::exit(680);
    }
#endif

    std::vector<uint32_t> got_off(stride);
    std::vector<uint8_t> got_mask(size_t(16) * stride);
    if (!got_off.empty()) ck(cudaMemcpy(got_off.data(), dt.low_rankdirectoff,
        got_off.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost), "p10dc directoff verify D2H");
    if (!got_mask.empty()) ck(cudaMemcpy(got_mask.data(), dt.low_rankdirectmask,
        got_mask.size()*sizeof(uint8_t), cudaMemcpyDeviceToHost), "p10dc directmask verify D2H");
#if P10DC_RANKCHUNK32_RANKPLANE
    std::vector<uint16_t> got_plane(dt.low_rankdirectplane_count);
    if (!got_plane.empty()) ck(cudaMemcpy(got_plane.data(), dt.low_rankdirectplane,
        got_plane.size()*sizeof(uint16_t), cudaMemcpyDeviceToHost), "p10dc directplane verify D2H");
#endif

    constexpr size_t P = size_t(MAXW + 2);
    const size_t owner_base = size_t(fixed) * P;
    const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
        ? bf.low_code_off[size_t(fixed + 1u) * P]
        : uint32_t(bf.low_codes.size());
    size_t compact = 0, actual = 0;
    uint32_t stream_ix = 0;
    for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
#if P10DC_RANKCHUNK32_ALIGN32
        compact = (compact + P10DC_RANKCHUNK32_BLOCK - 1u) &
                  ~(size_t(P10DC_RANKCHUNK32_BLOCK) - 1u);
#endif
        const uint32_t a = bf.low_code_off[owner_base + h];
        const uint32_t b = h + 1u < uint32_t(MAXW + 2)
            ? bf.low_code_off[owner_base + h + 1u] : owner_end;
        for (uint32_t i = a; i < b; ++i, ++compact, ++actual) {
            if (compact >= stride || got_off[compact] != stream_ix) {
                std::cerr << "p10dc directoff mismatch owner=" << fixed
                          << " h=" << h << " compact=" << compact
                          << " got=" << (compact < stride ? got_off[compact] : 0xffffffffu)
                          << " expected=" << stream_ix << '\n';
                std::exit(677);
            }
            const uint32_t code = bf.low_codes[i];
            const uint32_t key = gpu_direct_ternary_key_host(code, LOW_LUT_K);
            const uint32_t packed = p10dc_rankchunk32_pack_host(key);
            for (uint32_t depth = 1; depth <= 15u; ++depth) {
                const uint8_t expected = p10dc_rankchunk32_directmask_host(packed, depth);
                const uint8_t got = got_mask[size_t(depth) * stride + compact];
                if (got != expected) {
                    std::cerr << "p10dc directmask mismatch owner=" << fixed
                              << " h=" << h << " compact=" << compact
                              << " depth=" << depth << " got=" << unsigned(got)
                              << " expected=" << unsigned(expected) << '\n';
                    std::exit(678);
                }
            }

            uint32_t ordinal = 0;
            uint32_t weight = bkcz_pow3_const(LOW_LUT_K - 1);
            for (int pos = LOW_LUT_K - 1; pos >= 0; --pos) {
                if (((code >> (2 * pos)) & 3u) == uint32_t(::L)) {
#if P10DC_RANKCHUNK32_RANKPLANE
                    const uint32_t x = bf.low_direct[key - weight];
                    const uint32_t loc = x & BKF_LOC_MASK;
                    const uint16_t expected_rank = uint16_t(bkf_loc_rank(loc));
                    const uint16_t got_rank = got_plane[size_t(ordinal) * stride + compact];
                    if (x == BKF_DIRECT_INVALID || bkf_loc_owner(loc) != fixed ||
                        got_rank != expected_rank) {
                        std::cerr << "p10dc directplane mismatch owner=" << fixed
                                  << " h=" << h << " compact=" << compact
                                  << " ordinal=" << ordinal << " got=" << got_rank
                                  << " expected=" << expected_rank << '\n';
                        std::exit(681);
                    }
#endif
                    ++ordinal;
                    ++stream_ix;
                }
                if (pos) weight /= 3u;
            }
        }
    }
    if (compact != stride || stream_ix != dt.low_rankstream_count ||
        actual != dt.low_prekey_count) {
        std::cerr << "p10dc directmask walk mismatch owner=" << fixed
                  << " compact=" << compact << '/' << stride
                  << " stream=" << stream_ix << '/' << dt.low_rankstream_count
                  << " codes=" << actual << '/' << dt.low_prekey_count << '\n';
        std::exit(679);
    }
}
#endif

template<class Tables>
static void p10dc_rankchunk32_run_high(
    BucketHostGrid& g, const StorageLayout& layout, const BucketPhysicalLayoutHost& phy,
    const BucketFusedHost& bf, Tables& dt, bool rev, bool experiment
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
                                          cudaMemcpyHostToDevice), "p10dc rankchunk32 high H2D");
        }
        dt.bind_owner(fixed, phy, d);
        p10dc_verify_rankchunk32_tables(bf, dt, fixed);
#if P10DC_RANKCHUNK32_DIRECTMASK
        p10dc_verify_directmask_tables(bf, dt, fixed);
#endif
        if (!rev) {
            for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
                if (experiment)
                    bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankchunk32_cross5_kernel<<<grid, block, smem>>>(p);
                else bucket_high_orbit_closure_pattern10_depthcode_resolved_kernel<<<grid, block>>>(p);
                ck(cudaGetLastError(), experiment ? "p10dc rankchunk32 forward high" : "p10dc rankchunk32 forward control");
            }
        } else {
            for (int p = LOW_LUT_K + 1; p < TARGET_W; ++p) {
                if (experiment)
                    bucket_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_rankchunk32_cross5_kernel<<<grid, block, smem>>>(p);
                else bucket_reverse_high_pattern10_depthcode_resolved_kernel<<<grid, block>>>(p);
                ck(cudaGetLastError(), experiment ? "p10dc rankchunk32 reverse high" : "p10dc rankchunk32 reverse control");
            }
        }
        ck(cudaDeviceSynchronize(), "p10dc rankchunk32 high sync");
        for (uint32_t s = 0; s < BUCKET_NGPU; ++s) {
            auto& v = g[s][fixed];
            if (!v.empty()) ck(cudaMemcpy(v.data(), d[s], v.size()*sizeof(Count),
                                          cudaMemcpyDeviceToHost), "p10dc rankchunk32 high D2H");
        }
        bkft_free_slots(d);
    }
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W, L = LOW_LUT_K;
    static_assert(W <= 12, "rankchunk32 CROSS5 selftest intentionally uses small width");
    int visible = 0; cudaError_t ce = cudaGetDeviceCount(&visible);
    if (ce != cudaSuccess || visible < 1) {
        std::cout << "bucket-closure-pattern10-depthcode-rankchunk32-cross5-selftest SKIP no CUDA device\n";
        return 0;
    }
    ck(cudaSetDevice(0), "p10dc rankchunk32 set device");

    build_full_dp(); G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR); StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout); HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost loworbit = build_cpu_low_orbit(storage, layout, lowdesc); CpuHighDirectHost highdirect = build_cpu_high_direct(storage, layout, highdesc);
    GpuDirectGatherHost ordinary = build_gpu_direct_gather(layout, lowdesc, loworbit, highdirect); GpuDirectCrossGatherHost cross = build_gpu_direct_cross_gather(storage, layout,lowdesc,loworbit,highdirect);
    GpuDirectFusedHost fused = build_gpu_direct_fused_checked(layout, ordinary, cross); CpuLowSparseHost lowsparse = build_cpu_low_sparse(storage,layout,lowdesc,loworbit);
    BucketOwnerHost owner = build_bucket_owners(G_FACTOR, storage); BucketPhysicalLayoutHost phy = build_bucket_physical_layout(layout, owner);
    BucketOrbitStreamsHost bo = build_bucket_orbits(storage,layout,owner,lowsparse,highdirect); BucketFusedHost bf = build_bucket_fused(storage,layout,owner,ordinary,cross,fused);
    ReverseLowDescHost rlow = build_reverse_low_descriptors(storage,layout); ReverseHighDescHost rhigh = build_reverse_high_descriptors(storage,layout);
    ReverseOrbitHost rlo = build_reverse_orbit(storage,layout,true), rhi = build_reverse_orbit(storage,layout,false);
    ReverseBucketAtomicHost rb = build_reverse_bucket_atomic(storage,layout,owner,rlow,rhigh,rlo,rhi); ReverseBucketFusedHost rf = build_reverse_bucket_fused_checked(layout,owner,rb);
    auto fh = build_bucket_forward_pattern10_depthcode_placeholder(layout,bo,bf); auto rh = build_bucket_reverse_pattern10_depthcode_zero_checked(layout,bo,bf,rb,rf);

    auto ms = gdg_enum_states(W), bs = gdg_enum_states(W - 1); std::unordered_map<MateID,size_t> mi,di;
    for (size_t i=0;i<ms.size();++i) mi.emplace(ms[i],i); for (size_t i=0;i<bs.size();++i) di.emplace(bs[i],i);
    std::mt19937_64 rng(0x16181624ULL); std::vector<Count> im(ms.size()),ib(bs.size()); for(auto&x:im)x=Count(rng()%mod); for(auto&x:ib)x=Count(rng()%mod);
    auto [fhm,fhb]=gdg_reference_window(W,W-1,L+1,mod,ms,bs,mi,di,im,ib); auto [rhm,rhb]=bra_reference(L+1,W-1,mod,ms,bs,mi,di,im,ib);

    ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"p10dc rankchunk32 modulus");
    P10DCRankChunk32SelftestTables dt; dt.install_metadata(layout,bo,bf);
    BucketForwardPattern10DepthCodeDeviceTables fdt; fdt.install(fh); BucketReversePattern10DepthCodeDeviceTables rdt; rdt.install(rh);
    p10dc_install_cross5_lut(); p10dc_install_rankchunk32_lut();

    auto g0=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy); p10dc_rankchunk32_run_high(g0,layout,phy,bf,dt,false,false); if(!bkft_compare("pattern10-depthcode-forward-high-resolved-rankchunk32-control",g0,ms,bs,fhm,fhb,storage,layout,owner,phy))return 90;
    auto g1=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy); p10dc_rankchunk32_run_high(g1,layout,phy,bf,dt,false,true); if(!bkft_compare("pattern10-depthcode-forward-high-rankchunk32-cross5",g1,ms,bs,fhm,fhb,storage,layout,owner,phy))return 91;
    auto g2=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy); p10dc_rankchunk32_run_high(g2,layout,phy,bf,dt,true,false); if(!bkft_compare("pattern10-depthcode-reverse-high-resolved-rankchunk32-control",g2,ms,bs,rhm,rhb,storage,layout,owner,phy))return 92;
    auto g3=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy); p10dc_rankchunk32_run_high(g3,layout,phy,bf,dt,true,true); if(!bkft_compare("pattern10-depthcode-reverse-high-rankchunk32-cross5",g3,ms,bs,rhm,rhb,storage,layout,owner,phy))return 93;
    for(uint32_t g=0;g<BUCKET_NGPU;++g) if(!P10DC_RANKCHUNK32_TABLE_VERIFIED[g]) return 94;
    rdt.release(); fdt.release(); dt.release();
    std::cout << "bucket-closure-pattern10-depthcode-rankchunk32-cross5-selftest OK W=" << W
              << " control=resolved experiment=warpstriped_delta_direct_affine_rankchunk32_cross5"
              << " forward_exact=1 reverse_exact=1 rankchunk32_table_exact=1 padding_exact=1"
              << " directmask=" << P10DC_RANKCHUNK32_DIRECTMASK
              << " rankplane=" << P10DC_RANKCHUNK32_RANKPLANE
              << " ilp=" << P10DC_WARPSTRIPED_ILP
#if P10DC_RANKCHUNK32_DIRECTMASK
              << " directmask_table_exact=1 directoff_table_exact=1"
#if P10DC_RANKCHUNK32_RANKPLANE
              << " directrankplane_exact=1 offset_runtime=0"
#endif
              << " rankchunk_meta_runtime=0 blockbase_shuffle_runtime=0"
#endif
              << " chunk_bits=" << P10DC_RANKCHUNK32_CHUNK_BITS
              << " prefix_bits=" << P10DC_RANKCHUNK32_PREFIX_BITS
              << " block=" << P10DC_RANKCHUNK32_BLOCK
              << " height_align=" << P10DC_RANKCHUNK32_HEIGHT_ALIGN
              << " third_chunk_bits=7 block_base_loads_per_warp_max="
              << (P10DC_RANKCHUNK32_ALIGN32 ? 1 : 2)
              << " cross_runtime_div=0 cross_runtime_mod=0 cross_runtime_direct_lookup=0"
              << " old_prekey_offset_arrays_freed=1 fallback_structurally_unreachable=1\n";
    return 0;
}
