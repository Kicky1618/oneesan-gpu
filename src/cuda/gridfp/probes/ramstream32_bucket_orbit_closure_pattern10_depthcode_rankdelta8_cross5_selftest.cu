#pragma push_macro("main")
#undef main
#define main pattern10_depthcode_rankdelta8_reference_main_unused
#include "ramstream32_bucket_reverse_fused_selftest.cu"
#pragma pop_macro("main")

#include "../ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankdelta8_cross5.cuh"

static std::array<uint8_t, BUCKET_NGPU> P10DC_RANKDELTA8_TABLE_VERIFIED{};

static void p10dc_verify_rankdelta8_tables(
    const BucketFusedHost& bf, const BucketFusedDirectHighRowsRankDelta8Tables& dt,
    uint32_t fixed
) {
    if (P10DC_RANKDELTA8_TABLE_VERIFIED[fixed]) return;
    constexpr size_t P = size_t(MAXW + 2);
    const size_t owner_base = size_t(fixed) * P;
    const uint32_t owner_begin = bf.low_code_off[owner_base];
    const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
        ? bf.low_code_off[size_t(fixed + 1u) * P]
        : uint32_t(bf.low_codes.size());
    const size_t code_count = size_t(owner_end - owner_begin);
    const size_t meta_count = dt.low_rankdelta8_meta32_count;
    const size_t block_count = (meta_count + 31u) >> 5;
    if (meta_count < code_count || dt.low_rankdelta8_block32_count != block_count ||
        dt.low_rankdelta8_padding_count != meta_count - code_count || dt.low_prekey != nullptr) {
        std::cerr << "p10dc rankdelta8 count/lifetime mismatch owner=" << fixed
                  << " meta=" << meta_count << " codes=" << code_count
                  << " padding=" << dt.low_rankdelta8_padding_count
                  << " blocks=" << dt.low_rankdelta8_block32_count << '/' << block_count << '\n';
        std::exit(710);
    }

    std::vector<uint32_t> got_meta(meta_count), got_blocks(block_count);
    std::vector<uint8_t> got_stream(dt.low_rankdelta8_stream_count);
    if (!got_meta.empty()) ck(cudaMemcpy(got_meta.data(), dt.low_rankdelta8_meta32,
        got_meta.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost), "p10dc rankdelta8 meta verify D2H");
    if (!got_blocks.empty()) ck(cudaMemcpy(got_blocks.data(), dt.low_rankdelta8_block32,
        got_blocks.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost), "p10dc rankdelta8 blocks verify D2H");
    if (!got_stream.empty()) ck(cudaMemcpy(got_stream.data(), dt.low_rankdelta8_stream,
        got_stream.size(), cudaMemcpyDeviceToHost), "p10dc rankdelta8 stream verify D2H");
    std::array<uint32_t, MAXW + 2> got_hoff{};
    ck(cudaMemcpyFromSymbol(got_hoff.data(), D_P10DC_LOW_RANKDELTA8_HOFF,
        got_hoff.size()*sizeof(uint32_t)), "p10dc rankdelta8 hoff verify");

    size_t compact = 0, stream_ix = 0, actual_codes = 0, padding = 0;
    uint32_t current_block_base = 0;
    auto verify_block_start = [&](size_t c) {
        if ((c & 31u) != 0u) return;
        const size_t bi = c >> 5;
        current_block_base = uint32_t(stream_ix);
        if (bi >= got_blocks.size() || got_blocks[bi] != current_block_base) {
            std::cerr << "p10dc rankdelta8 block mismatch owner=" << fixed
                      << " compact=" << c << " block=" << bi << '\n';
            std::exit(711);
        }
    };

    for (uint32_t h = 0; h < uint32_t(MAXW + 2); ++h) {
#if P10DC_RANKDELTA8_ALIGN32
        while (compact & 31u) {
            verify_block_start(compact);
            if (compact >= got_meta.size() || got_meta[compact] != 0u) std::exit(712);
            ++compact; ++padding;
        }
#endif
        if (got_hoff[h] != compact ||
            (P10DC_RANKDELTA8_ALIGN32 && (got_hoff[h] & 31u))) {
            std::cerr << "p10dc rankdelta8 hoff mismatch owner=" << fixed
                      << " h=" << h << " got=" << got_hoff[h]
                      << " expected=" << compact << '\n';
            std::exit(713);
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
            const uint32_t expected_meta = chunks | (prefix << P10DC_RANKDELTA8_CHUNK_BITS);
            if ((chunks >> 23) != 0u || prefix >= 512u ||
                compact >= got_meta.size() || got_meta[compact] != expected_meta) {
                std::cerr << "p10dc rankdelta8 meta mismatch owner=" << fixed
                          << " h=" << h << " compact=" << compact
                          << " prefix=" << prefix << '\n';
                std::exit(714);
            }

            std::array<uint16_t, LOW_LUT_K> ranks{};
            uint32_t nr = 0, weight = bkcz_pow3_const(LOW_LUT_K - 1);
            for (int pos = LOW_LUT_K - 1; pos >= 0; --pos) {
                if (((code >> (2 * pos)) & 3u) == uint32_t(::L)) {
                    const uint32_t x = bf.low_direct[key - weight];
                    if (x == BKF_DIRECT_INVALID) std::exit(715);
                    const uint32_t loc = x & BKF_LOC_MASK;
                    if (bkf_loc_owner(loc) != fixed) std::exit(716);
                    ranks[nr++] = uint16_t(bkf_loc_rank(loc));
                }
                if (pos) weight /= 3u;
            }
            std::vector<uint8_t> expected;
            if (nr) {
                expected.push_back(uint8_t(ranks[0]));
                expected.push_back(uint8_t(ranks[0] >> 8));
                for (uint32_t j = 1; j < nr; ++j) {
                    if (ranks[j] <= ranks[j-1]) std::exit(717);
                    const uint32_t d = uint32_t(ranks[j] - ranks[j-1]);
                    if (d > 0xffffu) std::exit(718);
                    if (d <= 255u) expected.push_back(uint8_t(d));
                    else { expected.push_back(0u); expected.push_back(uint8_t(d)); expected.push_back(uint8_t(d >> 8)); }
                }
            }
            if (stream_ix + expected.size() > got_stream.size() ||
                !std::equal(expected.begin(), expected.end(), got_stream.begin() + stream_ix)) {
                std::cerr << "p10dc rankdelta8 stream mismatch owner=" << fixed
                          << " h=" << h << " compact=" << compact
                          << " stream_ix=" << stream_ix << '\n';
                std::exit(719);
            }
            stream_ix += expected.size();
        }
    }
    if (actual_codes != code_count || compact != meta_count || padding != dt.low_rankdelta8_padding_count ||
        stream_ix != dt.low_rankdelta8_stream_count) {
        std::cerr << "p10dc rankdelta8 walk mismatch owner=" << fixed
                  << " actual=" << actual_codes << '/' << code_count
                  << " meta=" << compact << '/' << meta_count
                  << " stream=" << stream_ix << '/' << dt.low_rankdelta8_stream_count << '\n';
        std::exit(720);
    }
    P10DC_RANKDELTA8_TABLE_VERIFIED[fixed] = 1;
}

static void p10dc_rankdelta8_run_high(
    BucketHostGrid& g, const StorageLayout& layout, const BucketPhysicalLayoutHost& phy,
    const BucketFusedHost& bf, BucketFusedDirectHighRowsRankDelta8Tables& dt,
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
            if (!v.empty()) ck(cudaMemcpy(d[s], v.data(), v.size()*sizeof(Count), cudaMemcpyHostToDevice), "p10dc rankdelta8 H2D");
        }
        dt.bind_owner(fixed, phy, d);
        p10dc_verify_rankdelta8_tables(bf, dt, fixed);
        if (!rev) {
            for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
                if (experiment) bucket_high_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankdelta8_cross5_kernel<<<grid,block,smem>>>(p);
                else bucket_high_orbit_closure_pattern10_depthcode_resolved_kernel<<<grid,block>>>(p);
                ck(cudaGetLastError(), experiment ? "p10dc rankdelta8 forward" : "p10dc rankdelta8 forward control");
            }
        } else {
            for (int p = LOW_LUT_K + 1; p < TARGET_W; ++p) {
                if (experiment) bucket_reverse_high_pattern10_depthcode_warpstriped_delta_direct_affine_rankdelta8_cross5_kernel<<<grid,block,smem>>>(p);
                else bucket_reverse_high_pattern10_depthcode_resolved_kernel<<<grid,block>>>(p);
                ck(cudaGetLastError(), experiment ? "p10dc rankdelta8 reverse" : "p10dc rankdelta8 reverse control");
            }
        }
        ck(cudaDeviceSynchronize(), "p10dc rankdelta8 sync");
        for (uint32_t s = 0; s < BUCKET_NGPU; ++s) {
            auto& v = g[s][fixed];
            if (!v.empty()) ck(cudaMemcpy(v.data(), d[s], v.size()*sizeof(Count), cudaMemcpyDeviceToHost), "p10dc rankdelta8 D2H");
        }
        bkft_free_slots(d);
    }
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W, L = LOW_LUT_K;
    static_assert(W <= 12, "rankdelta8 CROSS5 selftest intentionally uses small width");
    int visible = 0; cudaError_t ce = cudaGetDeviceCount(&visible);
    if (ce != cudaSuccess || visible < 1) {
        std::cout << "bucket-closure-pattern10-depthcode-rankdelta8-cross5-selftest SKIP no CUDA device\n";
        return 0;
    }
    ck(cudaSetDevice(0), "p10dc rankdelta8 set device");

    build_full_dp(); G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR); StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout); HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost loworbit = build_cpu_low_orbit(storage, layout, lowdesc); CpuHighDirectHost highdirect = build_cpu_high_direct(storage, layout, highdesc);
    GpuDirectGatherHost ordinary = build_gpu_direct_gather(layout, lowdesc, loworbit, highdirect); GpuDirectCrossGatherHost cross = build_gpu_direct_cross_gather(storage, layout, lowdesc, loworbit, highdirect);
    GpuDirectFusedHost fused = build_gpu_direct_fused_checked(layout, ordinary, cross); CpuLowSparseHost lowsparse = build_cpu_low_sparse(storage, layout, lowdesc, loworbit);
    BucketOwnerHost owner = build_bucket_owners(G_FACTOR, storage); BucketPhysicalLayoutHost phy = build_bucket_physical_layout(layout, owner);
    BucketOrbitStreamsHost bo = build_bucket_orbits(storage, layout, owner, lowsparse, highdirect); BucketFusedHost bf = build_bucket_fused(storage, layout, owner, ordinary, cross, fused);
    ReverseLowDescHost rlow = build_reverse_low_descriptors(storage, layout); ReverseHighDescHost rhigh = build_reverse_high_descriptors(storage, layout);
    ReverseOrbitHost rlo = build_reverse_orbit(storage, layout, true), rhi = build_reverse_orbit(storage, layout, false);
    ReverseBucketAtomicHost rb = build_reverse_bucket_atomic(storage, layout, owner, rlow, rhigh, rlo, rhi); ReverseBucketFusedHost rf = build_reverse_bucket_fused_checked(layout, owner, rb);
    auto fh = build_bucket_forward_pattern10_depthcode_placeholder(layout, bo, bf); auto rh = build_bucket_reverse_pattern10_depthcode_zero_checked(layout, bo,bf,rb,rf);

    auto ms = gdg_enum_states(W), bs = gdg_enum_states(W - 1); std::unordered_map<MateID,size_t> mi,di;
    for(size_t i=0;i<ms.size();++i)mi.emplace(ms[i],i); for(size_t i=0;i<bs.size();++i)di.emplace(bs[i],i);
    std::mt19937_64 rng(0x16181624ULL); std::vector<Count> im(ms.size()),ib(bs.size()); for(auto&x:im)x=Count(rng()%mod); for(auto&x:ib)x=Count(rng()%mod);
    auto [fhm,fhb]=gdg_reference_window(W,W-1,L+1,mod,ms,bs,mi,di,im,ib); auto [rhm,rhb]=bra_reference(L+1,W-1,mod,ms,bs,mi,di,im,ib);

    ck(cudaMemcpyToSymbol(D_MOD,&mod,sizeof(mod)),"p10dc rankdelta8 modulus");
    BucketFusedDirectHighRowsRankDelta8Tables dt; dt.install_metadata(layout,bo,bf);
    BucketForwardPattern10DepthCodeDeviceTables fdt; fdt.install(fh); BucketReversePattern10DepthCodeDeviceTables rdt; rdt.install(rh);
    p10dc_install_rankchunk32_lut();

    auto g0=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy); p10dc_rankdelta8_run_high(g0,layout,phy,bf,dt,false,false); if(!bkft_compare("rankdelta8-forward-control",g0,ms,bs,fhm,fhb,storage,layout,owner,phy))return 90;
    auto g1=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy); p10dc_rankdelta8_run_high(g1,layout,phy,bf,dt,false,true); if(!bkft_compare("rankdelta8-forward",g1,ms,bs,fhm,fhb,storage,layout,owner,phy))return 91;
    auto g2=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy); p10dc_rankdelta8_run_high(g2,layout,phy,bf,dt,true,false); if(!bkft_compare("rankdelta8-reverse-control",g2,ms,bs,rhm,rhb,storage,layout,owner,phy))return 92;
    auto g3=bkft_make_grid(ms,bs,im,ib,storage,layout,owner,phy); p10dc_rankdelta8_run_high(g3,layout,phy,bf,dt,true,true); if(!bkft_compare("rankdelta8-reverse",g3,ms,bs,rhm,rhb,storage,layout,owner,phy))return 93;
    for(uint32_t g=0;g<BUCKET_NGPU;++g) if(!P10DC_RANKDELTA8_TABLE_VERIFIED[g]) return 94;
    rdt.release(); fdt.release(); dt.release();
    std::cout << "bucket-closure-pattern10-depthcode-rankdelta8-cross5-selftest OK W=" << W
              << " forward_exact=1 reverse_exact=1 table_exact=1 stream_exact=1"
              << " chunk_bits=23 prefix_bits=9 block=32"
              << " height_align=" << (P10DC_RANKDELTA8_ALIGN32 ? 32 : 1)
              << " delta_fast8_escape16=1 fused16=" << P10DC_RANKCHUNK32_FUSED16
              << " cross_runtime_div=0 cross_runtime_mod=0\n";
    return 0;
}
