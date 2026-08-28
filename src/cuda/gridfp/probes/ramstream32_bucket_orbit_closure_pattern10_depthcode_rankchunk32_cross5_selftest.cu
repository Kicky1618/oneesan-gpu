#pragma push_macro("main")
#undef main
#define main pattern10_depthcode_rankchunk32_reference_main_unused
#include "ramstream32_bucket_reverse_fused_selftest.cu"
#pragma pop_macro("main")

#include "../ramstream32_bucket_orbit_closure_pattern10_depthcode_warpstriped_delta_direct_affine_rankchunk32_cross5.cuh"

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
        got_hoff.size()*sizeof(uint32_t)), "p10dc rankchunk32 aligned height offsets verify");

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
        std::cerr << "p10dc rankchunk32 padded walk mismatch owner=" << fixed
                  << " actual=" << actual_codes << '/' << code_count
                  << " meta=" << compact << '/' << meta_count
                  << " padding=" << padding << '/' << dt.low_rankchunk_padding_count
                  << " stream=" << stream_ix << '/' << dt.low_rankstream_count << '\n';
        std::exit(675);
    }
    P10DC_RANKCHUNK32_TABLE_VERIFIED[fixed] = 1;
}

static void p10dc_rankchunk32_run_high(
    BucketHostGrid& g, const StorageLayout& layout, const BucketPhysicalLayoutHost& phy,
    const BucketFusedHost& bf, BucketFusedDirectHighRowsRankChunk32Tables& dt,
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
                                          cudaMemcpyHostToDevice), "p10dc rankchunk32 high H2D");
        }
        dt.bind_owner(fixed, phy, d);
        p10dc_verify_rankchunk32_tables(bf, dt, fixed);
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
    LowDescHost lowdesc = build_low_descriptors(storage, layout); HighDescHost highdesc = build_high_descriptors(storage, layout, highdesc);
