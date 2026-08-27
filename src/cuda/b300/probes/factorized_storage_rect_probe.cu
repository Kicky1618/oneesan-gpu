// Host-side validation of the storage-major authoritative layout proposed for
// the B300 factorized backend.  This deliberately includes the production
// factorized implementation so that the probe uses exactly the same FBlock
// ordering and factor tables as the CUDA solver.

#ifndef TARGET_W
#define TARGET_W 28
#endif
#ifndef LOW_LUT_K
#define LOW_LUT_K 14
#endif
#ifndef HIGH_LUT_K
#define HIGH_LUT_K 13
#endif

#define main oneesan_factorized_batch_unused_main
#include "../oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu"
#undef main

#include "../../gridfp/ramstream32_factorized_storage.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

struct StorageRect {
    Code global = 0;
    Code local = 0;
    Code rows = 0;
    Code global_stride = 0;
    Code width = 0;
    // Number of LOW entries in one logical factorized row.  For fixed HIGH
    // occupancy the transfer rectangle is flattened to rows=1, so this field
    // is needed to recover (HIGH-local-rank, LOW-all-rank) in the validator.
    Code factor_stride = 0;
    uint8_t he = 0;
    uint8_t hs = 0;
    uint8_t c = 0;
};

static std::vector<StorageRect> storage_rects(
    const std::vector<FBlock>& local_blocks,
    const std::vector<StorageBlock>& storage_blocks,
    bool fix_low,
    uint32_t mask,
    const StorageFactorHost& storage
) {
    constexpr int S = StorageFactorHost::S;
    if (local_blocks.size() != storage_blocks.size()) {
        std::cerr << "block-count mismatch local=" << local_blocks.size()
                  << " storage=" << storage_blocks.size() << "\n";
        std::exit(70);
    }

    std::vector<StorageRect> out;
    out.reserve(local_blocks.size());
    for (size_t i = 0; i < local_blocks.size(); ++i) {
        const FBlock& fb = local_blocks[i];
        const StorageBlock& sb = storage_blocks[i];
        if (fb.end == fb.off) continue;
        if (!sb.valid) {
            std::cerr << "non-empty FBlock maps to invalid StorageBlock i=" << i << "\n";
            std::exit(71);
        }

        StorageRect r;
        r.local = fb.off;
        r.factor_stride = fb.stride;
        r.he = fb.he;
        r.hs = fb.hs;
        r.c = fb.c;
        if (!fix_low) {
            // Fixed HIGH occupancy: selected HIGH rows are contiguous and LOW is
            // completely free, so the entire FBlock is one contiguous run.
            uint32_t row0 = storage.high_mask_begin[size_t(mask) * S + fb.he];
            r.global = sb.off + Code(row0) * sb.cols;
            r.rows = 1;
            r.global_stride = fb.end - fb.off;
            r.width = fb.end - fb.off;
        } else {
            // Fixed LOW occupancy: every HIGH row contributes the same contiguous
            // LOW slice.  Keep this as one strided rectangle rather than expanding
            // it into one PeerInterval per row.
            uint32_t col0 = storage.low_mask_begin[size_t(mask) * S + fb.hs];
            r.width = fb.stride;
            r.rows = r.width ? (fb.end - fb.off) / r.width : 0;
            r.global = sb.off + col0;
            r.global_stride = sb.cols;
        }
        if (!r.rows || !r.width || !r.factor_stride
            || r.rows * r.width != fb.end - fb.off) {
            std::cerr << "invalid rectangle geometry i=" << i
                      << " rows=" << r.rows << " width=" << r.width
                      << " factor_stride=" << r.factor_stride
                      << " block=" << (fb.end - fb.off) << "\n";
            std::exit(72);
        }
        out.push_back(r);
    }
    return out;
}

static void logical_factor_coordinate(
    const StorageRect& r,
    Code transfer_row,
    Code transfer_col,
    bool fix_low,
    Code& factor_row,
    Code& factor_col
) {
    if (fix_low) {
        factor_row = transfer_row;
        factor_col = transfer_col;
        return;
    }
    Code flat = transfer_row * r.width + transfer_col;
    factor_row = flat / r.factor_stride;
    factor_col = flat % r.factor_stride;
}

static MateID main_mate_from_rect(
    const StorageRect& r,
    Code transfer_row,
    Code transfer_col,
    bool fix_low,
    uint32_t mask,
    const StorageFactorHost& storage
) {
    constexpr int L = LOW_LUT_K;
    constexpr int S = StorageFactorHost::S;

    Code row = 0, col = 0;
    logical_factor_coordinate(r, transfer_row, transfer_col, fix_low, row, col);

    uint32_t hc = 0, lc = 0;
    if (fix_low) {
        hc = storage.high_all_codes[storage.high_all_off[r.he] + row];
        uint32_t a = G_FACTOR.low_mask_off[size_t(mask) * S + r.hs];
        lc = G_FACTOR.low_mask_codes[a + col];
    } else {
        uint32_t a = G_FACTOR.high_mask_off[size_t(mask) * S + r.he];
        hc = G_FACTOR.high_mask_codes[a + row];
        lc = storage.low_all_codes[storage.low_all_off[r.hs] + col];
    }
    return MateID(lc)
        | (MateID(MateValue(r.c)) << (2 * L))
        | (MateID(hc) << (2 * (L + 1)));
}

static MateID block_mate_from_rect(
    const StorageRect& r,
    Code transfer_row,
    Code transfer_col,
    bool fix_low,
    uint32_t mask,
    const StorageFactorHost& storage
) {
    constexpr int L = LOW_LUT_K;
    constexpr int S = StorageFactorHost::S;

    Code row = 0, col = 0;
    logical_factor_coordinate(r, transfer_row, transfer_col, fix_low, row, col);

    uint32_t hc = 0, lc = 0;
    if (fix_low) {
        hc = storage.high_all_codes[storage.high_all_off[r.he] + row];
        uint32_t a = G_FACTOR.low_mask_off[size_t(mask) * S + r.hs];
        lc = G_FACTOR.low_mask_codes[a + col];
    } else {
        uint32_t a = G_FACTOR.high_mask_off[size_t(mask) * S + r.he];
        hc = G_FACTOR.high_mask_codes[a + row];
        lc = storage.low_all_codes[storage.low_all_off[r.hs] + col];
    }
    return MateID(lc) | (MateID(hc) << (2 * L));
}

static std::vector<Code> sample_axis(Code n) {
    std::vector<Code> v;
    if (!n) return v;
    v.push_back(0);
    if (n > 2) v.push_back(n / 2);
    if (n > 1) v.push_back(n - 1);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}

static void validate_rects(
    const std::vector<StorageRect>& rects,
    Code expected_size,
    bool main,
    bool fix_low,
    uint32_t mask,
    const StorageFactorHost& storage,
    const StorageLayout& layout
) {
    Code covered = 0;
    Code expected_local = 0;
    for (const StorageRect& r : rects) {
        if (r.local != expected_local) {
            std::cerr << "local gap: got=" << r.local
                      << " expected=" << expected_local << "\n";
            std::exit(73);
        }
        Code n = r.rows * r.width;
        expected_local += n;
        covered += n;

        for (Code row : sample_axis(r.rows)) {
            for (Code col : sample_axis(r.width)) {
                Code local = r.local + row * r.width + col;
                Code global = r.global + row * r.global_stride + col;
                MateID m = main
                    ? main_mate_from_rect(r, row, col, fix_low, mask, storage)
                    : block_mate_from_rect(r, row, col, fix_low, mask, storage);
                Code want = main
                    ? storage_rank_main_host(m, storage, layout)
                    : storage_rank_block_host(m, storage, layout);
                if (global != want) {
                    std::cerr << "storage rank mismatch main=" << main
                              << " fix_low=" << fix_low << " mask=" << mask
                              << " local=" << local << " global=" << global
                              << " want=" << want << " he=" << int(r.he)
                              << " hs=" << int(r.hs) << " c=" << int(r.c)
                              << " transfer_row=" << row
                              << " transfer_col=" << col << "\n";
                    std::exit(74);
                }
            }
        }
    }
    if (covered != expected_size || expected_local != expected_size) {
        std::cerr << "coverage mismatch covered=" << covered
                  << " expected=" << expected_size << "\n";
        std::exit(75);
    }
}

static std::vector<uint32_t> mask_set(int bits, bool full) {
    std::set<uint32_t> s;
    const uint32_t all = (1u << bits) - 1u;
    s.insert(0);
    s.insert(all);
    s.insert(all & 0x55555555u);
    s.insert(all & 0xaaaaaaaau);
    for (int b = 0; b < bits; ++b) s.insert(1u << b);
    if (full) {
        for (uint32_t m = 0; m <= all; ++m) s.insert(m);
    } else {
        std::mt19937 rng(0x51a7e5u + bits);
        for (int i = 0; i < 128; ++i) s.insert(rng() & all);
    }
    return {s.begin(), s.end()};
}

static void validate_mode(
    bool main,
    bool fix_low,
    bool full,
    const StorageFactorHost& storage,
    const StorageLayout& layout
) {
    const int bits = fix_low ? LOW_LUT_K : HIGH_LUT_K;
    auto masks = mask_set(bits, full);
    uint64_t groups = 0;
    uint64_t rect_count = 0;
    long double elems = 0;

    for (uint32_t mask : masks) {
        auto fb = main
            ? make_factor_main_blocks(fix_low, mask)
            : make_factor_block_blocks(fix_low, mask);
        auto rects = storage_rects(
            fb,
            main ? layout.main_blocks : layout.block_blocks,
            fix_low,
            mask,
            storage);
        Code expected = fb.empty() ? 0 : fb.back().end;
        validate_rects(rects, expected, main, fix_low, mask, storage, layout);
        ++groups;
        rect_count += rects.size();
        elems += expected;
    }

    std::cout << "mode=" << (main ? "main" : "blocked")
              << "/" << (fix_low ? "fix_low" : "fix_high")
              << " masks=" << groups
              << " rects=" << rect_count
              << " avg_rects=" << (groups ? double(rect_count) / groups : 0.0)
              << " elems=" << static_cast<double>(elems)
              << " ok\n";
}

int main(int argc, char** argv) {
    bool full = argc > 1 && std::string(argv[1]) == "--full";

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);

    std::cout << "factorized_storage_rect_probe"
              << " width=" << TARGET_W
              << " low=" << LOW_LUT_K
              << " high=" << HIGH_LUT_K
              << " main_states=" << layout.main_size
              << " blocked_states=" << layout.block_size
              << " mode=" << (full ? "full" : "quick") << "\n";

    validate_mode(true,  true,  full, storage, layout);
    validate_mode(true,  false, full, storage, layout);
    validate_mode(false, true,  full, storage, layout);
    validate_mode(false, false, full, storage, layout);

    std::cout << "PASS factorized storage rectangles\n";
    return 0;
}
