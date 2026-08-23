#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define main oneesan_factorized_hbm_unused_main
#include "../oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu"
#undef main

#include "../../gridfp/ramstream32_factorized_storage.hpp"
#define MASKSHARD_HIGH_CLOSURE_ROWS 1
#include "../../gridfp/ramstream32_highdesc.cuh"

int main(int argc, char** argv) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint64_t HC = (uint64_t(1) << (2 * H)) - 1u;
    const int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    if (n + 1 != TARGET_W || L + H != TARGET_W - 1) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    HighDescHost hd = build_high_descriptors(storage, layout);

    if (hd.closure_off[0] != 0 || hd.closure_off[H] != hd.closure_rows.size()) {
        std::cerr << "high closure row offset endpoints mismatch\n";
        return 2;
    }
    if (hd.closure_block_off.size() != size_t(H) * 65) return 3;

    uint64_t total = 0;
    for (int p = TARGET_W - 1; p >= L + 1; --p) {
        const uint32_t pi = uint32_t((TARGET_W - 1) - p);
        const uint32_t begin = hd.closure_off[pi];
        const uint32_t end = hd.closure_off[pi + 1];
        if (begin > end || end > hd.closure_rows.size()) return 4;
        std::vector<uint8_t> seen(hd.main_total, 0);

        if (hd.closure_block_off[size_t(pi) * 65] != begin
            || hd.closure_block_off[size_t(pi) * 65 + layout.main_blocks.size()] != end)
            return 5;

        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            const uint32_t a = hd.closure_block_off[size_t(pi) * 65 + bid];
            const uint32_t z = hd.closure_block_off[size_t(pi) * 65 + bid + 1];
            if (a > z || a < begin || z > end) return 6;
            for (uint32_t q = a; q < z; ++q) {
                if (highdesc_block(hd.closure_rows[q]) != bid) return 7;
            }
        }

        for (uint32_t q = begin; q < end; ++q) {
            const uint32_t src = hd.closure_rows[q];
            if (highdesc_kind(src) != HIGHDESC_MAIN) return 8;
            const uint32_t bid = highdesc_block(src);
            const uint32_t hr = highdesc_rank(src);
            if (bid >= layout.main_blocks.size()) return 9;
            const StorageBlock& sb = layout.main_blocks[bid];
            if (!sb.valid || !sb.rows || !sb.cols || hr >= sb.rows) return 10;
            const uint32_t ix = hd.main_base[bid] + hr;
            if (++seen[ix] != 1) return 11;

            const uint32_t lc = storage.low_all_codes[storage.low_all_off[sb.hs]];
            const uint32_t hc = storage.high_all_codes[storage.high_all_off[sb.he] + hr];
            const MateID m = MateID(lc)
                | (MateID(sb.c) << (2 * L))
                | (MateID(hc & HC) << (2 * (L + 1)));
            const MateValuePair w = mpair(m, p);
            if (w != LL && w != RR && w != RL) return 12;
            const uint32_t desc = hd.main_desc[size_t(pi) * hd.main_total + ix];
            const uint32_t kind = highdesc_kind(desc);
            if (kind != HIGHDESC_BLOCK && kind != HIGHDESC_CROSS) return 13;
        }

        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            const StorageBlock& sb = layout.main_blocks[bid];
            if (!sb.valid || !sb.rows || !sb.cols) continue;
            const uint32_t lc = storage.low_all_codes[storage.low_all_off[sb.hs]];
            for (uint32_t hr = 0; hr < sb.rows; ++hr) {
                const uint32_t hc = storage.high_all_codes[storage.high_all_off[sb.he] + hr];
                const MateID m = MateID(lc)
                    | (MateID(sb.c) << (2 * L))
                    | (MateID(hc & HC) << (2 * (L + 1)));
                const MateValuePair w = mpair(m, p);
                const uint32_t ix = hd.main_base[bid] + hr;
                const uint32_t desc = hd.main_desc[size_t(pi) * hd.main_total + ix];
                const uint32_t kind = highdesc_kind(desc);
                const bool expected = (w == LL || w == RR || w == RL)
                    && (kind == HIGHDESC_BLOCK || kind == HIGHDESC_CROSS);
                if (bool(seen[ix]) != expected) {
                    std::cerr << "closure row membership mismatch p=" << p
                              << " bid=" << bid << " hr=" << hr
                              << " pair=" << uint32_t(w) << " kind=" << kind << '\n';
                    return 14;
                }
            }
        }
        total += end - begin;
        std::cout << "p=" << p << " closure_rows=" << (end - begin) << '\n';
    }

    std::cout << "maskshard-highclosure-rows-hostplan OK n=" << n
              << " rows=" << total
              << " rows_mib=" << double(total * sizeof(uint32_t)) / double(1ULL << 20)
              << " block_off_kib="
              << double(hd.closure_block_off.size() * sizeof(uint32_t)) / 1024.0
              << '\n';
    return 0;
}
