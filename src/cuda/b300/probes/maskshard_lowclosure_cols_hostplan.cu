#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define main oneesan_factorized_hbm_unused_main
#include "../oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu"
#undef main

#include "../../gridfp/ramstream32_factorized_storage.hpp"
#include "../../gridfp/ramstream32_lowdesc.cuh"
#define MASKSHARD_LOW_CLOSURE_COLS 1
#include "../maskshard_lowclosure.cuh"

static uint32_t host_lowdesc_kind(uint32_t x) {
    return x >> LOWDESC_KIND_SHIFT;
}
static uint32_t host_lowdesc_block(uint32_t x) {
    return (x >> LOWDESC_BLOCK_SHIFT) & LOWDESC_BLOCK_MASK;
}
static uint32_t host_lowdesc_lr(uint32_t x) {
    return x & LOWDESC_LR_MASK;
}

int main(int argc, char** argv) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    const int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    if (n + 1 != TARGET_W || L + H != TARGET_W - 1) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost ld = build_low_descriptors(storage, layout);
    MaskShardLowClosureColsHost cc =
        build_maskshard_low_closure_cols(storage, layout, ld);

    if (cc.off[0] != 0 || cc.off[L] != cc.cols.size()) return 2;
    if (cc.block_off.size() != size_t(L) * 65) return 3;

    uint64_t total = 0;
    uint32_t expected_per_p = 0;
    for (int p = L; p >= 1; --p) {
        const uint32_t pi = uint32_t(L - p);
        const uint32_t begin = cc.off[pi];
        const uint32_t end = cc.off[pi + 1];
        if (begin > end || end > cc.cols.size()) return 4;
        if (cc.block_off[size_t(pi) * 65] != begin
            || cc.block_off[size_t(pi) * 65 + layout.main_blocks.size()] != end)
            return 5;

        std::vector<uint8_t> seen(ld.main_total, 0);
        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            const uint32_t a = cc.block_off[size_t(pi) * 65 + bid];
            const uint32_t z = cc.block_off[size_t(pi) * 65 + bid + 1];
            if (a > z || a < begin || z > end) return 6;
            for (uint32_t q = a; q < z; ++q) {
                const uint32_t src = cc.cols[q];
                if (host_lowdesc_kind(src) != LOWDESC_MAIN
                    || host_lowdesc_block(src) != bid) return 7;
                const uint32_t lr = host_lowdesc_lr(src);
                if (lr >= layout.main_blocks[bid].cols) return 8;
                const uint32_t ix = ld.main_base[bid] + lr;
                if (++seen[ix] != 1) return 9;
            }
        }

        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            const StorageBlock& sb = layout.main_blocks[bid];
            if (!sb.valid || !sb.rows || !sb.cols) continue;
            const uint32_t low0 = storage.low_all_off[sb.hs];
            for (uint32_t lr = 0; lr < sb.cols; ++lr) {
                const uint32_t lc = storage.low_all_codes[low0 + lr];
                const uint32_t active = lc | (uint32_t(sb.c) << (2 * L));
                const MateValuePair w = MateValuePair(
                    (active >> (2 * (p - 1))) & 15u);
                const uint32_t ix = ld.main_base[bid] + lr;
                const uint32_t desc = ld.main_desc[size_t(pi) * ld.main_total + ix];
                const bool expected = (w == LL || w == RR || w == RL)
                    && host_lowdesc_kind(desc) != LOWDESC_INVALID;
                if (bool(seen[ix]) != expected) {
                    std::cerr << "LOW closure column membership mismatch p=" << p
                              << " bid=" << bid << " lr=" << lr
                              << " pair=" << uint32_t(w)
                              << " kind=" << host_lowdesc_kind(desc) << '\n';
                    return 10;
                }
            }
        }

        const uint32_t count = end - begin;
        if (!expected_per_p) expected_per_p = count;
        else if (count != expected_per_p) return 11;
        total += count;
        std::cout << "p=" << p << " closure_cols=" << count << '\n';
    }

    std::cout << "maskshard-lowclosure-cols-hostplan OK n=" << n
              << " cols_per_p=" << expected_per_p
              << " entries=" << total
              << " cols_mib=" << double(total * sizeof(uint32_t)) / double(1ULL << 20)
              << " block_off_kib="
              << double(cc.block_off.size() * sizeof(uint32_t)) / 1024.0
              << '\n';
    return 0;
}
