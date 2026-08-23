#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

#define main oneesan_factorized_hbm_unused_main
#include "../oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu"
#undef main

#include "../../gridfp/ramstream32_factorized_storage.hpp"
#include "../../gridfp/ramstream32_highdesc.cuh"
#include "../../gridfp/ramstream32_lowdesc.cuh"
#include "../maskshard_layout.hpp"
#define MASKSHARD_ORBIT_AUX 1
#include "../maskshard_highio.cuh"

static uint32_t aux_kind(uint32_t x) { return x >> MS_ORBIT_AUX_KIND_SHIFT; }
static uint32_t aux_block(uint32_t x) {
    return (x >> MS_ORBIT_AUX_BLOCK_SHIFT) & MS_ORBIT_AUX_BLOCK_MASK;
}
static uint32_t aux_rank(uint32_t x) { return x & MS_ORBIT_AUX_RANK_MASK; }
static uint32_t hkind(uint32_t x) { return x >> HIGHDESC_KIND_SHIFT; }
static uint32_t lkind(uint32_t x) { return x >> LOWDESC_KIND_SHIFT; }

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    if (n + 1 != TARGET_W || LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    HighDescHost hd = build_high_descriptors(storage, layout);
    LowDescHost ld = build_low_descriptors(storage, layout);
    const auto& oa = MaskShardDeviceMeta::orbit_aux_host(layout);

    const size_t want_h = size_t(hd.main_total) * HIGH_LUT_K;
    const size_t want_l = size_t(ld.main_total) * LOW_LUT_K;
    if (oa.high_aux.size() != want_h || oa.low_aux.size() != want_l) {
        std::cerr << "orbit aux shape mismatch high=" << oa.high_aux.size() << '/' << want_h
                  << " low=" << oa.low_aux.size() << '/' << want_l << '\n';
        return 2;
    }

    uint64_t h_nn = 0, h_pair = 0, l_nn = 0, l_pair = 0, l_p1_pair = 0;
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        const uint32_t pi = uint32_t((TARGET_W - 1) - p);
        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            const StorageBlock& sb = layout.main_blocks[bid];
            for (uint32_t hr = 0; hr < sb.rows; ++hr) {
                const size_t ix = size_t(pi) * hd.main_total + hd.main_base[bid] + hr;
                const uint32_t a = oa.high_aux[ix];
                const uint32_t k = aux_kind(a);
                if (k == MS_ORBIT_AUX_INVALID) continue;
                if (k == MS_ORBIT_AUX_NN) {
                    ++h_nn;
                    if (hkind(hd.main_desc[ix]) != HIGHDESC_MAIN) return 10;
                    const uint32_t b = aux_block(a), r = aux_rank(a);
                    if (b >= layout.block_blocks.size() || r >= layout.block_blocks[b].rows) return 11;
                } else if (k == MS_ORBIT_AUX_PAIR) {
                    ++h_pair;
                    if (hkind(hd.main_desc[ix]) != HIGHDESC_BLOCK) return 12;
                    const uint32_t b = aux_block(a), r = aux_rank(a);
                    if (b >= layout.main_blocks.size() || r >= layout.main_blocks[b].rows) return 13;
                } else return 14;
            }
        }
    }

    for (int p = LOW_LUT_K; p >= 1; --p) {
        const uint32_t pi = uint32_t(LOW_LUT_K - p);
        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            const StorageBlock& sb = layout.main_blocks[bid];
            for (uint32_t lr = 0; lr < sb.cols; ++lr) {
                const size_t ix = size_t(pi) * ld.main_total + ld.main_base[bid] + lr;
                const uint32_t a = oa.low_aux[ix];
                const uint32_t k = aux_kind(a);
                if (k == MS_ORBIT_AUX_INVALID) continue;
                if (k == MS_ORBIT_AUX_NN) {
                    ++l_nn;
                    if (lkind(ld.main_desc[ix]) != LOWDESC_MAIN) return 20;
                    const uint32_t b = aux_block(a), r = aux_rank(a);
                    if (b >= layout.block_blocks.size() || r >= layout.block_blocks[b].cols) return 21;
                } else if (k == MS_ORBIT_AUX_PAIR) {
                    ++l_pair;
                    const uint32_t b = aux_block(a), r = aux_rank(a);
                    if (p == 1) {
                        ++l_p1_pair;
                        if (lkind(ld.main_desc[ix]) != LOWDESC_MAIN) return 22;
                        if (b >= layout.block_blocks.size() || r >= layout.block_blocks[b].cols) return 23;
                    } else {
                        if (lkind(ld.main_desc[ix]) != LOWDESC_BLOCK) return 24;
                        if (b >= layout.main_blocks.size() || r >= layout.main_blocks[b].cols) return 25;
                    }
                } else return 26;
            }
        }
    }

    std::cout << "maskshard-orbitaux-hostplan OK n=" << n
              << " high_aux_mib=" << double(oa.high_aux.size() * sizeof(uint32_t)) / double(1ULL << 20)
              << " low_aux_mib=" << double(oa.low_aux.size() * sizeof(uint32_t)) / double(1ULL << 20)
              << " high_nn=" << h_nn << " high_pair=" << h_pair
              << " low_nn=" << l_nn << " low_pair=" << l_pair
              << " low_p1_pair=" << l_p1_pair << '\n';
    return 0;
}
