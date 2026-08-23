#include <cuda_runtime.h>

#include <algorithm>
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
static bool orbit_rep(MateValuePair w) { return w == NN || w == NR || w == NL; }

int main(int argc, char** argv) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint64_t LOW_CODE_MASK = (uint64_t(1) << (2 * L)) - 1u;
    constexpr uint64_t HIGH_CODE_MASK = (uint64_t(1) << (2 * H)) - 1u;

    const int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    if (n + 1 != TARGET_W || L + H != TARGET_W - 1) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    HighDescHost hd = build_high_descriptors(storage, layout);
    LowDescHost ld = build_low_descriptors(storage, layout);
    const auto& oa = MaskShardDeviceMeta::orbit_aux_host(layout);

    const size_t want_h = size_t(hd.main_total) * H;
    const size_t want_l = size_t(ld.main_total) * L;
    if (oa.high_aux.size() != want_h || oa.low_aux.size() != want_l) {
        std::cerr << "orbit aux shape mismatch high=" << oa.high_aux.size() << '/' << want_h
                  << " low=" << oa.low_aux.size() << '/' << want_l << '\n';
        return 2;
    }

    uint64_t h_nn = 0, h_pair = 0, l_nn = 0, l_pair = 0, l_p1_pair = 0;

    for (int p = TARGET_W - 1; p >= L + 1; --p) {
        const uint32_t pi = uint32_t((TARGET_W - 1) - p);
        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            const StorageBlock& sb = layout.main_blocks[bid];
            if (!sb.valid || !sb.rows || !sb.cols) continue;
            const uint32_t lc = storage.low_all_codes[storage.low_all_off[sb.hs]];
            for (uint32_t hr = 0; hr < sb.rows; ++hr) {
                const uint32_t hc = storage.high_all_codes[storage.high_all_off[sb.he] + hr];
                const MateID m = MateID(lc)
                    | (MateID(sb.c) << (2 * L))
                    | (MateID(hc) << (2 * (L + 1)));
                const MateValuePair w = mpair(m, p);
                const size_t ix = size_t(pi) * hd.main_total + hd.main_base[bid] + hr;
                const uint32_t a = oa.high_aux[ix];
                const uint32_t k = aux_kind(a);
                if (!orbit_rep(w)) {
                    if (k != MS_ORBIT_AUX_INVALID) return 9;
                    continue;
                }

                const IncludeResult z = oneesan::gridfp::include_horizontal(m, TARGET_W, p);
                if (!z.valid) return 10;
                if (w == NN) {
                    ++h_nn;
                    if (k != MS_ORBIT_AUX_NN || hkind(hd.main_desc[ix]) != HIGHDESC_MAIN)
                        return 11;
                    const MateID dropped = mshrink(m, p);
                    const uint32_t dhc = uint32_t((dropped >> (2 * L)) & HIGH_CODE_MASK);
                    const int dh = seg_end_height_host(dhc, H);
                    const uint32_t packed = storage.high_packed_rank[dhc];
                    if (packed == 0xffffffffu || aux_block(a) != uint32_t(dh)
                        || aux_rank(a) != (packed >> H)) return 12;
                } else {
                    ++h_pair;
                    if (k != MS_ORBIT_AUX_PAIR || hkind(hd.main_desc[ix]) != HIGHDESC_BLOCK)
                        return 13;
                    const MateValuePair cw = w == NR ? RN : LN;
                    const MateID companion = msetpair(m, p, cw);
                    const uint32_t hc2 = uint32_t((companion >> (2 * (L + 1))) & HIGH_CODE_MASK);
                    const int he2 = seg_end_height_host(hc2, H);
                    const int cv2 = int(mget(companion, L));
                    const uint32_t packed = storage.high_packed_rank[hc2];
                    if (packed == 0xffffffffu
                        || aux_block(a) != uint32_t(3 * he2 + cv2)
                        || aux_rank(a) != (packed >> H)) return 14;
                    if (!z.blocked || z.mate != mshrink(m, p)) return 15;
                }
            }
        }
    }

    for (int p = L; p >= 1; --p) {
        const uint32_t pi = uint32_t(L - p);
        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            const StorageBlock& sb = layout.main_blocks[bid];
            if (!sb.valid || !sb.rows || !sb.cols) continue;
            const uint32_t hc = storage.high_all_codes[storage.high_all_off[sb.he]];
            for (uint32_t lr = 0; lr < sb.cols; ++lr) {
                const uint32_t lc = storage.low_all_codes[storage.low_all_off[sb.hs] + lr];
                const MateID m = MateID(lc)
                    | (MateID(sb.c) << (2 * L))
                    | (MateID(hc) << (2 * (L + 1)));
                const MateValuePair w = mpair(m, p);
                const size_t ix = size_t(pi) * ld.main_total + ld.main_base[bid] + lr;
                const uint32_t a = oa.low_aux[ix];
                const uint32_t k = aux_kind(a);
                if (!orbit_rep(w)) {
                    if (k != MS_ORBIT_AUX_INVALID) return 19;
                    continue;
                }

                const IncludeResult z = oneesan::gridfp::include_horizontal(m, TARGET_W, p);
                if (!z.valid) return 20;
                if (w == NN) {
                    ++l_nn;
                    if (k != MS_ORBIT_AUX_NN || lkind(ld.main_desc[ix]) != LOWDESC_MAIN)
                        return 21;
                    const MateID dropped = mshrink(m, p);
                    const uint32_t dlc = uint32_t(dropped & LOW_CODE_MASK);
                    const uint32_t packed = storage.low_packed_rank[dlc];
                    if (packed == 0xffffffffu || aux_block(a) != uint32_t(sb.he)
                        || aux_rank(a) != (packed >> L)) return 22;
                } else if (p == 1) {
                    ++l_pair;
                    ++l_p1_pair;
                    if (k != MS_ORBIT_AUX_PAIR || lkind(ld.main_desc[ix]) != LOWDESC_MAIN)
                        return 23;
                    const MateID dropped = mshrink(m, p);
                    const uint32_t dlc = uint32_t(dropped & LOW_CODE_MASK);
                    const uint32_t packed = storage.low_packed_rank[dlc];
                    if (packed == 0xffffffffu || aux_block(a) != uint32_t(sb.he)
                        || aux_rank(a) != (packed >> L)) return 24;
                    const MateValuePair cw = w == NR ? RN : LN;
                    if (z.blocked || z.mate != msetpair(m, p, cw)) return 25;
                } else {
                    ++l_pair;
                    if (k != MS_ORBIT_AUX_PAIR || lkind(ld.main_desc[ix]) != LOWDESC_BLOCK)
                        return 26;
                    const MateValuePair cw = w == NR ? RN : LN;
                    const MateID companion = msetpair(m, p, cw);
                    const uint32_t lc2 = uint32_t(companion & LOW_CODE_MASK);
                    const int cv2 = int(mget(companion, L));
                    const uint32_t packed = storage.low_packed_rank[lc2];
                    if (packed == 0xffffffffu
                        || aux_block(a) != uint32_t(3 * int(sb.he) + cv2)
                        || aux_rank(a) != (packed >> L)) return 27;
                    if (!z.blocked || z.mate != mshrink(m, p)) return 28;
                }

                const uint32_t zhc = z.blocked
                    ? uint32_t((z.mate >> (2 * L)) & HIGH_CODE_MASK)
                    : uint32_t((z.mate >> (2 * (L + 1))) & HIGH_CODE_MASK);
                if (zhc != hc) return 29;
            }
        }
    }

    std::cout << "maskshard-orbitaux-hostplan OK n=" << n
              << " high_aux_mib=" << double(oa.high_aux.size() * sizeof(uint32_t)) / double(1ULL << 20)
              << " low_aux_mib=" << double(oa.low_aux.size() * sizeof(uint32_t)) / double(1ULL << 20)
              << " high_nn=" << h_nn << " high_pair=" << h_pair
              << " low_nn=" << l_nn << " low_pair=" << l_pair
              << " low_p1_pair=" << l_p1_pair
              << " masks=" << (1u << H) << '/' << (1u << L) << '\n';
    return 0;
}
