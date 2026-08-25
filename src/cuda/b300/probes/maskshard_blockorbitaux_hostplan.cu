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
#define MASKSHARD_BLOCK_ORBIT_AUX 1
#include "../maskshard_highio.cuh"

static uint32_t aux_kind_host(uint32_t x) {
    return x >> MS_ORBIT_AUX_KIND_SHIFT;
}
static uint32_t aux_block_host(uint32_t x) {
    return (x >> MS_ORBIT_AUX_BLOCK_SHIFT) & MS_ORBIT_AUX_BLOCK_MASK;
}
static uint32_t aux_rank_host(uint32_t x) {
    return x & MS_ORBIT_AUX_RANK_MASK;
}

int main(int argc, char** argv) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint64_t LC = (uint64_t(1) << (2 * L)) - 1u;
    constexpr uint64_t HC = (uint64_t(1) << (2 * H)) - 1u;

    const int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    if (n + 1 != TARGET_W || L + H != TARGET_W - 1) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    HighDescHost hd = build_high_descriptors(storage, layout);
    LowDescHost ld = build_low_descriptors(storage, layout);
    const auto& oa = MaskShardDeviceMeta::orbit_aux_host(layout);

    const size_t want_h = size_t(hd.block_total) * H;
    const size_t want_l = size_t(ld.block_total) * L;
    if (oa.high_aux.size() != want_h || oa.low_aux.size() != want_l) {
        std::cerr << "compact orbit aux shape mismatch high=" << oa.high_aux.size()
                  << '/' << want_h << " low=" << oa.low_aux.size() << '/' << want_l << '\n';
        return 2;
    }

    uint64_t high_nn = 0, high_pair = 0;
    uint64_t low_nn = 0, low_pair = 0, low_p1_pair = 0;

    for (int p = TARGET_W - 1; p >= L + 1; --p) {
        const uint32_t pi = uint32_t((TARGET_W - 1) - p);
        for (size_t dbid = 0; dbid < layout.block_blocks.size(); ++dbid) {
            const StorageBlock& db = layout.block_blocks[dbid];
            if (!db.valid || !db.rows || !db.cols) continue;
            const uint32_t lc = storage.low_all_codes[storage.low_all_off[db.hs]];
            for (uint32_t dhr = 0; dhr < db.rows; ++dhr) {
                const size_t bdi = size_t(pi) * hd.block_total + hd.block_base[dbid] + dhr;
                const uint32_t bdesc = hd.block_desc[bdi];
                if (highdesc_kind(bdesc) != HIGHDESC_MAIN) return 10;

                const uint32_t hc = storage.high_all_codes[storage.high_all_off[db.he] + dhr];
                const MateID blocked = MateID(lc) | (MateID(hc) << (2 * L));
                const MateID rep = oneesan::gridfp::blocked_exclude(blocked, p);
                const MateValuePair w = mpair(rep, p);
                if (w != NN && w != NR && w != NL) return 11;

                const uint32_t rhc = uint32_t((rep >> (2 * (L + 1))) & HC);
                const int rhe = seg_end_height_host(rhc, H);
                const uint32_t rcv = uint32_t(mget(rep, L));
                const uint32_t rp = storage.high_packed_rank[rhc];
                if (rp == 0xffffffffu
                    || highdesc_block(bdesc) != uint32_t(3 * rhe + int(rcv))
                    || highdesc_rank(bdesc) != (rp >> H)) return 12;

                const uint32_t aux = oa.high_aux[bdi];
                if (w == NN) {
                    ++high_nn;
                    if (aux_kind_host(aux) != MS_ORBIT_AUX_NN) return 13;
                    const size_t sdi = size_t(pi) * hd.main_total
                        + hd.main_base[highdesc_block(bdesc)] + highdesc_rank(bdesc);
                    if (highdesc_kind(hd.main_desc[sdi]) != HIGHDESC_MAIN) return 14;
                } else {
                    ++high_pair;
                    if (aux_kind_host(aux) != MS_ORBIT_AUX_PAIR) return 15;
                    const MateValuePair cw = w == NR ? RN : LN;
                    const MateID companion = msetpair(rep, p, cw);
                    const uint32_t chc = uint32_t((companion >> (2 * (L + 1))) & HC);
                    const int che = seg_end_height_host(chc, H);
                    const uint32_t ccv = uint32_t(mget(companion, L));
                    const uint32_t cp = storage.high_packed_rank[chc];
                    if (cp == 0xffffffffu
                        || aux_block_host(aux) != uint32_t(3 * che + int(ccv))
                        || aux_rank_host(aux) != (cp >> H)) return 16;
                    const size_t sdi = size_t(pi) * hd.main_total
                        + hd.main_base[highdesc_block(bdesc)] + highdesc_rank(bdesc);
                    if (highdesc_kind(hd.main_desc[sdi]) != HIGHDESC_BLOCK
                        || highdesc_block(hd.main_desc[sdi]) != dbid
                        || highdesc_rank(hd.main_desc[sdi]) != dhr) return 17;
                }
            }
        }
    }

    for (int p = L; p >= 1; --p) {
        const uint32_t pi = uint32_t(L - p);
        for (size_t dbid = 0; dbid < layout.block_blocks.size(); ++dbid) {
            const StorageBlock& db = layout.block_blocks[dbid];
            if (!db.valid || !db.rows || !db.cols) continue;
            const uint32_t hc = storage.high_all_codes[storage.high_all_off[db.he]];
            for (uint32_t dlr = 0; dlr < db.cols; ++dlr) {
                const size_t bdi = size_t(pi) * ld.block_total + ld.block_base[dbid] + dlr;
                const uint32_t bdesc = ld.block_desc[bdi];
                if (lowdesc_kind(bdesc) != LOWDESC_MAIN) return 20;

                const uint32_t lc = storage.low_all_codes[storage.low_all_off[db.hs] + dlr];
                const MateID blocked = MateID(lc) | (MateID(hc) << (2 * L));
                const MateID rep = oneesan::gridfp::blocked_exclude(blocked, p);
                const MateValuePair w = mpair(rep, p);
                if (w != NN && w != NR && w != NL) return 21;

                const uint32_t rlc = uint32_t(rep & LC);
                const uint32_t rcv = uint32_t(mget(rep, L));
                const uint32_t rp = storage.low_packed_rank[rlc];
                if (rp == 0xffffffffu
                    || lowdesc_block(bdesc) != uint32_t(3 * int(db.he) + int(rcv))
                    || lowdesc_lr(bdesc) != (rp >> L)) return 22;

                const uint32_t aux = oa.low_aux[bdi];
                const size_t sdi = size_t(pi) * ld.main_total
                    + ld.main_base[lowdesc_block(bdesc)] + lowdesc_lr(bdesc);
                if (w == NN) {
                    ++low_nn;
                    if (aux_kind_host(aux) != MS_ORBIT_AUX_NN
                        || lowdesc_kind(ld.main_desc[sdi]) != LOWDESC_MAIN) return 23;
                } else if (p == 1) {
                    ++low_pair;
                    ++low_p1_pair;
                    if (aux_kind_host(aux) != MS_ORBIT_AUX_PAIR
                        || lowdesc_kind(ld.main_desc[sdi]) != LOWDESC_MAIN) return 24;
                    const MateValuePair cw = w == NR ? RN : LN;
                    const MateID companion = msetpair(rep, p, cw);
                    const uint32_t clc = uint32_t(companion & LC);
                    const uint32_t cp = storage.low_packed_rank[clc];
                    if (cp == 0xffffffffu
                        || lowdesc_block(ld.main_desc[sdi]) != uint32_t(3 * int(db.he) + int(mget(companion, L)))
                        || lowdesc_lr(ld.main_desc[sdi]) != (cp >> L)) return 25;
                } else {
                    ++low_pair;
                    if (aux_kind_host(aux) != MS_ORBIT_AUX_PAIR) return 26;
                    const MateValuePair cw = w == NR ? RN : LN;
                    const MateID companion = msetpair(rep, p, cw);
                    const uint32_t clc = uint32_t(companion & LC);
                    const uint32_t ccv = uint32_t(mget(companion, L));
                    const uint32_t cp = storage.low_packed_rank[clc];
                    if (cp == 0xffffffffu
                        || aux_block_host(aux) != uint32_t(3 * int(db.he) + int(ccv))
                        || aux_rank_host(aux) != (cp >> L)) return 27;
                    if (lowdesc_kind(ld.main_desc[sdi]) != LOWDESC_BLOCK
                        || lowdesc_block(ld.main_desc[sdi]) != dbid
                        || lowdesc_lr(ld.main_desc[sdi]) != dlr) return 28;
                }
            }
        }
    }

    std::cout << "maskshard-blockorbitaux-hostplan OK n=" << n
              << " high_aux_mib="
              << double(oa.high_aux.size() * sizeof(uint32_t)) / double(1ULL << 20)
              << " low_aux_mib="
              << double(oa.low_aux.size() * sizeof(uint32_t)) / double(1ULL << 20)
              << " high_nn=" << high_nn << " high_pair=" << high_pair
              << " low_nn=" << low_nn << " low_pair=" << low_pair
              << " low_p1_pair=" << low_p1_pair << '\n';
    return 0;
}
