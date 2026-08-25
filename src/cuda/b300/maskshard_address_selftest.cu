#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define main oneesan_factorized_hbm_unused_main
#include "oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu"
#undef main

#include "../gridfp/ramstream32_factorized_storage.hpp"
#include "maskshard_layout.hpp"

struct HostAddr {
    int owner = -1;
    Code off = 0;
};

static MateID unrank_full_host(Code rank, int width) {
    MateID m = 0;
    int h = 1;
    for (int pos = width - 1; pos >= 0; --pos) {
        Code z = H_DP[pos][h];
        if (rank < z) continue;
        rank -= z;
        if (h > 0) {
            z = H_DP[pos][h - 1];
            if (rank < z) {
                m |= MateID(R) << (2 * pos);
                --h;
                continue;
            }
            rank -= z;
        }
        m |= MateID(::L) << (2 * pos);
        ++h;
    }
    if (rank != 0 || h != 0) {
        std::cerr << "unrank_full_host residual rank=" << rank << " h=" << h << '\n';
        std::exit(150);
    }
    return m;
}

static Code rank_full_host(MateID m, int width) {
    Code rank = 0;
    int h = 1;
    for (int pos = width - 1; pos >= 0; --pos) {
        const MateValue v = mget(m, pos);
        if (v > N) rank += H_DP[pos][h];
        if (v > R && h > 0) rank += H_DP[pos][h - 1];
        if (v == R) --h;
        else if (v == ::L) ++h;
    }
    return rank;
}

static HostAddr addr_main(
    MateID m, const StorageFactorHost& storage,
    const StorageLayout& layout, const MaskShardLayout& shard
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LC = (1u << (2 * L)) - 1u;
    constexpr uint32_t HC = (1u << (2 * H)) - 1u;
    constexpr uint32_t HM = (1u << H) - 1u;
    const uint32_t lc = uint32_t(m) & LC;
    const uint32_t hc = uint32_t((m >> (2 * (L + 1))) & HC);
    const int he = seg_end_height_host(hc, H);
    const int cv = int(mget(m, L));
    const uint32_t bid = uint32_t(3 * he + cv);
    const uint32_t hp = storage.high_packed_rank[hc];
    const uint32_t lp = storage.low_packed_rank[lc];
    if (hp == 0xffffffffu || lp == 0xffffffffu || bid >= shard.main_nblocks) {
        std::cerr << "invalid main address state\n";
        std::exit(151);
    }
    const uint32_t mask = seg_occ(hc, H);
    const uint32_t mr = hp & HM;
    const uint32_t lr = lp >> L;
    const Code off = shard.main_base[mask]
        + shard.main_block_off[size_t(mask) * shard.main_nblocks + bid]
        + Code(mr) * layout.main_blocks[bid].cols + lr;
    return {int(shard.owner[mask]), off};
}

static HostAddr addr_block(
    MateID m, const StorageFactorHost& storage,
    const StorageLayout& layout, const MaskShardLayout& shard
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LC = (1u << (2 * L)) - 1u;
    constexpr uint32_t HC = (1u << (2 * H)) - 1u;
    constexpr uint32_t HM = (1u << H) - 1u;
    const uint32_t lc = uint32_t(m) & LC;
    const uint32_t hc = uint32_t((m >> (2 * L)) & HC);
    const int h = seg_end_height_host(hc, H);
    const uint32_t bid = uint32_t(h);
    const uint32_t hp = storage.high_packed_rank[hc];
    const uint32_t lp = storage.low_packed_rank[lc];
    if (hp == 0xffffffffu || lp == 0xffffffffu || bid >= shard.block_nblocks) {
        std::cerr << "invalid blocked address state\n";
        std::exit(152);
    }
    const uint32_t mask = seg_occ(hc, H);
    const uint32_t mr = hp & HM;
    const uint32_t lr = lp >> L;
    const Code off = shard.block_base[mask]
        + shard.block_block_off[size_t(mask) * shard.block_nblocks + bid]
        + Code(mr) * layout.block_blocks[bid].cols + lr;
    return {int(shard.owner[mask]), off};
}

static void mark_unique(
    std::vector<std::vector<uint8_t>>& seen, HostAddr a, const char* what
) {
    if (a.owner < 0 || a.owner >= int(seen.size()) || a.off >= seen[a.owner].size()) {
        std::cerr << what << " address out of range owner=" << a.owner
                  << " off=" << a.off << '\n';
        std::exit(153);
    }
    if (seen[a.owner][size_t(a.off)]) {
        std::cerr << what << " duplicate address owner=" << a.owner
                  << " off=" << a.off << '\n';
        std::exit(154);
    }
    seen[a.owner][size_t(a.off)] = 1;
}

static uint32_t high_mask_begin_base(uint32_t mask, int h) {
    constexpr int S = FactorTablesHost::STRIDE;
    return G_FACTOR.high_mask_off[size_t(mask) * S + h];
}
static uint32_t low_mask_begin_base(uint32_t mask, int h) {
    constexpr int S = FactorTablesHost::STRIDE;
    return G_FACTOR.low_mask_off[size_t(mask) * S + h];
}

int main() {
    static_assert(TARGET_W <= 14,
                  "address selftest is exhaustive; compile it only at a small width");
    static_assert(LOW_LUT_K + HIGH_LUT_K + 1 == TARGET_W, "bad factor split");
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr int S = FactorTablesHost::STRIDE;
    constexpr uint32_t HM = (1u << H) - 1u;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    const StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    const StorageLayout layout = build_storage_layout(storage);
    const int ngpu = std::min(8, 1 << H);
    const MaskShardLayout shard = build_high_mask_shard_layout(storage, layout, ngpu);

    std::vector<std::vector<uint8_t>> seen_main(ngpu), seen_block(ngpu);
    for (int d = 0; d < ngpu; ++d) {
        seen_main[d].assign(size_t(shard.gpu_main[d]), 0);
        seen_block[d].assign(size_t(shard.gpu_block[d]), 0);
    }

    // Independent canonical enumeration: every legal state must map to exactly
    // one owner-local authoritative address.
    for (Code r = 0; r < layout.main_size; ++r) {
        MateID m = unrank_full_host(r, TARGET_W);
        if (rank_full_host(m, TARGET_W) != r) {
            std::cerr << "main canonical rank roundtrip failed r=" << r << '\n';
            return 155;
        }
        mark_unique(seen_main, addr_main(m, storage, layout, shard), "main");
    }
    for (Code r = 0; r < layout.block_size; ++r) {
        MateID m = unrank_full_host(r, TARGET_W - 1);
        if (rank_full_host(m, TARGET_W - 1) != r) {
            std::cerr << "block canonical rank roundtrip failed r=" << r << '\n';
            return 156;
        }
        mark_unique(seen_block, addr_block(m, storage, layout, shard), "block");
    }
    for (int d = 0; d < ngpu; ++d) {
        if (std::find(seen_main[d].begin(), seen_main[d].end(), uint8_t(0)) != seen_main[d].end() ||
            std::find(seen_block[d].begin(), seen_block[d].end(), uint8_t(0)) != seen_block[d].end()) {
            std::cerr << "authoritative arena has an unreachable slot on gpu=" << d << '\n';
            return 157;
        }
    }

    // LOW window: fixed HIGH occupancy mask.  Verify factorized local rank is
    // literally the owner-local authoritative offset within that mask group.
    uint64_t low_main = 0, low_block = 0;
    for (uint32_t mask = 0; mask < (1u << H); ++mask) {
        const auto mb = make_factor_main_blocks(false, mask);
        const auto db = make_factor_block_blocks(false, mask);
        for (uint32_t bid = 0; bid < mb.size(); ++bid) {
            const FBlock& x = mb[bid];
            if (!x.stride) continue;
            const uint32_t rows = uint32_t((x.end - x.off) / x.stride);
            const uint32_t ha = high_mask_begin_base(mask, x.he);
            for (uint32_t hr = 0; hr < rows; ++hr) {
                const uint32_t hc = G_FACTOR.high_mask_codes[ha + hr];
                const uint32_t hp = storage.high_packed_rank[hc];
                if ((hp & HM) != hr) {
                    std::cerr << "LOW high mask-rank mismatch\n";
                    return 158;
                }
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    const uint32_t lc = storage.low_all_codes[storage.low_all_off[x.hs] + lr];
                    MateID m = MateID(lc) | (MateID(x.c) << (2 * L))
                        | (MateID(hc) << (2 * (L + 1)));
                    const HostAddr a = addr_main(m, storage, layout, shard);
                    const Code want = shard.main_base[mask]
                        + shard.main_block_off[size_t(mask) * shard.main_nblocks + bid]
                        + (x.off + Code(hr) * x.stride + lr - x.off);
                    if (a.owner != int(shard.owner[mask]) || a.off != want) {
                        std::cerr << "LOW main direct-rank mismatch mask=" << mask
                                  << " bid=" << bid << " hr=" << hr << " lr=" << lr << '\n';
                        return 159;
                    }
                    ++low_main;
                }
            }
        }
        for (uint32_t bid = 0; bid < db.size(); ++bid) {
            const FBlock& x = db[bid];
            if (!x.stride) continue;
            const uint32_t rows = uint32_t((x.end - x.off) / x.stride);
            const uint32_t ha = high_mask_begin_base(mask, x.he);
            for (uint32_t hr = 0; hr < rows; ++hr) {
                const uint32_t hc = G_FACTOR.high_mask_codes[ha + hr];
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    const uint32_t lc = storage.low_all_codes[storage.low_all_off[x.hs] + lr];
                    MateID m = MateID(lc) | (MateID(hc) << (2 * L));
                    const HostAddr a = addr_block(m, storage, layout, shard);
                    const Code want = shard.block_base[mask]
                        + shard.block_block_off[size_t(mask) * shard.block_nblocks + bid]
                        + Code(hr) * x.stride + lr;
                    if (a.owner != int(shard.owner[mask]) || a.off != want) {
                        std::cerr << "LOW block direct-rank mismatch mask=" << mask
                                  << " bid=" << bid << " hr=" << hr << " lr=" << lr << '\n';
                        return 160;
                    }
                    ++low_block;
                }
            }
        }
    }

    // HIGH window: fixed LOW occupancy mask.  Mirror the device address route
    // and compare it against the independently reconstructed state's address.
    std::vector<uint8_t> high_main_seen(size_t(layout.main_size), 0);
    std::vector<uint8_t> high_block_seen(size_t(layout.block_size), 0);
    uint64_t high_main = 0, high_block = 0;
    for (uint32_t low_mask = 0; low_mask < (1u << L); ++low_mask) {
        const auto mb = make_factor_main_blocks(true, low_mask);
        const auto db = make_factor_block_blocks(true, low_mask);
        for (uint32_t bid = 0; bid < mb.size(); ++bid) {
            const FBlock& x = mb[bid];
            if (!x.stride) continue;
            const uint32_t rows = uint32_t((x.end - x.off) / x.stride);
            const uint32_t la = low_mask_begin_base(low_mask, x.hs);
            for (uint32_t hr = 0; hr < rows; ++hr) {
                const uint32_t hc = storage.high_all_codes[storage.high_all_off[x.he] + hr];
                const uint32_t hp = storage.high_packed_rank[hc];
                const uint32_t high_mask = seg_occ(hc, H);
                const uint32_t mr = hp & HM;
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    const uint32_t lc = G_FACTOR.low_mask_codes[la + lr];
                    const uint32_t lar = storage.low_packed_rank[lc] >> L;
                    const HostAddr routed{
                        int(shard.owner[high_mask]),
                        shard.main_base[high_mask]
                            + shard.main_block_off[size_t(high_mask) * shard.main_nblocks + bid]
                            + Code(mr) * layout.main_blocks[bid].cols + lar
                    };
                    MateID m = MateID(lc) | (MateID(x.c) << (2 * L))
                        | (MateID(hc) << (2 * (L + 1)));
                    const HostAddr direct = addr_main(m, storage, layout, shard);
                    if (routed.owner != direct.owner || routed.off != direct.off) {
                        std::cerr << "HIGH main route mismatch low_mask=" << low_mask
                                  << " bid=" << bid << " hr=" << hr << " lr=" << lr << '\n';
                        return 161;
                    }
                    const Code cr = rank_full_host(m, TARGET_W);
                    if (cr >= layout.main_size || high_main_seen[size_t(cr)]++) {
                        std::cerr << "HIGH main partition duplicate rank=" << cr << '\n';
                        return 162;
                    }
                    ++high_main;
                }
            }
        }
        for (uint32_t bid = 0; bid < db.size(); ++bid) {
            const FBlock& x = db[bid];
            if (!x.stride) continue;
            const uint32_t rows = uint32_t((x.end - x.off) / x.stride);
            const uint32_t la = low_mask_begin_base(low_mask, x.hs);
            for (uint32_t hr = 0; hr < rows; ++hr) {
                const uint32_t hc = storage.high_all_codes[storage.high_all_off[x.he] + hr];
                const uint32_t hp = storage.high_packed_rank[hc];
                const uint32_t high_mask = seg_occ(hc, H);
                const uint32_t mr = hp & HM;
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    const uint32_t lc = G_FACTOR.low_mask_codes[la + lr];
                    const uint32_t lar = storage.low_packed_rank[lc] >> L;
                    const HostAddr routed{
                        int(shard.owner[high_mask]),
                        shard.block_base[high_mask]
                            + shard.block_block_off[size_t(high_mask) * shard.block_nblocks + bid]
                            + Code(mr) * layout.block_blocks[bid].cols + lar
                    };
                    MateID m = MateID(lc) | (MateID(hc) << (2 * L));
                    const HostAddr direct = addr_block(m, storage, layout, shard);
                    if (routed.owner != direct.owner || routed.off != direct.off) {
                        std::cerr << "HIGH block route mismatch low_mask=" << low_mask
                                  << " bid=" << bid << " hr=" << hr << " lr=" << lr << '\n';
                        return 163;
                    }
                    const Code cr = rank_full_host(m, TARGET_W - 1);
                    if (cr >= layout.block_size || high_block_seen[size_t(cr)]++) {
                        std::cerr << "HIGH block partition duplicate rank=" << cr << '\n';
                        return 164;
                    }
                    ++high_block;
                }
            }
        }
    }

    if (low_main != layout.main_size || high_main != layout.main_size ||
        low_block != layout.block_size || high_block != layout.block_size ||
        std::find(high_main_seen.begin(), high_main_seen.end(), uint8_t(0)) != high_main_seen.end() ||
        std::find(high_block_seen.begin(), high_block_seen.end(), uint8_t(0)) != high_block_seen.end()) {
        std::cerr << "partition coverage mismatch low=" << low_main << '/' << low_block
                  << " high=" << high_main << '/' << high_block
                  << " expected=" << layout.main_size << '/' << layout.block_size << '\n';
        return 165;
    }

    std::cout << "maskshard-address-selftest OK W=" << TARGET_W
              << " low=" << L << " high=" << H
              << " gpus=" << ngpu
              << " main_states=" << layout.main_size
              << " block_states=" << layout.block_size
              << " low_direct=1 high_route=1 bijection=1\n";
    return 0;
}
