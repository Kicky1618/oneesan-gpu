#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define main oneesan_factorized_hbm_unused_main
#include "../oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu"
#undef main
#include "../../gridfp/ramstream32_factorized_storage.hpp"
#include "../maskshard_layout.hpp"

static void enum_states_rec(int pos, int h, MateID m, std::vector<MateID>& out) {
    if (pos < 0) {
        if (h == 0) out.push_back(m);
        return;
    }
    enum_states_rec(pos - 1, h, m, out);
    if (h > 0)
        enum_states_rec(pos - 1, h - 1, m | (MateID(R) << (2 * pos)), out);
    enum_states_rec(pos - 1, h + 1, m | (MateID(::L) << (2 * pos)), out);
}

static std::vector<MateID> enum_states(int width) {
    std::vector<MateID> out;
    enum_states_rec(width - 1, 1, 0, out);
    return out;
}

struct Addr { int owner; Code off; };

static Addr main_addr(
    MateID m, const StorageFactorHost& storage,
    const StorageLayout& layout, const MaskShardLayout& shard
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LC = (1u << (2 * L)) - 1u;
    constexpr uint32_t HC = (1u << (2 * H)) - 1u;
    constexpr uint32_t HR = (1u << H) - 1u;
    const uint32_t lc = uint32_t(m) & LC;
    const uint32_t hc = uint32_t((m >> (2 * (L + 1))) & HC);
    const int he = seg_end_height_host(hc, H);
    const uint32_t bid = uint32_t(3 * he + int(mget(m, L)));
    const uint32_t mask = seg_occ(hc, H);
    const uint32_t hp = storage.high_packed_rank[hc];
    const uint32_t lp = storage.low_packed_rank[lc];
    if (hp == 0xffffffffu || lp == 0xffffffffu || bid >= layout.main_blocks.size())
        return {-1, 0};
    const uint32_t mr = hp & HR;
    const uint32_t lr = lp >> L;
    return {int(shard.owner[mask]),
            shard.main_base[mask]
              + shard.main_block_off[size_t(mask) * shard.main_nblocks + bid]
              + Code(mr) * layout.main_blocks[bid].cols + lr};
}

static Addr block_addr(
    MateID m, const StorageFactorHost& storage,
    const StorageLayout& layout, const MaskShardLayout& shard
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LC = (1u << (2 * L)) - 1u;
    constexpr uint32_t HC = (1u << (2 * H)) - 1u;
    constexpr uint32_t HR = (1u << H) - 1u;
    const uint32_t lc = uint32_t(m) & LC;
    const uint32_t hc = uint32_t((m >> (2 * L)) & HC);
    const int he = seg_end_height_host(hc, H);
    const uint32_t bid = uint32_t(he);
    const uint32_t mask = seg_occ(hc, H);
    const uint32_t hp = storage.high_packed_rank[hc];
    const uint32_t lp = storage.low_packed_rank[lc];
    if (hp == 0xffffffffu || lp == 0xffffffffu || bid >= layout.block_blocks.size())
        return {-1, 0};
    const uint32_t mr = hp & HR;
    const uint32_t lr = lp >> L;
    return {int(shard.owner[mask]),
            shard.block_base[mask]
              + shard.block_block_off[size_t(mask) * shard.block_nblocks + bid]
              + Code(mr) * layout.block_blocks[bid].cols + lr};
}

static bool mark_bijection(
    const char* tag, const std::vector<MateID>& states,
    const std::vector<Code>& prefix, Code total,
    const StorageFactorHost& storage, const StorageLayout& layout,
    const MaskShardLayout& shard, bool blocked
) {
    std::vector<uint8_t> seen(size_t(total));
    for (MateID m : states) {
        const Addr a = blocked ? block_addr(m, storage, layout, shard)
                               : main_addr(m, storage, layout, shard);
        if (a.owner < 0 || a.owner >= shard.ngpu) {
            std::cerr << tag << " invalid owner\n";
            return false;
        }
        const Code cap = blocked ? shard.gpu_block[a.owner] : shard.gpu_main[a.owner];
        if (a.off >= cap) {
            std::cerr << tag << " offset overflow owner=" << a.owner
                      << " off=" << a.off << " cap=" << cap << '\n';
            return false;
        }
        const Code flat = prefix[a.owner] + a.off;
        if (flat >= total || seen[size_t(flat)]) {
            std::cerr << tag << " duplicate/out-of-range flat=" << flat << '\n';
            return false;
        }
        seen[size_t(flat)] = 1;
    }
    if (std::find(seen.begin(), seen.end(), uint8_t(0)) != seen.end()) {
        std::cerr << tag << " address hole\n";
        return false;
    }
    return true;
}

static bool verify_high_group_io_formula(
    const StorageFactorHost& storage, const StorageLayout& layout,
    const MaskShardLayout& shard
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr int S = StorageFactorHost::S;
    constexpr uint32_t HM = (1u << H) - 1u;
    const uint32_t low_masks = 1u << L;

    for (uint32_t low_mask = 0; low_mask < low_masks; ++low_mask) {
        const auto mb = make_factor_main_blocks(true, low_mask);
        for (uint32_t bid = 0; bid < mb.size(); ++bid) {
            const FBlock& f = mb[bid];
            const StorageBlock& sb = layout.main_blocks[bid];
            if (!f.stride) continue;
            const Code rows = (f.end - f.off) / f.stride;
            const uint32_t low0 = G_FACTOR.low_mask_off[size_t(low_mask) * S + f.hs];
            for (Code hr = 0; hr < rows; ++hr) {
                const uint32_t hc = storage.high_all_codes[storage.high_all_off[f.he] + uint32_t(hr)];
                const uint32_t hmask = seg_occ(hc, H);
                const uint32_t mr = storage.high_packed_rank[hc] & HM;
                for (uint32_t lr = 0; lr < f.stride; ++lr) {
                    const uint32_t lc = G_FACTOR.low_mask_codes[low0 + lr];
                    const uint32_t lar = storage.low_packed_rank[lc] >> L;
                    const MateID m = MateID(lc) | (MateID(f.c) << (2 * L))
                                   | (MateID(hc) << (2 * (L + 1)));
                    const Addr expected = main_addr(m, storage, layout, shard);
                    const Addr direct{
                        int(shard.owner[hmask]),
                        shard.main_base[hmask]
                          + shard.main_block_off[size_t(hmask) * shard.main_nblocks + bid]
                          + Code(mr) * sb.cols + lar
                    };
                    if (expected.owner != direct.owner || expected.off != direct.off) {
                        std::cerr << "high main I/O formula mismatch low_mask=" << low_mask
                                  << " bid=" << bid << '\n';
                        return false;
                    }
                }
            }
        }

        const auto db = make_factor_block_blocks(true, low_mask);
        for (uint32_t bid = 0; bid < db.size(); ++bid) {
            const FBlock& f = db[bid];
            const StorageBlock& sb = layout.block_blocks[bid];
            if (!f.stride) continue;
            const Code rows = (f.end - f.off) / f.stride;
            const uint32_t low0 = G_FACTOR.low_mask_off[size_t(low_mask) * S + f.hs];
            for (Code hr = 0; hr < rows; ++hr) {
                const uint32_t hc = storage.high_all_codes[storage.high_all_off[f.he] + uint32_t(hr)];
                const uint32_t hmask = seg_occ(hc, H);
                const uint32_t mr = storage.high_packed_rank[hc] & HM;
                for (uint32_t lr = 0; lr < f.stride; ++lr) {
                    const uint32_t lc = G_FACTOR.low_mask_codes[low0 + lr];
                    const uint32_t lar = storage.low_packed_rank[lc] >> L;
                    const MateID m = MateID(lc) | (MateID(hc) << (2 * L));
                    const Addr expected = block_addr(m, storage, layout, shard);
                    const Addr direct{
                        int(shard.owner[hmask]),
                        shard.block_base[hmask]
                          + shard.block_block_off[size_t(hmask) * shard.block_nblocks + bid]
                          + Code(mr) * sb.cols + lar
                    };
                    if (expected.owner != direct.owner || expected.off != direct.off) {
                        std::cerr << "high block I/O formula mismatch low_mask=" << low_mask
                                  << " bid=" << bid << '\n';
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

int main() {
    static_assert(TARGET_W <= 12, "address selftest intentionally uses small W");
    build_full_dp();
    G_FACTOR = build_factor_tables();
    const StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    const StorageLayout layout = build_storage_layout(storage);
    const MaskShardLayout shard = build_high_mask_shard_layout(storage, layout, 4);

    const auto main_states = enum_states(TARGET_W);
    const auto block_states = enum_states(TARGET_W - 1);
    if (main_states.size() != layout.main_size || block_states.size() != layout.block_size) {
        std::cerr << "state/layout size mismatch\n";
        return 2;
    }

    std::vector<Code> mp(4), bp(4);
    for (int d = 1; d < 4; ++d) {
        mp[d] = mp[d - 1] + shard.gpu_main[d - 1];
        bp[d] = bp[d - 1] + shard.gpu_block[d - 1];
    }
    if (!mark_bijection("main", main_states, mp, layout.main_size,
                        storage, layout, shard, false)) return 3;
    if (!mark_bijection("block", block_states, bp, layout.block_size,
                        storage, layout, shard, true)) return 4;
    if (!verify_high_group_io_formula(storage, layout, shard)) return 5;

    std::cout << "maskshard-address-selftest OK"
              << " W=" << TARGET_W
              << " low=" << LOW_LUT_K << " high=" << HIGH_LUT_K
              << " main=" << main_states.size()
              << " block=" << block_states.size()
              << '\n';
    return 0;
}
