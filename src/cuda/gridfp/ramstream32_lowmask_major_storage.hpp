#pragma once

#include "ramstream32_factorized_storage.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

// Authoritative RAM layout specialized for the hybrid executor.
//
// The GPU HIGH window fixes the complete LOW occupancy mask.  Store that mask
// as the outermost dimension, then concatenate the exact factor blocks in the
// same order as make_factor_{main,block}_blocks(fix_low=true, mask).  Therefore
// one HIGH-window group is one contiguous range in main RAM and one contiguous
// range in blocked RAM.
//
// Within a (mask, factor-block) range the order remains
//   HIGH-all-rank x LOW-mask-local-rank,
// so the existing factorized GPU kernels need no indexing changes.
struct LowMaskMajorLayout {
    static constexpr int S = StorageFactorHost::S;

    uint32_t masks = 0;
    uint32_t main_nblocks = 0;
    uint32_t block_nblocks = 0;
    std::vector<Code> main_mask_off;   // [masks+1]
    std::vector<Code> block_mask_off;  // [masks+1]
    // Absolute authoritative offsets.  Last entry of each mask row is the end.
    std::vector<Code> main_block_off;  // [mask][main_nblocks+1]
    std::vector<Code> block_block_off; // [mask][block_nblocks+1]
    Code main_size = 0;
    Code block_size = 0;
};

static inline uint32_t lowmask_major_occ(uint32_t code, int len) {
    uint32_t m = 0;
    for (int p = 0; p < len; ++p)
        if (((code >> (2 * p)) & 3u) != uint32_t(N)) m |= 1u << p;
    return m;
}

static inline uint32_t lowmask_major_width(uint32_t mask, int h) {
    constexpr int S = FactorTablesHost::STRIDE;
    size_t ix = size_t(mask) * S + size_t(h);
    return G_FACTOR.low_mask_off[ix + 1] - G_FACTOR.low_mask_off[ix];
}

static LowMaskMajorLayout build_lowmask_major_layout(
    const StorageFactorHost& storage, const StorageLayout& logical
) {
    constexpr int L = LOW_LUT_K;
    const uint32_t NM = 1u << L;

    LowMaskMajorLayout x;
    x.masks = NM;
    x.main_nblocks = uint32_t(logical.main_blocks.size());
    x.block_nblocks = uint32_t(logical.block_blocks.size());
    x.main_mask_off.resize(size_t(NM) + 1);
    x.block_mask_off.resize(size_t(NM) + 1);
    x.main_block_off.resize(size_t(NM) * (size_t(x.main_nblocks) + 1));
    x.block_block_off.resize(size_t(NM) * (size_t(x.block_nblocks) + 1));

    Code off = 0;
    for (uint32_t mask = 0; mask < NM; ++mask) {
        x.main_mask_off[mask] = off;
        size_t row = size_t(mask) * (size_t(x.main_nblocks) + 1);
        for (uint32_t bid = 0; bid < x.main_nblocks; ++bid) {
            x.main_block_off[row + bid] = off;
            const StorageBlock& b = logical.main_blocks[bid];
            if (!b.valid || !b.rows) continue;
            uint32_t w = lowmask_major_width(mask, b.hs);
            off += Code(b.rows) * Code(w);
        }
        x.main_block_off[row + x.main_nblocks] = off;
    }
    x.main_mask_off[NM] = off;
    x.main_size = off;

    off = 0;
    for (uint32_t mask = 0; mask < NM; ++mask) {
        x.block_mask_off[mask] = off;
        size_t row = size_t(mask) * (size_t(x.block_nblocks) + 1);
        for (uint32_t bid = 0; bid < x.block_nblocks; ++bid) {
            x.block_block_off[row + bid] = off;
            const StorageBlock& b = logical.block_blocks[bid];
            if (!b.valid || !b.rows) continue;
            uint32_t w = lowmask_major_width(mask, b.hs);
            off += Code(b.rows) * Code(w);
        }
        x.block_block_off[row + x.block_nblocks] = off;
    }
    x.block_mask_off[NM] = off;
    x.block_size = off;

    if (x.main_size != logical.main_size || x.block_size != logical.block_size) {
        std::cerr << "lowmask-major size mismatch main=" << x.main_size << '/' << logical.main_size
                  << " block=" << x.block_size << '/' << logical.block_size << '\n';
        std::exit(110);
    }

    // Verify that every LOW-mask group has exactly the local factorized size
    // expected by the GPU HIGH-window codec.
    for (uint32_t mask = 0; mask < NM; ++mask) {
        auto mb = make_factor_main_blocks(true, mask);
        auto db = make_factor_block_blocks(true, mask);
        Code ms = mb.empty() ? 0 : mb.back().end;
        Code ds = db.empty() ? 0 : db.back().end;
        Code mm = x.main_mask_off[mask + 1] - x.main_mask_off[mask];
        Code md = x.block_mask_off[mask + 1] - x.block_mask_off[mask];
        if (ms != mm || ds != md) {
            std::cerr << "lowmask-major group mismatch mask=" << mask
                      << " main=" << mm << '/' << ms
                      << " block=" << md << '/' << ds << '\n';
            std::exit(111);
        }
    }

    std::cerr << "lowmask_major main_gib="
              << double(x.main_size * sizeof(Count)) / double(1ULL << 30)
              << " block_gib="
              << double(x.block_size * sizeof(Count)) / double(1ULL << 30)
              << " metadata_mib="
              << double((x.main_mask_off.size() + x.block_mask_off.size()
                         + x.main_block_off.size() + x.block_block_off.size())
                        * sizeof(Code)) / double(1 << 20)
              << '\n';
    return x;
}

static inline Code lowmask_major_main_block_base(
    const LowMaskMajorLayout& x, uint32_t mask, uint32_t bid
) {
    return x.main_block_off[
        size_t(mask) * (size_t(x.main_nblocks) + 1) + bid];
}
static inline Code lowmask_major_block_block_base(
    const LowMaskMajorLayout& x, uint32_t mask, uint32_t bid
) {
    return x.block_block_off[
        size_t(mask) * (size_t(x.block_nblocks) + 1) + bid];
}

static Code lowmask_major_rank_main_host(
    MateID m, const StorageFactorHost& storage, const StorageLayout& logical,
    const LowMaskMajorLayout& x
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LM = (1u << (2 * L)) - 1u;
    constexpr uint32_t HM = (1u << (2 * H)) - 1u;
    constexpr uint32_t LR_MASK = (1u << L) - 1u;

    uint32_t lc = uint32_t(m) & LM;
    uint32_t hc = uint32_t((m >> (2 * (L + 1))) & HM);
    uint32_t mask = lowmask_major_occ(lc, L);
    int he = seg_end_height_host(hc, H);
    int cv = int(mget(m, L));
    uint32_t bid = uint32_t(3 * he + cv);
    if (bid >= logical.main_blocks.size()) std::exit(112);
    const StorageBlock& b = logical.main_blocks[bid];
    uint32_t lp = storage.low_packed_rank[lc];
    uint32_t hp = storage.high_packed_rank[hc];
    if (!b.valid || lp == 0xffffffffu || hp == 0xffffffffu) std::exit(113);
    uint32_t lr = lp & LR_MASK;
    uint32_t hr = hp >> H;
    uint32_t w = lowmask_major_width(mask, b.hs);
    if (lr >= w || hr >= b.rows) std::exit(114);
    return lowmask_major_main_block_base(x, mask, bid) + Code(hr) * w + lr;
}

static Code lowmask_major_rank_block_host(
    MateID m, const StorageFactorHost& storage, const StorageLayout& logical,
    const LowMaskMajorLayout& x
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LM = (1u << (2 * L)) - 1u;
    constexpr uint32_t HM = (1u << (2 * H)) - 1u;
    constexpr uint32_t LR_MASK = (1u << L) - 1u;

    uint32_t lc = uint32_t(m) & LM;
    uint32_t hc = uint32_t((m >> (2 * L)) & HM);
    uint32_t mask = lowmask_major_occ(lc, L);
    int h = seg_end_height_host(hc, H);
    uint32_t bid = uint32_t(h);
    if (bid >= logical.block_blocks.size()) std::exit(115);
    const StorageBlock& b = logical.block_blocks[bid];
    uint32_t lp = storage.low_packed_rank[lc];
    uint32_t hp = storage.high_packed_rank[hc];
    if (lp == 0xffffffffu || hp == 0xffffffffu) std::exit(116);
    uint32_t lr = lp & LR_MASK;
    uint32_t hr = hp >> H;
    uint32_t w = lowmask_major_width(mask, b.hs);
    if (lr >= w || hr >= b.rows) std::exit(117);
    return lowmask_major_block_block_base(x, mask, bid) + Code(hr) * w + lr;
}
