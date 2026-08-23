#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"
#include "../ramstream32_low_orbit_device.cuh"
#include "../ramstream32_b300_compact_io.cuh"

static MateID host_local_unrank_main(
    Code i, bool fix_low, uint32_t mask, const std::vector<FBlock>& fb
) {
    constexpr int L = LOW_LUT_K, H = HIGH_LUT_K, S = MAXW + 2;
    size_t bid = 0;
    while (bid < fb.size() && i >= fb[bid].end) ++bid;
    if (bid >= fb.size()) std::exit(280);
    const FBlock& x = fb[bid];
    Code r = i - x.off;
    uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
    uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
    uint32_t hc, lc;
    if (fix_low) {
        hc = G_FACTOR.high_all_codes[G_FACTOR.high_all_off[x.he] + hr];
        lc = G_FACTOR.low_mask_codes[G_FACTOR.low_mask_off[size_t(mask) * S + x.hs] + lr];
    } else {
        hc = G_FACTOR.high_mask_codes[G_FACTOR.high_mask_off[size_t(mask) * S + x.he] + hr];
        lc = G_FACTOR.low_all_codes[G_FACTOR.low_all_off[x.hs] + lr];
    }
    return MateID(lc) | (MateID(x.c) << (2 * L)) | (MateID(hc) << (2 * (L + 1)));
}

static MateID host_local_unrank_block(
    Code i, bool fix_low, uint32_t mask, const std::vector<FBlock>& fb
) {
    constexpr int L = LOW_LUT_K, H = HIGH_LUT_K, S = MAXW + 2;
    size_t bid = 0;
    while (bid < fb.size() && i >= fb[bid].end) ++bid;
    if (bid >= fb.size()) std::exit(281);
    const FBlock& x = fb[bid];
    Code r = i - x.off;
    uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
    uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
    uint32_t hc, lc;
    if (fix_low) {
        hc = G_FACTOR.high_all_codes[G_FACTOR.high_all_off[x.he] + hr];
        lc = G_FACTOR.low_mask_codes[G_FACTOR.low_mask_off[size_t(mask) * S + x.hs] + lr];
    } else {
        hc = G_FACTOR.high_mask_codes[G_FACTOR.high_mask_off[size_t(mask) * S + x.he] + hr];
        lc = G_FACTOR.low_all_codes[G_FACTOR.low_all_off[x.hs] + lr];
    }
    return MateID(lc) | (MateID(hc) << (2 * L));
}

static Code host_compact_main(
    Code i, bool fix_low, uint32_t mask, const std::vector<FBlock>& fb,
    const CompactCanonicalRankHost& t
) {
    constexpr int L = LOW_LUT_K, S = MAXW + 2;
    size_t bid = 0;
    while (bid < fb.size() && i >= fb[bid].end) ++bid;
    const FBlock& x = fb[bid];
    Code r = i - x.off;
    uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
    uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
    uint32_t har, lar;
    if (fix_low) {
        har = hr;
        lar = t.low_mask_all_rank[G_FACTOR.low_mask_off[size_t(mask) * S + x.hs] + lr];
    } else {
        har = t.high_mask_all_rank[G_FACTOR.high_mask_off[size_t(mask) * S + x.he] + hr];
        lar = lr;
    }
    Code rank = G_FACTOR.high_main_base[G_FACTOR.high_all_off[x.he] + har];
    MateValue c = MateValue(x.c);
    if (c > N) rank += H_DP[L][x.he];
    if (c > R && x.he > 0) rank += H_DP[L][x.he - 1];
    return rank + lar;
}

static Code host_compact_block(
    Code i, bool fix_low, uint32_t mask, const std::vector<FBlock>& fb,
    const CompactCanonicalRankHost& t
) {
    constexpr int S = MAXW + 2;
    size_t bid = 0;
    while (bid < fb.size() && i >= fb[bid].end) ++bid;
    const FBlock& x = fb[bid];
    Code r = i - x.off;
    uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
    uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
    uint32_t har, lar;
    if (fix_low) {
        har = hr;
        lar = t.low_mask_all_rank[G_FACTOR.low_mask_off[size_t(mask) * S + x.hs] + lr];
    } else {
        har = t.high_mask_all_rank[G_FACTOR.high_mask_off[size_t(mask) * S + x.he] + hr];
        lar = lr;
    }
    return G_FACTOR.high_block_base[G_FACTOR.high_all_off[x.he] + har] + lar;
}

int main() {
    static_assert(TARGET_W <= 12, "selftest intended for reduced width");
    build_full_dp();
    G_FACTOR = build_factor_tables();
    CompactCanonicalRankHost maps = build_compact_canonical_ranks();

    uint64_t checked_main = 0, checked_block = 0;
    for (int mode = 0; mode < 2; ++mode) {
        bool fix_low = mode == 0;
        uint32_t nm = 1u << (fix_low ? LOW_LUT_K : HIGH_LUT_K);
        for (uint32_t mask = 0; mask < nm; ++mask) {
            auto mb = make_factor_main_blocks(fix_low, mask);
            auto db = make_factor_block_blocks(fix_low, mask);
            Code ms = mb.empty() ? 0 : mb.back().end;
            Code ds = db.empty() ? 0 : db.back().end;
            for (Code i = 0; i < ms; ++i) {
                MateID m = host_local_unrank_main(i, fix_low, mask, mb);
                Code want = rank_full_suffix_host(m, TARGET_W, 1);
                Code got = host_compact_main(i, fix_low, mask, mb, maps);
                if (want != got) {
                    std::cerr << "compact main mismatch mode=" << mode << " mask=" << mask
                              << " i=" << i << " want=" << want << " got=" << got << '\n';
                    return 282;
                }
                ++checked_main;
            }
            for (Code i = 0; i < ds; ++i) {
                MateID m = host_local_unrank_block(i, fix_low, mask, db);
                Code want = rank_full_suffix_host(m, TARGET_W - 1, 1);
                Code got = host_compact_block(i, fix_low, mask, db, maps);
                if (want != got) {
                    std::cerr << "compact block mismatch mode=" << mode << " mask=" << mask
                              << " i=" << i << " want=" << want << " got=" << got << '\n';
                    return 283;
                }
                ++checked_block;
            }
        }
    }
    std::cout << "b300-compact-io-selftest OK W=" << TARGET_W
              << " main=" << checked_main << " block=" << checked_block << '\n';
    return 0;
}
