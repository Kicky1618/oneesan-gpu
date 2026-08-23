#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"

static inline Count test_add(Count a, Count b, Count mod) {
    if (!b) return a;
    return (a >= mod - b) ? a - (mod - b) : a + b;
}

static MateID host_unrank_main_fixlow(
    Code i, uint32_t mask, const std::vector<FBlock>& mb,
    const StorageFactorHost& storage
) {
    constexpr int L = LOW_LUT_K;
    constexpr int S = StorageFactorHost::S;
    size_t bid = 0;
    while (bid < mb.size() && i >= mb[bid].end) ++bid;
    if (bid >= mb.size()) std::exit(180);
    const FBlock& x = mb[bid];
    Code r = i - x.off;
    uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
    uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
    uint32_t hc = storage.high_all_codes[storage.high_all_off[x.he] + hr];
    uint32_t a = G_FACTOR.low_mask_off[size_t(mask) * S + x.hs];
    uint32_t lc = G_FACTOR.low_mask_codes[a + lr];
    return MateID(lc) | (MateID(x.c) << (2 * L))
         | (MateID(hc) << (2 * (L + 1)));
}

static MateID host_unrank_block_fixlow(
    Code i, uint32_t mask, const std::vector<FBlock>& db,
    const StorageFactorHost& storage
) {
    constexpr int L = LOW_LUT_K;
    constexpr int S = StorageFactorHost::S;
    size_t bid = 0;
    while (bid < db.size() && i >= db[bid].end) ++bid;
    if (bid >= db.size()) std::exit(181);
    const FBlock& x = db[bid];
    Code r = i - x.off;
    uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
    uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
    uint32_t hc = storage.high_all_codes[storage.high_all_off[x.he] + hr];
    uint32_t a = G_FACTOR.low_mask_off[size_t(mask) * S + x.hs];
    uint32_t lc = G_FACTOR.low_mask_codes[a + lr];
    return MateID(lc) | (MateID(hc) << (2 * L));
}

static Code host_rank_main_fixlow(
    MateID m, uint32_t mask, const std::vector<FBlock>& mb,
    const StorageFactorHost& storage
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LM = (1u << (2 * L)) - 1u;
    constexpr uint32_t HM = (1u << (2 * H)) - 1u;
    uint32_t lc = uint32_t(m) & LM;
    uint32_t hc = uint32_t((m >> (2 * (L + 1))) & HM);
    if (lowmask_major_occ(lc, L) != mask) std::exit(182);
    int he = seg_end_height_host(hc, H);
    uint32_t bid = uint32_t(3 * he + int(mget(m, L)));
    if (bid >= mb.size()) std::exit(183);
    uint32_t lp = storage.low_packed_rank[lc];
    uint32_t hp = storage.high_packed_rank[hc];
    if (lp == 0xffffffffu || hp == 0xffffffffu) std::exit(184);
    uint32_t lr = lp & ((1u << L) - 1u);
    uint32_t hr = hp >> H;
    const FBlock& x = mb[bid];
    if (lr >= x.stride || x.off + Code(hr) * x.stride + lr >= x.end) std::exit(185);
    return x.off + Code(hr) * x.stride + lr;
}

static Code host_rank_block_fixlow(
    MateID m, uint32_t mask, const std::vector<FBlock>& db,
    const StorageFactorHost& storage
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LM = (1u << (2 * L)) - 1u;
    constexpr uint32_t HM = (1u << (2 * H)) - 1u;
    uint32_t lc = uint32_t(m) & LM;
    uint32_t hc = uint32_t((m >> (2 * L)) & HM);
    if (lowmask_major_occ(lc, L) != mask) std::exit(186);
    uint32_t bid = uint32_t(seg_end_height_host(hc, H));
    if (bid >= db.size()) std::exit(187);
    uint32_t lp = storage.low_packed_rank[lc];
    uint32_t hp = storage.high_packed_rank[hc];
    if (lp == 0xffffffffu || hp == 0xffffffffu) std::exit(188);
    uint32_t lr = lp & ((1u << L) - 1u);
    uint32_t hr = hp >> H;
    const FBlock& x = db[bid];
    if (lr >= x.stride || x.off + Code(hr) * x.stride + lr >= x.end) std::exit(189);
    return x.off + Code(hr) * x.stride + lr;
}

static Code closure_target_from_desc(
    MateID m, Code i, uint32_t mask, int p,
    const std::vector<FBlock>& mb, const std::vector<FBlock>& db,
    const StorageFactorHost& storage, const HighDescHost& desc
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr int S = StorageFactorHost::S;
    size_t bid = 0;
    while (bid < mb.size() && i >= mb[bid].end) ++bid;
    if (bid >= mb.size()) std::exit(190);
    const FBlock& x = mb[bid];
    Code r = i - x.off;
    uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
    uint32_t lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    uint32_t dw = desc.main_desc[size_t(pi) * desc.main_total + desc.main_base[bid] + hr];
    uint32_t kind = highdesc_kind(dw);
    if (kind == HIGHDESC_BLOCK) {
        const FBlock& y = db[highdesc_block(dw)];
        return y.off + Code(highdesc_rank(dw)) * y.stride + lr;
    }
    if (kind == HIGHDESC_CROSS) {
        uint32_t lc = uint32_t(m) & ((1u << (2 * L)) - 1u);
        uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(dw));
        if (lc2 == 0xffffffffu) std::exit(191);
        if (lowmask_major_occ(lc2, L) != mask) std::exit(192);
        const FBlock& y = db[highdesc_block(dw)];
        uint32_t packed = storage.low_packed_rank[lc2];
        if (packed == 0xffffffffu) std::exit(193);
        uint32_t lr2 = packed & ((1u << L) - 1u);
        if (lr2 >= y.stride) std::exit(194);
        return y.off + Code(highdesc_rank(dw)) * y.stride + lr2;
    }
    std::cerr << "closure descriptor kind is not blocked/cross kind=" << kind << '\n';
    std::exit(195);
}

int main() {
    static_assert(TARGET_W <= 12, "selftest is intended for reduced widths");
    constexpr Count mod = 4294967291u;
    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout logical = build_storage_layout(storage);
    HighDescHost highdesc = build_high_descriptors(storage, logical);
    HighOrbitHost orbit = build_high_orbit(storage, logical);
    WindowPlan wp = make_direct2d_window(true);

    uint64_t seed = 0x123456789abcdef0ULL;
    auto rnd = [&]() -> Count {
        seed ^= seed << 7; seed ^= seed >> 9; seed ^= seed << 8;
        return Count(seed % mod);
    };

    uint64_t checked_edges = 0, checked_masks = 0;
    for (uint32_t mask = 0; mask < (1u << LOW_LUT_K); ++mask) {
        auto mb = make_factor_main_blocks(true, mask);
        auto db = make_factor_block_blocks(true, mask);
        Code ms = mb.empty() ? 0 : mb.back().end;
        Code ds = db.empty() ? 0 : db.back().end;
        if (!ms && !ds) continue;
        ++checked_masks;

        for (int p = wp.p_hi; p >= wp.p_lo; --p) {
            std::vector<Count> in_m(size_t(ms)), in_d(size_t(ds));
            for (auto& x : in_m) x = rnd();
            for (auto& x : in_d) x = rnd();

            // Reference out-of-place recurrence = identity main + include + blocked exclude.
            std::vector<Count> ref_m = in_m;
            std::vector<Count> ref_d(size_t(ds), 0);
            for (Code i = 0; i < ms; ++i) {
                Count c = in_m[size_t(i)];
                MateID m = host_unrank_main_fixlow(i, mask, mb, storage);
                auto z = oneesan::gridfp::include_horizontal(m, TARGET_W, p);
                if (!z.valid) continue;
                if (z.blocked) {
                    Code j = host_rank_block_fixlow(z.mate, mask, db, storage);
                    ref_d[size_t(j)] = test_add(ref_d[size_t(j)], c, mod);
                } else {
                    Code j = host_rank_main_fixlow(z.mate, mask, mb, storage);
                    ref_m[size_t(j)] = test_add(ref_m[size_t(j)], c, mod);
                }
            }
            for (Code i = 0; i < ds; ++i) {
                Count c = in_d[size_t(i)];
                MateID m = host_unrank_block_fixlow(i, mask, db, storage);
                MateID z = oneesan::gridfp::blocked_exclude(m, p);
                Code j = host_rank_main_fixlow(z, mask, mb, storage);
                ref_m[size_t(j)] = test_add(ref_m[size_t(j)], c, mod);
            }

            // New in-place HIGH orbit.
            std::vector<Count> got_m = in_m, got_d = in_d;
            uint32_t pi = uint32_t((TARGET_W - 1) - p);
            for (size_t bid = 0; bid < mb.size(); ++bid) {
                const FBlock& x = mb[bid];
                if (!x.stride || x.end == x.off) continue;
                Code rows = (x.end - x.off) / x.stride;
                for (uint32_t hr = 0; hr < rows; ++hr) {
                    uint64_t ow = orbit.rec[size_t(pi) * orbit.main_total
                                            + orbit.main_base[bid] + hr];
                    uint32_t kind = high_orbit_kind(ow);
                    if (kind < HIGH_ORBIT_NN || kind > HIGH_ORBIT_NL) continue;
                    const FBlock& jy = mb[high_orbit_jblock(ow)];
                    const FBlock& dy = db[high_orbit_dblock(ow)];
                    for (uint32_t lr = 0; lr < x.stride; ++lr) {
                        Code i = x.off + Code(hr) * x.stride + lr;
                        Code j = jy.off + Code(high_orbit_jhr(ow)) * jy.stride + lr;
                        Code dj = dy.off + Code(high_orbit_dhr(ow)) * dy.stride + lr;
                        Count c = got_m[size_t(i)];
                        Count d = got_d[size_t(dj)];
                        if (kind == HIGH_ORBIT_NN) {
                            got_m[size_t(j)] = test_add(got_m[size_t(j)], c, mod);
                            got_m[size_t(i)] = test_add(c, d, mod);
                            got_d[size_t(dj)] = 0;
                        } else {
                            Count cc = got_m[size_t(j)];
                            got_m[size_t(i)] = test_add(test_add(c, cc, mod), d, mod);
                            got_d[size_t(dj)] = c;
                        }
                    }
                }
            }

            // Closure pass, using the exact compact HIGH descriptor target used by GPU.
            for (Code i = 0; i < ms; ++i) {
                MateID m = host_unrank_main_fixlow(i, mask, mb, storage);
                int he = seg_end_height_host(uint32_t(m >> (2 * (LOW_LUT_K + 1))), HIGH_LUT_K);
                uint32_t bid = uint32_t(3 * he + int(mget(m, LOW_LUT_K)));
                const FBlock& x = mb[bid];
                Code r = i - x.off;
                uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0;
                uint64_t ow = orbit.rec[size_t(pi) * orbit.main_total
                                        + orbit.main_base[bid] + hr];
                if (high_orbit_kind(ow) != HIGH_ORBIT_CLOSURE) continue;
                Count c = got_m[size_t(i)];
                if (!c) continue;
                Code j = closure_target_from_desc(m, i, mask, p, mb, db, storage, highdesc);
                // Cross-check descriptor target against direct include_horizontal rank.
                auto z = oneesan::gridfp::include_horizontal(m, TARGET_W, p);
                if (!z.valid || !z.blocked) std::exit(196);
                Code direct = host_rank_block_fixlow(z.mate, mask, db, storage);
                if (j != direct) {
                    std::cerr << "closure target mismatch mask=" << mask << " p=" << p
                              << " i=" << i << " desc=" << j << " direct=" << direct << '\n';
                    return 197;
                }
                got_d[size_t(j)] = test_add(got_d[size_t(j)], c, mod);
            }

            if (got_m != ref_m || got_d != ref_d) {
                size_t badm = 0; while (badm < got_m.size() && got_m[badm] == ref_m[badm]) ++badm;
                size_t badd = 0; while (badd < got_d.size() && got_d[badd] == ref_d[badd]) ++badd;
                std::cerr << "HIGH orbit mismatch mask=" << mask << " p=" << p
                          << " main_bad=" << badm << '/' << got_m.size()
                          << " block_bad=" << badd << '/' << got_d.size() << '\n';
                return 198;
            }
            ++checked_edges;
        }
    }

    std::cout << "high-orbit-selftest OK W=" << TARGET_W
              << " masks=" << checked_masks
              << " edge_groups=" << checked_edges
              << " orbit_mib=" << double(orbit.rec.size() * sizeof(uint64_t)) / double(1 << 20)
              << '\n';
    return 0;
}
