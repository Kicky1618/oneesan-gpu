#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"

static void high_test_enum_rec(int pos, int h, MateID m, std::vector<MateID>& out) {
    if (pos < 0) { if (h == 0) out.push_back(m); return; }
    high_test_enum_rec(pos - 1, h, mset(m, pos, N), out);
    if (h > 0) high_test_enum_rec(pos - 1, h - 1, mset(m, pos, R), out);
    high_test_enum_rec(pos - 1, h + 1, mset(m, pos, ::L), out);
}
static std::vector<MateID> high_test_enum(int width) {
    std::vector<MateID> out;
    high_test_enum_rec(width - 1, 1, 0, out);
    return out;
}
static inline Count high_test_add(Count a, Count b, Count mod) {
    return (a >= mod - b) ? a - (mod - b) : a + b;
}

// Independent host decoder for HIGH descriptors.  The production helpers are
// __device__ functions; duplicating the bit interpretation here keeps this
// selftest independent of the GPU decoder implementation while using the same
// published layout constants.
static inline uint32_t ht_highdesc_kind(uint32_t x) {
    return x >> HIGHDESC_KIND_SHIFT;
}
static inline uint32_t ht_highdesc_block(uint32_t x) {
    return (x >> HIGHDESC_BLOCK_SHIFT) & HIGHDESC_BLOCK_MASK;
}
static inline uint32_t ht_highdesc_rank(uint32_t x) {
    return x & HIGHDESC_RANK_MASK;
}
static inline uint32_t ht_highdesc_depth(uint32_t x) {
    return (x >> HIGHDESC_DEPTH_SHIFT) & HIGHDESC_DEPTH_MASK;
}
static inline uint32_t ht_highdesc_flip_low(uint32_t lc, uint32_t depth) {
    int s = int(depth);
    for (int pos = LOW_LUT_K - 1; pos >= 0; --pos) {
        MateValue v = MateValue((lc >> (2 * pos)) & 3u);
        if (v == ::L) {
            ++s;
        } else if (v == R) {
            if (--s == 0) {
                uint32_t z = 3u << (2 * pos);
                return (lc & ~z) | (uint32_t(::L) << (2 * pos));
            }
        }
    }
    return 0xffffffffu;
}
#define highdesc_kind ht_highdesc_kind
#define highdesc_block ht_highdesc_block
#define highdesc_rank ht_highdesc_rank
#define highdesc_depth ht_highdesc_depth
#define highdesc_flip_low ht_highdesc_flip_low

static void high_test_fill_factor(
    RamCounts& ma, RamCounts& ba,
    const std::vector<MateID>& ms, const std::vector<MateID>& bs,
    const std::vector<Count>& mv, const std::vector<Count>& bv,
    const StorageFactorHost& storage, const StorageLayout& layout
) {
    std::memset(ma.ptr, 0, ma.bytes);
    std::memset(ba.ptr, 0, ba.bytes);
    for (size_t i = 0; i < ms.size(); ++i)
        ma.ptr[storage_rank_main_host(ms[i], storage, layout)] = mv[i];
    for (size_t i = 0; i < bs.size(); ++i)
        ba.ptr[storage_rank_block_host(bs[i], storage, layout)] = bv[i];
}

static void high_test_pack(
    const RamCounts& auth, std::vector<Count>& local,
    const std::vector<FBlock>& fb, const std::vector<StorageBlock>& sb,
    uint32_t mask, const StorageFactorHost& storage
) {
    constexpr int S = StorageFactorHost::S;
    Code total = fb.empty() ? 0 : fb.back().end;
    local.assign(size_t(total), 0);
    for (size_t bid = 0; bid < fb.size(); ++bid) {
        const FBlock& x = fb[bid];
        const StorageBlock& y = sb[bid];
        if (!x.stride || x.end == x.off) continue;
        uint32_t col0 = storage.low_mask_begin[size_t(mask) * S + x.hs];
        Code rows = (x.end - x.off) / x.stride;
        for (Code hr = 0; hr < rows; ++hr) {
            std::memcpy(local.data() + x.off + hr * x.stride,
                        auth.ptr + y.off + hr * y.cols + col0,
                        size_t(x.stride) * sizeof(Count));
        }
    }
}

static void high_test_unpack(
    RamCounts& auth, const std::vector<Count>& local,
    const std::vector<FBlock>& fb, const std::vector<StorageBlock>& sb,
    uint32_t mask, const StorageFactorHost& storage
) {
    constexpr int S = StorageFactorHost::S;
    for (size_t bid = 0; bid < fb.size(); ++bid) {
        const FBlock& x = fb[bid];
        const StorageBlock& y = sb[bid];
        if (!x.stride || x.end == x.off) continue;
        uint32_t col0 = storage.low_mask_begin[size_t(mask) * S + x.hs];
        Code rows = (x.end - x.off) / x.stride;
        for (Code hr = 0; hr < rows; ++hr) {
            std::memcpy(auth.ptr + y.off + hr * y.cols + col0,
                        local.data() + x.off + hr * x.stride,
                        size_t(x.stride) * sizeof(Count));
        }
    }
}

static void high_test_run_group(
    RamCounts& ma, RamCounts& ba,
    uint32_t mask,
    const StorageFactorHost& storage, const StorageLayout& layout,
    const HighDescHost& desc, const HighOrbitHost& orbit, Count mod
) {
    constexpr int L = LOW_LUT_K;
    constexpr int S = FactorTablesHost::STRIDE;
    constexpr uint32_t LR_MASK = (1u << L) - 1u;
    auto mb = make_factor_main_blocks(true, mask);
    auto db = make_factor_block_blocks(true, mask);
    std::vector<Count> mainv, blockv;
    high_test_pack(ma, mainv, mb, layout.main_blocks, mask, storage);
    high_test_pack(ba, blockv, db, layout.block_blocks, mask, storage);

    for (int p = TARGET_W - 1; p >= L + 1; --p) {
        uint32_t pi = uint32_t((TARGET_W - 1) - p);
        for (size_t bid = 0; bid < mb.size(); ++bid) {
            const FBlock& x = mb[bid];
            if (!x.stride || x.end == x.off) continue;
            Code rows = (x.end - x.off) / x.stride;
            for (Code hr = 0; hr < rows; ++hr) {
                uint64_t ow = orbit.rec[
                    size_t(pi) * orbit.main_total + orbit.main_base[bid] + hr];
                uint32_t kind = high_orbit_kind(ow);
                if (kind < HIGH_ORBIT_NN || kind > HIGH_ORBIT_NL) continue;
                const FBlock& jy = mb[high_orbit_jblock(ow)];
                const FBlock& dy = db[high_orbit_dblock(ow)];
                if (jy.stride != x.stride || dy.stride != x.stride) {
                    std::cerr << "HIGH orbit stride mismatch mask=" << mask
                              << " p=" << p << " bid=" << bid << '\n';
                    std::exit(320);
                }
                Code ib = x.off + hr * x.stride;
                Code jb = jy.off + Code(high_orbit_jhr(ow)) * jy.stride;
                Code dd = dy.off + Code(high_orbit_dhr(ow)) * dy.stride;
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    Count c = mainv[ib + lr], d = blockv[dd + lr];
                    if (kind == HIGH_ORBIT_NN) {
                        mainv[jb + lr] = high_test_add(mainv[jb + lr], c, mod);
                        mainv[ib + lr] = high_test_add(c, d, mod);
                        blockv[dd + lr] = 0;
                    } else {
                        Count cc = mainv[jb + lr];
                        mainv[ib + lr] = high_test_add(high_test_add(c, cc, mod), d, mod);
                        blockv[dd + lr] = c;
                    }
                }
            }
        }

        for (size_t bid = 0; bid < mb.size(); ++bid) {
            const FBlock& x = mb[bid];
            if (!x.stride || x.end == x.off) continue;
            Code rows = (x.end - x.off) / x.stride;
            uint32_t low0 = G_FACTOR.low_mask_off[size_t(mask) * S + x.hs];
            for (Code hr = 0; hr < rows; ++hr) {
                uint64_t ow = orbit.rec[
                    size_t(pi) * orbit.main_total + orbit.main_base[bid] + hr];
                if (high_orbit_kind(ow) != HIGH_ORBIT_CLOSURE) continue;
                uint32_t word = desc.main_desc[
                    size_t(pi) * desc.main_total + desc.main_base[bid] + hr];
                uint32_t kind = highdesc_kind(word);
                const FBlock& y = db[highdesc_block(word)];
                Code ib = x.off + hr * x.stride;
                Code dstrow = y.off + Code(highdesc_rank(word)) * y.stride;
                if (kind == HIGHDESC_BLOCK) {
                    if (y.stride != x.stride) std::exit(321);
                    for (uint32_t lr = 0; lr < x.stride; ++lr) {
                        Count c = mainv[ib + lr];
                        blockv[dstrow + lr] = high_test_add(blockv[dstrow + lr], c, mod);
                    }
                } else if (kind == HIGHDESC_CROSS) {
                    for (uint32_t lr = 0; lr < x.stride; ++lr) {
                        Count c = mainv[ib + lr];
                        uint32_t lc = G_FACTOR.low_mask_codes[low0 + lr];
                        uint32_t lc2 = highdesc_flip_low(lc, highdesc_depth(word));
                        if (lc2 == 0xffffffffu) continue;
                        uint32_t packed = storage.low_packed_rank[lc2];
                        if (packed == 0xffffffffu) std::exit(322);
                        uint32_t lr2 = packed & LR_MASK;
                        if (lr2 >= y.stride) std::exit(323);
                        blockv[dstrow + lr2] = high_test_add(blockv[dstrow + lr2], c, mod);
                    }
                } else {
                    std::cerr << "unexpected HIGH closure kind=" << kind << '\n';
                    std::exit(324);
                }
            }
        }
    }

    high_test_unpack(ma, mainv, mb, layout.main_blocks, mask, storage);
    high_test_unpack(ba, blockv, db, layout.block_blocks, mask, storage);
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W;
    static_assert(W == LOW_LUT_K + HIGH_LUT_K + 1);
    static_assert(W <= 12, "HIGH orbit selftest intentionally uses small W");

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    HighDescHost desc = build_high_descriptors(storage, layout);
    HighOrbitHost orbit = build_high_orbit(storage, layout);

    auto ms = high_test_enum(W);
    auto bs = high_test_enum(W - 1);
    std::unordered_map<MateID,size_t> mi, bi;
    mi.reserve(ms.size() * 2); bi.reserve(bs.size() * 2);
    for (size_t i = 0; i < ms.size(); ++i) mi.emplace(ms[i], i);
    for (size_t i = 0; i < bs.size(); ++i) bi.emplace(bs[i], i);

    std::mt19937_64 rng(0xB3001618ULL);
    std::vector<Count> init_m(ms.size()), init_b(bs.size());
    for (auto& x : init_m) x = Count(rng() % mod);
    for (auto& x : init_b) x = Count(rng() % mod);

    std::vector<Count> rm = init_m, rb = init_b;
    for (int p = W - 1; p >= LOW_LUT_K + 1; --p) {
        std::vector<Count> nm = rm;
        std::vector<Count> nb(rb.size(), 0);
        for (size_t i = 0; i < ms.size(); ++i) {
            Count c = rm[i];
            auto z = oneesan::gridfp::include_horizontal(ms[i], W, p);
            if (!z.valid) continue;
            if (z.blocked) {
                auto it = bi.find(z.mate); if (it == bi.end()) return 330;
                nb[it->second] = high_test_add(nb[it->second], c, mod);
            } else {
                auto it = mi.find(z.mate); if (it == mi.end()) return 331;
                nm[it->second] = high_test_add(nm[it->second], c, mod);
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) {
            MateID z = oneesan::gridfp::blocked_exclude(bs[i], p);
            auto it = mi.find(z); if (it == mi.end()) return 332;
            nm[it->second] = high_test_add(nm[it->second], rb[i], mod);
        }
        rm.swap(nm); rb.swap(nb);
    }

    RamCounts ma, ba;
    ma.alloc(layout.main_size, "high-selftest main");
    ba.alloc(layout.block_size, "high-selftest block");
    high_test_fill_factor(ma, ba, ms, bs, init_m, init_b, storage, layout);

    uint32_t masks = 1u << LOW_LUT_K;
    for (uint32_t mask = 0; mask < masks; ++mask)
        high_test_run_group(ma, ba, mask, storage, layout, desc, orbit, mod);

    for (size_t i = 0; i < ms.size(); ++i) {
        Count got = ma.ptr[storage_rank_main_host(ms[i], storage, layout)];
        if (got != rm[i]) {
            std::cerr << "HIGH orbit main mismatch i=" << i
                      << " got=" << got << " want=" << rm[i] << '\n';
            return 333;
        }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        Count got = ba.ptr[storage_rank_block_host(bs[i], storage, layout)];
        if (got != rb[i]) {
            std::cerr << "HIGH orbit block mismatch i=" << i
                      << " got=" << got << " want=" << rb[i] << '\n';
            return 334;
        }
    }

    std::cout << "b300-high-orbit-selftest OK W=" << W
              << " main=" << ms.size() << " block=" << bs.size()
              << " masks=" << masks
              << " orbit_sources=" << orbit.orbit_sources
              << " closures=" << orbit.closures << '\n';
    ma.release(); ba.release();
    return 0;
}
