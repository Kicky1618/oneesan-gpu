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
#include "../ramstream32_cpu_low_inplace.hpp"
#include "../ramstream32_b300_sparse_actions.cuh"

static inline Count sa_add(Count a, Count b, Count mod) {
    return (a >= mod - b) ? a - (mod - b) : a + b;
}

static void sa_enum_rec(int pos, int h, MateID m, std::vector<MateID>& out) {
    if (pos < 0) { if (h == 0) out.push_back(m); return; }
    sa_enum_rec(pos - 1, h, mset(m, pos, N), out);
    if (h > 0) sa_enum_rec(pos - 1, h - 1, mset(m, pos, R), out);
    sa_enum_rec(pos - 1, h + 1, mset(m, pos, ::L), out);
}
static std::vector<MateID> sa_enum(int width) {
    std::vector<MateID> out;
    sa_enum_rec(width - 1, 1, 0, out);
    return out;
}

static inline uint32_t sa_ld_kind(uint32_t x) { return x >> LOWDESC_KIND_SHIFT; }
static inline uint32_t sa_ld_block(uint32_t x) { return (x >> LOWDESC_BLOCK_SHIFT) & LOWDESC_BLOCK_MASK; }
static inline uint32_t sa_ld_rank(uint32_t x) { return x & LOWDESC_LR_MASK; }
static inline uint32_t sa_ld_depth(uint32_t x) { return (x >> LOWDESC_DEPTH_SHIFT) & LOWDESC_DEPTH_MASK; }
static inline uint32_t sa_hd_kind(uint32_t x) { return x >> HIGHDESC_KIND_SHIFT; }
static inline uint32_t sa_hd_block(uint32_t x) { return (x >> HIGHDESC_BLOCK_SHIFT) & HIGHDESC_BLOCK_MASK; }
static inline uint32_t sa_hd_rank(uint32_t x) { return x & HIGHDESC_RANK_MASK; }
static inline uint32_t sa_hd_depth(uint32_t x) { return (x >> HIGHDESC_DEPTH_SHIFT) & HIGHDESC_DEPTH_MASK; }

static uint32_t sa_flip_high(uint32_t hc, uint32_t depth) {
    int s = int(depth);
    for (int pos = 0; pos < HIGH_LUT_K; ++pos) {
        MateValue v = MateValue((hc >> (2 * pos)) & 3u);
        if (v == ::L) {
            if (--s == 0) {
                uint32_t z = 3u << (2 * pos);
                return (hc & ~z) | (uint32_t(R) << (2 * pos));
            }
        } else if (v == R) ++s;
    }
    return 0xffffffffu;
}
static uint32_t sa_flip_low(uint32_t lc, uint32_t depth) {
    int s = int(depth);
    for (int pos = LOW_LUT_K - 1; pos >= 0; --pos) {
        MateValue v = MateValue((lc >> (2 * pos)) & 3u);
        if (v == ::L) ++s;
        else if (v == R) {
            if (--s == 0) {
                uint32_t z = 3u << (2 * pos);
                return (lc & ~z) | (uint32_t(::L) << (2 * pos));
            }
        }
    }
    return 0xffffffffu;
}

static void sa_pack(
    const RamCounts& auth, std::vector<Count>& local,
    const std::vector<FBlock>& fb, const std::vector<StorageBlock>& sb,
    bool fix_low, uint32_t mask, const StorageFactorHost& storage
) {
    constexpr int S = StorageFactorHost::S;
    Code total = fb.empty() ? 0 : fb.back().end;
    local.assign(size_t(total), 0);
    for (size_t bid = 0; bid < fb.size(); ++bid) {
        const FBlock& x = fb[bid];
        const StorageBlock& y = sb[bid];
        if (x.end == x.off || !x.stride) continue;
        if (fix_low) {
            uint32_t col0 = storage.low_mask_begin[size_t(mask) * S + x.hs];
            Code rows = (x.end - x.off) / x.stride;
            for (Code hr = 0; hr < rows; ++hr)
                std::memcpy(local.data() + x.off + hr * x.stride,
                            auth.ptr + y.off + hr * y.cols + col0,
                            size_t(x.stride) * sizeof(Count));
        } else {
            uint32_t row0 = storage.high_mask_begin[size_t(mask) * S + x.he];
            std::memcpy(local.data() + x.off,
                        auth.ptr + y.off + Code(row0) * y.cols,
                        size_t(x.end - x.off) * sizeof(Count));
        }
    }
}

static void sa_unpack(
    RamCounts& auth, const std::vector<Count>& local,
    const std::vector<FBlock>& fb, const std::vector<StorageBlock>& sb,
    bool fix_low, uint32_t mask, const StorageFactorHost& storage
) {
    constexpr int S = StorageFactorHost::S;
    for (size_t bid = 0; bid < fb.size(); ++bid) {
        const FBlock& x = fb[bid];
        const StorageBlock& y = sb[bid];
        if (x.end == x.off || !x.stride) continue;
        if (fix_low) {
            uint32_t col0 = storage.low_mask_begin[size_t(mask) * S + x.hs];
            Code rows = (x.end - x.off) / x.stride;
            for (Code hr = 0; hr < rows; ++hr)
                std::memcpy(auth.ptr + y.off + hr * y.cols + col0,
                            local.data() + x.off + hr * x.stride,
                            size_t(x.stride) * sizeof(Count));
        } else {
            uint32_t row0 = storage.high_mask_begin[size_t(mask) * S + x.he];
            std::memcpy(auth.ptr + y.off + Code(row0) * y.cols,
                        local.data() + x.off,
                        size_t(x.end - x.off) * sizeof(Count));
        }
    }
}

static void sa_run_high_group(
    RamCounts& ma, RamCounts& ba, uint32_t mask,
    const StorageFactorHost& storage, const StorageLayout& layout,
    const B300SparseActionsHost& sparse, Count mod
) {
    constexpr int S = FactorTablesHost::STRIDE;
    constexpr uint32_t LR_MASK = (1u << LOW_LUT_K) - 1u;
    auto mb = make_factor_main_blocks(true, mask);
    auto db = make_factor_block_blocks(true, mask);
    std::vector<Count> m, b;
    sa_pack(ma, m, mb, layout.main_blocks, true, mask, storage);
    sa_pack(ba, b, db, layout.block_blocks, true, mask, storage);

    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        uint32_t pi = uint32_t((TARGET_W - 1) - p);
        for (uint32_t q = sparse.high_orbit_off[pi]; q < sparse.high_orbit_off[pi + 1]; ++q) {
            const auto& op = sparse.high_orbit[q];
            const FBlock& x = mb[b300_sparse_sblock(op)];
            if (!x.stride) continue;
            uint32_t hr = b300_sparse_src(op);
            Code rows = (x.end - x.off) / x.stride;
            if (hr >= rows) return (void)std::exit(380);
            const FBlock& jy = mb[b300_sparse_jblock(op)];
            const FBlock& dy = db[b300_sparse_dblock(op)];
            if (jy.stride != x.stride || dy.stride != x.stride) std::exit(381);
            Code ib = x.off + Code(hr) * x.stride;
            Code jb = jy.off + Code(b300_sparse_jrank(op)) * jy.stride;
            Code dd = dy.off + Code(b300_sparse_drank(op)) * dy.stride;
            uint32_t kind = b300_sparse_kind(op);
            for (uint32_t lr = 0; lr < x.stride; ++lr) {
                Count c = m[ib + lr], d = b[dd + lr];
                if (kind == HIGH_ORBIT_NN) {
                    m[jb + lr] = sa_add(m[jb + lr], c, mod);
                    m[ib + lr] = sa_add(c, d, mod);
                    b[dd + lr] = 0;
                } else {
                    Count cc = m[jb + lr];
                    m[ib + lr] = sa_add(sa_add(c, cc, mod), d, mod);
                    b[dd + lr] = c;
                }
            }
        }

        for (uint32_t q = sparse.high_closure_off[pi]; q < sparse.high_closure_off[pi + 1]; ++q) {
            uint64_t op = sparse.high_closure[q];
            uint32_t sbid = b300_sparse_closure_sblock(op);
            uint32_t hr = b300_sparse_closure_src(op);
            uint32_t desc = b300_sparse_closure_desc(op);
            const FBlock& x = mb[sbid];
            if (!x.stride) continue;
            Code rows = (x.end - x.off) / x.stride;
            if (hr >= rows) std::exit(382);
            const FBlock& y = db[sa_hd_block(desc)];
            Code ib = x.off + Code(hr) * x.stride;
            Code dst = y.off + Code(sa_hd_rank(desc)) * y.stride;
            if (sa_hd_kind(desc) == HIGHDESC_BLOCK) {
                if (y.stride != x.stride) std::exit(383);
                for (uint32_t lr = 0; lr < x.stride; ++lr)
                    b[dst + lr] = sa_add(b[dst + lr], m[ib + lr], mod);
            } else if (sa_hd_kind(desc) == HIGHDESC_CROSS) {
                uint32_t low0 = G_FACTOR.low_mask_off[size_t(mask) * S + x.hs];
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    uint32_t lc = G_FACTOR.low_mask_codes[low0 + lr];
                    uint32_t lc2 = sa_flip_low(lc, sa_hd_depth(desc));
                    if (lc2 == 0xffffffffu) continue;
                    uint32_t packed = storage.low_packed_rank[lc2];
                    if (packed == 0xffffffffu) std::exit(384);
                    uint32_t lr2 = packed & LR_MASK;
                    if (lr2 >= y.stride) std::exit(385);
                    b[dst + lr2] = sa_add(b[dst + lr2], m[ib + lr], mod);
                }
            } else std::exit(386);
        }
    }

    sa_unpack(ma, m, mb, layout.main_blocks, true, mask, storage);
    sa_unpack(ba, b, db, layout.block_blocks, true, mask, storage);
}

static void sa_run_low_group(
    RamCounts& ma, RamCounts& ba, uint32_t mask,
    const StorageFactorHost& storage, const StorageLayout& layout,
    const B300SparseActionsHost& sparse, Count mod
) {
    constexpr int S = FactorTablesHost::STRIDE;
    constexpr uint32_t HR_MASK = (1u << HIGH_LUT_K) - 1u;
    auto mb = make_factor_main_blocks(false, mask);
    auto db = make_factor_block_blocks(false, mask);
    std::vector<Count> m, b;
    sa_pack(ma, m, mb, layout.main_blocks, false, mask, storage);
    sa_pack(ba, b, db, layout.block_blocks, false, mask, storage);

    for (int p = LOW_LUT_K; p >= 1; --p) {
        uint32_t pi = uint32_t(LOW_LUT_K - p);
        for (uint32_t q = sparse.low_orbit_off[pi]; q < sparse.low_orbit_off[pi + 1]; ++q) {
            const auto& op = sparse.low_orbit[q];
            const FBlock& x = mb[b300_sparse_sblock(op)];
            if (!x.stride) continue;
            uint32_t lr = b300_sparse_src(op);
            if (lr >= x.stride) std::exit(387);
            const FBlock& jy = mb[b300_sparse_jblock(op)];
            const FBlock& dy = db[b300_sparse_dblock(op)];
            Code rows = (x.end - x.off) / x.stride;
            uint32_t kind = b300_sparse_kind(op);
            for (Code hr = 0; hr < rows; ++hr) {
                Count* ip = m.data() + x.off + hr * x.stride + lr;
                Count* jp = m.data() + jy.off + hr * jy.stride + b300_sparse_jrank(op);
                Count* dp = b.data() + dy.off + hr * dy.stride + b300_sparse_drank(op);
                Count c = *ip, d = *dp;
                if (kind == CPU_ORBIT_NN) {
                    *jp = sa_add(*jp, c, mod);
                    *ip = sa_add(c, d, mod);
                    *dp = 0;
                } else {
                    Count cc = *jp;
                    Count all = sa_add(sa_add(c, cc, mod), d, mod);
                    if (p == 1) {
                        *ip = all;
                        *jp = sa_add(c, cc, mod);
                        *dp = 0;
                    } else {
                        *ip = all;
                        *dp = c;
                    }
                }
            }
        }

        for (uint32_t q = sparse.low_closure_off[pi]; q < sparse.low_closure_off[pi + 1]; ++q) {
            uint64_t op = sparse.low_closure[q];
            uint32_t sbid = b300_sparse_closure_sblock(op);
            uint32_t lr = b300_sparse_closure_src(op);
            uint32_t desc = b300_sparse_closure_desc(op);
            const FBlock& x = mb[sbid];
            if (!x.stride) continue;
            if (lr >= x.stride) std::exit(388);
            Code rows = (x.end - x.off) / x.stride;
            uint32_t kind = sa_ld_kind(desc);
            uint32_t high0 = G_FACTOR.high_mask_off[size_t(mask) * S + x.he];
            for (Code hr = 0; hr < rows; ++hr) {
                Count c = m[x.off + hr * x.stride + lr];
                if (!c) continue;
                if (kind == LOWDESC_MAIN) {
                    const FBlock& y = mb[sa_ld_block(desc)];
                    m[y.off + hr * y.stride + sa_ld_rank(desc)] =
                        sa_add(m[y.off + hr * y.stride + sa_ld_rank(desc)], c, mod);
                } else if (kind == LOWDESC_BLOCK) {
                    const FBlock& y = db[sa_ld_block(desc)];
                    b[y.off + hr * y.stride + sa_ld_rank(desc)] =
                        sa_add(b[y.off + hr * y.stride + sa_ld_rank(desc)], c, mod);
                } else if (kind == LOWDESC_CROSS) {
                    uint32_t hc = G_FACTOR.high_mask_codes[high0 + uint32_t(hr)];
                    uint32_t hc2 = sa_flip_high(hc, sa_ld_depth(desc));
                    if (hc2 == 0xffffffffu) continue;
                    uint32_t packed = storage.high_packed_rank[hc2];
                    if (packed == 0xffffffffu) std::exit(389);
                    uint32_t hr2 = packed & HR_MASK;
                    if (p == 1) {
                        const FBlock& y = mb[sa_ld_block(desc)];
                        if (hr2 >= (y.end - y.off) / y.stride) std::exit(390);
                        m[y.off + Code(hr2) * y.stride + sa_ld_rank(desc)] =
                            sa_add(m[y.off + Code(hr2) * y.stride + sa_ld_rank(desc)], c, mod);
                    } else {
                        const FBlock& y = db[sa_ld_block(desc)];
                        if (hr2 >= (y.end - y.off) / y.stride) std::exit(391);
                        b[y.off + Code(hr2) * y.stride + sa_ld_rank(desc)] =
                            sa_add(b[y.off + Code(hr2) * y.stride + sa_ld_rank(desc)], c, mod);
                    }
                } else std::exit(392);
            }
        }
    }

    sa_unpack(ma, m, mb, layout.main_blocks, false, mask, storage);
    // p=1 consumes blocked completely; resident runtime intentionally skips
    // the LOW blocked scatter and clears canonical blocked shards locally.
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W;
    static_assert(W <= 12, "sparse action selftest intentionally reduced width");
    static_assert(W == LOW_LUT_K + HIGH_LUT_K + 1);

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost loworbit = build_cpu_low_orbit(storage, layout, lowdesc);
    HighOrbitHost highorbit = build_high_orbit(storage, layout);
    B300SparseActionsHost sparse = build_b300_sparse_actions(
        layout, lowdesc, loworbit, highdesc, highorbit);

    auto ms = sa_enum(W);
    auto bs = sa_enum(W - 1);
    std::unordered_map<MateID,size_t> mi, bi;
    mi.reserve(ms.size() * 2); bi.reserve(bs.size() * 2);
    for (size_t i = 0; i < ms.size(); ++i) mi.emplace(ms[i], i);
    for (size_t i = 0; i < bs.size(); ++i) bi.emplace(bs[i], i);

    std::mt19937_64 rng(0x5A93E1618ULL);
    std::vector<Count> init(ms.size());
    for (auto& x : init) x = Count(rng() % mod);
    std::vector<Count> rm = init, rb(bs.size(), 0);
    for (int p = W - 1; p >= 1; --p) {
        std::vector<Count> nm = rm;
        std::vector<Count> nb(bs.size(), 0);
        for (size_t i = 0; i < ms.size(); ++i) {
            Count c = rm[i];
            auto z = oneesan::gridfp::include_horizontal(ms[i], W, p);
            if (!z.valid) continue;
            if (z.blocked) {
                auto it = bi.find(z.mate); if (it == bi.end()) return 393;
                nb[it->second] = sa_add(nb[it->second], c, mod);
            } else {
                auto it = mi.find(z.mate); if (it == mi.end()) return 394;
                nm[it->second] = sa_add(nm[it->second], c, mod);
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) {
            MateID z = oneesan::gridfp::blocked_exclude(bs[i], p);
            auto it = mi.find(z); if (it == mi.end()) return 395;
            nm[it->second] = sa_add(nm[it->second], rb[i], mod);
        }
        rm.swap(nm); rb.swap(nb);
    }

    RamCounts ma, ba;
    ma.alloc(layout.main_size, "sparse selftest main");
    ba.alloc(layout.block_size, "sparse selftest block");
    std::memset(ma.ptr, 0, ma.bytes);
    std::memset(ba.ptr, 0, ba.bytes);
    for (size_t i = 0; i < ms.size(); ++i)
        ma.ptr[storage_rank_main_host(ms[i], storage, layout)] = init[i];

    for (uint32_t mask = 0; mask < (1u << LOW_LUT_K); ++mask)
        sa_run_high_group(ma, ba, mask, storage, layout, sparse, mod);
    for (uint32_t mask = 0; mask < (1u << HIGH_LUT_K); ++mask)
        sa_run_low_group(ma, ba, mask, storage, layout, sparse, mod);
    std::memset(ba.ptr, 0, ba.bytes);

    for (size_t i = 0; i < ms.size(); ++i) {
        Count got = ma.ptr[storage_rank_main_host(ms[i], storage, layout)];
        if (got != rm[i]) {
            std::cerr << "sparse full-row main mismatch i=" << i
                      << " got=" << got << " want=" << rm[i] << '\n';
            return 396;
        }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        Count got = ba.ptr[storage_rank_block_host(bs[i], storage, layout)];
        if (got != rb[i]) {
            std::cerr << "sparse full-row block mismatch i=" << i
                      << " got=" << got << " want=" << rb[i] << '\n';
            return 397;
        }
    }

    std::cout << "b300-sparse-actions-selftest OK W=" << W
              << " main=" << ms.size() << " block=" << bs.size()
              << " sparse_mib=" << double(sparse.bytes()) / double(1 << 20)
              << " low_orbit=" << sparse.low_orbit.size()
              << " low_closure=" << sparse.low_closure.size()
              << " high_orbit=" << sparse.high_orbit.size()
              << " high_closure=" << sparse.high_closure.size() << '\n';
    ma.release(); ba.release();
    return 0;
}
