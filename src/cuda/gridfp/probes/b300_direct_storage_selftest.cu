#include <cuda_runtime.h>

#include <cstdint>
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

static inline Count ds_add(Count a, Count b, Count mod) {
    return (a >= mod - b) ? a - (mod - b) : a + b;
}

static void ds_enum_rec(int pos, int h, MateID m, std::vector<MateID>& out) {
    if (pos < 0) { if (h == 0) out.push_back(m); return; }
    ds_enum_rec(pos - 1, h, mset(m, pos, N), out);
    if (h > 0) ds_enum_rec(pos - 1, h - 1, mset(m, pos, R), out);
    ds_enum_rec(pos - 1, h + 1, mset(m, pos, ::L), out);
}
static std::vector<MateID> ds_enum(int width) {
    std::vector<MateID> out;
    ds_enum_rec(width - 1, 1, 0, out);
    return out;
}

static inline uint32_t ds_ld_kind(uint32_t x) { return x >> LOWDESC_KIND_SHIFT; }
static inline uint32_t ds_ld_block(uint32_t x) { return (x >> LOWDESC_BLOCK_SHIFT) & LOWDESC_BLOCK_MASK; }
static inline uint32_t ds_ld_rank(uint32_t x) { return x & LOWDESC_LR_MASK; }
static inline uint32_t ds_ld_depth(uint32_t x) { return (x >> LOWDESC_DEPTH_SHIFT) & LOWDESC_DEPTH_MASK; }
static inline uint32_t ds_hd_kind(uint32_t x) { return x >> HIGHDESC_KIND_SHIFT; }
static inline uint32_t ds_hd_block(uint32_t x) { return (x >> HIGHDESC_BLOCK_SHIFT) & HIGHDESC_BLOCK_MASK; }
static inline uint32_t ds_hd_rank(uint32_t x) { return x & HIGHDESC_RANK_MASK; }
static inline uint32_t ds_hd_depth(uint32_t x) { return (x >> HIGHDESC_DEPTH_SHIFT) & HIGHDESC_DEPTH_MASK; }

static uint32_t ds_flip_high(uint32_t hc, uint32_t depth) {
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
static uint32_t ds_flip_low(uint32_t lc, uint32_t depth) {
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

static void ds_run_high(
    std::vector<Count>& mainv, std::vector<Count>& blockv,
    const StorageFactorHost& storage, const StorageLayout& layout,
    const B300SparseActionsHost& sparse, Count mod
) {
    constexpr uint32_t LR_SHIFT = LOW_LUT_K;
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        uint32_t pi = uint32_t((TARGET_W - 1) - p);
        for (uint32_t q = sparse.high_orbit_off[pi]; q < sparse.high_orbit_off[pi + 1]; ++q) {
            const auto& op = sparse.high_orbit[q];
            const StorageBlock& x = layout.main_blocks[b300_sparse_sblock(op)];
            const StorageBlock& jy = layout.main_blocks[b300_sparse_jblock(op)];
            const StorageBlock& dy = layout.block_blocks[b300_sparse_dblock(op)];
            uint32_t hr = b300_sparse_src(op);
            uint32_t jhr = b300_sparse_jrank(op);
            uint32_t dhr = b300_sparse_drank(op);
            if (hr >= x.rows || jhr >= jy.rows || dhr >= dy.rows ||
                x.cols != jy.cols || x.cols != dy.cols) return (void)std::exit(410);
            Code ib = x.off + Code(hr) * x.cols;
            Code jb = jy.off + Code(jhr) * jy.cols;
            Code db = dy.off + Code(dhr) * dy.cols;
            uint32_t kind = b300_sparse_kind(op);
            for (uint32_t lr = 0; lr < x.cols; ++lr) {
                Count c = mainv[ib + lr], d = blockv[db + lr];
                if (kind == HIGH_ORBIT_NN) {
                    mainv[jb + lr] = ds_add(mainv[jb + lr], c, mod);
                    mainv[ib + lr] = ds_add(c, d, mod);
                    blockv[db + lr] = 0;
                } else {
                    Count cc = mainv[jb + lr];
                    mainv[ib + lr] = ds_add(ds_add(c, cc, mod), d, mod);
                    blockv[db + lr] = c;
                }
            }
        }

        for (uint32_t q = sparse.high_closure_off[pi]; q < sparse.high_closure_off[pi + 1]; ++q) {
            uint64_t op = sparse.high_closure[q];
            uint32_t sbid = b300_sparse_closure_sblock(op);
            uint32_t hr = b300_sparse_closure_src(op);
            uint32_t desc = b300_sparse_closure_desc(op);
            const StorageBlock& x = layout.main_blocks[sbid];
            const StorageBlock& y = layout.block_blocks[ds_hd_block(desc)];
            uint32_t dhr = ds_hd_rank(desc);
            if (hr >= x.rows || dhr >= y.rows) std::exit(411);
            Code ib = x.off + Code(hr) * x.cols;
            Code db = y.off + Code(dhr) * y.cols;
            uint32_t kind = ds_hd_kind(desc);
            if (kind == HIGHDESC_BLOCK) {
                if (x.cols != y.cols) std::exit(412);
                for (uint32_t lr = 0; lr < x.cols; ++lr)
                    blockv[db + lr] = ds_add(blockv[db + lr], mainv[ib + lr], mod);
            } else if (kind == HIGHDESC_CROSS) {
                for (uint32_t lr = 0; lr < x.cols; ++lr) {
                    Count c = mainv[ib + lr];
                    if (!c) continue;
                    uint32_t lc = storage.low_all_codes[storage.low_all_off[x.hs] + lr];
                    uint32_t lc2 = ds_flip_low(lc, ds_hd_depth(desc));
                    if (lc2 == 0xffffffffu) continue;
                    uint32_t packed = storage.low_packed_rank[lc2];
                    if (packed == 0xffffffffu) std::exit(413);
                    uint32_t lr2 = packed >> LR_SHIFT;
                    if (lr2 >= y.cols) std::exit(414);
                    blockv[db + lr2] = ds_add(blockv[db + lr2], c, mod);
                }
            } else std::exit(415);
        }
    }
}

static void ds_run_low(
    std::vector<Count>& mainv, std::vector<Count>& blockv,
    const StorageFactorHost& storage, const StorageLayout& layout,
    const B300SparseActionsHost& sparse, Count mod
) {
    constexpr uint32_t HR_SHIFT = HIGH_LUT_K;
    for (int p = LOW_LUT_K; p >= 1; --p) {
        uint32_t pi = uint32_t(LOW_LUT_K - p);
        for (uint32_t q = sparse.low_orbit_off[pi]; q < sparse.low_orbit_off[pi + 1]; ++q) {
            const auto& op = sparse.low_orbit[q];
            const StorageBlock& x = layout.main_blocks[b300_sparse_sblock(op)];
            const StorageBlock& jy = layout.main_blocks[b300_sparse_jblock(op)];
            const StorageBlock& dy = layout.block_blocks[b300_sparse_dblock(op)];
            uint32_t lr = b300_sparse_src(op);
            uint32_t jlr = b300_sparse_jrank(op);
            uint32_t dlr = b300_sparse_drank(op);
            if (lr >= x.cols || jlr >= jy.cols || dlr >= dy.cols ||
                x.rows != jy.rows || x.rows != dy.rows) std::exit(416);
            uint32_t kind = b300_sparse_kind(op);
            for (uint32_t hr = 0; hr < x.rows; ++hr) {
                Count* ip = mainv.data() + x.off + Code(hr) * x.cols + lr;
                Count* jp = mainv.data() + jy.off + Code(hr) * jy.cols + jlr;
                Count* dp = blockv.data() + dy.off + Code(hr) * dy.cols + dlr;
                Count c = *ip, d = *dp;
                if (kind == CPU_ORBIT_NN) {
                    *jp = ds_add(*jp, c, mod);
                    *ip = ds_add(c, d, mod);
                    *dp = 0;
                } else {
                    Count cc = *jp;
                    Count all = ds_add(ds_add(c, cc, mod), d, mod);
                    if (p == 1) {
                        *ip = all;
                        *jp = ds_add(c, cc, mod);
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
            const StorageBlock& x = layout.main_blocks[sbid];
            if (lr >= x.cols) std::exit(417);
            uint32_t kind = ds_ld_kind(desc);
            for (uint32_t hr = 0; hr < x.rows; ++hr) {
                Count c = mainv[x.off + Code(hr) * x.cols + lr];
                if (!c) continue;
                if (kind == LOWDESC_MAIN) {
                    const StorageBlock& y = layout.main_blocks[ds_ld_block(desc)];
                    if (hr >= y.rows || ds_ld_rank(desc) >= y.cols) std::exit(418);
                    Code dst = y.off + Code(hr) * y.cols + ds_ld_rank(desc);
                    mainv[dst] = ds_add(mainv[dst], c, mod);
                } else if (kind == LOWDESC_BLOCK) {
                    const StorageBlock& y = layout.block_blocks[ds_ld_block(desc)];
                    if (hr >= y.rows || ds_ld_rank(desc) >= y.cols) std::exit(419);
                    Code dst = y.off + Code(hr) * y.cols + ds_ld_rank(desc);
                    blockv[dst] = ds_add(blockv[dst], c, mod);
                } else if (kind == LOWDESC_CROSS) {
                    uint32_t hc = storage.high_all_codes[storage.high_all_off[x.he] + hr];
                    uint32_t hc2 = ds_flip_high(hc, ds_ld_depth(desc));
                    if (hc2 == 0xffffffffu) continue;
                    uint32_t packed = storage.high_packed_rank[hc2];
                    if (packed == 0xffffffffu) std::exit(420);
                    uint32_t hr2 = packed >> HR_SHIFT;
                    if (p == 1) {
                        const StorageBlock& y = layout.main_blocks[ds_ld_block(desc)];
                        if (hr2 >= y.rows || ds_ld_rank(desc) >= y.cols) std::exit(421);
                        Code dst = y.off + Code(hr2) * y.cols + ds_ld_rank(desc);
                        mainv[dst] = ds_add(mainv[dst], c, mod);
                    } else {
                        const StorageBlock& y = layout.block_blocks[ds_ld_block(desc)];
                        if (hr2 >= y.rows || ds_ld_rank(desc) >= y.cols) std::exit(422);
                        Code dst = y.off + Code(hr2) * y.cols + ds_ld_rank(desc);
                        blockv[dst] = ds_add(blockv[dst], c, mod);
                    }
                } else std::exit(423);
            }
        }
    }
}

int main() {
    constexpr Count mod = 4294967291u;
    constexpr int W = TARGET_W;
    static_assert(W <= 12, "direct storage selftest intentionally reduced width");
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

    auto ms = ds_enum(W);
    auto bs = ds_enum(W - 1);
    std::unordered_map<MateID,size_t> mi, bi;
    mi.reserve(ms.size() * 2); bi.reserve(bs.size() * 2);
    for (size_t i = 0; i < ms.size(); ++i) mi.emplace(ms[i], i);
    for (size_t i = 0; i < bs.size(); ++i) bi.emplace(bs[i], i);

    std::mt19937_64 rng(0xD1EC750A6EULL);
    std::vector<Count> init(ms.size());
    for (auto& x : init) x = Count(rng() % mod);

    std::vector<Count> refm = init, refb(bs.size(), 0);
    for (int p = W - 1; p >= 1; --p) {
        std::vector<Count> nm = refm;
        std::vector<Count> nb(bs.size(), 0);
        for (size_t i = 0; i < ms.size(); ++i) {
            Count c = refm[i];
            auto z = oneesan::gridfp::include_horizontal(ms[i], W, p);
            if (!z.valid) continue;
            if (z.blocked) {
                auto it = bi.find(z.mate); if (it == bi.end()) return 424;
                nb[it->second] = ds_add(nb[it->second], c, mod);
            } else {
                auto it = mi.find(z.mate); if (it == mi.end()) return 425;
                nm[it->second] = ds_add(nm[it->second], c, mod);
            }
        }
        for (size_t i = 0; i < bs.size(); ++i) {
            MateID z = oneesan::gridfp::blocked_exclude(bs[i], p);
            auto it = mi.find(z); if (it == mi.end()) return 426;
            nm[it->second] = ds_add(nm[it->second], refb[i], mod);
        }
        refm.swap(nm); refb.swap(nb);
    }

    std::vector<Count> mainv(size_t(layout.main_size), 0);
    std::vector<Count> blockv(size_t(layout.block_size), 0);
    for (size_t i = 0; i < ms.size(); ++i)
        mainv[size_t(storage_rank_main_host(ms[i], storage, layout))] = init[i];

    ds_run_high(mainv, blockv, storage, layout, sparse, mod);
    ds_run_low(mainv, blockv, storage, layout, sparse, mod);

    for (size_t i = 0; i < ms.size(); ++i) {
        Count got = mainv[size_t(storage_rank_main_host(ms[i], storage, layout))];
        if (got != refm[i]) {
            std::cerr << "direct storage main mismatch i=" << i
                      << " got=" << got << " want=" << refm[i] << '\n';
            return 427;
        }
    }
    for (size_t i = 0; i < bs.size(); ++i) {
        Count got = blockv[size_t(storage_rank_block_host(bs[i], storage, layout))];
        if (got != refb[i]) {
            std::cerr << "direct storage block mismatch i=" << i
                      << " got=" << got << " want=" << refb[i] << '\n';
            return 428;
        }
    }

    std::cout << "b300-direct-storage-selftest OK W=" << W
              << " main=" << ms.size() << " block=" << bs.size()
              << " runtime_groups=0"
              << " gather_scatter_bytes=0"
              << " sparse_mib=" << double(sparse.bytes()) / double(1 << 20)
              << '\n';
    return 0;
}
