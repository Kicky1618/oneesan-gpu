#pragma once

#include "ramstream32_cpu_low_sparse.hpp"
#include "ramstream32_lowmask_major_storage.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

static constexpr int CPU_MM_LOCAL_BITS = 10;
static constexpr uint32_t CPU_MM_LOCAL_MASK = (1u << CPU_MM_LOCAL_BITS) - 1u;
static constexpr uint32_t CPU_MM_LOC_MASK = (1u << 24) - 1u;
static_assert(LOW_LUT_K <= 14, "24-bit LOW-mask-major loc assumes LOW_LUT_K<=14");
static_assert(HIGH_LUT_K <= 15, "uint16 cross-rank cache assumes HIGH_LUT_K<=15");

struct CpuLowMaskOrbitOp { uint32_t w0 = 0, w1 = 0, w2 = 0; };
static_assert(sizeof(CpuLowMaskOrbitOp) == 12);
struct CpuLowMaskClosureOp { uint32_t w0 = 0, w1 = 0; };
static_assert(sizeof(CpuLowMaskClosureOp) == 8);

static inline CpuLowMaskOrbitOp cpu_mm_make_orbit(
    uint32_t src, uint32_t j, uint32_t d, uint32_t kind,
    uint32_t jbid, uint32_t dbid
) {
    if ((src | j | d) > CPU_MM_LOC_MASK || kind > 7 || jbid > 63 || dbid > 63)
        std::exit(118);
    CpuLowMaskOrbitOp z;
    z.w0 = src | ((j & 0xffu) << 24);
    z.w1 = ((j >> 8) & 0xffffu) | ((d & 0xffffu) << 16);
    z.w2 = ((d >> 16) & 0xffu) | (kind << 8) | (jbid << 11) | (dbid << 17);
    return z;
}
static inline uint32_t cpu_mm_orbit_src(const CpuLowMaskOrbitOp& z) { return z.w0 & CPU_MM_LOC_MASK; }
static inline uint32_t cpu_mm_orbit_j(const CpuLowMaskOrbitOp& z) {
    return ((z.w0 >> 24) & 0xffu) | ((z.w1 & 0xffffu) << 8);
}
static inline uint32_t cpu_mm_orbit_d(const CpuLowMaskOrbitOp& z) {
    return ((z.w1 >> 16) & 0xffffu) | ((z.w2 & 0xffu) << 16);
}
static inline uint32_t cpu_mm_orbit_kind(const CpuLowMaskOrbitOp& z) { return (z.w2 >> 8) & 7u; }
static inline uint32_t cpu_mm_orbit_jblock(const CpuLowMaskOrbitOp& z) { return (z.w2 >> 11) & 0x3fu; }
static inline uint32_t cpu_mm_orbit_dblock(const CpuLowMaskOrbitOp& z) { return (z.w2 >> 17) & 0x3fu; }

static inline CpuLowMaskClosureOp cpu_mm_make_closure(
    uint32_t src, uint32_t dst, uint32_t kind, uint32_t block, uint32_t depth
) {
    if ((src | dst) > CPU_MM_LOC_MASK || kind > 3 || block > 63 || depth > 15)
        std::exit(119);
    CpuLowMaskClosureOp z;
    z.w0 = src | ((dst & 0xffu) << 24);
    z.w1 = ((dst >> 8) & 0xffffu) | (kind << 16) | (block << 18) | (depth << 24);
    return z;
}
static inline uint32_t cpu_mm_closure_src(const CpuLowMaskClosureOp& z) { return z.w0 & CPU_MM_LOC_MASK; }
static inline uint32_t cpu_mm_closure_dst(const CpuLowMaskClosureOp& z) {
    return ((z.w0 >> 24) & 0xffu) | ((z.w1 & 0xffffu) << 8);
}
static inline uint32_t cpu_mm_closure_kind(const CpuLowMaskClosureOp& z) { return (z.w1 >> 16) & 3u; }
static inline uint32_t cpu_mm_closure_block(const CpuLowMaskClosureOp& z) { return (z.w1 >> 18) & 0x3fu; }
static inline uint32_t cpu_mm_closure_depth(const CpuLowMaskClosureOp& z) { return (z.w1 >> 24) & 0x0fu; }

struct CpuLowMaskSparseHost {
    std::vector<CpuLowMaskOrbitOp> orbit_ops;
    std::vector<CpuLowMaskClosureOp> closure_ops;
    std::vector<uint32_t> orbit_off;
    std::vector<uint32_t> closure_off;
    // depth-major table.  Source index is the index into
    // G_FACTOR.high_mask_codes; value is target rank inside the same HIGH
    // occupancy class after the boundary mate flip.  0xffff = invalid.
    std::vector<uint16_t> high_cross_rank;
    uint32_t high_cross_stride = 0;
    uint32_t nblocks = 0;
};

static inline uint32_t cpu_mm_low_loc(
    const StorageFactorHost& storage, int h, uint32_t all_rank
) {
    constexpr int L = LOW_LUT_K;
    constexpr uint32_t LR_MASK = (1u << L) - 1u;
    uint32_t a = storage.low_all_off[h];
    uint32_t b = storage.low_all_off[h + 1];
    if (a + all_rank >= b) {
        std::cerr << "cpu mask-major LOW all-rank overflow h=" << h
                  << " rank=" << all_rank << '\n';
        std::exit(120);
    }
    uint32_t code = storage.low_all_codes[a + all_rank];
    uint32_t packed = storage.low_packed_rank[code];
    if (packed == 0xffffffffu) std::exit(121);
    uint32_t mask = lowmask_major_occ(code, L);
    uint32_t local = packed & LR_MASK;
    uint32_t w = lowmask_major_width(mask, h);
    if (local >= w) std::exit(122);
    if (local > CPU_MM_LOCAL_MASK) {
        std::cerr << "mask-major local rank needs >10 bits mask=" << mask
                  << " h=" << h << " local=" << local << " width=" << w << '\n';
        std::exit(123);
    }
    return (mask << CPU_MM_LOCAL_BITS) | local;
}

static CpuLowMaskSparseHost build_cpu_low_maskmajor_sparse(
    const StorageFactorHost& storage, const StorageLayout& logical,
    const LowDescHost& desc, const LowOrbitHost& orbit
) {
    CpuLowMaskSparseHost s;
    s.nblocks = uint32_t(logical.main_blocks.size());
    size_t pitch = size_t(s.nblocks) + 1;
    s.orbit_off.resize(size_t(LOW_LUT_K) * pitch);
    s.closure_off.resize(size_t(LOW_LUT_K) * pitch);

    for (int p = LOW_LUT_K; p >= 1; --p) {
        uint32_t pi = uint32_t(LOW_LUT_K - p);
        for (uint32_t bid = 0; bid < s.nblocks; ++bid) {
            s.orbit_off[size_t(pi) * pitch + bid] = uint32_t(s.orbit_ops.size());
            s.closure_off[size_t(pi) * pitch + bid] = uint32_t(s.closure_ops.size());
            const StorageBlock& xb = logical.main_blocks[bid];
            for (uint32_t lr = 0; lr < xb.cols; ++lr) {
                uint64_t ow = orbit.rec[
                    size_t(pi) * orbit.main_total + orbit.main_base[bid] + lr];
                uint32_t k = cpu_orbit_kind(ow);
                if (k >= CPU_ORBIT_NN && k <= CPU_ORBIT_NL) {
                    uint32_t jbid = cpu_orbit_jblock(ow);
                    uint32_t dbid = cpu_orbit_dblock(ow);
                    if (jbid >= logical.main_blocks.size() || dbid >= logical.block_blocks.size())
                        std::exit(124);
                    uint32_t src = cpu_mm_low_loc(storage, xb.hs, lr);
                    uint32_t j = cpu_mm_low_loc(storage, logical.main_blocks[jbid].hs,
                                                cpu_orbit_jlr(ow));
                    uint32_t d = cpu_mm_low_loc(storage, logical.block_blocks[dbid].hs,
                                                cpu_orbit_dlr(ow));
                    s.orbit_ops.push_back(cpu_mm_make_orbit(src, j, d, k, jbid, dbid));
                } else if (k == CPU_ORBIT_CLOSURE) {
                    uint32_t dw = desc.main_desc[
                        size_t(pi) * desc.main_total + desc.main_base[bid] + lr];
                    uint32_t kind = cpu_low_kind(dw);
                    if (kind == LOWDESC_INVALID) continue;
                    uint32_t dbid = cpu_low_block(dw);
                    int dh = 0;
                    if (kind == LOWDESC_MAIN || (kind == LOWDESC_CROSS && p == 1)) {
                        if (dbid >= logical.main_blocks.size()) std::exit(125);
                        dh = logical.main_blocks[dbid].hs;
                    } else {
                        if (dbid >= logical.block_blocks.size()) std::exit(126);
                        dh = logical.block_blocks[dbid].hs;
                    }
                    uint32_t src = cpu_mm_low_loc(storage, xb.hs, lr);
                    uint32_t dst = cpu_mm_low_loc(storage, dh, cpu_low_lr(dw));
                    s.closure_ops.push_back(cpu_mm_make_closure(
                        src, dst, kind, dbid, cpu_low_depth(dw)));
                }
            }
        }
        s.orbit_off[size_t(pi) * pitch + s.nblocks] = uint32_t(s.orbit_ops.size());
        s.closure_off[size_t(pi) * pitch + s.nblocks] = uint32_t(s.closure_ops.size());
    }

    // CPU LOW cross transitions used to scan the HIGH mate then binary-search
    // the target occupancy class for every state.  The source high-mask entry
    // is already known as (high0+hr), so precompute the target local rank for
    // all 15 possible boundary depths.  At H=13 this costs ~24 MiB instead of
    // retaining the 256-MiB dense 4^H packed-rank table.
    s.high_cross_stride = uint32_t(G_FACTOR.high_mask_codes.size());
    s.high_cross_rank.assign(size_t(16) * s.high_cross_stride, uint16_t(0xffff));
    constexpr uint32_t HR_MASK = (1u << HIGH_LUT_K) - 1u;
    for (uint32_t ix = 0; ix < s.high_cross_stride; ++ix) {
        uint32_t hc = G_FACTOR.high_mask_codes[ix];
        for (uint32_t depth = 1; depth < 16; ++depth) {
            uint32_t hc2 = cpu_low_flip_high(hc, depth);
            if (hc2 == 0xffffffffu) continue;
            uint32_t packed = storage.high_packed_rank[hc2];
            if (packed == 0xffffffffu) continue;
            uint32_t hr2 = packed & HR_MASK;
            if (hr2 >= 0xffffu) std::exit(127);
            s.high_cross_rank[size_t(depth) * s.high_cross_stride + ix] = uint16_t(hr2);
        }
    }

    std::cerr << "cpu_low_maskmajor_sparse orbit_ops=" << s.orbit_ops.size()
              << " closure_ops=" << s.closure_ops.size()
              << " orbit_mib=" << double(s.orbit_ops.size() * sizeof(CpuLowMaskOrbitOp)) / (1<<20)
              << " closure_mib=" << double(s.closure_ops.size() * sizeof(CpuLowMaskClosureOp)) / (1<<20)
              << " cross_rank_mib=" << double(s.high_cross_rank.size() * sizeof(uint16_t)) / (1<<20)
              << '\n';
    return s;
}

static inline std::pair<uint32_t,uint32_t> cpu_mm_range(
    const std::vector<uint32_t>& off, uint32_t nblocks, uint32_t pi, uint32_t bid
) {
    size_t pitch = size_t(nblocks) + 1;
    return {off[size_t(pi) * pitch + bid], off[size_t(pi) * pitch + bid + 1]};
}
static inline uint32_t cpu_mm_loc_mask(uint32_t loc) { return loc >> CPU_MM_LOCAL_BITS; }
static inline uint32_t cpu_mm_loc_rank(uint32_t loc) { return loc & CPU_MM_LOCAL_MASK; }
static inline uint32_t cpu_mm_cross_rank(
    const CpuLowMaskSparseHost& s, uint32_t source_ix, uint32_t depth
) {
    if (depth >= 16 || source_ix >= s.high_cross_stride) return 0xffffffffu;
    uint16_t z = s.high_cross_rank[size_t(depth) * s.high_cross_stride + source_ix];
    return z == uint16_t(0xffff) ? 0xffffffffu : uint32_t(z);
}
static inline size_t cpu_mm_sparse_bytes(const CpuLowMaskSparseHost& s) {
    return s.orbit_ops.size() * sizeof(CpuLowMaskOrbitOp)
         + s.closure_ops.size() * sizeof(CpuLowMaskClosureOp)
         + (s.orbit_off.size() + s.closure_off.size()) * sizeof(uint32_t)
         + s.high_cross_rank.size() * sizeof(uint16_t);
}

struct CpuMmColumn { Count* base = nullptr; uint32_t stride = 0; };

static inline CpuMmColumn cpu_mm_main_col(
    RamCounts& auth, const LowMaskMajorLayout& mm,
    const StorageLayout& logical, uint32_t row0,
    uint32_t bid, uint32_t loc
) {
    if (bid >= logical.main_blocks.size()) std::exit(128);
    const StorageBlock& b = logical.main_blocks[bid];
    uint32_t low_mask = cpu_mm_loc_mask(loc);
    uint32_t lr = cpu_mm_loc_rank(loc);
    uint32_t w = lowmask_major_width(low_mask, b.hs);
    if (!w || lr >= w || row0 >= b.rows) std::exit(129);
    return {auth.ptr + lowmask_major_main_block_base(mm, low_mask, bid)
                    + Code(row0) * w + lr, w};
}
static inline CpuMmColumn cpu_mm_block_col(
    RamCounts& auth, const LowMaskMajorLayout& mm,
    const StorageLayout& logical, uint32_t row0,
    uint32_t bid, uint32_t loc
) {
    if (bid >= logical.block_blocks.size()) std::exit(130);
    const StorageBlock& b = logical.block_blocks[bid];
    uint32_t low_mask = cpu_mm_loc_mask(loc);
    uint32_t lr = cpu_mm_loc_rank(loc);
    uint32_t w = lowmask_major_width(low_mask, b.hs);
    if (!w || lr >= w || row0 >= b.rows) std::exit(131);
    return {auth.ptr + lowmask_major_block_block_base(mm, low_mask, bid)
                    + Code(row0) * w + lr, w};
}
static inline Count* cpu_mm_main_ptr_row0(
    RamCounts& auth, const LowMaskMajorLayout& mm,
    const StorageLayout& logical, uint32_t row0,
    uint32_t bid, uint32_t loc, Code hr
) {
    CpuMmColumn c = cpu_mm_main_col(auth, mm, logical, row0, bid, loc);
    return c.base + hr * c.stride;
}
static inline Count* cpu_mm_block_ptr_row0(
    RamCounts& auth, const LowMaskMajorLayout& mm,
    const StorageLayout& logical, uint32_t row0,
    uint32_t bid, uint32_t loc, Code hr
) {
    CpuMmColumn c = cpu_mm_block_col(auth, mm, logical, row0, bid, loc);
    return c.base + hr * c.stride;
}

struct CpuLowMaskMajorStats {
    double kernel_s = 0.0;
    uint64_t groups = 0;
    uint64_t orbit_columns = 0;
    uint64_t closure_columns = 0;
};

static void process_cpu_low_group_maskmajor(
    CpuLowMaskMajorStats& stats, const CpuLowJob& job,
    RamCounts& main_auth, RamCounts& block_auth,
    const StorageFactorHost& storage, const StorageLayout& logical,
    const LowMaskMajorLayout& mm, const CpuLowMaskSparseHost& sparse, Count mod
) {
    if (!job.main_size && !job.block_size) return;

    std::array<uint32_t, 3 * (MAXW + 2)> main_row0{};
    std::array<uint32_t, MAXW + 2> block_row0{};
    for (uint32_t bid = 0; bid < logical.main_blocks.size(); ++bid) {
        const StorageBlock& b = logical.main_blocks[bid];
        main_row0[bid] = storage.high_mask_begin[
            size_t(job.mask) * StorageFactorHost::S + b.he];
    }
    for (uint32_t bid = 0; bid < logical.block_blocks.size(); ++bid) {
        const StorageBlock& b = logical.block_blocks[bid];
        block_row0[bid] = storage.high_mask_begin[
            size_t(job.mask) * StorageFactorHost::S + b.he];
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int p = LOW_LUT_K; p >= 1; --p) {
        uint32_t pi = uint32_t(LOW_LUT_K - p);

        // N* owner orbits preserve HIGH, so resolve each physical column once
        // and stream through all HIGH rows.
        for (uint32_t bid = 0; bid < sparse.nblocks; ++bid) {
            const FBlock& xb = job.main_blocks[bid];
            if (!xb.stride || xb.end == xb.off) continue;
            Code rows = (xb.end - xb.off) / xb.stride;
            auto [oa, ob] = cpu_mm_range(sparse.orbit_off, sparse.nblocks, pi, bid);
            for (uint32_t q = oa; q < ob; ++q) {
                const CpuLowMaskOrbitOp& op = sparse.orbit_ops[q];
                uint32_t kind = cpu_mm_orbit_kind(op);
                uint32_t jbid = cpu_mm_orbit_jblock(op);
                uint32_t dbid = cpu_mm_orbit_dblock(op);
                CpuMmColumn ic = cpu_mm_main_col(
                    main_auth, mm, logical, main_row0[bid], bid, cpu_mm_orbit_src(op));
                CpuMmColumn jc = cpu_mm_main_col(
                    main_auth, mm, logical, main_row0[jbid], jbid, cpu_mm_orbit_j(op));
                CpuMmColumn dc = cpu_mm_block_col(
                    block_auth, mm, logical, block_row0[dbid], dbid, cpu_mm_orbit_d(op));
                ++stats.orbit_columns;
                for (Code hr = 0; hr < rows; ++hr) {
                    Count* ip = ic.base + hr * ic.stride;
                    Count* jp = jc.base + hr * jc.stride;
                    Count* dd = dc.base + hr * dc.stride;
                    Count c = *ip;
                    Count d = *dd;
                    if (kind == CPU_ORBIT_NN) {
                        *jp = cpu_low_add(*jp, c, mod);
                        *ip = cpu_low_add(c, d, mod);
                        *dd = 0;
                    } else {
                        Count cc = *jp;
                        Count all = cpu_low_add(cpu_low_add(c, cc, mod), d, mod);
                        if (p == 1) {
                            *ip = all;
                            *jp = cpu_low_add(c, cc, mod);
                            *dd = 0;
                        } else {
                            *ip = all;
                            *dd = c;
                        }
                    }
                }
            }
        }

        // For p>1 every closure reads main and only accumulates into blocked.
        // No closure can change another closure source, and modular addition is
        // commutative, so op/row loop interchange is exact.  p=1 can target
        // main and retains the previously validated row-major order below.
        if (p > 1) {
            for (uint32_t bid = 0; bid < sparse.nblocks; ++bid) {
                const FBlock& xb = job.main_blocks[bid];
                if (!xb.stride || xb.end == xb.off) continue;
                Code rows = (xb.end - xb.off) / xb.stride;
                auto [ca, cb] = cpu_mm_range(sparse.closure_off, sparse.nblocks, pi, bid);
                uint32_t high0 = G_FACTOR.high_mask_off[
                    size_t(job.mask) * FactorTablesHost::STRIDE + xb.he];
                for (uint32_t q = ca; q < cb; ++q) {
                    const CpuLowMaskClosureOp& op = sparse.closure_ops[q];
                    uint32_t kind = cpu_mm_closure_kind(op);
                    uint32_t dbid = cpu_mm_closure_block(op);
                    uint32_t dst = cpu_mm_closure_dst(op);
                    CpuMmColumn sc = cpu_mm_main_col(
                        main_auth, mm, logical, main_row0[bid], bid, cpu_mm_closure_src(op));
                    ++stats.closure_columns;
                    if (kind == LOWDESC_BLOCK) {
                        CpuMmColumn dc = cpu_mm_block_col(
                            block_auth, mm, logical, block_row0[dbid], dbid, dst);
                        for (Code hr = 0; hr < rows; ++hr) {
                            Count c = sc.base[hr * sc.stride];
                            if (!c) continue;
                            Count* j = dc.base + hr * dc.stride;
                            *j = cpu_low_add(*j, c, mod);
                        }
                    } else if (kind == LOWDESC_CROSS) {
                        CpuMmColumn dc = cpu_mm_block_col(
                            block_auth, mm, logical, block_row0[dbid], dbid, dst);
                        uint32_t depth = cpu_mm_closure_depth(op);
                        for (Code hr = 0; hr < rows; ++hr) {
                            Count c = sc.base[hr * sc.stride];
                            if (!c) continue;
                            uint32_t hr2 = cpu_mm_cross_rank(
                                sparse, high0 + uint32_t(hr), depth);
                            if (hr2 == 0xffffffffu) continue;
                            Count* j = dc.base + Code(hr2) * dc.stride;
                            *j = cpu_low_add(*j, c, mod);
                        }
                    } else if (kind == LOWDESC_MAIN) {
                        // Defensive fallback; the mathematical p>1 closure
                        // cases are blocked/cross, but keep semantics exact if
                        // descriptor construction is extended later.
                        CpuMmColumn dc = cpu_mm_main_col(
                            main_auth, mm, logical, main_row0[dbid], dbid, dst);
                        for (Code hr = 0; hr < rows; ++hr) {
                            Count c = sc.base[hr * sc.stride];
                            if (!c) continue;
                            Count* j = dc.base + hr * dc.stride;
                            *j = cpu_low_add(*j, c, mod);
                        }
                    }
                }
            }
            continue;
        }

        // p=1: retain the proven row-major closure order because destinations
        // are main states and can participate in the same final-edge phase.
        for (uint32_t bid = 0; bid < sparse.nblocks; ++bid) {
            const FBlock& xb = job.main_blocks[bid];
            if (!xb.stride || xb.end == xb.off) continue;
            Code rows = (xb.end - xb.off) / xb.stride;
            auto [ca, cb] = cpu_mm_range(sparse.closure_off, sparse.nblocks, pi, bid);
            uint32_t high0 = G_FACTOR.high_mask_off[
                size_t(job.mask) * FactorTablesHost::STRIDE + xb.he];
            for (Code hr = 0; hr < rows; ++hr) {
                for (uint32_t q = ca; q < cb; ++q) {
                    const CpuLowMaskClosureOp& op = sparse.closure_ops[q];
                    Count* src = cpu_mm_main_ptr_row0(
                        main_auth, mm, logical, main_row0[bid],
                        bid, cpu_mm_closure_src(op), hr);
                    Count c = *src;
                    if (!c) continue;
                    uint32_t kind = cpu_mm_closure_kind(op);
                    uint32_t dbid = cpu_mm_closure_block(op);
                    uint32_t dst = cpu_mm_closure_dst(op);
                    if (kind == LOWDESC_MAIN) {
                        Count* j = cpu_mm_main_ptr_row0(
                            main_auth, mm, logical, main_row0[dbid], dbid, dst, hr);
                        *j = cpu_low_add(*j, c, mod);
                    } else if (kind == LOWDESC_BLOCK) {
                        Count* j = cpu_mm_block_ptr_row0(
                            block_auth, mm, logical, block_row0[dbid], dbid, dst, hr);
                        *j = cpu_low_add(*j, c, mod);
                    } else if (kind == LOWDESC_CROSS) {
                        uint32_t hr2 = cpu_mm_cross_rank(
                            sparse, high0 + uint32_t(hr), cpu_mm_closure_depth(op));
                        if (hr2 == 0xffffffffu) continue;
                        Count* j = cpu_mm_main_ptr_row0(
                            main_auth, mm, logical, main_row0[dbid], dbid, dst, hr2);
                        *j = cpu_low_add(*j, c, mod);
                    }
                }
            }
        }
    }

    stats.kernel_s += ram_seconds_since(t0);
    ++stats.groups;
}

struct CpuLowMaskMajorPool {
    int workers = 1;
    std::vector<CpuLowMaskMajorStats> stats;
    double wall_s = 0.0;

    explicit CpuLowMaskMajorPool(int n)
        : workers(std::max(1, n)), stats(size_t(std::max(1, n))) {}

    void run(
        const std::vector<CpuLowJob>& jobs,
        RamCounts& main_auth, RamCounts& block_auth,
        const StorageFactorHost& storage, const StorageLayout& logical,
        const LowMaskMajorLayout& mm, const CpuLowMaskSparseHost& sparse, Count mod
    ) {
        auto t0 = std::chrono::steady_clock::now();
        std::atomic<size_t> next{0};
        std::vector<std::thread> ts;
        ts.reserve(workers);
        for (int w = 0; w < workers; ++w) {
            ts.emplace_back([&, w] {
                for (;;) {
                    size_t q = next.fetch_add(1, std::memory_order_relaxed);
                    if (q >= jobs.size()) break;
                    if (!jobs[q].main_size && !jobs[q].block_size) continue;
                    process_cpu_low_group_maskmajor(
                        stats[size_t(w)], jobs[q], main_auth, block_auth,
                        storage, logical, mm, sparse, mod);
                }
            });
        }
        for (auto& t : ts) t.join();
        wall_s += ram_seconds_since(t0);
    }

    double kernel_s() const { double z=0; for (const auto& s:stats) z+=s.kernel_s; return z; }
    uint64_t groups() const { uint64_t z=0; for (const auto& s:stats) z+=s.groups; return z; }
    uint64_t orbit_columns() const { uint64_t z=0; for (const auto& s:stats) z+=s.orbit_columns; return z; }
    uint64_t closure_columns() const { uint64_t z=0; for (const auto& s:stats) z+=s.closure_columns; return z; }
};
