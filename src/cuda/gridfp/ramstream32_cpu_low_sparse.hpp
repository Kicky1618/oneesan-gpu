#pragma once

#include "ramstream32_cpu_low_direct.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

// Sparse CPU instruction stream for the LOW window. The dense orbit table is
// convenient for construction/proofs, but direct authoritative execution does
// not need to inspect states whose local pair has no action. Compress only the
// N* orbit representatives and LL/RR/RL closure states into per-(p,FBlock)
// streams.
//
// v4.4 compacts an orbit operation from 12 to 8 bytes. The source block and p
// already determine both destination factor blocks.
//
// v4.5 pre-ranks LOWDESC_CROSS on the inactive HIGH half, replacing an O(H)
// bracket scan plus binary search with one uint16_t lookup.
//
// v4.6 separates closure operations into LOCAL and CROSS streams. Local
// destination storage is determined entirely by p: p=1 writes main, p>1 writes
// blocked. CROSS operations run in a separate branch-free loop.
//
// v4.7 separates N* orbit representatives into NN, NR and NL streams. These
// three classes are disjoint local orbits: the partner states are LR, RN and LN
// respectively, none of which is another N* representative. Reordering the
// representatives by class therefore preserves the in-place orbit algebra.
// Runtime no longer decodes orbit kind or branches on it. The p==1 split for
// NR/NL is also hoisted outside the operation loop.
using CpuLowSparseOrbitOp = uint64_t;
using CpuLowSparseClosureOp = uint64_t;
static_assert(sizeof(CpuLowSparseOrbitOp) == 8);
static_assert(sizeof(CpuLowSparseClosureOp) == 8);
static_assert(HIGH_LUT_K <= 15,
              "uint16_t sparse CROSS rank requires HIGH_LUT_K <= 15");

static constexpr uint64_t CPU_SPARSE_RANK_MASK = (1ull << 20) - 1ull;
static constexpr int CPU_SPARSE_JLR_SHIFT = 20;
static constexpr int CPU_SPARSE_DLR_SHIFT = 40;
static constexpr int CPU_SPARSE_KIND_SHIFT = 60;
static constexpr uint16_t CPU_SPARSE_CROSS_INVALID = 0xffffu;

static constexpr int CPU_SPARSE_CLOSURE_DST_LR_SHIFT = 20;
static constexpr int CPU_SPARSE_CLOSURE_BLOCK_SHIFT = 40;
static constexpr int CPU_SPARSE_CLOSURE_DEPTH_SHIFT = 46;

static inline CpuLowSparseOrbitOp cpu_sparse_orbit_pack(
    uint32_t src_lr, uint32_t kind, uint32_t jlr, uint32_t dlr
) {
    if (src_lr > CPU_ORBIT_LR_MASK || jlr > CPU_ORBIT_LR_MASK
        || dlr > CPU_ORBIT_LR_MASK
        || kind < CPU_ORBIT_NN || kind > CPU_ORBIT_NL) {
        std::cerr << "cpu sparse orbit encoding overflow\n";
        std::exit(100);
    }
    uint64_t kind2 = uint64_t(kind - CPU_ORBIT_NN);
    return uint64_t(src_lr)
        | (uint64_t(jlr) << CPU_SPARSE_JLR_SHIFT)
        | (uint64_t(dlr) << CPU_SPARSE_DLR_SHIFT)
        | (kind2 << CPU_SPARSE_KIND_SHIFT);
}
static inline uint32_t cpu_sparse_src(CpuLowSparseOrbitOp z) {
    return uint32_t(z & CPU_SPARSE_RANK_MASK);
}
static inline uint32_t cpu_sparse_jlr(CpuLowSparseOrbitOp z) {
    return uint32_t((z >> CPU_SPARSE_JLR_SHIFT) & CPU_SPARSE_RANK_MASK);
}
static inline uint32_t cpu_sparse_dlr(CpuLowSparseOrbitOp z) {
    return uint32_t((z >> CPU_SPARSE_DLR_SHIFT) & CPU_SPARSE_RANK_MASK);
}
static inline uint32_t cpu_sparse_kind(CpuLowSparseOrbitOp z) {
    return uint32_t((z >> CPU_SPARSE_KIND_SHIFT) & 3u) + CPU_ORBIT_NN;
}

static inline uint32_t cpu_sparse_jblock(
    uint32_t source_bid, const FBlock& source, int p, uint32_t kind
) {
    if (p != LOW_LUT_K) return source_bid;
    uint32_t center = (kind == CPU_ORBIT_NR) ? uint32_t(R) : uint32_t(::L);
    return 3u * uint32_t(source.he) + center;
}

static inline CpuLowSparseClosureOp cpu_sparse_closure_pack(
    uint32_t src_lr, uint32_t block, uint32_t dst_lr, uint32_t depth = 0
) {
    if (src_lr > CPU_SPARSE_RANK_MASK || dst_lr > CPU_SPARSE_RANK_MASK
        || block > LOWDESC_BLOCK_MASK || depth > LOWDESC_DEPTH_MASK) {
        std::cerr << "cpu sparse closure encoding overflow src=" << src_lr
                  << " block=" << block << " dst=" << dst_lr
                  << " depth=" << depth << '\n';
        std::exit(101);
    }
    return uint64_t(src_lr)
        | (uint64_t(dst_lr) << CPU_SPARSE_CLOSURE_DST_LR_SHIFT)
        | (uint64_t(block) << CPU_SPARSE_CLOSURE_BLOCK_SHIFT)
        | (uint64_t(depth) << CPU_SPARSE_CLOSURE_DEPTH_SHIFT);
}
static inline uint32_t cpu_sparse_closure_src(CpuLowSparseClosureOp z) {
    return uint32_t(z & CPU_SPARSE_RANK_MASK);
}
static inline uint32_t cpu_sparse_closure_lr(CpuLowSparseClosureOp z) {
    return uint32_t((z >> CPU_SPARSE_CLOSURE_DST_LR_SHIFT) & CPU_SPARSE_RANK_MASK);
}
static inline uint32_t cpu_sparse_closure_block(CpuLowSparseClosureOp z) {
    return uint32_t((z >> CPU_SPARSE_CLOSURE_BLOCK_SHIFT) & LOWDESC_BLOCK_MASK);
}
static inline uint32_t cpu_sparse_closure_depth(CpuLowSparseClosureOp z) {
    return uint32_t((z >> CPU_SPARSE_CLOSURE_DEPTH_SHIFT) & LOWDESC_DEPTH_MASK);
}

struct CpuLowSparseHost {
    // v4.7 split orbit streams. The legacy aggregate vector is intentionally
    // kept empty for source compatibility with older probes; production code
    // uses the three split streams below.
    std::vector<CpuLowSparseOrbitOp> orbit_ops;
    std::vector<CpuLowSparseOrbitOp> nn_orbit_ops;
    std::vector<CpuLowSparseOrbitOp> nr_orbit_ops;
    std::vector<CpuLowSparseOrbitOp> nl_orbit_ops;
    std::vector<CpuLowSparseClosureOp> local_closure_ops;
    std::vector<CpuLowSparseClosureOp> cross_closure_ops;

    // flattened [pi * (nblocks+1) + bid]
    std::vector<uint32_t> nn_orbit_off;
    std::vector<uint32_t> nr_orbit_off;
    std::vector<uint32_t> nl_orbit_off;
    std::vector<uint32_t> local_closure_off;
    std::vector<uint32_t> cross_closure_off;
    uint32_t nblocks = 0;

    // flattened [(depth-1) * high_cross_pitch + global high-mask-code index].
    // Value is the target mask-local HIGH rank, or 0xffff when no matching L
    // exists. G_FACTOR.high_mask_codes supplies the source indexing.
    std::vector<uint16_t> high_cross_rank;
    uint32_t high_cross_pitch = 0;

    size_t orbit_count() const {
        return nn_orbit_ops.size() + nr_orbit_ops.size() + nl_orbit_ops.size();
    }
};

static CpuLowSparseHost build_cpu_low_sparse(
    const StorageFactorHost& storage, const StorageLayout& layout,
    const LowDescHost& desc, const LowOrbitHost& orbit
) {
    CpuLowSparseHost s;
    s.nblocks = uint32_t(layout.main_blocks.size());
    size_t pitch = size_t(s.nblocks) + 1;
    size_t noff = size_t(LOW_LUT_K) * pitch;
    s.nn_orbit_off.resize(noff);
    s.nr_orbit_off.resize(noff);
    s.nl_orbit_off.resize(noff);
    s.local_closure_off.resize(noff);
    s.cross_closure_off.resize(noff);

    for (int p = LOW_LUT_K; p >= 1; --p) {
        uint32_t pi = uint32_t(LOW_LUT_K - p);
        for (uint32_t bid = 0; bid < s.nblocks; ++bid) {
            size_t oi = size_t(pi) * pitch + bid;
            s.nn_orbit_off[oi] = uint32_t(s.nn_orbit_ops.size());
            s.nr_orbit_off[oi] = uint32_t(s.nr_orbit_ops.size());
            s.nl_orbit_off[oi] = uint32_t(s.nl_orbit_ops.size());
            s.local_closure_off[oi] = uint32_t(s.local_closure_ops.size());
            s.cross_closure_off[oi] = uint32_t(s.cross_closure_ops.size());

            uint32_t cols = layout.main_blocks[bid].cols;
            for (uint32_t lr = 0; lr < cols; ++lr) {
                uint64_t ow = orbit.rec[
                    size_t(pi) * orbit.main_total + orbit.main_base[bid] + lr];
                uint32_t k = cpu_orbit_kind(ow);
                if (k >= CPU_ORBIT_NN && k <= CPU_ORBIT_NL) {
                    const StorageBlock& sb = layout.main_blocks[bid];
                    FBlock source{};
                    source.he = sb.he;
                    uint32_t derived_j = cpu_sparse_jblock(bid, source, p, k);
                    uint32_t derived_d = uint32_t(sb.he);
                    if (derived_j != cpu_orbit_jblock(ow)
                        || derived_d != cpu_orbit_dblock(ow)) {
                        std::cerr << "cpu sparse derived block mismatch p=" << p
                                  << " bid=" << bid << " kind=" << k
                                  << " j=" << derived_j << "/" << cpu_orbit_jblock(ow)
                                  << " d=" << derived_d << "/" << cpu_orbit_dblock(ow)
                                  << '\n';
                        std::exit(102);
                    }
                    CpuLowSparseOrbitOp op = cpu_sparse_orbit_pack(
                        lr, k, cpu_orbit_jlr(ow), cpu_orbit_dlr(ow));
                    if (k == CPU_ORBIT_NN) s.nn_orbit_ops.push_back(op);
                    else if (k == CPU_ORBIT_NR) s.nr_orbit_ops.push_back(op);
                    else s.nl_orbit_ops.push_back(op);
                } else if (k == CPU_ORBIT_CLOSURE) {
                    uint32_t dw = desc.main_desc[
                        size_t(pi) * desc.main_total + desc.main_base[bid] + lr];
                    uint32_t kind = cpu_low_kind(dw);
                    if (kind == LOWDESC_INVALID) continue;
                    if (kind == LOWDESC_CROSS) {
                        uint32_t depth = cpu_low_depth(dw);
                        if (!depth) {
                            std::cerr << "cpu sparse CROSS with zero depth p=" << p
                                      << " bid=" << bid << " lr=" << lr << '\n';
                            std::exit(104);
                        }
                        s.cross_closure_ops.push_back(cpu_sparse_closure_pack(
                            lr, cpu_low_block(dw), cpu_low_lr(dw), depth));
                    } else {
                        uint32_t expected = p == 1 ? LOWDESC_MAIN : LOWDESC_BLOCK;
                        if (kind != expected) {
                            std::cerr << "cpu sparse local closure kind mismatch p=" << p
                                      << " bid=" << bid << " lr=" << lr
                                      << " kind=" << kind << " expected=" << expected << '\n';
                            std::exit(105);
                        }
                        s.local_closure_ops.push_back(cpu_sparse_closure_pack(
                            lr, cpu_low_block(dw), cpu_low_lr(dw)));
                    }
                }
            }
        }
        size_t end = size_t(pi) * pitch + s.nblocks;
        s.nn_orbit_off[end] = uint32_t(s.nn_orbit_ops.size());
        s.nr_orbit_off[end] = uint32_t(s.nr_orbit_ops.size());
        s.nl_orbit_off[end] = uint32_t(s.nl_orbit_ops.size());
        s.local_closure_off[end] = uint32_t(s.local_closure_ops.size());
        s.cross_closure_off[end] = uint32_t(s.cross_closure_ops.size());
    }

    // Pre-rank every possible inactive-HIGH CROSS operation. The dense storage
    // rank is still available here and already stores the mask-local rank in
    // its low HIGH_LUT_K bits. Flipping L->R preserves occupancy, so no mask
    // conversion is necessary.
    s.high_cross_pitch = uint32_t(G_FACTOR.high_mask_codes.size());
    s.high_cross_rank.assign(
        size_t(HIGH_LUT_K) * s.high_cross_pitch, CPU_SPARSE_CROSS_INVALID);
    constexpr uint32_t HIGH_MASK_RANK_MASK = (1u << HIGH_LUT_K) - 1u;
    for (uint32_t depth = 1; depth <= uint32_t(HIGH_LUT_K); ++depth) {
        uint16_t* dst = s.high_cross_rank.data()
            + size_t(depth - 1) * s.high_cross_pitch;
        for (uint32_t i = 0; i < s.high_cross_pitch; ++i) {
            uint32_t hc = G_FACTOR.high_mask_codes[i];
            uint32_t hc2 = cpu_low_flip_high(hc, depth);
            if (hc2 == 0xffffffffu) continue;
            uint32_t packed = storage.high_packed_rank[hc2];
            if (packed == 0xffffffffu) continue;
            uint32_t hr2 = packed & HIGH_MASK_RANK_MASK;
            if (hr2 >= uint32_t(CPU_SPARSE_CROSS_INVALID)) {
                std::cerr << "cpu sparse cross rank overflow depth=" << depth
                          << " rank=" << hr2 << '\n';
                std::exit(103);
            }
            dst[i] = uint16_t(hr2);
        }
    }

    std::cerr << "cpu_low_sparse"
              << " nn_orbit_ops=" << s.nn_orbit_ops.size()
              << " nr_orbit_ops=" << s.nr_orbit_ops.size()
              << " nl_orbit_ops=" << s.nl_orbit_ops.size()
              << " orbit_ops=" << s.orbit_count()
              << " local_closure_ops=" << s.local_closure_ops.size()
              << " cross_closure_ops=" << s.cross_closure_ops.size()
              << " orbit_op_bytes=" << sizeof(CpuLowSparseOrbitOp)
              << " closure_op_bytes=" << sizeof(CpuLowSparseClosureOp)
              << " orbit_mib="
              << double(s.orbit_count() * sizeof(CpuLowSparseOrbitOp)) / (1<<20)
              << " local_closure_mib="
              << double(s.local_closure_ops.size() * sizeof(CpuLowSparseClosureOp)) / (1<<20)
              << " cross_closure_mib="
              << double(s.cross_closure_ops.size() * sizeof(CpuLowSparseClosureOp)) / (1<<20)
              << " cross_rank_mib="
              << double(s.high_cross_rank.size() * sizeof(uint16_t)) / (1<<20)
              << '\n';
    return s;
}

static inline std::pair<uint32_t,uint32_t> cpu_sparse_range(
    const std::vector<uint32_t>& off, uint32_t nblocks, uint32_t pi, uint32_t bid
) {
    size_t pitch = size_t(nblocks) + 1;
    return {off[size_t(pi)*pitch+bid], off[size_t(pi)*pitch+bid+1]};
}

struct CpuLowSparseStats {
    double kernel_s = 0.0;
    uint64_t groups = 0;
};

static void process_cpu_low_group_sparse(
    CpuLowSparseStats& stats, const CpuLowJob& job,
    RamCounts& main_auth, RamCounts& block_auth,
    const StorageFactorHost& storage, const StorageLayout& layout,
    const CpuLowSparseHost& sparse, Count mod
) {
    if (!job.main_size && !job.block_size) return;
    std::vector<Count*> mp(job.main_blocks.size(), nullptr);
    std::vector<Count*> dp(job.block_blocks.size(), nullptr);
    for (size_t bid = 0; bid < job.main_blocks.size(); ++bid)
        mp[bid] = cpu_low_direct_block_ptr(
            main_auth, layout.main_blocks[bid], job.main_blocks[bid], job.mask, storage);
    for (size_t bid = 0; bid < job.block_blocks.size(); ++bid)
        dp[bid] = cpu_low_direct_block_ptr(
            block_auth, layout.block_blocks[bid], job.block_blocks[bid], job.mask, storage);

    auto t0 = std::chrono::steady_clock::now();
    for (int p = LOW_LUT_K; p >= 1; --p) {
        uint32_t pi = uint32_t(LOW_LUT_K - p);

        for (uint32_t bid = 0; bid < sparse.nblocks; ++bid) {
            const FBlock& x = job.main_blocks[bid];
            Count* xb = mp[bid];
            if (!xb || !x.stride) continue;
            uint32_t dbid = uint32_t(x.he);
            Code rows = (x.end - x.off) / x.stride;

            auto [na, nb] = cpu_sparse_range(
                sparse.nn_orbit_off, sparse.nblocks, pi, bid);
            if (na != nb) {
                uint32_t jbid = cpu_sparse_jblock(bid, x, p, CPU_ORBIT_NN);
                const FBlock& jy = job.main_blocks[jbid];
                const FBlock& dy = job.block_blocks[dbid];
                for (Code hr = 0; hr < rows; ++hr) {
                    Count* xr = xb + hr * x.stride;
                    Count* jr = mp[jbid] + hr * jy.stride;
                    Count* dr = dp[dbid] + hr * dy.stride;
                    for (uint32_t q = na; q < nb; ++q) {
                        CpuLowSparseOrbitOp op = sparse.nn_orbit_ops[q];
                        Count* ip = xr + cpu_sparse_src(op);
                        Count* jp = jr + cpu_sparse_jlr(op);
                        Count* dd = dr + cpu_sparse_dlr(op);
                        Count c = *ip;
                        Count d = *dd;
                        *jp = cpu_low_add(*jp, c, mod);
                        *ip = cpu_low_add(c, d, mod);
                        *dd = 0;
                    }
                }
            }

            auto [ra, rb] = cpu_sparse_range(
                sparse.nr_orbit_off, sparse.nblocks, pi, bid);
            auto [la, lb] = cpu_sparse_range(
                sparse.nl_orbit_off, sparse.nblocks, pi, bid);
            if (p == 1) {
                if (ra != rb) {
                    uint32_t jbid = cpu_sparse_jblock(bid, x, p, CPU_ORBIT_NR);
                    const FBlock& jy = job.main_blocks[jbid];
                    const FBlock& dy = job.block_blocks[dbid];
                    for (Code hr = 0; hr < rows; ++hr) {
                        Count* xr = xb + hr * x.stride;
                        Count* jr = mp[jbid] + hr * jy.stride;
                        Count* dr = dp[dbid] + hr * dy.stride;
                        for (uint32_t q = ra; q < rb; ++q) {
                            CpuLowSparseOrbitOp op = sparse.nr_orbit_ops[q];
                            Count* ip = xr + cpu_sparse_src(op);
                            Count* jp = jr + cpu_sparse_jlr(op);
                            Count* dd = dr + cpu_sparse_dlr(op);
                            Count c = *ip;
                            Count cc = *jp;
                            Count d = *dd;
                            *ip = cpu_low_add(cpu_low_add(c, cc, mod), d, mod);
                            *jp = cpu_low_add(c, cc, mod);
                            *dd = 0;
                        }
                    }
                }
                if (la != lb) {
                    uint32_t jbid = cpu_sparse_jblock(bid, x, p, CPU_ORBIT_NL);
                    const FBlock& jy = job.main_blocks[jbid];
                    const FBlock& dy = job.block_blocks[dbid];
                    for (Code hr = 0; hr < rows; ++hr) {
                        Count* xr = xb + hr * x.stride;
                        Count* jr = mp[jbid] + hr * jy.stride;
                        Count* dr = dp[dbid] + hr * dy.stride;
                        for (uint32_t q = la; q < lb; ++q) {
                            CpuLowSparseOrbitOp op = sparse.nl_orbit_ops[q];
                            Count* ip = xr + cpu_sparse_src(op);
                            Count* jp = jr + cpu_sparse_jlr(op);
                            Count* dd = dr + cpu_sparse_dlr(op);
                            Count c = *ip;
                            Count cc = *jp;
                            Count d = *dd;
                            *ip = cpu_low_add(cpu_low_add(c, cc, mod), d, mod);
                            *jp = cpu_low_add(c, cc, mod);
                            *dd = 0;
                        }
                    }
                }
            } else {
                if (ra != rb) {
                    uint32_t jbid = cpu_sparse_jblock(bid, x, p, CPU_ORBIT_NR);
                    const FBlock& jy = job.main_blocks[jbid];
                    const FBlock& dy = job.block_blocks[dbid];
                    for (Code hr = 0; hr < rows; ++hr) {
                        Count* xr = xb + hr * x.stride;
                        Count* jr = mp[jbid] + hr * jy.stride;
                        Count* dr = dp[dbid] + hr * dy.stride;
                        for (uint32_t q = ra; q < rb; ++q) {
                            CpuLowSparseOrbitOp op = sparse.nr_orbit_ops[q];
                            Count* ip = xr + cpu_sparse_src(op);
                            Count* jp = jr + cpu_sparse_jlr(op);
                            Count* dd = dr + cpu_sparse_dlr(op);
                            Count c = *ip;
                            Count cc = *jp;
                            Count d = *dd;
                            *ip = cpu_low_add(cpu_low_add(c, cc, mod), d, mod);
                            *dd = c;
                        }
                    }
                }
                if (la != lb) {
                    uint32_t jbid = cpu_sparse_jblock(bid, x, p, CPU_ORBIT_NL);
                    const FBlock& jy = job.main_blocks[jbid];
                    const FBlock& dy = job.block_blocks[dbid];
                    for (Code hr = 0; hr < rows; ++hr) {
                        Count* xr = xb + hr * x.stride;
                        Count* jr = mp[jbid] + hr * jy.stride;
                        Count* dr = dp[dbid] + hr * dy.stride;
                        for (uint32_t q = la; q < lb; ++q) {
                            CpuLowSparseOrbitOp op = sparse.nl_orbit_ops[q];
                            Count* ip = xr + cpu_sparse_src(op);
                            Count* jp = jr + cpu_sparse_jlr(op);
                            Count* dd = dr + cpu_sparse_dlr(op);
                            Count c = *ip;
                            Count cc = *jp;
                            Count d = *dd;
                            *ip = cpu_low_add(cpu_low_add(c, cc, mod), d, mod);
                            *dd = c;
                        }
                    }
                }
            }
        }

        // Local closures are branch-free with respect to destination kind.
        // The builder proved p=1 -> main and p>1 -> blocked.
        if (p == 1) {
            for (uint32_t bid = 0; bid < sparse.nblocks; ++bid) {
                const FBlock& x = job.main_blocks[bid];
                Count* xb = mp[bid];
                if (!xb || !x.stride) continue;
                auto [ca, cb] = cpu_sparse_range(
                    sparse.local_closure_off, sparse.nblocks, pi, bid);
                if (ca == cb) continue;
                Code rows = (x.end - x.off) / x.stride;
                for (Code hr = 0; hr < rows; ++hr) {
                    Count* xr = xb + hr * x.stride;
                    for (uint32_t q = ca; q < cb; ++q) {
                        CpuLowSparseClosureOp op = sparse.local_closure_ops[q];
                        Count c = xr[cpu_sparse_closure_src(op)];
                        if (!c) continue;
                        uint32_t jbid = cpu_sparse_closure_block(op);
                        Count* j = mp[jbid] + hr * job.main_blocks[jbid].stride
                            + cpu_sparse_closure_lr(op);
                        *j = cpu_low_add(*j, c, mod);
                    }
                }
            }
        } else {
            for (uint32_t bid = 0; bid < sparse.nblocks; ++bid) {
                const FBlock& x = job.main_blocks[bid];
                Count* xb = mp[bid];
                if (!xb || !x.stride) continue;
                auto [ca, cb] = cpu_sparse_range(
                    sparse.local_closure_off, sparse.nblocks, pi, bid);
                if (ca == cb) continue;
                Code rows = (x.end - x.off) / x.stride;
                for (Code hr = 0; hr < rows; ++hr) {
                    Count* xr = xb + hr * x.stride;
                    for (uint32_t q = ca; q < cb; ++q) {
                        CpuLowSparseClosureOp op = sparse.local_closure_ops[q];
                        Count c = xr[cpu_sparse_closure_src(op)];
                        if (!c) continue;
                        uint32_t dbid = cpu_sparse_closure_block(op);
                        Count* j = dp[dbid] + hr * job.block_blocks[dbid].stride
                            + cpu_sparse_closure_lr(op);
                        *j = cpu_low_add(*j, c, mod);
                    }
                }
            }
        }

        // CROSS closures use the pre-ranked HIGH table and contain no kind
        // dispatch. Hoist p=1 vs p>1 outside the operation loop as well.
        if (p == 1) {
            for (uint32_t bid = 0; bid < sparse.nblocks; ++bid) {
                const FBlock& x = job.main_blocks[bid];
                Count* xb = mp[bid];
                if (!xb || !x.stride) continue;
                auto [ca, cb] = cpu_sparse_range(
                    sparse.cross_closure_off, sparse.nblocks, pi, bid);
                if (ca == cb) continue;
                Code rows = (x.end - x.off) / x.stride;
                uint32_t high0 = G_FACTOR.high_mask_off[
                    size_t(job.mask) * FactorTablesHost::STRIDE + x.he];
                for (Code hr = 0; hr < rows; ++hr) {
                    Count* xr = xb + hr * x.stride;
                    for (uint32_t q = ca; q < cb; ++q) {
                        CpuLowSparseClosureOp op = sparse.cross_closure_ops[q];
                        Count c = xr[cpu_sparse_closure_src(op)];
                        if (!c) continue;
                        uint32_t depth = cpu_sparse_closure_depth(op);
                        uint16_t hr16 = sparse.high_cross_rank[
                            size_t(depth - 1) * sparse.high_cross_pitch + high0 + hr];
                        if (hr16 == CPU_SPARSE_CROSS_INVALID) continue;
                        uint32_t jbid = cpu_sparse_closure_block(op);
                        const FBlock& y = job.main_blocks[jbid];
                        Count* j = mp[jbid] + Code(hr16) * y.stride
                            + cpu_sparse_closure_lr(op);
                        *j = cpu_low_add(*j, c, mod);
                    }
                }
            }
        } else {
            for (uint32_t bid = 0; bid < sparse.nblocks; ++bid) {
                const FBlock& x = job.main_blocks[bid];
                Count* xb = mp[bid];
                if (!xb || !x.stride) continue;
                auto [ca, cb] = cpu_sparse_range(
                    sparse.cross_closure_off, sparse.nblocks, pi, bid);
                if (ca == cb) continue;
                Code rows = (x.end - x.off) / x.stride;
                uint32_t high0 = G_FACTOR.high_mask_off[
                    size_t(job.mask) * FactorTablesHost::STRIDE + x.he];
                for (Code hr = 0; hr < rows; ++hr) {
                    Count* xr = xb + hr * x.stride;
                    for (uint32_t q = ca; q < cb; ++q) {
                        CpuLowSparseClosureOp op = sparse.cross_closure_ops[q];
                        Count c = xr[cpu_sparse_closure_src(op)];
                        if (!c) continue;
                        uint32_t depth = cpu_sparse_closure_depth(op);
                        uint16_t hr16 = sparse.high_cross_rank[
                            size_t(depth - 1) * sparse.high_cross_pitch + high0 + hr];
                        if (hr16 == CPU_SPARSE_CROSS_INVALID) continue;
                        uint32_t dbid = cpu_sparse_closure_block(op);
                        const FBlock& y = job.block_blocks[dbid];
                        Count* j = dp[dbid] + Code(hr16) * y.stride
                            + cpu_sparse_closure_lr(op);
                        *j = cpu_low_add(*j, c, mod);
                    }
                }
            }
        }
    }
    stats.kernel_s += ram_seconds_since(t0);
    ++stats.groups;
}

struct CpuLowSparsePool {
    int workers = 1;
    std::vector<CpuLowSparseStats> stats;
    double wall_s = 0.0;
    explicit CpuLowSparsePool(int n)
        : workers(std::max(1,n)), stats(size_t(std::max(1,n))) {}

    void run(
        const std::vector<CpuLowJob>& jobs,
        RamCounts& main_auth, RamCounts& block_auth,
        const StorageFactorHost& storage, const StorageLayout& layout,
        const CpuLowSparseHost& sparse, Count mod
    ) {
        auto t0 = std::chrono::steady_clock::now();
        std::atomic<size_t> next{0};
        std::vector<std::thread> ts;
        ts.reserve(workers);
        for (int w=0; w<workers; ++w) {
            ts.emplace_back([&,w]{
                for (;;) {
                    size_t q=next.fetch_add(1,std::memory_order_relaxed);
                    if (q>=jobs.size()) break;
                    if (!jobs[q].main_size && !jobs[q].block_size) continue;
                    process_cpu_low_group_sparse(
                        stats[w], jobs[q], main_auth, block_auth,
                        storage, layout, sparse, mod);
                }
            });
        }
        for(auto& t:ts)t.join();
        wall_s += ram_seconds_since(t0);
    }
    double kernel_s() const { double z=0; for(auto const&s:stats)z+=s.kernel_s; return z; }
    uint64_t groups() const { uint64_t z=0; for(auto const&s:stats)z+=s.groups; return z; }
};