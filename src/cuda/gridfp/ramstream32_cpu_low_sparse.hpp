#pragma once

#include "ramstream32_cpu_low_direct.hpp"

#include <cstdint>
#include <iostream>
#include <vector>

// Sparse CPU instruction stream for the LOW window.  The dense orbit table is
// convenient for construction/proofs, but direct authoritative execution does
// not need to inspect states whose local pair has no action.  Compress only the
// N* orbit representatives and LL/RR/RL closure states into per-(p,FBlock)
// streams.

struct CpuLowSparseOrbitOp {
    // a: src_lr[19:0], kind[22:20], jblock[28:23]
    // b: jlr[19:0], dblock[25:20]
    // c: dlr[19:0]
    uint32_t a = 0, b = 0, c = 0;
};

static inline CpuLowSparseOrbitOp cpu_sparse_orbit_pack(
    uint32_t src_lr, uint32_t kind, uint32_t jblock, uint32_t jlr,
    uint32_t dblock, uint32_t dlr
) {
    if (src_lr > CPU_ORBIT_LR_MASK || jlr > CPU_ORBIT_LR_MASK
        || dlr > CPU_ORBIT_LR_MASK || kind > 7
        || jblock > CPU_ORBIT_BLOCK_MASK || dblock > CPU_ORBIT_BLOCK_MASK) {
        std::cerr << "cpu sparse orbit encoding overflow\n";
        std::exit(100);
    }
    CpuLowSparseOrbitOp z;
    z.a = src_lr | (kind << 20) | (jblock << 23);
    z.b = jlr | (dblock << 20);
    z.c = dlr;
    return z;
}
static inline uint32_t cpu_sparse_src(const CpuLowSparseOrbitOp& z) { return z.a & ((1u<<20)-1u); }
static inline uint32_t cpu_sparse_kind(const CpuLowSparseOrbitOp& z) { return (z.a >> 20) & 7u; }
static inline uint32_t cpu_sparse_jblock(const CpuLowSparseOrbitOp& z) { return (z.a >> 23) & 0x3fu; }
static inline uint32_t cpu_sparse_jlr(const CpuLowSparseOrbitOp& z) { return z.b & ((1u<<20)-1u); }
static inline uint32_t cpu_sparse_dblock(const CpuLowSparseOrbitOp& z) { return (z.b >> 20) & 0x3fu; }
static inline uint32_t cpu_sparse_dlr(const CpuLowSparseOrbitOp& z) { return z.c & ((1u<<20)-1u); }

struct CpuLowSparseHost {
    std::vector<CpuLowSparseOrbitOp> orbit_ops;
    // closure op: low 20 bits = source lr; bits 20..51 = lowdesc word.
    std::vector<uint64_t> closure_ops;
    // flattened [pi * (nblocks+1) + bid]
    std::vector<uint32_t> orbit_off;
    std::vector<uint32_t> closure_off;
    uint32_t nblocks = 0;
};

static inline uint64_t cpu_sparse_closure_pack(uint32_t src_lr, uint32_t desc) {
    if (src_lr >= (1u << 20)) {
        std::cerr << "cpu sparse closure rank overflow\n";
        std::exit(101);
    }
    return uint64_t(src_lr) | (uint64_t(desc) << 20);
}
static inline uint32_t cpu_sparse_closure_src(uint64_t z) { return uint32_t(z & ((1ull<<20)-1)); }
static inline uint32_t cpu_sparse_closure_desc(uint64_t z) { return uint32_t(z >> 20); }

static CpuLowSparseHost build_cpu_low_sparse(
    const StorageLayout& layout, const LowDescHost& desc, const LowOrbitHost& orbit
) {
    CpuLowSparseHost s;
    s.nblocks = uint32_t(layout.main_blocks.size());
    size_t pitch = size_t(s.nblocks) + 1;
    s.orbit_off.resize(size_t(LOW_LUT_K) * pitch);
    s.closure_off.resize(size_t(LOW_LUT_K) * pitch);

    for (int p = LOW_LUT_K; p >= 1; --p) {
        uint32_t pi = uint32_t(LOW_LUT_K - p);
        for (uint32_t bid = 0; bid < s.nblocks; ++bid) {
            s.orbit_off[size_t(pi) * pitch + bid] = uint32_t(s.orbit_ops.size());
            s.closure_off[size_t(pi) * pitch + bid] = uint32_t(s.closure_ops.size());
            uint32_t cols = layout.main_blocks[bid].cols;
            for (uint32_t lr = 0; lr < cols; ++lr) {
                uint64_t ow = orbit.rec[
                    size_t(pi) * orbit.main_total + orbit.main_base[bid] + lr];
                uint32_t k = cpu_orbit_kind(ow);
                if (k >= CPU_ORBIT_NN && k <= CPU_ORBIT_NL) {
                    s.orbit_ops.push_back(cpu_sparse_orbit_pack(
                        lr, k, cpu_orbit_jblock(ow), cpu_orbit_jlr(ow),
                        cpu_orbit_dblock(ow), cpu_orbit_dlr(ow)));
                } else if (k == CPU_ORBIT_CLOSURE) {
                    uint32_t dw = desc.main_desc[
                        size_t(pi) * desc.main_total + desc.main_base[bid] + lr];
                    // Invalid closure states are harmless, but dropping them
                    // reduces both metadata and runtime branches.
                    if (cpu_low_kind(dw) != LOWDESC_INVALID)
                        s.closure_ops.push_back(cpu_sparse_closure_pack(lr, dw));
                }
            }
        }
        s.orbit_off[size_t(pi) * pitch + s.nblocks] = uint32_t(s.orbit_ops.size());
        s.closure_off[size_t(pi) * pitch + s.nblocks] = uint32_t(s.closure_ops.size());
    }

    std::cerr << "cpu_low_sparse orbit_ops=" << s.orbit_ops.size()
              << " closure_ops=" << s.closure_ops.size()
              << " orbit_mib=" << double(s.orbit_ops.size() * sizeof(CpuLowSparseOrbitOp)) / (1<<20)
              << " closure_mib=" << double(s.closure_ops.size() * sizeof(uint64_t)) / (1<<20)
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
            auto [oa, ob] = cpu_sparse_range(sparse.orbit_off, sparse.nblocks, pi, bid);
            if (oa == ob) continue;
            Code rows = (x.end - x.off) / x.stride;
            for (Code hr = 0; hr < rows; ++hr) {
                Count* xr = xb + hr * x.stride;
                for (uint32_t q = oa; q < ob; ++q) {
                    const CpuLowSparseOrbitOp& op = sparse.orbit_ops[q];
                    uint32_t kind = cpu_sparse_kind(op);
                    Count* ip = xr + cpu_sparse_src(op);
                    uint32_t jbid = cpu_sparse_jblock(op);
                    uint32_t dbid = cpu_sparse_dblock(op);
                    Count* jp = mp[jbid] + hr * job.main_blocks[jbid].stride + cpu_sparse_jlr(op);
                    Count* dd = dp[dbid] + hr * job.block_blocks[dbid].stride + cpu_sparse_dlr(op);
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

        for (uint32_t bid = 0; bid < sparse.nblocks; ++bid) {
            const FBlock& x = job.main_blocks[bid];
            Count* xb = mp[bid];
            if (!xb || !x.stride) continue;
            auto [ca, cb] = cpu_sparse_range(sparse.closure_off, sparse.nblocks, pi, bid);
            if (ca == cb) continue;
            Code rows = (x.end - x.off) / x.stride;
            uint32_t high0 = G_FACTOR.high_mask_off[
                size_t(job.mask) * FactorTablesHost::STRIDE + x.he];
            for (Code hr = 0; hr < rows; ++hr) {
                Count* xr = xb + hr * x.stride;
                for (uint32_t q = ca; q < cb; ++q) {
                    uint64_t op = sparse.closure_ops[q];
                    uint32_t src_lr = cpu_sparse_closure_src(op);
                    Count c = xr[src_lr];
                    if (!c) continue;
                    uint32_t word = cpu_sparse_closure_desc(op);
                    uint32_t kind = cpu_low_kind(word);
                    if (kind == LOWDESC_MAIN) {
                        uint32_t jbid = cpu_low_block(word);
                        Count* j = mp[jbid] + hr * job.main_blocks[jbid].stride + cpu_low_lr(word);
                        *j = cpu_low_add(*j, c, mod);
                    } else if (kind == LOWDESC_BLOCK) {
                        uint32_t dbid = cpu_low_block(word);
                        Count* j = dp[dbid] + hr * job.block_blocks[dbid].stride + cpu_low_lr(word);
                        *j = cpu_low_add(*j, c, mod);
                    } else if (kind == LOWDESC_CROSS) {
                        uint32_t hc = G_FACTOR.high_mask_codes[high0 + hr];
                        uint32_t hc2 = cpu_low_flip_high(hc, cpu_low_depth(word));
                        if (hc2 == 0xffffffffu) continue;
                        if (p == 1) {
                            uint32_t jbid = cpu_low_block(word);
                            const FBlock& y = job.main_blocks[jbid];
                            uint32_t hr2 = cpu_high_mask_rank(job.mask, hc2, y.he);
                            if (hr2 == 0xffffffffu) continue;
                            Count* j = mp[jbid] + Code(hr2) * y.stride + cpu_low_lr(word);
                            *j = cpu_low_add(*j, c, mod);
                        } else {
                            uint32_t dbid = cpu_low_block(word);
                            const FBlock& y = job.block_blocks[dbid];
                            uint32_t hr2 = cpu_high_mask_rank(job.mask, hc2, y.he);
                            if (hr2 == 0xffffffffu) continue;
                            Count* j = dp[dbid] + Code(hr2) * y.stride + cpu_low_lr(word);
                            *j = cpu_low_add(*j, c, mod);
                        }
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
