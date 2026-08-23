#pragma once

#include "ramstream32_cpu_low_inplace.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

// Zero-scratch CPU LOW executor.  For the LOW+center window HIGH occupancy is
// fixed, therefore every factor block's group slice is already a contiguous row
// range in the authoritative HIGH x LOW storage matrix.  The in-place orbit is
// closed inside those slices, so copying to a temporary group buffer is
// unnecessary.

static Count* cpu_low_direct_block_ptr(
    RamCounts& auth, const StorageBlock& sb, const FBlock& fb,
    uint32_t mask, const StorageFactorHost& storage
) {
    if (fb.end == fb.off) return nullptr;
    constexpr int S = StorageFactorHost::S;
    uint32_t row0 = storage.high_mask_begin[size_t(mask) * S + fb.he];
    if (fb.stride != sb.cols) {
        std::cerr << "cpu-low direct stride mismatch factor=" << fb.stride
                  << " storage=" << sb.cols << '\n';
        std::exit(90);
    }
    return auth.ptr + sb.off + Code(row0) * sb.cols;
}

struct CpuLowDirectStats {
    double kernel_s = 0.0;
    uint64_t groups = 0;
};

static void process_cpu_low_group_direct(
    CpuLowDirectStats& stats, const CpuLowJob& job,
    RamCounts& main_auth, RamCounts& block_auth,
    const StorageFactorHost& storage, const StorageLayout& layout,
    const LowDescHost& desc, const LowOrbitHost& orbit, Count mod
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

        // Orbit pass.  Each source with p=N is the unique representative of
        // its local main/main/blocked orbit, so no two worker groups alias.
        for (size_t bid = 0; bid < job.main_blocks.size(); ++bid) {
            const FBlock& x = job.main_blocks[bid];
            Count* xb = mp[bid];
            if (!xb || !x.stride) continue;
            Code rows = (x.end - x.off) / x.stride;
            for (Code hr = 0; hr < rows; ++hr) {
                Count* xr = xb + hr * x.stride;
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    uint64_t ow = orbit.rec[
                        size_t(pi) * orbit.main_total + orbit.main_base[bid] + lr];
                    uint32_t kind = cpu_orbit_kind(ow);
                    if (kind < CPU_ORBIT_NN || kind > CPU_ORBIT_NL) continue;

                    uint32_t jbid = cpu_orbit_jblock(ow);
                    uint32_t dbid = cpu_orbit_dblock(ow);
                    Count* jp = mp[jbid] + hr * job.main_blocks[jbid].stride
                        + cpu_orbit_jlr(ow);
                    Count* dd = dp[dbid] + hr * job.block_blocks[dbid].stride
                        + cpu_orbit_dlr(ow);
                    Count* ip = xr + lr;
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

        // Closure pass.  It must follow the complete orbit pass because the
        // closure value is the post-orbit value for this edge position.
        for (size_t bid = 0; bid < job.main_blocks.size(); ++bid) {
            const FBlock& x = job.main_blocks[bid];
            Count* xb = mp[bid];
            if (!xb || !x.stride) continue;
            Code rows = (x.end - x.off) / x.stride;
            uint32_t high0 = G_FACTOR.high_mask_off[
                size_t(job.mask) * FactorTablesHost::STRIDE + x.he];
            for (Code hr = 0; hr < rows; ++hr) {
                Count* xr = xb + hr * x.stride;
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    uint64_t ow = orbit.rec[
                        size_t(pi) * orbit.main_total + orbit.main_base[bid] + lr];
                    if (cpu_orbit_kind(ow) != CPU_ORBIT_CLOSURE) continue;
                    Count c = xr[lr];
                    if (!c) continue;
                    uint32_t word = desc.main_desc[
                        size_t(pi) * desc.main_total + desc.main_base[bid] + lr];
                    uint32_t kind = cpu_low_kind(word);
                    if (kind == LOWDESC_MAIN) {
                        uint32_t jbid = cpu_low_block(word);
                        Count* j = mp[jbid] + hr * job.main_blocks[jbid].stride
                            + cpu_low_lr(word);
                        *j = cpu_low_add(*j, c, mod);
                    } else if (kind == LOWDESC_BLOCK) {
                        uint32_t dbid = cpu_low_block(word);
                        Count* j = dp[dbid] + hr * job.block_blocks[dbid].stride
                            + cpu_low_lr(word);
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

struct CpuLowDirectPool {
    int workers = 1;
    std::vector<CpuLowDirectStats> stats;
    double wall_s = 0.0;

    explicit CpuLowDirectPool(int n)
        : workers(std::max(1, n)), stats(size_t(std::max(1, n))) {}

    void run(
        const std::vector<CpuLowJob>& jobs,
        RamCounts& main_auth, RamCounts& block_auth,
        const StorageFactorHost& storage, const StorageLayout& layout,
        const LowDescHost& desc, const LowOrbitHost& orbit, Count mod
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
                    process_cpu_low_group_direct(
                        stats[w], jobs[q], main_auth, block_auth,
                        storage, layout, desc, orbit, mod);
                }
            });
        }
        for (auto& t : ts) t.join();
        wall_s += ram_seconds_since(t0);
    }

    double kernel_s() const { double z=0; for(auto const&s:stats)z+=s.kernel_s; return z; }
    uint64_t groups() const { uint64_t z=0; for(auto const&s:stats)z+=s.groups; return z; }
};
