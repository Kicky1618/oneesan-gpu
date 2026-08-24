#pragma once

#include "ramstream32_cpu_high.hpp"
#include "ramstream32_cpu_affinity.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

// Zero-scratch CPU HIGH executor.
//
// For every active HIGH position p >= LOW_LUT_K+1, p is strictly greater than
// one. The N* main states and their partner/drop states therefore form the same
// in-place three-state orbits used by the CPU LOW direct executor, but without
// the p==1 special case. Closure states are processed after the complete orbit
// pass for each p. Fixed LOW occupancy makes different jobs disjoint.
//
// v5.1 stores each orbit in 64 bits: three 20-bit HIGH ranks. Destination block
// IDs are topology invariants and are reconstructed at runtime. v5.2 splits the
// orbit stream into NN and NR/NL classes. Stream identity replaces the remaining
// per-operation kind decode/branch and lets partner/drop block reconstruction be
// hoisted outside the inner operation loop. v5.3 likewise separates ordinary
// blocked closures from boundary CROSS closures, removing the last descriptor-
// kind branch from the CPU HIGH direct hot loops. v5.5 optionally pins workers
// through CPU_HIGH_CPU_LIST so direct work does not migrate across CPU sockets.
// v5.6 computes the exact direct cell work of each selected occupancy group once
// and schedules groups in descending work order. The atomic queue remains, so
// workers still dynamically steal jobs, but the largest tail risks are launched
// first instead of relying on the older transfer-size proxy.

enum CpuHighOrbitKind : uint8_t {
    CPU_HIGH_ORBIT_NN = 1,
    CPU_HIGH_ORBIT_NR = 2,
    CPU_HIGH_ORBIT_NL = 3,
};

using CpuHighOrbitOp = uint64_t;
static_assert(sizeof(CpuHighOrbitOp) == 8);
static constexpr uint64_t CPU_HIGH_ORBIT_RANK_MASK = (1ull << 20) - 1ull;
static constexpr int CPU_HIGH_ORBIT_PARTNER_SHIFT = 20;
static constexpr int CPU_HIGH_ORBIT_DROP_SHIFT = 40;

static inline CpuHighOrbitOp cpu_high_orbit_pack(
    uint32_t src_hr, uint32_t partner_hr, uint32_t drop_hr
) {
    if (src_hr > CPU_HIGH_ORBIT_RANK_MASK
        || partner_hr > CPU_HIGH_ORBIT_RANK_MASK
        || drop_hr > CPU_HIGH_ORBIT_RANK_MASK) {
        std::cerr << "cpu high orbit encoding overflow src=" << src_hr
                  << " partner=" << partner_hr << " drop=" << drop_hr << '\n';
        std::exit(119);
    }
    return uint64_t(src_hr)
        | (uint64_t(partner_hr) << CPU_HIGH_ORBIT_PARTNER_SHIFT)
        | (uint64_t(drop_hr) << CPU_HIGH_ORBIT_DROP_SHIFT);
}
static inline uint32_t cpu_high_orbit_src(CpuHighOrbitOp op) {
    return uint32_t(op & CPU_HIGH_ORBIT_RANK_MASK);
}
static inline uint32_t cpu_high_orbit_partner(CpuHighOrbitOp op) {
    return uint32_t((op >> CPU_HIGH_ORBIT_PARTNER_SHIFT) & CPU_HIGH_ORBIT_RANK_MASK);
}
static inline uint32_t cpu_high_orbit_drop(CpuHighOrbitOp op) {
    return uint32_t((op >> CPU_HIGH_ORBIT_DROP_SHIFT) & CPU_HIGH_ORBIT_RANK_MASK);
}

static inline uint32_t cpu_high_orbit_partner_block(
    uint32_t source_bid, const FBlock& source, int p, bool nn_stream
) {
    if (p != LOW_LUT_K + 1) return source_bid;

    uint32_t center = nn_stream ? uint32_t(R) : uint32_t(N);
    int he = int(source.hs) + (center == uint32_t(R) ? 1 : 0);
    return uint32_t(3 * he + int(center));
}
static inline uint32_t cpu_high_orbit_drop_block(const FBlock& source) {
    return uint32_t(source.hs);
}

struct CpuHighClosureOp {
    uint32_t src_hr = 0;
    uint32_t desc = 0;
};
static_assert(sizeof(CpuHighClosureOp) == 8);

struct CpuHighOrbitStreams {
    std::vector<CpuHighOrbitOp> nn;
    std::vector<CpuHighOrbitOp> nrnl;
    size_t size() const { return nn.size() + nrnl.size(); }
};

struct CpuHighOrbitOffsets {
    std::vector<uint32_t> nn;
    std::vector<uint32_t> nrnl;
    size_t size() const { return nn.size() + nrnl.size(); }
};

struct CpuHighClosureStreams {
    std::vector<CpuHighClosureOp> block;
    std::vector<CpuHighClosureOp> cross;
    size_t size() const { return block.size() + cross.size(); }
};

struct CpuHighClosureOffsets {
    std::vector<uint32_t> block;
    std::vector<uint32_t> cross;
    size_t size() const { return block.size() + cross.size(); }
};

struct CpuHighDirectHost {
    CpuHighOrbitStreams orbit_ops;
    CpuHighClosureStreams closure_ops;
    CpuHighOrbitOffsets orbit_off;
    CpuHighClosureOffsets closure_off;
    uint32_t nblocks = 0;
};

static CpuHighDirectHost build_cpu_high_direct(
    const StorageFactorHost& storage, const StorageLayout& layout,
    const HighDescHost& desc
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LM = (1u << (2 * L)) - 1u;
    constexpr uint32_t HM = (1u << (2 * H)) - 1u;

    CpuHighDirectHost out;
    out.nblocks = uint32_t(layout.main_blocks.size());
    size_t pitch = size_t(out.nblocks) + 1;
    out.orbit_off.nn.resize(size_t(H) * pitch);
    out.orbit_off.nrnl.resize(size_t(H) * pitch);
    out.closure_off.block.resize(size_t(H) * pitch);
    out.closure_off.cross.resize(size_t(H) * pitch);

    auto representative_low = [&](int hs) -> uint32_t {
        uint32_t a = storage.low_all_off[hs];
        uint32_t b = storage.low_all_off[hs + 1];
        return a < b ? storage.low_all_codes[a] : 0xffffffffu;
    };

    for (int p = TARGET_W - 1; p >= L + 1; --p) {
        uint32_t pi = uint32_t((TARGET_W - 1) - p);
        for (uint32_t bid = 0; bid < out.nblocks; ++bid) {
            size_t oi = size_t(pi) * pitch + bid;
            out.orbit_off.nn[oi] = uint32_t(out.orbit_ops.nn.size());
            out.orbit_off.nrnl[oi] = uint32_t(out.orbit_ops.nrnl.size());
            out.closure_off.block[oi] = uint32_t(out.closure_ops.block.size());
            out.closure_off.cross[oi] = uint32_t(out.closure_ops.cross.size());

            const StorageBlock& sb = layout.main_blocks[bid];
            if (!sb.valid || !sb.rows || !sb.cols) continue;
            uint32_t lc = representative_low(sb.hs);
            if (lc == 0xffffffffu) continue;
            uint32_t high0 = storage.high_all_off[sb.he];

            for (uint32_t hr = 0; hr < sb.rows; ++hr) {
                uint32_t hc = storage.high_all_codes[high0 + hr];
                MateID m = MateID(lc)
                    | (MateID(sb.c) << (2 * L))
                    | (MateID(hc) << (2 * (L + 1)));
                MateValue a = mget(m, p);
                MateValue b = mget(m, p - 1);

                CpuHighOrbitKind okind{};
                MateValuePair partner_pair = NN;
                if (a == N && b == N) {
                    okind = CPU_HIGH_ORBIT_NN;
                    partner_pair = LR;
                } else if (a == N && b == R) {
                    okind = CPU_HIGH_ORBIT_NR;
                    partner_pair = RN;
                } else if (a == N && b == ::L) {
                    okind = CPU_HIGH_ORBIT_NL;
                    partner_pair = LN;
                }

                if (okind) {
                    MateID jm = msetpair(m, p, partner_pair);
                    uint32_t jlc = uint32_t(jm) & LM;
                    if (jlc != lc) {
                        std::cerr << "cpu high orbit partner changed LOW\n";
                        std::exit(120);
                    }
                    uint32_t jhc = uint32_t((jm >> (2 * (L + 1))) & HM);
                    uint32_t jp = storage.high_packed_rank[jhc];
                    if (jp == 0xffffffffu) {
                        std::cerr << "cpu high orbit partner rank missing\n";
                        std::exit(121);
                    }
                    uint32_t jhr = jp >> H;
                    int jhe = seg_end_height_host(jhc, H);
                    int jcv = int(mget(jm, L));
                    uint32_t jbid = uint32_t(3 * jhe + jcv);
                    if (jbid >= layout.main_blocks.size()
                        || jhr >= layout.main_blocks[jbid].rows) {
                        std::cerr << "cpu high orbit partner block mismatch\n";
                        std::exit(122);
                    }

                    MateID dm = mshrink(m, p);
                    uint32_t dlc = uint32_t(dm) & LM;
                    if (dlc != lc) {
                        std::cerr << "cpu high orbit drop changed LOW\n";
                        std::exit(123);
                    }
                    uint32_t dhc = uint32_t((dm >> (2 * L)) & HM);
                    uint32_t dp = storage.high_packed_rank[dhc];
                    if (dp == 0xffffffffu) {
                        std::cerr << "cpu high orbit drop rank missing\n";
                        std::exit(124);
                    }
                    uint32_t dhr = dp >> H;
                    uint32_t dbid = uint32_t(seg_end_height_host(dhc, H));
                    if (dbid >= layout.block_blocks.size()
                        || dhr >= layout.block_blocks[dbid].rows) {
                        std::cerr << "cpu high orbit drop block mismatch\n";
                        std::exit(125);
                    }

                    FBlock source_fb{};
                    source_fb.he = sb.he;
                    source_fb.hs = sb.hs;
                    source_fb.c = sb.c;
                    bool nn_stream = okind == CPU_HIGH_ORBIT_NN;
                    uint32_t derived_jbid = cpu_high_orbit_partner_block(
                        bid, source_fb, p, nn_stream);
                    uint32_t derived_dbid = cpu_high_orbit_drop_block(source_fb);
                    if (derived_jbid != jbid || derived_dbid != dbid) {
                        std::cerr << "cpu high orbit derived block mismatch p=" << p
                                  << " bid=" << bid
                                  << " kind=" << unsigned(okind)
                                  << " partner=" << derived_jbid << '/' << jbid
                                  << " drop=" << derived_dbid << '/' << dbid << '\n';
                        std::exit(129);
                    }

                    CpuHighOrbitOp op = cpu_high_orbit_pack(hr, jhr, dhr);
                    if (nn_stream) out.orbit_ops.nn.push_back(op);
                    else out.orbit_ops.nrnl.push_back(op);
                    continue;
                }

                bool closure = (a == ::L && b == ::L)
                    || (a == R && b == R)
                    || (a == R && b == ::L);
                if (!closure) continue;

                uint32_t word = desc.main_desc[
                    size_t(pi) * desc.main_total + desc.main_base[bid] + hr];
                uint32_t kind = cpu_high_desc_kind(word);
                if (kind == HIGHDESC_INVALID) continue;
                if (kind == HIGHDESC_BLOCK) {
                    out.closure_ops.block.push_back({hr, word});
                } else if (kind == HIGHDESC_CROSS) {
                    out.closure_ops.cross.push_back({hr, word});
                } else {
                    std::cerr << "cpu high closure expected blocked destination p="
                              << p << " bid=" << bid << " hr=" << hr
                              << " kind=" << kind << '\n';
                    std::exit(126);
                }
            }
        }
        size_t end = size_t(pi) * pitch + out.nblocks;
        out.orbit_off.nn[end] = uint32_t(out.orbit_ops.nn.size());
        out.orbit_off.nrnl[end] = uint32_t(out.orbit_ops.nrnl.size());
        out.closure_off.block[end] = uint32_t(out.closure_ops.block.size());
        out.closure_off.cross[end] = uint32_t(out.closure_ops.cross.size());
    }

    std::cerr << "cpu_high_direct nn_orbit_ops=" << out.orbit_ops.nn.size()
              << " nrnl_orbit_ops=" << out.orbit_ops.nrnl.size()
              << " block_closure_ops=" << out.closure_ops.block.size()
              << " cross_closure_ops=" << out.closure_ops.cross.size()
              << " orbit_ops=" << out.orbit_ops.size()
              << " closure_ops=" << out.closure_ops.size()
              << " orbit_mib="
              << double(out.orbit_ops.size() * sizeof(CpuHighOrbitOp))/(1<<20)
              << " closure_mib="
              << double(out.closure_ops.size() * sizeof(CpuHighClosureOp))/(1<<20)
              << '\n';
    return out;
}

static inline std::pair<uint32_t,uint32_t> cpu_high_direct_range(
    const std::vector<uint32_t>& off, uint32_t nblocks,
    uint32_t pi, uint32_t bid
) {
    size_t pitch = size_t(nblocks) + 1;
    return {off[size_t(pi)*pitch+bid], off[size_t(pi)*pitch+bid+1]};
}

static uint64_t cpu_high_direct_job_cells(
    const CpuHighJob& job, const CpuHighDirectHost& direct
) {
    uint64_t cells = 0;
    for (uint32_t bid = 0; bid < direct.nblocks; ++bid) {
        const FBlock& x = job.main_blocks[bid];
        uint64_t width = x.stride;
        if (!width) continue;
        for (uint32_t pi = 0; pi < uint32_t(HIGH_LUT_K); ++pi) {
            auto [na, nb] = cpu_high_direct_range(
                direct.orbit_off.nn, direct.nblocks, pi, bid);
            auto [ra, rb] = cpu_high_direct_range(
                direct.orbit_off.nrnl, direct.nblocks, pi, bid);
            auto [ba, bb] = cpu_high_direct_range(
                direct.closure_off.block, direct.nblocks, pi, bid);
            auto [ca, cb] = cpu_high_direct_range(
                direct.closure_off.cross, direct.nblocks, pi, bid);
            uint64_t ops = uint64_t(nb - na) + uint64_t(rb - ra)
                + uint64_t(bb - ba) + uint64_t(cb - ca);
            if (ops && width > UINT64_MAX / ops) {
                std::cerr << "cpu high direct job cell count overflow\n";
                std::exit(133);
            }
            cells += ops * width;
        }
    }
    return cells;
}

static Count* cpu_high_direct_row_ptr(
    RamCounts& auth, const StorageBlock& sb, const FBlock& fb,
    uint32_t mask, const StorageFactorHost& storage, uint32_t hr
) {
    if (!fb.stride || hr >= sb.rows) return nullptr;
    constexpr int S = FactorTablesHost::STRIDE;
    uint32_t col0 = storage.low_mask_begin[size_t(mask) * S + fb.hs];
    return auth.ptr + sb.off + Code(hr) * sb.cols + col0;
}

struct CpuHighDirectStats {
    double kernel_s = 0.0;
    uint64_t groups = 0;
};

static void process_cpu_high_group_direct(
    CpuHighDirectStats& stats, const CpuHighJob& job,
    RamCounts& main_auth, RamCounts& block_auth,
    const StorageFactorHost& storage, const StorageLayout& layout,
    const CpuHighDirectHost& direct, const CpuHighCrossHost& cross, Count mod
) {
    if (!job.main_size && !job.block_size) return;
    auto t0 = std::chrono::steady_clock::now();

    constexpr int S = FactorTablesHost::STRIDE;
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        uint32_t pi = uint32_t((TARGET_W - 1) - p);

        for (uint32_t bid = 0; bid < direct.nblocks; ++bid) {
            const FBlock& x = job.main_blocks[bid];
            if (!x.stride) continue;

            auto [na, nb] = cpu_high_direct_range(
                direct.orbit_off.nn, direct.nblocks, pi, bid);
            auto [ra, rb] = cpu_high_direct_range(
                direct.orbit_off.nrnl, direct.nblocks, pi, bid);
            if (na == nb && ra == rb) continue;

            uint32_t drop_block = cpu_high_orbit_drop_block(x);
            if (drop_block >= job.block_blocks.size()) {
                std::cerr << "cpu high direct drop block out of range\n";
                std::exit(130);
            }
            const FBlock& dy = job.block_blocks[drop_block];
            if (dy.stride != x.stride) {
                std::cerr << "cpu high direct drop LOW-width mismatch\n";
                std::exit(127);
            }

            if (na != nb) {
                uint32_t partner_block = cpu_high_orbit_partner_block(
                    bid, x, p, true);
                if (partner_block >= job.main_blocks.size()) {
                    std::cerr << "cpu high direct NN partner block out of range\n";
                    std::exit(130);
                }
                const FBlock& jy = job.main_blocks[partner_block];
                if (jy.stride != x.stride) {
                    std::cerr << "cpu high direct NN LOW-width mismatch\n";
                    std::exit(127);
                }
                for (uint32_t q = na; q < nb; ++q) {
                    CpuHighOrbitOp op = direct.orbit_ops.nn[q];
                    Count* ip = cpu_high_direct_row_ptr(
                        main_auth, layout.main_blocks[bid], x,
                        job.mask, storage, cpu_high_orbit_src(op));
                    Count* jp = cpu_high_direct_row_ptr(
                        main_auth, layout.main_blocks[partner_block], jy,
                        job.mask, storage, cpu_high_orbit_partner(op));
                    Count* dp = cpu_high_direct_row_ptr(
                        block_auth, layout.block_blocks[drop_block], dy,
                        job.mask, storage, cpu_high_orbit_drop(op));
                    if (!ip || !jp || !dp) continue;
                    for (uint32_t lr = 0; lr < x.stride; ++lr) {
                        Count c = ip[lr];
                        Count d = dp[lr];
                        jp[lr] = cpu_high_add(jp[lr], c, mod);
                        ip[lr] = cpu_high_add(c, d, mod);
                        dp[lr] = 0;
                    }
                }
            }

            if (ra != rb) {
                uint32_t partner_block = cpu_high_orbit_partner_block(
                    bid, x, p, false);
                if (partner_block >= job.main_blocks.size()) {
                    std::cerr << "cpu high direct NRNL partner block out of range\n";
                    std::exit(130);
                }
                const FBlock& jy = job.main_blocks[partner_block];
                if (jy.stride != x.stride) {
                    std::cerr << "cpu high direct NRNL LOW-width mismatch\n";
                    std::exit(127);
                }
                for (uint32_t q = ra; q < rb; ++q) {
                    CpuHighOrbitOp op = direct.orbit_ops.nrnl[q];
                    Count* ip = cpu_high_direct_row_ptr(
                        main_auth, layout.main_blocks[bid], x,
                        job.mask, storage, cpu_high_orbit_src(op));
                    Count* jp = cpu_high_direct_row_ptr(
                        main_auth, layout.main_blocks[partner_block], jy,
                        job.mask, storage, cpu_high_orbit_partner(op));
                    Count* dp = cpu_high_direct_row_ptr(
                        block_auth, layout.block_blocks[drop_block], dy,
                        job.mask, storage, cpu_high_orbit_drop(op));
                    if (!ip || !jp || !dp) continue;
                    for (uint32_t lr = 0; lr < x.stride; ++lr) {
                        Count c = ip[lr];
                        Count cc = jp[lr];
                        Count d = dp[lr];
                        ip[lr] = cpu_high_add(cpu_high_add(c, cc, mod), d, mod);
                        dp[lr] = c;
                    }
                }
            }
        }

        for (uint32_t bid = 0; bid < direct.nblocks; ++bid) {
            const FBlock& x = job.main_blocks[bid];
            if (!x.stride) continue;

            auto [ba, bb] = cpu_high_direct_range(
                direct.closure_off.block, direct.nblocks, pi, bid);
            for (uint32_t q = ba; q < bb; ++q) {
                const CpuHighClosureOp& op = direct.closure_ops.block[q];
                Count* src = cpu_high_direct_row_ptr(
                    main_auth, layout.main_blocks[bid], x,
                    job.mask, storage, op.src_hr);
                if (!src) continue;
                uint32_t dbid = cpu_high_desc_block(op.desc);
                const FBlock& y = job.block_blocks[dbid];
                Count* dst = cpu_high_direct_row_ptr(
                    block_auth, layout.block_blocks[dbid], y,
                    job.mask, storage, cpu_high_desc_rank(op.desc));
                if (!dst) continue;
                if (y.stride != x.stride) {
                    std::cerr << "cpu high direct closure LOW-width mismatch\n";
                    std::exit(128);
                }
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    Count c = src[lr];
                    if (c) dst[lr] = cpu_high_add(dst[lr], c, mod);
                }
            }

            auto [ca, cb] = cpu_high_direct_range(
                direct.closure_off.cross, direct.nblocks, pi, bid);
            if (ca == cb) continue;
            uint32_t low0 = G_FACTOR.low_mask_off[
                size_t(job.mask) * S + x.hs];
            for (uint32_t q = ca; q < cb; ++q) {
                const CpuHighClosureOp& op = direct.closure_ops.cross[q];
                Count* src = cpu_high_direct_row_ptr(
                    main_auth, layout.main_blocks[bid], x,
                    job.mask, storage, op.src_hr);
                if (!src) continue;
                uint32_t dbid = cpu_high_desc_block(op.desc);
                const FBlock& y = job.block_blocks[dbid];
                Count* dst = cpu_high_direct_row_ptr(
                    block_auth, layout.block_blocks[dbid], y,
                    job.mask, storage, cpu_high_desc_rank(op.desc));
                if (!dst) continue;
                uint32_t depth = cpu_high_desc_depth(op.desc);
                if (!depth || depth > uint32_t(LOW_LUT_K)) continue;
                const uint16_t* rank_row = cross.low_cross_rank.data()
                    + size_t(depth - 1) * cross.pitch + low0;
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    Count c = src[lr];
                    if (!c) continue;
                    uint16_t lr2 = rank_row[lr];
                    if (lr2 == CPU_HIGH_CROSS_INVALID) continue;
                    dst[lr2] = cpu_high_add(dst[lr2], c, mod);
                }
            }
        }
    }

    stats.kernel_s += ram_seconds_since(t0);
    ++stats.groups;
}

struct CpuHighDirectPool {
    int workers = 1;
    std::vector<CpuHighDirectStats> stats;
    double wall_s = 0.0;
    double schedule_build_s = 0.0;
    std::vector<const CpuHighJob*> schedule_source;
    std::vector<const CpuHighJob*> scheduled_jobs;

    explicit CpuHighDirectPool(int n)
        : workers(std::max(1, n)), stats(size_t(std::max(1, n))) {}

    void prepare_schedule(
        const std::vector<const CpuHighJob*>& jobs,
        const CpuHighDirectHost& direct
    ) {
        if (schedule_source.size() == jobs.size()
            && std::equal(schedule_source.begin(), schedule_source.end(), jobs.begin()))
            return;

        auto t0 = std::chrono::steady_clock::now();
        schedule_source = jobs;
        std::vector<std::pair<const CpuHighJob*,uint64_t>> ranked;
        ranked.reserve(jobs.size());
        uint64_t total_cells = 0;
        uint64_t max_cells = 0;
        uint64_t min_cells = UINT64_MAX;
        for (const CpuHighJob* job : jobs) {
            uint64_t cells = cpu_high_direct_job_cells(*job, direct);
            ranked.push_back({job, cells});
            total_cells += cells;
            max_cells = std::max(max_cells, cells);
            min_cells = std::min(min_cells, cells);
        }
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            if (a.second != b.second) return a.second > b.second;
            if (a.first->scratch_bytes != b.first->scratch_bytes)
                return a.first->scratch_bytes > b.first->scratch_bytes;
            return a.first->g < b.first->g;
        });
        scheduled_jobs.clear();
        scheduled_jobs.reserve(ranked.size());
        for (const auto& x : ranked) scheduled_jobs.push_back(x.first);
        if (jobs.empty()) min_cells = 0;
        double dt = ram_seconds_since(t0);
        schedule_build_s += dt;
        std::cerr << "cpu_high_direct_schedule jobs=" << jobs.size()
                  << " total_cells=" << total_cells
                  << " max_cells=" << max_cells
                  << " min_cells=" << min_cells
                  << " build_s=" << dt << '\n';
    }

    void run(
        const std::vector<const CpuHighJob*>& jobs,
        RamCounts& main_auth, RamCounts& block_auth,
        const StorageFactorHost& storage, const StorageLayout& layout,
        const CpuHighDirectHost& direct, const CpuHighCrossHost& cross, Count mod
    ) {
        prepare_schedule(jobs, direct);
        auto t0 = std::chrono::steady_clock::now();
        std::atomic<size_t> next{0};
        std::vector<std::thread> ts;
        ts.reserve(workers);
        for (int w = 0; w < workers; ++w) {
            ts.emplace_back([&, w] {
                cpu_high_bind_worker(w);
                for (;;) {
                    size_t q = next.fetch_add(1, std::memory_order_relaxed);
                    if (q >= scheduled_jobs.size()) break;
                    process_cpu_high_group_direct(
                        stats[w], *scheduled_jobs[q], main_auth, block_auth,
                        storage, layout, direct, cross, mod);
                }
            });
        }
        for (auto& thread : ts) thread.join();
        wall_s += ram_seconds_since(t0);
    }

    double kernel_s() const {
        double z = 0; for (const auto& x : stats) z += x.kernel_s; return z;
    }
    uint64_t groups() const {
        uint64_t z = 0; for (const auto& x : stats) z += x.groups; return z;
    }
};