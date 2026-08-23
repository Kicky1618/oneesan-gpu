#pragma once

#include "ramstream32_cpu_low.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

// In-place LOW-window executor.  The algebra is the same main/block orbit used
// by the proven GPU orbit prototype (src/cpp/probes/orbit_inplace_equiv.cpp),
// but all local destinations are pre-ranked once while the dense host rank
// tables still exist.  Runtime then needs only one main and one blocked buffer.

struct LowOrbitHost {
    // One 64-bit record per (p, main factor block, LOW all-rank).
    // kind: 0 none, 1 NN, 2 NR, 3 NL, 4 closure(LL/RR/RL).
    // For kinds 1..3:
    //   bits  0..19  partner-main LOW rank
    //   bits 20..25  partner-main block
    //   bits 26..45  dropped-N blocked LOW rank
    //   bits 46..51  dropped-N blocked block
    //   bits 52..54  kind
    std::vector<uint64_t> rec;
    std::array<uint32_t, 64> main_base{};
    uint32_t main_total = 0;
    uint64_t orbit_sources = 0;
    uint64_t closures = 0;
};

static constexpr uint64_t CPU_ORBIT_LR_MASK = (1ull << 20) - 1ull;
static constexpr uint64_t CPU_ORBIT_BLOCK_MASK = 0x3full;
static constexpr int CPU_ORBIT_JBLOCK_SHIFT = 20;
static constexpr int CPU_ORBIT_DLR_SHIFT = 26;
static constexpr int CPU_ORBIT_DBLOCK_SHIFT = 46;
static constexpr int CPU_ORBIT_KIND_SHIFT = 52;

enum CpuOrbitKind : uint32_t {
    CPU_ORBIT_NONE = 0,
    CPU_ORBIT_NN = 1,
    CPU_ORBIT_NR = 2,
    CPU_ORBIT_NL = 3,
    CPU_ORBIT_CLOSURE = 4,
};

static uint64_t cpu_orbit_pack(
    CpuOrbitKind kind, uint32_t jblock = 0, uint32_t jlr = 0,
    uint32_t dblock = 0, uint32_t dlr = 0
) {
    if (jlr > CPU_ORBIT_LR_MASK || dlr > CPU_ORBIT_LR_MASK
        || jblock > CPU_ORBIT_BLOCK_MASK || dblock > CPU_ORBIT_BLOCK_MASK) {
        std::cerr << "cpu orbit encoding overflow\n";
        std::exit(80);
    }
    return uint64_t(jlr)
        | (uint64_t(jblock) << CPU_ORBIT_JBLOCK_SHIFT)
        | (uint64_t(dlr) << CPU_ORBIT_DLR_SHIFT)
        | (uint64_t(dblock) << CPU_ORBIT_DBLOCK_SHIFT)
        | (uint64_t(kind) << CPU_ORBIT_KIND_SHIFT);
}

static inline uint32_t cpu_orbit_kind(uint64_t x) {
    return uint32_t((x >> CPU_ORBIT_KIND_SHIFT) & 7u);
}
static inline uint32_t cpu_orbit_jlr(uint64_t x) {
    return uint32_t(x & CPU_ORBIT_LR_MASK);
}
static inline uint32_t cpu_orbit_jblock(uint64_t x) {
    return uint32_t((x >> CPU_ORBIT_JBLOCK_SHIFT) & CPU_ORBIT_BLOCK_MASK);
}
static inline uint32_t cpu_orbit_dlr(uint64_t x) {
    return uint32_t((x >> CPU_ORBIT_DLR_SHIFT) & CPU_ORBIT_LR_MASK);
}
static inline uint32_t cpu_orbit_dblock(uint64_t x) {
    return uint32_t((x >> CPU_ORBIT_DBLOCK_SHIFT) & CPU_ORBIT_BLOCK_MASK);
}

static LowOrbitHost build_cpu_low_orbit(
    const StorageFactorHost& storage, const StorageLayout& layout,
    const LowDescHost& desc
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LM = (1u << (2 * L)) - 1u;
    constexpr uint32_t HM = (1u << (2 * H)) - 1u;

    LowOrbitHost o;
    o.main_base = desc.main_base;
    o.main_total = desc.main_total;
    o.rec.assign(size_t(o.main_total) * L, 0);

    auto representative_high = [&](int he) -> uint32_t {
        uint32_t a = storage.high_all_off[he];
        uint32_t b = storage.high_all_off[he + 1];
        return a < b ? storage.high_all_codes[a] : 0xffffffffu;
    };

    for (int p = L; p >= 1; --p) {
        uint32_t pi = uint32_t(L - p);
        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            const StorageBlock& sb = layout.main_blocks[bid];
            if (!sb.valid || !sb.cols || !sb.rows) continue;
            uint32_t hc = representative_high(sb.he);
            if (hc == 0xffffffffu) continue;
            uint32_t low0 = storage.low_all_off[sb.hs];

            for (uint32_t lr = 0; lr < sb.cols; ++lr) {
                uint32_t lc = storage.low_all_codes[low0 + lr];
                MateID m = MateID(lc)
                    | (MateID(sb.c) << (2 * L))
                    | (MateID(hc) << (2 * (L + 1)));
                MateValue a = mget(m, p);
                MateValue b = mget(m, p - 1);
                uint64_t word = 0;

                if (a == N) {
                    CpuOrbitKind kind = CPU_ORBIT_NONE;
                    MateValuePair pair = NN;
                    if (b == N) { kind = CPU_ORBIT_NN; pair = LR; }
                    else if (b == R) { kind = CPU_ORBIT_NR; pair = RN; }
                    else if (b == ::L) { kind = CPU_ORBIT_NL; pair = LN; }
                    if (kind != CPU_ORBIT_NONE) {
                        MateID jm = msetpair(m, p, pair);
                        uint32_t jlc = uint32_t(jm) & LM;
                        uint32_t jp = storage.low_packed_rank[jlc];
                        if (jp == 0xffffffffu) {
                            std::cerr << "cpu orbit partner rank missing\n";
                            std::exit(81);
                        }
                        uint32_t jlr = jp >> L;
                        int jcv = int(mget(jm, L));
                        uint32_t jbid = uint32_t(3 * sb.he + jcv);
                        if (jbid >= layout.main_blocks.size()
                            || jlr >= layout.main_blocks[jbid].cols) {
                            std::cerr << "cpu orbit partner block mismatch\n";
                            std::exit(82);
                        }

                        MateID dm = mshrink(m, p);
                        uint32_t dhc = uint32_t((dm >> (2 * L)) & HM);
                        if (dhc != hc) {
                            std::cerr << "cpu orbit drop unexpectedly changes HIGH\n";
                            std::exit(83);
                        }
                        uint32_t dlc = uint32_t(dm) & LM;
                        uint32_t dp = storage.low_packed_rank[dlc];
                        if (dp == 0xffffffffu) {
                            std::cerr << "cpu orbit dropped rank missing\n";
                            std::exit(84);
                        }
                        uint32_t dlr = dp >> L;
                        uint32_t dbid = uint32_t(sb.he);
                        if (dbid >= layout.block_blocks.size()
                            || dlr >= layout.block_blocks[dbid].cols) {
                            std::cerr << "cpu orbit dropped block mismatch\n";
                            std::exit(85);
                        }
                        word = cpu_orbit_pack(kind, jbid, jlr, dbid, dlr);
                        ++o.orbit_sources;
                    }
                } else if ((a == ::L && b == ::L) || (a == R && b == R)
                           || (a == R && b == ::L)) {
                    word = cpu_orbit_pack(CPU_ORBIT_CLOSURE);
                    ++o.closures;
                }
                o.rec[size_t(pi) * o.main_total + o.main_base[bid] + lr] = word;
            }
        }
    }

    std::cerr << "cpu_low_orbit mib="
              << double(o.rec.size() * sizeof(uint64_t)) / (1 << 20)
              << " orbit_sources=" << o.orbit_sources
              << " closures=" << o.closures << '\n';
    return o;
}

struct CpuLowInplaceScratch {
    uint8_t* arena = nullptr;
    size_t cap_bytes = 0;
    Count *mainv = nullptr, *blockv = nullptr;
    double pack_s = 0.0, kernel_s = 0.0, unpack_s = 0.0;
    uint64_t groups = 0;

    void ensure(Code m, Code d) {
        auto al = [](size_t x) { return (x + 4095) & ~size_t(4095); };
        size_t mb = al(size_t(m) * sizeof(Count));
        size_t db = al(size_t(d) * sizeof(Count));
        size_t need = mb + db;
        if (need > cap_bytes) {
            if (arena) munmap(arena, cap_bytes);
            int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
            flags |= MAP_NORESERVE;
#endif
            void* p = mmap(nullptr, need, PROT_READ | PROT_WRITE, flags, -1, 0);
            if (p == MAP_FAILED) {
                perror("mmap cpu-low inplace scratch");
                std::exit(86);
            }
            arena = static_cast<uint8_t*>(p);
            cap_bytes = need;
#ifdef MADV_HUGEPAGE
            madvise(arena, cap_bytes, MADV_HUGEPAGE);
#endif
        }
        mainv = reinterpret_cast<Count*>(arena);
        blockv = reinterpret_cast<Count*>(arena + mb);
    }

    void release() {
        if (arena) munmap(arena, cap_bytes);
        arena = nullptr;
        cap_bytes = 0;
        mainv = blockv = nullptr;
    }
};

static void process_cpu_low_group_inplace(
    CpuLowInplaceScratch& s, const CpuLowJob& job,
    RamCounts& main_auth, RamCounts& block_auth,
    const StorageFactorHost& storage, const StorageLayout& layout,
    const LowDescHost& desc, const LowOrbitHost& orbit, Count mod
) {
    if (!job.main_size && !job.block_size) return;
    s.ensure(job.main_size, job.block_size);

    auto t = std::chrono::steady_clock::now();
    if (job.main_size)
        cpu_low_pack_blocks(main_auth, s.mainv, job.main_blocks, layout.main_blocks,
                            job.mask, storage);
    if (job.block_size)
        cpu_low_pack_blocks(block_auth, s.blockv, job.block_blocks, layout.block_blocks,
                            job.mask, storage);
    s.pack_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    for (int p = LOW_LUT_K; p >= 1; --p) {
        uint32_t pi = uint32_t(LOW_LUT_K - p);

        // Orbit pass: fold identity + included local pair + old blocked value.
        for (size_t bid = 0; bid < job.main_blocks.size(); ++bid) {
            const FBlock& x = job.main_blocks[bid];
            if (x.end == x.off || !x.stride) continue;
            Code rows = (x.end - x.off) / x.stride;
            for (Code hr = 0; hr < rows; ++hr) {
                Code base = x.off + hr * x.stride;
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    uint64_t ow = orbit.rec[
                        size_t(pi) * orbit.main_total + orbit.main_base[bid] + lr];
                    uint32_t kind = cpu_orbit_kind(ow);
                    if (kind < CPU_ORBIT_NN || kind > CPU_ORBIT_NL) continue;

                    Code i = base + lr;
                    const FBlock& jy = job.main_blocks[cpu_orbit_jblock(ow)];
                    const FBlock& dy = job.block_blocks[cpu_orbit_dblock(ow)];
                    Code j = jy.off + hr * jy.stride + cpu_orbit_jlr(ow);
                    Code dj = dy.off + hr * dy.stride + cpu_orbit_dlr(ow);
                    Count c = s.mainv[i];
                    Count d = s.blockv[dj];

                    if (kind == CPU_ORBIT_NN) {
                        s.mainv[j] = cpu_low_add(s.mainv[j], c, mod);
                        s.mainv[i] = cpu_low_add(c, d, mod);
                        s.blockv[dj] = 0;
                    } else {
                        Count cc = s.mainv[j];
                        Count all = cpu_low_add(cpu_low_add(c, cc, mod), d, mod);
                        if (p == 1) {
                            s.mainv[i] = all;
                            s.mainv[j] = cpu_low_add(c, cc, mod);
                            s.blockv[dj] = 0;
                        } else {
                            s.mainv[i] = all;
                            s.blockv[dj] = c;
                            // Partner RN/LN keeps its identity value in mainv[j].
                        }
                    }
                }
            }
        }

        // Closure pass: LL/RR/RL branches.  The current main value already
        // contains every identity/orbit contribution for this position.
        for (size_t bid = 0; bid < job.main_blocks.size(); ++bid) {
            const FBlock& x = job.main_blocks[bid];
            if (x.end == x.off || !x.stride) continue;
            Code rows = (x.end - x.off) / x.stride;
            uint32_t high0 = G_FACTOR.high_mask_off[
                size_t(job.mask) * FactorTablesHost::STRIDE + x.he];
            for (Code hr = 0; hr < rows; ++hr) {
                Code base = x.off + hr * x.stride;
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    uint64_t ow = orbit.rec[
                        size_t(pi) * orbit.main_total + orbit.main_base[bid] + lr];
                    if (cpu_orbit_kind(ow) != CPU_ORBIT_CLOSURE) continue;
                    Count c = s.mainv[base + lr];
                    if (!c) continue;
                    uint32_t word = desc.main_desc[
                        size_t(pi) * desc.main_total + desc.main_base[bid] + lr];
                    uint32_t kind = cpu_low_kind(word);
                    if (kind == LOWDESC_MAIN) {
                        const FBlock& y = job.main_blocks[cpu_low_block(word)];
                        Code j = y.off + hr * y.stride + cpu_low_lr(word);
                        s.mainv[j] = cpu_low_add(s.mainv[j], c, mod);
                    } else if (kind == LOWDESC_BLOCK) {
                        const FBlock& y = job.block_blocks[cpu_low_block(word)];
                        Code j = y.off + hr * y.stride + cpu_low_lr(word);
                        s.blockv[j] = cpu_low_add(s.blockv[j], c, mod);
                    } else if (kind == LOWDESC_CROSS) {
                        uint32_t hc = G_FACTOR.high_mask_codes[high0 + hr];
                        uint32_t hc2 = cpu_low_flip_high(hc, cpu_low_depth(word));
                        if (hc2 == 0xffffffffu) continue;
                        if (p == 1) {
                            const FBlock& y = job.main_blocks[cpu_low_block(word)];
                            uint32_t hr2 = cpu_high_mask_rank(job.mask, hc2, y.he);
                            if (hr2 == 0xffffffffu) continue;
                            Code j = y.off + Code(hr2) * y.stride + cpu_low_lr(word);
                            s.mainv[j] = cpu_low_add(s.mainv[j], c, mod);
                        } else {
                            const FBlock& y = job.block_blocks[cpu_low_block(word)];
                            uint32_t hr2 = cpu_high_mask_rank(job.mask, hc2, y.he);
                            if (hr2 == 0xffffffffu) continue;
                            Code j = y.off + Code(hr2) * y.stride + cpu_low_lr(word);
                            s.blockv[j] = cpu_low_add(s.blockv[j], c, mod);
                        }
                    }
                }
            }
        }
    }
    s.kernel_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (job.main_size)
        cpu_low_unpack_blocks(main_auth, s.mainv, job.main_blocks, layout.main_blocks,
                              job.mask, storage);
    if (job.block_size)
        cpu_low_unpack_blocks(block_auth, s.blockv, job.block_blocks, layout.block_blocks,
                              job.mask, storage);
    s.unpack_s += ram_seconds_since(t);
    ++s.groups;
}

struct CpuLowInplacePool {
    int workers = 1;
    std::vector<CpuLowInplaceScratch> scratch;
    double wall_s = 0.0;

    explicit CpuLowInplacePool(int n)
        : workers(std::max(1, n)), scratch(size_t(std::max(1, n))) {}

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
                    process_cpu_low_group_inplace(
                        scratch[w], jobs[q], main_auth, block_auth,
                        storage, layout, desc, orbit, mod);
                }
            });
        }
        for (auto& t : ts) t.join();
        wall_s += ram_seconds_since(t0);
    }

    size_t peak_scratch_bytes() const {
        size_t z = 0;
        for (auto const& s : scratch) z += s.cap_bytes;
        return z;
    }
    double pack_s() const { double z=0; for(auto const&s:scratch)z+=s.pack_s; return z; }
    double kernel_s() const { double z=0; for(auto const&s:scratch)z+=s.kernel_s; return z; }
    double unpack_s() const { double z=0; for(auto const&s:scratch)z+=s.unpack_s; return z; }
    uint64_t groups() const { uint64_t z=0; for(auto const&s:scratch)z+=s.groups; return z; }
    void release() { for (auto& s : scratch) s.release(); }
};
