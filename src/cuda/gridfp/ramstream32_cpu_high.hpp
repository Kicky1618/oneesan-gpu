#pragma once

#include "ramstream32_highdesc.cuh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include <sys/mman.h>

// CPU executor for selected HIGH-window groups.
//
// A HIGH-window group fixes LOW occupancy. All transitions preserve that
// occupancy: ordinary HIGH transitions leave LOW untouched, while the only
// boundary-crossing case flips one occupied LOW symbol R -> L. Therefore each
// LOW-occupancy group is transition-closed and can be assigned independently to
// CPU or GPU.
//
// This executor uses the same factorized local layout as the GPU HIGH window:
// rows are all HIGH codes at a factor-block height and columns are the selected
// LOW occupancy slice. Small groups can therefore stay entirely in System RAM,
// removing their H2D+D2H traffic at the cost of CPU transition work.

static inline uint32_t cpu_high_desc_kind(uint32_t x) {
    return x >> HIGHDESC_KIND_SHIFT;
}
static inline uint32_t cpu_high_desc_block(uint32_t x) {
    return (x >> HIGHDESC_BLOCK_SHIFT) & HIGHDESC_BLOCK_MASK;
}
static inline uint32_t cpu_high_desc_rank(uint32_t x) {
    return x & HIGHDESC_RANK_MASK;
}
static inline uint32_t cpu_high_desc_depth(uint32_t x) {
    return (x >> HIGHDESC_DEPTH_SHIFT) & HIGHDESC_DEPTH_MASK;
}

static inline uint32_t cpu_high_flip_low(uint32_t lc, uint32_t depth) {
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

struct CpuHighCrossHost {
    // [(depth-1) * pitch + global low-mask-code index] -> destination
    // mask-local LOW rank. 0xffff is invalid.
    std::vector<uint16_t> low_cross_rank;
    uint32_t pitch = 0;
};

static constexpr uint16_t CPU_HIGH_CROSS_INVALID = 0xffffu;
static_assert(LOW_LUT_K <= 15,
              "uint16_t CPU HIGH CROSS rank requires LOW_LUT_K <= 15");

static CpuHighCrossHost build_cpu_high_cross(
    const StorageFactorHost& storage
) {
    CpuHighCrossHost out;
    out.pitch = uint32_t(G_FACTOR.low_mask_codes.size());
    out.low_cross_rank.assign(
        size_t(LOW_LUT_K) * out.pitch, CPU_HIGH_CROSS_INVALID);

    constexpr uint32_t LOW_MASK_RANK_MASK = (1u << LOW_LUT_K) - 1u;
    for (uint32_t depth = 1; depth <= uint32_t(LOW_LUT_K); ++depth) {
        uint16_t* dst = out.low_cross_rank.data()
            + size_t(depth - 1) * out.pitch;
        for (uint32_t i = 0; i < out.pitch; ++i) {
            uint32_t lc = G_FACTOR.low_mask_codes[i];
            uint32_t lc2 = cpu_high_flip_low(lc, depth);
            if (lc2 == 0xffffffffu) continue;
            uint32_t packed = storage.low_packed_rank[lc2];
            if (packed == 0xffffffffu) continue;
            uint32_t lr2 = packed & LOW_MASK_RANK_MASK;
            if (lr2 >= uint32_t(CPU_HIGH_CROSS_INVALID)) {
                std::cerr << "cpu high cross rank overflow depth=" << depth
                          << " rank=" << lr2 << '\n';
                std::exit(110);
            }
            dst[i] = uint16_t(lr2);
        }
    }

    std::cerr << "cpu_high_cross pitch=" << out.pitch
              << " mib="
              << double(out.low_cross_rank.size() * sizeof(uint16_t)) / (1 << 20)
              << '\n';
    return out;
}

struct CpuHighJob {
    int g = 0;
    uint32_t mask = 0;
    Code main_size = 0;
    Code block_size = 0;
    size_t scratch_bytes = 0;
    std::vector<FBlock> main_blocks;
    std::vector<FBlock> block_blocks;
};

static std::vector<CpuHighJob> make_cpu_high_jobs(
    int W, const WindowPlan& wp
) {
    if (wp.p_hi <= LOW_LUT_K) {
        std::cerr << "CpuHighJob requires the HIGH window\n";
        std::exit(111);
    }

    int groups = 1 << int(wp.fixed_pos.size());
    std::vector<CpuHighJob> jobs;
    jobs.reserve(groups);
    for (int g = 0; g < groups; ++g) {
        uint32_t mf, mo, bf, bo;
        window_masks(W, wp.p_hi, wp.p_lo, wp.fixed_pos, uint32_t(g),
                     mf, mo, bf, bo);
        GroupSpec ms = make_spec(W, mf, mo);
        GroupSpec ds = make_spec(W - 1, bf, bo);
        uint32_t mask = mo & ((1u << LOW_LUT_K) - 1u);
        auto mb = make_factor_main_blocks(true, mask);
        auto db = make_factor_block_blocks(true, mask);
        if (mb.empty() || db.empty()
            || mb.back().end != ms.size || db.back().end != ds.size) {
            std::cerr << "cpu-high group size mismatch g=" << g
                      << " main=" << (mb.empty() ? 0 : mb.back().end)
                      << '/' << ms.size
                      << " block=" << (db.empty() ? 0 : db.back().end)
                      << '/' << ds.size << '\n';
            std::exit(112);
        }

        CpuHighJob job;
        job.g = g;
        job.mask = mask;
        job.main_size = ms.size;
        job.block_size = ds.size;
        job.scratch_bytes = size_t(2 * ms.size + 2 * ds.size) * sizeof(Count);
        job.main_blocks = std::move(mb);
        job.block_blocks = std::move(db);
        jobs.push_back(std::move(job));
    }

    std::sort(jobs.begin(), jobs.end(), [](const CpuHighJob& a, const CpuHighJob& b) {
        return a.scratch_bytes > b.scratch_bytes;
    });
    return jobs;
}

struct CpuHighScratch {
    uint8_t* arena = nullptr;
    size_t cap_bytes = 0;
    Count *a = nullptr, *b = nullptr, *d = nullptr, *e = nullptr;
    double pack_s = 0.0;
    double kernel_s = 0.0;
    double unpack_s = 0.0;
    uint64_t groups = 0;

    void ensure(Code m, Code dcount) {
        auto align_page = [](size_t x) { return (x + 4095) & ~size_t(4095); };
        size_t mb = align_page(size_t(m) * sizeof(Count));
        size_t db = align_page(size_t(dcount) * sizeof(Count));
        size_t need = 2 * mb + 2 * db;
        if (need <= cap_bytes) {
            size_t off = 0;
            a = reinterpret_cast<Count*>(arena + off); off += mb;
            b = reinterpret_cast<Count*>(arena + off); off += mb;
            d = reinterpret_cast<Count*>(arena + off); off += db;
            e = reinterpret_cast<Count*>(arena + off);
            return;
        }

        if (arena) munmap(arena, cap_bytes);
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
        flags |= MAP_NORESERVE;
#endif
        void* p = mmap(nullptr, need, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (p == MAP_FAILED) {
            perror("mmap cpu-high scratch");
            std::exit(113);
        }
        arena = static_cast<uint8_t*>(p);
        cap_bytes = need;
#ifdef MADV_HUGEPAGE
        madvise(arena, cap_bytes, MADV_HUGEPAGE);
#endif
        size_t off = 0;
        a = reinterpret_cast<Count*>(arena + off); off += mb;
        b = reinterpret_cast<Count*>(arena + off); off += mb;
        d = reinterpret_cast<Count*>(arena + off); off += db;
        e = reinterpret_cast<Count*>(arena + off);
    }

    void release() {
        if (arena) munmap(arena, cap_bytes);
        arena = nullptr;
        cap_bytes = 0;
        a = b = d = e = nullptr;
    }
};

static void cpu_high_pack_blocks(
    const RamCounts& auth, Count* local,
    const std::vector<FBlock>& blocks,
    const std::vector<StorageBlock>& storage_blocks,
    uint32_t mask, const StorageFactorHost& storage
) {
    constexpr int S = StorageFactorHost::S;
    for (size_t bid = 0; bid < blocks.size(); ++bid) {
        const FBlock& fb = blocks[bid];
        if (fb.end == fb.off || !fb.stride) continue;
        const StorageBlock& sb = storage_blocks[bid];
        uint32_t col0 = storage.low_mask_begin[size_t(mask) * S + fb.hs];
        Code rows = (fb.end - fb.off) / fb.stride;
        for (Code hr = 0; hr < rows; ++hr) {
            std::memcpy(local + fb.off + hr * fb.stride,
                        auth.ptr + sb.off + hr * sb.cols + col0,
                        size_t(fb.stride) * sizeof(Count));
        }
    }
}

static void cpu_high_unpack_blocks(
    RamCounts& auth, const Count* local,
    const std::vector<FBlock>& blocks,
    const std::vector<StorageBlock>& storage_blocks,
    uint32_t mask, const StorageFactorHost& storage
) {
    constexpr int S = StorageFactorHost::S;
    for (size_t bid = 0; bid < blocks.size(); ++bid) {
        const FBlock& fb = blocks[bid];
        if (fb.end == fb.off || !fb.stride) continue;
        const StorageBlock& sb = storage_blocks[bid];
        uint32_t col0 = storage.low_mask_begin[size_t(mask) * S + fb.hs];
        Code rows = (fb.end - fb.off) / fb.stride;
        for (Code hr = 0; hr < rows; ++hr) {
            std::memcpy(auth.ptr + sb.off + hr * sb.cols + col0,
                        local + fb.off + hr * fb.stride,
                        size_t(fb.stride) * sizeof(Count));
        }
    }
}

static inline Count cpu_high_add(Count a, Count b, Count mod) {
    if (!b) return a;
    return (a >= mod - b) ? a - (mod - b) : a + b;
}

static void process_cpu_high_group(
    CpuHighScratch& scratch, const CpuHighJob& job,
    RamCounts& main_auth, RamCounts& block_auth,
    const StorageFactorHost& storage, const StorageLayout& layout,
    const HighDescHost& desc, const CpuHighCrossHost& cross, Count mod
) {
    if (!job.main_size && !job.block_size) return;
    scratch.ensure(job.main_size, job.block_size);

    auto t = std::chrono::steady_clock::now();
    if (job.main_size)
        cpu_high_pack_blocks(main_auth, scratch.a, job.main_blocks,
                             layout.main_blocks, job.mask, storage);
    if (job.block_size)
        cpu_high_pack_blocks(block_auth, scratch.d, job.block_blocks,
                             layout.block_blocks, job.mask, storage);
    scratch.pack_s += ram_seconds_since(t);

    Count* cur = scratch.a;
    Count* nxt = scratch.b;
    Count* dcur = scratch.d;
    Count* dnext = scratch.e;
    t = std::chrono::steady_clock::now();

    constexpr int S = FactorTablesHost::STRIDE;
    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        uint32_t pi = uint32_t((TARGET_W - 1) - p);
        if (job.main_size)
            std::memcpy(nxt, cur, size_t(job.main_size) * sizeof(Count));
        if (job.block_size)
            std::memset(dnext, 0, size_t(job.block_size) * sizeof(Count));

        for (uint32_t bid = 0; bid < job.main_blocks.size(); ++bid) {
            const FBlock& x = job.main_blocks[bid];
            if (x.end == x.off || !x.stride) continue;
            Code rows = (x.end - x.off) / x.stride;
            uint32_t low0 = G_FACTOR.low_mask_off[
                size_t(job.mask) * S + x.hs];

            for (Code hr = 0; hr < rows; ++hr) {
                uint32_t word = desc.main_desc[
                    size_t(pi) * desc.main_total + desc.main_base[bid] + hr];
                uint32_t kind = cpu_high_desc_kind(word);
                if (kind == HIGHDESC_INVALID) continue;

                const Count* src = cur + x.off + hr * x.stride;
                if (kind == HIGHDESC_MAIN) {
                    uint32_t jbid = cpu_high_desc_block(word);
                    const FBlock& y = job.main_blocks[jbid];
                    Count* dst = nxt + y.off
                        + Code(cpu_high_desc_rank(word)) * y.stride;
                    for (uint32_t lr = 0; lr < x.stride; ++lr) {
                        Count c = src[lr];
                        if (c) dst[lr] = cpu_high_add(dst[lr], c, mod);
                    }
                } else if (kind == HIGHDESC_BLOCK) {
                    uint32_t dbid = cpu_high_desc_block(word);
                    const FBlock& y = job.block_blocks[dbid];
                    Count* dst = dnext + y.off
                        + Code(cpu_high_desc_rank(word)) * y.stride;
                    for (uint32_t lr = 0; lr < x.stride; ++lr) {
                        Count c = src[lr];
                        if (c) dst[lr] = cpu_high_add(dst[lr], c, mod);
                    }
                } else {
                    uint32_t dbid = cpu_high_desc_block(word);
                    const FBlock& y = job.block_blocks[dbid];
                    Count* dst = dnext + y.off
                        + Code(cpu_high_desc_rank(word)) * y.stride;
                    uint32_t depth = cpu_high_desc_depth(word);
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

        for (uint32_t bid = 0; bid < job.block_blocks.size(); ++bid) {
            const FBlock& x = job.block_blocks[bid];
            if (x.end == x.off || !x.stride) continue;
            Code rows = (x.end - x.off) / x.stride;
            for (Code hr = 0; hr < rows; ++hr) {
                uint32_t word = desc.block_desc[
                    size_t(pi) * desc.block_total + desc.block_base[bid] + hr];
                if (cpu_high_desc_kind(word) != HIGHDESC_MAIN) continue;
                uint32_t jbid = cpu_high_desc_block(word);
                const FBlock& y = job.main_blocks[jbid];
                const Count* src = dcur + x.off + hr * x.stride;
                Count* dst = nxt + y.off
                    + Code(cpu_high_desc_rank(word)) * y.stride;
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    Count c = src[lr];
                    if (c) dst[lr] = cpu_high_add(dst[lr], c, mod);
                }
            }
        }

        std::swap(cur, nxt);
        std::swap(dcur, dnext);
    }
    scratch.kernel_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (job.main_size)
        cpu_high_unpack_blocks(main_auth, cur, job.main_blocks,
                               layout.main_blocks, job.mask, storage);
    if (job.block_size)
        cpu_high_unpack_blocks(block_auth, dcur, job.block_blocks,
                               layout.block_blocks, job.mask, storage);
    scratch.unpack_s += ram_seconds_since(t);
    ++scratch.groups;
}

struct CpuHighPool {
    int workers = 1;
    std::vector<CpuHighScratch> scratch;
    double wall_s = 0.0;

    explicit CpuHighPool(int n)
        : workers(std::max(1, n)), scratch(size_t(std::max(1, n))) {}

    void run(
        const std::vector<const CpuHighJob*>& jobs,
        RamCounts& main_auth, RamCounts& block_auth,
        const StorageFactorHost& storage, const StorageLayout& layout,
        const HighDescHost& desc, const CpuHighCrossHost& cross, Count mod
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
                    process_cpu_high_group(
                        scratch[w], *jobs[q], main_auth, block_auth,
                        storage, layout, desc, cross, mod);
                }
            });
        }
        for (auto& thread : ts) thread.join();
        wall_s += ram_seconds_since(t0);
    }

    double pack_s() const {
        double z = 0; for (const auto& x : scratch) z += x.pack_s; return z;
    }
    double kernel_s() const {
        double z = 0; for (const auto& x : scratch) z += x.kernel_s; return z;
    }
    double unpack_s() const {
        double z = 0; for (const auto& x : scratch) z += x.unpack_s; return z;
    }
    uint64_t groups() const {
        uint64_t z = 0; for (const auto& x : scratch) z += x.groups; return z;
    }
    size_t peak_scratch_bytes() const {
        size_t z = 0; for (const auto& x : scratch) z = std::max(z, x.cap_bytes); return z;
    }
    void release() {
        for (auto& x : scratch) x.release();
    }
};
