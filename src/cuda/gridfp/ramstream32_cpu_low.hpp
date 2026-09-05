#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include <sys/mman.h>

// CPU executor for the LOW+center window used by the n=27 RAM-heavy profile.
// The factorized authoritative layout is row-major HIGH x LOW.  A LOW-window
// group fixes HIGH occupancy, so every factor block belonging to that group is
// a contiguous run in ordinary system RAM.  We therefore keep this window off
// PCIe entirely: copy the closed group into ordinary mmap scratch, execute all
// LOW positions on CPU, then copy it back.  Different HIGH-occupancy groups are
// transition-closed and can be processed by independent worker threads.

struct CpuLowJob {
    int g = 0;
    uint32_t mask = 0;
    Code main_size = 0;
    Code block_size = 0;
    size_t scratch_bytes = 0;
    std::vector<FBlock> main_blocks;
    std::vector<FBlock> block_blocks;
};

static std::vector<CpuLowJob> make_cpu_low_jobs(int W, const WindowPlan& wp) {
    int groups = 1 << int(wp.fixed_pos.size());
    std::vector<CpuLowJob> jobs;
    jobs.reserve(groups);
    for (int g = 0; g < groups; ++g) {
        uint32_t mf, mo, bf, bo;
        window_masks(W, wp.p_hi, wp.p_lo, wp.fixed_pos, uint32_t(g), mf, mo, bf, bo);
        GroupSpec ms = make_spec(W, mf, mo);
        GroupSpec ds = make_spec(W - 1, bf, bo);
        bool fix_low = wp.p_hi > LOW_LUT_K;
        if (fix_low) {
            std::cerr << "CpuLowJob requires the LOW window\n";
            std::exit(70);
        }
        uint32_t mask = (mo >> (LOW_LUT_K + 1)) & ((1u << HIGH_LUT_K) - 1u);
        auto mb = make_factor_main_blocks(false, mask);
        auto db = make_factor_block_blocks(false, mask);
        if (mb.empty() || db.empty() || mb.back().end != ms.size || db.back().end != ds.size) {
            std::cerr << "cpu-low group size mismatch g=" << g << "\n";
            std::exit(71);
        }
        CpuLowJob j;
        j.g = g;
        j.mask = mask;
        j.main_size = ms.size;
        j.block_size = ds.size;
        j.scratch_bytes = size_t(2 * ms.size + 2 * ds.size) * sizeof(Count);
        j.main_blocks = std::move(mb);
        j.block_blocks = std::move(db);
        jobs.push_back(std::move(j));
    }
    std::sort(jobs.begin(), jobs.end(), [](const CpuLowJob& a, const CpuLowJob& b) {
        return a.scratch_bytes > b.scratch_bytes;
    });
    return jobs;
}

struct CpuLowScratch {
    uint8_t* arena = nullptr;
    size_t cap_bytes = 0;
    Count *a = nullptr, *b = nullptr, *d = nullptr, *e = nullptr;
    double pack_s = 0.0, kernel_s = 0.0, unpack_s = 0.0;
    uint64_t groups = 0;

    void ensure(Code m, Code n) {
        auto al = [](size_t x) { return (x + 4095) & ~size_t(4095); };
        size_t mb = al(size_t(m) * sizeof(Count));
        size_t db = al(size_t(n) * sizeof(Count));
        size_t need = 2 * mb + 2 * db;
        if (need > cap_bytes) {
            if (arena) munmap(arena, cap_bytes);
            int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
            flags |= MAP_NORESERVE;
#endif
            void* p = mmap(nullptr, need, PROT_READ | PROT_WRITE, flags, -1, 0);
            if (p == MAP_FAILED) {
                perror("mmap cpu-low scratch");
                std::exit(72);
            }
            arena = static_cast<uint8_t*>(p);
            cap_bytes = need;
#ifdef MADV_HUGEPAGE
            madvise(arena, cap_bytes, MADV_HUGEPAGE);
#endif
        }
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

static inline Count cpu_low_add(Count a, Count b, Count mod) {
    if (!b) return a;
    return (a >= mod - b) ? a - (mod - b) : a + b;
}

static inline uint32_t cpu_low_kind(uint32_t x) { return x >> LOWDESC_KIND_SHIFT; }
static inline uint32_t cpu_low_block(uint32_t x) {
    return (x >> LOWDESC_BLOCK_SHIFT) & LOWDESC_BLOCK_MASK;
}
static inline uint32_t cpu_low_lr(uint32_t x) { return x & LOWDESC_LR_MASK; }
static inline uint32_t cpu_low_depth(uint32_t x) {
    return (x >> LOWDESC_DEPTH_SHIFT) & LOWDESC_DEPTH_MASK;
}

static uint32_t cpu_low_flip_high(uint32_t hc, uint32_t depth) {
    int s = int(depth);
    for (int pos = 0; pos < HIGH_LUT_K; ++pos) {
        MateValue v = MateValue((hc >> (2 * pos)) & 3u);
        if (v == ::L) {
            if (--s == 0) {
                uint32_t z = 3u << (2 * pos);
                return (hc & ~z) | (uint32_t(R) << (2 * pos));
            }
        } else if (v == R) {
            ++s;
        }
    }
    return 0xffffffffu;
}

static uint32_t cpu_high_mask_rank(uint32_t mask, uint32_t code, int h) {
    constexpr int S = FactorTablesHost::STRIDE;
    size_t ix = size_t(mask) * S + h;
    uint32_t a = G_FACTOR.high_mask_off[ix];
    uint32_t b = G_FACTOR.high_mask_off[ix + 1];
    auto first = G_FACTOR.high_mask_codes.begin() + a;
    auto last = G_FACTOR.high_mask_codes.begin() + b;
    auto it = std::lower_bound(first, last, code);
    if (it == last || *it != code) return 0xffffffffu;
    return uint32_t(it - first);
}

static void cpu_low_pack_blocks(
    const RamCounts& auth, Count* local,
    const std::vector<FBlock>& blocks, const std::vector<StorageBlock>& storage_blocks,
    uint32_t mask, const StorageFactorHost& storage
) {
    constexpr int S = StorageFactorHost::S;
    for (size_t bid = 0; bid < blocks.size(); ++bid) {
        const FBlock& fb = blocks[bid];
        if (fb.end == fb.off) continue;
        const StorageBlock& sb = storage_blocks[bid];
        uint32_t row0 = storage.high_mask_begin[size_t(mask) * S + fb.he];
        Code elems = fb.end - fb.off;
        std::memcpy(local + fb.off,
                    auth.ptr + sb.off + Code(row0) * sb.cols,
                    size_t(elems) * sizeof(Count));
    }
}

static void cpu_low_unpack_blocks(
    RamCounts& auth, const Count* local,
    const std::vector<FBlock>& blocks, const std::vector<StorageBlock>& storage_blocks,
    uint32_t mask, const StorageFactorHost& storage
) {
    constexpr int S = StorageFactorHost::S;
    for (size_t bid = 0; bid < blocks.size(); ++bid) {
        const FBlock& fb = blocks[bid];
        if (fb.end == fb.off) continue;
        const StorageBlock& sb = storage_blocks[bid];
        uint32_t row0 = storage.high_mask_begin[size_t(mask) * S + fb.he];
        Code elems = fb.end - fb.off;
        std::memcpy(auth.ptr + sb.off + Code(row0) * sb.cols,
                    local + fb.off,
                    size_t(elems) * sizeof(Count));
    }
}

static void process_cpu_low_group(
    CpuLowScratch& s, const CpuLowJob& job,
    RamCounts& main_auth, RamCounts& block_auth,
    const StorageFactorHost& storage, const StorageLayout& layout,
    const LowDescHost& desc, Count mod
) {
    if (!job.main_size && !job.block_size) return;
    s.ensure(job.main_size, job.block_size);

    auto t = std::chrono::steady_clock::now();
    if (job.main_size)
        cpu_low_pack_blocks(main_auth, s.a, job.main_blocks, layout.main_blocks,
                            job.mask, storage);
    if (job.block_size)
        cpu_low_pack_blocks(block_auth, s.d, job.block_blocks, layout.block_blocks,
                            job.mask, storage);
    s.pack_s += ram_seconds_since(t);

    Count* cur = s.a;
    Count* nxt = s.b;
    Count* dcur = s.d;
    Count* dnext = s.e;
    t = std::chrono::steady_clock::now();

    for (int p = LOW_LUT_K; p >= 1; --p) {
        uint32_t pi = uint32_t(LOW_LUT_K - p);
        if (job.main_size)
            std::memcpy(nxt, cur, size_t(job.main_size) * sizeof(Count));
        if (job.block_size)
            std::memset(dnext, 0, size_t(job.block_size) * sizeof(Count));

        for (size_t bid = 0; bid < job.main_blocks.size(); ++bid) {
            const FBlock& x = job.main_blocks[bid];
            if (x.end == x.off || !x.stride) continue;
            Code rows = (x.end - x.off) / x.stride;
            uint32_t high0 = G_FACTOR.high_mask_off[
                size_t(job.mask) * FactorTablesHost::STRIDE + x.he];
            for (Code hr = 0; hr < rows; ++hr) {
                Code base = x.off + hr * x.stride;
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    Count c = cur[base + lr];
                    if (!c) continue;
                    uint32_t word = desc.main_desc[
                        size_t(pi) * desc.main_total + desc.main_base[bid] + lr];
                    uint32_t kind = cpu_low_kind(word);
                    if (kind == LOWDESC_MAIN) {
                        const FBlock& y = job.main_blocks[cpu_low_block(word)];
                        Code j = y.off + hr * y.stride + cpu_low_lr(word);
                        nxt[j] = cpu_low_add(nxt[j], c, mod);
                    } else if (kind == LOWDESC_BLOCK) {
                        const FBlock& y = job.block_blocks[cpu_low_block(word)];
                        Code j = y.off + hr * y.stride + cpu_low_lr(word);
                        dnext[j] = cpu_low_add(dnext[j], c, mod);
                    } else if (kind == LOWDESC_CROSS) {
                        uint32_t hc = G_FACTOR.high_mask_codes[high0 + hr];
                        uint32_t hc2 = cpu_low_flip_high(hc, cpu_low_depth(word));
                        if (hc2 == 0xffffffffu) continue;
                        if (p == 1) {
                            const FBlock& y = job.main_blocks[cpu_low_block(word)];
                            uint32_t hr2 = cpu_high_mask_rank(job.mask, hc2, y.he);
                            if (hr2 == 0xffffffffu) continue;
                            Code j = y.off + Code(hr2) * y.stride + cpu_low_lr(word);
                            nxt[j] = cpu_low_add(nxt[j], c, mod);
                        } else {
                            const FBlock& y = job.block_blocks[cpu_low_block(word)];
                            uint32_t hr2 = cpu_high_mask_rank(job.mask, hc2, y.he);
                            if (hr2 == 0xffffffffu) continue;
                            Code j = y.off + Code(hr2) * y.stride + cpu_low_lr(word);
                            dnext[j] = cpu_low_add(dnext[j], c, mod);
                        }
                    }
                }
            }
        }

        for (size_t bid = 0; bid < job.block_blocks.size(); ++bid) {
            const FBlock& x = job.block_blocks[bid];
            if (x.end == x.off || !x.stride) continue;
            Code rows = (x.end - x.off) / x.stride;
            for (Code hr = 0; hr < rows; ++hr) {
                Code base = x.off + hr * x.stride;
                for (uint32_t lr = 0; lr < x.stride; ++lr) {
                    Count c = dcur[base + lr];
                    if (!c) continue;
                    uint32_t word = desc.block_desc[
                        size_t(pi) * desc.block_total + desc.block_base[bid] + lr];
                    if (cpu_low_kind(word) != LOWDESC_MAIN) continue;
                    const FBlock& y = job.main_blocks[cpu_low_block(word)];
                    Code j = y.off + hr * y.stride + cpu_low_lr(word);
                    nxt[j] = cpu_low_add(nxt[j], c, mod);
                }
            }
        }
        std::swap(cur, nxt);
        std::swap(dcur, dnext);
    }
    s.kernel_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (job.main_size)
        cpu_low_unpack_blocks(main_auth, cur, job.main_blocks, layout.main_blocks,
                              job.mask, storage);
    if (job.block_size)
        cpu_low_unpack_blocks(block_auth, dcur, job.block_blocks, layout.block_blocks,
                              job.mask, storage);
    s.unpack_s += ram_seconds_since(t);
    ++s.groups;
}

struct CpuLowPool {
    int workers = 1;
    std::vector<CpuLowScratch> scratch;
    double wall_s = 0.0;

    explicit CpuLowPool(int n): workers(std::max(1, n)), scratch(size_t(std::max(1, n))) {}

    void run(
        const std::vector<CpuLowJob>& jobs,
        RamCounts& main_auth, RamCounts& block_auth,
        const StorageFactorHost& storage, const StorageLayout& layout,
        const LowDescHost& desc, Count mod
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
                    process_cpu_low_group(scratch[w], jobs[q], main_auth, block_auth,
                                          storage, layout, desc, mod);
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
