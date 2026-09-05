#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

// Reuse the proven factorized topology codec and transition kernels.  This
// translation unit supplies a different authoritative-memory backend below.
#define main oneesan_factorized_hbm_unused_main
#include "../b300/oneesan_cuda_gridfp_b300_hbm32_factorized_batch.cu"
#undef main

static_assert(LOW_LUT_K > 0, "ramstream32 factorized backend requires LOW_LUT_K > 0");
static_assert(HIGH_LUT_K > 0, "ramstream32 factorized backend requires HIGH_LUT_K > 0");
static_assert(LOW_LUT_K + HIGH_LUT_K + 1 == TARGET_W,
              "TARGET_W must equal HIGH_LUT_K + 1 + LOW_LUT_K");

struct RamCounts {
    Count* ptr = nullptr;
    size_t bytes = 0;

    void alloc(Code n, const char* what) {
        bytes = size_t(n) * sizeof(Count);
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
        flags |= MAP_NORESERVE;
#endif
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (p == MAP_FAILED) {
            perror(what);
            std::exit(2);
        }
        ptr = static_cast<Count*>(p);
#ifdef MADV_HUGEPAGE
        madvise(ptr, bytes, MADV_HUGEPAGE);
#endif
    }

    void release() {
        if (ptr) munmap(ptr, bytes);
        ptr = nullptr;
        bytes = 0;
    }
};

struct PinnedCountsV3 {
    Count* ptr = nullptr;
    Code cap = 0;

    void ensure(Code n) {
        if (n <= cap) return;
        if (ptr) ck(cudaFreeHost(ptr), "free pinned staging");
        cap = n;
        ck(cudaHostAlloc(reinterpret_cast<void**>(&ptr), size_t(cap) * sizeof(Count),
                         cudaHostAllocPortable),
           "alloc pinned staging");
    }

    void release() {
        if (ptr) cudaFreeHost(ptr);
        ptr = nullptr;
        cap = 0;
    }
};

template<class F>
static void ram_parallel(Code n, int threads, F&& f) {
    if (!n) return;
    threads = std::max(1, threads);
    if (threads == 1 || n < Code(threads) * 1024) {
        for (Code i = 0; i < n; ++i) f(i);
        return;
    }
    Code chunk = (n + Code(threads) - 1) / Code(threads);
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        Code begin = Code(t) * chunk;
        Code end = std::min(n, begin + chunk);
        if (begin >= end) break;
        workers.emplace_back([=, &f] {
            for (Code i = begin; i < end; ++i) f(i);
        });
    }
    for (auto& w : workers) w.join();
}

static double ram_seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

// The device factorized codec has two rank fields packed in one uint32_t:
//   high bits = rank among all legal codes of the given height
//   low bits  = rank among codes with the fixed occupancy mask.
//
// For the RAM backend we intentionally redefine the "all" rank.  Instead of
// canonical code order, all codes of a fixed height are ordered by
//   (occupancy mask, mask-local rank).
// This makes every occupancy class contiguous in each matrix dimension.
struct StorageFactorHost {
    static constexpr int S = MAXW + 2;

    std::vector<uint32_t> low_all_codes;
    std::vector<uint32_t> high_all_codes;
    std::vector<uint32_t> low_packed_rank;
    std::vector<uint32_t> high_packed_rank;

    // Rank, within a fixed-height all-code list, where this occupancy mask starts.
    std::vector<uint32_t> low_mask_begin;
    std::vector<uint32_t> high_mask_begin;

    std::array<uint32_t, MAXW + 2> low_all_off{};
    std::array<uint32_t, MAXW + 2> high_all_off{};
};

static StorageFactorHost build_storage_factor_tables(const FactorTablesHost& base) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr int S = StorageFactorHost::S;
    const uint32_t LM = 1u << L;
    const uint32_t HM = 1u << H;

    StorageFactorHost f;
    f.low_packed_rank.assign(size_t(1) << (2 * L), 0xffffffffu);
    f.high_packed_rank.assign(size_t(1) << (2 * H), 0xffffffffu);
    f.low_mask_begin.resize(size_t(LM) * S);
    f.high_mask_begin.resize(size_t(HM) * S);

    for (int h = 0; h <= MAXW; ++h) {
        f.low_all_off[h] = uint32_t(f.low_all_codes.size());
        uint32_t storage_rank = 0;
        for (uint32_t mask = 0; mask < LM; ++mask) {
            size_t ix = size_t(mask) * S + h;
            f.low_mask_begin[ix] = storage_rank;
            uint32_t a = base.low_mask_off[ix];
            uint32_t b = base.low_mask_off[ix + 1];
            for (uint32_t p = a; p < b; ++p) {
                uint32_t code = base.low_mask_codes[p];
                uint32_t mask_rank = p - a;
                f.low_all_codes.push_back(code);
                f.low_packed_rank[code] = (storage_rank << L) | mask_rank;
                ++storage_rank;
            }
        }
        uint32_t expected = base.low_all_off[h + 1] - base.low_all_off[h];
        if (storage_rank != expected) {
            std::cerr << "low storage count mismatch h=" << h
                      << " got=" << storage_rank << " expected=" << expected << "\n";
            std::exit(30);
        }
    }
    f.low_all_off[MAXW + 1] = uint32_t(f.low_all_codes.size());

    for (int h = 0; h <= MAXW; ++h) {
        f.high_all_off[h] = uint32_t(f.high_all_codes.size());
        uint32_t storage_rank = 0;
        for (uint32_t mask = 0; mask < HM; ++mask) {
            size_t ix = size_t(mask) * S + h;
            f.high_mask_begin[ix] = storage_rank;
            uint32_t a = base.high_mask_off[ix];
            uint32_t b = base.high_mask_off[ix + 1];
            for (uint32_t p = a; p < b; ++p) {
                uint32_t code = base.high_mask_codes[p];
                uint32_t mask_rank = p - a;
                f.high_all_codes.push_back(code);
                f.high_packed_rank[code] = (storage_rank << H) | mask_rank;
                ++storage_rank;
            }
        }
        uint32_t expected = base.high_all_off[h + 1] - base.high_all_off[h];
        if (storage_rank != expected) {
            std::cerr << "high storage count mismatch h=" << h
                      << " got=" << storage_rank << " expected=" << expected << "\n";
            std::exit(31);
        }
    }
    f.high_all_off[MAXW + 1] = uint32_t(f.high_all_codes.size());

    std::cerr
        << "storage_factor low_codes=" << f.low_all_codes.size()
        << " high_codes=" << f.high_all_codes.size()
        << " low_rank_mib=" << double(f.low_packed_rank.size() * sizeof(uint32_t)) / (1 << 20)
        << " high_rank_mib=" << double(f.high_packed_rank.size() * sizeof(uint32_t)) / (1 << 20)
        << "\n";
    return f;
}

struct StorageBlock {
    Code off = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint8_t he = 0;
    uint8_t hs = 0;
    uint8_t c = 0;
    uint8_t valid = 0;
};

struct StorageLayout {
    std::vector<StorageBlock> main_blocks;
    std::vector<StorageBlock> block_blocks;
    Code main_size = 0;
    Code block_size = 0;
};

static StorageLayout build_storage_layout(const StorageFactorHost& f) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;

    StorageLayout s;
    s.main_blocks.resize(3 * (H + 2));
    Code off = 0;
    for (int he = 0; he <= H + 1; ++he) {
        uint32_t hc = f.high_all_off[he + 1] - f.high_all_off[he];
        for (int cv = 0; cv < 3; ++cv) {
            int hs = he + (cv == int(::L) ? 1 : cv == int(::R) ? -1 : 0);
            StorageBlock b;
            b.off = off;
            b.he = uint8_t(he);
            b.c = uint8_t(cv);
            if (hs >= 0 && hs <= L + 1) {
                b.hs = uint8_t(hs);
                b.rows = hc;
                b.cols = f.low_all_off[hs + 1] - f.low_all_off[hs];
                b.valid = 1;
                off += Code(b.rows) * b.cols;
            }
            s.main_blocks[3 * he + cv] = b;
        }
    }
    s.main_size = off;

    s.block_blocks.resize(H + 2);
    off = 0;
    for (int h = 0; h <= H + 1; ++h) {
        StorageBlock b;
        b.off = off;
        b.he = b.hs = uint8_t(h);
        b.valid = 1;
        b.rows = f.high_all_off[h + 1] - f.high_all_off[h];
        b.cols = f.low_all_off[h + 1] - f.low_all_off[h];
        off += Code(b.rows) * b.cols;
        s.block_blocks[h] = b;
    }
    s.block_size = off;

    if (s.main_size != H_DP[TARGET_W][1] || s.block_size != H_DP[TARGET_W - 1][1]) {
        std::cerr << "storage layout size mismatch main=" << s.main_size
                  << "/" << H_DP[TARGET_W][1]
                  << " block=" << s.block_size
                  << "/" << H_DP[TARGET_W - 1][1] << "\n";
        std::exit(32);
    }
    return s;
}

static int seg_end_height_host(uint32_t code, int len) {
    int h = 1;
    for (int p = len - 1; p >= 0; --p) {
        MateValue v = MateValue((code >> (2 * p)) & 3u);
        if (v == R) --h;
        else if (v == ::L) ++h;
    }
    return h;
}

static Code storage_rank_main_host(
    MateID m, const StorageFactorHost& f, const StorageLayout& layout
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LM = (1u << (2 * L)) - 1u;
    constexpr uint32_t HM = (1u << (2 * H)) - 1u;

    uint32_t lc = uint32_t(m) & LM;
    uint32_t hc = uint32_t((m >> (2 * (L + 1))) & HM);
    int he = seg_end_height_host(hc, H);
    int cv = int(mget(m, L));
    const StorageBlock& b = layout.main_blocks[3 * he + cv];
    uint32_t lp = f.low_packed_rank[lc];
    uint32_t hp = f.high_packed_rank[hc];
    if (!b.valid || lp == 0xffffffffu || hp == 0xffffffffu) {
        std::cerr << "invalid main storage rank\n";
        std::exit(33);
    }
    uint32_t lr = lp >> L;
    uint32_t hr = hp >> H;
    return b.off + Code(hr) * b.cols + lr;
}

static Code storage_rank_block_host(
    MateID m, const StorageFactorHost& f, const StorageLayout& layout
) {
    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t LM = (1u << (2 * L)) - 1u;
    constexpr uint32_t HM = (1u << (2 * H)) - 1u;

    uint32_t lc = uint32_t(m) & LM;
    uint32_t hc = uint32_t((m >> (2 * L)) & HM);
    int h = seg_end_height_host(hc, H);
    const StorageBlock& b = layout.block_blocks[h];
    uint32_t lp = f.low_packed_rank[lc];
    uint32_t hp = f.high_packed_rank[hc];
    if (lp == 0xffffffffu || hp == 0xffffffffu) {
        std::cerr << "invalid block storage rank\n";
        std::exit(34);
    }
    uint32_t lr = lp >> L;
    uint32_t hr = hp >> H;
    return b.off + Code(hr) * b.cols + lr;
}

struct StorageDeviceTables {
    uint32_t *low_all = nullptr, *low_mask = nullptr, *low_off = nullptr, *low_rank = nullptr;
    uint32_t *high_all = nullptr, *high_mask = nullptr, *high_off = nullptr, *high_rank = nullptr;

    static void copy_u32(uint32_t** dst, const std::vector<uint32_t>& v, const char* what) {
        if (v.empty()) return;
        ck(cudaMalloc(dst, v.size() * sizeof(uint32_t)), what);
        ck(cudaMemcpy(*dst, v.data(), v.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), what);
    }

    void install(const StorageFactorHost& storage, const FactorTablesHost& base) {
        copy_u32(&low_all, storage.low_all_codes, "storage low all");
        copy_u32(&low_mask, base.low_mask_codes, "storage low mask");
        copy_u32(&low_off, base.low_mask_off, "storage low mask off");
        copy_u32(&low_rank, storage.low_packed_rank, "storage low packed rank");
        copy_u32(&high_all, storage.high_all_codes, "storage high all");
        copy_u32(&high_mask, base.high_mask_codes, "storage high mask");
        copy_u32(&high_off, base.high_mask_off, "storage high mask off");
        copy_u32(&high_rank, storage.high_packed_rank, "storage high packed rank");

        ck(cudaMemcpyToSymbol(D_F_LOW_ALL_CODES, &low_all, sizeof(low_all)), "storage ptr low all");
        ck(cudaMemcpyToSymbol(D_F_LOW_MASK_CODES, &low_mask, sizeof(low_mask)), "storage ptr low mask");
        ck(cudaMemcpyToSymbol(D_F_LOW_MASK_OFF, &low_off, sizeof(low_off)), "storage ptr low off");
        ck(cudaMemcpyToSymbol(D_F_LOW_PACKED_RANK, &low_rank, sizeof(low_rank)), "storage ptr low rank");
        ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_CODES, &high_all, sizeof(high_all)), "storage ptr high all");
        ck(cudaMemcpyToSymbol(D_F_HIGH_MASK_CODES, &high_mask, sizeof(high_mask)), "storage ptr high mask");
        ck(cudaMemcpyToSymbol(D_F_HIGH_MASK_OFF, &high_off, sizeof(high_off)), "storage ptr high off");
        ck(cudaMemcpyToSymbol(D_F_HIGH_PACKED_RANK, &high_rank, sizeof(high_rank)), "storage ptr high rank");
        ck(cudaMemcpyToSymbol(D_F_LOW_ALL_OFF, storage.low_all_off.data(),
                              sizeof(uint32_t) * (MAXW + 2)), "storage low all off");
        ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_OFF, storage.high_all_off.data(),
                              sizeof(uint32_t) * (MAXW + 2)), "storage high all off");
        ck(cudaMemcpyToSymbol(D_FULL_DP, H_DP, sizeof(H_DP)), "full dp");
    }

    void release() {
        if (low_all) cudaFree(low_all);
        if (low_mask) cudaFree(low_mask);
        if (low_off) cudaFree(low_off);
        if (low_rank) cudaFree(low_rank);
        if (high_all) cudaFree(high_all);
        if (high_mask) cudaFree(high_mask);
        if (high_off) cudaFree(high_off);
        if (high_rank) cudaFree(high_rank);
        low_all = low_mask = low_off = low_rank = nullptr;
        high_all = high_mask = high_off = high_rank = nullptr;
    }
};

struct RamFactorCtx {
    uint8_t* arena = nullptr;
    size_t cap_arena = 0;
    Count *dA = nullptr, *dB = nullptr, *dD = nullptr, *dE = nullptr;
    PinnedCountsV3 hM, hD;
    double pack_s = 0, h2d_s = 0, kernel_s = 0, d2h_s = 0, unpack_s = 0;
    uint64_t groups = 0;
    uint64_t memcpy_runs = 0;
    long double memcpy_elems = 0;

    void init(Count mod) {
        ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "copy modulus");
    }

    void ensure(Code m, Code d) {
        auto align256 = [](size_t x) { return (x + 255) & ~size_t(255); };
        size_t mb = align256(size_t(m) * sizeof(Count));
        size_t db = align256(size_t(d) * sizeof(Count));
        size_t need = 2 * mb + 2 * db;
        if (need > cap_arena) {
            if (arena) cudaFree(arena);
            cap_arena = need;
            ck(cudaMalloc(&arena, cap_arena), "ramstream factor scratch");
        }
        size_t off = 0;
        dA = reinterpret_cast<Count*>(arena + off); off += mb;
        dB = reinterpret_cast<Count*>(arena + off); off += mb;
        dD = reinterpret_cast<Count*>(arena + off); off += db;
        dE = reinterpret_cast<Count*>(arena + off);
        hM.ensure(m);
        hD.ensure(d);
    }

    void destroy() {
        if (arena) cudaFree(arena);
        arena = nullptr;
        cap_arena = 0;
        hM.release();
        hD.release();
    }
};

static void copy_rect_to_local(
    const RamCounts& auth, Count* local, const StorageBlock& sb, const FBlock& fb,
    bool fix_low, uint32_t mask, const StorageFactorHost& f,
    int cpu_threads, RamFactorCtx& stats
) {
    constexpr int S = StorageFactorHost::S;
    if (fb.end == fb.off) return;

    if (!fix_low) {
        // Fixed HIGH occupancy: selected rows are consecutive, and every row
        // contains the complete LOW dimension.  The whole rectangle is one run.
        uint32_t row0 = f.high_mask_begin[size_t(mask) * S + fb.he];
        Code elems = fb.end - fb.off;
        std::memcpy(local + fb.off,
                    auth.ptr + sb.off + Code(row0) * sb.cols,
                    size_t(elems) * sizeof(Count));
        ++stats.memcpy_runs;
        stats.memcpy_elems += elems;
        return;
    }

    // Fixed LOW occupancy: each HIGH row contributes one consecutive slice.
    uint32_t col0 = f.low_mask_begin[size_t(mask) * S + fb.hs];
    uint32_t width = fb.stride;
    Code rows = width ? (fb.end - fb.off) / width : 0;
    ram_parallel(rows, cpu_threads, [&](Code r) {
        std::memcpy(local + fb.off + r * width,
                    auth.ptr + sb.off + r * sb.cols + col0,
                    size_t(width) * sizeof(Count));
    });
    stats.memcpy_runs += uint64_t(rows);
    stats.memcpy_elems += Code(rows) * width;
}

static void copy_rect_from_local(
    RamCounts& auth, const Count* local, const StorageBlock& sb, const FBlock& fb,
    bool fix_low, uint32_t mask, const StorageFactorHost& f,
    int cpu_threads, RamFactorCtx& stats
) {
    constexpr int S = StorageFactorHost::S;
    if (fb.end == fb.off) return;

    if (!fix_low) {
        uint32_t row0 = f.high_mask_begin[size_t(mask) * S + fb.he];
        Code elems = fb.end - fb.off;
        std::memcpy(auth.ptr + sb.off + Code(row0) * sb.cols,
                    local + fb.off,
                    size_t(elems) * sizeof(Count));
        ++stats.memcpy_runs;
        stats.memcpy_elems += elems;
        return;
    }

    uint32_t col0 = f.low_mask_begin[size_t(mask) * S + fb.hs];
    uint32_t width = fb.stride;
    Code rows = width ? (fb.end - fb.off) / width : 0;
    ram_parallel(rows, cpu_threads, [&](Code r) {
        std::memcpy(auth.ptr + sb.off + r * sb.cols + col0,
                    local + fb.off + r * width,
                    size_t(width) * sizeof(Count));
    });
    stats.memcpy_runs += uint64_t(rows);
    stats.memcpy_elems += Code(rows) * width;
}

static void pack_main_factor(
    const RamCounts& auth, Count* local, const std::vector<FBlock>& fb,
    bool fix_low, uint32_t mask, const StorageFactorHost& f,
    const StorageLayout& layout, int cpu_threads, RamFactorCtx& stats
) {
    for (size_t i = 0; i < fb.size(); ++i) {
        copy_rect_to_local(auth, local, layout.main_blocks[i], fb[i],
                           fix_low, mask, f, cpu_threads, stats);
    }
}

static void unpack_main_factor(
    RamCounts& auth, const Count* local, const std::vector<FBlock>& fb,
    bool fix_low, uint32_t mask, const StorageFactorHost& f,
    const StorageLayout& layout, int cpu_threads, RamFactorCtx& stats
) {
    for (size_t i = 0; i < fb.size(); ++i) {
        copy_rect_from_local(auth, local, layout.main_blocks[i], fb[i],
                             fix_low, mask, f, cpu_threads, stats);
    }
}

static void pack_block_factor(
    const RamCounts& auth, Count* local, const std::vector<FBlock>& fb,
    bool fix_low, uint32_t mask, const StorageFactorHost& f,
    const StorageLayout& layout, int cpu_threads, RamFactorCtx& stats
) {
    for (size_t i = 0; i < fb.size(); ++i) {
        copy_rect_to_local(auth, local, layout.block_blocks[i], fb[i],
                           fix_low, mask, f, cpu_threads, stats);
    }
}

static void unpack_block_factor(
    RamCounts& auth, const Count* local, const std::vector<FBlock>& fb,
    bool fix_low, uint32_t mask, const StorageFactorHost& f,
    const StorageLayout& layout, int cpu_threads, RamFactorCtx& stats
) {
    for (size_t i = 0; i < fb.size(); ++i) {
        copy_rect_from_local(auth, local, layout.block_blocks[i], fb[i],
                             fix_low, mask, f, cpu_threads, stats);
    }
}

static void process_group_ramfactor(
    RamFactorCtx& c, RamCounts& main_auth, RamCounts& block_auth,
    const StorageFactorHost& storage, const StorageLayout& layout,
    int W, const WindowPlan& wp, int g, int gpu_threads, int cpu_threads
) {
    uint32_t mf, mo, bf, bo;
    window_masks(W, wp.p_hi, wp.p_lo, wp.fixed_pos, uint32_t(g), mf, mo, bf, bo);
    auto ms = make_spec(W, mf, mo);
    auto ds = make_spec(W - 1, bf, bo);
    if (!ms.size && !ds.size) return;

    bool fix_low = wp.p_hi > LOW_LUT_K;
    uint32_t mask = fix_low
        ? (mo & ((1u << LOW_LUT_K) - 1u))
        : ((mo >> (LOW_LUT_K + 1)) & ((1u << HIGH_LUT_K) - 1u));

    auto fmb = make_factor_main_blocks(fix_low, mask);
    auto fdb = make_factor_block_blocks(fix_low, mask);
    if (fmb.empty() || fdb.empty() ||
        fmb.back().end != ms.size || fdb.back().end != ds.size) {
        std::cerr << "ramfactor group size mismatch main="
                  << (fmb.empty() ? 0 : fmb.back().end) << "/" << ms.size
                  << " block=" << (fdb.empty() ? 0 : fdb.back().end) << "/" << ds.size
                  << " fix_low=" << fix_low << " mask=" << mask << "\n";
        std::exit(35);
    }

    int fm = int(fmb.size()), fd = int(fdb.size()), fl = fix_low ? 1 : 0;
    ck(cudaMemcpyToSymbol(D_F_MAIN_BLOCKS, fmb.data(), fmb.size() * sizeof(FBlock)),
       "ramfactor main blocks");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_BLOCKS, fdb.data(), fdb.size() * sizeof(FBlock)),
       "ramfactor block blocks");
    ck(cudaMemcpyToSymbol(D_F_MAIN_NBLOCKS, &fm, sizeof(fm)), "ramfactor main nblocks");
    ck(cudaMemcpyToSymbol(D_F_BLOCK_NBLOCKS, &fd, sizeof(fd)), "ramfactor block nblocks");
    ck(cudaMemcpyToSymbol(D_F_MASK, &mask, sizeof(mask)), "ramfactor mask");
    ck(cudaMemcpyToSymbol(D_F_FIX_LOW, &fl, sizeof(fl)), "ramfactor mode");
    ck(cudaMemcpyToSymbol(D_MAIN_DP, ms.dp, sizeof(ms.dp)), "ramfactor main dp");
    ck(cudaMemcpyToSymbol(D_BLOCK_DP, ds.dp, sizeof(ds.dp)), "ramfactor block dp");
    ck(cudaMemcpyToSymbol(D_MAIN_FIXED, &mf, sizeof(mf)), "ramfactor main fixed");
    ck(cudaMemcpyToSymbol(D_MAIN_OCC, &mo, sizeof(mo)), "ramfactor main occ");
    ck(cudaMemcpyToSymbol(D_BLOCK_FIXED, &bf, sizeof(bf)), "ramfactor block fixed");
    ck(cudaMemcpyToSymbol(D_BLOCK_OCC, &bo, sizeof(bo)), "ramfactor block occ");

    c.ensure(ms.size, ds.size);

    auto t = std::chrono::steady_clock::now();
    if (ms.size) pack_main_factor(main_auth, c.hM.ptr, fmb, fix_low, mask,
                                  storage, layout, cpu_threads, c);
    if (ds.size) pack_block_factor(block_auth, c.hD.ptr, fdb, fix_low, mask,
                                   storage, layout, cpu_threads, c);
    c.pack_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (ms.size) ck(cudaMemcpy(c.dA, c.hM.ptr, size_t(ms.size) * sizeof(Count),
                               cudaMemcpyHostToDevice), "ramfactor H2D main");
    if (ds.size) ck(cudaMemcpy(c.dD, c.hD.ptr, size_t(ds.size) * sizeof(Count),
                               cudaMemcpyHostToDevice), "ramfactor H2D block");
    c.h2d_s += ram_seconds_since(t);

    int bm = int(std::min<Code>(65535, (ms.size + gpu_threads - 1) / gpu_threads));
    int bd = int(std::min<Code>(65535, (ds.size + gpu_threads - 1) / gpu_threads));

    t = std::chrono::steady_clock::now();
    Count* cur = c.dA;
    Count* nxt = c.dB;
    Count* dcur = c.dD;
    Count* dnext = c.dE;
    for (int p = wp.p_hi; p >= wp.p_lo; --p) {
        if (ms.size) ck(cudaMemcpy(nxt, cur, size_t(ms.size) * sizeof(Count),
                                   cudaMemcpyDeviceToDevice), "ramfactor identity");
        if (ds.size) ck(cudaMemset(dnext, 0, size_t(ds.size) * sizeof(Count)),
                        "ramfactor clear block");
        if (ms.size) main_group_kernel<<<bm, gpu_threads>>>(cur, nullptr, ms.size, nxt, dnext, p);
        if (ds.size) blocked_group_kernel<<<bd, gpu_threads>>>(dcur, ds.size, nxt, p);
        ck(cudaGetLastError(), "ramfactor transition");
        std::swap(cur, nxt);
        std::swap(dcur, dnext);
    }
    ck(cudaDeviceSynchronize(), "ramfactor transition sync");
    c.kernel_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (ms.size) ck(cudaMemcpy(c.hM.ptr, cur, size_t(ms.size) * sizeof(Count),
                               cudaMemcpyDeviceToHost), "ramfactor D2H main");
    if (ds.size) ck(cudaMemcpy(c.hD.ptr, dcur, size_t(ds.size) * sizeof(Count),
                               cudaMemcpyDeviceToHost), "ramfactor D2H block");
    c.d2h_s += ram_seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (ms.size) unpack_main_factor(main_auth, c.hM.ptr, fmb, fix_low, mask,
                                    storage, layout, cpu_threads, c);
    if (ds.size) unpack_block_factor(block_auth, c.hD.ptr, fdb, fix_low, mask,
                                     storage, layout, cpu_threads, c);
    c.unpack_s += ram_seconds_since(t);
    ++c.groups;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 4294967291u;
    int target_mib = argc > 3 ? std::atoi(argv[3]) : 4096;
    int max_window = argc > 4 ? std::atoi(argv[4]) : LOW_LUT_K;
    int cpu_threads = argc > 5
        ? std::atoi(argv[5])
        : int(std::max(1u, std::thread::hardware_concurrency()));
    int W = n + 1;

    if (W != TARGET_W || n < 2 || W > MAXW) {
        std::cerr << "binary specialized for n=" << (TARGET_W - 1) << "\n";
        return 1;
    }
    if (target_mib <= 0 || max_window <= 0 || cpu_threads <= 0) {
        std::cerr << "target_mib, max_window and cpu_threads must be positive\n";
        return 1;
    }

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (visible < 1) {
        std::cerr << "need a CUDA GPU\n";
        return 2;
    }
    ck(cudaSetDevice(0), "cudaSetDevice");

    StorageDeviceTables device_tables;
    device_tables.install(storage, G_FACTOR);

    RamCounts main_auth, block_auth;
    main_auth.alloc(layout.main_size, "mmap factorized main");
    block_auth.alloc(layout.block_size, "mmap factorized block");

    MateID init = MateID(R) << (2 * (W - 1));
    main_auth.ptr[storage_rank_main_host(init, storage, layout)] = 1;

    RamFactorCtx ctx;
    ctx.init(mod);

    size_t target = size_t(target_mib) << 20;
    int gpu_threads = 256;
    int total_windows = 0;
    int max_groups = 0;
    auto wall0 = std::chrono::steady_clock::now();

    for (int row = 0; row < W; ++row) {
        int hi = W - 1;
        while (hi >= 1) {
            WindowPlan wp;
            bool found = false;
            for (int lo = std::max(1, hi - max_window + 1); lo <= hi; ++lo) {
                auto candidate = plan_window(W, hi, lo, target);
                if (candidate.max_bytes && candidate.max_bytes <= target) {
                    wp = std::move(candidate);
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::cerr << "cannot fit factorized window hi=" << hi
                          << " target_mib=" << target_mib << "\n";
                return 4;
            }

            int groups = 1 << int(wp.fixed_pos.size());
            max_groups = std::max(max_groups, groups);
            ++total_windows;

            struct Job { int g; Code work; };
            std::vector<Job> jobs;
            jobs.reserve(groups);
            for (int g = 0; g < groups; ++g) {
                uint32_t mf, mo, bf, bo;
                window_masks(W, wp.p_hi, wp.p_lo, wp.fixed_pos, uint32_t(g), mf, mo, bf, bo);
                auto ms = make_spec(W, mf, mo);
                auto ds = make_spec(W - 1, bf, bo);
                jobs.push_back({g, 2 * ms.size + 2 * ds.size});
            }
            std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) {
                return a.work > b.work;
            });

            for (const Job& job : jobs) {
                process_group_ramfactor(ctx, main_auth, block_auth, storage, layout,
                                        W, wp, job.g, gpu_threads, cpu_threads);
            }
            hi = wp.p_lo - 1;
        }
        std::cerr << "row " << (row + 1) << "/" << W
                  << " windows=" << total_windows
                  << " groups=" << ctx.groups
                  << " memcpy_runs=" << ctx.memcpy_runs << "\n";
    }

    double wall_s = ram_seconds_since(wall0);
    MateID final_mate = MateID(R);
    Count answer = main_auth.ptr[storage_rank_main_host(final_mate, storage, layout)];
    double auth_gib = double(layout.main_size + layout.block_size) * sizeof(Count) /
                      double(1ULL << 30);
    double avg_memcpy_elems = ctx.memcpy_runs
        ? double(ctx.memcpy_elems / ctx.memcpy_runs)
        : 0.0;

    std::cout
        << "backend=gridfp-ramstream32-factorized-v3"
        << " n=" << n
        << " residue=" << answer
        << " modulus=" << mod
        << " main_states=" << layout.main_size
        << " blocked_states=" << layout.block_size
        << " auth_gib=" << auth_gib
        << " scratch_target_mib=" << target_mib
        << " max_window=" << max_window
        << " cpu_threads=" << cpu_threads
        << " low_lut_k=" << LOW_LUT_K
        << " high_lut_k=" << HIGH_LUT_K
        << " windows=" << total_windows
        << " max_groups=" << max_groups
        << " groups=" << ctx.groups
        << " memcpy_runs=" << ctx.memcpy_runs
        << " avg_memcpy_elems=" << avg_memcpy_elems
        << " pack_s=" << ctx.pack_s
        << " h2d_s=" << ctx.h2d_s
        << " kernel_s=" << ctx.kernel_s
        << " d2h_s=" << ctx.d2h_s
        << " unpack_s=" << ctx.unpack_s
        << " wall_s=" << wall_s
        << "\n";

    ctx.destroy();
    device_tables.release();
    main_auth.release();
    block_auth.release();
    return 0;
}
