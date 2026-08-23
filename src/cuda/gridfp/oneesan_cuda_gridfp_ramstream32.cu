#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

#include <sys/mman.h>
#include <unistd.h>

using Count = uint32_t;
using MateID = unsigned long long;
using Code = unsigned long long;

static constexpr int MAXW = 28;
#ifndef TARGET_W
#define TARGET_W 28
#endif

enum MateValue : uint8_t { N = 0, R = 1, L = 2, X = 3 };
enum MateValuePair : uint8_t {
    NN = 0x0, NR = 0x1, NL = 0x2, NX = 0x3,
    RN = 0x4, RR = 0x5, RL = 0x6, RX = 0x7,
    LN = 0x8, LR = 0x9, LL = 0xa, LX = 0xb,
    XN = 0xc, XR = 0xd, XL = 0xe, XX = 0xf,
};

static Code H_DP[MAXW + 1][MAXW + 2];
__constant__ Code D_MAIN_DP[MAXW + 1][MAXW + 2];
__constant__ Code D_BLOCK_DP[MAXW + 1][MAXW + 2];
__constant__ uint32_t D_MAIN_FIXED, D_MAIN_OCC, D_BLOCK_FIXED, D_BLOCK_OCC;
__constant__ Count D_MOD;

__host__ __device__ static inline MateValue mget(MateID m, int k) {
    return MateValue((m >> (2 * k)) & 3ULL);
}
__host__ __device__ static inline MateValuePair mpair(MateID m, int p) {
    return MateValuePair((m >> (2 * (p - 1))) & 15ULL);
}
__host__ __device__ static inline MateID mset(MateID m, int k, MateValue v) {
    MateID z = 3ULL << (2 * k);
    return (m & ~z) | (MateID(v) << (2 * k));
}
__host__ __device__ static inline MateID msetpair(MateID m, int p, MateValuePair v) {
    MateID z = 15ULL << (2 * (p - 1));
    return (m & ~z) | (MateID(v) << (2 * (p - 1));
}
__host__ __device__ static inline MateID mshrink(MateID m, int k) {
    MateID mask = (1ULL << (2 * k)) - 1ULL;
    return ((m & ~mask) >> 2) | (m & mask);
}
__host__ __device__ static inline MateID minsert(MateID m, int k, MateValue v) {
    MateID lowmask = k ? ((1ULL << (2 * k)) - 1ULL) : 0ULL;
    MateID lo = m & lowmask, hi = m & ~lowmask;
    return lo | (MateID(v) << (2 * k)) | (hi << 2);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}

static void build_full_dp() {
    for (int h = 0; h <= MAXW + 1; ++h) H_DP[0][h] = (h == 0);
    for (int w = 1; w <= MAXW; ++w) {
        for (int h = 0; h <= MAXW; ++h) {
            Code x = H_DP[w - 1][h];
            if (h > 0) x += H_DP[w - 1][h - 1];
            if (h < MAXW + 1) x += H_DP[w - 1][h + 1];
            H_DP[w][h] = x;
        }
    }
}

struct GroupSpec {
    int width = 0;
    uint32_t fixed = 0, occ = 0;
    Code dp[MAXW + 1][MAXW + 2]{};
    Code size = 0;
};

static GroupSpec make_spec(int width, uint32_t fixed, uint32_t occ) {
    GroupSpec s;
    s.width = width;
    s.fixed = fixed;
    s.occ = occ;
    for (int h = 0; h <= MAXW + 1; ++h) s.dp[0][h] = (h == 0);
    for (int w = 1; w <= width; ++w) {
        int pos = w - 1;
        bool f = (fixed >> pos) & 1u;
        bool o = (occ >> pos) & 1u;
        for (int h = 0; h <= MAXW; ++h) {
            Code x = 0;
            if (!f || !o) x += s.dp[w - 1][h];
            if (!f || o) {
                if (h > 0) x += s.dp[w - 1][h - 1];
                if (h < MAXW + 1) x += s.dp[w - 1][h + 1];
            }
            s.dp[w][h] = x;
        }
    }
    s.size = s.dp[width][1];
    return s;
}

static std::vector<int> window_candidates(int W, int hi, int lo) {
    std::vector<int> v;
    for (int q = W - 1; q >= 0; --q) if (q < lo - 1 || q > hi) v.push_back(q);
    return v;
}

static void window_masks(
    int W, int hi, int lo, const std::vector<int>& fp, uint32_t group,
    uint32_t& mf, uint32_t& mo, uint32_t& bf, uint32_t& bo
) {
    mf = mo = bf = bo = 0;
    for (size_t i = 0; i < fp.size(); ++i) {
        int q = fp[i];
        bool one = (group >> i) & 1u;
        mf |= 1u << q;
        if (one) mo |= 1u << q;
        int bq = (q < lo - 1) ? q : q - 1;
        bf |= 1u << bq;
        if (one) bo |= 1u << bq;
    }
}

struct WindowPlan {
    int p_hi = 0, p_lo = 0;
    std::vector<int> fixed_pos;
    size_t max_bytes = 0;
    Code max_main = 0, max_block = 0;
};

static WindowPlan plan_window(int W, int hi, int lo, size_t target, int maxbits = 20) {
    WindowPlan best;
    best.p_hi = hi;
    best.p_lo = lo;
    auto cand = window_candidates(W, hi, lo);
    int klim = std::min<int>(cand.size(), maxbits);
    for (int k = 0; k <= klim; ++k) {
        std::vector<int> fp(cand.begin(), cand.begin() + k);
        uint64_t ng = 1ULL << k;
        size_t mx = 0;
        Code mm = 0, md = 0;
        for (uint64_t g = 0; g < ng; ++g) {
            uint32_t mf, mo, bf, bo;
            window_masks(W, hi, lo, fp, uint32_t(g), mf, mo, bf, bo);
            auto ms = make_spec(W, mf, mo);
            auto ds = make_spec(W - 1, bf, bo);
            size_t bytes = size_t(2 * ms.size + ds.size) * sizeof(Count);
            if (bytes > mx) {
                mx = bytes;
                mm = ms.size;
                md = ds.size;
            }
            if (mx > target && k < klim) break;
        }
        if (mx <= target || k == klim) {
            best.fixed_pos = std::move(fp);
            best.max_bytes = mx;
            best.max_main = mm;
            best.max_block = md;
            return best;
        }
    }
    return best;
}

__device__ __forceinline__ bool allowed_dev(uint32_t fixed, uint32_t occ, int pos, MateValue v) {
    if (!((fixed >> pos) & 1u)) return v != X;
    bool o = (occ >> pos) & 1u;
    return o ? (v == R || v == L) : (v == N);
}

template<int WIDTH>
__device__ __forceinline__ MateID unrank_group_t(
    Code rank, uint32_t fixed, uint32_t occ,
    const Code dp[MAXW + 1][MAXW + 2]
) {
    MateID m = 0;
    int h = 1;
#pragma unroll
    for (int pos = WIDTH - 1; pos >= 0; --pos) {
        if (allowed_dev(fixed, occ, pos, N)) {
            Code z = dp[pos][h];
            if (rank < z) continue;
            rank -= z;
        }
        if (h > 0 && allowed_dev(fixed, occ, pos, R)) {
            Code z = dp[pos][h - 1];
            if (rank < z) {
                m |= MateID(R) << (2 * pos);
                --h;
                continue;
            }
            rank -= z;
        }
        m |= MateID(L) << (2 * pos);
        ++h;
    }
    return m;
}

template<int WIDTH>
__device__ __forceinline__ Code rank_group_t(
    MateID m, uint32_t fixed, uint32_t occ,
    const Code dp[MAXW + 1][MAXW + 2]
) {
    Code rank = 0;
    int h = 1;
#pragma unroll
    for (int pos = WIDTH - 1; pos >= 0; --pos) {
        MateValue v = mget(m, pos);
        if (v > N && allowed_dev(fixed, occ, pos, N)) rank += dp[pos][h];
        if (v > R && h > 0 && allowed_dev(fixed, occ, pos, R)) rank += dp[pos][h - 1];
        if (v == R) --h;
        else if (v == L) ++h;
    }
    return rank;
}

__device__ __forceinline__ void atomic_add_mod(Count* p, Count v) {
    if (!v) return;
    Count mod = D_MOD;
    Count old = atomicCAS(p, 0u, 0u);
    for (;;) {
        Count neu = (old >= mod - v) ? old - (mod - v) : old + v;
        Count seen = atomicCAS(p, old, neu);
        if (seen == old) return;
        old = seen;
    }
}

__global__ void blocked_group_kernel(const Count* in, Code n, Count* out_main, int p) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code stride = Code(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        Count c = in[i];
        if (!c) continue;
        MateID sm = unrank_group_t<TARGET_W - 1>(i, D_BLOCK_FIXED, D_BLOCK_OCC, D_BLOCK_DP);
        MateID t = minsert(sm, p, N);
        Code j = rank_group_t<TARGET_W>(t, D_MAIN_FIXED, D_MAIN_OCC, D_MAIN_DP);
        atomic_add_mod(out_main + j, c);
    }
}

__global__ void main_group_kernel(const Count* in, Code n, Count* out_main, Count* out_block, int p) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code stride = Code(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        Count c = in[i];
        if (!c) continue;
        MateID m = unrank_group_t<TARGET_W>(i, D_MAIN_FIXED, D_MAIN_OCC, D_MAIN_DP);
        MateValuePair w = mpair(m, p);
        switch (w) {
            case NN: {
                MateID t = msetpair(m, p, LR);
                atomic_add_mod(out_main + rank_group_t<TARGET_W>(t, D_MAIN_FIXED, D_MAIN_OCC, D_MAIN_DP), c);
                break;
            }
            case NR:
            case NL: {
                if (p == 1) {
                    MateID t = msetpair(m, p, w == NR ? RN : LN);
                    atomic_add_mod(out_main + rank_group_t<TARGET_W>(t, D_MAIN_FIXED, D_MAIN_OCC, D_MAIN_DP), c);
                } else {
                    MateID t = mshrink(m, p);
                    atomic_add_mod(out_block + rank_group_t<TARGET_W - 1>(t, D_BLOCK_FIXED, D_BLOCK_OCC, D_BLOCK_DP), c);
                }
                break;
            }
            case RN: {
                MateID t = msetpair(m, p, NR);
                atomic_add_mod(out_main + rank_group_t<TARGET_W>(t, D_MAIN_FIXED, D_MAIN_OCC, D_MAIN_DP), c);
                break;
            }
            case LN: {
                MateID t = msetpair(m, p, NL);
                atomic_add_mod(out_main + rank_group_t<TARGET_W>(t, D_MAIN_FIXED, D_MAIN_OCC, D_MAIN_DP), c);
                break;
            }
            case LL: {
                MateID t = msetpair(m, p, NN);
                int q = p - 1, s = 1;
                while (s) {
                    --q;
                    auto v = mget(t, q);
                    if (v == L) ++s;
                    else if (v == R) --s;
                }
                t = mset(t, q, L);
                if (p == 1) {
                    atomic_add_mod(out_main + rank_group_t<TARGET_W>(t, D_MAIN_FIXED, D_MAIN_OCC, D_MAIN_DP), c);
                } else {
                    t = mshrink(t, p - 1);
                    atomic_add_mod(out_block + rank_group_t<TARGET_W - 1>(t, D_BLOCK_FIXED, D_BLOCK_OCC, D_BLOCK_DP), c);
                }
                break;
            }
            case RR: {
                MateID t = msetpair(m, p, NN);
                int q = p, s = 1;
                while (s) {
                    ++q;
                    auto v = mget(t, q);
                    if (v == L) --s;
                    else if (v == R) ++s;
                }
                t = mset(t, q, R);
                if (p == 1) {
                    atomic_add_mod(out_main + rank_group_t<TARGET_W>(t, D_MAIN_FIXED, D_MAIN_OCC, D_MAIN_DP), c);
                } else {
                    t = mshrink(t, p - 1);
                    atomic_add_mod(out_block + rank_group_t<TARGET_W - 1>(t, D_BLOCK_FIXED, D_BLOCK_OCC, D_BLOCK_DP), c);
                }
                break;
            }
            case RL: {
                MateID t = msetpair(m, p, NN);
                if (p == 1) {
                    atomic_add_mod(out_main + rank_group_t<TARGET_W>(t, D_MAIN_FIXED, D_MAIN_OCC, D_MAIN_DP), c);
                } else {
                    t = mshrink(t, p - 1);
                    atomic_add_mod(out_block + rank_group_t<TARGET_W - 1>(t, D_BLOCK_FIXED, D_BLOCK_OCC, D_BLOCK_DP), c);
                }
                break;
            }
            default:
                break;
        }
    }
}

static bool allowed_host(uint32_t fixed, uint32_t occ, int pos, MateValue v) {
    if (!((fixed >> pos) & 1u)) return v != X;
    bool o = (occ >> pos) & 1u;
    return o ? (v == R || v == L) : (v == N);
}

static MateID unrank_group_host(Code rank, const GroupSpec& s) {
    MateID m = 0;
    int h = 1;
    for (int pos = s.width - 1; pos >= 0; --pos) {
        if (allowed_host(s.fixed, s.occ, pos, N)) {
            Code z = s.dp[pos][h];
            if (rank < z) continue;
            rank -= z;
        }
        if (h > 0 && allowed_host(s.fixed, s.occ, pos, R)) {
            Code z = s.dp[pos][h - 1];
            if (rank < z) {
                m |= MateID(R) << (2 * pos);
                --h;
                continue;
            }
            rank -= z;
        }
        m |= MateID(L) << (2 * pos);
        ++h;
    }
    return m;
}

static Code rank_full(MateID m, int width) {
    Code rank = 0;
    int h = 1;
    for (int pos = width - 1; pos >= 0; --pos) {
        MateValue v = mget(m, pos);
        if (v > N) rank += H_DP[pos][h];
        if (v > R && h > 0) rank += H_DP[pos][h - 1];
        if (v == R) --h;
        else if (v == L) ++h;
    }
    return rank;
}

template<class F>
static void cpu_parallel(Code n, int threads, F&& f) {
    if (!n) return;
    threads = std::max(1, threads);
    if (threads == 1 || n < Code(threads) * 4096) {
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

struct HostCounts {
    Count* ptr = nullptr;
    size_t bytes = 0;

    void alloc(Code n) {
        bytes = size_t(n) * sizeof(Count);
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
#ifdef MAP_NORESERVE
        flags |= MAP_NORESERVE;
#endif
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, flags, -1, 0);
        if (p == MAP_FAILED) {
            perror("mmap authoritative");
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

struct PinnedCounts {
    Count* ptr = nullptr;
    Code cap = 0;

    void ensure(Code n) {
        if (n <= cap) return;
        if (ptr) ck(cudaFreeHost(ptr), "free pinned staging");
        cap = n;
        ck(cudaHostAlloc(reinterpret_cast<void**>(&ptr), size_t(cap) * sizeof(Count), cudaHostAllocPortable), "alloc pinned staging");
    }

    void release() {
        if (ptr) cudaFreeHost(ptr);
        ptr = nullptr;
        cap = 0;
    }
};

struct DeviceCtx {
    Count *dA = nullptr, *dB = nullptr, *dD = nullptr;
    Code capM = 0, capD = 0;
    PinnedCounts hM, hD;
    double pack_s = 0, h2d_s = 0, kernel_s = 0, d2h_s = 0, unpack_s = 0;
    uint64_t groups = 0;

    void init(Count mod) {
        ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "copy modulus");
    }

    void ensure(Code main_n, Code block_n) {
        if (main_n > capM) {
            if (dA) cudaFree(dA);
            if (dB) cudaFree(dB);
            capM = main_n;
            ck(cudaMalloc(&dA, size_t(capM) * sizeof(Count)), "alloc scratch A");
            ck(cudaMalloc(&dB, size_t(capM) * sizeof(Count)), "alloc scratch B");
            hM.ensure(capM);
        }
        if (block_n > capD) {
            if (dD) cudaFree(dD);
            capD = block_n;
            ck(cudaMalloc(&dD, size_t(capD) * sizeof(Count)), "alloc scratch D");
            hD.ensure(capD);
        }
    }

    void destroy() {
        if (dA) cudaFree(dA);
        if (dB) cudaFree(dB);
        if (dD) cudaFree(dD);
        hM.release();
        hD.release();
    }
};

static void gather_group(const HostCounts& auth, Count* staging, const GroupSpec& spec, int cpu_threads) {
    cpu_parallel(spec.size, cpu_threads, [&](Code i) {
        MateID m = unrank_group_host(i, spec);
        staging[i] = auth.ptr[rank_full(m, spec.width)];
    });
}

static void scatter_group(HostCounts& auth, const Count* staging, const GroupSpec& spec, int cpu_threads) {
    cpu_parallel(spec.size, cpu_threads, [&](Code i) {
        MateID m = unrank_group_host(i, spec);
        auth.ptr[rank_full(m, spec.width)] = staging[i];
    });
}

static double seconds_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

static void process_group(
    DeviceCtx& c, HostCounts& main_auth, HostCounts& block_auth,
    int W, const WindowPlan& wp, int g, int gpu_threads, int cpu_threads
) {
    uint32_t mf, mo, bf, bo;
    window_masks(W, wp.p_hi, wp.p_lo, wp.fixed_pos, uint32_t(g), mf, mo, bf, bo);
    auto ms = make_spec(W, mf, mo);
    auto ds = make_spec(W - 1, bf, bo);
    if (!ms.size && !ds.size) return;
    if (!ms.size && ds.size) {
        std::cerr << "invalid group: blocked states without main states\n";
        std::exit(5);
    }

    c.ensure(ms.size, ds.size);

    auto t = std::chrono::steady_clock::now();
    if (ms.size) gather_group(main_auth, c.hM.ptr, ms, cpu_threads);
    if (ds.size) gather_group(block_auth, c.hD.ptr, ds, cpu_threads);
    c.pack_s += seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (ms.size) ck(cudaMemcpy(c.dA, c.hM.ptr, size_t(ms.size) * sizeof(Count), cudaMemcpyHostToDevice), "H2D main");
    if (ds.size) ck(cudaMemcpy(c.dD, c.hD.ptr, size_t(ds.size) * sizeof(Count), cudaMemcpyHostToDevice), "H2D block");
    c.h2d_s += seconds_since(t);

    ck(cudaMemcpyToSymbol(D_MAIN_DP, ms.dp, sizeof(ms.dp)), "main dp");
    ck(cudaMemcpyToSymbol(D_BLOCK_DP, ds.dp, sizeof(ds.dp)), "block dp");
    ck(cudaMemcpyToSymbol(D_MAIN_FIXED, &mf, sizeof(mf)), "main fixed");
    ck(cudaMemcpyToSymbol(D_MAIN_OCC, &mo, sizeof(mo)), "main occ");
    ck(cudaMemcpyToSymbol(D_BLOCK_FIXED, &bf, sizeof(bf)), "block fixed");
    ck(cudaMemcpyToSymbol(D_BLOCK_OCC, &bo, sizeof(bo)), "block occ");

    int bm = int(std::min<Code>(65535, (ms.size + gpu_threads - 1) / gpu_threads));
    int bd = int(std::min<Code>(65535, (ds.size + gpu_threads - 1) / gpu_threads));

    t = std::chrono::steady_clock::now();
    Count* cur = c.dA;
    Count* nxt = c.dB;
    for (int p = wp.p_hi; p >= wp.p_lo; --p) {
        if (ms.size) ck(cudaMemcpy(nxt, cur, size_t(ms.size) * sizeof(Count), cudaMemcpyDeviceToDevice), "identity");
        if (ds.size) blocked_group_kernel<<<bd, gpu_threads>>>(c.dD, ds.size, nxt, p);
        if (ds.size) ck(cudaMemset(c.dD, 0, size_t(ds.size) * sizeof(Count)), "clear block scratch");
        if (ms.size) main_group_kernel<<<bm, gpu_threads>>>(cur, ms.size, nxt, c.dD, p);
        ck(cudaGetLastError(), "transition kernel");
        std::swap(cur, nxt);
    }
    ck(cudaDeviceSynchronize(), "transition sync");
    c.kernel_s += seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (ms.size) ck(cudaMemcpy(c.hM.ptr, cur, size_t(ms.size) * sizeof(Count), cudaMemcpyDeviceToHost), "D2H main");
    if (ds.size) ck(cudaMemcpy(c.hD.ptr, c.dD, size_t(ds.size) * sizeof(Count), cudaMemcpyDeviceToHost), "D2H block");
    c.d2h_s += seconds_since(t);

    t = std::chrono::steady_clock::now();
    if (ms.size) scatter_group(main_auth, c.hM.ptr, ms, cpu_threads);
    if (ds.size) scatter_group(block_auth, c.hD.ptr, ds, cpu_threads);
    c.unpack_s += seconds_since(t);
    ++c.groups;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    Count mod = argc > 2 ? Count(std::strtoul(argv[2], nullptr, 10)) : 2147483647u;
    int target_mib = argc > 3 ? std::atoi(argv[3]) : 4096;
    int max_window = argc > 4 ? std::atoi(argv[4]) : 14;
    int cpu_threads = argc > 5 ? std::atoi(argv[5]) : int(std::max(1u, std::thread::hardware_concurrency()));
    int W = n + 1;

    if (n < 2 || W > MAXW) {
        std::cerr << "n=2..27\n";
        return 1;
    }
    if (W != TARGET_W) {
        std::cerr << "specialized for width " << TARGET_W << " (n=" << (TARGET_W - 1) << ")\n";
        return 1;
    }
    if (target_mib <= 0 || max_window <= 0 || cpu_threads <= 0) {
        std::cerr << "target_mib, max_window and cpu_threads must be positive\n";
        return 1;
    }

    build_full_dp();
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "cudaGetDeviceCount");
    if (visible < 1) {
        std::cerr << "need a CUDA GPU\n";
        return 2;
    }
    ck(cudaSetDevice(0), "cudaSetDevice");

    Code mainN = H_DP[W][1];
    Code blockN = H_DP[W - 1][1];
    HostCounts main_auth, block_auth;
    main_auth.alloc(mainN);
    block_auth.alloc(blockN);

    MateID init = MateID(R) << (2 * (W - 1));
    main_auth.ptr[rank_full(init, W)] = 1;

    DeviceCtx ctx;
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
                std::cerr << "cannot fit window hi=" << hi << " target_mib=" << target_mib << "\n";
                return 4;
            }

            int groups = 1 << int(wp.fixed_pos.size());
            max_groups = std::max(max_groups, groups);
            ++total_windows;

            struct Job { int group; Code work; };
            std::vector<Job> jobs;
            jobs.reserve(groups);
            for (int g = 0; g < groups; ++g) {
                uint32_t mf, mo, bf, bo;
                window_masks(W, wp.p_hi, wp.p_lo, wp.fixed_pos, uint32_t(g), mf, mo, bf, bo);
                auto ms = make_spec(W, mf, mo);
                auto ds = make_spec(W - 1, bf, bo);
                jobs.push_back({g, 2 * ms.size + ds.size});
            }
            std::sort(jobs.begin(), jobs.end(), [](const Job& a, const Job& b) { return a.work > b.work; });

            for (const auto& job : jobs) {
                process_group(ctx, main_auth, block_auth, W, wp, job.group, gpu_threads, cpu_threads);
            }
            hi = wp.p_lo - 1;
        }
        std::cerr << "row " << (row + 1) << "/" << W
                  << " windows=" << total_windows
                  << " groups=" << ctx.groups << "\n";
    }

    double wall_s = seconds_since(wall0);
    Code final_rank = rank_full(MateID(R), W);
    Count answer = main_auth.ptr[final_rank];
    double auth_gib = double(mainN + blockN) * sizeof(Count) / double(1ULL << 30);

    std::cout
        << "backend=gridfp-ramstream32-v1"
        << " n=" << n
        << " residue=" << answer
        << " modulus=" << mod
        << " main_states=" << mainN
        << " blocked_states=" << blockN
        << " auth_gib=" << auth_gib
        << " scratch_target_mib=" << target_mib
        << " max_window=" << max_window
        << " cpu_threads=" << cpu_threads
        << " windows=" << total_windows
        << " max_groups=" << max_groups
        << " groups=" << ctx.groups
        << " pack_s=" << ctx.pack_s
        << " h2d_s=" << ctx.h2d_s
        << " kernel_s=" << ctx.kernel_s
        << " d2h_s=" << ctx.d2h_s
        << " unpack_s=" << ctx.unpack_s
        << " wall_s=" << wall_s
        << "\n";

    ctx.destroy();
    main_auth.release();
    block_auth.release();
    return 0;
}
