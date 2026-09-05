#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

constexpr std::uint32_t MOD = 65521;
constexpr std::uint32_t ZETA = 61640;
constexpr std::uint32_t ZETA_INV = 19685; // ZETA^15 mod MOD

#define CUDA_CHECK(expr)                                                                       \
    do {                                                                                       \
        const cudaError_t err__ = (expr);                                                      \
        if (err__ != cudaSuccess) {                                                            \
            throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(err__)); \
        }                                                                                      \
    } while (0)

__device__ __forceinline__ std::uint32_t add_mod(std::uint32_t a, std::uint32_t b) {
    std::uint32_t x = a + b;
    if (x >= MOD) x -= MOD;
    return x;
}

// MOD = 2^16 - 15, so 2^16 == 15 (mod MOD).  Inputs are < MOD.
__device__ __forceinline__ std::uint32_t mul_mod(std::uint32_t a, std::uint32_t b) {
    const std::uint32_t x = a * b;
    std::uint32_t r = (x & 0xffffu) + 15u * (x >> 16);
    r = (r & 0xffffu) + 15u * (r >> 16);
    if (r >= MOD) r -= MOD;
    return r;
}

__global__ void init_source(std::uint16_t* a, std::uint64_t total) {
    const std::uint64_t i = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < total) a[i] = 0;
    if (i == 0) {
        // After processing the top-left source:
        // digit 0 = down(+): endpoint correction zeta^-1
        // digit 1 = right(+): endpoint correction 1.
        a[1] = ZETA_INV;
        a[3] = 1;
    }
}

// Apply the 13-nonzero local oriented-loop gate to adjacent trits p,p+1.
// One thread handles all 9 local input/output configurations for fixed other trits.
__global__ void local_gate(const std::uint16_t* __restrict__ in,
                           std::uint16_t* __restrict__ out,
                           std::uint64_t groups,
                           std::uint64_t stride,
                           std::uint16_t allowed_mask) {
    const std::uint64_t g = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (g >= groups) return;

    const std::uint64_t low = g % stride;
    const std::uint64_t high = g / stride;
    const std::uint64_t base = high * (9 * stride) + low;

    const std::uint32_t a0 = in[base + 0 * stride];
    const std::uint32_t a1 = in[base + 1 * stride];
    const std::uint32_t a2 = in[base + 2 * stride];
    const std::uint32_t a3 = in[base + 3 * stride];
    const std::uint32_t a5 = in[base + 5 * stride];
    const std::uint32_t a6 = in[base + 6 * stride];
    const std::uint32_t a7 = in[base + 7 * stride];

    // Local index is down + 3*right.  These equations are the complete
    // 9x9 degree-(0 or 2), flow-conserving vertex tensor.
    std::uint32_t o0 = add_mod(a0, add_mod(mul_mod(a7, ZETA), mul_mod(a5, ZETA_INV)));
    std::uint32_t o1 = add_mod(a3, mul_mod(a1, ZETA_INV));
    std::uint32_t o2 = add_mod(a6, mul_mod(a2, ZETA));
    std::uint32_t o3 = add_mod(mul_mod(a3, ZETA), a1);
    constexpr std::uint32_t o4 = 0;
    std::uint32_t o5 = mul_mod(a0, ZETA_INV);
    std::uint32_t o6 = add_mod(mul_mod(a6, ZETA_INV), a2);
    std::uint32_t o7 = mul_mod(a0, ZETA);
    constexpr std::uint32_t o8 = 0;

    const std::uint32_t v[9] = {o0, o1, o2, o3, o4, o5, o6, o7, o8};
#pragma unroll
    for (int k = 0; k < 9; ++k) {
        out[base + std::uint64_t{k} * stride] = (allowed_mask & (1u << k)) ? v[k] : 0u;
    }
}


struct V9 {
    std::uint32_t v0,v1,v2,v3,v4,v5,v6,v7,v8;
};

__device__ __forceinline__ V9 gate9(V9 a) {
    return {
        add_mod(a.v0, add_mod(mul_mod(a.v7, ZETA), mul_mod(a.v5, ZETA_INV))),
        add_mod(a.v3, mul_mod(a.v1, ZETA_INV)),
        add_mod(a.v6, mul_mod(a.v2, ZETA)),
        add_mod(mul_mod(a.v3, ZETA), a.v1),
        0,
        mul_mod(a.v0, ZETA_INV),
        add_mod(mul_mod(a.v6, ZETA_INV), a.v2),
        mul_mod(a.v0, ZETA),
        0
    };
}

// Fuse two consecutive full interior gates. A thread owns all 27 amplitudes
// for trits p,p+1,p+2 and keeps the intermediate 27-vector in registers.
__global__ void local_gate_fused2_thread(const std::uint16_t* __restrict__ in,
                                         std::uint16_t* __restrict__ out,
                                         std::uint64_t groups,
                                         std::uint64_t stride) {
    const std::uint64_t g = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const std::uint64_t low = g % stride;
    const std::uint64_t high = g / stride;
    const std::uint64_t base = high * (27 * stride) + low;
#define LD(k) static_cast<std::uint32_t>(in[base + std::uint64_t{k} * stride])
#define ST(k,x) out[base + std::uint64_t{k} * stride] = static_cast<std::uint16_t>(x)
    const V9 m0 = gate9({LD(0),LD(1),LD(2),LD(3),LD(4),LD(5),LD(6),LD(7),LD(8)});
    const V9 m1 = gate9({LD(9),LD(10),LD(11),LD(12),LD(13),LD(14),LD(15),LD(16),LD(17)});
    const V9 m2 = gate9({LD(18),LD(19),LD(20),LD(21),LD(22),LD(23),LD(24),LD(25),LD(26)});

    // Second gate is on digits 1,2 at fixed digit 0.
    const V9 o0 = gate9({m0.v0,m0.v3,m0.v6,m1.v0,m1.v3,m1.v6,m2.v0,m2.v3,m2.v6});
    const V9 o1 = gate9({m0.v1,m0.v4,m0.v7,m1.v1,m1.v4,m1.v7,m2.v1,m2.v4,m2.v7});
    const V9 o2 = gate9({m0.v2,m0.v5,m0.v8,m1.v2,m1.v5,m1.v8,m2.v2,m2.v5,m2.v8});

    // Scatter local k=d1+3*d2 back to global k=d0+3*d1+9*d2.
    ST(0,o0.v0); ST(3,o0.v1); ST(6,o0.v2); ST(9,o0.v3); ST(12,o0.v4); ST(15,o0.v5); ST(18,o0.v6); ST(21,o0.v7); ST(24,o0.v8);
    ST(1,o1.v0); ST(4,o1.v1); ST(7,o1.v2); ST(10,o1.v3); ST(13,o1.v4); ST(16,o1.v5); ST(19,o1.v6); ST(22,o1.v7); ST(25,o1.v8);
    ST(2,o2.v0); ST(5,o2.v1); ST(8,o2.v2); ST(11,o2.v3); ST(14,o2.v4); ST(17,o2.v5); ST(20,o2.v6); ST(23,o2.v7); ST(26,o2.v8);
#undef LD
#undef ST
}


// Fuse three consecutive full interior gates. Three independent 81-amplitude
// groups are packed into each warp (lanes 0..8, 9..17, 18..26). Intermediate
// values live in shared memory; global memory is touched only once at each end.
__global__ void local_gate_fused3(const std::uint16_t* __restrict__ in,
                                  std::uint16_t* __restrict__ out,
                                  std::uint64_t groups,
                                  std::uint64_t stride) {
    constexpr unsigned groups_per_warp = 3;
    constexpr unsigned active_lanes = 27;
    constexpr unsigned values_per_group = 81;
    constexpr unsigned warps_per_block = 8; // launched with 256 threads
    __shared__ std::uint16_t sh[warps_per_block * groups_per_warp * values_per_group];

    const unsigned lane = threadIdx.x & 31u;
    if (lane >= active_lanes) return;
    const unsigned warp_local = threadIdx.x >> 5;
    const unsigned sub = lane / 9;
    const unsigned q = lane % 9;
    const std::uint64_t warp_global = (std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x) >> 5;
    const std::uint64_t g = warp_global * groups_per_warp + sub;
    if (g >= groups) return;

    const unsigned sync_mask = 0x1ffu << (sub * 9);
    const unsigned sbase = (warp_local * groups_per_warp + sub) * values_per_group;
    const std::uint64_t low = g % stride;
    const std::uint64_t high = g / stride;
    const std::uint64_t base = high * (values_per_group * stride) + low;
#define GLD(k) static_cast<std::uint32_t>(in[base + std::uint64_t{k} * stride])
#define GST(k,x) out[base + std::uint64_t{k} * stride] = static_cast<std::uint16_t>(x)
#define SLD(k) static_cast<std::uint32_t>(sh[sbase + (k)])
#define SST(k,x) sh[sbase + (k)] = static_cast<std::uint16_t>(x)

    // Stage 1: gate on digits 0,1, fixed (d2,d3)=q.
    const int b1 = 9 * q;
    V9 a = {GLD(b1+0),GLD(b1+1),GLD(b1+2),GLD(b1+3),GLD(b1+4),GLD(b1+5),GLD(b1+6),GLD(b1+7),GLD(b1+8)};
    V9 b = gate9(a);
    SST(b1+0,b.v0); SST(b1+1,b.v1); SST(b1+2,b.v2); SST(b1+3,b.v3); SST(b1+4,b.v4); SST(b1+5,b.v5); SST(b1+6,b.v6); SST(b1+7,b.v7); SST(b1+8,b.v8);
    __syncwarp(sync_mask);

    // Stage 2: gate on digits 1,2, fixed (d0,d3)=q.
    const int d0 = q % 3, d3 = q / 3;
#define I2(j) (d0 + 3*(j) + 27*d3)
    a = {SLD(I2(0)),SLD(I2(1)),SLD(I2(2)),SLD(I2(3)),SLD(I2(4)),SLD(I2(5)),SLD(I2(6)),SLD(I2(7)),SLD(I2(8))};
    b = gate9(a);
    __syncwarp(sync_mask); // all stage-2 inputs have been captured
    SST(I2(0),b.v0); SST(I2(1),b.v1); SST(I2(2),b.v2); SST(I2(3),b.v3); SST(I2(4),b.v4); SST(I2(5),b.v5); SST(I2(6),b.v6); SST(I2(7),b.v7); SST(I2(8),b.v8);
    __syncwarp(sync_mask);
#undef I2

    // Stage 3: gate on digits 2,3, fixed (d0,d1)=q; write straight out.
    const int d0b = q % 3, d1 = q / 3;
#define I3(j) (d0b + 3*d1 + 9*(j))
    a = {SLD(I3(0)),SLD(I3(1)),SLD(I3(2)),SLD(I3(3)),SLD(I3(4)),SLD(I3(5)),SLD(I3(6)),SLD(I3(7)),SLD(I3(8))};
    b = gate9(a);
    GST(I3(0),b.v0); GST(I3(1),b.v1); GST(I3(2),b.v2); GST(I3(3),b.v3); GST(I3(4),b.v4); GST(I3(5),b.v5); GST(I3(6),b.v6); GST(I3(7),b.v7); GST(I3(8),b.v8);
#undef I3
#undef GLD
#undef GST
#undef SLD
#undef SST
}

// At the end of a row, right-carry digit n is guaranteed zero.  Reinterpret
// down[0..n-1] as up[0..n-1] for the next row by shifting all trits one slot.
__global__ void row_shift(const std::uint16_t* __restrict__ in,
                          std::uint16_t* __restrict__ out,
                          std::uint64_t total,
                          std::uint64_t valid_old) {
    const std::uint64_t i = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i >= total) return;
    if (i % 3u == 0u) {
        const std::uint64_t old = i / 3u;
        out[i] = old < valid_old ? in[old] : 0u;
    } else {
        out[i] = 0u;
    }
}

std::uint64_t pow3(unsigned e) {
    std::uint64_t x = 1;
    while (e--) x *= 3;
    return x;
}

struct SolveResult {
    std::uint32_t residue = 0;
    float kernel_ms = 0;
    double wall_ms = 0;
    std::uint64_t states = 0;
};

SolveResult solve(unsigned n) {
    if (n < 2 || n > 18) throw std::runtime_error("n must be in [2,18]");
    const std::uint64_t total = pow3(n + 1);
    const std::uint64_t bytes = total * sizeof(std::uint16_t);

    std::uint16_t *a = nullptr, *b = nullptr;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));

    constexpr int threads = 256;
    auto blocks_for = [&](std::uint64_t count) {
        return static_cast<unsigned>((count + threads - 1) / threads);
    };

    cudaEvent_t ev0{}, ev1{};
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    const auto wall0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(ev0));
    init_source<<<blocks_for(total), threads>>>(a, total);
    CUDA_CHECK(cudaGetLastError());

    // source (0,0) was applied by init_source.
    for (unsigned y = 0; y < n; ++y) {
        unsigned x = (y == 0 ? 1u : 0u);
        if (y + 1 < n) {
            while (x + 3 < n) {
                const std::uint64_t stride = pow3(x);
                const std::uint64_t groups = total / 81;
                constexpr std::uint64_t groups_per_block = 8 * 3;
                const unsigned blocks = static_cast<unsigned>((groups + groups_per_block - 1) / groups_per_block);
                local_gate_fused3<<<blocks, threads>>>(a, b, groups, stride);
                CUDA_CHECK(cudaGetLastError());
                std::swap(a, b);
                x += 3;
            }
            if (x + 2 < n) {
                const std::uint64_t stride = pow3(x);
                const std::uint64_t groups = total / 27;
                local_gate_fused2_thread<<<blocks_for(groups), threads>>>(a, b, groups, stride);
                CUDA_CHECK(cudaGetLastError());
                std::swap(a, b);
                x += 2;
            }
        }
        for (; x < n; ++x) {
            if (y + 1 == n && x + 1 == n) break; // target evaluated separately
            const std::uint64_t stride = pow3(x);
            const std::uint64_t groups = total / 9;
            const bool allow_down = y + 1 < n;
            const bool allow_right = x + 1 < n;
            std::uint16_t mask = 0x1ffu;
            if (!allow_down) mask &= (1u << 0) | (1u << 3) | (1u << 6); // down=0
            if (!allow_right) mask &= (1u << 0) | (1u << 1) | (1u << 2); // right=0
            local_gate<<<blocks_for(groups), threads>>>(a, b, groups, stride, mask);
            CUDA_CHECK(cudaGetLastError());
            std::swap(a, b);
        }

        if (y + 1 < n) {
            const std::uint64_t valid_old = pow3(n); // highest (right-carry) trit is zero
            row_shift<<<blocks_for(total), threads>>>(a, b, total, valid_old);
            CUDA_CHECK(cudaGetLastError());
            std::swap(a, b);
        }
    }

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    const auto wall1 = std::chrono::steady_clock::now();

    // Before target: either the left edge (digit n-1) or up edge (digit n)
    // is the unique + arrow.  Target endpoint correction is 1 or zeta.
    std::uint16_t left16 = 0, up16 = 0;
    const std::uint64_t left_idx = pow3(n - 1);
    const std::uint64_t up_idx = pow3(n);
    CUDA_CHECK(cudaMemcpy(&left16, a + left_idx, sizeof(left16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&up16, a + up_idx, sizeof(up16), cudaMemcpyDeviceToHost));
    const std::uint32_t left = left16, up = up16;
    std::uint32_t residue = left + static_cast<std::uint32_t>((std::uint64_t{up} * ZETA) % MOD);
    if (residue >= MOD) residue -= MOD;

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFree(a));
    CUDA_CHECK(cudaFree(b));

    return {residue, ms,
            std::chrono::duration<double, std::milli>(wall1 - wall0).count(), total};
}

} // namespace

int main(int argc, char** argv) {
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cerr << "gpu=" << prop.name << " mod=" << MOD << " zeta=" << ZETA << '\n';

    unsigned first = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 2;
    unsigned last = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : first;
    unsigned reps = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 3;

    // Pay CUDA context/lazy module-loading costs before timed runs.
    (void)solve(2);

    for (unsigned n = first; n <= last; ++n) {
        try {
            SolveResult best{};
            best.kernel_ms = 1e30f;
            double sum_ms = 0.0;
            for (unsigned rep = 0; rep < reps; ++rep) {
                const auto r = solve(n);
                sum_ms += r.kernel_ms;
                if (r.kernel_ms < best.kernel_ms) best = r;
            }
            std::cout << "n=" << n << " residue=" << best.residue
                      << " dense_states=" << best.states
                      << " best_ms=" << std::fixed << std::setprecision(3) << best.kernel_ms
                      << " avg_ms=" << (sum_ms / reps) << '\n';
        } catch (const std::exception& e) {
            std::cerr << "n=" << n << " error: " << e.what() << '\n';
            return 1;
        }
    }
}
