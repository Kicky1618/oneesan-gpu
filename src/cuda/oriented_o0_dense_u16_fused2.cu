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


__device__ __forceinline__ std::uint32_t gate_component_warp(std::uint32_t self,
                                                               int out_k,
                                                               int lane_base,
                                                               int lane_step) {
    const unsigned mask = 0xffffffffu;
    // Every lane executes the same shuffle sequence; only the source lane differs.
    const auto a0 = __shfl_sync(mask, self, lane_base + lane_step * 0);
    const auto a1 = __shfl_sync(mask, self, lane_base + lane_step * 1);
    const auto a2 = __shfl_sync(mask, self, lane_base + lane_step * 2);
    const auto a3 = __shfl_sync(mask, self, lane_base + lane_step * 3);
    const auto a5 = __shfl_sync(mask, self, lane_base + lane_step * 5);
    const auto a6 = __shfl_sync(mask, self, lane_base + lane_step * 6);
    const auto a7 = __shfl_sync(mask, self, lane_base + lane_step * 7);
    switch (out_k) {
        case 0: return add_mod(a0, add_mod(mul_mod(a7, ZETA), mul_mod(a5, ZETA_INV)));
        case 1: return add_mod(a3, mul_mod(a1, ZETA_INV));
        case 2: return add_mod(a6, mul_mod(a2, ZETA));
        case 3: return add_mod(mul_mod(a3, ZETA), a1);
        case 4: return 0;
        case 5: return mul_mod(a0, ZETA_INV);
        case 6: return add_mod(mul_mod(a6, ZETA_INV), a2);
        case 7: return mul_mod(a0, ZETA);
        default: return 0;
    }
}

// Fuse two consecutive interior vertices. One warp owns the 27 amplitudes of
// three adjacent trits p,p+1,p+2. The first 9x9 gate acts on p,p+1, then the
// second on p+1,p+2; the intermediate vector never touches global memory.
__global__ void local_gate_fused2(const std::uint16_t* __restrict__ in,
                                  std::uint16_t* __restrict__ out,
                                  std::uint64_t groups,
                                  std::uint64_t stride) {
    const unsigned lane = threadIdx.x & 31u;
    const std::uint64_t warp = (std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x) >> 5;
    if (warp >= groups) return;

    const std::uint64_t low = warp % stride;
    const std::uint64_t high = warp / stride;
    const std::uint64_t base = high * (27 * stride) + low;

    std::uint32_t v = 0;
    if (lane < 27) v = in[base + std::uint64_t{lane} * stride];

    // Gate 1 on digits (d0,d1), holding d2 fixed. Local k = d0 + 3*d1.
    const bool valid = lane < 27;
    const int d0 = valid ? int(lane % 3) : 0;
    const int d1 = valid ? int((lane / 3) % 3) : 0;
    const int d2 = valid ? int(lane / 9) : 0;
    const int k01 = d0 + 3 * d1;
    const int base01 = 9 * d2;
    std::uint32_t mid = gate_component_warp(v, k01, base01, 1);
    if (!valid) mid = 0;

    // Gate 2 on digits (d1,d2), holding d0 fixed. For source local index k,
    // global warp lane is d0 + 3*(k%3) + 9*(k/3) = d0 + 3*k.
    const int k12 = d1 + 3 * d2;
    std::uint32_t fin = gate_component_warp(mid, k12, d0, 3);
    if (!valid) fin = 0;

    if (lane < 27) out[base + std::uint64_t{lane} * stride] = static_cast<std::uint16_t>(fin);
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

        // Interior rows: fuse two adjacent full gates at a time. The bottom row
        // needs down=0 masks, and the right boundary needs right=0, so those
        // remain on the scalar gate path.
        if (y + 1 < n) {
            while (x + 2 < n) {
                const std::uint64_t stride = pow3(x);
                const std::uint64_t groups = total / 27;
                constexpr int fused_threads = 128;
                const std::uint64_t warps_per_block = fused_threads / 32;
                const unsigned blocks = static_cast<unsigned>((groups + warps_per_block - 1) / warps_per_block);
                local_gate_fused2<<<blocks, fused_threads>>>(a, b, groups, stride);
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
