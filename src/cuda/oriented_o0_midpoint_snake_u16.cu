#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include "o0_midpoint_tables.hpp"

namespace {
constexpr std::uint32_t MOD = 65521;
constexpr std::uint32_t ZETA = 61640;
constexpr std::uint32_t ZETA_INV = 19685;

#define CUDA_CHECK(expr) do { \
    const cudaError_t err__ = (expr); \
    if (err__ != cudaSuccess) throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(err__)); \
} while (0)

__constant__ std::uint16_t D_CF[144];
__constant__ std::uint16_t D_CR[144];
__constant__ std::uint16_t D_JFR[16];
__constant__ std::uint16_t D_JRF[16];
__constant__ std::uint16_t D_BF[36];

__device__ __forceinline__ std::uint32_t reduce_mod(std::uint32_t x) {
    std::uint32_t r = (x & 0xffffu) + 15u * (x >> 16);
    r = (r & 0xffffu) + 15u * (r >> 16);
    if (r >= MOD) r -= MOD;
    return r;
}

__device__ __forceinline__ std::uint32_t mul_mod(std::uint32_t a, std::uint32_t b) {
    return reduce_mod(a * b);
}

__device__ __forceinline__ std::uint32_t add2(std::uint32_t a, std::uint32_t b) {
    std::uint32_t x = a + b; if (x >= MOD) x -= MOD; return x;
}
__device__ __forceinline__ std::uint32_t add3(std::uint32_t a, std::uint32_t b, std::uint32_t c) {
    return add2(add2(a, b), c);
}

template <bool REV, bool BOTTOM = false>
__global__ void apply12_sparse(std::uint16_t* v, std::uint64_t groups, std::uint64_t stride) {
    const std::uint64_t g = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const std::uint64_t low = g % stride;
    const std::uint64_t high = g / stride;
    const std::uint64_t base = high * (12 * stride) + low;
    std::uint32_t x[12];
#pragma unroll
    for (int i = 0; i < 12; ++i) x[i] = v[base + std::uint64_t(i) * stride];
    std::uint32_t o[12]{};
    if constexpr (!REV) {
        o[0]  = add3(x[0], x[9], mul_mod(ZETA, x[10]));
        o[1]  = add2(mul_mod(57852, x[1]), x[11]);
        o[2]  = x[8];
        o[3]  = add2(x[1], mul_mod(ZETA, x[2]));
        o[4]  = x[3];
        o[5]  = x[0];
        o[6]  = add2(mul_mod(53583, x[4]), mul_mod(ZETA, x[11]));
        o[7]  = mul_mod(ZETA_INV, x[5]);
        o[8]  = mul_mod(ZETA_INV, x[9]);
        o[9]  = add3(x[3], x[5], mul_mod(ZETA, x[6]));
        o[10] = x[7];
        o[11] = add2(mul_mod(8031, x[1]), x[4]);
    } else {
        o[0]  = add3(x[0], x[5], mul_mod(ZETA_INV, x[8]));
        o[1]  = add2(x[3], mul_mod(ZETA_INV, x[6]));
        o[2]  = add2(mul_mod(16855, x[1]), mul_mod(ZETA_INV, x[11]));
        o[3]  = add3(x[4], mul_mod(ZETA_INV, x[7]), x[9]);
        o[4]  = add2(mul_mod(8031, x[3]), x[11]);
        o[5]  = x[9];
        o[6]  = mul_mod(ZETA, x[4]);
        o[7]  = x[10];
        o[8]  = x[2];
        o[9]  = x[0];
        o[10] = mul_mod(ZETA, x[5]);
        o[11] = add2(x[1], mul_mod(57852, x[3]));
    }
#pragma unroll
    for (int i = 0; i < 12; ++i) {
        if constexpr (BOTTOM) v[base + std::uint64_t(i) * stride] = (i % 3 == 0) ? o[i] : 0;
        else v[base + std::uint64_t(i) * stride] = o[i];
    }
}

template <bool REV>
__global__ void apply4_sparse(std::uint16_t* v, std::uint64_t groups, std::uint64_t stride) {
    const std::uint64_t g = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const std::uint64_t low = g % stride;
    const std::uint64_t high = g / stride;
    const std::uint64_t base = high * (4 * stride) + low;
    const std::uint32_t x0=v[base], x1=v[base+stride], x2=v[base+2*stride], x3=v[base+3*stride];
    v[base] = x0;
    v[base+stride] = add2(x1, mul_mod(REV ? ZETA_INV : ZETA, x2));
    v[base+2*stride] = 0;
    v[base+3*stride] = x3;
}

__global__ void init_state(std::uint16_t* v, std::uint64_t total) {
    const std::uint64_t i = std::uint64_t{blockIdx.x} * blockDim.x + threadIdx.x;
    if (i < total) v[i] = 0;
}

std::uint64_t pow3(unsigned e) {
    std::uint64_t x = 1;
    while (e--) x *= 3;
    return x;
}

struct Result {
    std::uint32_t residue = 0;
    std::uint64_t states = 0;
    float kernel_ms = 0;
    double wall_ms = 0;
};

void upload_tables() {
    CUDA_CHECK(cudaMemcpyToSymbol(D_CF, o0mid::C_F, sizeof(o0mid::C_F)));
    CUDA_CHECK(cudaMemcpyToSymbol(D_CR, o0mid::C_R, sizeof(o0mid::C_R)));
    CUDA_CHECK(cudaMemcpyToSymbol(D_JFR, o0mid::J_FR, sizeof(o0mid::J_FR)));
    CUDA_CHECK(cudaMemcpyToSymbol(D_JRF, o0mid::J_RF, sizeof(o0mid::J_RF)));
    CUDA_CHECK(cudaMemcpyToSymbol(D_BF, o0mid::B_F, sizeof(o0mid::B_F)));
}

Result solve(unsigned n) {
    if (n < 3 || !(n & 1) || n > 19) throw std::runtime_error("odd n in [3,19] only");
    const std::uint64_t stride_last = pow3(n - 2);
    const std::uint64_t total = 4 * pow3(n - 1);
    const std::uint64_t bytes = total * sizeof(std::uint16_t);

    std::uint16_t* v = nullptr;
    CUDA_CHECK(cudaMalloc(&v, bytes));
    constexpr int threads = 256;
    auto blocks = [&](std::uint64_t count) { return static_cast<unsigned>((count + threads - 1) / threads); };

    cudaEvent_t ev0{}, ev1{};
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    const auto wall0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(ev0));

    init_state<<<blocks(total), threads>>>(v, total);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Source at top-left. Layout after source+A(vertex 1): d0, K1, u2...
    // Only 8 entries can be non-zero, write them from host for simplicity.
    std::uint16_t init[12]{};
    for (unsigned d0 = 0; d0 < 2; ++d0) {
        const unsigned r = d0 == 0 ? 1u : 0u;
        const std::uint32_t sw = d0 == 0 ? 1u : ZETA_INV;
        for (int k = 0; k < 4; ++k) {
            const auto a = o0mid::A_F[k * 9 + r];
            if (!a) continue;
            const auto w = static_cast<std::uint16_t>((std::uint64_t{sw} * a) % MOD);
            init[d0 + 3 * k] = static_cast<std::uint16_t>((init[d0 + 3 * k] + w) % MOD);
        }
    }
    CUDA_CHECK(cudaMemcpy(v, init, sizeof(init), cudaMemcpyHostToDevice));

    auto apply12 = [&](std::uint64_t stride, int kind, bool bottom = false) {
        const std::uint64_t groups = total / 12;
        if (kind == 0 && bottom) apply12_sparse<false, true><<<blocks(groups), threads>>>(v, groups, stride);
        else if (kind == 0) apply12_sparse<false, false><<<blocks(groups), threads>>>(v, groups, stride);
        else apply12_sparse<true, false><<<blocks(groups), threads>>>(v, groups, stride);
        CUDA_CHECK(cudaGetLastError());
    };
    auto apply4 = [&](std::uint64_t stride, int kind) {
        const std::uint64_t groups = total / 4;
        if (kind == 0) apply4_sparse<false><<<blocks(groups), threads>>>(v, groups, stride);
        else apply4_sparse<true><<<blocks(groups), threads>>>(v, groups, stride);
        CUDA_CHECK(cudaGetLastError());
    };

    for (unsigned p = 1; p + 1 < n; ++p) apply12(pow3(p), 0);

    bool forward = true;
    for (unsigned y = 1; y + 1 < n; ++y) {
        if (forward) {
            apply4(pow3(n - 1), 0);
            forward = false;
            for (int p = int(n) - 1; p > 0; --p) apply12(pow3(p - 1), 1);
        } else {
            apply4(1, 1);
            forward = true;
            for (unsigned p = 0; p + 1 < n; ++p) apply12(pow3(p), 0);
        }
    }
    if (forward) throw std::runtime_error("unexpected orientation before bottom");
    apply4(1, 1);
    for (unsigned p = 0; p + 2 < n; ++p) apply12(pow3(p), 0, true);

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    const auto wall1 = std::chrono::steady_clock::now();

    // Gather only target-support coefficients: lower n-2 qutrits are zero.
    std::uint32_t ans = 0;
    for (int k = 0; k < 4; ++k) {
        for (unsigned u = 0; u < 3; ++u) {
            std::uint32_t tw = 0;
            for (unsigned r = 0; r < 3; ++r) {
                std::uint32_t ew = 0;
                if (r == 1 && u == 0) ew = 1;
                else if (r == 0 && u == 1) ew = ZETA;
                if (!ew) continue;
                const auto b = o0mid::B_F[(3 * r) * 4 + k];
                tw = (tw + std::uint64_t{b} * ew) % MOD;
            }
            if (!tw) continue;
            const std::uint64_t idx = (std::uint64_t(k) + 4 * u) * stride_last;
            std::uint16_t val = 0;
            CUDA_CHECK(cudaMemcpy(&val, v + idx, sizeof(val), cudaMemcpyDeviceToHost));
            ans = (ans + std::uint64_t{val} * tw) % MOD;
        }
    }

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
    CUDA_CHECK(cudaFree(v));
    return {ans, total, ms, std::chrono::duration<double, std::milli>(wall1 - wall0).count()};
}

} // namespace

int main(int argc, char** argv) {
    CUDA_CHECK(cudaSetDevice(0));
    upload_tables();
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cerr << "gpu=" << prop.name << " mod=" << MOD << "\n";
    unsigned first = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 3;
    unsigned last = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : first;
    unsigned reps = argc > 3 ? std::strtoul(argv[3], nullptr, 10) : 1;
    for (unsigned n = first; n <= last; n += 2) {
        Result best{}; best.kernel_ms = 1e30f;
        double sum = 0;
        for (unsigned rep = 0; rep < reps; ++rep) {
            auto r = solve(n); sum += r.kernel_ms; if (r.kernel_ms < best.kernel_ms) best = r;
        }
        std::cout << "n=" << n << " residue=" << best.residue << " midpoint_states=" << best.states
                  << " workspace_mib=" << std::fixed << std::setprecision(1)
                  << (best.states * sizeof(std::uint16_t) / 1048576.0)
                  << " best_ms=" << std::setprecision(3) << best.kernel_ms
                  << " avg_ms=" << (sum / reps) << "\n";
    }
}
