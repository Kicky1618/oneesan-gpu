#include "two_cell_component_warp_arithmetic.cuh"
#include "../../common/two_cell_recoupling_rank.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

using namespace oneesan::twocell;

constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS = 32 * WARPS_PER_BLOCK;
constexpr int MAX_STATES = kMaxComponentMatching;
constexpr std::uint32_t MOD = 1000000007u;

__constant__ RankTables TC_WARP_ARITH_TABLES;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at " << what << ": " << cudaGetErrorString(e) << '\n';
        std::exit(420);
    }
}

__device__ __forceinline__ std::uint32_t ref_add(
    std::uint32_t a, std::uint32_t b
) {
    const unsigned long long z = static_cast<unsigned long long>(a) + b;
    return static_cast<std::uint32_t>(z >= MOD ? z - MOD : z);
}

__global__ void component_warp_arithmetic_kernel(
    int W,
    Rank components,
    unsigned long long* checked,
    unsigned long long* singleton,
    unsigned long long* triple,
    unsigned long long* deep_rn,
    unsigned long long* deep_lr,
    unsigned long long* deep_ln,
    int* error
) {
    __shared__ PackedKey sh_src[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_x[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_y[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_ref[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ PackedWord sh_label[WARPS_PER_BLOCK];
    __shared__ int sh_ns[WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank first = Rank(blockIdx.x) * WARPS_PER_BLOCK + Rank(warp);
    const Rank stride = Rank(gridDim.x) * WARPS_PER_BLOCK;

    for (Rank cr = first; cr < components; cr += stride) {
        if (lane == 0) {
            const PackedKey lk = component_label_unrank(W, cr, TC_WARP_ARITH_TABLES);
            sh_label[warp] = PackedWord{
                lk.support, lk.left, static_cast<std::uint8_t>(W - 2)};
        }
        __syncwarp();

        for (int i = 0; i <= W - 4; ++i) {
            if (lane == 0) {
                const auto src = direct_component_sources(sh_label[warp], W, i);
                sh_ns[warp] = 0;
                if (src.overflow || src.size <= 0 || src.size > MAX_STATES) {
                    atomicCAS(error, 0, 421);
                } else {
                    sh_ns[warp] = src.size;
                    for (int s = 0; s < src.size; ++s) {
                        sh_src[warp][s] = src.value[s];
                        sh_x[warp][s] = static_cast<std::uint32_t>(
                            1 + ((cr * 1315423911ULL + Rank(i) * 97ULL +
                                  Rank(s) * 65537ULL) % (MOD - 1ULL)));
                        sh_ref[warp][s] = 0;
                    }
                }
            }
            __syncwarp();

            const int ns = sh_ns[warp];
            const PackedKey source = lane < ns ? sh_src[warp][lane] : PackedKey{};
            const std::uint32_t x = lane < ns ? sh_x[warp][lane] : 0u;
            const std::uint32_t y =
                oneesan::twocell::cuda_component::apply_closed_component_warp(
                    sh_label[warp], source, ns, W, i, x, MOD, error);
            if (lane < ns) sh_y[warp][lane] = y;
            __syncwarp();

            if (lane == 0 && ns > 0) {
                for (int s = 0; s < ns; ++s) {
                    const auto edges = K_step(sh_src[warp][s], W, i);
                    if (edges.overflow) {
                        atomicCAS(error, 0, 422);
                        continue;
                    }
                    for (int e = 0; e < edges.size; ++e) {
                        const int t = coordinate_index_for_destination(
                            sh_src[warp], ns, edges.value[e], i);
                        if (t < 0) {
                            atomicCAS(error, 0, 423);
                            continue;
                        }
                        sh_ref[warp][t] = ref_add(
                            sh_ref[warp][t], sh_x[warp][s]);
                    }
                }
                for (int t = 0; t < ns; ++t)
                    if (sh_y[warp][t] != sh_ref[warp][t])
                        atomicCAS(error, 0, 424);

                if (ns == 1) {
                    atomicAdd(singleton, 1ULL);
                } else if (ns == 3) {
                    atomicAdd(triple, 1ULL);
                } else {
                    const Symbol a = symbol(sh_label[warp], i);
                    const Symbol b = symbol(sh_label[warp], i + 1);
                    if (a == TC_R && b == TC_N) atomicAdd(deep_rn, 1ULL);
                    else if (a == TC_L && b == TC_R) atomicAdd(deep_lr, 1ULL);
                    else if (a == TC_L && b == TC_N) atomicAdd(deep_ln, 1ULL);
                    else atomicCAS(error, 0, 425);
                }
                atomicAdd(checked, 1ULL);
            }
            __syncwarp();
        }
    }
}

bool has_arg(int argc, char** argv, const char* needle) {
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == needle) return true;
    return false;
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 9;
    const unsigned blocks = argc > 2
        ? static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10)) : 4096u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 5 || W > kMaxWidth || blocks == 0) return 2;

    const RankTables rt = make_rank_tables();
    const Rank components = component_label_count(W, rt);
    if (plan_only) {
        std::cout << "two-cell-component-warp-arithmetic-plan"
                  << " W=" << W
                  << " components=" << components
                  << " component_positions=" << (W - 3)
                  << " component_value_shared_bytes=0"
                  << " component_rank_shared_bytes=0"
                  << " arithmetic=register+shuffle"
                  << " oracle=K_step_edges\n";
        return 0;
    }
    if (W > 11) {
        std::cerr << "execution mode intentionally limited to W<=11; use --plan-only above that\n";
        return 3;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "warp arithmetic device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "warp arithmetic set device");
    ck(cudaMemcpyToSymbol(TC_WARP_ARITH_TABLES, &rt, sizeof(rt)),
       "warp arithmetic copy tables");

    unsigned long long *d_checked = nullptr, *d_singleton = nullptr, *d_triple = nullptr;
    unsigned long long *d_rn = nullptr, *d_lr = nullptr, *d_ln = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_checked, sizeof(unsigned long long)), "warp arith alloc checked");
    ck(cudaMalloc(&d_singleton, sizeof(unsigned long long)), "warp arith alloc singleton");
    ck(cudaMalloc(&d_triple, sizeof(unsigned long long)), "warp arith alloc triple");
    ck(cudaMalloc(&d_rn, sizeof(unsigned long long)), "warp arith alloc rn");
    ck(cudaMalloc(&d_lr, sizeof(unsigned long long)), "warp arith alloc lr");
    ck(cudaMalloc(&d_ln, sizeof(unsigned long long)), "warp arith alloc ln");
    ck(cudaMalloc(&d_error, sizeof(int)), "warp arith alloc error");
    ck(cudaMemset(d_checked, 0, sizeof(unsigned long long)), "warp arith zero checked");
    ck(cudaMemset(d_singleton, 0, sizeof(unsigned long long)), "warp arith zero singleton");
    ck(cudaMemset(d_triple, 0, sizeof(unsigned long long)), "warp arith zero triple");
    ck(cudaMemset(d_rn, 0, sizeof(unsigned long long)), "warp arith zero rn");
    ck(cudaMemset(d_lr, 0, sizeof(unsigned long long)), "warp arith zero lr");
    ck(cudaMemset(d_ln, 0, sizeof(unsigned long long)), "warp arith zero ln");
    ck(cudaMemset(d_error, 0, sizeof(int)), "warp arith zero error");

    const Rank one_pass_blocks = (components + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned grid = static_cast<unsigned>(std::max<Rank>(
        1, std::min<Rank>(blocks, one_pass_blocks)));
    component_warp_arithmetic_kernel<<<grid, THREADS>>>(
        W, components, d_checked, d_singleton, d_triple,
        d_rn, d_lr, d_ln, d_error);
    ck(cudaGetLastError(), "warp arithmetic launch");
    ck(cudaDeviceSynchronize(), "warp arithmetic sync");

    unsigned long long checked = 0, singleton = 0, triple = 0, rn = 0, lr = 0, ln = 0;
    int error = 0;
    ck(cudaMemcpy(&checked, d_checked, sizeof(checked), cudaMemcpyDeviceToHost),
       "warp arith copy checked");
    ck(cudaMemcpy(&singleton, d_singleton, sizeof(singleton), cudaMemcpyDeviceToHost),
       "warp arith copy singleton");
    ck(cudaMemcpy(&triple, d_triple, sizeof(triple), cudaMemcpyDeviceToHost),
       "warp arith copy triple");
    ck(cudaMemcpy(&rn, d_rn, sizeof(rn), cudaMemcpyDeviceToHost), "warp arith copy rn");
    ck(cudaMemcpy(&lr, d_lr, sizeof(lr), cudaMemcpyDeviceToHost), "warp arith copy lr");
    ck(cudaMemcpy(&ln, d_ln, sizeof(ln), cudaMemcpyDeviceToHost), "warp arith copy ln");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "warp arith copy error");

    const unsigned long long expected =
        static_cast<unsigned long long>(components) * static_cast<unsigned long long>(W - 3);
    if (error || checked != expected || singleton + triple + rn + lr + ln != checked) {
        std::cerr << "FAIL warp arithmetic W=" << W
                  << " error=" << error
                  << " checked=" << checked << "/" << expected << '\n';
        return 5;
    }

    std::cout << "two-cell-component-warp-arithmetic"
              << " W=" << W
              << " checked=" << checked
              << " singleton=" << singleton
              << " triple=" << triple
              << " RN=" << rn
              << " LR=" << lr
              << " LN=" << ln
              << " component_value_shared_bytes=0"
              << " component_rank_shared_bytes=0"
              << " K_edge_oracle=OK\n";
    std::cout << "ALL_OK register_only_component_warp_arithmetic=1 W=" << W << '\n';

    cudaFree(d_checked);
    cudaFree(d_singleton);
    cudaFree(d_triple);
    cudaFree(d_rn);
    cudaFree(d_lr);
    cudaFree(d_ln);
    cudaFree(d_error);
    return 0;
}
