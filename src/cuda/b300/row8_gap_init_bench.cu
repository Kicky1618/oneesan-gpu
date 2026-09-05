#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

using Count = std::uint32_t;
using Code = std::uint64_t;
static constexpr int TARGET_W = 19;
static Code H_DP[64][64]{};
static inline void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(where) + ": " + cudaGetErrorString(e));
}

#include "row8_tensor_init.cuh"
#include "row8_structural_tensor_init.cuh"
#include "row8_gap_tensor_init.cuh"

static void build_full_dp() {
    for (int h = 0; h < 64; ++h) H_DP[0][h] = (h == 0);
    for (int w = 1; w < 64; ++w) {
        for (int h = 0; h < 63; ++h) {
            Code x = H_DP[w - 1][h];
            if (h > 0) x += H_DP[w - 1][h - 1];
            x += H_DP[w - 1][h + 1];
            H_DP[w][h] = x;
        }
    }
}

struct Sum {
    double upload = 0, prefix = 0, suffix = 0, transpose = 0, join = 0, total = 0;
    int n = 0;
    template<class S> void add(const S& s) {
        upload += s.upload_s; prefix += s.prefix_s; suffix += s.suffix_s;
        transpose += s.transpose_s; join += s.join_s; total += s.gpu_init_s; ++n;
    }
    void print(const char* name) const {
        const double d = n ? double(n) : 1.0;
        std::cout << name
                  << " reps=" << n
                  << " upload_ms=" << 1000.0 * upload / d
                  << " prefix_ms=" << 1000.0 * prefix / d
                  << " suffix_ms=" << 1000.0 * suffix / d
                  << " transpose_ms=" << 1000.0 * transpose / d
                  << " join_ms=" << 1000.0 * join / d
                  << " total_ms=" << 1000.0 * total / d << '\n';
    }
};

int main(int argc, char** argv) {
    const int reps = argc > 1 ? std::max(1, std::atoi(argv[1])) : 10;
    const std::uint32_t mod = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 1000000007u;
    build_full_dp();
    const Code states = H_DP[TARGET_W][1];
    Count* out = nullptr;
    ck(cudaMalloc(&out, std::size_t(states) * sizeof(Count)), "output alloc");

    // Load caches, JIT kernels, and warm clocks before measurements.
    (void)oneesan::row8struct::init_single_gpu(TARGET_W, mod, out, 0);
    (void)oneesan::row8gap::init_single_gpu(TARGET_W, mod, out, 0);

    Sum ss, gs;
    for (int i = 0; i < reps; ++i) {
        if ((i & 1) == 0) {
            ss.add(oneesan::row8struct::init_single_gpu(TARGET_W, mod, out, 0));
            gs.add(oneesan::row8gap::init_single_gpu(TARGET_W, mod, out, 0));
        } else {
            gs.add(oneesan::row8gap::init_single_gpu(TARGET_W, mod, out, 0));
            ss.add(oneesan::row8struct::init_single_gpu(TARGET_W, mod, out, 0));
        }
    }
    ss.print("structural");
    gs.print("gap");
    std::cout << "speedup_total=" << ss.total / gs.total
              << " speedup_prefix=" << ss.prefix / gs.prefix
              << " speedup_suffix=" << ss.suffix / gs.suffix
              << " states=" << states << '\n';
    cudaFree(out);
    return 0;
}
