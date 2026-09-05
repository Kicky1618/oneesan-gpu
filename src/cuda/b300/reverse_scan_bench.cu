// Isolate predecessor enumeration; this does not measure full solver throughput.
#include <cuda_runtime.h>
#include "../../common/gridfp_reverse.hpp"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
using oneesan::gridfp::MateID;
static void check(cudaError_t e) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s\n", cudaGetErrorString(e)); std::exit(1); }
}
template<bool Sparse>
__global__ void enumerate(const MateID* input, uint64_t* output, size_t n, int p) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += size_t(gridDim.x) * blockDim.x) {
        uint64_t digest = 0;
        oneesan::gridfp::reverse_block_predecessors<Sparse>(input[i], 28, p,
            [&](MateID m){ digest = digest * 0x9e3779b97f4a7c15ULL + m + 1; });
        output[i] = digest;
    }
}
int main() {
    constexpr size_t n = 1u << 20;
    uint64_t ways[28][30]{};
    ways[0][0] = 1;
    for (int w = 1; w <= 27; ++w)
        for (int h = 0; h <= 27; ++h)
            ways[w][h] = ways[w-1][h] + (h ? ways[w-1][h-1] : 0) + ways[w-1][h+1];
    std::mt19937_64 rng(20260905);
    std::vector<MateID> input(n);
    for (auto& b : input) {
        uint64_t rank = rng() % ways[27][1]; int h = 1;
        for (int pos = 26; pos >= 0; --pos) {
            if (rank < ways[pos][h]) continue;
            rank -= ways[pos][h];
            if (h && rank < ways[pos][h-1]) {
                b |= MateID(1) << (2*pos); --h;
            } else {
                if (h) rank -= ways[pos][h-1];
                b |= MateID(2) << (2*pos); ++h;
            }
        }
    }
    MateID* src; uint64_t *dst[2];
    check(cudaMalloc(&src, n * sizeof(MateID)));
    for (auto& ptr : dst) check(cudaMalloc(&ptr, n * sizeof(uint64_t)));
    check(cudaMemcpy(src, input.data(), n * sizeof(MateID), cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
    for (int p : {2, 14, 27}) {
        std::vector<float> time[2];
        for (int repeat = 0; repeat < 11; ++repeat) {
            for (int order = 0; order < 2; ++order) {
                int mode = order ^ (repeat & 1);
                check(cudaEventRecord(start));
                for (int j = 0; j < 10; ++j) {
                    if (mode) enumerate<true><<<4096,256>>>(src,dst[mode],n,p);
                    else enumerate<false><<<4096,256>>>(src,dst[mode],n,p);
                }
                check(cudaGetLastError()); check(cudaEventRecord(stop));
                check(cudaEventSynchronize(stop));
                float ms; check(cudaEventElapsedTime(&ms,start,stop));
                if (repeat) time[mode].push_back(ms / 10);
            }
        }
        std::vector<uint64_t> a(n), b(n);
        check(cudaMemcpy(a.data(),dst[0],n*sizeof(uint64_t),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(b.data(),dst[1],n*sizeof(uint64_t),cudaMemcpyDeviceToHost));
        if (a != b) { std::fprintf(stderr,"scan mismatch p=%d\n",p); return 2; }
        for (auto& t : time) std::sort(t.begin(),t.end());
        float scalar=(time[0][4]+time[0][5])/2, sparse=(time[1][4]+time[1][5])/2;
        std::printf("p=%d targets=%zu scalar_ms=%.6f sparse_ms=%.6f speedup=%.4f PASS\n",
                    p,n,scalar,sparse,scalar/sparse);
    }
    check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    check(cudaFree(src)); for (auto ptr : dst) check(cudaFree(ptr));
}
