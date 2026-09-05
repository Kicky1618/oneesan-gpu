#include "two_cell_parallel_face_device.cuh"
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
constexpr int MAX_COMPONENT = oneesan::twocell::cuda_face::kMaxComponent;
constexpr int MAX_CANDIDATES = oneesan::twocell::cuda_face::kMaxCandidates;

__constant__ RankTables TC_FACE_ORDER_TABLES;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at " << what << ": " << cudaGetErrorString(e) << '\n';
        std::exit(340);
    }
}

__global__ void parallel_face_order_kernel(
    int W,
    Rank components,
    unsigned long long* deep_checked,
    unsigned long long* ordered_equal,
    int* worst_partner_rounds,
    int* error
) {
    __shared__ PackedKey sh_out[WARPS_PER_BLOCK][MAX_COMPONENT];
    __shared__ PackedKey sh_candidate[WARPS_PER_BLOCK][MAX_CANDIDATES];
    __shared__ std::uint8_t sh_valid[WARPS_PER_BLOCK][MAX_CANDIDATES];
    __shared__ int sh_size[WARPS_PER_BLOCK];
    __shared__ int sh_rounds[WARPS_PER_BLOCK];
    __shared__ PackedWord sh_label[WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank first = Rank(blockIdx.x) * WARPS_PER_BLOCK + Rank(warp);
    const Rank stride = Rank(gridDim.x) * WARPS_PER_BLOCK;

    for (Rank component = first; component < components; component += stride) {
        if (lane == 0) {
            const PackedKey label_key = component_label_unrank(
                W, component, TC_FACE_ORDER_TABLES);
            sh_label[warp] = PackedWord{
                label_key.support,
                label_key.left,
                static_cast<std::uint8_t>(W - 2)
            };
        }
        __syncwarp();

        for (int i = 0; i <= W - 4; ++i) {
            PackedWord collapsed{};
            const bool deep = deep_collapse(sh_label[warp], i, collapsed);
            if (!__any_sync(0xffffffffu, deep)) continue;

            oneesan::twocell::cuda_face::deep_component_sources(
                sh_label[warp], W, i,
                sh_out[warp], &sh_size[warp],
                sh_candidate[warp], sh_valid[warp],
                &sh_rounds[warp], error);
            __syncwarp();

            if (lane == 0) {
                const auto serial = direct_component_sources(sh_label[warp], W, i);
                atomicAdd(deep_checked, 1ULL);
                if (serial.overflow || serial.size != sh_size[warp]) {
                    atomicCAS(error, 0, 341);
                } else {
                    bool same = true;
                    for (int q = 0; q < serial.size; ++q) {
                        if (!equal(serial.value[q], sh_out[warp][q])) {
                            same = false;
                            atomicCAS(error, 0, 342);
                            break;
                        }
                    }
                    if (same) atomicAdd(ordered_equal, 1ULL);
                }
                atomicMax(worst_partner_rounds, sh_rounds[warp]);
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
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const unsigned blocks = argc > 2
        ? static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10)) : 4096u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 5 || W > kMaxWidth || blocks == 0) return 2;

    const RankTables tables = make_rank_tables();
    const Rank components = component_label_count(W, tables);
    if (plan_only) {
        std::cout << "two-cell-parallel-face-order-plan"
                  << " W=" << W
                  << " components=" << components
                  << " compare=index-by-index"
                  << " serial=direct_component_sources"
                  << " parallel=cuda_face::deep_component_sources"
                  << " max_sources=17 candidate_slots=34\n";
        return 0;
    }
    if (W > 11) {
        std::cerr << "execution mode intentionally limited to W<=11; use --plan-only above that\n";
        return 3;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "face order device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "face order set device");
    ck(cudaMemcpyToSymbol(TC_FACE_ORDER_TABLES, &tables, sizeof(tables)),
       "face order copy tables");

    unsigned long long *d_deep = nullptr, *d_equal = nullptr;
    int *d_rounds = nullptr, *d_error = nullptr;
    ck(cudaMalloc(&d_deep, sizeof(unsigned long long)), "face order alloc deep");
    ck(cudaMalloc(&d_equal, sizeof(unsigned long long)), "face order alloc equal");
    ck(cudaMalloc(&d_rounds, sizeof(int)), "face order alloc rounds");
    ck(cudaMalloc(&d_error, sizeof(int)), "face order alloc error");
    ck(cudaMemset(d_deep, 0, sizeof(unsigned long long)), "face order zero deep");
    ck(cudaMemset(d_equal, 0, sizeof(unsigned long long)), "face order zero equal");
    ck(cudaMemset(d_rounds, 0, sizeof(int)), "face order zero rounds");
    ck(cudaMemset(d_error, 0, sizeof(int)), "face order zero error");

    const Rank one_pass_blocks = (components + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned grid = static_cast<unsigned>(std::max<Rank>(
        1, std::min<Rank>(blocks, one_pass_blocks)));
    parallel_face_order_kernel<<<grid, THREADS>>>(
        W, components, d_deep, d_equal, d_rounds, d_error);
    ck(cudaGetLastError(), "face order launch");
    ck(cudaDeviceSynchronize(), "face order sync");

    unsigned long long deep = 0, equal = 0;
    int rounds = 0, error = 0;
    ck(cudaMemcpy(&deep, d_deep, sizeof(deep), cudaMemcpyDeviceToHost),
       "face order copy deep");
    ck(cudaMemcpy(&equal, d_equal, sizeof(equal), cudaMemcpyDeviceToHost),
       "face order copy equal");
    ck(cudaMemcpy(&rounds, d_rounds, sizeof(rounds), cudaMemcpyDeviceToHost),
       "face order copy rounds");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "face order copy error");

    if (error || deep != equal) {
        std::cerr << "FAIL face order W=" << W
                  << " error=" << error
                  << " deep=" << deep
                  << " ordered_equal=" << equal << '\n';
        return 5;
    }

    std::cout << "two-cell-parallel-face-order"
              << " W=" << W
              << " deep_components=" << deep
              << " ordered_equal=" << equal
              << " worst_partner_rounds=" << rounds
              << " canonical_source_order=OK\n";
    std::cout << "ALL_OK parallel_face_canonical_order=1 W=" << W << '\n';

    cudaFree(d_deep);
    cudaFree(d_equal);
    cudaFree(d_rounds);
    cudaFree(d_error);
    return 0;
}
