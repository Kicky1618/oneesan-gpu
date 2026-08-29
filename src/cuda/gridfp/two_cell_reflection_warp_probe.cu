#pragma push_macro("main")
#undef main
#define main two_cell_packed_component_probe_main_unused
#include "../../cpp/probes/two_cell_packed_component_probe.cpp"
#pragma pop_macro("main")

#include "two_cell_reflection_warp.cuh"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

using DeviceWord = oneesan::twocell::PackedWord;

void ck_reflect(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at " << what << ": "
                  << cudaGetErrorString(e) << '\n';
        std::exit(560);
    }
}

DeviceWord to_device_word(const Word& w) {
    DeviceWord z{};
    z.len = static_cast<std::uint8_t>(w.size());
    for (int p = 0; p < static_cast<int>(w.size()); ++p) {
        const std::uint32_t bit = std::uint32_t(1) << p;
        if (w[p] != N) z.support |= bit;
        if (w[p] == L) z.left |= bit;
    }
    return z;
}

bool same_word(DeviceWord a, DeviceWord b) {
    return a.support == b.support && a.left == b.left && a.len == b.len;
}

__global__ void reflection_warp_kernel(
    const DeviceWord* __restrict__ input,
    DeviceWord* __restrict__ output,
    int count,
    int* error
) {
    const int lane = threadIdx.x & 31;
    const int warp_global = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (warp_global >= count) return;
    const DeviceWord w = input[warp_global];
    const DeviceWord z = oneesan::twocell::cuda_reflect::reflect_word_warp(
        w, error);
    if (lane == 0) output[warp_global] = z;
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 14;
    if (maxW < 1 || maxW > 15) return 2;

    int visible = 0;
    ck_reflect(cudaGetDeviceCount(&visible), "device count");
    if (visible < 1) return 3;
    ck_reflect(cudaSetDevice(0), "set device");

    for (int W = 1; W <= maxW; ++W) {
        const auto words = gen_words(W);
        std::vector<DeviceWord> input;
        input.reserve(words.size());
        for (const Word& w : words) input.push_back(to_device_word(w));
        std::vector<DeviceWord> output(input.size());

        DeviceWord *d_input = nullptr, *d_output = nullptr;
        int* d_error = nullptr;
        ck_reflect(cudaMalloc(&d_input, input.size() * sizeof(DeviceWord)),
                   "alloc input");
        ck_reflect(cudaMalloc(&d_output, output.size() * sizeof(DeviceWord)),
                   "alloc output");
        ck_reflect(cudaMalloc(&d_error, sizeof(int)), "alloc error");
        ck_reflect(cudaMemcpy(d_input, input.data(), input.size() * sizeof(DeviceWord),
                              cudaMemcpyHostToDevice), "copy input");
        ck_reflect(cudaMemset(d_error, 0, sizeof(int)), "zero error");

        constexpr int threads = 128;
        const int warps = static_cast<int>(input.size());
        const int blocks = std::max(1, (warps * 32 + threads - 1) / threads);
        reflection_warp_kernel<<<blocks, threads>>>(
            d_input, d_output, warps, d_error);
        ck_reflect(cudaGetLastError(), "launch");
        ck_reflect(cudaDeviceSynchronize(), "sync");
        int error = 0;
        ck_reflect(cudaMemcpy(output.data(), d_output,
                              output.size() * sizeof(DeviceWord),
                              cudaMemcpyDeviceToHost), "copy output");
        ck_reflect(cudaMemcpy(&error, d_error, sizeof(error),
                              cudaMemcpyDeviceToHost), "copy error");
        if (error) {
            std::cerr << "FAIL reflection warp error W=" << W
                      << " error=" << error << '\n';
            return 4;
        }

        for (std::size_t q = 0; q < input.size(); ++q) {
            const DeviceWord expected = oneesan::twocell::reflect_packed_word(input[q]);
            if (!same_word(output[q], expected)) {
                std::cerr << "FAIL reflection mismatch W=" << W
                          << " index=" << q << '\n';
                return 5;
            }
        }

        std::cout << "W=" << W
                  << " words=" << input.size()
                  << " warp_reflection=OK"
                  << " bit_reverse_loop=0 root_position_loop=0\n";
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_error);
    }
    std::cout << "ALL_OK warp_packed_reflection=1\n";
    return 0;
}
