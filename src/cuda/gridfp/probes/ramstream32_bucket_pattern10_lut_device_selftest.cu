#include "../ramstream32_bucket_orbit_closure_pattern10_depth8_lut.cuh"

#include <cstdint>
#include <iostream>
#include <vector>

__global__ void p10lut_low_test_kernel(int p, uint32_t begin, uint32_t count, uint32_t* out) {
    for (uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < count; id += blockDim.x * gridDim.x) {
        uint16_t lm = 0, rm = 0;
        bkcp10_decode_lut(uint16_t(id), LOW_LUT_K + 1, p, lm, rm);
        out[begin + id] = uint32_t(lm) | (uint32_t(rm) << 16);
    }
}

__global__ void p10lut_high_test_kernel(int p, uint32_t begin, uint32_t count, uint32_t* out) {
    for (uint32_t id = blockIdx.x * blockDim.x + threadIdx.x; id < count; id += blockDim.x * gridDim.x) {
        uint16_t lm = 0, rm = 0;
        bkcp10_decode_lut(uint16_t(id), HIGH_LUT_K + 1, p, lm, rm);
        out[begin + id] = uint32_t(lm) | (uint32_t(rm) << 16);
    }
}

int main() {
    int visible = 0;
    cudaError_t ce = cudaGetDeviceCount(&visible);
    if (ce != cudaSuccess || visible < 1) {
        std::cout << "bucket-pattern10-lut-device-selftest SKIP no CUDA device\n";
        return 0;
    }
    ck(cudaSetDevice(0), "pattern10 LUT selftest set device");
    BucketPattern10DecodeLutHost h = build_bucket_pattern10_decode_lut();
    BucketPattern10DecodeLutDeviceTables dt;
    dt.install(h);

    uint32_t *dl = nullptr, *dh = nullptr;
    ck(cudaMalloc(&dl, h.low.packed.size() * sizeof(uint32_t)), "pattern10 LUT test low out");
    ck(cudaMalloc(&dh, h.high.packed.size() * sizeof(uint32_t)), "pattern10 LUT test high out");
    for (int p = 1; p <= LOW_LUT_K; ++p) {
        uint32_t begin = h.low.off[size_t(p)], count = h.low.off[size_t(p + 1)] - begin;
        p10lut_low_test_kernel<<<32, 128>>>(p, begin, count, dl);
        ck(cudaGetLastError(), "pattern10 LUT low kernel");
    }
    for (int p = 1; p <= HIGH_LUT_K; ++p) {
        uint32_t begin = h.high.off[size_t(p)], count = h.high.off[size_t(p + 1)] - begin;
        p10lut_high_test_kernel<<<32, 128>>>(p, begin, count, dh);
        ck(cudaGetLastError(), "pattern10 LUT high kernel");
    }
    ck(cudaDeviceSynchronize(), "pattern10 LUT selftest sync");

    std::vector<uint32_t> gl(h.low.packed.size()), gh(h.high.packed.size());
    ck(cudaMemcpy(gl.data(), dl, gl.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost), "pattern10 LUT low D2H");
    ck(cudaMemcpy(gh.data(), dh, gh.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost), "pattern10 LUT high D2H");
    if (gl != h.low.packed || gh != h.high.packed) {
        std::cerr << "pattern10 LUT device decode mismatch\n";
        return 2;
    }
    std::cout << "bucket-pattern10-lut-device-selftest OK low_entries=" << gl.size()
              << " high_entries=" << gh.size()
              << " payload_bytes=" << (gl.size() + gh.size()) * sizeof(uint32_t)
              << '\n';
    cudaFree(dl); cudaFree(dh); dt.release();
    return 0;
}
