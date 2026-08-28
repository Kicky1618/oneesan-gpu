#include "../ramstream32_bucket_closure_cross5_rankchunk32.cuh"

__global__ void p10dc_rankchunk32_codec_kernel(uint32_t* bad) {
    constexpr uint32_t KEYN = 4782969u; // 3^14
    constexpr uint32_t P9 = 19683u;     // 3^9
    constexpr uint32_t P4 = 81u;        // 3^4
    constexpr uint32_t PREFIX_N = 1u << P10DC_RANKCHUNK32_PREFIX_BITS;
    for (uint32_t key = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
         key < KEYN;
         key += uint32_t(gridDim.x) * blockDim.x) {
        const uint32_t c0 = (key / P9) % 243u;
        const uint32_t c1 = (key / P4) % 243u;
        const uint32_t c2 = key % P4;
        const uint32_t packed = c0 | (c1 << 8) | (c2 << 16);
        const uint32_t prefix = key & (PREFIX_N - 1u);
        const uint32_t meta = packed | (prefix << P10DC_RANKCHUNK32_CHUNK_BITS);
        const uint32_t chunks = meta & P10DC_RANKCHUNK32_CHUNK_MASK;
        const uint32_t recovered_prefix = meta >> P10DC_RANKCHUNK32_CHUNK_BITS;
        if (packed > P10DC_RANKCHUNK32_CHUNK_MASK ||
            chunks != packed || recovered_prefix != prefix ||
            p10dc_rankchunk32_chunk_device(chunks, 0u) != c0 ||
            p10dc_rankchunk32_chunk_device(chunks, 1u) != c1 ||
            p10dc_rankchunk32_chunk_device(chunks, 2u) != c2)
            atomicAdd(bad, 1u);
    }
}

int main() {
    static_assert(LOW_LUT_K == 14, "rankchunk32 codec selftest requires K=14");
    static_assert(P10DC_RANKCHUNK32_BLOCK == 32u);
    static_assert(P10DC_RANKCHUNK32_CHUNK_BITS == (P10DC_RANKCHUNK32_BYTEPACK ? 24u : 23u));
    static_assert(P10DC_RANKCHUNK32_PREFIX_BITS == (P10DC_RANKCHUNK32_BYTEPACK ? 8u : 9u));

    int visible = 0;
    cudaError_t ce = cudaGetDeviceCount(&visible);
    if (ce != cudaSuccess || visible < 1) {
        std::cout << "rankchunk32-codec-cuda-selftest SKIP no CUDA device\n";
        return 0;
    }
    ck(cudaSetDevice(0), "rankchunk32 codec set device");
    uint32_t* d_bad = nullptr;
    ck(cudaMalloc(&d_bad, sizeof(uint32_t)), "rankchunk32 codec bad alloc");
    ck(cudaMemset(d_bad, 0, sizeof(uint32_t)), "rankchunk32 codec bad clear");
    p10dc_rankchunk32_codec_kernel<<<128, 256>>>(d_bad);
    ck(cudaGetLastError(), "rankchunk32 codec kernel launch");
    ck(cudaDeviceSynchronize(), "rankchunk32 codec kernel sync");
    uint32_t bad = 0;
    ck(cudaMemcpy(&bad, d_bad, sizeof(uint32_t), cudaMemcpyDeviceToHost),
       "rankchunk32 codec bad D2H");
    cudaFree(d_bad);
    if (bad) {
        std::cerr << "rankchunk32-codec-cuda-selftest FAIL bad=" << bad << '\n';
        return 2;
    }
    std::cout << "rankchunk32-codec-cuda-selftest OK"
              << " keys=4782969 chunk_bits=" << P10DC_RANKCHUNK32_CHUNK_BITS
              << " prefix_bits=" << P10DC_RANKCHUNK32_PREFIX_BITS
              << " block=32 chunk0_bits=8 chunk1_bits=8 chunk2_value_bits=7"
              << " byte_aligned_chunks=" << P10DC_RANKCHUNK32_BYTEPACK
              << " prefix_isolation_exact=1 device_decode_exact=1\n";
    return 0;
}