#include "../ramstream32_bucket_low_rankchunk32_basepair64.cuh"

static void bp64_ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::cerr << what << ": " << cudaGetErrorString(e) << '\n';
        std::exit(2);
    }
}

__global__ void p10dc_rankchunk32_basepair64_warp_probe(
    uint32_t* bad, const uint16_t* stream_base
) {
    const uint32_t start = uint32_t(blockIdx.x);       // 0..63
    const uint32_t width = uint32_t(blockIdx.y) + 1u; // 1..32
    const uint32_t lane = uint32_t(threadIdx.x) & 31u;
    if (lane >= width) return;

    const uint32_t rank = start + lane;
    uint32_t chunks = 0;
    const uint16_t* row = nullptr;
    p10dc_low_rankchunk32_basepair64_row_warpstripe(0u, rank, chunks, row);

    const uint32_t compact = rank;
    const uint32_t prefix = (compact * 13u) % 251u;
    const uint32_t expected_chunks =
        (compact & 0xffu) | (((compact * 3u) & 0xffu) << 8) |
        (((compact * 5u) & 0xffu) << 16);
    const uint32_t block32 = compact >> 5;
    uint32_t expected_base = 0;
    if (block32 == 0u) expected_base = 100u;
    else if (block32 == 1u) expected_base = 107u;
    else if (block32 == 2u) expected_base = 200u;
    else expected_base = 211u;

    if (chunks != expected_chunks ||
        row != stream_base + expected_base + prefix)
        atomicAdd(bad, 1u);
}

int main() {
    static_assert(P10DC_RANKCHUNK32_BYTEPACK == 1,
                  "basepair64 warp probe requires bytepack");
    static_assert(P10DC_RANKCHUNK32_BLOCK64 == 0,
                  "basepair64 warp probe keeps block32 prefix metadata");
    static_assert(P10DC_RANKCHUNK32_BLOCK == 32u);
    static_assert(P10DC_RANKCHUNK32_PREFIX_BITS == 8u);

    int visible = 0;
    cudaError_t ce = cudaGetDeviceCount(&visible);
    if (ce != cudaSuccess || visible < 1) {
        std::cout << "rankchunk32-basepair64-warp-cuda-selftest SKIP no CUDA device\n";
        return 0;
    }
    bp64_ck(cudaSetDevice(0), "basepair64 warp set device");

    constexpr size_t META_N = 96;
    constexpr size_t STREAM_N = 4096;
    std::array<uint32_t, META_N> meta{};
    for (uint32_t i = 0; i < uint32_t(META_N); ++i) {
        const uint32_t prefix = (i * 13u) % 251u;
        const uint32_t chunks =
            (i & 0xffu) | (((i * 3u) & 0xffu) << 8) |
            (((i * 5u) & 0xffu) << 16);
        meta[i] = chunks | (prefix << 24);
    }
    const std::array<uint32_t, 2> pairs = {
        100u | (7u << P10DC_RANKCHUNK32_BASEPAIR_BASE_BITS),
        200u | (11u << P10DC_RANKCHUNK32_BASEPAIR_BASE_BITS),
    };
    std::array<uint32_t, MAXW + 2> hoff{};

    uint32_t* d_meta = nullptr;
    uint32_t* d_pairs = nullptr;
    uint16_t* d_stream = nullptr;
    uint32_t* d_bad = nullptr;
    bp64_ck(cudaMalloc(&d_meta, meta.size() * sizeof(uint32_t)), "basepair64 warp meta alloc");
    bp64_ck(cudaMalloc(&d_pairs, pairs.size() * sizeof(uint32_t)), "basepair64 warp pair alloc");
    bp64_ck(cudaMalloc(&d_stream, STREAM_N * sizeof(uint16_t)), "basepair64 warp stream alloc");
    bp64_ck(cudaMalloc(&d_bad, sizeof(uint32_t)), "basepair64 warp bad alloc");
    bp64_ck(cudaMemcpy(d_meta, meta.data(), meta.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "basepair64 warp meta H2D");
    bp64_ck(cudaMemcpy(d_pairs, pairs.data(), pairs.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
            "basepair64 warp pairs H2D");
    bp64_ck(cudaMemset(d_stream, 0, STREAM_N * sizeof(uint16_t)), "basepair64 warp stream clear");
    bp64_ck(cudaMemset(d_bad, 0, sizeof(uint32_t)), "basepair64 warp bad clear");
    bp64_ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKCHUNKMETA32, &d_meta, sizeof(d_meta)),
            "basepair64 warp meta ptr");
    bp64_ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKCHUNKBLOCK16, &d_pairs, sizeof(d_pairs)),
            "basepair64 warp pair ptr");
    bp64_ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKCHUNK_HOFF, hoff.data(), hoff.size() * sizeof(uint32_t)),
            "basepair64 warp hoff");
    bp64_ck(cudaMemcpyToSymbol(D_P10DC_LOW_RANKSTREAM, &d_stream, sizeof(d_stream)),
            "basepair64 warp stream ptr");

    p10dc_rankchunk32_basepair64_warp_probe<<<dim3(64, 32), 32>>>(d_bad, d_stream);
    bp64_ck(cudaGetLastError(), "basepair64 warp kernel launch");
    bp64_ck(cudaDeviceSynchronize(), "basepair64 warp kernel sync");
    uint32_t bad = 0;
    bp64_ck(cudaMemcpy(&bad, d_bad, sizeof(uint32_t), cudaMemcpyDeviceToHost),
            "basepair64 warp bad D2H");

    cudaFree(d_bad);
    cudaFree(d_stream);
    cudaFree(d_pairs);
    cudaFree(d_meta);
    if (bad) {
        std::cerr << "rankchunk32-basepair64-warp-cuda-selftest FAIL bad=" << bad << '\n';
        return 3;
    }
    std::cout << "rankchunk32-basepair64-warp-cuda-selftest OK"
              << " cases=2048 starts=64 partial_widths=32"
              << " align=" << P10DC_RANKCHUNK32_ALIGN32
              << " bytepack=1 pair_decode_exact=1 row_pointer_exact=1"
              << " base_bits=22 delta_bits=8\n";
    return 0;
}
