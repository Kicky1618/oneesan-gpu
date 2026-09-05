#include "../ramstream32_bucket_closure_cross5.cuh"

#include <cstdint>
#include <iostream>
#include <vector>

static uint32_t cross5_pack_ternary(uint32_t key) {
    uint32_t code = 0;
    for (int pos = 0; pos < LOW_LUT_K; ++pos) {
        uint32_t v = key % 3u;
        key /= 3u;
        code |= v << (2 * pos);
    }
    return code;
}

__global__ void cross5_compare_kernel(
    const uint32_t* codes, uint32_t ncode, const Count* source_row,
    uint32_t* mismatches, uint32_t* checked
) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = ncode * 15u;
    if (i >= total) return;
    uint32_t ci = i / 15u;
    uint32_t depth = 1u + i % 15u;
    uint32_t code = codes[ci];
    BkczCrossAccum a = p10dc_resolved_low_preimages(code, depth, source_row);
    BkczCrossAccum b = p10dc_resolved_low_preimages_cross5(code, depth, source_row);
    BkczCrossAccum c = p10dc_resolved_low_preimages_cross5_prekey(code, ci, depth, source_row);
    if (a != b || a != c) atomicAdd(mismatches, 1u);
    atomicAdd(checked, 1u);
}

int main() {
    static_assert(LOW_LUT_K > 0 && LOW_LUT_K <= 14);
    static_assert(LOW_LUT_K <= 6,
                  "exhaustive CUDA CROSS5 helper selftest intentionally targets a small factor");
    int visible = 0;
    cudaError_t ce = cudaGetDeviceCount(&visible);
    if (ce != cudaSuccess || visible < 1) {
        std::cout << "bucket-cross5-selftest SKIP no CUDA device\n";
        return 0;
    }
    ck(cudaSetDevice(0), "cross5 selftest set device");
    p10dc_install_cross5_lut();

    uint32_t ncode = 1;
    for (int i = 0; i < LOW_LUT_K; ++i) ncode *= 3u;
    std::vector<uint32_t> h_codes(ncode), h_direct(ncode);
    std::vector<Count> h_row(ncode);
    for (uint32_t k = 0; k < ncode; ++k) {
        h_codes[k] = cross5_pack_ternary(k);
        // For this helper test only the owner-local rank is consumed. At W10
        // ncode=243, so the ternary key itself is a valid synthetic rank.
        h_direct[k] = k;
        h_row[k] = Count((uint64_t(k) * 2654435761ull + 0x1618u) % 4294967291ull);
    }

    uint32_t *d_codes = nullptr, *d_direct = nullptr, *d_mismatch = nullptr, *d_checked = nullptr;
    Count* d_row = nullptr;
    ck(cudaMalloc(&d_codes, h_codes.size() * sizeof(uint32_t)), "cross5 codes alloc");
    ck(cudaMalloc(&d_direct, h_direct.size() * sizeof(uint32_t)), "cross5 direct alloc");
    ck(cudaMalloc(&d_row, h_row.size() * sizeof(Count)), "cross5 row alloc");
    ck(cudaMalloc(&d_mismatch, sizeof(uint32_t)), "cross5 mismatch alloc");
    ck(cudaMalloc(&d_checked, sizeof(uint32_t)), "cross5 checked alloc");
    ck(cudaMemcpy(d_codes, h_codes.data(), h_codes.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "cross5 codes H2D");
    ck(cudaMemcpy(d_direct, h_direct.data(), h_direct.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "cross5 direct H2D");
    ck(cudaMemcpy(d_row, h_row.data(), h_row.size() * sizeof(Count), cudaMemcpyHostToDevice), "cross5 row H2D");
    ck(cudaMemset(d_mismatch, 0, sizeof(uint32_t)), "cross5 mismatch clear");
    ck(cudaMemset(d_checked, 0, sizeof(uint32_t)), "cross5 checked clear");
    ck(cudaMemcpyToSymbol(D_BKF_LOW_DIRECT, &d_direct, sizeof(d_direct)), "cross5 bind low direct");
    Count mod = 4294967291u;
    ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "cross5 modulus");

    uint32_t total = ncode * 15u;
    cross5_compare_kernel<<<(total + 255u) / 256u, 256>>>(d_codes, ncode, d_row, d_mismatch, d_checked);
    ck(cudaGetLastError(), "cross5 compare launch");
    ck(cudaDeviceSynchronize(), "cross5 compare sync");
    uint32_t mismatch = 0, checked = 0;
    ck(cudaMemcpy(&mismatch, d_mismatch, sizeof(mismatch), cudaMemcpyDeviceToHost), "cross5 mismatch D2H");
    ck(cudaMemcpy(&checked, d_checked, sizeof(checked), cudaMemcpyDeviceToHost), "cross5 checked D2H");

    cudaFree(d_checked); cudaFree(d_mismatch); cudaFree(d_row); cudaFree(d_direct); cudaFree(d_codes);
    if (mismatch || checked != total) {
        std::cerr << "bucket-cross5-selftest mismatch=" << mismatch
                  << " checked=" << checked << " expected=" << total << '\n';
        return 2;
    }
    std::cout << "bucket-cross5-selftest OK W=" << TARGET_W
              << " low_k=" << LOW_LUT_K
              << " codes=" << ncode
              << " depths=15 checked=" << checked
              << " mismatches=0 table_bytes=6561 scalar_equivalent=1 prekey_equivalent=1 pm_accum=" << GPU_DIRECT_PM_ACCUM
              << '\n';
    return 0;
}
