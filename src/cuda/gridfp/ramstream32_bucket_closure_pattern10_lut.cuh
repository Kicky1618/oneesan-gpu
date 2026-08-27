#pragma once

#include "../../common/gridfp_closure_pattern10_lut.hpp"
#include "ramstream32_bucket_onepass_pattern10_depth8_alias.cuh"

#include <array>
#include <cstdint>

__constant__ uint32_t* D_P10LUT_LOW = nullptr;
__constant__ uint32_t* D_P10LUT_HIGH = nullptr;
__constant__ uint32_t D_P10LUT_LOW_OFF[LOW_LUT_K + 2];
__constant__ uint32_t D_P10LUT_HIGH_OFF[HIGH_LUT_K + 2];

struct BucketPattern10DecodeLutHost {
    oneesan::gridfp::ClosurePattern10LutHost<LOW_LUT_K> low;
    oneesan::gridfp::ClosurePattern10LutHost<HIGH_LUT_K> high;
    size_t bytes() const { return low.bytes() + high.bytes(); }
};

static BucketPattern10DecodeLutHost build_bucket_pattern10_decode_lut() {
    BucketPattern10DecodeLutHost out;
    out.low = oneesan::gridfp::build_closure_pattern10_lut<LOW_LUT_K>();
    out.high = oneesan::gridfp::build_closure_pattern10_lut<HIGH_LUT_K>();
    std::cerr << "bucket_pattern10_decode_lut low_entries=" << out.low.packed.size()
              << " high_entries=" << out.high.packed.size()
              << " bytes=" << out.bytes()
              << " kib=" << double(out.bytes()) / 1024.0 << '\n';
    return out;
}

struct BucketPattern10DecodeLutDeviceTables {
    uint32_t* low = nullptr;
    uint32_t* high = nullptr;

    static void cp(uint32_t*& d, const std::vector<uint32_t>& s, const char* what) {
        if (s.empty()) return;
        ck(cudaMalloc(&d, s.size() * sizeof(uint32_t)), what);
        ck(cudaMemcpy(d, s.data(), s.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), what);
    }
    void install(const BucketPattern10DecodeLutHost& h) {
        cp(low, h.low.packed, "pattern10 low decode LUT");
        cp(high, h.high.packed, "pattern10 high decode LUT");
        ck(cudaMemcpyToSymbol(D_P10LUT_LOW, &low, sizeof(low)), "pattern10 low LUT ptr");
        ck(cudaMemcpyToSymbol(D_P10LUT_HIGH, &high, sizeof(high)), "pattern10 high LUT ptr");
        ck(cudaMemcpyToSymbol(D_P10LUT_LOW_OFF, h.low.off.data(), h.low.off.size() * sizeof(uint32_t)), "pattern10 low LUT offsets");
        ck(cudaMemcpyToSymbol(D_P10LUT_HIGH_OFF, h.high.off.data(), h.high.off.size() * sizeof(uint32_t)), "pattern10 high LUT offsets");
    }
    void release() {
        cudaFree(low); cudaFree(high); low = high = nullptr;
    }
};

__device__ __forceinline__ void bkcp10_decode_low_lut(
    uint16_t id, int, int p, uint16_t& lm, uint16_t& rm
) {
    if (id == oneesan::gridfp::CLOSURE_PATTERN10_NONE || p <= 0 || p > LOW_LUT_K) {
        lm = rm = 0; return;
    }
    uint32_t z = D_P10LUT_LOW[D_P10LUT_LOW_OFF[p] + uint32_t(id)];
    lm = uint16_t(z); rm = uint16_t(z >> 16);
}

__device__ __forceinline__ void bkcp10_decode_high_lut(
    uint16_t id, int, int p, uint16_t& lm, uint16_t& rm
) {
    if (id == oneesan::gridfp::CLOSURE_PATTERN10_NONE || p <= 0 || p > HIGH_LUT_K) {
        lm = rm = 0; return;
    }
    uint32_t z = D_P10LUT_HIGH[D_P10LUT_HIGH_OFF[p] + uint32_t(id)];
    lm = uint16_t(z); rm = uint16_t(z >> 16);
}
