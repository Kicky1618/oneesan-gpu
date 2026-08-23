#pragma once

#include "ramstream32_lowdesc.cuh"
#include "ramstream32_highdesc.cuh"

// The resident in-place orbit kernels consume blocked states through their
// owner orbit.  They never execute blocked_group_* descriptor kernels, so the
// LOW/HIGH blocked descriptor arrays are dead device metadata.  Upload only the
// main descriptors used by closure passes and explicitly null the blocked
// symbols to catch accidental future use.
struct B300MainDescDeviceTables {
    uint32_t* low_main = nullptr;
    uint32_t* high_main = nullptr;

    void install(const LowDescHost& low, const HighDescHost& high) {
        if (!low.main_desc.empty()) {
            ck(cudaMalloc(&low_main, low.main_desc.size() * sizeof(uint32_t)),
               "b300 low main desc alloc");
            ck(cudaMemcpy(low_main, low.main_desc.data(),
                          low.main_desc.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "b300 low main desc copy");
        }
        if (!high.main_desc.empty()) {
            ck(cudaMalloc(&high_main, high.main_desc.size() * sizeof(uint32_t)),
               "b300 high main desc alloc");
            ck(cudaMemcpy(high_main, high.main_desc.data(),
                          high.main_desc.size() * sizeof(uint32_t),
                          cudaMemcpyHostToDevice), "b300 high main desc copy");
        }

        uint32_t* nullp = nullptr;
        ck(cudaMemcpyToSymbol(D_LOWDESC_MAIN, &low_main, sizeof(low_main)),
           "b300 low main desc ptr");
        ck(cudaMemcpyToSymbol(D_LOWDESC_BLOCK, &nullp, sizeof(nullp)),
           "b300 low blocked desc null");
        ck(cudaMemcpyToSymbol(D_LOWDESC_MAIN_BASE, low.main_base.data(),
                              sizeof(uint32_t) * low.main_base.size()),
           "b300 low main desc base");
        ck(cudaMemcpyToSymbol(D_LOWDESC_MAIN_TOTAL, &low.main_total,
                              sizeof(low.main_total)), "b300 low main desc total");
        uint32_t zero = 0;
        ck(cudaMemcpyToSymbol(D_LOWDESC_BLOCK_TOTAL, &zero, sizeof(zero)),
           "b300 low blocked desc total zero");

        ck(cudaMemcpyToSymbol(D_HIGHDESC_MAIN, &high_main, sizeof(high_main)),
           "b300 high main desc ptr");
        ck(cudaMemcpyToSymbol(D_HIGHDESC_BLOCK, &nullp, sizeof(nullp)),
           "b300 high blocked desc null");
        ck(cudaMemcpyToSymbol(D_HIGHDESC_MAIN_BASE, high.main_base.data(),
                              sizeof(uint32_t) * high.main_base.size()),
           "b300 high main desc base");
        ck(cudaMemcpyToSymbol(D_HIGHDESC_MAIN_TOTAL, &high.main_total,
                              sizeof(high.main_total)), "b300 high main desc total");
        ck(cudaMemcpyToSymbol(D_HIGHDESC_BLOCK_TOTAL, &zero, sizeof(zero)),
           "b300 high blocked desc total zero");
    }

    size_t bytes(const LowDescHost& low, const HighDescHost& high) const {
        return (low.main_desc.size() + high.main_desc.size()) * sizeof(uint32_t);
    }

    static size_t saved_bytes(const LowDescHost& low, const HighDescHost& high) {
        return (low.block_desc.size() + high.block_desc.size()) * sizeof(uint32_t);
    }

    void release() {
        if (low_main) cudaFree(low_main);
        if (high_main) cudaFree(high_main);
        low_main = high_main = nullptr;
    }
};
