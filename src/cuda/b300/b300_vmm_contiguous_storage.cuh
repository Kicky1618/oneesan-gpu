#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <vector>

namespace b300_vmm {

inline void ck_driver(CUresult e, const char* where) {
    if (e == CUDA_SUCCESS) return;
    const char* name = nullptr;
    const char* text = nullptr;
    cuGetErrorName(e, &name);
    cuGetErrorString(e, &text);
    std::fprintf(stderr, "%s: %s: %s (%d)\n", where,
                 name ? name : "CUDA_ERROR",
                 text ? text : "unknown driver error", int(e));
    std::exit(1);
}

inline void ck_runtime(cudaError_t e, const char* where) {
    if (e == cudaSuccess) return;
    std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e));
    std::exit(1);
}

inline std::size_t checked_lcm(std::size_t a, std::size_t b) {
    const std::size_t g = std::gcd(a, b);
    const std::size_t q = b / g;
    if (a > std::numeric_limits<std::size_t>::max() / q) {
        std::fprintf(stderr, "VMM granularity LCM overflow: %zu x %zu\n", a, b);
        std::exit(1);
    }
    return a * q;
}

inline std::size_t ceil_div(std::size_t a, std::size_t b) {
    return a / b + std::size_t(a % b != 0);
}

struct ContiguousStorage {
    CUdeviceptr va = 0;
    std::size_t logical_bytes = 0;
    std::size_t mapped_bytes = 0;
    std::size_t granularity = 0;
    std::size_t mapped_units = 0;
    int ngpu = 0;
    int extra_rotation = 0;
    std::vector<CUmemGenericAllocationHandle> handles;
    std::vector<std::size_t> offsets;
    std::vector<std::size_t> segment_bytes;

    template<class T>
    T* base_as() const {
        return reinterpret_cast<T*>(static_cast<std::uintptr_t>(va));
    }

    bool active() const { return va != 0; }

    void create(std::uint64_t elements, int devices, int rotation, const char* tag) {
        if (active()) {
            std::fprintf(stderr, "%s VMM storage already active\n", tag);
            std::exit(1);
        }
        if (devices < 1 || devices > 8) {
            std::fprintf(stderr, "%s VMM invalid device count %d\n", tag, devices);
            std::exit(1);
        }
        if (elements > std::uint64_t(std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t))) {
            std::fprintf(stderr, "%s VMM logical size overflow\n", tag);
            std::exit(1);
        }

        ck_driver(cuInit(0), "cuInit");
        ngpu = devices;
        extra_rotation = ((rotation % ngpu) + ngpu) % ngpu;
        logical_bytes = std::size_t(elements) * sizeof(std::uint32_t);

        std::vector<CUmemAllocationProp> props(static_cast<std::size_t>(ngpu));
        granularity = 1;
        for (int d = 0; d < ngpu; ++d) {
            ck_runtime(cudaSetDevice(d), "VMM set device for capability");
            ck_runtime(cudaFree(nullptr), "VMM initialize runtime context");
            CUdevice dev{};
            ck_driver(cuDeviceGet(&dev, d), "cuDeviceGet");
            int supported = 0;
            ck_driver(cuDeviceGetAttribute(&supported,
                                           CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,
                                           dev),
                      "cuDeviceGetAttribute(VMM)");
            if (!supported) {
                std::fprintf(stderr, "%s VMM unsupported on device %d\n", tag, d);
                std::exit(3);
            }
            auto& p = props[static_cast<std::size_t>(d)];
            p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            p.location.id = d;
            p.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
            std::size_t g = 0;
            ck_driver(cuMemGetAllocationGranularity(&g, &p,
                                                     CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                      "cuMemGetAllocationGranularity");
            if (g == 0 || (g % sizeof(std::uint32_t)) != 0) {
                std::fprintf(stderr, "%s VMM invalid granularity %zu on device %d\n", tag, g, d);
                std::exit(1);
            }
            granularity = checked_lcm(granularity, g);
        }

        mapped_units = ceil_div(logical_bytes, granularity);
        if (mapped_units < static_cast<std::size_t>(ngpu)) {
            std::fprintf(stderr,
                         "%s VMM needs at least one granularity unit per GPU: units=%zu ngpu=%d\n",
                         tag, mapped_units, ngpu);
            std::exit(1);
        }
        if (mapped_units > std::numeric_limits<std::size_t>::max() / granularity) {
            std::fprintf(stderr, "%s VMM mapped size overflow\n", tag);
            std::exit(1);
        }
        mapped_bytes = mapped_units * granularity;

        const std::size_t q = mapped_units / static_cast<std::size_t>(ngpu);
        const std::size_t r = mapped_units % static_cast<std::size_t>(ngpu);
        offsets.assign(static_cast<std::size_t>(ngpu) + 1, 0);
        segment_bytes.assign(static_cast<std::size_t>(ngpu), 0);
        handles.assign(static_cast<std::size_t>(ngpu), CUmemGenericAllocationHandle{});
        for (int d = 0; d < ngpu; ++d) {
            const int rel = (d - extra_rotation + ngpu) % ngpu;
            const std::size_t units = q + std::size_t(rel < int(r));
            const std::size_t bytes = units * granularity;
            segment_bytes[static_cast<std::size_t>(d)] = bytes;
            offsets[static_cast<std::size_t>(d) + 1] =
                offsets[static_cast<std::size_t>(d)] + bytes;
        }
        if (offsets.back() != mapped_bytes) {
            std::fprintf(stderr, "%s VMM internal layout mismatch\n", tag);
            std::exit(1);
        }

        ck_runtime(cudaSetDevice(0), "VMM set reserve device");
        ck_driver(cuMemAddressReserve(&va, mapped_bytes, granularity, 0, 0),
                  "cuMemAddressReserve");
        for (int d = 0; d < ngpu; ++d) {
            const std::size_t i = static_cast<std::size_t>(d);
            ck_driver(cuMemCreate(&handles[i], segment_bytes[i], &props[i], 0),
                      "cuMemCreate");
            ck_driver(cuMemMap(va + CUdeviceptr(offsets[i]), segment_bytes[i], 0,
                               handles[i], 0),
                      "cuMemMap");
        }

        std::vector<CUmemAccessDesc> access(static_cast<std::size_t>(ngpu));
        for (int d = 0; d < ngpu; ++d) {
            auto& a = access[static_cast<std::size_t>(d)];
            a.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            a.location.id = d;
            a.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        }
        ck_driver(cuMemSetAccess(va, mapped_bytes, access.data(), access.size()),
                  "cuMemSetAccess");

        std::size_t min_segment = segment_bytes.front();
        std::size_t max_segment = segment_bytes.front();
        for (std::size_t x : segment_bytes) {
            min_segment = std::min(min_segment, x);
            max_segment = std::max(max_segment, x);
        }
        std::fprintf(stderr,
                     "%s VMM: logical_gib=%.6f mapped_gib=%.6f padding_mib=%.6f "
                     "granularity_kib=%.3f min_segment_gib=%.6f max_segment_gib=%.6f "
                     "extra_rotation=%d\n",
                     tag,
                     double(logical_bytes) / double(1ull << 30),
                     double(mapped_bytes) / double(1ull << 30),
                     double(mapped_bytes - logical_bytes) / double(1ull << 20),
                     double(granularity) / 1024.0,
                     double(min_segment) / double(1ull << 30),
                     double(max_segment) / double(1ull << 30),
                     extra_rotation);
    }

    void zero_local_segments() const {
        for (int d = 0; d < ngpu; ++d) {
            const std::size_t i = static_cast<std::size_t>(d);
            ck_runtime(cudaSetDevice(d), "VMM zero set device");
            void* p = reinterpret_cast<void*>(
                static_cast<std::uintptr_t>(va + CUdeviceptr(offsets[i])));
            ck_runtime(cudaMemset(p, 0, segment_bytes[i]), "VMM zero segment");
        }
        for (int d = 0; d < ngpu; ++d) {
            ck_runtime(cudaSetDevice(d), "VMM zero sync set device");
            ck_runtime(cudaDeviceSynchronize(), "VMM zero sync");
        }
    }

    void destroy() {
        if (!active()) return;
        for (int d = 0; d < ngpu; ++d) {
            ck_runtime(cudaSetDevice(d), "VMM destroy set device");
            ck_runtime(cudaDeviceSynchronize(), "VMM destroy sync");
        }
        ck_runtime(cudaSetDevice(0), "VMM destroy reserve device");
        for (int d = 0; d < ngpu; ++d) {
            const std::size_t i = static_cast<std::size_t>(d);
            ck_driver(cuMemUnmap(va + CUdeviceptr(offsets[i]), segment_bytes[i]),
                      "cuMemUnmap");
        }
        for (auto h : handles) ck_driver(cuMemRelease(h), "cuMemRelease");
        ck_driver(cuMemAddressFree(va, mapped_bytes), "cuMemAddressFree");
        va = 0;
        logical_bytes = mapped_bytes = granularity = mapped_units = 0;
        ngpu = 0;
        extra_rotation = 0;
        handles.clear();
        offsets.clear();
        segment_bytes.clear();
    }
};

} // namespace b300_vmm
