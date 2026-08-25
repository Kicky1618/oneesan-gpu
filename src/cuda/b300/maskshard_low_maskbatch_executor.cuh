#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "maskshard_low_maskbatch_kernels.cuh"

#ifndef MASKSHARD_LOW_MASKBATCH_TARGET_TASKS_PER_CTA
#define MASKSHARD_LOW_MASKBATCH_TARGET_TASKS_PER_CTA 16384
#endif
#ifndef MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS
#define MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS 16
#endif
static_assert(MASKSHARD_LOW_MASKBATCH_TARGET_TASKS_PER_CTA >= 1,
              "LOW mask-batch target tasks per CTA must be positive");
static_assert(MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS >= 1
              && MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS <= 65535,
              "LOW mask-batch replica cap must fit uint16");

struct MaskShardLowMaskBatchRowDevicePlan {
    std::uint32_t orbit_count = 0;
    std::array<std::uint32_t, LOW_LUT_K + 1> closure_off{};
};

#ifdef MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS
struct MaskShardLowMaskBatchResidentCapPlan {
    std::uint32_t orbit_off = 0;
    std::uint32_t orbit_count = 0;
    std::array<std::uint32_t, LOW_LUT_K + 1> closure_off{};
};
#endif

struct MaskShardLowMaskBatchExecutor {
    static constexpr int ROW_CAPS = (TARGET_W + 1) / 2;

    int ngpu = 0;
    std::uint64_t target_tasks_per_cta =
        std::uint64_t(MASKSHARD_LOW_MASKBATCH_TARGET_TASKS_PER_CTA);
    int max_replicas = MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS;
    MaskShardLowMaskBatchTablesHost tasks;
    std::array<MaskShardLowMaskBatchDeviceTables, 8> cfg;
    std::array<MaskShardLowBatchDeviceDesc*, 8> d_orbit{};
    std::array<MaskShardLowBatchDeviceDesc*, 8> d_closure{};

#ifndef MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS
    std::array<std::size_t, 8> orbit_cap{};
    std::array<std::size_t, 8> closure_cap{};
    std::array<std::vector<MaskShardLowBatchDeviceDesc>, 8> h_orbit;
    std::array<std::vector<MaskShardLowBatchDeviceDesc>, 8> h_closure;
    std::array<MaskShardLowMaskBatchRowDevicePlan, 8> row_dev{};
#else
    std::array<std::vector<MaskShardLowBatchDeviceDesc>, 8> h_orbit_all;
    std::array<std::vector<MaskShardLowBatchDeviceDesc>, 8> h_closure_all;
    std::array<std::array<MaskShardLowMaskBatchResidentCapPlan, ROW_CAPS + 1>, 8>
        resident{};
#endif

#ifndef MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS
    static void ensure_desc(
        MaskShardLowBatchDeviceDesc** ptr,
        std::size_t& cap,
        std::size_t need,
        const char* what
    ) {
        if (need <= cap) return;
        if (*ptr) ck(cudaFree(*ptr), what);
        cap = std::max<std::size_t>(need, cap ? cap * 2 : 1024);
        ck(cudaMalloc(ptr, cap * sizeof(MaskShardLowBatchDeviceDesc)), what);
    }
#endif

#ifdef MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS
    void build_resident_rows(const MaskShardLayout& shard) {
        for (int cap = 1; cap <= ROW_CAPS; ++cap) {
            const MaskShardLowMaskBatchRowPlan row =
                maskshard_build_low_maskbatch_row_plan(
                    shard, tasks, cap - 1,
                    target_tasks_per_cta, max_replicas);
            for (int d = 0; d < ngpu; ++d) {
                MaskShardLowMaskBatchResidentCapPlan& rp = resident[d][cap];
                rp.orbit_off = std::uint32_t(h_orbit_all[d].size());
                std::vector<MaskShardLowBatchDeviceDesc> orbit =
                    cfg[d].bind(row.orbit[d]);
                h_orbit_all[d].insert(
                    h_orbit_all[d].end(), orbit.begin(), orbit.end());
                rp.orbit_count = std::uint32_t(orbit.size());

                rp.closure_off[0] = std::uint32_t(h_closure_all[d].size());
                for (int pi = 0; pi < LOW_LUT_K; ++pi) {
                    std::vector<MaskShardLowBatchDeviceDesc> closure =
                        cfg[d].bind(row.closure[d][pi]);
                    h_closure_all[d].insert(
                        h_closure_all[d].end(), closure.begin(), closure.end());
                    rp.closure_off[pi + 1] =
                        std::uint32_t(h_closure_all[d].size());
                }
            }
        }

        for (int d = 0; d < ngpu; ++d) {
            ck(cudaSetDevice(d), "LOW mask-batch resident row device");
            if (!h_orbit_all[d].empty()) {
                ck(cudaMalloc(&d_orbit[d],
                              h_orbit_all[d].size() * sizeof(h_orbit_all[d][0])),
                   "LOW mask-batch resident orbit desc alloc");
                ck(cudaMemcpy(d_orbit[d], h_orbit_all[d].data(),
                              h_orbit_all[d].size() * sizeof(h_orbit_all[d][0]),
                              cudaMemcpyHostToDevice),
                   "LOW mask-batch resident orbit desc copy");
            }
            if (!h_closure_all[d].empty()) {
                ck(cudaMalloc(&d_closure[d],
                              h_closure_all[d].size() * sizeof(h_closure_all[d][0])),
                   "LOW mask-batch resident closure desc alloc");
                ck(cudaMemcpy(d_closure[d], h_closure_all[d].data(),
                              h_closure_all[d].size() * sizeof(h_closure_all[d][0]),
                              cudaMemcpyHostToDevice),
                   "LOW mask-batch resident closure desc copy");
            }
            const std::size_t bytes =
                h_orbit_all[d].size() * sizeof(MaskShardLowBatchDeviceDesc)
                + h_closure_all[d].size() * sizeof(MaskShardLowBatchDeviceDesc);
            std::cerr << "LOW mask-batch resident rows dev=" << d
                      << " caps=" << ROW_CAPS
                      << " orbit_desc=" << h_orbit_all[d].size()
                      << " closure_desc=" << h_closure_all[d].size()
                      << " mib="
                      << double(bytes) / double(1ULL << 20) << '\n';
        }
    }
#endif

    void install(const MaskShardLayout& shard) {
        ngpu = shard.ngpu;
        tasks = maskshard_build_low_maskbatch_tables();
        for (int d = 0; d < ngpu; ++d) {
            ck(cudaSetDevice(d), "LOW mask-batch install device");
            cfg[d].install(d, shard, tasks);
        }
#ifdef MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS
        build_resident_rows(shard);
#endif
        std::cerr << "LOW mask-batch executor installed target_tasks_per_cta="
                  << target_tasks_per_cta
                  << " max_replicas=" << max_replicas
#ifdef MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS
                  << " resident_rows=1"
#else
                  << " resident_rows=0"
#endif
                  << '\n';
    }

    void prepare_row(const MaskShardLayout& shard, int zero_based_row) {
#ifdef MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS
        (void)shard;
        (void)zero_based_row;
#else
        const MaskShardLowMaskBatchRowPlan row =
            maskshard_build_low_maskbatch_row_plan(
                shard, tasks, zero_based_row,
                target_tasks_per_cta, max_replicas);

        for (int d = 0; d < ngpu; ++d) {
            h_orbit[d] = cfg[d].bind(row.orbit[d]);
            h_closure[d].clear();
            row_dev[d].closure_off.fill(0);
            for (int pi = 0; pi < LOW_LUT_K; ++pi) {
                row_dev[d].closure_off[pi] =
                    std::uint32_t(h_closure[d].size());
                std::vector<MaskShardLowBatchDeviceDesc> bound =
                    cfg[d].bind(row.closure[d][pi]);
                h_closure[d].insert(
                    h_closure[d].end(), bound.begin(), bound.end());
            }
            row_dev[d].closure_off[LOW_LUT_K] =
                std::uint32_t(h_closure[d].size());
            row_dev[d].orbit_count = std::uint32_t(h_orbit[d].size());

            ck(cudaSetDevice(d), "LOW mask-batch row device");
            ensure_desc(&d_orbit[d], orbit_cap[d], h_orbit[d].size(),
                        "LOW mask-batch orbit desc alloc");
            ensure_desc(&d_closure[d], closure_cap[d], h_closure[d].size(),
                        "LOW mask-batch closure desc alloc");
            if (!h_orbit[d].empty())
                ck(cudaMemcpy(d_orbit[d], h_orbit[d].data(),
                              h_orbit[d].size() * sizeof(h_orbit[d][0]),
                              cudaMemcpyHostToDevice),
                   "LOW mask-batch orbit desc copy");
            if (!h_closure[d].empty())
                ck(cudaMemcpy(d_closure[d], h_closure[d].data(),
                              h_closure[d].size() * sizeof(h_closure[d][0]),
                              cudaMemcpyHostToDevice),
                   "LOW mask-batch closure desc copy");
        }
#endif
    }

    void run_device_row(
        int d,
        Count* authoritative_main,
        Count* authoritative_block,
        int zero_based_row,
        int threads
    ) {
        if (d < 0 || d >= ngpu) return;
        ck(cudaSetDevice(d), "LOW mask-batch run device");
        const int cap = std::min(zero_based_row + 1, ROW_CAPS);

#ifdef MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS
        const MaskShardLowMaskBatchResidentCapPlan& rp = resident[d][cap];
        const MaskShardLowBatchDeviceDesc* orbit_desc =
            d_orbit[d] ? d_orbit[d] + rp.orbit_off : nullptr;
#else
        const MaskShardLowMaskBatchRowDevicePlan& rp = row_dev[d];
        const MaskShardLowBatchDeviceDesc* orbit_desc = d_orbit[d];
#endif

        for (int p = LOW_LUT_K; p >= 1; --p) {
            if (rp.orbit_count)
                maskshard_low_maskbatch_orbit_kernel<<<rp.orbit_count, threads>>>(
                    authoritative_main, authoritative_block,
                    orbit_desc, cfg[d].d_static, cfg[d].d_dynamic,
                    std::min(cap, TARGET_W / 2), p);
            ck(cudaGetLastError(), "LOW mask-batch orbit launch");

            const int pi = LOW_LUT_K - p;
            const std::uint32_t a = rp.closure_off[pi];
            const std::uint32_t z = rp.closure_off[pi + 1];
            if (z > a)
                maskshard_low_maskbatch_closure_kernel<<<z - a, threads>>>(
                    authoritative_main, authoritative_block,
                    d_closure[d] + a, cfg[d].d_static, cfg[d].d_dynamic,
                    cap, p);
            ck(cudaGetLastError(), "LOW mask-batch closure launch");
        }
        ck(cudaDeviceSynchronize(), "LOW mask-batch row sync");
    }

    void release() {
        for (int d = 0; d < ngpu; ++d) {
            ck(cudaSetDevice(d), "LOW mask-batch release device");
            if (d_orbit[d]) cudaFree(d_orbit[d]);
            if (d_closure[d]) cudaFree(d_closure[d]);
            d_orbit[d] = d_closure[d] = nullptr;
#ifndef MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS
            orbit_cap[d] = closure_cap[d] = 0;
            h_orbit[d].clear();
            h_closure[d].clear();
#else
            h_orbit_all[d].clear();
            h_closure_all[d].clear();
#endif
            cfg[d].release();
        }
        ngpu = 0;
    }
};
