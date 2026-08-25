#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "maskshard_low_maskbatch_range_kernels.cuh"

#ifndef MASKSHARD_LOW_MASKBATCH_RESIDENT_ROWS
#error "compact LOW batch ranges require resident-row planning"
#endif
#ifndef MASKSHARD_LOW_MASKBATCH_REBUILD_DYNAMIC
#error "compact LOW batch ranges require rebuild-dynamic mode"
#endif
#ifndef MASKSHARD_LOW_MASKBATCH_TARGET_TASKS_PER_CTA
#define MASKSHARD_LOW_MASKBATCH_TARGET_TASKS_PER_CTA 16384
#endif
#ifndef MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS
#define MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS 1024
#endif
static_assert(MASKSHARD_LOW_MASKBATCH_TARGET_TASKS_PER_CTA >= 1,
              "LOW range target tasks per CTA must be positive");
static_assert(MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS >= 1
              && MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS <= 65535,
              "LOW range replica cap must fit uint16");

struct MaskShardLowMaskBatchRangeResidentCapPlan {
    std::uint32_t orbit_group_off = 0;
    std::uint32_t orbit_group_count = 0;
    std::uint32_t orbit_ctas = 0;
    std::array<std::uint32_t, LOW_LUT_K + 1> closure_group_off{};
    std::array<std::uint32_t, LOW_LUT_K> closure_ctas{};
};

struct MaskShardLowMaskBatchExecutor {
    static constexpr int ROW_CAPS = (TARGET_W + 1) / 2;

    int ngpu = 0;
    std::uint64_t target_tasks_per_cta =
        std::uint64_t(MASKSHARD_LOW_MASKBATCH_TARGET_TASKS_PER_CTA);
    int max_replicas = MASKSHARD_LOW_MASKBATCH_MAX_REPLICAS;
    MaskShardLowMaskBatchTablesHost tasks;
    std::array<MaskShardLowMaskBatchDeviceTables, 8> cfg;
    std::array<MaskShardLowBatchRangeDeviceDesc*, 8> d_orbit{};
    std::array<MaskShardLowBatchRangeDeviceDesc*, 8> d_closure{};
    std::array<std::vector<MaskShardLowBatchRangeDeviceDesc>, 8> h_orbit;
    std::array<std::vector<MaskShardLowBatchRangeDeviceDesc>, 8> h_closure;
    std::array<
        std::array<MaskShardLowMaskBatchRangeResidentCapPlan, ROW_CAPS + 1>, 8>
        resident{};

    void build_resident_rows(const MaskShardLayout& shard) {
        for (int cap = 1; cap <= ROW_CAPS; ++cap) {
            const MaskShardLowMaskBatchRangeRowPlan row =
                maskshard_build_low_maskbatch_range_row_plan(
                    shard, tasks, cap - 1,
                    target_tasks_per_cta, max_replicas);
            for (int d = 0; d < ngpu; ++d) {
                MaskShardLowMaskBatchRangeResidentCapPlan& rp = resident[d][cap];

                rp.orbit_group_off = std::uint32_t(h_orbit[d].size());
                const MaskShardLowBatchBoundRanges orbit =
                    maskshard_bind_low_batch_ranges(cfg[d], row.orbit[d]);
                h_orbit[d].insert(h_orbit[d].end(),
                                  orbit.groups.begin(), orbit.groups.end());
                rp.orbit_group_count = std::uint32_t(orbit.groups.size());
                rp.orbit_ctas = orbit.ctas;

                rp.closure_group_off[0] = std::uint32_t(h_closure[d].size());
                for (int pi = 0; pi < LOW_LUT_K; ++pi) {
                    const MaskShardLowBatchBoundRanges closure =
                        maskshard_bind_low_batch_ranges(
                            cfg[d], row.closure[d][pi]);
                    h_closure[d].insert(h_closure[d].end(),
                                        closure.groups.begin(), closure.groups.end());
                    rp.closure_group_off[pi + 1] =
                        std::uint32_t(h_closure[d].size());
                    rp.closure_ctas[pi] = closure.ctas;
                }
            }
        }

        for (int d = 0; d < ngpu; ++d) {
            ck(cudaSetDevice(d), "LOW range resident row device");
            if (!h_orbit[d].empty()) {
                ck(cudaMalloc(&d_orbit[d],
                              h_orbit[d].size() * sizeof(h_orbit[d][0])),
                   "LOW range resident orbit alloc");
                ck(cudaMemcpy(d_orbit[d], h_orbit[d].data(),
                              h_orbit[d].size() * sizeof(h_orbit[d][0]),
                              cudaMemcpyHostToDevice),
                   "LOW range resident orbit copy");
            }
            if (!h_closure[d].empty()) {
                ck(cudaMalloc(&d_closure[d],
                              h_closure[d].size() * sizeof(h_closure[d][0])),
                   "LOW range resident closure alloc");
                ck(cudaMemcpy(d_closure[d], h_closure[d].data(),
                              h_closure[d].size() * sizeof(h_closure[d][0]),
                              cudaMemcpyHostToDevice),
                   "LOW range resident closure copy");
            }
            const std::size_t bytes =
                (h_orbit[d].size() + h_closure[d].size())
                * sizeof(MaskShardLowBatchRangeDeviceDesc);
            std::uint64_t max_cap_ctas = 0;
            for (int cap = 1; cap <= ROW_CAPS; ++cap) {
                const auto& rp = resident[d][cap];
                std::uint64_t z = std::uint64_t(LOW_LUT_K) * rp.orbit_ctas;
                for (int pi = 0; pi < LOW_LUT_K; ++pi)
                    z += rp.closure_ctas[pi];
                max_cap_ctas = std::max(max_cap_ctas, z);
            }
            std::cerr << "LOW mask-batch compact ranges dev=" << d
                      << " caps=" << ROW_CAPS
                      << " orbit_groups=" << h_orbit[d].size()
                      << " closure_groups=" << h_closure[d].size()
                      << " descriptor_mib="
                      << double(bytes) / double(1ULL << 20)
                      << " max_cap_ctas=" << max_cap_ctas << '\n';
        }
    }

    void install(const MaskShardLayout& shard) {
        ngpu = shard.ngpu;
        tasks = maskshard_build_low_maskbatch_tables();
        for (int d = 0; d < ngpu; ++d) {
            ck(cudaSetDevice(d), "LOW range install device");
            cfg[d].install(d, shard, tasks);
        }
        build_resident_rows(shard);
        std::cerr << "LOW mask-batch compact-range executor installed target_tasks_per_cta="
                  << target_tasks_per_cta
                  << " max_replicas=" << max_replicas
                  << " descriptor_mode=range-prefix\n";
    }

    void prepare_row(const MaskShardLayout&, int) {}

    void run_device_row(
        int d,
        Count* authoritative_main,
        Count* authoritative_block,
        int zero_based_row,
        int threads
    ) {
        if (d < 0 || d >= ngpu) return;
        ck(cudaSetDevice(d), "LOW range run device");
        const int cap = std::min(zero_based_row + 1, ROW_CAPS);
        const MaskShardLowMaskBatchRangeResidentCapPlan& rp = resident[d][cap];
        const MaskShardLowBatchRangeDeviceDesc* orbit_groups =
            d_orbit[d] ? d_orbit[d] + rp.orbit_group_off : nullptr;

        for (int p = LOW_LUT_K; p >= 1; --p) {
            if (rp.orbit_ctas) {
                maskshard_low_maskbatch_range_orbit_kernel<<<rp.orbit_ctas, threads>>>(
                    authoritative_main, authoritative_block,
                    orbit_groups, int(rp.orbit_group_count), cfg[d].d_static,
                    std::min(cap, TARGET_W / 2), p);
                ck(cudaGetLastError(), "LOW range orbit launch");
            }

            const int pi = LOW_LUT_K - p;
            const std::uint32_t ga = rp.closure_group_off[pi];
            const std::uint32_t gz = rp.closure_group_off[pi + 1];
            if (rp.closure_ctas[pi]) {
                maskshard_low_maskbatch_range_closure_kernel<<<
                    rp.closure_ctas[pi], threads>>>(
                    authoritative_main, authoritative_block,
                    d_closure[d] + ga, int(gz - ga), cfg[d].d_static,
                    cap, p);
                ck(cudaGetLastError(), "LOW range closure launch");
            }
        }
        ck(cudaDeviceSynchronize(), "LOW range row sync");
    }

    void release() {
        for (int d = 0; d < ngpu; ++d) {
            ck(cudaSetDevice(d), "LOW range release device");
            if (d_orbit[d]) cudaFree(d_orbit[d]);
            if (d_closure[d]) cudaFree(d_closure[d]);
            d_orbit[d] = d_closure[d] = nullptr;
            h_orbit[d].clear();
            h_closure[d].clear();
            cfg[d].release();
        }
        ngpu = 0;
    }
};
