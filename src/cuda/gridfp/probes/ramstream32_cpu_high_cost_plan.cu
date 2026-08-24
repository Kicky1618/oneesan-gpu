#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_high_direct.hpp"

static uint32_t group_from_low_mask(uint32_t mask) {
    uint32_t g = 0;
    for (int pos = 0; pos < LOW_LUT_K; ++pos)
        if (mask & (1u << pos)) g |= 1u << (LOW_LUT_K - 1 - pos);
    return g;
}

static unsigned long long boundary_exposure_upper(
    unsigned long long rows, uint32_t width, uint32_t col0,
    uint32_t total_cols, unsigned long long page_bytes
) {
    if (!rows || !width) return 0;
    unsigned boundaries = 0;
    if (col0 != 0) ++boundaries;
    if (uint64_t(col0) + width != total_cols) ++boundaries;
    if (!boundaries) return 0;
    unsigned long long interval_bytes =
        static_cast<unsigned long long>(width) * sizeof(Count);
    unsigned long long exposed_per_row = std::min(
        interval_bytes, static_cast<unsigned long long>(boundaries) * page_bytes);
    return rows * exposed_per_row;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? std::atoi(argv[1]) : TARGET_W - 1;
    int W = n + 1;
    if (W != TARGET_W || n < 2 || W > MAXW) return 1;
    if constexpr (LOW_LUT_K + HIGH_LUT_K != TARGET_W - 1) return 1;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    CpuHighDirectHost direct = build_cpu_high_direct(storage, layout, highdesc);

    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr int S = FactorTablesHost::STRIDE;
    constexpr unsigned long long PAGE4K = 4096ull;
    constexpr unsigned long long PAGE2M = 2ull << 20;
    const uint32_t nmasks = 1u << L;

    unsigned long long sum_main = 0;
    unsigned long long sum_block = 0;

    std::cout
        << "group\tmask\troundtrip_bytes\tmain_states\tblocked_states"
        << "\tauthoritative_bytes\tgpu_state_steps\tpcie_copy_calls"
        << "\tpage4k_boundary_upper_bytes\tpage2m_boundary_upper_bytes"
        << "\tnn_cells\tnrnl_cells\tblock_closure_cells\tcross_closure_cells"
        << "\ttotal_cells\n";

    for (uint32_t mask = 0; mask < nmasks; ++mask) {
        unsigned long long main_states = 0;
        unsigned long long blocked_states = 0;
        unsigned long long pcie_copy_calls = 0;
        unsigned long long page4k_upper = 0;
        unsigned long long page2m_upper = 0;
        unsigned long long nn_cells = 0;
        unsigned long long nrnl_cells = 0;
        unsigned long long block_cells = 0;
        unsigned long long cross_cells = 0;

        for (uint32_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            const StorageBlock& sb = layout.main_blocks[bid];
            if (!sb.valid || !sb.rows) continue;
            uint32_t width = factor_count(G_FACTOR.low_mask_off, mask, sb.hs);
            uint32_t col0 = storage.low_mask_begin[size_t(mask) * S + sb.hs];
            if (width) pcie_copy_calls += 2; // H2D + D2H for this main slice.
            main_states += static_cast<unsigned long long>(sb.rows) * width;
            page4k_upper += boundary_exposure_upper(
                sb.rows, width, col0, sb.cols, PAGE4K);
            page2m_upper += boundary_exposure_upper(
                sb.rows, width, col0, sb.cols, PAGE2M);

            for (uint32_t pi = 0; pi < uint32_t(H); ++pi) {
                auto [na, nb] = cpu_high_direct_range(
                    direct.orbit_off.nn, direct.nblocks, pi, bid);
                auto [ra, rb] = cpu_high_direct_range(
                    direct.orbit_off.nrnl, direct.nblocks, pi, bid);
                auto [ba, bb] = cpu_high_direct_range(
                    direct.closure_off.block, direct.nblocks, pi, bid);
                auto [ca, cb] = cpu_high_direct_range(
                    direct.closure_off.cross, direct.nblocks, pi, bid);
                nn_cells += static_cast<unsigned long long>(nb - na) * width;
                nrnl_cells += static_cast<unsigned long long>(rb - ra) * width;
                block_cells += static_cast<unsigned long long>(bb - ba) * width;
                cross_cells += static_cast<unsigned long long>(cb - ca) * width;
            }
        }

        for (const StorageBlock& sb : layout.block_blocks) {
            if (!sb.valid || !sb.rows) continue;
            uint32_t width = factor_count(G_FACTOR.low_mask_off, mask, sb.hs);
            uint32_t col0 = storage.low_mask_begin[size_t(mask) * S + sb.hs];
            if (width) pcie_copy_calls += 2; // H2D + D2H for this blocked slice.
            blocked_states += static_cast<unsigned long long>(sb.rows) * width;
            page4k_upper += boundary_exposure_upper(
                sb.rows, width, col0, sb.cols, PAGE4K);
            page2m_upper += boundary_exposure_upper(
                sb.rows, width, col0, sb.cols, PAGE2M);
        }

        unsigned long long total_cells = nn_cells + nrnl_cells + block_cells + cross_cells;
        unsigned long long authoritative_bytes =
            (main_states + blocked_states) * sizeof(Count);
        unsigned long long roundtrip_bytes = 2ull * authoritative_bytes;
        unsigned long long gpu_state_steps =
            static_cast<unsigned long long>(H) * (main_states + blocked_states);
        uint32_t group = group_from_low_mask(mask);

        std::cout << group << '\t' << mask << '\t' << roundtrip_bytes
                  << '\t' << main_states << '\t' << blocked_states
                  << '\t' << authoritative_bytes << '\t' << gpu_state_steps
                  << '\t' << pcie_copy_calls
                  << '\t' << page4k_upper << '\t' << page2m_upper
                  << '\t' << nn_cells << '\t' << nrnl_cells
                  << '\t' << block_cells << '\t' << cross_cells
                  << '\t' << total_cells << '\n';

        sum_main += main_states;
        sum_block += blocked_states;
    }

    if (sum_main != layout.main_size || sum_block != layout.block_size) {
        std::cerr << "CPU HIGH cost plan partition mismatch main="
                  << sum_main << '/' << layout.main_size
                  << " block=" << sum_block << '/' << layout.block_size << '\n';
        return 2;
    }

    std::cerr << "cpu_high_cost_plan OK n=" << n
              << " groups=" << nmasks
              << " main_states=" << sum_main
              << " blocked_states=" << sum_block
              << '\n';
    return 0;
}
