#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"
#include "../ramstream32_cpu_low_inplace.hpp"
#include "../ramstream32_b300_sparse_actions.cuh"

static uint32_t mp_high_mask_from_rank(
    const StorageFactorHost& storage, int h, uint32_t hr
) {
    uint32_t code = storage.high_all_codes[storage.high_all_off[h] + hr];
    return seg_occ(code, HIGH_LUT_K);
}

int main() {
    constexpr int NG = 8;
    constexpr int S = MAXW + 2;
    constexpr uint32_t NM = 1u << HIGH_LUT_K;

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost low = build_cpu_low_orbit(storage, layout, lowdesc);
    HighOrbitHost high = build_high_orbit(storage, layout);
    B300SparseActionsHost sparse = build_b300_sparse_actions(
        layout, lowdesc, low, highdesc, high);

    std::vector<unsigned long long> weight(NM, 0);
    auto add_block_weight = [&](const StorageBlock& b) {
        if (!b.valid || !b.rows || !b.cols) return;
        for (uint32_t mask = 0; mask < NM; ++mask) {
            size_t ix = size_t(mask) * S + b.he;
            uint32_t a = G_FACTOR.high_mask_off[ix];
            uint32_t e = G_FACTOR.high_mask_off[ix + 1];
            weight[mask] += (unsigned long long)(e - a) * b.cols * sizeof(Count);
        }
    };
    for (const auto& b : layout.main_blocks) add_block_weight(b);
    for (const auto& b : layout.block_blocks) add_block_weight(b);

    // Largest-processing-time greedy bin packing by authoritative HBM bytes.
    std::vector<uint32_t> order(NM);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        if (weight[a] != weight[b]) return weight[a] > weight[b];
        return a < b;
    });
    std::array<unsigned long long, NG> gpu_bytes{};
    std::vector<uint8_t> owner(NM, 0);
    for (uint32_t mask : order) {
        int g = int(std::min_element(gpu_bytes.begin(), gpu_bytes.end()) - gpu_bytes.begin());
        owner[mask] = uint8_t(g);
        gpu_bytes[g] += weight[mask];
    }

    std::array<unsigned long long, NG> high_source_cells{};
    std::array<unsigned long long, NG> high_orbit_ops{}, high_closure_ops{};
    unsigned long long high_orbit_cells = 0;
    unsigned long long high_partner_remote = 0;
    unsigned long long high_drop_remote = 0;
    unsigned long long high_closure_cells = 0;
    unsigned long long high_closure_remote = 0;

    for (const auto& op : sparse.high_orbit) {
        uint32_t sb = b300_sparse_sblock(op);
        uint32_t jb = b300_sparse_jblock(op);
        uint32_t db = b300_sparse_dblock(op);
        const auto& x = layout.main_blocks[sb];
        const auto& jy = layout.main_blocks[jb];
        const auto& dy = layout.block_blocks[db];
        uint32_t sm = mp_high_mask_from_rank(storage, x.he, b300_sparse_src(op));
        uint32_t jm = mp_high_mask_from_rank(storage, jy.he, b300_sparse_jrank(op));
        uint32_t dm = mp_high_mask_from_rank(storage, dy.he, b300_sparse_drank(op));
        int so = owner[sm];
        ++high_orbit_ops[so];
        high_source_cells[so] += x.cols;
        high_orbit_cells += x.cols;
        if (owner[jm] != so) high_partner_remote += x.cols;
        if (owner[dm] != so) high_drop_remote += x.cols;
    }
    for (uint64_t op : sparse.high_closure) {
        uint32_t sb = b300_sparse_closure_sblock(op);
        uint32_t desc = b300_sparse_closure_desc(op);
        const auto& x = layout.main_blocks[sb];
        const auto& y = layout.block_blocks[highdesc_block(desc)];
        uint32_t sm = mp_high_mask_from_rank(storage, x.he, b300_sparse_closure_src(op));
        uint32_t dm = mp_high_mask_from_rank(storage, y.he, highdesc_rank(desc));
        int so = owner[sm];
        ++high_closure_ops[so];
        high_source_cells[so] += x.cols;
        high_closure_cells += x.cols;
        if (owner[dm] != so) high_closure_remote += x.cols;
    }

    uint64_t low_sparse_bytes = sparse.low_orbit.size() * sizeof(B300SparseOrbitOp)
                              + sparse.low_closure.size() * sizeof(uint64_t)
                              + (sparse.low_orbit_off.size() + sparse.low_closure_off.size()) * sizeof(uint32_t);
    uint64_t common_bytes = uint64_t(storage.low_all_codes.size() + storage.high_all_codes.size()) * sizeof(uint32_t)
                          + uint64_t(storage.low_mask_begin.size() + storage.high_mask_begin.size()) * sizeof(uint32_t)
                          + uint64_t(G_FACTOR.low_mask_codes.size() + G_FACTOR.low_mask_off.size()
                                   + G_FACTOR.high_mask_codes.size() + G_FACTOR.high_mask_off.size()) * sizeof(uint32_t)
                          + uint64_t(NM) * sizeof(uint8_t);
    std::array<unsigned long long, NG> high_sparse_bytes{};
    for (int g = 0; g < NG; ++g)
        high_sparse_bytes[g] = high_orbit_ops[g] * sizeof(B300SparseOrbitOp)
                             + high_closure_ops[g] * sizeof(uint64_t)
                             + 2ull * (HIGH_LUT_K + 1) * sizeof(uint32_t);

    auto gib = [](long double x) { return double(x / (1ull << 30)); };
    auto mib = [](long double x) { return double(x / (1ull << 20)); };
    auto frac = [](unsigned long long a, unsigned long long b) {
        return b ? double(a) / double(b) : 0.0;
    };

    unsigned long long mn = gpu_bytes[0], mx = gpu_bytes[0];
    unsigned long long work_mn = high_source_cells[0], work_mx = high_source_cells[0];
    long double max_need = 0;
    for (int g = 0; g < NG; ++g) {
        mn = std::min(mn, gpu_bytes[g]);
        mx = std::max(mx, gpu_bytes[g]);
        work_mn = std::min(work_mn, high_source_cells[g]);
        work_mx = std::max(work_mx, high_source_cells[g]);
        max_need = std::max(max_need, (long double)gpu_bytes[g] + low_sparse_bytes
                                      + common_bytes + high_sparse_bytes[g]);
    }

    std::cout << std::fixed << std::setprecision(6)
        << "b300-direct-maskshard-plan W=" << TARGET_W << " gpus=" << NG
        << " masks=" << NM
        << " auth_min_gib=" << gib(mn)
        << " auth_max_gib=" << gib(mx)
        << " auth_imbalance=" << (mn ? double(mx) / double(mn) : 0.0)
        << " max_need_gib=" << gib(max_need)
        << " headroom_288GB_gib=" << gib(288.0e9L - max_need)
        << " headroom_279GB_gib=" << gib(279.0e9L - max_need)
        << '\n';
    for (int g = 0; g < NG; ++g)
        std::cout << "maskshard_gpu=" << g
                  << " auth_gib=" << gib(gpu_bytes[g])
                  << " high_sparse_mib=" << mib(high_sparse_bytes[g])
                  << " high_source_cells=" << high_source_cells[g] << '\n';
    std::cout
        << "low_sparse_replicated_mib=" << mib(low_sparse_bytes)
        << " common_meta_mib=" << mib(common_bytes)
        << " high_work_imbalance=" << (work_mn ? double(work_mx) / double(work_mn) : 0.0)
        << '\n'
        << "low_orbit_remote_fraction=0.000000"
        << " low_closure_remote_fraction=0.000000"
        << " low_cross_remote_fraction=0.000000"
        << " low_reason=HIGH_occupancy_is_invariant"
        << '\n'
        << "high_orbit_cells=" << high_orbit_cells
        << " high_partner_remote_cells=" << high_partner_remote
        << " high_partner_remote_fraction=" << frac(high_partner_remote, high_orbit_cells)
        << " high_drop_remote_cells=" << high_drop_remote
        << " high_drop_remote_fraction=" << frac(high_drop_remote, high_orbit_cells)
        << " high_closure_cells=" << high_closure_cells
        << " high_closure_remote_cells=" << high_closure_remote
        << " high_closure_remote_fraction=" << frac(high_closure_remote, high_closure_cells)
        << '\n'
        << "runtime_groups=0 bulk_gather_scatter_bytes=0"
        << " shard_unit=HIGH_occupancy_class"
        << '\n';
    return 0;
}
