#include <cuda_runtime.h>

#include <cstdint>
#include <iomanip>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"
#include "../ramstream32_cpu_low_inplace.hpp"
#include "../ramstream32_b300_sparse_actions.cuh"

int main() {
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

    uint64_t low_obs = uint64_t(low.main_total) * LOW_LUT_K;
    uint64_t high_obs = uint64_t(high.main_total) * HIGH_LUT_K;
    uint64_t low_actions = sparse.low_orbit.size() + sparse.low_closure.size();
    uint64_t high_actions = sparse.high_orbit.size() + sparse.high_closure.size();

    uint64_t low_sparse = sparse.low_orbit.size() * sizeof(B300SparseOrbitOp)
                        + sparse.low_closure.size() * sizeof(uint64_t)
                        + (sparse.low_orbit_off.size() + sparse.low_closure_off.size()) * sizeof(uint32_t);
    uint64_t high_sparse = sparse.high_orbit.size() * sizeof(B300SparseOrbitOp)
                         + sparse.high_closure.size() * sizeof(uint64_t)
                         + (sparse.high_orbit_off.size() + sparse.high_closure_off.size()) * sizeof(uint32_t);

    uint64_t dense_orbit = low.rec.size() * sizeof(uint64_t)
                         + high.rec.size() * sizeof(uint64_t);
    uint64_t dense_desc_all = (lowdesc.main_desc.size() + lowdesc.block_desc.size()
                             + highdesc.main_desc.size() + highdesc.block_desc.size())
                            * sizeof(uint32_t);
    uint64_t dense_desc_main = (lowdesc.main_desc.size() + highdesc.main_desc.size())
                             * sizeof(uint32_t);
    uint64_t dense_all = dense_orbit + dense_desc_all;
    uint64_t dense_main_only = dense_orbit + dense_desc_main;
    uint64_t sparse_combined = sparse.bytes();

    auto mib = [](uint64_t x) { return double(x) / double(1ull << 20); };
    std::cout << std::fixed << std::setprecision(3)
        << "b300-sparse-action-plan W=" << TARGET_W
        << " low_observations=" << low_obs
        << " low_orbit=" << sparse.low_orbit.size()
        << " low_closure=" << sparse.low_closure.size()
        << " low_action_fraction=" << (low_obs ? double(low_actions) / low_obs : 0.0)
        << " low_noop_fraction=" << (low_obs ? 1.0 - double(low_actions) / low_obs : 0.0)
        << " low_sparse_mib=" << mib(low_sparse)
        << '\n'
        << "high_observations=" << high_obs
        << " high_orbit=" << sparse.high_orbit.size()
        << " high_closure=" << sparse.high_closure.size()
        << " high_action_fraction=" << (high_obs ? double(high_actions) / high_obs : 0.0)
        << " high_noop_fraction=" << (high_obs ? 1.0 - double(high_actions) / high_obs : 0.0)
        << " high_sparse_mib=" << mib(high_sparse)
        << '\n'
        << "dense_orbit_mib=" << mib(dense_orbit)
        << " dense_descriptor_all_mib=" << mib(dense_desc_all)
        << " dense_descriptor_main_only_mib=" << mib(dense_desc_main)
        << " dense_all_mib=" << mib(dense_all)
        << " dense_main_only_mib=" << mib(dense_main_only)
        << " sparse_combined_mib=" << mib(sparse_combined)
        << " saved_vs_dense_all_mib=" << mib(dense_all - sparse_combined)
        << " saved_vs_dense_main_only_mib=" << mib(dense_main_only - sparse_combined)
        << " saved_vs_dense_all_fraction="
        << (dense_all ? 1.0 - double(sparse_combined) / dense_all : 0.0)
        << '\n';

    for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) {
        std::cout << "high_edge p=" << p
                  << " orbit_ops=" << b300_sparse_high_orbit_count(sparse, p)
                  << " closure_ops=" << b300_sparse_high_closure_count(sparse, p) << '\n';
    }
    for (int p = LOW_LUT_K; p >= 1; --p) {
        std::cout << "low_edge p=" << p
                  << " orbit_ops=" << b300_sparse_low_orbit_count(sparse, p)
                  << " closure_ops=" << b300_sparse_low_closure_count(sparse, p) << '\n';
    }
    return 0;
}
