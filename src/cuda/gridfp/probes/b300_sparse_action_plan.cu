#include <cuda_runtime.h>

#include <cstdint>
#include <iomanip>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"
#include "../ramstream32_cpu_low_inplace.hpp"

int main() {
    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost low = build_cpu_low_orbit(storage, layout, lowdesc);
    HighOrbitHost high = build_high_orbit(storage, layout);

    uint64_t low_obs = uint64_t(low.main_total) * LOW_LUT_K;
    uint64_t high_obs = uint64_t(high.main_total) * HIGH_LUT_K;
    uint64_t low_actions = low.orbit_sources + low.closures;
    uint64_t high_actions = high.orbit_sources + high.closures;

    // Proposed sparse GPU stream:
    //   orbit owner: 12 bytes = source rank + partner rank/block + dropped rank/block/kind
    //   closure:      8 bytes = source rank + packed descriptor destination
    // Per-(p,bid) uint32 offsets are negligible but included below.
    uint64_t low_sparse = low.orbit_sources * 12ull + low.closures * 8ull;
    uint64_t high_sparse = high.orbit_sources * 12ull + high.closures * 8ull;
    uint64_t low_offsets = uint64_t(LOW_LUT_K) * (layout.main_blocks.size() + 1) * 2ull * 4ull;
    uint64_t high_offsets = uint64_t(HIGH_LUT_K) * (layout.main_blocks.size() + 1) * 2ull * 4ull;
    low_sparse += low_offsets;
    high_sparse += high_offsets;

    uint64_t dense_orbit = low.rec.size() * sizeof(uint64_t)
                         + high.rec.size() * sizeof(uint64_t);
    uint64_t dense_desc = (lowdesc.main_desc.size() + lowdesc.block_desc.size()
                         + highdesc.main_desc.size() + highdesc.block_desc.size())
                        * sizeof(uint32_t);
    uint64_t dense_combined = dense_orbit + dense_desc;
    uint64_t sparse_combined = low_sparse + high_sparse;

    auto mib = [](uint64_t x) { return double(x) / double(1ull << 20); };
    std::cout << std::fixed << std::setprecision(3)
        << "b300-sparse-action-plan W=" << TARGET_W
        << " low_observations=" << low_obs
        << " low_orbit=" << low.orbit_sources
        << " low_closure=" << low.closures
        << " low_action_fraction=" << (low_obs ? double(low_actions) / low_obs : 0.0)
        << " low_noop_fraction=" << (low_obs ? 1.0 - double(low_actions) / low_obs : 0.0)
        << " low_sparse_mib=" << mib(low_sparse)
        << '\n'
        << "high_observations=" << high_obs
        << " high_orbit=" << high.orbit_sources
        << " high_closure=" << high.closures
        << " high_action_fraction=" << (high_obs ? double(high_actions) / high_obs : 0.0)
        << " high_noop_fraction=" << (high_obs ? 1.0 - double(high_actions) / high_obs : 0.0)
        << " high_sparse_mib=" << mib(high_sparse)
        << '\n'
        << "dense_orbit_mib=" << mib(dense_orbit)
        << " dense_descriptor_mib=" << mib(dense_desc)
        << " dense_combined_mib=" << mib(dense_combined)
        << " sparse_combined_mib=" << mib(sparse_combined)
        << " metadata_saved_mib=" << mib(dense_combined - sparse_combined)
        << " metadata_saved_fraction="
        << (dense_combined ? 1.0 - double(sparse_combined) / dense_combined : 0.0)
        << '\n';
    return 0;
}
