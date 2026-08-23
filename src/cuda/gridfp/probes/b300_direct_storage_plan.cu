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

    constexpr uint64_t S = MAXW + 2;
    uint64_t all_codes_bytes = uint64_t(storage.low_all_codes.size()
                                      + storage.high_all_codes.size()) * sizeof(uint32_t);
    uint64_t mask_begin_bytes = uint64_t(storage.low_mask_begin.size()
                                       + storage.high_mask_begin.size()) * sizeof(uint32_t);
    uint64_t mask_tables_bytes = uint64_t(G_FACTOR.low_mask_codes.size()
                                        + G_FACTOR.high_mask_codes.size()
                                        + G_FACTOR.low_mask_off.size()
                                        + G_FACTOR.high_mask_off.size()) * sizeof(uint32_t);
    uint64_t block_meta_bytes = uint64_t(layout.main_blocks.size()
                                       + layout.block_blocks.size()) * sizeof(StorageBlock);
    uint64_t direct_meta_bytes = sparse.bytes() + all_codes_bytes
                               + mask_begin_bytes + mask_tables_bytes + block_meta_bytes;

    uint64_t auth_bytes = uint64_t(layout.main_size + layout.block_size) * sizeof(Count);
    long double canonical_copy_per_grid_row =
        (4.0L * layout.main_size + 2.0L * layout.block_size) * sizeof(Count);
    long double canonical_copy_per_residue = canonical_copy_per_grid_row * TARGET_W;
    long double ramstream_pcie_per_grid_row =
        (2.0L * layout.main_size + 1.0L * layout.block_size) * sizeof(Count);
    long double ramstream_pcie_per_residue = ramstream_pcie_per_grid_row * TARGET_W;

    auto mib = [](long double x) { return double(x / (1ull << 20)); };
    auto gib = [](long double x) { return double(x / (1ull << 30)); };
    auto tib = [](long double x) { return double(x / (1ull << 40)); };

    long double auth_shard_bytes = (long double)auth_bytes / 8.0L;
    long double direct_need_bytes = auth_shard_bytes + direct_meta_bytes;
    long double cap288 = 288.0L * 1000.0L * 1000.0L * 1000.0L;
    long double cap279 = 279.0L * 1000.0L * 1000.0L * 1000.0L;

    std::cout << std::fixed << std::setprecision(3)
        << "b300-direct-storage-plan W=" << TARGET_W
        << " main_states=" << layout.main_size
        << " block_states=" << layout.block_size
        << " auth_total_gib=" << gib(auth_bytes)
        << " auth_shard_gib_x8=" << gib(auth_shard_bytes)
        << '\n'
        << "runtime_high_groups=0 runtime_low_groups=0 scratch_gib=0.000"
        << " canonical_maps_mib=0.000 dense_transition_meta_mib=0.000"
        << '\n'
        << "sparse_actions_mib=" << mib(sparse.bytes())
        << " storage_all_codes_mib=" << mib(all_codes_bytes)
        << " mask_begin_mib=" << mib(mask_begin_bytes)
        << " compact_mask_tables_mib=" << mib(mask_tables_bytes)
        << " block_meta_mib=" << mib(block_meta_bytes)
        << " direct_meta_mib=" << mib(direct_meta_bytes)
        << '\n'
        << "estimated_need_gib_per_gpu=" << gib(direct_need_bytes)
        << " headroom_288GB_gib=" << gib(cap288 - direct_need_bytes)
        << " headroom_279GB_gib=" << gib(cap279 - direct_need_bytes)
        << '\n'
        << "canonical_resident_gather_scatter_tib_per_residue="
        << tib(canonical_copy_per_residue)
        << " direct_gather_scatter_tib_per_residue=0.000"
        << " eliminated_tib=" << tib(canonical_copy_per_residue)
        << '\n'
        << "ramstream_v6_pcie_tib_per_residue=" << tib(ramstream_pcie_per_residue)
        << " direct_bulk_host_pcie_tib_per_residue=0.000"
        << " eliminated_host_pcie_tib=" << tib(ramstream_pcie_per_residue)
        << '\n'
        << "cross_rank_strategy=occupancy_preserving-flip+mask-local-rank"
        << " low_masks=" << (1u << LOW_LUT_K)
        << " high_masks=" << (1u << HIGH_LUT_K)
        << " height_stride=" << S
        << '\n';
    return 0;
}
