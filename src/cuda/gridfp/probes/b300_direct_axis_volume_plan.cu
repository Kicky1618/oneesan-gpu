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

static uint32_t av_low_mask(const StorageFactorHost& f, const StorageBlock& b, uint32_t lr) {
    return seg_occ(f.low_all_codes[f.low_all_off[b.hs] + lr], LOW_LUT_K);
}
static uint32_t av_high_mask(const StorageFactorHost& f, const StorageBlock& b, uint32_t hr) {
    return seg_occ(f.high_all_codes[f.high_all_off[b.he] + hr], HIGH_LUT_K);
}

int main() {
    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost f = build_storage_factor_tables(G_FACTOR);
    StorageLayout l = build_storage_layout(f);
    LowDescHost ld = build_low_descriptors(f, l);
    HighDescHost hd = build_high_descriptors(f, l);
    LowOrbitHost lo = build_cpu_low_orbit(f, l, ld);
    HighOrbitHost ho = build_high_orbit(f, l);
    B300SparseActionsHost s = build_b300_sparse_actions(l, ld, lo, hd, ho);

    uint64_t high_orbit_cells = 0, high_closure_cells = 0;
    uint64_t high_orbit_mask_change_cells = 0, high_closure_mask_change_cells = 0;
    for (const auto& op : s.high_orbit) {
        const auto& x = l.main_blocks[b300_sparse_sblock(op)];
        const auto& j = l.main_blocks[b300_sparse_jblock(op)];
        const auto& d = l.block_blocks[b300_sparse_dblock(op)];
        uint32_t sm = av_high_mask(f, x, b300_sparse_src(op));
        uint32_t jm = av_high_mask(f, j, b300_sparse_jrank(op));
        uint32_t dm = av_high_mask(f, d, b300_sparse_drank(op));
        high_orbit_cells += 2ull * x.cols;
        if (sm != jm) high_orbit_mask_change_cells += x.cols;
        if (sm != dm) high_orbit_mask_change_cells += x.cols;
    }
    for (uint64_t op : s.high_closure) {
        const auto& x = l.main_blocks[b300_sparse_closure_sblock(op)];
        uint32_t desc = b300_sparse_closure_desc(op);
        const auto& d = l.block_blocks[highdesc_block(desc)];
        uint32_t sm = av_high_mask(f, x, b300_sparse_closure_src(op));
        uint32_t dm = av_high_mask(f, d, highdesc_rank(desc));
        high_closure_cells += x.cols;
        if (sm != dm) high_closure_mask_change_cells += x.cols;
    }

    uint64_t low_orbit_cells = 0, low_closure_cells = 0;
    uint64_t low_orbit_mask_change_cells = 0, low_closure_mask_change_cells = 0;
    for (const auto& op : s.low_orbit) {
        const auto& x = l.main_blocks[b300_sparse_sblock(op)];
        const auto& j = l.main_blocks[b300_sparse_jblock(op)];
        const auto& d = l.block_blocks[b300_sparse_dblock(op)];
        uint32_t sm = av_low_mask(f, x, b300_sparse_src(op));
        uint32_t jm = av_low_mask(f, j, b300_sparse_jrank(op));
        uint32_t dm = av_low_mask(f, d, b300_sparse_drank(op));
        low_orbit_cells += 2ull * x.rows;
        if (sm != jm) low_orbit_mask_change_cells += x.rows;
        if (sm != dm) low_orbit_mask_change_cells += x.rows;
    }
    for (uint64_t op : s.low_closure) {
        const auto& x = l.main_blocks[b300_sparse_closure_sblock(op)];
        uint32_t desc = b300_sparse_closure_desc(op);
        uint32_t kind = lowdesc_kind(desc);
        uint32_t sm = av_low_mask(f, x, b300_sparse_closure_src(op));
        uint32_t dm = sm;
        if (kind == LOWDESC_MAIN) {
            const auto& d = l.main_blocks[lowdesc_block(desc)];
            dm = av_low_mask(f, d, lowdesc_lr(desc));
        } else {
            const auto& d = l.block_blocks[lowdesc_block(desc)];
            dm = av_low_mask(f, d, lowdesc_lr(desc));
        }
        low_closure_cells += x.rows;
        if (sm != dm) low_closure_mask_change_cells += x.rows;
    }

    auto frac = [](uint64_t a, uint64_t b) { return b ? double(a) / double(b) : 0.0; };
    auto tib_per_residue = [](uint64_t cells) {
        return double((long double)cells * sizeof(Count) * TARGET_W / (1ull << 40));
    };
    std::cout << std::fixed << std::setprecision(6)
        << "b300-direct-axis-volume W=" << TARGET_W << '\n'
        << "high_mask_sharding_low_window_p2p_cells=0"
        << " residual_high_orbit_target_cells=" << high_orbit_cells
        << " residual_high_closure_target_cells=" << high_closure_cells
        << " high_orbit_mask_change_fraction=" << frac(high_orbit_mask_change_cells, high_orbit_cells)
        << " high_closure_mask_change_fraction=" << frac(high_closure_mask_change_cells, high_closure_cells)
        << " high_all_target_payload_tib_per_residue=" << tib_per_residue(high_orbit_cells + high_closure_cells)
        << '\n'
        << "low_mask_sharding_high_window_p2p_cells=0"
        << " residual_low_orbit_target_cells=" << low_orbit_cells
        << " residual_low_closure_target_cells=" << low_closure_cells
        << " low_orbit_mask_change_fraction=" << frac(low_orbit_mask_change_cells, low_orbit_cells)
        << " low_closure_mask_change_fraction=" << frac(low_closure_mask_change_cells, low_closure_cells)
        << " low_all_target_payload_tib_per_residue=" << tib_per_residue(low_orbit_cells + low_closure_cells)
        << '\n';
    return 0;
}
