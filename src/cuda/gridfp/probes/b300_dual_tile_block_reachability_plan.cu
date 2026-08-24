#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"
#include "../ramstream32_cpu_low_inplace.hpp"
#include "../ramstream32_b300_sparse_actions.cuh"
#include "../ramstream32_b300_dual_tile_precomputed_w28.cuh"

namespace {

struct Reach {
    std::array<uint8_t, 64> main{};
    std::array<uint8_t, 32> block{};
};

uint32_t locate_main_block(Code rank, const StorageLayout& l) {
    for (uint32_t b = 0; b < l.main_blocks.size(); ++b) {
        const auto& x = l.main_blocks[b];
        if (!x.valid) continue;
        Code n = Code(x.rows) * x.cols;
        if (rank >= x.off && rank < x.off + n) return b;
    }
    std::cerr << "initial main rank not found\n";
    std::exit(630);
}

void high_edge_reach(Reach& r, const B300SparseActionsHost& s, int p) {
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    for (uint32_t q = s.high_orbit_off[pi]; q < s.high_orbit_off[pi + 1]; ++q) {
        const auto& op = s.high_orbit[q];
        uint32_t sb = b300_sparse_sblock(op), jb = b300_sparse_jblock(op), db = b300_sparse_dblock(op);
        uint32_t kind = b300_sparse_kind(op);
        bool sm = r.main[sb], jm = r.main[jb], bd = r.block[db];
        if (kind == HIGH_ORBIT_NN) {
            // jp' = jp + ip, ip' = ip + dp.  Keeping blocked reachability is
            // conservative because only this exact dropped rank is cleared.
            if (jm || sm) r.main[jb] = 1;
            if (sm || bd) r.main[sb] = 1;
        } else {
            // ip' = ip + jp + dp, dp' = ip, jp is retained.
            if (sm || jm || bd) r.main[sb] = 1;
            if (bd || sm) r.block[db] = 1;
        }
    }
    for (uint32_t q = s.high_closure_off[pi]; q < s.high_closure_off[pi + 1]; ++q) {
        uint64_t op = s.high_closure[q];
        uint32_t sb = b300_sparse_closure_sblock(op);
        if (!r.main[sb]) continue;
        uint32_t desc = b300_sparse_closure_desc(op);
        uint32_t kind = b300_host_high_kind(desc);
        if (kind == HIGHDESC_BLOCK || kind == HIGHDESC_CROSS)
            r.block[(desc >> HIGHDESC_BLOCK_SHIFT) & HIGHDESC_BLOCK_MASK] = 1;
    }
}

void low_edge_reach(Reach& r, const B300SparseActionsHost& s, int p) {
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    for (uint32_t q = s.low_orbit_off[pi]; q < s.low_orbit_off[pi + 1]; ++q) {
        const auto& op = s.low_orbit[q];
        uint32_t sb = b300_sparse_sblock(op), jb = b300_sparse_jblock(op), db = b300_sparse_dblock(op);
        uint32_t kind = b300_sparse_kind(op);
        bool sm = r.main[sb], jm = r.main[jb], bd = r.block[db];
        if (kind == CPU_ORBIT_NN) {
            if (jm || sm) r.main[jb] = 1;
            if (sm || bd) r.main[sb] = 1;
        } else if (p == 1) {
            // p=1 special: ip'=ip+jp+dp, jp'=ip+jp, dp'=0.
            if (sm || jm || bd) r.main[sb] = 1;
            if (jm || sm) r.main[jb] = 1;
        } else {
            if (sm || jm || bd) r.main[sb] = 1;
            if (bd || sm) r.block[db] = 1;
        }
    }
    for (uint32_t q = s.low_closure_off[pi]; q < s.low_closure_off[pi + 1]; ++q) {
        uint64_t op = s.low_closure[q];
        uint32_t sb = b300_sparse_closure_sblock(op);
        if (!r.main[sb]) continue;
        uint32_t desc = b300_sparse_closure_desc(op);
        uint32_t kind = b300_host_low_kind(desc);
        uint32_t db = (desc >> LOWDESC_BLOCK_SHIFT) & LOWDESC_BLOCK_MASK;
        if (kind == LOWDESC_MAIN) r.main[db] = 1;
        else if (kind == LOWDESC_BLOCK) r.block[db] = 1;
        else if (kind == LOWDESC_CROSS) {
            if (p == 1) r.main[db] = 1;
            else r.block[db] = 1;
        }
    }
    // The proven row-boundary invariant: after LOW p=1 the entire blocked
    // vector is zero, not merely the ranks represented by the coarse flags.
    if (p == 1) r.block.fill(0);
}

long double active_offgpu_bytes(
    const B300DualTileHost& z, const StorageLayout& l,
    const Reach& r, bool include_main, bool include_block
) {
    long double bytes = 0;
    if (include_main) {
        for (uint32_t bid = 0; bid < l.main_blocks.size(); ++bid) if (r.main[bid]) {
            const auto& b = l.main_blocks[bid];
            if (!b.valid) continue;
            for (int hi = 0; hi < z.ngpu; ++hi) for (int lo = 0; lo < z.ngpu; ++lo)
                if (hi != lo)
                    bytes += (long double)z.high_count[hi][b.he] * z.low_count[lo][b.hs] * sizeof(Count);
        }
    }
    if (include_block) {
        for (uint32_t bid = 0; bid < l.block_blocks.size(); ++bid) if (r.block[bid]) {
            const auto& b = l.block_blocks[bid];
            if (!b.valid) continue;
            for (int hi = 0; hi < z.ngpu; ++hi) for (int lo = 0; lo < z.ngpu; ++lo)
                if (hi != lo)
                    bytes += (long double)z.high_count[hi][b.he] * z.low_count[lo][b.hs] * sizeof(Count);
        }
    }
    return bytes;
}

int count_main(const Reach& r) { return int(std::count(r.main.begin(), r.main.end(), uint8_t(1))); }
int count_block(const Reach& r) { return int(std::count(r.block.begin(), r.block.end(), uint8_t(1))); }
static double gib(long double x) { return double(x / (1ull << 30)); }
static double tib(long double x) { return double(x / (1ull << 40)); }

} // namespace

int main() {
    constexpr int NG = 8;
    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost f = build_storage_factor_tables(G_FACTOR);
    StorageLayout l = build_storage_layout(f);
    LowDescHost ld = build_low_descriptors(f, l);
    HighDescHost hd = build_high_descriptors(f, l);
    LowOrbitHost lo = build_cpu_low_orbit(f, l, ld);
    HighOrbitHost ho = build_high_orbit(f, l);
    B300SparseActionsHost s = build_b300_sparse_actions(l, ld, lo, hd, ho);
    B300DualTileHost z = build_b300_dual_tile_layout_w28_precomputed(f, l, NG);

    Reach r;
    Code initial = storage_rank_main_host(MateID(R) << (2 * (TARGET_W - 1)), f, l);
    uint32_t ib = locate_main_block(initial, l);
    r.main[ib] = 1;

    long double pruned_total = 0, full_total = 0;
    long double full_l2h = 0, full_h2l = 0;
    Reach all;
    for (uint32_t b = 0; b < l.main_blocks.size(); ++b) if (l.main_blocks[b].valid) all.main[b] = 1;
    for (uint32_t b = 0; b < l.block_blocks.size(); ++b) if (l.block_blocks[b].valid) all.block[b] = 1;
    full_l2h = active_offgpu_bytes(z, l, all, true, true);
    full_h2l = active_offgpu_bytes(z, l, all, true, false);

    std::cout << std::fixed << std::setprecision(6)
              << "b300-dual-tile-block-reachability W=" << TARGET_W
              << " initial_block=" << ib
              << " full_l2h_gib=" << gib(full_l2h)
              << " full_h2l_gib=" << gib(full_h2l) << '\n';

    for (int row = 0; row < TARGET_W; ++row) {
        for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p) high_edge_reach(r, s, p);
        long double l2h = active_offgpu_bytes(z, l, r, true, true);
        pruned_total += l2h;
        full_total += full_l2h;
        std::cout << "reach_row=" << row
                  << " stage=L2H main_blocks=" << count_main(r)
                  << " block_blocks=" << count_block(r)
                  << " active_gib=" << gib(l2h)
                  << " full_fraction=" << double(l2h / full_l2h) << '\n';

        for (int p = LOW_LUT_K; p >= 1; --p) low_edge_reach(r, s, p);
        if (row + 1 < TARGET_W) {
            long double h2l = active_offgpu_bytes(z, l, r, true, false);
            pruned_total += h2l;
            full_total += full_h2l;
            std::cout << "reach_row=" << row
                      << " stage=H2L main_blocks=" << count_main(r)
                      << " block_blocks=" << count_block(r)
                      << " active_gib=" << gib(h2l)
                      << " full_fraction=" << double(h2l / full_h2l) << '\n';
        }
    }

    std::cout << "reachability_summary"
              << " full_tib_per_residue=" << tib(full_total)
              << " conservative_pruned_tib_per_residue=" << tib(pruned_total)
              << " reduction=" << (full_total ? 1.0 - double(pruned_total / full_total) : 0.0)
              << " final_main_blocks=" << count_main(r)
              << " final_block_blocks=" << count_block(r) << '\n';
    return 0;
}
