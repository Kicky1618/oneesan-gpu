#pragma once

#include "ramstream32_b300_sparse_actions.cuh"
#include "ramstream32_b300_dual_tile_precomputed_w28.cuh"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <vector>

// Conservative, purely structural reachability at the granularity of
//   (StorageBlock, HIGH-owner, LOW-owner).
// A set bit means that at least one exact rank in the ownership tile may be
// nonzero.  Bits are only added (except the exact p=1 blocked reset), so a zero
// bit is a proof that the whole physical tile is zero.  This is deliberately an
// over-approximation: it is safe for skipping NVLink transfers, never for
// pruning arithmetic inside an active tile.
struct B300DualReachStage {
    std::array<uint64_t, 64> main{};
    std::array<uint64_t, 32> block{};
};

struct B300DualReachSchedule {
    int ngpu = 0;
    // State immediately after the HIGH window, before LOW->HIGH shuffle.
    std::vector<B300DualReachStage> l2h;
    // State immediately after the LOW window, before HIGH->LOW main shuffle.
    // The final row has no H2L shuffle and is therefore omitted.
    std::vector<B300DualReachStage> h2l;
};

static inline uint64_t b300_dt_reach_bit(int hi, int lo, int ngpu) {
    return uint64_t(1) << (hi * ngpu + lo);
}
static inline bool b300_dt_reach_get(uint64_t m, int hi, int lo, int ngpu) {
    return (m & b300_dt_reach_bit(hi, lo, ngpu)) != 0;
}
static inline void b300_dt_reach_put(uint64_t& m, int hi, int lo, int ngpu) {
    m |= b300_dt_reach_bit(hi, lo, ngpu);
}

static inline int b300_dt_reach_low_owner(
    const B300DualTileHost& z, const StorageFactorHost& f,
    const StorageBlock& b, uint32_t lr
) {
    return z.low_owner[f.low_all_off[b.hs] + lr];
}
static inline int b300_dt_reach_high_owner(
    const B300DualTileHost& z, const StorageFactorHost& f,
    const StorageBlock& b, uint32_t hr
) {
    return z.high.high_owner[f.high_all_off[b.he] + hr];
}

static inline uint32_t b300_dt_reach_locate_main(
    Code rank, const StorageLayout& l, uint32_t& hr, uint32_t& lr
) {
    for (uint32_t bid = 0; bid < l.main_blocks.size(); ++bid) {
        const auto& b = l.main_blocks[bid];
        if (!b.valid) continue;
        Code n = Code(b.rows) * b.cols;
        if (rank >= b.off && rank < b.off + n) {
            Code q = rank - b.off;
            hr = uint32_t(q / b.cols);
            lr = uint32_t(q % b.cols);
            return bid;
        }
    }
    std::exit(650);
}

static inline void b300_dt_reach_high_edge(
    B300DualReachStage& r, const B300SparseActionsHost& s,
    const StorageFactorHost& f, const StorageLayout& l,
    const B300DualTileHost& z, int p
) {
    const int ngpu = z.ngpu;
    uint32_t pi = uint32_t((TARGET_W - 1) - p);
    for (uint32_t q = s.high_orbit_off[pi]; q < s.high_orbit_off[pi + 1]; ++q) {
        const auto& op = s.high_orbit[q];
        uint32_t sb = b300_sparse_sblock(op), jb = b300_sparse_jblock(op),
                 db = b300_sparse_dblock(op);
        const auto& sx = l.main_blocks[sb];
        const auto& jx = l.main_blocks[jb];
        const auto& dx = l.block_blocks[db];
        int so = b300_dt_reach_high_owner(z, f, sx, b300_sparse_src(op));
        int jo = b300_dt_reach_high_owner(z, f, jx, b300_sparse_jrank(op));
        int bo = b300_dt_reach_high_owner(z, f, dx, b300_sparse_drank(op));
        uint32_t kind = b300_sparse_kind(op);
        for (int lo = 0; lo < ngpu; ++lo) {
            bool sm = b300_dt_reach_get(r.main[sb], so, lo, ngpu);
            bool jm = b300_dt_reach_get(r.main[jb], jo, lo, ngpu);
            bool bd = b300_dt_reach_get(r.block[db], bo, lo, ngpu);
            if (kind == HIGH_ORBIT_NN) {
                if (jm || sm) b300_dt_reach_put(r.main[jb], jo, lo, ngpu);
                if (sm || bd) b300_dt_reach_put(r.main[sb], so, lo, ngpu);
                // Exact dp rank is cleared, but other ranks in the same coarse
                // ownership tile may remain nonzero: never clear a tile here.
            } else {
                if (sm || jm || bd) b300_dt_reach_put(r.main[sb], so, lo, ngpu);
                if (bd || sm) b300_dt_reach_put(r.block[db], bo, lo, ngpu);
            }
        }
    }

    for (uint32_t q = s.high_closure_off[pi]; q < s.high_closure_off[pi + 1]; ++q) {
        uint64_t op = s.high_closure[q];
        uint32_t sb = b300_sparse_closure_sblock(op);
        uint32_t src = b300_sparse_closure_src(op);
        uint32_t d = b300_sparse_closure_desc(op);
        uint32_t kind = b300_host_high_kind(d);
        if (kind != HIGHDESC_BLOCK && kind != HIGHDESC_CROSS) continue;
        uint32_t db = (d >> HIGHDESC_BLOCK_SHIFT) & HIGHDESC_BLOCK_MASK;
        const auto& sx = l.main_blocks[sb];
        const auto& dx = l.block_blocks[db];
        int so = b300_dt_reach_high_owner(z, f, sx, src);
        int bo = b300_dt_reach_high_owner(z, f, dx, d & HIGHDESC_RANK_MASK);
        for (int lo = 0; lo < ngpu; ++lo)
            if (b300_dt_reach_get(r.main[sb], so, lo, ngpu))
                b300_dt_reach_put(r.block[db], bo, lo, ngpu);
        // HIGH CROSS flips LOW topology but preserves LOW occupancy, hence lo.
    }
}

static inline void b300_dt_reach_low_edge(
    B300DualReachStage& r, const B300SparseActionsHost& s,
    const StorageFactorHost& f, const StorageLayout& l,
    const B300DualTileHost& z, int p
) {
    const int ngpu = z.ngpu;
    uint32_t pi = uint32_t(LOW_LUT_K - p);
    for (uint32_t q = s.low_orbit_off[pi]; q < s.low_orbit_off[pi + 1]; ++q) {
        const auto& op = s.low_orbit[q];
        uint32_t sb = b300_sparse_sblock(op), jb = b300_sparse_jblock(op),
                 db = b300_sparse_dblock(op);
        const auto& sx = l.main_blocks[sb];
        const auto& jx = l.main_blocks[jb];
        const auto& dx = l.block_blocks[db];
        int so = b300_dt_reach_low_owner(z, f, sx, b300_sparse_src(op));
        int jo = b300_dt_reach_low_owner(z, f, jx, b300_sparse_jrank(op));
        int bo = b300_dt_reach_low_owner(z, f, dx, b300_sparse_drank(op));
        uint32_t kind = b300_sparse_kind(op);
        for (int hi = 0; hi < ngpu; ++hi) {
            bool sm = b300_dt_reach_get(r.main[sb], hi, so, ngpu);
            bool jm = b300_dt_reach_get(r.main[jb], hi, jo, ngpu);
            bool bd = b300_dt_reach_get(r.block[db], hi, bo, ngpu);
            if (kind == CPU_ORBIT_NN) {
                if (jm || sm) b300_dt_reach_put(r.main[jb], hi, jo, ngpu);
                if (sm || bd) b300_dt_reach_put(r.main[sb], hi, so, ngpu);
            } else if (p == 1) {
                if (sm || jm || bd) b300_dt_reach_put(r.main[sb], hi, so, ngpu);
                if (jm || sm) b300_dt_reach_put(r.main[jb], hi, jo, ngpu);
            } else {
                if (sm || jm || bd) b300_dt_reach_put(r.main[sb], hi, so, ngpu);
                if (bd || sm) b300_dt_reach_put(r.block[db], hi, bo, ngpu);
            }
        }
    }

    for (uint32_t q = s.low_closure_off[pi]; q < s.low_closure_off[pi + 1]; ++q) {
        uint64_t op = s.low_closure[q];
        uint32_t sb = b300_sparse_closure_sblock(op);
        uint32_t src = b300_sparse_closure_src(op);
        uint32_t d = b300_sparse_closure_desc(op);
        uint32_t kind = b300_host_low_kind(d);
        uint32_t db = (d >> LOWDESC_BLOCK_SHIFT) & LOWDESC_BLOCK_MASK;
        const auto& sx = l.main_blocks[sb];
        int so = b300_dt_reach_low_owner(z, f, sx, src);
        int dest_owner = -1;
        bool main_dest = false;
        if (kind == LOWDESC_MAIN) {
            main_dest = true;
            dest_owner = b300_dt_reach_low_owner(z, f, l.main_blocks[db], d & LOWDESC_LR_MASK);
        } else if (kind == LOWDESC_BLOCK) {
            dest_owner = b300_dt_reach_low_owner(z, f, l.block_blocks[db], d & LOWDESC_LR_MASK);
        } else if (kind == LOWDESC_CROSS) {
            main_dest = (p == 1);
            if (main_dest)
                dest_owner = b300_dt_reach_low_owner(z, f, l.main_blocks[db], d & LOWDESC_LR_MASK);
            else
                dest_owner = b300_dt_reach_low_owner(z, f, l.block_blocks[db], d & LOWDESC_LR_MASK);
        } else {
            continue;
        }
        for (int hi = 0; hi < ngpu; ++hi) if (b300_dt_reach_get(r.main[sb], hi, so, ngpu)) {
            if (main_dest) b300_dt_reach_put(r.main[db], hi, dest_owner, ngpu);
            else b300_dt_reach_put(r.block[db], hi, dest_owner, ngpu);
        }
        // LOW CROSS flips HIGH topology but preserves HIGH occupancy, hence hi.
    }

    // Exact row-boundary invariant of the p=1 transition.
    if (p == 1) r.block.fill(0);
}

static inline B300DualReachSchedule build_b300_dual_reach_schedule(
    const B300SparseActionsHost& s, const StorageFactorHost& f,
    const StorageLayout& l, const B300DualTileHost& z
) {
    if (z.ngpu < 1 || z.ngpu > 8) std::exit(651);
    B300DualReachSchedule out;
    out.ngpu = z.ngpu;
    out.l2h.reserve(TARGET_W);
    out.h2l.reserve(TARGET_W - 1);

    B300DualReachStage r;
    uint32_t hr = 0, lr = 0;
    Code rank = storage_rank_main_host(MateID(R) << (2 * (TARGET_W - 1)), f, l);
    uint32_t bid = b300_dt_reach_locate_main(rank, l, hr, lr);
    int hi = b300_dt_reach_high_owner(z, f, l.main_blocks[bid], hr);
    int lo = b300_dt_reach_low_owner(z, f, l.main_blocks[bid], lr);
    b300_dt_reach_put(r.main[bid], hi, lo, z.ngpu);

    for (int row = 0; row < TARGET_W; ++row) {
        for (int p = TARGET_W - 1; p >= LOW_LUT_K + 1; --p)
            b300_dt_reach_high_edge(r, s, f, l, z, p);
        out.l2h.push_back(r);

        for (int p = LOW_LUT_K; p >= 1; --p)
            b300_dt_reach_low_edge(r, s, f, l, z, p);
        if (row + 1 < TARGET_W) out.h2l.push_back(r);
    }
    return out;
}

static inline bool b300_dt_reach_main_active(
    const B300DualReachStage& r, uint32_t bid, int hi, int lo, int ngpu
) {
    return b300_dt_reach_get(r.main[bid], hi, lo, ngpu);
}
static inline bool b300_dt_reach_block_active(
    const B300DualReachStage& r, uint32_t bid, int hi, int lo, int ngpu
) {
    return b300_dt_reach_get(r.block[bid], hi, lo, ngpu);
}
