#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_set>

// The LOW p==1 closure writes the main authoritative array in place.
// Destination-oriented gather is race/order independent only when no active-
// half closure source can also be an active-half closure destination.  The
// ordinary-only builder already checks this for ordinary edges; fused execution
// also includes CROSS edges, so validate the union here.
//
// Active-half disjointness is a stronger condition than full-state
// disjointness: if (factor block, LOW rank) differs, the complete state differs
// for every HIGH row, including all CROSS preimages.
static void validate_gpu_direct_fused_low_p1_disjoint(
    const StorageLayout& layout,
    const GpuDirectGatherHost& ordinary,
    const GpuDirectCrossGatherHost& cross,
    const GpuDirectFusedHost& fused
) {
    if (LOW_LUT_K <= 0) return;
    const uint32_t pi = uint32_t(LOW_LUT_K - 1); // p == 1
    const uint32_t nt = uint32_t(layout.main_blocks.size());

    auto key = [](uint32_t bid, uint32_t rank) -> uint64_t {
        return (uint64_t(bid) << 32) | rank;
    };
    auto ordinary_block = [](uint32_t x) -> uint32_t {
        return x >> GPU_DIRECT_GATHER_SRC_BLOCK_SHIFT;
    };
    auto ordinary_rank = [](uint32_t x) -> uint32_t {
        return x & GPU_DIRECT_GATHER_RANK_MASK;
    };
    auto cross_block = [](uint32_t x) -> uint32_t {
        return (x >> GPU_DIRECT_CROSS_OP_BLOCK_SHIFT)
            & GPU_DIRECT_CROSS_OP_BLOCK_MASK;
    };
    auto cross_rank = [](uint32_t x) -> uint32_t {
        return x & GPU_DIRECT_CROSS_OP_RANK_MASK;
    };

    std::unordered_set<uint64_t> sources;
    std::unordered_set<uint64_t> destinations;
    size_t ordinary_edges = 0, cross_edges = 0;

    for (uint32_t dbid = 0; dbid < nt; ++dbid) {
        size_t oi = size_t(pi) * fused.low_pitch + dbid;
        uint32_t a = fused.low_off[oi], b = fused.low_off[oi + 1];
        for (uint32_t q = a; q < b; ++q) {
            const GpuDirectFusedDst& rec = fused.low_dst[q];
            destinations.insert(key(dbid, rec.dst_rank));
            uint32_t lc = rec.counts & 0xffffu;
            uint32_t cc = rec.counts >> 16;
            for (uint32_t e = rec.local_begin; e < rec.local_begin + lc; ++e) {
                uint32_t x = ordinary.low_src[e];
                sources.insert(key(ordinary_block(x), ordinary_rank(x)));
                ++ordinary_edges;
            }
            for (uint32_t e = rec.cross_begin; e < rec.cross_begin + cc; ++e) {
                uint32_t x = cross.low_op[e];
                sources.insert(key(cross_block(x), cross_rank(x)));
                ++cross_edges;
            }
        }
    }

    for (uint64_t d : destinations) {
        if (sources.count(d)) {
            uint32_t bid = uint32_t(d >> 32);
            uint32_t rank = uint32_t(d);
            std::cerr << "gpu direct fused LOW p=1 source/destination alias"
                      << " block=" << bid << " rank=" << rank << '\n';
            std::exit(171);
        }
    }

    std::cerr << "gpu_direct_fused_p1_disjoint"
              << " sources=" << sources.size()
              << " destinations=" << destinations.size()
              << " ordinary_edges=" << ordinary_edges
              << " cross_edges=" << cross_edges
              << " overlap=0\n";
}
