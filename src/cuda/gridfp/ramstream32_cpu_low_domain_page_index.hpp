#pragma once

#include "ramstream32_cpu_low_sparse_persistent.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

// Height-indexed successor table for HIGH occupancy masks.
// For a threshold t and endpoint height h, next_nonempty(h,t) returns the
// smallest mask >= t that has at least one HIGH code at h, or nmasks if none.
// first_nonempty[h] is next_nonempty(h,0).
//
// Page-boundary analysis repeatedly asks exactly this predecessor/successor
// question. Building the table once changes each boundary signature from a
// scan over up to 2^HIGH_LUT_K masks per factor block to O(1) lookup.

struct CpuLowDomainPageMaskIndex {
    uint32_t nmasks = 0;
    uint32_t stride = 0;
    std::vector<uint32_t> first_nonempty;
    std::vector<uint32_t> next_nonempty;

    uint32_t next(uint32_t h, uint32_t threshold) const {
        if (h >= stride || threshold > nmasks) {
            std::cerr << "cpu LOW page mask index lookup out of range"
                      << " h=" << h
                      << " threshold=" << threshold
                      << " stride=" << stride
                      << " nmasks=" << nmasks << '\n';
            std::exit(165);
        }
        return next_nonempty[size_t(h) * (size_t(nmasks) + 1) + threshold];
    }
};

static CpuLowDomainPageMaskIndex cpu_low_build_domain_page_mask_index() {
    constexpr uint32_t S = FactorTablesHost::STRIDE;
    static_assert(HIGH_LUT_K > 0 && HIGH_LUT_K < 31,
                  "page mask index expects a practical 32-bit HIGH mask");
    const uint32_t nmasks = uint32_t(1) << HIGH_LUT_K;

    CpuLowDomainPageMaskIndex out;
    out.nmasks = nmasks;
    out.stride = S;
    out.first_nonempty.assign(S, nmasks);

    const size_t row = size_t(nmasks) + 1;
    if (row > std::numeric_limits<size_t>::max() / S) {
        std::cerr << "cpu LOW page mask index size overflow\n";
        std::exit(166);
    }
    out.next_nonempty.assign(row * S, nmasks);

    for (uint32_t h = 0; h < S; ++h) {
        uint32_t next = nmasks;
        out.next_nonempty[size_t(h) * row + nmasks] = nmasks;
        for (uint32_t mask = nmasks; mask-- > 0;) {
            size_t ix = size_t(mask) * S + h;
            if (G_FACTOR.high_mask_off[ix + 1] != G_FACTOR.high_mask_off[ix])
                next = mask;
            out.next_nonempty[size_t(h) * row + mask] = next;
        }
        out.first_nonempty[h] = out.next_nonempty[size_t(h) * row];
    }
    return out;
}
