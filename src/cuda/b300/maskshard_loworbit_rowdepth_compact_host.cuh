#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>

#ifndef MASKSHARD_LOW_ORBIT_ROW_DEPTH_COMPACT
#error "compact LOW orbit host hook requires compact task mapping"
#endif

// Reuse the existing per-row setter and per-LOW-group configure call instead of
// adding another call site to the shared batch host.  The row setter records the
// current row for each CUDA device.  configure_low_group(mask) then installs the
// exact compact task prefix once for that mask/row before all LOW positions run.
static std::array<int, 8> G_MS_LOW_ORBIT_COMPACT_ROW{};
static std::array<Code, 8> G_MS_LOW_ORBIT_COMPACT_TOTAL{};

static void maskshard_set_row_depth_loworbit_compact_row(int zero_based_row) {
    maskshard_set_row_depth_exact_io_row(zero_based_row);
    int dev = -1;
    ck(cudaGetDevice(&dev), "LOW orbit compact row get device");
    if (dev < 0 || dev >= int(G_MS_LOW_ORBIT_COMPACT_ROW.size())) {
        std::cerr << "LOW orbit compact row unsupported device " << dev << '\n';
        std::exit(324);
    }
    G_MS_LOW_ORBIT_COMPACT_ROW[size_t(dev)] = zero_based_row;
}

#ifdef maskshard_set_row_depth_fblock_io_row
#undef maskshard_set_row_depth_fblock_io_row
#endif
#define maskshard_set_row_depth_fblock_io_row \
        maskshard_set_row_depth_loworbit_compact_row

static void maskshard_configure_low_group_loworbit_compact(std::uint32_t mask) {
    maskshard_configure_low_group(mask);
    int dev = -1;
    ck(cudaGetDevice(&dev), "LOW orbit compact group get device");
    if (dev < 0 || dev >= int(G_MS_LOW_ORBIT_COMPACT_ROW.size())) {
        std::cerr << "LOW orbit compact group unsupported device " << dev << '\n';
        std::exit(325);
    }
    const int cap = std::min(
        G_MS_LOW_ORBIT_COMPACT_ROW[size_t(dev)] + 1, TARGET_W / 2);
    G_MS_LOW_ORBIT_COMPACT_TOTAL[size_t(dev)] =
        maskshard_configure_loworbit_rowdepth_compact_group(mask, cap);
}

#define maskshard_configure_low_group \
        maskshard_configure_low_group_loworbit_compact

static Code maskshard_loworbit_compact_total_current_device() {
    int dev = -1;
    ck(cudaGetDevice(&dev), "LOW orbit compact total get device");
    if (dev < 0 || dev >= int(G_MS_LOW_ORBIT_COMPACT_TOTAL.size())) {
        std::cerr << "LOW orbit compact total unsupported device " << dev << '\n';
        std::exit(326);
    }
    return G_MS_LOW_ORBIT_COMPACT_TOTAL[size_t(dev)];
}
