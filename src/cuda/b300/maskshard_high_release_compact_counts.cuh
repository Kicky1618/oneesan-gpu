#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#ifndef MASKSHARD_HIGH_RELEASE_COMPACT_COUNTS
#error "HIGH compact-count release header requires MASKSHARD_HIGH_RELEASE_COMPACT_COUNTS"
#endif
#ifndef MASKSHARD_HIGH_ROW_BATCH_ASYNC
#error "HIGH compact-count release requires row-batch plan caching"
#endif
#ifndef MASKSHARD_HIGH_ROW_PLAN_CLASS_CACHE
#error "HIGH compact-count release is only enabled after class plans are materialized"
#endif
#ifndef MASKSHARD_ROW_DEPTH_ORBIT_COMPACT
#error "HIGH compact-count release requires compact row-depth metadata"
#endif
#ifndef MASKSHARD_HIGH_CLOSURE_ROW_DEPTH_COMPACT_LAUNCH
#error "HIGH compact-count release must run after exact HIGH closure launch setup"
#endif

// v0.69: the cumulative compact counts are needed by two setup consumers:
//   1. v0.67 materializes every unsaturated HIGH orbit plan during the layout
//      report hook;
//   2. v0.23 builds the exact HIGH closure launch-count table later, after
//      high_jobs have been constructed.
// Only after (2) completes are low_count/high_count dead. Hook that later setup
// call rather than report_high_mask_shard_layout(); releasing at the earlier
// report hook would invalidate the v0.23 table build.
static void maskshard_release_high_compact_count_vectors() {
    auto& compact = maskshard_row_depth_orbit_compact_cache();
    const std::size_t low_bytes =
        compact.low_count.capacity() * sizeof(std::uint16_t);
    const std::size_t high_bytes =
        compact.high_count.capacity() * sizeof(std::uint32_t);
    const std::size_t released = low_bytes + high_bytes;

    std::vector<std::uint16_t>().swap(compact.low_count);
    std::vector<std::uint32_t>().swap(compact.high_count);

    std::cerr << "HIGH compact count tables released host_mib="
              << double(released) / double(1ULL << 20)
              << " after=highclosure-launch-cache"
              << " low_capacity=0 high_capacity=0\n";
}

static void maskshard_prepare_highclosure_rowdepth_compact_release_counts(
    const HighDescHost& high_desc, int ngpu
) {
    // The v0.23 header has already defined this concrete function before its
    // macro redirects the shared setup call. Invoke it directly, then release.
    maskshard_prepare_highclosure_rowdepth_compact_launch(high_desc, ngpu);
    maskshard_release_high_compact_count_vectors();
}

#ifdef maskshard_prepare_highclosure_rowdepth_compact
#undef maskshard_prepare_highclosure_rowdepth_compact
#endif
#define maskshard_prepare_highclosure_rowdepth_compact \
        maskshard_prepare_highclosure_rowdepth_compact_release_counts
