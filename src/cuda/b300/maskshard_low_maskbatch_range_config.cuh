#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "maskshard_low_maskbatch_config.cuh"
#include "maskshard_low_maskbatch_range_plan.hpp"

struct MaskShardLowBatchRangeDeviceDesc {
    std::uint32_t cta_end = 0;
    std::uint16_t mask = 0;
    std::uint16_t local = 0;
};
static_assert(sizeof(MaskShardLowBatchRangeDeviceDesc) == 8,
              "LOW mask-batch range device descriptor ABI changed");

struct MaskShardLowBatchBoundRanges {
    std::vector<MaskShardLowBatchRangeDeviceDesc> groups;
    std::uint32_t ctas = 0;
};

static MaskShardLowBatchBoundRanges maskshard_bind_low_batch_ranges(
    const MaskShardLowMaskBatchDeviceTables& cfg,
    const std::vector<MaskShardLowBatchRangeHostDesc>& src
) {
    MaskShardLowBatchBoundRanges out;
    out.groups.reserve(src.size());
    std::uint64_t end = 0;
    for (const MaskShardLowBatchRangeHostDesc& x : src) {
        if (x.mask >= cfg.local_of_mask.size()
            || cfg.local_of_mask[x.mask] == std::uint16_t(0xffffu)
            || x.replicas == 0) {
            std::cerr << "LOW mask-batch range owner/replica mismatch dev="
                      << cfg.dev << " mask=" << x.mask
                      << " replicas=" << x.replicas << '\n';
            std::exit(355);
        }
        end += x.replicas;
        if (end > std::numeric_limits<std::uint32_t>::max()) {
            std::cerr << "LOW mask-batch range CTA prefix overflow dev="
                      << cfg.dev << " ctas=" << end << '\n';
            std::exit(356);
        }
        out.groups.push_back({std::uint32_t(end), x.mask,
                              cfg.local_of_mask[x.mask]});
    }
    out.ctas = std::uint32_t(end);
    return out;
}
