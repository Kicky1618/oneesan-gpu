#pragma once

#include "two_cell_fusion_sectors.hpp"

#if defined(__CUDACC__)
#define ONEESAN_TC_FSLICE_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_TC_FSLICE_HD inline
#endif

namespace oneesan::twocell {

ONEESAN_TC_FSLICE_HD Rank primitive_slice_begin(
    Rank count,
    int owner,
    int owners
) {
    return (count * Rank(owner)) / Rank(owners);
}

ONEESAN_TC_FSLICE_HD Rank primitive_slice_end(
    Rank count,
    int owner,
    int owners
) {
    return (count * Rank(owner + 1)) / Rank(owners);
}

ONEESAN_TC_FSLICE_HD Rank primitive_slice_count(
    Rank count,
    int owner,
    int owners
) {
    return primitive_slice_end(count, owner, owners) -
           primitive_slice_begin(count, owner, owners);
}

ONEESAN_TC_FSLICE_HD int primitive_slice_owner(
    Rank primitive,
    Rank count,
    int owners
) {
    if (!count || primitive >= count || owners <= 0) return -1;
    int owner = static_cast<int>((primitive * Rank(owners)) / count);
    if (owner >= owners) owner = owners - 1;
    return owner;
}

// Return the fixed sector index of a state inside a fused block.  The sector is
// independent of active position after C support is rebased to Q_start.
ONEESAN_TC_FSLICE_HD int fusion_sector_index_at(
    PackedKey key,
    int start,
    int steps,
    int active
) {
    if (key.type == 0) {
        return static_cast<int>(
            (key.support >> start) & low_mask(steps + 2));
    }
    const std::uint32_t support = stationary_c_rebase_support(
        key.support, active, start);
    const int ccode = static_cast<int>(
        (support >> (start + 1)) & low_mask(steps));
    return (1 << (steps + 2)) + ccode;
}

ONEESAN_TC_FSLICE_HD int fusion_sector_index(
    PackedKey key,
    int start,
    int steps
) {
    return fusion_sector_index_at(key, start, steps, start);
}

// Offset of one owner's slice of sector q in a CTA-local concatenated DSM
// partition. This uses only sector counts, so it can be precomputed once per
// outer-support block and reused in both phases.
template <class SectorArray>
ONEESAN_TC_FSLICE_HD Rank fusion_slice_local_base(
    const SectorArray& sectors,
    int sector,
    int owner,
    int owners
) {
    Rank base = 0;
    for (int q = 0; q < sector; ++q) {
        if (!sectors[q].valid) continue;
        base += primitive_slice_count(sectors[q].count, owner, owners);
    }
    return base;
}

} // namespace oneesan::twocell

#undef ONEESAN_TC_FSLICE_HD
