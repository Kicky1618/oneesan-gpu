#pragma once

#include "gridfp_reduced_production_turn_owner_device.cuh"

namespace oneesan::gridfp::reducedprod {

// High-edge counterpart of the low Q1 compression label codec.  The physical
// high window is [W-L,W-1] and the boundary-compression missing bit is W-2.
__device__ __forceinline__ MateID turn_compress_high_label_unrank_planned_device(
    int W,
    int K,
    const OwnerComponentPlanDevice& plan,
    Rank64 rank
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = W - L;
    const int hi = W - 1;
    const int missing = W - 2;

    int outer_ones = -1;
    Rank64 local = 0;
    for (int r = 0; r <= O; ++r) {
        if (rank < plan.prefix[r + 1]) {
            outer_ones = r;
            local = rank - plan.prefix[r];
            break;
        }
    }
    if (outer_ones < 0) return 0;
    const Rank64 group = plan.component_group[outer_ones];
    if (!group) return 0;
    const Rank64 outer_sr = plan.sr_begin[outer_ones] + local / group;
    Rank64 within = local % group;
    const std::uint32_t outer = support_unrank_mask_device(O, outer_ones, outer_sr);

    int local_ones = -1;
    Rank64 local_sr = 0, primitive_rank = 0;
    for (int l = 0; l <= L - 1; ++l) {
        const int occupied = outer_ones + l;
        if (!(occupied & 1)) continue;
        const Rank64 pc = RP_PRIMITIVE[occupied][1];
        const Rank64 n = choose_device(L - 1, l) * pc;
        if (within < n) {
            local_ones = l;
            local_sr = within / pc;
            primitive_rank = within % pc;
            break;
        }
        within -= n;
    }
    if (local_ones < 0) return 0;

    const std::uint32_t local_support = support_unrank_mask_device(
        L - 1, local_ones, local_sr);
    std::uint32_t full = 0;
    owner_expand_outer_support_device(outer, W, lo, hi, full);
    int cp = 0;
    for (int bit = lo; bit <= hi; ++bit) {
        if (bit == missing) continue;
        if ((local_support >> cp) & 1u) full |= std::uint32_t(1) << bit;
        ++cp;
    }
    const int occupied = outer_ones + local_ones;
    const std::uint32_t label_support = owner_label_lr_support_device(full, W, missing);
    return materialize_primitive_device(label_support, W - 1, occupied, primitive_rank);
}

__device__ __forceinline__ DeviceKey turn_compress_high_seed(MateID label, int W) {
    if (mget(label, W - 2) != N) return DeviceKey{label, 1};
    return DeviceKey{blocked_exclude_reverse(label, W, W - 1), 0};
}

__device__ __forceinline__ DeviceKey turn_expand_high_seed(MateID label, int W) {
    return DeviceKey{blocked_exclude(label, W - 1), 0};
}

__device__ __forceinline__ bool turn_small_compress_high_step(
    DeviceKey src, int W, SmallTerms& z
) {
    SmallTerms tmp;
    const DeviceKey mirrored = mirror_key_device(src, W);
    if (!turn_small_compress_step(mirrored, W, tmp)) return false;
    for (int i = 0; i < tmp.n; ++i) {
        if (!small_add(z, mirror_key_device(tmp.v[i].key, W), tmp.v[i].coef))
            return false;
    }
    return true;
}

__device__ __forceinline__ bool turn_small_expand_high_step(
    DeviceKey src, int W, SmallTerms& z
) {
    if (src.blocked) return false;
    return small_step(src, W, W - 1, false, z);
}

struct TurnHighMirrorSink {
    DeviceKeySetSink sink{};
    int W = 0;
    __device__ __forceinline__ bool emit(DeviceKey k) {
        return sink.emit(mirror_key_device(k, W));
    }
};

struct TurnHighMainOnlySink {
    DeviceKeySetSink sink{};
    __device__ __forceinline__ bool emit(DeviceKey k) {
        if (k.blocked) return true;
        return sink.emit(k);
    }
};

__device__ __forceinline__ bool turn_high_discover_inverse_to_set(
    DeviceKey dest,
    int W,
    bool expand,
    DeviceKey* source_set,
    int& source_count,
    int capacity
) {
    DeviceKeySetSink base{source_set, &source_count, capacity};
    if (!expand) {
        TurnHighMirrorSink sink{base, W};
        return turn_discover_compress_inverse(
            mirror_key_device(dest, W), W, sink);
    }
    TurnHighMainOnlySink sink{base};
    return discover_inverse_reduced_forward(dest, W, W - 1, sink);
}

} // namespace oneesan::gridfp::reducedprod
