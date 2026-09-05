#pragma once

#include "gridfp_reduced_production_turn_device.cuh"
#include "gridfp_reduced_production_owner_component_plan_device.cuh"
#include "gridfp_reduced_production_discovery_device.cuh"

namespace oneesan::gridfp::reducedprod {

__device__ __forceinline__ Rank64 turn_compress_component_group_size_device(
    int L, int outer_ones
) {
    Rank64 total = 0;
    for (int local = 0; local <= L - 1; ++local) {
        const int occupied = outer_ones + local;
        if (!(occupied & 1)) continue;
        total += choose_device(L - 1, local) * RP_PRIMITIVE[occupied][1];
    }
    return total;
}

// Owner-local Q1-forward compression label.  The full seed support always has
// physical bit 1 missing; all other L-1 local support positions are free.
__device__ __forceinline__ MateID turn_compress_label_unrank_planned_device(
    int W,
    int K,
    const OwnerComponentPlanDevice& plan,
    Rank64 rank
) {
    const int L = K + 2;
    const int O = W - L;
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

    constexpr int missing = 1;
    const std::uint32_t local_support = support_unrank_mask_device(
        L - 1, local_ones, local_sr);
    std::uint32_t full = 0;
    owner_expand_outer_support_device(outer, W, 0, L - 1, full);
    int cp = 0;
    for (int bit = 0; bit < L; ++bit) {
        if (bit == missing) continue;
        if ((local_support >> cp) & 1u) full |= std::uint32_t(1) << bit;
        ++cp;
    }
    const int occupied = outer_ones + local_ones;
    const std::uint32_t label_support = owner_label_lr_support_device(full, W, missing);
    return materialize_primitive_device(label_support, W - 1, occupied, primitive_rank);
}

__device__ __forceinline__ bool turn_small_compress_step(
    DeviceKey src, int W, SmallTerms& z
) {
    if (!src.blocked) {
        if (!small_add(z, src, 1)) return false;
        const IncludeResult x = include_horizontal(src.mate, W, 1);
        if (!x.valid) return true;
        if (x.blocked) return false;
        return small_add(z, DeviceKey{x.mate, 0}, 1);
    }
    return small_add(z, DeviceKey{blocked_exclude(src.mate, 1), 0}, 1);
}

__device__ __forceinline__ bool turn_small_expand_step(
    DeviceKey src, int W, SmallTerms& z
) {
    if (src.blocked) return false;
    return small_step(src, W, 1, true, z);
}

template<class Sink>
__device__ __forceinline__ bool turn_discover_try_compress_main(
    MateID x, MateID dest, int W, Sink& sink
) {
    if (!valid_mate_device(x, W)) return true;
    const IncludeResult z = include_horizontal(x, W, 1);
    if (z.valid && !z.blocked && z.mate == dest)
        return sink.emit(DeviceKey{x, 0});
    return true;
}

template<class Sink>
__device__ __forceinline__ bool turn_discover_compress_inverse(
    DeviceKey dest, int W, Sink& sink
) {
    if (dest.blocked) return false;
    const MateID d = dest.mate;
    if (!sink.emit(DeviceKey{d, 0})) return false;

    const MateValuePair pair = mpair(d, 1);
    if (pair == LR && !turn_discover_try_compress_main(msetpair(d, 1, NN), d, W, sink)) return false;
    if (pair == RN && !turn_discover_try_compress_main(msetpair(d, 1, NR), d, W, sink)) return false;
    if (pair == LN && !turn_discover_try_compress_main(msetpair(d, 1, NL), d, W, sink)) return false;
    if (pair == NR && !turn_discover_try_compress_main(msetpair(d, 1, RN), d, W, sink)) return false;
    if (pair == NL && !turn_discover_try_compress_main(msetpair(d, 1, LN), d, W, sink)) return false;

    if (pair == NN) {
        if (!turn_discover_try_compress_main(msetpair(d, 1, RL), d, W, sink)) return false;
        int bal = 0;
        for (int q = 2; q < W; ++q) {
            const MateValue v = mget(d, q);
            if (bal == 0 && v == R) {
                MateID x = msetpair(d, 1, RR);
                x = mset(x, q, L);
                if (!turn_discover_try_compress_main(x, d, W, sink)) return false;
            }
            if (v == R) ++bal;
            else if (v == L) --bal;
            if (bal < 0) break;
        }
    }

    if (mget(d, 1) == N && is_endpoint(mget(d, 0))) {
        const MateID b = mshrink(d, 1);
        if (valid_mate_device(b, W - 1) && mget(b, 0) != N && blocked_exclude(b, 1) == d) {
            if (!sink.emit(DeviceKey{b, 1})) return false;
        }
    }
    return true;
}

struct TurnMainMirroredSink {
    DeviceKeySetSink sink{};
    int W = 0;
    __device__ __forceinline__ bool emit(DeviceKey k) {
        const DeviceKey z = mirror_key_device(k, W);
        if (z.blocked) return true;
        return sink.emit(z);
    }
};

__device__ __forceinline__ bool turn_discover_inverse_to_set(
    DeviceKey dest,
    int W,
    bool expand,
    DeviceKey* source_set,
    int& source_count,
    int capacity
) {
    DeviceKeySetSink base{source_set, &source_count, capacity};
    if (!expand) return turn_discover_compress_inverse(dest, W, base);

    // reverse p=1 is mirror-conjugate to forward p=W-1; retain main sources only.
    TurnMainMirroredSink sink{base, W};
    return discover_inverse_reduced_forward(
        mirror_key_device(dest, W), W, W - 1, sink);
}

} // namespace oneesan::gridfp::reducedprod
