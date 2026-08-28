#pragma once

#include "gridfp_reduced_production_runtime_subwarp.cuh"

namespace oneesan::gridfp::reducedprod {

__device__ __forceinline__ Rank64 runtime_turn_compress_group_size_device(
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

__device__ __forceinline__ MateID runtime_turn_compress_label_unrank_device(
    int W,
    int K,
    bool high,
    const OwnerComponentPlanDevice& plan,
    Rank64 rank
) {
    const int L = K + 2;
    const int O = W - L;
    const int lo = high ? W - L : 0;
    const int hi = lo + L - 1;
    const int missing = high ? W - 2 : 1;

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
    Rank64 local_sr = 0;
    Rank64 primitive_rank = 0;
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
        if ((local_support >> cp) & 1u)
            full |= std::uint32_t(1) << bit;
        ++cp;
    }
    const int occupied = outer_ones + local_ones;
    const std::uint32_t label_support = owner_label_lr_support_device(
        full, W, missing);
    return materialize_primitive_device(
        label_support, W - 1, occupied, primitive_rank);
}

__device__ __forceinline__ DeviceKey runtime_turn_compress_seed(
    MateID label, int W, bool high
) {
    if (!high) {
        if (mget(label, 0) != N) return DeviceKey{label, 1};
        return DeviceKey{blocked_exclude(label, 1), 0};
    }
    if (mget(label, W - 2) != N) return DeviceKey{label, 1};
    return DeviceKey{blocked_exclude_reverse(label, W, W - 1), 0};
}

__device__ __forceinline__ DeviceKey runtime_turn_expand_seed(
    MateID label, int W, bool high
) {
    return high
        ? DeviceKey{blocked_exclude(label, W - 1), 0}
        : DeviceKey{blocked_exclude_reverse(label, W, 1), 0};
}

__device__ __forceinline__ bool runtime_turn_compress_low_step(
    DeviceKey src, int W, RuntimeSmallTerms& z
) {
    if (!src.blocked) {
        if (!runtime_small_add(z, src, 1)) return false;
        const IncludeResult x = include_horizontal(src.mate, W, 1);
        if (!x.valid) return true;
        if (x.blocked) return false;
        return runtime_small_add(z, DeviceKey{x.mate, 0}, 1);
    }
    return runtime_small_add(
        z, DeviceKey{blocked_exclude(src.mate, 1), 0}, 1);
}

__device__ __forceinline__ bool runtime_turn_step(
    DeviceKey src, int W, bool high, bool expand, RuntimeSmallTerms& z
) {
    if (!high) {
        if (expand) {
            if (src.blocked) return false;
            return runtime_small_step(src, W, 1, true, z);
        }
        return runtime_turn_compress_low_step(src, W, z);
    }

    if (expand) {
        if (src.blocked) return false;
        return runtime_small_step(src, W, W - 1, false, z);
    }
    RuntimeSmallTerms tmp;
    if (!runtime_turn_compress_low_step(mirror_key_device(src, W), W, tmp))
        return false;
    for (int i = 0; i < tmp.n; ++i) {
        if (!runtime_small_add(
                z, mirror_key_device(tmp.v[i].key, W), tmp.v[i].coef))
            return false;
    }
    return true;
}

template<class Sink>
__device__ __forceinline__ bool runtime_turn_try_compress_main(
    MateID x, MateID dest, int W, Sink& sink
) {
    if (!valid_mate_device(x, W)) return true;
    const IncludeResult z = include_horizontal(x, W, 1);
    if (z.valid && !z.blocked && z.mate == dest)
        return sink.emit(DeviceKey{x, 0});
    return true;
}

template<class Sink>
__device__ __forceinline__ bool runtime_turn_discover_compress_low(
    DeviceKey dest, int W, Sink& sink
) {
    if (dest.blocked) return false;
    const MateID d = dest.mate;
    if (!sink.emit(DeviceKey{d, 0})) return false;

    const MateValuePair pair = mpair(d, 1);
    if (pair == LR && !runtime_turn_try_compress_main(msetpair(d, 1, NN), d, W, sink)) return false;
    if (pair == RN && !runtime_turn_try_compress_main(msetpair(d, 1, NR), d, W, sink)) return false;
    if (pair == LN && !runtime_turn_try_compress_main(msetpair(d, 1, NL), d, W, sink)) return false;
    if (pair == NR && !runtime_turn_try_compress_main(msetpair(d, 1, RN), d, W, sink)) return false;
    if (pair == NL && !runtime_turn_try_compress_main(msetpair(d, 1, LN), d, W, sink)) return false;

    if (pair == NN) {
        if (!runtime_turn_try_compress_main(msetpair(d, 1, RL), d, W, sink))
            return false;
        int bal = 0;
        for (int q = 2; q < W; ++q) {
            const MateValue v = mget(d, q);
            if (bal == 0 && v == R) {
                MateID x = msetpair(d, 1, RR);
                x = mset(x, q, L);
                if (!runtime_turn_try_compress_main(x, d, W, sink))
                    return false;
            }
            if (v == R) ++bal;
            else if (v == L) --bal;
            if (bal < 0) break;
        }
    }

    if (mget(d, 1) == N && is_endpoint(mget(d, 0))) {
        const MateID b = mshrink(d, 1);
        if (valid_mate_device(b, W - 1) && mget(b, 0) != N &&
            blocked_exclude(b, 1) == d) {
            if (!sink.emit(DeviceKey{b, 1})) return false;
        }
    }
    return true;
}

struct RuntimeMirrorSink {
    DeviceKeySetSink sink{};
    int W = 0;
    bool main_only = false;

    __device__ __forceinline__ bool emit(DeviceKey k) {
        const DeviceKey z = mirror_key_device(k, W);
        if (main_only && z.blocked) return true;
        return sink.emit(z);
    }
};

struct RuntimeMainOnlySink {
    DeviceKeySetSink sink{};
    __device__ __forceinline__ bool emit(DeviceKey k) {
        if (k.blocked) return true;
        return sink.emit(k);
    }
};

__device__ __forceinline__ bool runtime_turn_discover_inverse(
    DeviceKey dest,
    int W,
    bool high,
    bool expand,
    DeviceKey* source_set,
    int& source_count,
    int capacity
) {
    DeviceKeySetSink base{source_set, &source_count, capacity};
    if (!high && !expand)
        return runtime_turn_discover_compress_low(dest, W, base);

    if (high && !expand) {
        RuntimeMirrorSink sink{base, W, false};
        return runtime_turn_discover_compress_low(
            mirror_key_device(dest, W), W, sink);
    }

    if (!high && expand) {
        RuntimeMirrorSink sink{base, W, true};
        return discover_inverse_reduced_forward(
            mirror_key_device(dest, W), W, W - 1, sink);
    }

    RuntimeMainOnlySink sink{base};
    return discover_inverse_reduced_forward(dest, W, W - 1, sink);
}

// Counter-free turn kernel for both physical row edges and both phases.
// Compression enumerates M_{W-1} components; expansion enumerates the ordinary
// reduced component set M_{W-1}-M_{W-3}.  Four independent components share a
// warp using 8-lane subgroups.
__global__ void owner_turn_runtime_subwarp_kernel(
    std::uint32_t* __restrict__ state,
    Rank64 local_components,
    int W,
    int K,
    bool high,
    bool expand,
    int gpu_id,
    int ngpu,
    const Rank64* __restrict__ owner_begin,
    const Rank64* __restrict__ component_prefix,
    const Rank64* __restrict__ component_sr_begin,
    const Rank64* __restrict__ component_group,
    std::uint32_t mod,
    int* error
) {
    __shared__ DeviceKey sh_src[RP_RUNTIME_WARPS_PER_BLOCK]
                               [RP_RUNTIME_SUBGROUPS_PER_WARP]
                               [RP_RUNTIME_MAX_PAIRS];
    __shared__ DeviceKey sh_dst[RP_RUNTIME_WARPS_PER_BLOCK]
                               [RP_RUNTIME_SUBGROUPS_PER_WARP]
                               [RP_RUNTIME_MAX_PAIRS];
    __shared__ std::uint32_t sh_value[RP_RUNTIME_WARPS_PER_BLOCK]
                                     [RP_RUNTIME_SUBGROUPS_PER_WARP]
                                     [RP_RUNTIME_MAX_PAIRS];
    __shared__ GroupedComponentContextDevice sh_ctx[RP_RUNTIME_WARPS_PER_BLOCK]
                                                   [RP_RUNTIME_SUBGROUPS_PER_WARP];
    __shared__ int sh_ns[RP_RUNTIME_WARPS_PER_BLOCK][RP_RUNTIME_SUBGROUPS_PER_WARP];
    __shared__ int sh_nd[RP_RUNTIME_WARPS_PER_BLOCK][RP_RUNTIME_SUBGROUPS_PER_WARP];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int subgroup = lane / RP_RUNTIME_SUBGROUP_WIDTH;
    const int sublane = lane & (RP_RUNTIME_SUBGROUP_WIDTH - 1);
    const unsigned subgroup_mask = 0xffu << (subgroup * RP_RUNTIME_SUBGROUP_WIDTH);
    const Rank64 warp_global = Rank64(blockIdx.x) * RP_RUNTIME_WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 first = warp_global * RP_RUNTIME_SUBGROUPS_PER_WARP + Rank64(subgroup);
    const Rank64 stride = Rank64(gridDim.x) * RP_RUNTIME_WARPS_PER_BLOCK *
                          RP_RUNTIME_SUBGROUPS_PER_WARP;
    const OwnerComponentPlanDevice plan{
        component_prefix, component_sr_begin, component_group};

    const int source_q = high ? W - 1 : 1;
    const bool source_reverse = high ? !expand : expand;
    const int tile_start = high
        ? (expand ? W - 1 : K + 1)
        : (expand ? 1 : K + 1);
    const int destination_q = high
        ? (expand ? W - 2 : W - 1)
        : (expand ? 2 : 1);

    for (Rank64 local_rank = first; local_rank < local_components; local_rank += stride) {
        if (sublane == 0) {
            sh_ns[warp][subgroup] = 0;
            sh_nd[warp][subgroup] = 0;
            const MateID label = expand
                ? owner_component_label_unrank_planned_device(
                    W,
                    high ? W - 1 : 1,
                    high ? false : true,
                    high ? W - 1 : 1,
                    K,
                    plan,
                    local_rank)
                : runtime_turn_compress_label_unrank_device(
                    W, K, high, plan, local_rank);
            const DeviceKey seed = expand
                ? runtime_turn_expand_seed(label, W, high)
                : runtime_turn_compress_seed(label, W, high);
            const GroupedComponentContextDevice ctx = grouped_component_context_device(
                seed, W, source_q, source_reverse, tile_start, K, ngpu, owner_begin);
            sh_ctx[warp][subgroup] = ctx;
            if (ctx.owner != gpu_id) {
                runtime_set_error(error, 321);
            } else {
                sh_src[warp][subgroup][0] = seed;
                sh_ns[warp][subgroup] = 1;
                int cursor = 0;
                while (cursor < sh_ns[warp][subgroup]) {
                    RuntimeSmallTerms edge;
                    if (!runtime_turn_step(
                            sh_src[warp][subgroup][cursor++], W, high, expand, edge)) {
                        runtime_set_error(error, 322);
                        break;
                    }
                    for (int ei = 0; ei < edge.n; ++ei) {
                        if (!edge.v[ei].coef) continue;
                        const DeviceKey d = edge.v[ei].key;
                        if (runtime_find_key(
                                sh_dst[warp][subgroup], sh_nd[warp][subgroup], d) >= 0)
                            continue;
                        if (sh_nd[warp][subgroup] >= RP_RUNTIME_MAX_PAIRS) {
                            runtime_set_error(error, 323);
                            break;
                        }
                        sh_dst[warp][subgroup][sh_nd[warp][subgroup]++] = d;
                        if (!runtime_turn_discover_inverse(
                                d, W, high, expand,
                                sh_src[warp][subgroup], sh_ns[warp][subgroup],
                                RP_RUNTIME_MAX_PAIRS)) {
                            runtime_set_error(error, 324);
                            break;
                        }
                    }
                    if (error && *error) break;
                }
            }
        }
        __syncwarp(subgroup_mask);

        const int ns = sh_ns[warp][subgroup];
        const int nd = sh_nd[warp][subgroup];
        const GroupedComponentContextDevice ctx = sh_ctx[warp][subgroup];
        for (int i = sublane; i < ns; i += RP_RUNTIME_SUBGROUP_WIDTH) {
            const GroupedDeviceRank gr = grouped_rank_in_component_device(
                sh_src[warp][subgroup][i], W, source_q, source_reverse, ctx);
            if (gr.owner != gpu_id) {
                runtime_set_error(error, 325);
                sh_value[warp][subgroup][i] = 0;
            } else {
                sh_value[warp][subgroup][i] = state[gr.local];
            }
        }
        __syncwarp(subgroup_mask);

        for (int di = sublane; di < nd; di += RP_RUNTIME_SUBGROUP_WIDTH) {
            const DeviceKey mine = sh_dst[warp][subgroup][di];
            const GroupedDeviceRank dgr = grouped_rank_in_component_device(
                mine, W, destination_q, source_reverse, ctx);
            if (dgr.owner != gpu_id) {
                runtime_set_error(error, 326);
                continue;
            }
            long long acc = 0;
            for (int si = 0; si < ns; ++si) {
                RuntimeSmallTerms edge;
                if (!runtime_turn_step(
                        sh_src[warp][subgroup][si], W, high, expand, edge)) {
                    runtime_set_error(error, 327);
                    continue;
                }
                for (int ei = 0; ei < edge.n; ++ei) {
                    if (key_equal(edge.v[ei].key, mine))
                        acc += static_cast<long long>(edge.v[ei].coef) *
                               static_cast<long long>(sh_value[warp][subgroup][si]);
                }
            }
            long long z = acc % static_cast<long long>(mod);
            if (z < 0) z += mod;
            state[dgr.local] = static_cast<std::uint32_t>(z);
        }
        __syncwarp(subgroup_mask);
    }
}

} // namespace oneesan::gridfp::reducedprod
