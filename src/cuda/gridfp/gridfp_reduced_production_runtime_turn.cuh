#pragma once

#include "gridfp_reduced_production_runtime_subwarp.cuh"

#ifndef RP_RUNTIME_TURN_LOCAL_SECTOR_TABLE
#define RP_RUNTIME_TURN_LOCAL_SECTOR_TABLE 1
#endif
#ifndef RP_RUNTIME_TURN_LOCAL_SECTOR_W28_TREE
#define RP_RUNTIME_TURN_LOCAL_SECTOR_W28_TREE 0
#endif
#ifndef RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN
#define RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN 0
#endif
#ifndef RP_RUNTIME_TURN_DIRECT_COMPRESS_INVERSE
#define RP_RUNTIME_TURN_DIRECT_COMPRESS_INVERSE 1
#endif
static_assert(RP_RUNTIME_TURN_LOCAL_SECTOR_TABLE == 0 ||
              RP_RUNTIME_TURN_LOCAL_SECTOR_TABLE == 1,
              "RP_RUNTIME_TURN_LOCAL_SECTOR_TABLE must be 0 or 1");
static_assert(RP_RUNTIME_TURN_LOCAL_SECTOR_W28_TREE == 0 ||
              RP_RUNTIME_TURN_LOCAL_SECTOR_W28_TREE == 1,
              "RP_RUNTIME_TURN_LOCAL_SECTOR_W28_TREE must be 0 or 1");
static_assert(RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN == 0 ||
              RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN == 1,
              "RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN must be 0 or 1");
static_assert(RP_RUNTIME_TURN_DIRECT_COMPRESS_INVERSE == 0 ||
              RP_RUNTIME_TURN_DIRECT_COMPRESS_INVERSE == 1,
              "RP_RUNTIME_TURN_DIRECT_COMPRESS_INVERSE must be 0 or 1");

namespace oneesan::gridfp::reducedprod {

static constexpr int RP_RUNTIME_TURN_LOCAL_SECTOR_END_ENTRIES = 550;
__device__ __constant__ std::uint32_t
RP_RUNTIME_TURN_LOCAL_SECTOR_END[RP_RUNTIME_TURN_LOCAL_SECTOR_END_ENTRIES] = {
#include "gridfp_reduced_production_runtime_turn_local_sector_end_values.inc"
};
static_assert(sizeof(RP_RUNTIME_TURN_LOCAL_SECTOR_END) == 2200);

__device__ __forceinline__ int runtime_turn_local_sector_width_base_device(int W) {
    switch (W) {
    case 8: return 0; case 10: return 10; case 12: return 25;
    case 14: return 46; case 16: return 74; case 18: return 110;
    case 20: return 155; case 22: return 210; case 24: return 276;
    case 26: return 354; case 28: return 445; default: return -1;
    }
}

__device__ __forceinline__ void runtime_turn_local_sector_w28_tree_device(
    int row,
    int first,
    bool eight,
    Rank64 within,
    int& local_ones,
    Rank64& local_within
) {
    // W=28,L=15 has seven active sectors for even outer popcount and eight
    // for odd outer popcount. Production `within` is a remainder modulo the
    // row group, hence is below the final endpoint. A fixed lower_bound tree
    // can therefore omit the sentinel branch and reuse the compared
    // predecessor as `begin`: at most three constant-table loads vs up to
    // four comparisons plus one begin reload in the generic path.
    const Rank64 e3 = RP_RUNTIME_TURN_LOCAL_SECTOR_END[row + 3];
    int index = 0;
    Rank64 begin = 0;
    if (within < e3) {
        const Rank64 e1 = RP_RUNTIME_TURN_LOCAL_SECTOR_END[row + 1];
        if (within < e1) {
            const Rank64 e0 = RP_RUNTIME_TURN_LOCAL_SECTOR_END[row];
            if (within < e0) {
                index = 0;
            } else {
                index = 1;
                begin = e0;
            }
        } else {
            const Rank64 e2 = RP_RUNTIME_TURN_LOCAL_SECTOR_END[row + 2];
            if (within < e2) {
                index = 2;
                begin = e1;
            } else {
                index = 3;
                begin = e2;
            }
        }
    } else {
        const Rank64 e5 = RP_RUNTIME_TURN_LOCAL_SECTOR_END[row + 5];
        if (within < e5) {
            const Rank64 e4 = RP_RUNTIME_TURN_LOCAL_SECTOR_END[row + 4];
            if (within < e4) {
                index = 4;
                begin = e3;
            } else {
                index = 5;
                begin = e4;
            }
        } else if (eight) {
            const Rank64 e6 = RP_RUNTIME_TURN_LOCAL_SECTOR_END[row + 6];
            if (within < e6) {
                index = 6;
                begin = e5;
            } else {
                index = 7;
                begin = e6;
            }
        } else {
            index = 6;
            begin = e5;
        }
    }
    local_ones = first + (index << 1);
    local_within = within - begin;
}

__device__ __forceinline__ bool runtime_turn_local_sector_device(
    int W,
    int L,
    int outer_ones,
    Rank64 within,
    int& local_ones,
    Rank64& local_within
) {
#if RP_RUNTIME_TURN_LOCAL_SECTOR_TABLE
    const int base = runtime_turn_local_sector_width_base_device(W);
    const int O = W - L;
    if (base >= 0 && L == W / 2 + 1 && outer_ones >= 0 && outer_ones <= O) {
        const int odd_l = L >> 1;
        const int even_l = (L + 1) >> 1;
        const int prior_even_r = (outer_ones + 1) >> 1;
        const int prior_odd_r = outer_ones >> 1;
        const int row = base + prior_even_r * odd_l + prior_odd_r * even_l;
        const int first = (outer_ones & 1) ? 0 : 1;
        const int count = (outer_ones & 1) ? even_l : odd_l;
#if RP_RUNTIME_TURN_LOCAL_SECTOR_W28_TREE
        if (W == 28 && L == 15) {
            runtime_turn_local_sector_w28_tree_device(
                row, first, count == 8, within, local_ones, local_within);
            return true;
        }
#endif
        int lo = 0;
        int hi = count;
        while (lo < hi) {
            const int mid = lo + ((hi - lo) >> 1);
            if (within < RP_RUNTIME_TURN_LOCAL_SECTOR_END[row + mid]) hi = mid;
            else lo = mid + 1;
        }
        if (lo >= count) return false;
        const Rank64 begin = lo ? RP_RUNTIME_TURN_LOCAL_SECTOR_END[row + lo - 1] : 0;
        local_ones = first + (lo << 1);
        local_within = within - begin;
        return true;
    }
#endif
    return false;
}

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

    const int outer_ones = runtime_owner_prefix_sector_device(plan.prefix, O, rank);
    if (outer_ones < 0) return 0;
    const Rank64 local = rank - plan.prefix[outer_ones];
    const Rank64 group = plan.component_group[outer_ones];
    if (!group) return 0;
    Rank64 outer_delta = 0;
    Rank64 within = 0;
    runtime_fastdivmod64_magic(
        local, group, runtime_turn_compress_group_magic(W, outer_ones),
        outer_delta, within);
    const Rank64 outer_sr = plan.sr_begin[outer_ones] + outer_delta;
    const std::uint32_t outer = support_unrank_mask_device(O, outer_ones, outer_sr);

    int local_ones = -1;
    Rank64 local_sr = 0;
    Rank64 primitive_rank = 0;
    Rank64 local_within = 0;
    if (runtime_turn_local_sector_device(
            W, L, outer_ones, within, local_ones, local_within)) {
        const int occupied = outer_ones + local_ones;
        const Rank64 pc = RP_PRIMITIVE[occupied][1];
        runtime_fastdivmod64_magic(
            local_within, pc, RP_RUNTIME_PRIMITIVE1_MAGIC[occupied],
            local_sr, primitive_rank);
    } else {
        Rank64 scan_within = within;
        for (int l = 0; l <= L - 1; ++l) {
            const int occupied = outer_ones + l;
            if (!(occupied & 1)) continue;
            const Rank64 pc = RP_PRIMITIVE[occupied][1];
            const Rank64 n = choose_device(L - 1, l) * pc;
            if (scan_within < n) {
                local_ones = l;
                runtime_fastdivmod64_magic(
                    scan_within, pc, RP_RUNTIME_PRIMITIVE1_MAGIC[occupied],
                    local_sr, primitive_rank);
                break;
            }
            scan_within -= n;
        }
    }
    if (local_ones < 0) return 0;

    const std::uint32_t local_support = support_unrank_mask_device(
        L - 1, local_ones, local_sr);
    std::uint32_t full = 0;
    owner_expand_outer_support_device(outer, W, lo, hi, full);
    owner_expand_local_support_device(local_support, L, lo, missing, full);
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
#if RP_RUNTIME_TURN_DIRECT_COMPRESS_INVERSE
    if (pair == LR && !sink.emit(DeviceKey{msetpair(d, 1, NN), 0})) return false;
    if (pair == RN && !sink.emit(DeviceKey{msetpair(d, 1, NR), 0})) return false;
    if (pair == NR && !sink.emit(DeviceKey{msetpair(d, 1, RN), 0})) return false;

    if (pair == NN) {
        int bal = 0;
#if RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN
        std::uint32_t mask = mate_non_n_mask(d, W) & ~std::uint32_t(3u);
        while (mask) {
            const int q = mate_lsb_index32(mask);
            const MateValue v = mget(d, q);
            if (bal == 0 && v == R) {
                MateID x = msetpair(d, 1, RR);
                x = mset(x, q, L);
                if (!sink.emit(DeviceKey{x, 0})) return false;
            }
            if (v == R) ++bal;
            else if (v == L) --bal;
            if (bal < 0) break;
            mask &= mask - 1u;
        }
#else
        for (int q = 2; q < W; ++q) {
            const MateValue v = mget(d, q);
            if (bal == 0 && v == R) {
                MateID x = msetpair(d, 1, RR);
                x = mset(x, q, L);
                if (!sink.emit(DeviceKey{x, 0})) return false;
            }
            if (v == R) ++bal;
            else if (v == L) --bal;
            if (bal < 0) break;
        }
#endif
    }
    if (pair == NR && !sink.emit(DeviceKey{mshrink(d, 1), 1})) return false;
    return true;
#else
    if (pair == LR && !runtime_turn_try_compress_main(msetpair(d, 1, NN), d, W, sink)) return false;
    if (pair == RN && !runtime_turn_try_compress_main(msetpair(d, 1, NR), d, W, sink)) return false;
    if (pair == LN && !runtime_turn_try_compress_main(msetpair(d, 1, NL), d, W, sink)) return false;
    if (pair == NR && !runtime_turn_try_compress_main(msetpair(d, 1, RN), d, W, sink)) return false;
    if (pair == NL && !runtime_turn_try_compress_main(msetpair(d, 1, LN), d, W, sink)) return false;

    if (pair == NN) {
        if (!runtime_turn_try_compress_main(msetpair(d, 1, RL), d, W, sink))
            return false;
        int bal = 0;
#if RP_RUNTIME_TURN_DISCOVERY_NONN_SCAN
        std::uint32_t mask = mate_non_n_mask(d, W) & ~std::uint32_t(3u);
        while (mask) {
            const int q = mate_lsb_index32(mask);
            const MateValue v = mget(d, q);
            if (bal == 0 && v == R) {
                MateID x = msetpair(d, 1, RR);
                x = mset(x, q, L);
                if (!runtime_turn_try_compress_main(x, d, W, sink)) return false;
            }
            if (v == R) ++bal;
            else if (v == L) --bal;
            if (bal < 0) break;
            mask &= mask - 1u;
        }
#else
        for (int q = 2; q < W; ++q) {
            const MateValue v = mget(d, q);
            if (bal == 0 && v == R) {
                MateID x = msetpair(d, 1, RR);
                x = mset(x, q, L);
                if (!runtime_turn_try_compress_main(x, d, W, sink)) return false;
            }
            if (v == R) ++bal;
            else if (v == L) --bal;
            if (bal < 0) break;
        }
#endif
    }

    if (mget(d, 1) == N && is_endpoint(mget(d, 0))) {
        const MateID b = mshrink(d, 1);
        if (valid_mate_device(b, W - 1) && mget(b, 0) != N &&
            blocked_exclude(b, 1) == d) {
            if (!sink.emit(DeviceKey{b, 1})) return false;
        }
    }
    return true;
#endif
}

template<class Sink>
struct RuntimeMirrorSink {
    Sink sink{};
    int W = 0;
    bool main_only = false;

    __device__ __forceinline__ bool emit(DeviceKey k) {
        const DeviceKey z = mirror_key_device(k, W);
        if (main_only && z.blocked) return true;
        return sink.emit(z);
    }
};

template<class Sink>
struct RuntimeMainOnlySink {
    Sink sink{};
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
    RuntimeSharedKey* source_set,
    int& source_count,
    int capacity,
    std::uint64_t* source_signature = nullptr
) {
    RuntimeSharedKeySetSink base{
        source_set, &source_count, capacity, source_signature};
    if (!high && !expand)
        return runtime_turn_discover_compress_low(dest, W, base);

    if (high && !expand) {
        RuntimeMirrorSink<RuntimeSharedKeySetSink> sink{base, W, false};
        return runtime_turn_discover_compress_low(
            mirror_key_device(dest, W), W, sink);
    }

    if (!high && expand) {
        RuntimeMirrorSink<RuntimeSharedKeySetSink> sink{base, W, true};
        return discover_inverse_reduced_forward(
            mirror_key_device(dest, W), W, W - 1, sink);
    }

    RuntimeMainOnlySink<RuntimeSharedKeySetSink> sink{base};
    return discover_inverse_reduced_forward(dest, W, W - 1, sink);
}

__device__ __forceinline__ bool runtime_turn_discover_inverse_indexed(
    DeviceKey dest,
    int W,
    bool high,
    bool expand,
    RuntimeSharedKey* source_set,
    int& source_count,
    int capacity,
    std::uint64_t& source_occupancy,
    RuntimeFindIndexCache& source_cache
) {
    RuntimeIndexedSharedKeySetSink base{
        source_set, &source_count, capacity, &source_occupancy, &source_cache};
    if (!high && !expand)
        return runtime_turn_discover_compress_low(dest, W, base);

    if (high && !expand) {
        RuntimeMirrorSink<RuntimeIndexedSharedKeySetSink> sink{base, W, false};
        return runtime_turn_discover_compress_low(
            mirror_key_device(dest, W), W, sink);
    }

    if (!high && expand) {
        RuntimeMirrorSink<RuntimeIndexedSharedKeySetSink> sink{base, W, true};
        return discover_inverse_reduced_forward(
            mirror_key_device(dest, W), W, W - 1, sink);
    }

    RuntimeMainOnlySink<RuntimeIndexedSharedKeySetSink> sink{base};
    return discover_inverse_reduced_forward(dest, W, W - 1, sink);
}

// Counter-free turn kernel for both physical row edges and both phases.
// Compression enumerates M_{W-1} components; expansion enumerates the ordinary
// reduced component set M_{W-1}-M_{W-3}. Four independent components share a
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
    __shared__ RuntimeSharedKey sh_src[RP_RUNTIME_WARPS_PER_BLOCK]
                                      [RP_RUNTIME_SUBGROUPS_PER_WARP]
                                      [RP_RUNTIME_MAX_PAIRS];
    __shared__ RuntimeSharedKey sh_dst[RP_RUNTIME_WARPS_PER_BLOCK]
                                      [RP_RUNTIME_SUBGROUPS_PER_WARP]
                                      [RP_RUNTIME_MAX_PAIRS];
    __shared__ std::uint32_t sh_value[RP_RUNTIME_WARPS_PER_BLOCK]
                                     [RP_RUNTIME_SUBGROUPS_PER_WARP]
                                     [RP_RUNTIME_MAX_PAIRS];
    __shared__ GroupedComponentContextDevice sh_ctx[RP_RUNTIME_WARPS_PER_BLOCK]
                                                   [RP_RUNTIME_SUBGROUPS_PER_WARP];
    __shared__ int sh_ns[RP_RUNTIME_WARPS_PER_BLOCK][RP_RUNTIME_SUBGROUPS_PER_WARP];
    __shared__ int sh_nd[RP_RUNTIME_WARPS_PER_BLOCK][RP_RUNTIME_SUBGROUPS_PER_WARP];
#if RP_RUNTIME_CACHE_EDGES
    __shared__ RuntimeEdgeCache sh_edge[RP_RUNTIME_WARPS_PER_BLOCK]
                                       [RP_RUNTIME_SUBGROUPS_PER_WARP];
#endif
#if RP_RUNTIME_FIND_INDEX_CACHE
    __shared__ RuntimeFindIndexCache sh_src_find[RP_RUNTIME_WARPS_PER_BLOCK]
                                            [RP_RUNTIME_SUBGROUPS_PER_WARP];
    __shared__ RuntimeFindIndexCache sh_dst_find[RP_RUNTIME_WARPS_PER_BLOCK]
                                            [RP_RUNTIME_SUBGROUPS_PER_WARP];
#endif

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
#if RP_RUNTIME_FIND_INDEX_CACHE
                std::uint64_t source_occupancy = 0;
                std::uint64_t destination_occupancy = 0;
#elif RP_RUNTIME_FIND_SIGNATURE_FILTER
                std::uint64_t source_signature = runtime_shared_key_signature_bit(seed);
                std::uint64_t destination_signature = 0;
#endif
                sh_src[warp][subgroup][0] = runtime_shared_key_encode(seed);
                sh_ns[warp][subgroup] = 1;
#if RP_RUNTIME_FIND_INDEX_CACHE
                runtime_find_index_record(
                    sh_src_find[warp][subgroup], source_occupancy, seed, 0);
#endif
                int cursor = 0;
                while (cursor < sh_ns[warp][subgroup]) {
                    const int source_ix = cursor++;
#if RP_RUNTIME_CACHE_EDGES
                    sh_edge[warp][subgroup].count[source_ix] = 0;
#endif
                    RuntimeSmallTerms edge;
                    if (!runtime_turn_step(
                            runtime_shared_key_decode(sh_src[warp][subgroup][source_ix]),
                            W, high, expand, edge)) {
                        runtime_set_error(error, 322);
                        break;
                    }
                    for (int ei = 0; ei < edge.n; ++ei) {
                        if (!edge.v[ei].coef) continue;
                        const DeviceKey d = edge.v[ei].key;
#if RP_RUNTIME_FIND_INDEX_CACHE
                        int destination_ix = runtime_find_shared_key_indexed(
                            sh_dst[warp][subgroup], sh_nd[warp][subgroup], d,
                            destination_occupancy, sh_dst_find[warp][subgroup]);
#elif RP_RUNTIME_FIND_SIGNATURE_FILTER
                        int destination_ix = runtime_find_shared_key_filtered(
                            sh_dst[warp][subgroup], sh_nd[warp][subgroup], d,
                            destination_signature);
#else
                        int destination_ix = runtime_find_shared_key(
                            sh_dst[warp][subgroup], sh_nd[warp][subgroup], d);
#endif
                        if (destination_ix < 0) {
                            if (sh_nd[warp][subgroup] >= RP_RUNTIME_MAX_PAIRS) {
                                runtime_set_error(error, 323);
                                cursor = RP_RUNTIME_MAX_PAIRS;
                                break;
                            }
                            destination_ix = sh_nd[warp][subgroup]++;
                            sh_dst[warp][subgroup][destination_ix] =
                                runtime_shared_key_encode(d);
#if RP_RUNTIME_FIND_INDEX_CACHE
                            runtime_find_index_record(
                                sh_dst_find[warp][subgroup], destination_occupancy,
                                d, destination_ix);
                            if (!runtime_turn_discover_inverse_indexed(
                                    d, W, high, expand,
                                    sh_src[warp][subgroup], sh_ns[warp][subgroup],
                                    RP_RUNTIME_MAX_PAIRS, source_occupancy,
                                    sh_src_find[warp][subgroup])) {
#elif RP_RUNTIME_FIND_SIGNATURE_FILTER
                            destination_signature |= runtime_shared_key_signature_bit(d);
                            if (!runtime_turn_discover_inverse(
                                    d, W, high, expand,
                                    sh_src[warp][subgroup], sh_ns[warp][subgroup],
                                    RP_RUNTIME_MAX_PAIRS, &source_signature)) {
#else
                            if (!runtime_turn_discover_inverse(
                                    d, W, high, expand,
                                    sh_src[warp][subgroup], sh_ns[warp][subgroup],
                                    RP_RUNTIME_MAX_PAIRS)) {
#endif
                                runtime_set_error(error, 324);
                                cursor = RP_RUNTIME_MAX_PAIRS;
                                break;
                            }
                        }
#if RP_RUNTIME_CACHE_EDGES
                        if (!runtime_edge_cache_append(
                                sh_edge[warp][subgroup], source_ix,
                                destination_ix, int(edge.v[ei].coef))) {
                            runtime_set_error(error, 328);
                            cursor = RP_RUNTIME_MAX_PAIRS;
                            break;
                        }
#endif
                    }
#if RP_RUNTIME_POLL_GLOBAL_ERROR
                    if (error && *error) break;
#endif
                }
            }
        }
        __syncwarp(subgroup_mask);

        const int ns = sh_ns[warp][subgroup];
        const int nd = sh_nd[warp][subgroup];
        const GroupedComponentContextDevice ctx = sh_ctx[warp][subgroup];
        for (int i = sublane; i < ns; i += RP_RUNTIME_SUBGROUP_WIDTH) {
            const GroupedDeviceRank gr = grouped_rank_in_component_device(
                runtime_shared_key_decode(sh_src[warp][subgroup][i]),
                W, source_q, source_reverse, ctx);
            if (gr.owner != gpu_id) {
                runtime_set_error(error, 325);
                sh_value[warp][subgroup][i] = 0;
            } else {
                sh_value[warp][subgroup][i] = state[gr.local];
            }
        }
        __syncwarp(subgroup_mask);

#if RP_RUNTIME_CACHE_EDGES
        long long accum[RP_RUNTIME_MAX_DESTINATIONS_PER_LANE] = {0, 0, 0};
        for (int si = 0; si < ns; ++si) {
            const long long value = static_cast<long long>(sh_value[warp][subgroup][si]);
            const std::uint8_t nedge = sh_edge[warp][subgroup].count[si];
            for (std::uint8_t ei = 0; ei < nedge; ++ei) {
                const std::uint8_t packed = sh_edge[warp][subgroup].packed[si][ei];
                const std::uint8_t destination_ix = runtime_edge_destination(packed);
                if ((destination_ix & (RP_RUNTIME_SUBGROUP_WIDTH - 1)) == sublane) {
                    const int slot = destination_ix >> 3;
                    accum[slot] +=
                        static_cast<long long>(runtime_edge_coefficient(packed)) * value;
                }
            }
        }
        int accumulator_ix = 0;
        for (int di = sublane; di < nd;
             di += RP_RUNTIME_SUBGROUP_WIDTH, ++accumulator_ix) {
            const DeviceKey mine =
                runtime_shared_key_decode(sh_dst[warp][subgroup][di]);
            const GroupedDeviceRank dgr = grouped_rank_in_component_device(
                mine, W, destination_q, source_reverse, ctx);
            if (dgr.owner != gpu_id) {
                runtime_set_error(error, 326);
                continue;
            }
            state[dgr.local] = runtime_reduce_accum(accum[accumulator_ix], mod);
        }
#else
        for (int di = sublane; di < nd; di += RP_RUNTIME_SUBGROUP_WIDTH) {
            const DeviceKey mine =
                runtime_shared_key_decode(sh_dst[warp][subgroup][di]);
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
                        runtime_shared_key_decode(sh_src[warp][subgroup][si]),
                        W, high, expand, edge)) {
                    runtime_set_error(error, 327);
                    continue;
                }
                for (int ei = 0; ei < edge.n; ++ei) {
                    if (key_equal(edge.v[ei].key, mine))
                        acc += static_cast<long long>(edge.v[ei].coef) *
                               static_cast<long long>(sh_value[warp][subgroup][si]);
                }
            }
            state[dgr.local] = runtime_reduce_accum(acc, mod);
        }
#endif
        __syncwarp(subgroup_mask);
    }
}

} // namespace oneesan::gridfp::reducedprod
