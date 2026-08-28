#pragma once

#include "two_cell_component_device.cuh"

#if defined(__CUDACC__)
#define ONEESAN_TC_MATCH_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_TC_MATCH_HD inline
#endif

namespace oneesan::twocell {

constexpr int kMaxComponentMatching = 18;

struct ComponentMatching {
    std::uint32_t adjacency[kMaxComponentMatching]{}; // source -> stationary destination index bits
    std::uint8_t src_to_dst[kMaxComponentMatching]{};
    std::uint8_t dst_to_src[kMaxComponentMatching]{};
    int size = 0;
    int edges = 0;
    int residual_edges = 0;
    bool ok = false;
};

ONEESAN_TC_MATCH_HD int coordinate_index_for_destination(
    const PackedKey* src,
    int n,
    PackedKey dst,
    int i
) {
    for (int t = 0; t < n; ++t)
        if (equal(recouple_coordinate(src[t], i), dst)) return t;
    return -1;
}

// The support graph of every interior two-cell component is a balanced tree.
// recouple_coordinate() only identifies the destination coordinate set with the
// stationary source address set; it is NOT generally the matrix matching edge.
// Recover the unique perfect matching locally by repeatedly peeling a degree-1
// source or destination.  n<=17 at W=28, so no global matching table is needed.
ONEESAN_TC_MATCH_HD ComponentMatching build_component_matching(
    const PackedKey* src,
    int n,
    int W,
    int i
) {
    ComponentMatching out{};
    out.size = n;
    if (n <= 0 || n > kMaxComponentMatching) return out;
    for (int q = 0; q < kMaxComponentMatching; ++q) {
        out.src_to_dst[q] = 0xffu;
        out.dst_to_src[q] = 0xffu;
    }

    for (int s = 0; s < n; ++s) {
        const auto edges = K_step(src[s], W, i);
        if (edges.overflow || edges.size <= 0 || edges.size > 3) return out;
        std::uint32_t mask = 0;
        for (int e = 0; e < edges.size; ++e) {
            const int t = coordinate_index_for_destination(src, n, edges.value[e], i);
            if (t < 0) return out;
            mask |= std::uint32_t(1) << t;
        }
        if (popcount32(mask) != edges.size) return out;
        out.adjacency[s] = mask;
        out.edges += edges.size;
    }

    std::uint32_t alive_s = low_mask(n);
    std::uint32_t alive_d = low_mask(n);
    int matched = 0;
    while (matched < n) {
        bool progress = false;

        for (int s = 0; s < n; ++s) {
            if (!((alive_s >> s) & 1u)) continue;
            const std::uint32_t m = out.adjacency[s] & alive_d;
            if (popcount32(m) != 1) continue;
            const int t = ctz32(m);
            out.src_to_dst[s] = static_cast<std::uint8_t>(t);
            out.dst_to_src[t] = static_cast<std::uint8_t>(s);
            alive_s &= ~(std::uint32_t(1) << s);
            alive_d &= ~(std::uint32_t(1) << t);
            ++matched;
            progress = true;
            break;
        }
        if (progress) continue;

        for (int t = 0; t < n; ++t) {
            if (!((alive_d >> t) & 1u)) continue;
            int only = -1;
            int degree = 0;
            for (int s = 0; s < n; ++s) {
                if (!((alive_s >> s) & 1u)) continue;
                if (!((out.adjacency[s] >> t) & 1u)) continue;
                only = s;
                ++degree;
            }
            if (degree != 1) continue;
            out.src_to_dst[only] = static_cast<std::uint8_t>(t);
            out.dst_to_src[t] = static_cast<std::uint8_t>(only);
            alive_s &= ~(std::uint32_t(1) << only);
            alive_d &= ~(std::uint32_t(1) << t);
            ++matched;
            progress = true;
            break;
        }
        if (!progress) return out;
    }

    for (int s = 0; s < n; ++s) {
        const int t = out.src_to_dst[s];
        if (t < 0 || t >= n || !((out.adjacency[s] >> t) & 1u)) return out;
    }
    for (int t = 0; t < n; ++t) {
        const int s = out.dst_to_src[t];
        if (s < 0 || s >= n || out.src_to_dst[s] != t) return out;
    }

    out.residual_edges = out.edges - n;
    if (out.residual_edges != n - 1) return out;
    out.ok = true;
    return out;
}

// Exact unit-coefficient component multiply after all source values have been
// captured locally.  Matching edges become a permutation copy; every remaining
// tree edge is exactly one modular addition.  This keeps the minimal n-1 adds
// without requiring the matching permutation to be identity in stationary
// coordinates.
template <class Value>
ONEESAN_TC_MATCH_HD bool apply_component_matching(
    const ComponentMatching& m,
    const Value* input,
    Value* output,
    std::uint32_t mod
) {
    if (!m.ok) return false;
    for (int t = 0; t < m.size; ++t)
        output[t] = input[m.dst_to_src[t]];
    for (int s = 0; s < m.size; ++s) {
        std::uint32_t residual = m.adjacency[s] &
            ~(std::uint32_t(1) << m.src_to_dst[s]);
        while (residual) {
            const int t = ctz32(residual);
            const unsigned long long z =
                static_cast<unsigned long long>(output[t]) +
                static_cast<unsigned long long>(input[s]);
            output[t] = static_cast<Value>(z >= mod ? z - mod : z);
            residual &= residual - 1;
        }
    }
    return true;
}

} // namespace oneesan::twocell

#undef ONEESAN_TC_MATCH_HD
