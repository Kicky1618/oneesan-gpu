#pragma once

#include "two_cell_component_device.cuh"

#if defined(__CUDACC__)
#define ONEESAN_TC_MATCH_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_TC_MATCH_HD inline
#endif

namespace oneesan::twocell {

constexpr int kMaxComponentMatching = 18;

enum ComponentFastKind : std::uint8_t {
    TC_MATCH_GENERIC = 0,
    TC_MATCH_SINGLETON = 1,
    TC_MATCH_TRIPLE = 2,
    TC_MATCH_DEEP_RN = 3,
    TC_MATCH_DEEP_LR = 4,
};

struct ComponentMatching {
    std::uint32_t adjacency[kMaxComponentMatching]{};
    std::uint8_t src_to_dst[kMaxComponentMatching]{};
    std::uint8_t dst_to_src[kMaxComponentMatching]{};
    int size = 0;
    int edges = 0;
    int residual_edges = 0;
    ComponentFastKind fast_kind = TC_MATCH_GENERIC;
    bool ok = false;
};

ONEESAN_TC_MATCH_HD void clear_component_matching(ComponentMatching& out, int n) {
    out = ComponentMatching{};
    out.size = n;
    for (int q = 0; q < kMaxComponentMatching; ++q) {
        out.src_to_dst[q] = 0xffu;
        out.dst_to_src[q] = 0xffu;
    }
}

ONEESAN_TC_MATCH_HD ComponentFastKind classify_component_fastpath(
    const PackedKey* src,
    int n,
    int W,
    int i
) {
    if (n == 1) return TC_MATCH_SINGLETON;
    if (n == 3) return TC_MATCH_TRIPLE;
    if (n < 5 || src[0].type != 1) return TC_MATCH_GENERIC;
    const PackedWord u = state_word(src[0], W);
    const Symbol a = symbol(u, i);
    const Symbol b = symbol(u, i + 1);
    if (a == TC_R && b == TC_N) return TC_MATCH_DEEP_RN;
    if (a == TC_L && b == TC_R) return TC_MATCH_DEEP_LR;
    return TC_MATCH_GENERIC; // deep LN remains topology-dependent
}

// direct_component_sources() has a canonical source order.  In this order four
// component families have closed support graphs.  Only deep LN components need
// K_step() edge generation and leaf peeling.
ONEESAN_TC_MATCH_HD bool build_component_matching_fastpath(
    const PackedKey* src,
    int n,
    int W,
    int i,
    ComponentMatching& out
) {
    clear_component_matching(out, n);
    const ComponentFastKind kind = classify_component_fastpath(src, n, W, i);
    out.fast_kind = kind;

    if (kind == TC_MATCH_SINGLETON) {
        out.adjacency[0] = 0x1u;
        out.src_to_dst[0] = 0;
        out.dst_to_src[0] = 0;
        out.edges = 1;
        out.residual_edges = 0;
        out.ok = true;
        return true;
    }

    if (kind == TC_MATCH_TRIPLE) {
        out.adjacency[0] = 0x4u;
        out.adjacency[1] = 0x2u;
        out.adjacency[2] = 0x7u;
        out.src_to_dst[0] = 2;
        out.src_to_dst[1] = 1;
        out.src_to_dst[2] = 0;
        out.dst_to_src[0] = 2;
        out.dst_to_src[1] = 1;
        out.dst_to_src[2] = 0;
        out.edges = 5;
        out.residual_edges = 2;
        out.ok = true;
        return true;
    }

    if (kind == TC_MATCH_DEEP_LR) {
        out.adjacency[0] = std::uint32_t(1) << 2;
        out.adjacency[1] = std::uint32_t(1) << 1;
        out.adjacency[2] = 0x7u;
        out.adjacency[3] = (std::uint32_t(1) << 1) | (std::uint32_t(1) << 3);
        for (int s = 4; s < n; ++s)
            out.adjacency[s] = (std::uint32_t(1) << 3) | (std::uint32_t(1) << s);
        out.src_to_dst[0] = 2;
        out.src_to_dst[1] = 1;
        out.src_to_dst[2] = 0;
        out.dst_to_src[0] = 2;
        out.dst_to_src[1] = 1;
        out.dst_to_src[2] = 0;
        for (int s = 3; s < n; ++s) {
            out.src_to_dst[s] = static_cast<std::uint8_t>(s);
            out.dst_to_src[s] = static_cast<std::uint8_t>(s);
        }
        out.edges = 2 * n - 1;
        out.residual_edges = n - 1;
        out.ok = true;
        return true;
    }

    if (kind == TC_MATCH_DEEP_RN) {
        out.adjacency[0] = std::uint32_t(1) << 2;
        out.adjacency[1] = std::uint32_t(1) << 1;
        out.adjacency[2] = (std::uint32_t(1) << 1) |
                           (std::uint32_t(1) << 2) |
                           (std::uint32_t(1) << (n - 1));
        out.adjacency[3] = (std::uint32_t(1) << 0) | (std::uint32_t(1) << 3);
        for (int s = 4; s < n; ++s)
            out.adjacency[s] = (std::uint32_t(1) << 3) | (std::uint32_t(1) << s);

        out.src_to_dst[0] = 2;
        out.src_to_dst[1] = 1;
        out.src_to_dst[2] = static_cast<std::uint8_t>(n - 1);
        out.src_to_dst[3] = 0;
        out.dst_to_src[0] = 3;
        out.dst_to_src[1] = 1;
        out.dst_to_src[2] = 0;
        out.dst_to_src[n - 1] = 2;
        for (int s = 4; s < n - 1; ++s) {
            out.src_to_dst[s] = static_cast<std::uint8_t>(s);
            out.dst_to_src[s] = static_cast<std::uint8_t>(s);
        }
        out.src_to_dst[n - 1] = 3;
        out.dst_to_src[3] = static_cast<std::uint8_t>(n - 1);
        out.edges = 2 * n - 1;
        out.residual_edges = n - 1;
        out.ok = true;
        return true;
    }

    return false;
}

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

ONEESAN_TC_MATCH_HD ComponentMatching build_component_matching(
    const PackedKey* src,
    int n,
    int W,
    int i
) {
    ComponentMatching out{};
    if (n <= 0 || n > kMaxComponentMatching) return out;
    if (build_component_matching_fastpath(src, n, W, i, out)) return out;
    clear_component_matching(out, n);

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

template <class Value>
ONEESAN_TC_MATCH_HD Value component_add_mod(Value a, Value b, std::uint32_t mod) {
    const unsigned long long z =
        static_cast<unsigned long long>(a) + static_cast<unsigned long long>(b);
    return static_cast<Value>(z >= mod ? z - mod : z);
}

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
            output[t] = component_add_mod(output[t], input[s], mod);
            residual &= residual - 1;
        }
    }
    return true;
}

// Descriptor-free arithmetic for all closed families.  Returns false only for
// topology-dependent deep LN components.
template <class Value>
ONEESAN_TC_MATCH_HD bool apply_component_fastpath(
    const PackedKey* src,
    int n,
    int W,
    int i,
    const Value* input,
    Value* output,
    std::uint32_t mod
) {
    const ComponentFastKind kind = classify_component_fastpath(src, n, W, i);
    if (kind == TC_MATCH_SINGLETON) {
        output[0] = input[0];
        return true;
    }
    if (kind == TC_MATCH_TRIPLE) {
        output[0] = input[2];
        output[1] = component_add_mod(input[1], input[2], mod);
        output[2] = component_add_mod(input[0], input[2], mod);
        return true;
    }
    if (kind == TC_MATCH_DEEP_LR) {
        output[0] = input[2];
        output[1] = input[1];
        output[2] = input[0];
        for (int q = 3; q < n; ++q) output[q] = input[q];
        output[1] = component_add_mod(output[1], input[2], mod);
        output[2] = component_add_mod(output[2], input[2], mod);
        output[1] = component_add_mod(output[1], input[3], mod);
        for (int q = 4; q < n; ++q)
            output[3] = component_add_mod(output[3], input[q], mod);
        return true;
    }
    if (kind == TC_MATCH_DEEP_RN) {
        output[0] = input[3];
        output[1] = input[1];
        output[2] = input[0];
        output[3] = input[n - 1];
        for (int q = 4; q < n - 1; ++q) output[q] = input[q];
        output[n - 1] = input[2];
        output[1] = component_add_mod(output[1], input[2], mod);
        output[2] = component_add_mod(output[2], input[2], mod);
        output[3] = component_add_mod(output[3], input[3], mod);
        for (int q = 4; q < n - 1; ++q)
            output[3] = component_add_mod(output[3], input[q], mod);
        output[n - 1] = component_add_mod(output[n - 1], input[n - 1], mod);
        return true;
    }
    return false;
}

} // namespace oneesan::twocell

#undef ONEESAN_TC_MATCH_HD
