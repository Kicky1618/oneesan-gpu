#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <unordered_set>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_cpu_low_inplace.hpp"

struct DSU {
    std::vector<uint32_t> p, sz;
    explicit DSU(size_t n): p(n), sz(n,1) { std::iota(p.begin(), p.end(), 0); }
    uint32_t find(uint32_t x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }
    void unite(uint32_t a, uint32_t b) {
        a=find(a); b=find(b); if(a==b) return;
        if(sz[a]<sz[b]) std::swap(a,b);
        p[b]=a; sz[a]+=sz[b];
    }
};

static uint64_t high_row_state_weight(const StorageLayout& logical, int he) {
    uint64_t z = 0;
    for (int c = 0; c < 3; ++c) {
        const StorageBlock& b = logical.main_blocks[size_t(3 * he + c)];
        if (b.valid) z += b.cols;
    }
    if (he < int(logical.block_blocks.size())) {
        const StorageBlock& b = logical.block_blocks[size_t(he)];
        if (b.valid) z += b.cols;
    }
    return z;
}

int main() {
    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout logical = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, logical);

    constexpr int L = LOW_LUT_K;
    constexpr int H = HIGH_LUT_K;
    constexpr uint32_t HM = (1u << (2 * H)) - 1u;

    // Valid HIGH topology nodes are exactly storage.high_all_codes.  Node id is
    // its index in that height-major array.  Deduplicate the millions of LOW
    // cross descriptors down to the tiny set of (source-height, depth) actions.
    bool used[MAXW + 2][16]{};
    uint64_t cross_desc = 0;
    for (int p = L; p >= 1; --p) {
        uint32_t pi = uint32_t(L - p);
        for (size_t bid = 0; bid < logical.main_blocks.size(); ++bid) {
            const StorageBlock& b = logical.main_blocks[bid];
            if (!b.valid) continue;
            for (uint32_t lr = 0; lr < b.cols; ++lr) {
                uint32_t w = lowdesc.main_desc[
                    size_t(pi) * lowdesc.main_total + lowdesc.main_base[bid] + lr];
                if (cpu_low_kind(w) != LOWDESC_CROSS) continue;
                uint32_t d = cpu_low_depth(w);
                if (!d || d >= 16) std::exit(250);
                used[b.he][d] = true;
                ++cross_desc;
            }
        }
    }

    DSU dsu(storage.high_all_codes.size());
    uint64_t edge_attempts = 0, edges = 0;
    uint32_t signatures = 0;
    for (int he = 0; he <= H + 1; ++he) {
        uint32_t a = storage.high_all_off[he], b = storage.high_all_off[he + 1];
        for (uint32_t depth = 1; depth < 16; ++depth) {
            if (!used[he][depth]) continue;
            ++signatures;
            for (uint32_t ix = a; ix < b; ++ix) {
                uint32_t hc = storage.high_all_codes[ix] & HM;
                uint32_t hc2 = cpu_low_flip_high(hc, depth);
                ++edge_attempts;
                if (hc2 == 0xffffffffu) continue;
                uint32_t packed = storage.high_packed_rank[hc2];
                if (packed == 0xffffffffu) continue;
                int he2 = seg_end_height_host(hc2, H);
                uint32_t j = storage.high_all_off[he2] + (packed >> H);
                if (j >= storage.high_all_codes.size() || storage.high_all_codes[j] != hc2)
                    std::exit(251);
                dsu.unite(ix, j);
                ++edges;
            }
        }
    }

    std::vector<uint64_t> comp_states(storage.high_all_codes.size(), 0);
    std::vector<uint32_t> comp_nodes(storage.high_all_codes.size(), 0);
    for (uint32_t ix = 0; ix < storage.high_all_codes.size(); ++ix) {
        uint32_t r = dsu.find(ix);
        int he = seg_end_height_host(storage.high_all_codes[ix], H);
        comp_states[r] += high_row_state_weight(logical, he);
        ++comp_nodes[r];
    }

    std::vector<uint64_t> state_sizes;
    std::vector<uint32_t> node_sizes;
    for (uint32_t i = 0; i < comp_nodes.size(); ++i) if (comp_nodes[i]) {
        node_sizes.push_back(comp_nodes[i]);
        state_sizes.push_back(comp_states[i]);
    }
    std::sort(node_sizes.begin(), node_sizes.end());
    std::sort(state_sizes.begin(), state_sizes.end());
    auto q64 = [&](double q) -> uint64_t {
        if (state_sizes.empty()) return 0;
        size_t i = std::min(state_sizes.size()-1, size_t(q * double(state_sizes.size()-1)));
        return state_sizes[i];
    };
    auto q32 = [&](double q) -> uint32_t {
        if (node_sizes.empty()) return 0;
        size_t i = std::min(node_sizes.size()-1, size_t(q * double(node_sizes.size()-1)));
        return node_sizes[i];
    };
    uint64_t max_states = state_sizes.empty()?0:state_sizes.back();
    uint32_t max_nodes = node_sizes.empty()?0:node_sizes.back();

    std::cout
        << "low-highcomponent-probe W=" << TARGET_W
        << " high_codes=" << storage.high_all_codes.size()
        << " cross_desc=" << cross_desc
        << " signatures=" << signatures
        << " edge_attempts=" << edge_attempts
        << " edges=" << edges
        << " components=" << state_sizes.size()
        << " node_p50=" << q32(0.50)
        << " node_p90=" << q32(0.90)
        << " node_p99=" << q32(0.99)
        << " node_max=" << max_nodes
        << " data_p50_mib=" << double(q64(0.50) * sizeof(Count)) / (1<<20)
        << " data_p90_mib=" << double(q64(0.90) * sizeof(Count)) / (1<<20)
        << " data_p99_mib=" << double(q64(0.99) * sizeof(Count)) / (1<<20)
        << " data_max_mib=" << double(max_states * sizeof(Count)) / (1<<20)
        << '\n';
    return 0;
}
