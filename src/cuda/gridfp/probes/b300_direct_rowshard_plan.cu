#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <unordered_map>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"
#include "../ramstream32_cpu_low_inplace.hpp"
#include "../ramstream32_b300_sparse_actions.cuh"

static uint32_t rp_flip_high(uint32_t hc, uint32_t depth) {
    int s = int(depth);
    for (int pos = 0; pos < HIGH_LUT_K; ++pos) {
        MateValue v = MateValue((hc >> (2 * pos)) & 3u);
        if (v == ::L) {
            if (--s == 0) {
                uint32_t z = 3u << (2 * pos);
                return (hc & ~z) | (uint32_t(R) << (2 * pos));
            }
        } else if (v == R) ++s;
    }
    return 0xffffffffu;
}

struct RpCrossStat { uint64_t rows = 0, remote = 0; };

int main() {
    constexpr int NG = 8;
    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost low = build_cpu_low_orbit(storage, layout, lowdesc);
    HighOrbitHost high = build_high_orbit(storage, layout);
    B300SparseActionsHost sparse = build_b300_sparse_actions(
        layout, lowdesc, low, highdesc, high);

    std::array<unsigned long long, NG> auth_bytes{};
    auto add_blocks = [&](const std::vector<StorageBlock>& blocks) {
        for (const auto& b : blocks) if (b.valid && b.rows && b.cols) {
            for (uint32_t hr = 0; hr < b.rows; ++hr)
                auth_bytes[hr % NG] += (unsigned long long)b.cols * sizeof(Count);
        }
    };
    add_blocks(layout.main_blocks);
    add_blocks(layout.block_blocks);

    std::array<unsigned long long, NG> high_orbit_ops{}, high_closure_ops{};
    std::array<unsigned long long, NG> high_source_cells{};
    unsigned long long high_partner_remote_cells = 0;
    unsigned long long high_drop_remote_cells = 0;
    unsigned long long high_closure_remote_cells = 0;
    unsigned long long high_closure_cells = 0;

    for (const auto& op : sparse.high_orbit) {
        const auto& x = layout.main_blocks[b300_sparse_sblock(op)];
        uint32_t so = b300_sparse_src(op) % NG;
        uint32_t jo = b300_sparse_jrank(op) % NG;
        uint32_t do_ = b300_sparse_drank(op) % NG;
        ++high_orbit_ops[so];
        high_source_cells[so] += x.cols;
        if (jo != so) high_partner_remote_cells += x.cols;
        if (do_ != so) high_drop_remote_cells += x.cols;
    }
    for (uint64_t op : sparse.high_closure) {
        const auto& x = layout.main_blocks[b300_sparse_closure_sblock(op)];
        uint32_t desc = b300_sparse_closure_desc(op);
        uint32_t so = b300_sparse_closure_src(op) % NG;
        uint32_t to = highdesc_rank(desc) % NG;
        ++high_closure_ops[so];
        high_source_cells[so] += x.cols;
        high_closure_cells += x.cols;
        if (to != so) high_closure_remote_cells += x.cols;
    }

    unsigned long long low_orbit_cells = 0;
    unsigned long long low_closure_cells = 0;
    unsigned long long low_cross_cells = 0;
    unsigned long long low_cross_remote_cells = 0;
    for (const auto& op : sparse.low_orbit) {
        const auto& x = layout.main_blocks[b300_sparse_sblock(op)];
        low_orbit_cells += x.rows;
    }

    // Key: source ending height, target ending height, matching depth.
    // LOW CROSS changes one L/R endpoint, hence occupancy stays fixed; only the
    // HIGH row rank can move to another modulo-8 owner.
    std::unordered_map<uint32_t, RpCrossStat> cross_cache;
    for (int p = LOW_LUT_K; p >= 1; --p) {
        uint32_t pi = uint32_t(LOW_LUT_K - p);
        for (uint32_t q = sparse.low_closure_off[pi]; q < sparse.low_closure_off[pi + 1]; ++q) {
            uint64_t op = sparse.low_closure[q];
            uint32_t sbid = b300_sparse_closure_sblock(op);
            uint32_t desc = b300_sparse_closure_desc(op);
            const auto& x = layout.main_blocks[sbid];
            uint32_t kind = lowdesc_kind(desc);
            low_closure_cells += x.rows;
            if (kind != LOWDESC_CROSS) continue;
            const auto& y = (p == 1)
                ? layout.main_blocks[lowdesc_block(desc)]
                : layout.block_blocks[lowdesc_block(desc)];
            uint32_t depth = lowdesc_depth(desc);
            uint32_t key = uint32_t(x.he) | (uint32_t(y.he) << 8) | (depth << 16);
            auto it = cross_cache.find(key);
            if (it == cross_cache.end()) {
                RpCrossStat st;
                st.rows = x.rows;
                for (uint32_t hr = 0; hr < x.rows; ++hr) {
                    uint32_t hc = storage.high_all_codes[storage.high_all_off[x.he] + hr];
                    uint32_t hc2 = rp_flip_high(hc, depth);
                    if (hc2 == 0xffffffffu) continue;
                    uint32_t packed = storage.high_packed_rank[hc2];
                    if (packed == 0xffffffffu) std::exit(430);
                    uint32_t hr2 = packed >> HIGH_LUT_K;
                    if (hr2 >= y.rows) std::exit(431);
                    if ((hr2 % NG) != (hr % NG)) ++st.remote;
                }
                it = cross_cache.emplace(key, st).first;
            }
            low_cross_cells += it->second.rows;
            low_cross_remote_cells += it->second.remote;
        }
    }

    uint64_t low_sparse_bytes = sparse.low_orbit.size() * sizeof(B300SparseOrbitOp)
                              + sparse.low_closure.size() * sizeof(uint64_t)
                              + (sparse.low_orbit_off.size() + sparse.low_closure_off.size()) * sizeof(uint32_t);
    uint64_t common_bytes = uint64_t(storage.low_all_codes.size() + storage.high_all_codes.size()) * sizeof(uint32_t)
                          + uint64_t(storage.low_mask_begin.size() + storage.high_mask_begin.size()) * sizeof(uint32_t)
                          + uint64_t(G_FACTOR.low_mask_codes.size() + G_FACTOR.low_mask_off.size()
                                   + G_FACTOR.high_mask_codes.size() + G_FACTOR.high_mask_off.size()) * sizeof(uint32_t);

    std::array<unsigned long long, NG> high_sparse_bytes{};
    for (int g = 0; g < NG; ++g) {
        high_sparse_bytes[g] = high_orbit_ops[g] * sizeof(B300SparseOrbitOp)
                             + high_closure_ops[g] * sizeof(uint64_t)
                             + 2ull * (HIGH_LUT_K + 1) * sizeof(uint32_t);
    }

    auto gib = [](long double x) { return double(x / (1ull << 30)); };
    auto mib = [](long double x) { return double(x / (1ull << 20)); };
    auto frac = [](unsigned long long a, unsigned long long b) {
        return b ? double(a) / double(b) : 0.0;
    };

    unsigned long long auth_min = auth_bytes[0], auth_max = auth_bytes[0];
    unsigned long long meta_min = low_sparse_bytes + common_bytes + high_sparse_bytes[0];
    unsigned long long meta_max = meta_min;
    for (int g = 0; g < NG; ++g) {
        auth_min = std::min(auth_min, auth_bytes[g]);
        auth_max = std::max(auth_max, auth_bytes[g]);
        unsigned long long mb = low_sparse_bytes + common_bytes + high_sparse_bytes[g];
        meta_min = std::min(meta_min, mb);
        meta_max = std::max(meta_max, mb);
    }

    std::cout << std::fixed << std::setprecision(6)
        << "b300-direct-rowshard-plan W=" << TARGET_W << " gpus=" << NG
        << " auth_min_gib=" << gib(auth_min)
        << " auth_max_gib=" << gib(auth_max)
        << " auth_imbalance=" << (auth_min ? double(auth_max) / double(auth_min) : 0.0)
        << '\n'
        << "low_sparse_replicated_mib=" << mib(low_sparse_bytes)
        << " common_meta_mib=" << mib(common_bytes)
        << " high_sparse_partitioned_min_mib=" << mib(*std::min_element(high_sparse_bytes.begin(), high_sparse_bytes.end()))
        << " high_sparse_partitioned_max_mib=" << mib(*std::max_element(high_sparse_bytes.begin(), high_sparse_bytes.end()))
        << " meta_min_mib=" << mib(meta_min)
        << " meta_max_mib=" << mib(meta_max)
        << " max_need_gib=" << gib(auth_max + meta_max)
        << '\n'
        << "high_partner_remote_fraction=" << frac(high_partner_remote_cells,
             high_partner_remote_cells + 0ULL + [&](){ unsigned long long z=0; for (auto x: high_source_cells) z+=x; return z; }())
        << " high_partner_remote_cells=" << high_partner_remote_cells
        << " high_drop_remote_cells=" << high_drop_remote_cells
        << " high_closure_remote_fraction=" << frac(high_closure_remote_cells, high_closure_cells)
        << " high_closure_remote_cells=" << high_closure_remote_cells
        << '\n'
        << "low_orbit_cells=" << low_orbit_cells
        << " low_orbit_remote_cells=0"
        << " low_closure_cells=" << low_closure_cells
        << " low_cross_cells=" << low_cross_cells
        << " low_cross_remote_cells=" << low_cross_remote_cells
        << " low_cross_remote_fraction=" << frac(low_cross_remote_cells, low_cross_cells)
        << '\n'
        << "runtime_groups=0 bulk_gather_scatter_bytes=0"
        << " low_non_cross_source_target_local=1"
        << " high_action_partition=source_row_owner"
        << '\n';
    return 0;
}
