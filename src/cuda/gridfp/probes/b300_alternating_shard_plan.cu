#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"
#include "../ramstream32_cpu_low_inplace.hpp"
#include "../ramstream32_b300_sparse_actions.cuh"
#include "../ramstream32_b300_precomputed_w28_partition.cuh"

struct AltLowShard {
    std::vector<uint8_t> owner;
    std::array<uint64_t, MAXGPU> bytes{};
};

static AltLowShard build_alt_low_lpt(
    const StorageFactorHost& f, const StorageLayout& l, int ngpu
) {
    constexpr int S = MAXW + 2;
    constexpr uint32_t NM = 1u << LOW_LUT_K;
    std::vector<uint64_t> w(NM, 0);
    auto add = [&](const StorageBlock& b) {
        if (!b.valid || !b.rows || !b.cols) return;
        for (uint32_t m = 0; m < NM; ++m) {
            size_t ix = size_t(m) * S + b.hs;
            uint64_t nc = G_FACTOR.low_mask_off[ix + 1] - G_FACTOR.low_mask_off[ix];
            w[m] += nc * uint64_t(b.rows) * sizeof(Count);
        }
    };
    for (const auto& b : l.main_blocks) add(b);
    for (const auto& b : l.block_blocks) add(b);

    std::vector<uint32_t> ord(NM);
    std::iota(ord.begin(), ord.end(), 0u);
    std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) {
        return w[a] != w[b] ? w[a] > w[b] : a < b;
    });
    AltLowShard z;
    z.owner.assign(NM, 0);
    for (uint32_t m : ord) {
        int g = 0;
        for (int q = 1; q < ngpu; ++q) if (z.bytes[q] < z.bytes[g]) g = q;
        z.owner[m] = uint8_t(g);
        z.bytes[g] += w[m];
    }
    return z;
}

static std::array<uint32_t, MAXGPU> alt_high_rows_per_gpu(
    const StorageFactorHost& f, const StorageBlock& b,
    const B300DirectMaskShardHost& high, int ngpu
) {
    std::array<uint32_t, MAXGPU> z{};
    uint32_t off = f.high_all_off[b.he];
    for (uint32_t hr = 0; hr < b.rows; ++hr) ++z[high.high_owner[off + hr]];
    return z;
}

static std::array<uint32_t, MAXGPU> alt_low_cols_per_gpu(
    const StorageFactorHost& f, const StorageBlock& b,
    const AltLowShard& low, int ngpu
) {
    std::array<uint32_t, MAXGPU> z{};
    uint32_t off = f.low_all_off[b.hs];
    for (uint32_t lr = 0; lr < b.cols; ++lr) {
        uint32_t code = f.low_all_codes[off + lr];
        ++z[low.owner[seg_occ(code, LOW_LUT_K)]];
    }
    return z;
}

static void alt_add_matrix(
    std::array<std::array<long double, MAXGPU>, MAXGPU>& mat,
    const std::array<uint32_t, MAXGPU>& r,
    const std::array<uint32_t, MAXGPU>& c,
    int ngpu
) {
    for (int a = 0; a < ngpu; ++a)
        for (int b = 0; b < ngpu; ++b)
            mat[a][b] += (long double)r[a] * c[b] * sizeof(Count);
}

int main() {
    constexpr int NG = 8;
    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost f = build_storage_factor_tables(G_FACTOR);
    StorageLayout l = build_storage_layout(f);

    B300DirectMaskShardHost high = build_b300_direct_mask_shards_w28_precomputed(f, l, NG);
    AltLowShard low = build_alt_low_lpt(f, l, NG);

    std::array<std::array<long double, MAXGPU>, MAXGPU> main_h2l{};
    std::array<std::array<long double, MAXGPU>, MAXGPU> main_l2h{};
    std::array<std::array<long double, MAXGPU>, MAXGPU> block_l2h{};
    for (const auto& b : l.main_blocks) if (b.valid && b.rows && b.cols) {
        auto r = alt_high_rows_per_gpu(f, b, high, NG);
        auto c = alt_low_cols_per_gpu(f, b, low, NG);
        alt_add_matrix(main_h2l, r, c, NG);
        for (int a = 0; a < NG; ++a) for (int d = 0; d < NG; ++d)
            main_l2h[a][d] += (long double)c[a] * r[d] * sizeof(Count);
    }
    for (const auto& b : l.block_blocks) if (b.valid && b.rows && b.cols) {
        auto r = alt_high_rows_per_gpu(f, b, high, NG);
        auto c = alt_low_cols_per_gpu(f, b, low, NG);
        for (int a = 0; a < NG; ++a) for (int d = 0; d < NG; ++d)
            block_l2h[a][d] += (long double)c[a] * r[d] * sizeof(Count);
    }

    auto gib = [](long double x) { return double(x / (1ull << 30)); };
    auto tib = [](long double x) { return double(x / (1ull << 40)); };
    auto gb = [](long double x) { return double(x / 1.0e9L); };

    std::array<long double, NG> high_auth{}, low_auth{};
    for (int g = 0; g < NG; ++g)
        high_auth[g] = (long double)(high.main_count[g] + high.block_count[g]) * sizeof(Count);
    low_auth = {};
    for (const auto& b : l.main_blocks) if (b.valid && b.rows && b.cols) {
        auto c = alt_low_cols_per_gpu(f, b, low, NG);
        for (int g = 0; g < NG; ++g) low_auth[g] += (long double)c[g] * b.rows * sizeof(Count);
    }
    for (const auto& b : l.block_blocks) if (b.valid && b.rows && b.cols) {
        auto c = alt_low_cols_per_gpu(f, b, low, NG);
        for (int g = 0; g < NG; ++g) low_auth[g] += (long double)c[g] * b.rows * sizeof(Count);
    }

    long double total_h2l = 0, total_l2h = 0, off_h2l = 0, off_l2h = 0;
    long double max_pair_h2l = 0, max_pair_l2h = 0;
    std::array<long double, NG> send_h2l{}, recv_h2l{}, send_l2h{}, recv_l2h{};
    for (int a = 0; a < NG; ++a) for (int b = 0; b < NG; ++b) {
        long double x = main_h2l[a][b];
        long double y = main_l2h[a][b] + block_l2h[a][b];
        total_h2l += x; total_l2h += y;
        if (a != b) {
            off_h2l += x; off_l2h += y;
            send_h2l[a] += x; recv_h2l[b] += x;
            send_l2h[a] += y; recv_l2h[b] += y;
            max_pair_h2l = std::max(max_pair_h2l, x);
            max_pair_l2h = std::max(max_pair_l2h, y);
        }
    }

    long double max_port_row = 0;
    for (int g = 0; g < NG; ++g)
        max_port_row = std::max(max_port_row,
            std::max(send_h2l[g], recv_h2l[g]) + std::max(send_l2h[g], recv_l2h[g]));

    long double all_per_row = total_h2l + total_l2h;
    long double off_per_row = off_h2l + off_l2h;
    long double all_per_residue = all_per_row * TARGET_W;
    long double off_per_residue = off_per_row * TARGET_W;
    long double max_port_residue = max_port_row * TARGET_W;
    long double nvlink_port_decimal = 1.8e12L;

    long double high_min = high_auth[0], high_max = high_auth[0];
    long double low_min = low_auth[0], low_max = low_auth[0];
    for (int g = 1; g < NG; ++g) {
        high_min = std::min(high_min, high_auth[g]); high_max = std::max(high_max, high_auth[g]);
        low_min = std::min(low_min, low_auth[g]); low_max = std::max(low_max, low_auth[g]);
    }

    std::cout << std::fixed << std::setprecision(6)
        << "b300-alternating-shard-plan W=" << TARGET_W << " gpus=" << NG
        << " high_auth_min_gib=" << gib(high_min) << " high_auth_max_gib=" << gib(high_max)
        << " low_auth_min_gib=" << gib(low_min) << " low_auth_max_gib=" << gib(low_max) << '\n'
        << "high_to_low_main_total_gib_per_row=" << gib(total_h2l)
        << " high_to_low_offgpu_gib_per_row=" << gib(off_h2l)
        << " high_to_low_offgpu_fraction=" << double(off_h2l / total_h2l)
        << " high_to_low_max_pair_gib=" << gib(max_pair_h2l) << '\n'
        << "low_to_high_main_block_total_gib_per_row=" << gib(total_l2h)
        << " low_to_high_offgpu_gib_per_row=" << gib(off_l2h)
        << " low_to_high_offgpu_fraction=" << double(off_l2h / total_l2h)
        << " low_to_high_max_pair_gib=" << gib(max_pair_l2h) << '\n'
        << "logical_shuffle_tib_per_residue=" << tib(all_per_residue)
        << " physical_offgpu_tib_per_residue=" << tib(off_per_residue)
        << " max_gpu_port_tib_per_residue=" << tib(max_port_residue)
        << " ideal_1p8TBs_port_seconds_per_residue=" << double(max_port_residue / nvlink_port_decimal)
        << '\n'
        << "row_boundary_blocked_free_gib=" << gib((long double)l.block_size * sizeof(Count))
        << " row_boundary_blocked_free_gib_per_gpu_avg=" << gib((long double)l.block_size * sizeof(Count) / NG)
        << " chunked_pair_exchange_possible=1"
        << " required_full_double_buffer=0"
        << '\n';

    for (int g = 0; g < NG; ++g) {
        std::cout << "alt_gpu=" << g
                  << " h2l_send_gib=" << gib(send_h2l[g])
                  << " h2l_recv_gib=" << gib(recv_h2l[g])
                  << " l2h_send_gib=" << gib(send_l2h[g])
                  << " l2h_recv_gib=" << gib(recv_l2h[g]) << '\n';
    }
    return 0;
}
