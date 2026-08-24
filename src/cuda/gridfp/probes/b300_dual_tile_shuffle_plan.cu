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

struct DTLowShard {
    std::vector<uint8_t> owner;
    std::array<uint64_t, MAXGPU> bytes{};
};

static DTLowShard dt_build_low_lpt(const StorageFactorHost& f, const StorageLayout& l, int ngpu) {
    constexpr int S = MAXW + 2;
    constexpr uint32_t NM = 1u << LOW_LUT_K;
    std::vector<uint64_t> w(NM, 0);
    auto add = [&](const StorageBlock& b) {
        if (!b.valid || !b.rows || !b.cols) return;
        for (uint32_t m = 0; m < NM; ++m) {
            size_t ix = size_t(m) * S + b.hs;
            uint64_t n = G_FACTOR.low_mask_off[ix + 1] - G_FACTOR.low_mask_off[ix];
            w[m] += n * uint64_t(b.rows) * sizeof(Count);
        }
    };
    for (const auto& b : l.main_blocks) add(b);
    for (const auto& b : l.block_blocks) add(b);
    std::vector<uint32_t> ord(NM); std::iota(ord.begin(), ord.end(), 0u);
    std::sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) {
        return w[a] != w[b] ? w[a] > w[b] : a < b;
    });
    DTLowShard z; z.owner.assign(NM, 0);
    for (uint32_t m : ord) {
        int g = 0; for (int q = 1; q < ngpu; ++q) if (z.bytes[q] < z.bytes[g]) g = q;
        z.owner[m] = uint8_t(g); z.bytes[g] += w[m];
    }
    return z;
}

using DTMat = std::array<std::array<long double, MAXGPU>, MAXGPU>;

static DTMat dt_matrix(const std::vector<StorageBlock>& blocks,
                       const StorageFactorHost& f,
                       const B300DirectMaskShardHost& high,
                       const DTLowShard& low, int ngpu) {
    DTMat z{};
    for (const auto& b : blocks) if (b.valid && b.rows && b.cols) {
        std::array<uint32_t, MAXGPU> nr{}, nc{};
        uint32_t ho = f.high_all_off[b.he];
        for (uint32_t hr = 0; hr < b.rows; ++hr) ++nr[high.high_owner[ho + hr]];
        uint32_t lo = f.low_all_off[b.hs];
        for (uint32_t lr = 0; lr < b.cols; ++lr) {
            uint32_t code = f.low_all_codes[lo + lr];
            ++nc[low.owner[seg_occ(code, LOW_LUT_K)]];
        }
        for (int i = 0; i < ngpu; ++i) for (int j = 0; j < ngpu; ++j)
            z[i][j] += (long double)nr[i] * nc[j] * sizeof(Count);
    }
    return z;
}

int main() {
    constexpr int NG = 8;
    build_full_dp(); G_FACTOR = build_factor_tables();
    StorageFactorHost f = build_storage_factor_tables(G_FACTOR);
    StorageLayout l = build_storage_layout(f);
    B300DirectMaskShardHost high = build_b300_direct_mask_shards_w28_precomputed(f, l, NG);
    DTLowShard low = dt_build_low_lpt(f, l, NG);

    DTMat mm = dt_matrix(l.main_blocks, f, high, low, NG);
    DTMat bm = dt_matrix(l.block_blocks, f, high, low, NG);

    std::array<long double, NG> hmain{}, lmain{}, hblock{}, lblock{};
    std::array<long double, NG> maincap{}, blockcap{}, totalcap{};
    std::array<long double, NG> h2lsend{}, h2lrecv{}, l2hsend{}, l2hrecv{};
    long double h2loff = 0, l2hoff = 0, maxpairmain = 0, maxpairblock = 0;
    for (int i = 0; i < NG; ++i) {
        for (int j = 0; j < NG; ++j) {
            hmain[i] += mm[i][j]; lmain[i] += mm[j][i];
            hblock[i] += bm[i][j]; lblock[i] += bm[j][i];
            if (i == j) continue;
            h2lsend[i] += mm[i][j]; h2lrecv[j] += mm[i][j];
            l2hsend[j] += mm[i][j] + bm[i][j];
            l2hrecv[i] += mm[i][j] + bm[i][j];
            h2loff += mm[i][j]; l2hoff += mm[i][j] + bm[i][j];
            maxpairmain = std::max(maxpairmain, mm[i][j]);
            maxpairblock = std::max(maxpairblock, bm[i][j]);
        }
        maincap[i] = mm[i][i]; blockcap[i] = bm[i][i];
        for (int j = 0; j < NG; ++j) if (i != j) {
            maincap[i] += std::max(mm[i][j], mm[j][i]);
            blockcap[i] += std::max(bm[i][j], bm[j][i]);
        }
        totalcap[i] = maincap[i] + blockcap[i];
    }

    long double maxcap = *std::max_element(totalcap.begin(), totalcap.end());
    long double maxauth = 0, maxover = 0, maxportrow = 0;
    for (int i = 0; i < NG; ++i) {
        long double auth = std::max(hmain[i] + hblock[i], lmain[i] + lblock[i]);
        maxauth = std::max(maxauth, auth);
        maxover = std::max(maxover, totalcap[i] - auth);
        maxportrow = std::max(maxportrow,
            std::max(h2lsend[i], h2lrecv[i]) + std::max(l2hsend[i], l2hrecv[i]));
    }

    auto gib = [](long double x) { return double(x / (1ull << 30)); };
    auto mib = [](long double x) { return double(x / (1ull << 20)); };
    auto tib = [](long double x) { return double(x / (1ull << 40)); };
    long double offres = (h2loff + l2hoff) * TARGET_W;
    long double portres = maxportrow * TARGET_W;
    long double cap288 = 288.0e9L, cap279 = 279.0e9L;
    long double chunk = 512.0L * (1ull << 20);

    std::cout << std::fixed << std::setprecision(6)
        << "b300-dual-tile-shuffle-plan W=" << TARGET_W << " gpus=" << NG
        << " exchange_rounds=7 simultaneous_pairs_per_round=4"
        << " tile_pack_unpack_bytes=0 full_double_buffer=0"
        << " swap_chunk_mib=" << mib(chunk) << '\n'
        << "h2l_main_offgpu_tib_per_residue=" << tib(h2loff * TARGET_W)
        << " l2h_main_block_offgpu_tib_per_residue=" << tib(l2hoff * TARGET_W)
        << " total_offgpu_tib_per_residue=" << tib(offres)
        << " max_gpu_port_tib_per_residue=" << tib(portres)
        << " ideal_1p8TBs_port_seconds_per_residue=" << double(portres / 1.8e12L)
        << '\n'
        << "pairslot_max_auth_gib=" << gib(maxauth)
        << " pairslot_max_arena_gib=" << gib(maxcap)
        << " pairslot_max_overhead_gib=" << gib(maxover)
        << " plus_512mib_scratch_gib=" << gib(maxcap + chunk)
        << " headroom_288GB_after_arena_chunk_gib=" << gib(cap288 - maxcap - chunk)
        << " headroom_279GB_after_arena_chunk_gib=" << gib(cap279 - maxcap - chunk)
        << " max_directed_main_tile_gib=" << gib(maxpairmain)
        << " max_directed_block_tile_gib=" << gib(maxpairblock)
        << '\n';

    for (int i = 0; i < NG; ++i) {
        std::cout << "dual_gpu=" << i
                  << " high_main_gib=" << gib(hmain[i])
                  << " low_main_gib=" << gib(lmain[i])
                  << " high_block_gib=" << gib(hblock[i])
                  << " low_block_gib=" << gib(lblock[i])
                  << " main_pairslot_gib=" << gib(maincap[i])
                  << " block_pairslot_gib=" << gib(blockcap[i])
                  << " total_pairslot_gib=" << gib(totalcap[i])
                  << " h2l_send_gib=" << gib(h2lsend[i])
                  << " h2l_recv_gib=" << gib(h2lrecv[i])
                  << " l2h_send_gib=" << gib(l2hsend[i])
                  << " l2h_recv_gib=" << gib(l2hrecv[i]) << '\n';
    }

    // Round-robin edge coloring of K8.  Each GPU participates in one pair per
    // round, so all four pair swaps can use their NVLink ports concurrently.
    std::array<int, NG> ring{}; std::iota(ring.begin(), ring.end(), 0);
    for (int r = 0; r < NG - 1; ++r) {
        std::cout << "exchange_round=" << r;
        for (int k = 0; k < NG / 2; ++k)
            std::cout << " pair=" << ring[k] << '-' << ring[NG - 1 - k];
        std::cout << '\n';
        int last = ring[NG - 1];
        for (int k = NG - 1; k >= 2; --k) ring[k] = ring[k - 1];
        ring[1] = last;
    }
    return 0;
}
