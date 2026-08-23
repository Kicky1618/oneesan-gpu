#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../oneesan_cuda_gridfp_ramstream32_factorized_bidesc_compact.cu"
#undef RAMSTREAM_BIDESC_COMPACT_NO_MAIN
#include "../ramstream32_high_orbit.cuh"

struct ForcedPeak {
    Code main_states = 0;
    Code block_states = 0;
    uint32_t group = 0;
    size_t bytes = 0;
};

static ForcedPeak forced_peak(int W, int hi, int lo) {
    std::vector<int> fp = window_candidates(W, hi, lo);
    if (fp.size() >= 31) std::exit(260);
    uint32_t ng = 1u << fp.size();
    ForcedPeak z;
    for (uint32_t g = 0; g < ng; ++g) {
        uint32_t mf, mo, bf, bo;
        window_masks(W, hi, lo, fp, g, mf, mo, bf, bo);
        auto ms = make_spec(W, mf, mo);
        auto ds = make_spec(W - 1, bf, bo);
        size_t bytes = size_t(ms.size + ds.size) * sizeof(Count);
        if (bytes > z.bytes) {
            z = {ms.size, ds.size, g, bytes};
        }
    }
    return z;
}

static double gib(long double bytes) {
    return double(bytes / (long double(1ULL << 30)));
}
static double gb(long double bytes) {
    return double(bytes / 1.0e9L);
}

int main() {
    constexpr int W = TARGET_W;
    constexpr int NG = 8;
    static_assert(LOW_LUT_K + HIGH_LUT_K + 1 == W);

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout logical = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, logical);
    HighDescHost highdesc = build_high_descriptors(storage, logical);
    HighOrbitHost highorbit = build_high_orbit(storage, logical);

    Code main_n = H_DP[W][1];
    Code block_n = H_DP[W - 1][1];
    Code main_chunk = (main_n + NG - 1) / NG;
    Code block_chunk = (block_n + NG - 1) / NG;
    long double auth_bytes = (long double(main_chunk) + block_chunk) * sizeof(Count);

    ForcedPeak high = forced_peak(W, W - 1, LOW_LUT_K + 1);
    ForcedPeak low = forced_peak(W, LOW_LUT_K, 1);
    size_t scratch_peak = std::max(high.bytes, low.bytes);

    long double desc_bytes =
        (long double(lowdesc.main_desc.size()) + lowdesc.block_desc.size()
       + highdesc.main_desc.size() + highdesc.block_desc.size()) * sizeof(uint32_t);
    long double orbit_bytes = (long double)highorbit.rec.size() * sizeof(uint64_t);
    long double mask_bytes =
        (long double(G_FACTOR.low_mask_codes.size()) + G_FACTOR.low_mask_off.size()
       + G_FACTOR.high_mask_codes.size() + G_FACTOR.high_mask_off.size()) * sizeof(uint32_t);
    long double compact_meta = desc_bytes + orbit_bytes + mask_bytes;

    // Old LOW LUT from the legacy B300 source: dense 4^LOW uint32 rank plus
    // approximately one 8-byte entry per legal LOW code and offsets.  Report
    // dense rank separately because compact descriptors remove it completely.
    long double old_low_dense = (long double(size_t(1) << (2 * LOW_LUT_K))) * sizeof(uint32_t);

    long double need_compact = auth_bytes + scratch_peak + compact_meta;
    long double cap_288 = 288.0e9L;
    long double cap_279 = 279.0e9L;

    std::cout << std::fixed << std::setprecision(3)
        << "b300-hbm-capacity W=" << W
        << " main_states=" << main_n
        << " block_states=" << block_n
        << " auth_total_gib=" << gib((long double(main_n + block_n) * sizeof(Count))
        << " auth_shard_gib=" << gib(auth_bytes)
        << '\n'
        << "forced_high groups=" << (1u << LOW_LUT_K)
        << " max_main=" << high.main_states
        << " max_block=" << high.block_states
        << " scratch_gib=" << gib(high.bytes)
        << " group=" << high.group
        << '\n'
        << "forced_low groups=" << (1u << HIGH_LUT_K)
        << " max_main=" << low.main_states
        << " max_block=" << low.block_states
        << " scratch_gib=" << gib(low.bytes)
        << " group=" << low.group
        << '\n'
        << "scratch_peak_gib=" << gib(scratch_peak)
        << " descriptors_mib=" << double(desc_bytes / (1 << 20))
        << " high_orbit_mib=" << double(orbit_bytes / (1 << 20))
        << " mask_tables_mib=" << double(mask_bytes / (1 << 20))
        << " compact_meta_mib=" << double(compact_meta / (1 << 20))
        << " legacy_low_dense_rank_mib=" << double(old_low_dense / (1 << 20))
        << '\n'
        << "compact_need_gib=" << gib(need_compact)
        << " compact_need_gb=" << gb(need_compact)
        << " headroom_288GB_gib=" << gib(cap_288 - need_compact)
        << " headroom_279GB_gib=" << gib(cap_279 - need_compact)
        << '\n';

    for (int reserve_gib : {2, 4, 8}) {
        long double r = long double(reserve_gib) * (1ULL << 30);
        std::cout << "reserve_check reserve_gib=" << reserve_gib
                  << " fits_288GB=" << (need_compact + r <= cap_288)
                  << " fits_279GB=" << (need_compact + r <= cap_279)
                  << '\n';
    }
    return 0;
}
