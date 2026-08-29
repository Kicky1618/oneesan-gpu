#pragma once

#include "ramstream32_bucket_closure_cross5_rankstream.cuh"
#include "ramstream32_bucket_low_rankchunk32.cuh"
#include <cstdio>

#ifndef P10DC_RANKCHUNK32_FUSED16
#define P10DC_RANKCHUNK32_FUSED16 0
#endif
static_assert(P10DC_RANKCHUNK32_FUSED16 == 0 || P10DC_RANKCHUNK32_FUSED16 == 1,
              "P10DC_RANKCHUNK32_FUSED16 must be 0 or 1");

#ifndef P10DC_RANKCHUNK32_DIRECT3
#define P10DC_RANKCHUNK32_DIRECT3 0
#endif
static_assert(P10DC_RANKCHUNK32_DIRECT3 == 0 || P10DC_RANKCHUNK32_DIRECT3 == 1,
              "P10DC_RANKCHUNK32_DIRECT3 must be 0 or 1");

#ifndef P10DC_RANKCHUNK32_RANKMASK_PROFILE
#define P10DC_RANKCHUNK32_RANKMASK_PROFILE 0
#endif
static_assert(P10DC_RANKCHUNK32_RANKMASK_PROFILE == 0 || P10DC_RANKCHUNK32_RANKMASK_PROFILE == 1,
              "P10DC_RANKCHUNK32_RANKMASK_PROFILE must be 0 or 1");

#ifndef P10DC_RANKCHUNK32_RANKMASK_PROFILE_LOG2
#define P10DC_RANKCHUNK32_RANKMASK_PROFILE_LOG2 0
#endif
static_assert(P10DC_RANKCHUNK32_RANKMASK_PROFILE_LOG2 >= 0 &&
              P10DC_RANKCHUNK32_RANKMASK_PROFILE_LOG2 <= 16,
              "P10DC_RANKCHUNK32_RANKMASK_PROFILE_LOG2 must be in [0,16]");

#if P10DC_RANKCHUNK32_RANKMASK_PROFILE
// Profiling-only rankmask histogram. The production shape proof says only
// {0,1,2,3,5,7} are reachable, but keep 32 bins so sampled real traffic also
// checks that assumption instead of baking it into the profiler.
__device__ unsigned long long D_P10DC_RANKCHUNK32_RANKMASK_PROFILE[32];
// Dynamic sampled warp events at the profiler point: total, all-zero,
// all-nonzero, mixed zero/nonzero. These expose whether an outer rankmask!=0
// guard can skip whole warps or merely introduce divergence.
__device__ unsigned long long D_P10DC_RANKCHUNK32_RANKMASK_WARP_PROFILE[4];

__device__ __forceinline__ bool p10dc_rankchunk32_profile_sample_warp() {
#if P10DC_RANKCHUNK32_RANKMASK_PROFILE_LOG2 == 0
    return true;
#else
    const unsigned linear_thread = unsigned(threadIdx.x) + unsigned(blockDim.x) *
        (unsigned(threadIdx.y) + unsigned(blockDim.y) * unsigned(threadIdx.z));
    const unsigned threads_per_block =
        unsigned(blockDim.x) * unsigned(blockDim.y) * unsigned(blockDim.z);
    const unsigned warps_per_block = (threads_per_block + 31u) >> 5;
    const unsigned long long linear_block = static_cast<unsigned long long>(blockIdx.x) +
        static_cast<unsigned long long>(gridDim.x) *
        (static_cast<unsigned long long>(blockIdx.y) +
         static_cast<unsigned long long>(gridDim.y) * static_cast<unsigned long long>(blockIdx.z));
    const unsigned long long warp_id =
        linear_block * static_cast<unsigned long long>(warps_per_block) +
        static_cast<unsigned long long>(linear_thread >> 5);
    // Use high bits of the multiplicative hash. Low bits of an odd multiply are
    // only a permutation modulo 2^k and would reduce to periodic stride sampling.
    const unsigned long long h = warp_id * 0x9e3779b97f4a7c15ull;
    constexpr unsigned sample_shift =
        64u - unsigned(P10DC_RANKCHUNK32_RANKMASK_PROFILE_LOG2);
    return (h >> sample_shift) == 0ull;
#endif
}

__device__ __forceinline__ void p10dc_rankchunk32_profile_rankmask(uint8_t rankmask) {
    if (!p10dc_rankchunk32_profile_sample_warp()) return;
    const unsigned active = __activemask();
    const unsigned peers = __match_any_sync(active, unsigned(rankmask));
    const unsigned zeros = __ballot_sync(active, rankmask == 0u);
    const unsigned lane = unsigned(threadIdx.x) & 31u;
    const unsigned peer_leader = unsigned(__ffs(int(peers)) - 1);
    const unsigned active_leader = unsigned(__ffs(int(active)) - 1);
    if (lane == peer_leader) {
        atomicAdd(&D_P10DC_RANKCHUNK32_RANKMASK_PROFILE[unsigned(rankmask)],
                  static_cast<unsigned long long>(__popc(peers)));
    }
    if (lane == active_leader) {
        atomicAdd(&D_P10DC_RANKCHUNK32_RANKMASK_WARP_PROFILE[0], 1ull);
        const unsigned kind = zeros == active ? 1u : (zeros == 0u ? 2u : 3u);
        atomicAdd(&D_P10DC_RANKCHUNK32_RANKMASK_WARP_PROFILE[kind], 1ull);
    }
}

static void p10dc_rankchunk32_report_rankmask_profile_devices(int ngpu) {
    std::array<unsigned long long, 32> total{};
    std::array<unsigned long long, 4> warps{};
    int restore = 0;
    ck(cudaGetDevice(&restore), "rankchunk32 profile get device");
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "rankchunk32 profile set device");
        std::array<unsigned long long, 32> one{};
        std::array<unsigned long long, 4> one_warps{};
        ck(cudaMemcpyFromSymbol(one.data(), D_P10DC_RANKCHUNK32_RANKMASK_PROFILE,
                                one.size() * sizeof(unsigned long long)),
           "rankchunk32 profile D2H");
        ck(cudaMemcpyFromSymbol(one_warps.data(), D_P10DC_RANKCHUNK32_RANKMASK_WARP_PROFILE,
                                one_warps.size() * sizeof(unsigned long long)),
           "rankchunk32 warp profile D2H");
        for (size_t i = 0; i < one.size(); ++i) total[i] += one[i];
        for (size_t i = 0; i < one_warps.size(); ++i) warps[i] += one_warps[i];
    }
    ck(cudaSetDevice(restore), "rankchunk32 profile restore device");

    unsigned long long calls = 0, selected = 0, other = 0;
    for (unsigned i = 0; i < total.size(); ++i) {
        calls += total[i];
        unsigned bits = i, pc = 0;
        while (bits) { pc += bits & 1u; bits >>= 1; }
        selected += total[i] * static_cast<unsigned long long>(pc);
        if (i >= 8u) other += total[i];
    }
    const unsigned long long disallowed = total[4] + total[6] + other;
    const double zero_frac = calls ? double(total[0]) / double(calls) : 0.0;
    const double nonzero_frac = calls ? 1.0 - zero_frac : 0.0;
    const double avg_popcount = calls ? double(selected) / double(calls) : 0.0;
    const double all_zero_warp_frac = warps[0] ? double(warps[1]) / double(warps[0]) : 0.0;
    const double all_nonzero_warp_frac = warps[0] ? double(warps[2]) / double(warps[0]) : 0.0;
    const double mixed_warp_frac = warps[0] ? double(warps[3]) / double(warps[0]) : 0.0;
    const double avg_active_lanes = warps[0] ? double(calls) / double(warps[0]) : 0.0;
    constexpr unsigned long long sample_one_in =
        1ull << P10DC_RANKCHUNK32_RANKMASK_PROFILE_LOG2;
    std::fprintf(stderr,
        "rankchunk32_rankmask_profile scope=sampled_process_all_gpus_all_moduli warp_aggregated=1 sample_log2=%d sample_one_in=%llu total=%llu m0=%llu m1=%llu m2=%llu m3=%llu m4=%llu m5=%llu m6=%llu m7=%llu other=%llu disallowed=%llu zero_frac=%.9f nonzero_frac=%.9f avg_popcount=%.9f warp_events=%llu warp_all_zero=%llu warp_all_nonzero=%llu warp_mixed=%llu warp_all_zero_frac=%.9f warp_all_nonzero_frac=%.9f warp_mixed_frac=%.9f avg_active_lanes=%.9f\n",
        P10DC_RANKCHUNK32_RANKMASK_PROFILE_LOG2, sample_one_in,
        calls, total[0], total[1], total[2], total[3], total[4], total[5], total[6], total[7],
        other, disallowed, zero_frac, nonzero_frac, avg_popcount,
        warps[0], warps[1], warps[2], warps[3], all_zero_warp_frac,
        all_nonzero_warp_frac, mixed_warp_frac, avg_active_lanes);
}
#endif

#if P10DC_RANKCHUNK32_FUSED16
#if P10DC_RANKSTREAM_LUT_LDG
__device__ __align__(128) uint16_t
    D_P10DC_RANKCHUNK32_FUSED16[P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE];
#else
__constant__ uint16_t
    D_P10DC_RANKCHUNK32_FUSED16[P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE];
#endif

static std::array<uint16_t, P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE>
p10dc_rankchunk32_fused16_table() {
    std::array<uint16_t, P10DC_CROSS5_STATES * P10DC_RANKSTREAM_LUT_STRIDE> out{};
    for (uint32_t s = 0; s < P10DC_CROSS5_STATES; ++s) {
        for (uint32_t k = 0; k < P10DC_CROSS5_KEYS; ++k) {
            const uint16_t e = uint16_t(p10dc_rankstream_host_entry(k, s));
            const uint16_t meta = uint16_t(p10dc_rankstream_meta_host(k));
            out[size_t(s) * P10DC_RANKSTREAM_LUT_STRIDE + k] = e | (meta << 8);
        }
    }
    return out;
}

static void p10dc_install_rankchunk32_lut() {
    static const auto table = p10dc_rankchunk32_fused16_table();
    ck(cudaMemcpyToSymbol(D_P10DC_RANKCHUNK32_FUSED16, table.data(),
                          table.size() * sizeof(uint16_t)),
       "p10dc rankchunk32 fused16 CROSS5 table");
}

__device__ __forceinline__ uint16_t p10dc_rankchunk32_pair_load(
    uint32_t state, uint32_t chunk
) {
    const size_t ix = size_t(state) * P10DC_RANKSTREAM_LUT_STRIDE + chunk;
#if P10DC_RANKSTREAM_LUT_LDG
    return __ldg(D_P10DC_RANKCHUNK32_FUSED16 + ix);
#else
    return D_P10DC_RANKCHUNK32_FUSED16[ix];
#endif
}
#else
static void p10dc_install_rankchunk32_lut() {
    p10dc_install_rankstream_lut();
}
#endif

// Centralized production decoder. Compact mode uses 7 physical bits for the
// third chunk because bit 23 belongs to the 9-bit prefix. Bytepack mode starts
// the prefix at bit 24, so all three chunk extracts are ordinary byte masks.
__device__ __forceinline__ uint32_t p10dc_rankchunk32_chunk_device(
    uint32_t packed_chunks, uint32_t slot
) {
    if (slot == 0u) return packed_chunks & 0xffu;
    if (slot == 1u) return (packed_chunks >> 8) & 0xffu;
    return (packed_chunks >> 16) & (P10DC_RANKCHUNK32_BYTEPACK ? 0xffu : 0x7fu);
}

__device__ __forceinline__ uint32_t p10dc_cross5_apply_rankchunk32(
    uint32_t chunk, uint32_t& state, const Count* source_row,
    const uint16_t* rank_row, uint32_t& lbase, BkczCrossAccum& sum
) {
#if P10DC_RANKCHUNK32_FUSED16
    const uint16_t pair = p10dc_rankchunk32_pair_load(state, chunk);
    const uint8_t e = uint8_t(pair);
#else
    const uint8_t e = p10dc_rankstream_entry_load(
        size_t(state) * P10DC_RANKSTREAM_LUT_STRIDE + chunk);
#endif
    const uint8_t rankmask = uint8_t(e & P10DC_CROSS5_MASK_MASK);
#if P10DC_RANKCHUNK32_RANKMASK_PROFILE
    p10dc_rankchunk32_profile_rankmask(rankmask);
#endif
#if P10DC_RANKCHUNK32_DIRECT3
    // Exhaustive CROSS5 shape proof over all 25*243 production table inputs:
    // rankmask is always one of {0,1,2,3,5,7}; bits 3 and 4 are unreachable.
    // Avoid the dependent ffs/clear loop and test the only three ordinals that
    // can occur. Keep this behind an A/B switch until measured on B300.
    if (rankmask & 0x01u) {
        const uint16_t source_rank = rank_row[lbase];
        sum = bkcz_cross_add(sum, source_row[uint32_t(source_rank)]);
    }
    if (rankmask & 0x02u) {
        const uint16_t source_rank = rank_row[lbase + 1u];
        sum = bkcz_cross_add(sum, source_row[uint32_t(source_rank)]);
    }
    if (rankmask & 0x04u) {
        const uint16_t source_rank = rank_row[lbase + 2u];
        sum = bkcz_cross_add(sum, source_row[uint32_t(source_rank)]);
    }
#else
    uint8_t pending = rankmask;
    while (pending) {
        const int ordinal = __ffs(int(pending)) - 1;
        pending = uint8_t(pending & uint8_t(pending - 1));
        const uint16_t source_rank = rank_row[lbase + uint32_t(ordinal)];
        sum = bkcz_cross_add(sum, source_row[uint32_t(source_rank)]);
    }
#endif
    if (((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) != 0) return 1u;

#if P10DC_RANKCHUNK32_FUSED16
    const uint8_t meta = uint8_t(pair >> 8);
#else
    const uint8_t meta = p10dc_rankstream_meta_load(chunk);
#endif
    lbase += uint32_t(meta & P10DC_RANKSTREAM_META_LCOUNT_MASK);
    state = uint32_t(
        int(state) + int(meta >> P10DC_RANKSTREAM_META_DELTA_SHIFT)
        - P10DC_RANKSTREAM_META_DELTA_BIAS);
    return 0u;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankchunk32_fast(
    uint32_t packed_chunks, uint32_t depth, const Count* source_row,
    const uint16_t* rank_row
) {
    if (!depth) return BkczCrossAccum(0);
    uint32_t state = depth, lbase = 0;
    BkczCrossAccum sum = 0;

    uint32_t st = p10dc_cross5_apply_rankchunk32(
        p10dc_rankchunk32_chunk_device(packed_chunks, 0u),
        state, source_row, rank_row, lbase, sum);
    if (st == 1u) return sum;

    constexpr int L0 = LOW_LUT_K >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : LOW_LUT_K;
    constexpr int S0 = LOW_LUT_K - L0;
    if constexpr (S0 > 0) {
        st = p10dc_cross5_apply_rankchunk32(
            p10dc_rankchunk32_chunk_device(packed_chunks, 1u),
            state, source_row, rank_row, lbase, sum);
        if (st == 1u) return sum;
        constexpr int L1 = S0 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S0;
        constexpr int S1 = S0 - L1;
        if constexpr (S1 > 0) {
            constexpr int L2 = S1 >= P10DC_CROSS5_CHUNK ? P10DC_CROSS5_CHUNK : S1;
            constexpr int S2 = S1 - L2;
            static_assert(S2 == 0, "rankchunk32 K<=14 must fit in three chunks");
            p10dc_cross5_apply_rankchunk32(
                p10dc_rankchunk32_chunk_device(packed_chunks, 2u),
                state, source_row, rank_row, lbase, sum);
        }
    }
    return sum;
}

__device__ __forceinline__ BkczCrossAccum
p10dc_resolved_low_preimages_cross5_rankchunk32_fixed(
    uint32_t h, uint32_t rank, uint32_t depth, const Count* source_row
) {
    uint32_t packed_chunks = 0;
    const uint16_t* rank_row = nullptr;
    // The low-rankchunk header owns the warp-striped block mapping and the
    // P10DC_RANKCHUNK32_ONESHFL A/B switch. Keep exactly one implementation so
    // packing/block-layout changes cannot drift between the two hot paths.
    p10dc_low_rankchunk32_row_warpstripe(h, rank, packed_chunks, rank_row);
    return p10dc_resolved_low_preimages_cross5_rankchunk32_fast(
        packed_chunks, depth, source_row, rank_row);
}
