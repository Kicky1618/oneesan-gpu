#define BUCKET_SNAKE_REVERSE_FUSED 1
#define main rankchunk32_exact_rankmask_traffic_driver_main_unused
#include "../../b300/oneesan_cuda_gridfp_b300_bucket_snake_atomic.cu"
#undef main

#include "../ramstream32_bucket_onepass_pattern10_depthcode.hpp"
#include "../ramstream32_bucket_closure_cross5_rankchunk32.cuh"

#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace {

struct Traffic {
    std::array<unsigned long long, 32> mask{};
    unsigned long long resolved_calls = 0;
    unsigned long long chunk_calls = 0;
    unsigned long long halt_calls = 0;
    unsigned long long selected = 0;

    Traffic& operator+=(const Traffic& o) {
        for (size_t i = 0; i < mask.size(); ++i) mask[i] += o.mask[i];
        resolved_calls += o.resolved_calls;
        chunk_calls += o.chunk_calls;
        halt_calls += o.halt_calls;
        selected += o.selected;
        return *this;
    }

    Traffic scaled(unsigned long long n) const {
        Traffic out;
        for (size_t i = 0; i < mask.size(); ++i) out.mask[i] = mask[i] * n;
        out.resolved_calls = resolved_calls * n;
        out.chunk_calls = chunk_calls * n;
        out.halt_calls = halt_calls * n;
        out.selected = selected * n;
        return out;
    }
};

static unsigned popcount32(unsigned x) {
    unsigned n = 0;
    while (x) { n += x & 1u; x >>= 1; }
    return n;
}

static uint32_t packed_chunk_host(uint32_t packed, uint32_t slot) {
    if (slot == 0u) return packed & 0xffu;
    if (slot == 1u) return (packed >> 8) & 0xffu;
    // The legal <=4-digit tail is < 3^4 = 81, so compact and bytepack modes
    // have the same numeric chunk value here.
    return (packed >> 16) & 0xffu;
}

static void add_key_traffic(uint32_t key, uint32_t depth, Traffic& out) {
    if (depth == 0u) return;
    if (depth >= P10DC_CROSS5_STATES) {
        std::cerr << "rankchunk32 exact traffic initial state overflow depth=" << depth << '\n';
        std::exit(701);
    }

    ++out.resolved_calls;
    uint32_t state = depth;
    const uint32_t packed = p10dc_rankchunk32_pack_host(key);
    constexpr uint32_t chunks = uint32_t((LOW_LUT_K + P10DC_CROSS5_CHUNK - 1) /
                                         P10DC_CROSS5_CHUNK);
    static_assert(chunks >= 1u && chunks <= 3u);

    for (uint32_t slot = 0; slot < chunks; ++slot) {
        const uint32_t chunk = packed_chunk_host(packed, slot);
        if (chunk >= P10DC_CROSS5_KEYS || state >= P10DC_CROSS5_STATES) {
            std::cerr << "rankchunk32 exact traffic table overflow slot=" << slot
                      << " chunk=" << chunk << " state=" << state << '\n';
            std::exit(702);
        }
        const uint8_t e = p10dc_rankstream_host_entry(chunk, state);
        const uint8_t rankmask = uint8_t(e & P10DC_CROSS5_MASK_MASK);
        ++out.chunk_calls;
        ++out.mask[rankmask];
        out.selected += popcount32(rankmask);
        if ((e >> P10DC_CROSS5_HALT_SHIFT) & 1u) {
            ++out.halt_calls;
            return;
        }
        const uint8_t meta = p10dc_rankstream_meta_host(chunk);
        state = uint32_t(
            int(state) + int(meta >> P10DC_RANKSTREAM_META_DELTA_SHIFT)
            - P10DC_RANKSTREAM_META_DELTA_BIAS);
    }
}

static unsigned long long disallowed(const Traffic& t) {
    unsigned long long n = t.mask[4] + t.mask[6];
    for (size_t i = 8; i < t.mask.size(); ++i) n += t.mask[i];
    return n;
}

static void print_traffic(const char* scope, const Traffic& t) {
    const double zero_frac = t.chunk_calls ? double(t.mask[0]) / double(t.chunk_calls) : 0.0;
    const double nonzero_frac = t.chunk_calls ? 1.0 - zero_frac : 0.0;
    const double avg_popcount = t.chunk_calls ? double(t.selected) / double(t.chunk_calls) : 0.0;
    const double chunks_per_resolved = t.resolved_calls
        ? double(t.chunk_calls) / double(t.resolved_calls) : 0.0;
    const double halt_frac = t.resolved_calls
        ? double(t.halt_calls) / double(t.resolved_calls) : 0.0;
    unsigned long long other = 0;
    for (size_t i = 8; i < t.mask.size(); ++i) other += t.mask[i];
    std::cout << std::setprecision(12)
              << "rankchunk32_exact_rankmask_traffic scope=" << scope
              << " resolved_calls=" << t.resolved_calls
              << " chunk_calls=" << t.chunk_calls
              << " m0=" << t.mask[0]
              << " m1=" << t.mask[1]
              << " m2=" << t.mask[2]
              << " m3=" << t.mask[3]
              << " m4=" << t.mask[4]
              << " m5=" << t.mask[5]
              << " m6=" << t.mask[6]
              << " m7=" << t.mask[7]
              << " other=" << other
              << " disallowed=" << disallowed(t)
              << " zero_frac=" << zero_frac
              << " nonzero_frac=" << nonzero_frac
              << " avg_popcount=" << avg_popcount
              << " chunks_per_resolved=" << chunks_per_resolved
              << " halt_calls=" << t.halt_calls
              << " halt_frac=" << halt_frac << '\n';
}

}  // namespace

int main() {
    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);
    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost loworbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuHighDirectHost highdirect = build_cpu_high_direct(storage, layout, highdesc);
    GpuDirectGatherHost ordinary = build_gpu_direct_gather(layout, lowdesc, loworbit, highdirect);
    GpuDirectCrossGatherHost cross =
        build_gpu_direct_cross_gather(storage, layout, lowdesc, loworbit, highdirect);
    GpuDirectFusedHost fused = build_gpu_direct_fused_checked(layout, ordinary, cross);
    CpuLowSparseHost lowsparse = build_cpu_low_sparse(storage, layout, lowdesc, loworbit);
    BucketOwnerHost owner = build_bucket_owners(G_FACTOR, storage);
    BucketPhysicalLayoutHost phy = build_bucket_physical_layout(layout, owner);
    BucketOrbitStreamsHost bo =
        build_bucket_orbits(storage, layout, owner, lowsparse, highdirect);
    BucketFusedHost bf = build_bucket_fused(storage, layout, owner, ordinary, cross, fused);

    ReverseLowDescHost rlow = build_reverse_low_descriptors(storage, layout);
    ReverseHighDescHost rhigh = build_reverse_high_descriptors(storage, layout);
    ReverseOrbitHost rlo = build_reverse_orbit(storage, layout, true);
    ReverseOrbitHost rhi = build_reverse_orbit(storage, layout, false);
    ReverseBucketAtomicHost rb =
        build_reverse_bucket_atomic(storage, layout, owner, rlow, rhigh, rlo, rhi);
    ReverseBucketFusedHost rf = build_reverse_bucket_fused_checked(layout, owner, rb);
    auto fh = build_bucket_forward_pattern10_depthcode_placeholder(layout, bo, bf);
    auto rh = build_bucket_reverse_pattern10_depthcode_zero_checked(layout, bo, bf, rb, rf);
    (void)fh;

    constexpr size_t H = size_t(MAXW + 2);
    constexpr size_t D = 16;
    constexpr size_t P = size_t(MAXW + 2);
    std::array<std::array<unsigned long long, H>, BUCKET_NGPU> low_count{};
    std::array<std::array<Traffic, H>, D> depth_height{};

    for (uint32_t fixed = 0; fixed < BUCKET_NGPU; ++fixed) {
        const size_t owner_base = size_t(fixed) * P;
        const uint32_t owner_end = fixed + 1u < BUCKET_NGPU
            ? bf.low_code_off[size_t(fixed + 1u) * P]
            : uint32_t(bf.low_codes.size());
        for (uint32_t h = 0; h < uint32_t(H); ++h) {
            const uint32_t a = bf.low_code_off[owner_base + h];
            const uint32_t b = h + 1u < uint32_t(H)
                ? bf.low_code_off[owner_base + h + 1u] : owner_end;
            if (a > b || b > bf.low_codes.size()) {
                std::cerr << "rankchunk32 exact traffic low-code range invalid fixed=" << fixed
                          << " h=" << h << " a=" << a << " b=" << b << '\n';
                return 703;
            }
            low_count[fixed][h] = b - a;
            for (uint32_t i = a; i < b; ++i) {
                const uint32_t key = gpu_direct_ternary_key_host(bf.low_codes[i], LOW_LUT_K);
                for (uint32_t depth = 1; depth < D; ++depth)
                    add_key_traffic(key, depth, depth_height[depth][h]);
            }
        }
    }

    // Prove the host mapping used above is the same fixed-owner/height column
    // mapping used by the HIGH physical blocks installed on each GPU.
    unsigned long long physical_blocks_checked = 0;
    for (uint32_t high_owner = 0; high_owner < BUCKET_NGPU; ++high_owner) {
        for (uint32_t fixed = 0; fixed < BUCKET_NGPU; ++fixed) {
            const auto& q = phy.pair[high_owner][fixed];
            for (const auto& b : q.main_blocks) {
                if (!b.valid) continue;
                if (b.hs >= H || b.cols != low_count[fixed][b.hs]) {
                    std::cerr << "rankchunk32 exact traffic physical column mismatch high_owner="
                              << high_owner << " fixed=" << fixed
                              << " hs=" << unsigned(b.hs) << " cols=" << b.cols
                              << " expected=" << (b.hs < H ? low_count[fixed][b.hs] : 0ull)
                              << '\n';
                    return 704;
                }
                ++physical_blocks_checked;
            }
        }
    }

    Traffic forward, reverse;
    std::array<unsigned long long, D> forward_depth_entries{};
    std::array<unsigned long long, D> reverse_depth_entries{};
    unsigned long long forward_entries = 0, reverse_entries = 0;

    p10dc_for_each_entry_direct(layout, bo, rh.split, bf, [&](P10DepthCodeEntryView e) {
        if (!e.high) return;
        const uint32_t depth = e.pair == P10DC_NONE_PAIR ? 0u : uint32_t(e.pair & 15u);
        if (depth >= D || e.h >= H) {
            std::cerr << "rankchunk32 exact traffic entry overflow depth=" << depth
                      << " h=" << e.h << '\n';
            std::exit(705);
        }
        if (e.rev) {
            ++reverse_entries;
            ++reverse_depth_entries[depth];
            reverse += depth_height[depth][e.h];
        } else {
            ++forward_entries;
            ++forward_depth_entries[depth];
            forward += depth_height[depth][e.h];
        }
    });

    const unsigned long long expected_forward = bo.high_nn.size() + bo.high_nrnl.size();
    const unsigned long long expected_reverse = rh.split.high.ops();
    if (forward_entries != expected_forward || reverse_entries != expected_reverse) {
        std::cerr << "rankchunk32 exact traffic high-entry mismatch forward=" << forward_entries
                  << '/' << expected_forward << " reverse=" << reverse_entries
                  << '/' << expected_reverse << '\n';
        return 706;
    }
    if (disallowed(forward) || disallowed(reverse)) {
        std::cerr << "rankchunk32 exact traffic rankmask shape violation forward="
                  << disallowed(forward) << " reverse=" << disallowed(reverse) << '\n';
        return 707;
    }

    constexpr unsigned long long forward_launches = (TARGET_W + 1u) / 2u;
    constexpr unsigned long long reverse_launches = TARGET_W / 2u;
    Traffic residue = forward.scaled(forward_launches);
    residue += reverse.scaled(reverse_launches);

    print_traffic("forward_high_sweep", forward);
    print_traffic("reverse_high_sweep", reverse);
    print_traffic("one_residue", residue);

    unsigned long long total_low_codes = 0;
    for (uint32_t fixed = 0; fixed < BUCKET_NGPU; ++fixed)
        for (uint32_t h = 0; h < uint32_t(H); ++h)
            total_low_codes += low_count[fixed][h];

    std::cout << "rankchunk32-exact-rankmask-traffic OK W=" << TARGET_W
              << " low_k=" << LOW_LUT_K
              << " high_k=" << HIGH_LUT_K
              << " fixed_owners=" << BUCKET_NGPU
              << " low_codes=" << total_low_codes
              << " forward_high_entries=" << forward_entries
              << " reverse_high_entries=" << reverse_entries
              << " forward_graph_launches=" << forward_launches
              << " reverse_graph_launches=" << reverse_launches
              << " physical_cols_checked=" << physical_blocks_checked
              << " physical_cols_exact=1"
              << " count_value_independent=1"
              << " modulus_independent=1"
              << " host_exact=1"
              << " runtime_gpu_required=0\n";

    std::cout << "forward_depth_entries=";
    for (size_t d = 0; d < D; ++d) std::cout << (d ? "," : "") << forward_depth_entries[d];
    std::cout << '\n';
    std::cout << "reverse_depth_entries=";
    for (size_t d = 0; d < D; ++d) std::cout << (d ? "," : "") << reverse_depth_entries[d];
    std::cout << '\n';
    return 0;
}
