#define main pattern10_depth4_direct_reference_main_unused
#include "ramstream32_bucket_reverse_fused_selftest.cu"
#undef main

#include "../ramstream32_bucket_orbit_closure_pattern10_depth4.cuh"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

static bool p10d4_check_depth(
    const char* name,
    const BucketDepth4Host& got,
    const std::vector<uint8_t>& ref
) {
    if (got.count != ref.size()) {
        std::cerr << name << " count mismatch got=" << got.count
                  << " expected=" << ref.size() << '\n';
        return false;
    }
    const size_t expected_bytes = (ref.size() + 1) / 2;
    if (got.bytes() != expected_bytes) {
        std::cerr << name << " byte-size mismatch got=" << got.bytes()
                  << " expected=" << expected_bytes << '\n';
        return false;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        const uint8_t x = got.get(i);
        if (x != ref[i]) {
            std::cerr << name << " depth mismatch q=" << i
                      << " got=" << unsigned(x)
                      << " expected=" << unsigned(ref[i]) << '\n';
            return false;
        }
    }
    return true;
}

template<class T>
static bool p10d4_check_vec(
    const char* name,
    const std::vector<T>& got,
    const std::vector<T>& ref
) {
    if (got == ref) return true;
    std::cerr << name << " mismatch got_size=" << got.size()
              << " expected_size=" << ref.size() << '\n';
    const size_t n = std::min(got.size(), ref.size());
    for (size_t i = 0; i < n; ++i) {
        if (got[i] != ref[i]) {
            std::cerr << name << " first mismatch q=" << i << '\n';
            break;
        }
    }
    return false;
}

static bool p10d4_check_split(
    const ReverseSplit54Host& got,
    const ReverseSplit54Host& ref
) {
    if (got.nblocks != ref.nblocks) {
        std::cerr << "split54 nblocks mismatch got=" << got.nblocks
                  << " expected=" << ref.nblocks << '\n';
        return false;
    }
    bool ok = true;
    ok &= p10d4_check_vec("split.low.nn", got.low.nn, ref.low.nn);
    ok &= p10d4_check_vec("split.low.nr", got.low.nr, ref.low.nr);
    ok &= p10d4_check_vec("split.low.nl", got.low.nl, ref.low.nl);
    ok &= p10d4_check_vec("split.low.nn_off", got.low.nn_off, ref.low.nn_off);
    ok &= p10d4_check_vec("split.low.nr_off", got.low.nr_off, ref.low.nr_off);
    ok &= p10d4_check_vec("split.low.nl_off", got.low.nl_off, ref.low.nl_off);
    ok &= p10d4_check_vec("split.high.nn", got.high.nn, ref.high.nn);
    ok &= p10d4_check_vec("split.high.nr", got.high.nr, ref.high.nr);
    ok &= p10d4_check_vec("split.high.nl", got.high.nl, ref.high.nl);
    ok &= p10d4_check_vec("split.high.nn_off", got.high.nn_off, ref.high.nn_off);
    ok &= p10d4_check_vec("split.high.nr_off", got.high.nr_off, ref.high.nr_off);
    ok &= p10d4_check_vec("split.high.nl_off", got.high.nl_off, ref.high.nl_off);
    return ok;
}

int main() {
    constexpr int W = TARGET_W;
    static_assert(W <= 12, "direct depth4 plan probe intentionally uses small width");

    build_full_dp();
    G_FACTOR = build_factor_tables();
    StorageFactorHost storage = build_storage_factor_tables(G_FACTOR);
    StorageLayout layout = build_storage_layout(storage);

    LowDescHost lowdesc = build_low_descriptors(storage, layout);
    HighDescHost highdesc = build_high_descriptors(storage, layout);
    LowOrbitHost loworbit = build_cpu_low_orbit(storage, layout, lowdesc);
    CpuHighDirectHost highdirect = build_cpu_high_direct(storage, layout, highdesc);
    GpuDirectGatherHost ordinary = build_gpu_direct_gather(layout, lowdesc, loworbit, highdirect);
    GpuDirectCrossGatherHost cross = build_gpu_direct_cross_gather(storage, layout, lowdesc, loworbit, highdirect);
    GpuDirectFusedHost fused = build_gpu_direct_fused_checked(layout, ordinary, cross);
    CpuLowSparseHost lowsparse = build_cpu_low_sparse(storage, layout, lowdesc, loworbit);
    BucketOwnerHost owner = build_bucket_owners(G_FACTOR, storage);
    BucketPhysicalLayoutHost phy = build_bucket_physical_layout(layout, owner);
    (void)phy;
    BucketOrbitStreamsHost bo = build_bucket_orbits(storage, layout, owner, lowsparse, highdirect);
    BucketFusedHost bf = build_bucket_fused(storage, layout, owner, ordinary, cross, fused);

    // Build an independent byte-per-orbit oracle before the direct builders
    // release closure-only host metadata.
    build_bucket_forward_pattern10(layout, bo, bf);

    ReverseLowDescHost rlow = build_reverse_low_descriptors(storage, layout);
    ReverseHighDescHost rhigh = build_reverse_high_descriptors(storage, layout);
    ReverseOrbitHost rlo = build_reverse_orbit(storage, layout, true);
    ReverseOrbitHost rhi = build_reverse_orbit(storage, layout, false);
    ReverseBucketAtomicHost rb = build_reverse_bucket_atomic(
        storage, layout, owner, rlow, rhigh, rlo, rhi);
    ReverseBucketFusedHost rf = build_reverse_bucket_fused_checked(layout, owner, rb);

    ReverseSplit54Host split_ref = build_reverse_split54(layout, rb, false);
    build_reverse_split54_pattern10(layout, bf, split_ref);
    BucketPattern10Depth8Host depth8 =
        build_bucket_pattern10_depth8(layout, bf, bo, split_ref);

    // Production-style direct nibble builders.  These must be exactly the
    // packed representation of the independent depth8 oracle above.
    BucketForwardPattern10Depth4Host direct_f =
        build_bucket_forward_pattern10_depth4_zero(layout, bo, bf);
    BucketReversePattern10Depth4Host direct_r =
        build_bucket_reverse_pattern10_depth4_zero_checked(
            layout, bo, bf, rb, rf);

    bool ok = true;
    ok &= p10d4_check_depth("forward.low.nn", direct_f.low_nn, depth8.f_low_nn);
    ok &= p10d4_check_depth("forward.low.nr", direct_f.low_nr, depth8.f_low_nr);
    ok &= p10d4_check_depth("forward.low.nl", direct_f.low_nl, depth8.f_low_nl);
    ok &= p10d4_check_depth("forward.high.nn", direct_f.high_nn, depth8.f_high_nn);
    ok &= p10d4_check_depth("forward.high.nrnl", direct_f.high_nrnl, depth8.f_high_nrnl);
    ok &= p10d4_check_depth("reverse.low.nn", direct_r.low_nn, depth8.r_low_nn);
    ok &= p10d4_check_depth("reverse.low.nr", direct_r.low_nr, depth8.r_low_nr);
    ok &= p10d4_check_depth("reverse.low.nl", direct_r.low_nl, depth8.r_low_nl);
    ok &= p10d4_check_depth("reverse.high.nn", direct_r.high_nn, depth8.r_high_nn);
    ok &= p10d4_check_depth("reverse.high.nr", direct_r.high_nr, depth8.r_high_nr);
    ok &= p10d4_check_depth("reverse.high.nl", direct_r.high_nl, depth8.r_high_nl);
    ok &= p10d4_check_split(direct_r.split, split_ref);

    size_t expected_depth_bytes = 0;
    auto add_expected = [&](const auto& v) { expected_depth_bytes += (v.size() + 1) / 2; };
    add_expected(depth8.f_low_nn);
    add_expected(depth8.f_low_nr);
    add_expected(depth8.f_low_nl);
    add_expected(depth8.f_high_nn);
    add_expected(depth8.f_high_nrnl);
    add_expected(depth8.r_low_nn);
    add_expected(depth8.r_low_nr);
    add_expected(depth8.r_low_nl);
    add_expected(depth8.r_high_nn);
    add_expected(depth8.r_high_nr);
    add_expected(depth8.r_high_nl);

    const size_t direct_depth_bytes = direct_f.bytes() + direct_r.depth_bytes();
    if (direct_depth_bytes != expected_depth_bytes) {
        std::cerr << "direct depth byte accounting mismatch got=" << direct_depth_bytes
                  << " expected=" << expected_depth_bytes << '\n';
        ok = false;
    }

    if (!ok) return 1;
    std::cout << "pattern10-depth4-direct-plan OK W=" << W
              << " depth8_bytes=" << depth8.bytes()
              << " depth4_bytes=" << direct_depth_bytes
              << " forward_ops=" << direct_f.ops()
              << " reverse_ops=" << direct_r.ops()
              << " split54_bytes=" << direct_r.split.bytes()
              << '\n';
    return 0;
}
