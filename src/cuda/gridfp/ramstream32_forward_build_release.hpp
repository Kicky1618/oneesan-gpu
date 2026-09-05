#pragma once

#include <cstddef>
#include <iostream>
#include <vector>

// Release construction-only forward metadata as soon as its bucket-local
// replacement has been materialized.  These helpers intentionally preserve
// scalar topology fields and only return vector capacity to the allocator.
template<class T>
static size_t bsn_forward_release_vector(std::vector<T>& v) {
    const size_t bytes = v.capacity() * sizeof(T);
    std::vector<T>().swap(v);
    return bytes;
}

static void bsn_forward_release_highdesc(HighDescHost& h) {
    size_t bytes = 0;
    bytes += bsn_forward_release_vector(h.main_desc);
    bytes += bsn_forward_release_vector(h.block_desc);
    std::cerr << "forward_build release_stage=highdesc released_mib="
              << double(bytes) / double(1 << 20) << '\n';
}

static void bsn_forward_release_lowdesc_orbit(LowDescHost& d, LowOrbitHost& o) {
    size_t bytes = 0;
    bytes += bsn_forward_release_vector(d.main_desc);
    bytes += bsn_forward_release_vector(d.block_desc);
    bytes += bsn_forward_release_vector(o.rec);
    std::cerr << "forward_build release_stage=lowdesc-loworbit released_mib="
              << double(bytes) / double(1 << 20) << '\n';
}

static void bsn_forward_release_bucket_orbit_inputs(
    CpuLowSparseHost& low, CpuHighDirectHost& high
) {
    size_t bytes = 0;
    bytes += bsn_forward_release_vector(low.orbit_ops);
    bytes += bsn_forward_release_vector(low.nn_orbit_ops);
    bytes += bsn_forward_release_vector(low.nr_orbit_ops);
    bytes += bsn_forward_release_vector(low.nl_orbit_ops);
    bytes += bsn_forward_release_vector(low.local_closure_ops);
    bytes += bsn_forward_release_vector(low.cross_closure_ops);
    bytes += bsn_forward_release_vector(low.nn_orbit_off);
    bytes += bsn_forward_release_vector(low.nr_orbit_off);
    bytes += bsn_forward_release_vector(low.nl_orbit_off);
    bytes += bsn_forward_release_vector(low.local_closure_off);
    bytes += bsn_forward_release_vector(low.cross_closure_off);
    bytes += bsn_forward_release_vector(low.high_cross_rank);

    bytes += bsn_forward_release_vector(high.orbit_ops.nn);
    bytes += bsn_forward_release_vector(high.orbit_ops.nrnl);
    bytes += bsn_forward_release_vector(high.closure_ops.block);
    bytes += bsn_forward_release_vector(high.closure_ops.cross);
    bytes += bsn_forward_release_vector(high.orbit_off.nn);
    bytes += bsn_forward_release_vector(high.orbit_off.nrnl);
    bytes += bsn_forward_release_vector(high.closure_off.block);
    bytes += bsn_forward_release_vector(high.closure_off.cross);

    std::cerr << "forward_build release_stage=bucket-orbit-inputs released_mib="
              << double(bytes) / double(1 << 20) << '\n';
}

static void bsn_forward_release_bucket_fused_inputs(
    GpuDirectGatherHost& ordinary, GpuDirectCrossGatherHost& cross,
    GpuDirectFusedHost& fused
) {
    size_t bytes = 0;
    bytes += bsn_forward_release_vector(ordinary.low_dst);
    bytes += bsn_forward_release_vector(ordinary.low_src);
    bytes += bsn_forward_release_vector(ordinary.low_off);
    bytes += bsn_forward_release_vector(ordinary.low_cross);
    bytes += bsn_forward_release_vector(ordinary.low_cross_off);
    bytes += bsn_forward_release_vector(ordinary.high_dst);
    bytes += bsn_forward_release_vector(ordinary.high_src);
    bytes += bsn_forward_release_vector(ordinary.high_off);

    bytes += bsn_forward_release_vector(cross.low_dst);
    bytes += bsn_forward_release_vector(cross.low_op);
    bytes += bsn_forward_release_vector(cross.low_off);
    bytes += bsn_forward_release_vector(cross.high_dst);
    bytes += bsn_forward_release_vector(cross.high_op);
    bytes += bsn_forward_release_vector(cross.high_off);
    bytes += bsn_forward_release_vector(cross.high_codes);
    bytes += bsn_forward_release_vector(cross.low_codes);
    bytes += bsn_forward_release_vector(cross.high_direct);
    bytes += bsn_forward_release_vector(cross.low_direct);

    bytes += bsn_forward_release_vector(fused.low_dst);
    bytes += bsn_forward_release_vector(fused.low_off);
    bytes += bsn_forward_release_vector(fused.high_dst);
    bytes += bsn_forward_release_vector(fused.high_off);

    std::cerr << "forward_build release_stage=bucket-fused-inputs released_mib="
              << double(bytes) / double(1 << 20) << '\n';
}
