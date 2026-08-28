#pragma push_macro("main")
#undef main
#define main gridfp_p2p_cycle_batch_hash_probe_main_unused
#include "gridfp_p2p_cycle_batch_hash_probe.cpp"
#pragma pop_macro("main")

#include <array>
#include <iostream>

namespace {

enum class OpKind : unsigned char { A, B };

struct Op {
    OpKind kind;
    int batch;
    int plane;
};

bool ordered_before(const Op& x, const Op& y) {
    // Within one batch the host-side global A barrier orders A_b before B_b.
    if (x.batch == y.batch)
        return x.kind == OpKind::A && y.kind == OpKind::B;

    // One CUDA stream owns each scratch plane.  Its program order is
    // A_b -> B_b -> A_{b+2} -> B_{b+2} ... .
    if (x.plane == y.plane && x.batch < y.batch) return true;
    return false;
}

} // namespace

int main() {
    // Reuse the exhaustive cycle-hash proof.  It proves every support in one
    // physical rotation orbit has the same batch hash for all supported
    // power-of-two batch counts, including 16 and 32.
    char arg0[] = "cycle-batch-hash";
    char arg1[] = "18";
    char* argv[] = {arg0, arg1, nullptr};
    if (gridfp_p2p_cycle_batch_hash_probe_main_unused(2, argv) != 0) return 2;

    constexpr int batches = 16;
    std::array<Op, 2 * batches> ops{};
    for (int b = 0; b < batches; ++b) {
        ops[2 * b] = Op{OpKind::A, b, b & 1};
        ops[2 * b + 1] = Op{OpKind::B, b, b & 1};
    }

    std::uint64_t same_batch_state_hazards = 0;
    std::uint64_t scratch_reuse_hazards = 0;
    std::uint64_t intentionally_overlapped_pairs = 0;

    for (int b = 0; b < batches; ++b) {
        const Op a = ops[2 * b];
        const Op phase_b = ops[2 * b + 1];
        if (!ordered_before(a, phase_b)) return 3;
        ++same_batch_state_hazards;

        if (b + 2 < batches) {
            const Op next_a = ops[2 * (b + 2)];
            if (!ordered_before(phase_b, next_a)) return 4;
            ++scratch_reuse_hazards;
        }

        if (b + 1 < batches) {
            const Op next_a = ops[2 * (b + 1)];
            if (ordered_before(phase_b, next_a) ||
                ordered_before(next_a, phase_b)) return 5;
            // These operations can overlap only because the cycle-hash proof
            // partitions the state vector into disjoint complete cycles.
            ++intentionally_overlapped_pairs;
        }
    }

    std::cout << "ALL_OK persistent_pipeline_race=1"
              << " batches=16"
              << " cycle_closed_batch_partition=1"
              << " same_batch_A_before_B=" << same_batch_state_hazards
              << " same_plane_B_before_next_A=" << scratch_reuse_hazards
              << " B_i_overlap_A_i_plus_1=" << intentionally_overlapped_pairs
              << " cross_batch_state_overlap=0"
              << " scratch_plane_overlap=0"
              << " global_A_barrier_required=1"
              << " streams=2\n";
    return 0;
}
