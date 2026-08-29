#pragma once

#include <cstdint>

// Linkable correctness-stage API.  Each function mutates one stationary GPU
// vector in place and returns zero on success.  These wrappers intentionally
// favor independent translation units over one enormous include chain: every
// stage owns/installs the small constant tables and temporary primitive LUTs
// needed by its kernel family.

extern "C" int oneesan_two_cell_forward2_stage(
    std::uint32_t* d_values,
    int W,
    int start,
    int requested_max_cluster,
    std::uint64_t shared_limit_bytes,
    std::uint32_t mod);

extern "C" int oneesan_two_cell_right_boundary_stage(
    std::uint32_t* d_values,
    int W,
    int requested_max_cluster,
    std::uint64_t shared_limit_bytes,
    std::uint32_t mod);

extern "C" int oneesan_two_cell_reverse2_stage(
    std::uint32_t* d_values,
    int W,
    int start,
    int requested_max_cluster,
    std::uint64_t shared_limit_bytes,
    std::uint32_t mod);

extern "C" int oneesan_two_cell_left_boundary_stage(
    std::uint32_t* d_values,
    int W,
    int requested_max_cluster,
    std::uint64_t shared_limit_bytes,
    std::uint32_t mod);
