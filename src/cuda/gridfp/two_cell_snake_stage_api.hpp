#pragma once

#include <cstdint>

// Linkable correctness-stage API. Each function mutates one stationary GPU
// vector in place. Runtime variants choose the smallest executable cluster up
// to requested_max_cluster. Forced variants require exactly cluster=2/4/8 and
// are intended to exercise remote DSM even at small widths.

extern "C" int oneesan_two_cell_forward2_stage(
    std::uint32_t* d_values, int W, int start, int requested_max_cluster,
    std::uint64_t shared_limit_bytes, std::uint32_t mod);
extern "C" int oneesan_two_cell_forward2_stage_forced(
    std::uint32_t* d_values, int W, int start, int forced_cluster,
    std::uint64_t shared_limit_bytes, std::uint32_t mod);

extern "C" int oneesan_two_cell_right_boundary_stage(
    std::uint32_t* d_values, int W, int requested_max_cluster,
    std::uint64_t shared_limit_bytes, std::uint32_t mod);
extern "C" int oneesan_two_cell_right_boundary_stage_forced(
    std::uint32_t* d_values, int W, int forced_cluster,
    std::uint64_t shared_limit_bytes, std::uint32_t mod);

extern "C" int oneesan_two_cell_reverse2_stage(
    std::uint32_t* d_values, int W, int start, int requested_max_cluster,
    std::uint64_t shared_limit_bytes, std::uint32_t mod);
extern "C" int oneesan_two_cell_reverse2_stage_forced(
    std::uint32_t* d_values, int W, int start, int forced_cluster,
    std::uint64_t shared_limit_bytes, std::uint32_t mod);

extern "C" int oneesan_two_cell_left_boundary_stage(
    std::uint32_t* d_values, int W, int requested_max_cluster,
    std::uint64_t shared_limit_bytes, std::uint32_t mod);
extern "C" int oneesan_two_cell_left_boundary_stage_forced(
    std::uint32_t* d_values, int W, int forced_cluster,
    std::uint64_t shared_limit_bytes, std::uint32_t mod);
