#include <cstdint>
#include <iostream>

namespace {
constexpr std::uint64_t CUDA_CONSTANT_BUDGET = 64u * 1024u;

// gridfp_reduced_production_device.cuh
constexpr std::uint64_t CORE_CHOOSE = 29u * 29u * 8u;
constexpr std::uint64_t CORE_PRIMITIVE = 29u * 30u * 8u;
constexpr std::uint64_t CORE_MOTZKIN = 29u * 30u * 8u;
constexpr std::uint64_t CORE_SECTOR_OFFSET = 17u * 8u;
constexpr std::uint64_t CORE_SECTOR_MAIN = 16u * 8u;
constexpr std::uint64_t CORE_SECTOR_PRIMITIVE = 16u * 8u;
constexpr std::uint64_t CORE_TOTAL = CORE_CHOOSE + CORE_PRIMITIVE + CORE_MOTZKIN +
                                     CORE_SECTOR_OFFSET + CORE_SECTOR_MAIN +
                                     CORE_SECTOR_PRIMITIVE;

// gridfp_reduced_production_runtime_fastdiv64.cuh
constexpr std::uint64_t FASTDIV_PRIMITIVE_MAGIC = 29u * 8u;
constexpr std::uint64_t FASTDIV_OWNER_MAGIC = 11u * 14u * 8u;
constexpr std::uint64_t FASTDIV_TURN_MAGIC = 11u * 14u * 8u;
constexpr std::uint64_t FASTDIV_TOTAL = FASTDIV_PRIMITIVE_MAGIC +
                                        FASTDIV_OWNER_MAGIC + FASTDIV_TURN_MAGIC;

// Cumulative fast stack: fixed52 owner mode.
constexpr std::uint64_t RANK_SECTOR_OFFSET = 1199u * 4u;
constexpr std::uint64_t RANK_OUTER_GROUP_SIZE = 99u * 4u;
constexpr std::uint64_t RANK_OUTER_GROUP_PREFIX = 99u * 8u;
constexpr std::uint64_t OWNER_FIXED52_MAGIC = 11u * 8u;
constexpr std::uint64_t OWNER_LOCAL_SECTOR_END = 1100u * 4u;
constexpr std::uint64_t TURN_LOCAL_SECTOR_END = 550u * 4u;
constexpr std::uint64_t STACK_TABLE_TOTAL = RANK_SECTOR_OFFSET +
    RANK_OUTER_GROUP_SIZE + RANK_OUTER_GROUP_PREFIX + OWNER_FIXED52_MAGIC +
    OWNER_LOCAL_SECTOR_END + TURN_LOCAL_SECTOR_END;

constexpr std::uint64_t KNOWN_TOTAL = CORE_TOTAL + FASTDIV_TOTAL + STACK_TABLE_TOTAL;
constexpr std::uint64_t HEADROOM = CUDA_CONSTANT_BUDGET - KNOWN_TOTAL;
static_assert(CORE_TOTAL == 21040u);
static_assert(FASTDIV_TOTAL == 2696u);
static_assert(STACK_TABLE_TOTAL == 12672u);
static_assert(KNOWN_TOTAL == 36408u);
static_assert(KNOWN_TOTAL < CUDA_CONSTANT_BUDGET);
static_assert(HEADROOM == 29128u);
}

int main() {
    std::cout << "gridfp-runtime-constant-budget-proof OK"
              << " cuda_constant_budget=" << CUDA_CONSTANT_BUDGET
              << " core_bytes=" << CORE_TOTAL
              << " fastdiv_bytes=" << FASTDIV_TOTAL
              << " stack_table_bytes=" << STACK_TABLE_TOTAL
              << " known_total_bytes=" << KNOWN_TOTAL
              << " headroom_bytes=" << HEADROOM
              << " fixed52_owner=1 under_budget=1\n";
    return 0;
}
