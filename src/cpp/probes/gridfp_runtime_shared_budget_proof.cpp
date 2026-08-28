#include <cstdint>
#include <iostream>

namespace {
constexpr std::uint64_t WARPS = 8;
constexpr std::uint64_t SUBGROUPS = 4;
constexpr std::uint64_t PAIRS = 20;
constexpr std::uint64_t GROUPS = WARPS * SUBGROUPS;

constexpr std::uint64_t PACKED_KEY_BYTES = 8;
constexpr std::uint64_t DEVICE_KEY_BYTES = 16;
constexpr std::uint64_t CONTEXT_BYTES = 24;
constexpr std::uint64_t EDGE_BYTES_PER_GROUP = 80;

constexpr std::uint64_t PACKED_KEYS = 2 * GROUPS * PAIRS * PACKED_KEY_BYTES;
constexpr std::uint64_t UNPACKED_KEYS = 2 * GROUPS * PAIRS * DEVICE_KEY_BYTES;
constexpr std::uint64_t VALUES = GROUPS * PAIRS * 4;
constexpr std::uint64_t CONTEXTS = GROUPS * CONTEXT_BYTES;
constexpr std::uint64_t COUNTERS = 2 * GROUPS * 4;
constexpr std::uint64_t EDGES = GROUPS * EDGE_BYTES_PER_GROUP;

constexpr std::uint64_t BASELINE = UNPACKED_KEYS + VALUES + CONTEXTS + COUNTERS;
constexpr std::uint64_t FAST_NO_INDEX =
    PACKED_KEYS + VALUES + CONTEXTS + COUNTERS + EDGES;

constexpr std::uint64_t index_bytes(int buckets) {
    return GROUPS * 2u * std::uint64_t(buckets);
}
constexpr std::uint64_t fast_bytes(int buckets) {
    return FAST_NO_INDEX + index_bytes(buckets);
}
constexpr std::int64_t delta_bytes(int buckets) {
    return std::int64_t(fast_bytes(buckets)) - std::int64_t(BASELINE);
}

static_assert(PACKED_KEYS == 10240);
static_assert(UNPACKED_KEYS == 20480);
static_assert(VALUES == 2560);
static_assert(CONTEXTS == 768);
static_assert(COUNTERS == 256);
static_assert(EDGES == 2560);
static_assert(BASELINE == 24064);
static_assert(FAST_NO_INDEX == 16384);
static_assert(index_bytes(16) == 1024);
static_assert(index_bytes(32) == 2048);
static_assert(index_bytes(64) == 4096);
static_assert(fast_bytes(16) == 17408);
static_assert(fast_bytes(32) == 18432);
static_assert(fast_bytes(64) == 20480);
static_assert(delta_bytes(16) == -6656);
static_assert(delta_bytes(32) == -5632);
static_assert(delta_bytes(64) == -3584);
}

int main() {
    std::cout << "gridfp-runtime-shared-budget-proof OK"
              << " groups=" << GROUPS
              << " baseline_bytes=" << BASELINE
              << " fast_no_index_bytes=" << FAST_NO_INDEX
              << " index16_bytes=" << index_bytes(16)
              << " fast16_bytes=" << fast_bytes(16)
              << " delta16_bytes=" << delta_bytes(16)
              << " index32_bytes=" << index_bytes(32)
              << " fast32_bytes=" << fast_bytes(32)
              << " delta32_bytes=" << delta_bytes(32)
              << " index64_bytes=" << index_bytes(64)
              << " fast64_bytes=" << fast_bytes(64)
              << " delta64_bytes=" << delta_bytes(64)
              << " packed_key_bytes=" << PACKED_KEY_BYTES
              << " device_key_bytes=" << DEVICE_KEY_BYTES
              << " context_bytes=" << CONTEXT_BYTES
              << " exact=1\n";
    return 0;
}
