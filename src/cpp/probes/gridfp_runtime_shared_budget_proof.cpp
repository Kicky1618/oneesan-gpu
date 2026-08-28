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
constexpr std::uint64_t INDEX_CACHE_BYTES_PER_GROUP = 128;

constexpr std::uint64_t PACKED_KEYS = 2 * GROUPS * PAIRS * PACKED_KEY_BYTES;
constexpr std::uint64_t UNPACKED_KEYS = 2 * GROUPS * PAIRS * DEVICE_KEY_BYTES;
constexpr std::uint64_t VALUES = GROUPS * PAIRS * 4;
constexpr std::uint64_t CONTEXTS = GROUPS * CONTEXT_BYTES;
constexpr std::uint64_t COUNTERS = 2 * GROUPS * 4;
constexpr std::uint64_t EDGES = GROUPS * EDGE_BYTES_PER_GROUP;
constexpr std::uint64_t INDEX_CACHE = GROUPS * INDEX_CACHE_BYTES_PER_GROUP;

constexpr std::uint64_t BASELINE = UNPACKED_KEYS + VALUES + CONTEXTS + COUNTERS;
constexpr std::uint64_t FAST_NO_INDEX =
    PACKED_KEYS + VALUES + CONTEXTS + COUNTERS + EDGES;
constexpr std::uint64_t FAST_INDEX = FAST_NO_INDEX + INDEX_CACHE;
constexpr std::int64_t FAST_INDEX_DELTA =
    std::int64_t(FAST_INDEX) - std::int64_t(BASELINE);

static_assert(PACKED_KEYS == 10240);
static_assert(UNPACKED_KEYS == 20480);
static_assert(VALUES == 2560);
static_assert(CONTEXTS == 768);
static_assert(COUNTERS == 256);
static_assert(EDGES == 2560);
static_assert(INDEX_CACHE == 4096);
static_assert(BASELINE == 24064);
static_assert(FAST_NO_INDEX == 16384);
static_assert(FAST_INDEX == 20480);
static_assert(FAST_INDEX_DELTA == -3584);
}

int main() {
    std::cout << "gridfp-runtime-shared-budget-proof OK"
              << " groups=" << GROUPS
              << " baseline_bytes=" << BASELINE
              << " fast_no_index_bytes=" << FAST_NO_INDEX
              << " index_cache_bytes=" << INDEX_CACHE
              << " fast_index_bytes=" << FAST_INDEX
              << " fast_index_delta_bytes=" << FAST_INDEX_DELTA
              << " packed_key_bytes=" << PACKED_KEY_BYTES
              << " device_key_bytes=" << DEVICE_KEY_BYTES
              << " context_bytes=" << CONTEXT_BYTES
              << " exact=1\n";
    return 0;
}
