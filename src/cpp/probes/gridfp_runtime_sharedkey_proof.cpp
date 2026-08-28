#include <array>
#include <cstdint>
#include <iostream>

namespace {

constexpr int MAX_W = 28;
constexpr int WARPS_PER_BLOCK = 8;
constexpr int SUBGROUPS_PER_WARP = 4;
constexpr int MAX_PAIRS = 20;
constexpr std::uint64_t BLOCKED_BIT = 1ULL << 63;
constexpr std::uint64_t MATE_MASK = BLOCKED_BIT - 1ULL;

struct DeviceKeyModel {
    std::uint64_t mate = 0;
    std::uint8_t blocked = 0;
};

constexpr std::uint64_t encode(std::uint64_t mate, bool blocked) {
    return mate | (blocked ? BLOCKED_BIT : 0ULL);
}

constexpr std::uint64_t decode_mate(std::uint64_t packed) {
    return packed & MATE_MASK;
}

constexpr bool decode_blocked(std::uint64_t packed) {
    return (packed & BLOCKED_BIT) != 0;
}

static_assert(2 * MAX_W < 63);
static_assert(sizeof(DeviceKeyModel) == 16);
static_assert(sizeof(std::uint64_t) == 8);

} // namespace

int main() {
    std::uint64_t cases = 0;
    std::uint64_t x = 0x9e3779b97f4a7c15ULL;
    for (int W = 1; W <= MAX_W; ++W) {
        const int bits = 2 * W;
        const std::uint64_t mask = (1ULL << bits) - 1ULL;
        const std::array<std::uint64_t, 5> boundary = {
            0ULL, 1ULL, mask >> 1, mask - 1ULL, mask};
        for (const std::uint64_t mate : boundary) {
            for (int blocked = 0; blocked <= 1; ++blocked) {
                const std::uint64_t packed = encode(mate, blocked != 0);
                if (decode_mate(packed) != mate ||
                    decode_blocked(packed) != (blocked != 0) ||
                    (packed & ~(mask | BLOCKED_BIT)) != 0) {
                    std::cerr << "shared-key boundary mismatch W=" << W
                              << " mate=" << mate
                              << " blocked=" << blocked << '\n';
                    return 2;
                }
                ++cases;
            }
        }
        for (int i = 0; i < 4096; ++i) {
            x ^= x << 7;
            x ^= x >> 9;
            x ^= x << 8;
            const std::uint64_t mate = x & mask;
            const bool blocked = (x >> 62) & 1ULL;
            const std::uint64_t packed = encode(mate, blocked);
            if (decode_mate(packed) != mate || decode_blocked(packed) != blocked) {
                std::cerr << "shared-key randomized mismatch W=" << W
                          << " iteration=" << i << '\n';
                return 3;
            }
            ++cases;
        }
    }

    constexpr int entries =
        2 * WARPS_PER_BLOCK * SUBGROUPS_PER_WARP * MAX_PAIRS;
    constexpr int unpacked_bytes = entries * int(sizeof(DeviceKeyModel));
    constexpr int packed_bytes = entries * int(sizeof(std::uint64_t));
    constexpr int saved_bytes = unpacked_bytes - packed_bytes;
    static_assert(entries == 1280);
    static_assert(unpacked_bytes == 20480);
    static_assert(packed_bytes == 10240);
    static_assert(saved_bytes == 10240);

    std::cout << "gridfp-runtime-sharedkey-proof OK"
              << " cases=" << cases
              << " max_w=" << MAX_W
              << " mate_bits_max=" << 2 * MAX_W
              << " blocked_bit=63"
              << " devicekey_model_bytes=" << sizeof(DeviceKeyModel)
              << " packed_key_bytes=" << sizeof(std::uint64_t)
              << " shared_key_entries_per_block=" << entries
              << " unpacked_shared_key_bytes_per_block=" << unpacked_bytes
              << " packed_shared_key_bytes_per_block=" << packed_bytes
              << " shared_key_bytes_saved_per_block=" << saved_bytes
              << " shared_key_reduction_pct=50"
              << " roundtrip_exact=1\n";
    return 0;
}
