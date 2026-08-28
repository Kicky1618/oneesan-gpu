#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <utility>

namespace {

using Rank = std::uint64_t;
using Mask = std::uint32_t;

constexpr int W = 28;
constexpr int K = 13;
constexpr int L = K + 2;
constexpr int O = W - L;
constexpr int NGPU = 8;
constexpr int BATCHES = 8;
constexpr unsigned DESCRIPTOR_BYTES = 32;

std::array<std::array<Rank, 32>, 32> choose{};
std::array<std::array<Rank, 32>, 32> primitive{};
std::array<Rank, O + 1> group_size{};
std::array<Rank, O + 1> group_prefix{};
std::array<unsigned char, 1 << O> owner_lut{};
std::array<Rank, NGPU> owner_states{};

struct DirectionStats {
    Rank words[NGPU][BATCHES]{};
    Rank segments[NGPU][BATCHES]{};
    Rank local_entries[NGPU]{};
    Rank main_cycles = 0;
    Rank blocked_cycles = 0;
};

int necklace_bits[W + 1]{};

Mask rotate_bits(Mask x, int len, int shift) {
    shift %= len;
    if (shift < 0) shift += len;
    if (!shift) return x;
    const Mask mask = (Mask(1) << len) - 1u;
    return ((x << shift) | (x >> (len - shift))) & mask;
}

Rank support_rank(Mask mask, int len, int ones) {
    Rank rank = 0;
    int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        if (((mask >> pos) & 1u) == 0) continue;
        rank += choose[len - pos - 1][left];
        --left;
    }
    return rank;
}

Mask mix32(Mask x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

int main_batch(Mask support) {
    Mask h = Mask(__builtin_popcount(support)) * 0x9e3779b1u;
    constexpr std::pair<int, Mask> terms[] = {
        {1, 0x85ebca6bu},
        {3, 0xc2b2ae35u},
        {5, 0x27d4eb2fu},
        {7, 0x165667b1u},
    };
    for (const auto [distance, coefficient] : terms) {
        h ^= Mask(__builtin_popcount(
                 support & rotate_bits(support, W, distance))) *
             coefficient;
    }
    return int(mix32(h) & (BATCHES - 1));
}

int blocked_batch(Mask a, Mask b, int occupied) {
    const Mask lo = std::min(a, b);
    const Mask hi = std::max(a, b);
    Mask h = lo * 0x9e3779b1u;
    h ^= hi * 0x85ebca6bu;
    h ^= Mask(occupied) * 0xc2b2ae35u;
    return int(mix32(h) & (BATCHES - 1));
}

int owner_of_support(Mask support, bool reverse) {
    const Mask outer = reverse ? ((support >> 15) & 8191u)
                               : (support & 8191u);
    return owner_lut[outer];
}

void process_main_necklace(
    DirectionStats& stats,
    int period,
    bool reverse
) {
    int occupied = 0;
    Mask support = 0;
    for (int i = 1; i <= W; ++i) {
        occupied += necklace_bits[i];
        if (necklace_bits[i]) support |= Mask(1) << (i - 1);
    }
    if (!(occupied & 1)) return;

    const Rank primitive_count = primitive[occupied][1];
    const int batch = main_batch(support);
    const int step = reverse ? 15 : 13;
    const int first_owner = owner_of_support(support, reverse);
    int previous_owner = first_owner;
    int crossings = 0;
    Mask current = rotate_bits(support, W, step);

    for (int hop = 1; hop < period; ++hop) {
        const int current_owner = owner_of_support(current, reverse);
        if (current_owner != previous_owner) {
            stats.words[previous_owner][batch] += primitive_count;
            ++stats.segments[previous_owner][batch];
            ++crossings;
        }
        previous_owner = current_owner;
        current = rotate_bits(current, W, step);
    }
    if (previous_owner != first_owner) {
        stats.words[previous_owner][batch] += primitive_count;
        ++stats.segments[previous_owner][batch];
        ++crossings;
    }

    if (!crossings && period > 1) ++stats.local_entries[first_owner];
    ++stats.main_cycles;
}

void generate_binary_necklaces(
    int t,
    int period,
    DirectionStats& forward,
    DirectionStats& reverse
) {
    if (t > W) {
        if (W % period == 0) {
            process_main_necklace(forward, period, false);
            process_main_necklace(reverse, period, true);
        }
        return;
    }

    necklace_bits[t] = necklace_bits[t - period];
    generate_binary_necklaces(t + 1, period, forward, reverse);
    for (int bit = necklace_bits[t - period] + 1; bit <= 1; ++bit) {
        necklace_bits[t] = bit;
        generate_binary_necklaces(t + 1, t, forward, reverse);
    }
}

void initialize_tables() {
    for (int n = 0; n < 32; ++n) {
        choose[n][0] = choose[n][n] = 1;
        for (int k = 1; k < n; ++k)
            choose[n][k] = choose[n - 1][k - 1] + choose[n - 1][k];
    }

    primitive[0][0] = 1;
    for (int rem = 1; rem < 32; ++rem) {
        for (int h = 0; h < 31; ++h) {
            primitive[rem][h] =
                primitive[rem - 1][h + 1] +
                (h ? primitive[rem - 1][h - 1] : 0);
        }
    }
}

Rank initialize_owner_lut() {
    Rank total_states = 0;
    for (int outer_ones = 0; outer_ones <= O; ++outer_ones) {
        Rank size = 0;
        for (int local_ones = 0; local_ones <= L; ++local_ones) {
            const int occupied = outer_ones + local_ones;
            if (!(occupied & 1)) continue;
            const Rank supports =
                choose[L][local_ones] +
                (local_ones ? choose[L - 2][local_ones - 1] : 0);
            size += supports * primitive[occupied][1];
        }
        group_size[outer_ones] = size;
        group_prefix[outer_ones] =
            outer_ones
                ? group_prefix[outer_ones - 1] +
                      choose[O][outer_ones - 1] *
                      group_size[outer_ones - 1]
                : 0;
        total_states += choose[O][outer_ones] * size;
    }

    for (Mask outer = 0; outer < (Mask(1) << O); ++outer) {
        const int outer_ones = __builtin_popcount(outer);
        const Rank base =
            group_prefix[outer_ones] +
            support_rank(outer, O, outer_ones) * group_size[outer_ones];
        const Rank midpoint = base + group_size[outer_ones] / 2;
        int owner = int((__uint128_t(midpoint) * NGPU) / total_states);
        if (owner >= NGPU) owner = NGPU - 1;
        owner_lut[outer] = static_cast<unsigned char>(owner);
        owner_states[owner] += group_size[outer_ones];
    }
    return total_states;
}

void enumerate_blocked_cycles(
    DirectionStats& forward,
    DirectionStats& reverse
) {
    for (Mask a = 0; a < (Mask(1) << K); ++a) {
        for (Mask b = 0; b <= a; ++b) {
            const int free_occupied = __builtin_popcount(a) + __builtin_popcount(b);
            if (free_occupied & 1) continue;

            if (a == b) {
                ++forward.blocked_cycles;
                ++reverse.blocked_cycles;
                continue;
            }

            const int owner_a = owner_lut[a];
            const int owner_b = owner_lut[b];
            const int occupied = free_occupied + 1;
            const Rank primitive_count = primitive[occupied][1];
            const int batch = blocked_batch(a, b, occupied);

            if (owner_a != owner_b) {
                for (DirectionStats* stats : {&forward, &reverse}) {
                    stats->words[owner_a][batch] += primitive_count;
                    ++stats->segments[owner_a][batch];
                    stats->words[owner_b][batch] += primitive_count;
                    ++stats->segments[owner_b][batch];
                }
            } else {
                ++forward.local_entries[owner_a];
                ++reverse.local_entries[owner_a];
            }
            ++forward.blocked_cycles;
            ++reverse.blocked_cycles;
        }
    }
}

void print_direction(const char* name, const DirectionStats& stats) {
    Rank total_words = 0;
    Rank total_segments = 0;
    Rank total_local = 0;

    for (int gpu = 0; gpu < NGPU; ++gpu) {
        Rank owner_segments = 0;
        for (int batch = 0; batch < BATCHES; ++batch) {
            total_words += stats.words[gpu][batch];
            total_segments += stats.segments[gpu][batch];
            owner_segments += stats.segments[gpu][batch];
            std::cout
                << "persistent-exact-cell"
                << " direction=" << name
                << " gpu=" << gpu
                << " batch=" << batch
                << " scratch_words=" << stats.words[gpu][batch]
                << " scratch_GiB="
                << double(stats.words[gpu][batch]) * 4.0 / double(1ULL << 30)
                << " segments=" << stats.segments[gpu][batch]
                << " descriptor_MiB="
                << double(stats.segments[gpu][batch]) * DESCRIPTOR_BYTES /
                       double(1ULL << 20)
                << '\n';
        }
        total_local += stats.local_entries[gpu];
        const Rank list_entries = owner_segments + stats.local_entries[gpu];
        std::cout
            << "persistent-exact-owner"
            << " direction=" << name
            << " gpu=" << gpu
            << " cross_segments=" << owner_segments
            << " local_entries=" << stats.local_entries[gpu]
            << " list_entries=" << list_entries
            << " list_MiB=" << double(list_entries) * 4.0 / double(1ULL << 20)
            << '\n';
    }

    std::cout
        << "persistent-exact-direction"
        << " direction=" << name
        << " logical_peer_values=" << total_words
        << " logical_peer_GiB="
        << double(total_words) * 4.0 / double(1ULL << 30)
        << " cross_segments=" << total_segments
        << " local_entries=" << total_local
        << " main_cycles=" << stats.main_cycles
        << " blocked_cycles=" << stats.blocked_cycles
        << '\n';
}

} // namespace

int main() {
    initialize_tables();
    const Rank total_states = initialize_owner_lut();
    if (total_states != 473397057701ULL) return 2;

    DirectionStats forward{};
    DirectionStats reverse{};
    const auto start = std::chrono::steady_clock::now();
    necklace_bits[0] = 0;
    generate_binary_necklaces(1, 1, forward, reverse);
    enumerate_blocked_cycles(forward, reverse);
    const double enumerate_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();

    std::cout << std::fixed << std::setprecision(9);
    print_direction("forward", forward);
    print_direction("reverse", reverse);

    if (forward.main_cycles != 4793492ULL ||
        forward.blocked_cycles != 16781312ULL ||
        reverse.main_cycles != forward.main_cycles ||
        reverse.blocked_cycles != forward.blocked_cycles) return 3;

    Rank forward_words = 0;
    Rank reverse_words = 0;
    Rank forward_segments = 0;
    Rank reverse_segments = 0;
    Rank forward_local = 0;
    Rank reverse_local = 0;
    for (int gpu = 0; gpu < NGPU; ++gpu) {
        forward_local += forward.local_entries[gpu];
        reverse_local += reverse.local_entries[gpu];
        for (int batch = 0; batch < BATCHES; ++batch) {
            forward_words += forward.words[gpu][batch];
            reverse_words += reverse.words[gpu][batch];
            forward_segments += forward.segments[gpu][batch];
            reverse_segments += reverse.segments[gpu][batch];
        }
    }

    if (forward_words != 409769189454ULL || reverse_words != forward_words ||
        forward_segments != 117118478ULL ||
        reverse_segments != forward_segments ||
        forward_local != 5910700ULL || reverse_local != forward_local) return 4;

    constexpr double B300_GIB = 288e9 / double(1ULL << 30);
    double worst_peak_gib = 0.0;
    int worst_gpu = -1;

    for (int gpu = 0; gpu < NGPU; ++gpu) {
        Rank max_scratch_words = 0;
        Rank max_descriptors = 0;
        Rank dual_list_entries =
            forward.local_entries[gpu] + reverse.local_entries[gpu];
        for (int batch = 0; batch < BATCHES; ++batch) {
            max_scratch_words = std::max({
                max_scratch_words,
                forward.words[gpu][batch],
                reverse.words[gpu][batch],
            });
            max_descriptors = std::max({
                max_descriptors,
                forward.segments[gpu][batch],
                reverse.segments[gpu][batch],
            });
            dual_list_entries +=
                forward.segments[gpu][batch] + reverse.segments[gpu][batch];
        }

        const double state_gib =
            double(owner_states[gpu]) * 4.0 / double(1ULL << 30);
        const double dual_list_gib =
            double(dual_list_entries) * 4.0 / double(1ULL << 30);
        const double scratch_gib =
            double(max_scratch_words) * 4.0 / double(1ULL << 30);
        const double descriptor_gib =
            double(max_descriptors) * DESCRIPTOR_BYTES / double(1ULL << 30);
        const double peak_gib =
            state_gib + dual_list_gib + scratch_gib + descriptor_gib;
        if (peak_gib > worst_peak_gib) {
            worst_peak_gib = peak_gib;
            worst_gpu = gpu;
        }

        std::cout
            << "persistent-exact-peak"
            << " gpu=" << gpu
            << " state_GiB=" << state_gib
            << " dual_list_MiB=" << dual_list_gib * 1024.0
            << " max_scratch_GiB=" << scratch_gib
            << " max_descriptor_MiB=" << descriptor_gib * 1024.0
            << " peak_GiB=" << peak_gib
            << " B300_headroom_GiB=" << (B300_GIB - peak_gib)
            << '\n';
    }

    const double headroom_gib = B300_GIB - worst_peak_gib;
    if (worst_gpu != 0 || headroom_gib < 20.0) return 5;

    std::cout
        << "ALL_OK gridfp_persistent_exact_plan=1"
        << " W=28 K=13 ngpu=8 batches=8"
        << " states=" << total_states
        << " logical_peer_values=" << forward_words
        << " logical_peer_GiB="
        << double(forward_words) * 4.0 / double(1ULL << 30)
        << " cross_segments=" << forward_segments
        << " local_entries_per_direction=" << forward_local
        << " worst_gpu=" << worst_gpu
        << " worst_peak_GiB=" << worst_peak_gib
        << " B300_GiB=" << B300_GIB
        << " B300_headroom_GiB=" << headroom_gib
        << " enumerate_ms=" << enumerate_ms
        << " runtime_support_scan_passes=0"
        << " runtime_count_passes=0"
        << " exact=OK\n";
    return 0;
}
