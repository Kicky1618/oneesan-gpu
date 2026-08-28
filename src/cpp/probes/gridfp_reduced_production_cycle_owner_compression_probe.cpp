#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <utility>

namespace {

using Rank = std::uint64_t;
using Mask = std::uint32_t;

constexpr int W = 28;
constexpr int K = 13;
constexpr int L = 15;
constexpr int O = 13;
constexpr int NGPU = 8;
constexpr int BATCHES = 8;
constexpr int MAX_SEGMENTS_PER_OWNER = 14;

Rank choose[32][32]{};
Rank primitive[32][32]{};
Rank group_size[O + 1]{};
Rank group_prefix[O + 1]{};
Rank owner_states[NGPU]{};
std::uint8_t owner_lut[1 << O]{};
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

Rank state_group_size(int outer_ones) {
    Rank total = 0;
    for (int local = 0; local <= L; ++local) {
        const int occupied = outer_ones + local;
        if (!(occupied & 1)) continue;
        total += (choose[L][local] +
                  (local ? choose[L - 2][local - 1] : 0)) *
                 primitive[occupied][1];
    }
    return total;
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

    Rank total_states = 0;
    for (int r = 0; r <= O; ++r) {
        group_size[r] = state_group_size(r);
        group_prefix[r] = r
            ? group_prefix[r - 1] + choose[O][r - 1] * group_size[r - 1]
            : 0;
        total_states += choose[O][r] * group_size[r];
    }
    if (total_states != 473397057701ULL) {
        std::cerr << "production state dimension mismatch\n";
        std::exit(2);
    }

    for (Mask outer = 0; outer < (Mask(1) << O); ++outer) {
        const int r = __builtin_popcount(outer);
        const Rank base =
            group_prefix[r] + support_rank(outer, O, r) * group_size[r];
        const Rank midpoint = base + group_size[r] / 2;
        int owner = static_cast<int>(
            (__uint128_t(midpoint) * NGPU) / total_states);
        if (owner >= NGPU) owner = NGPU - 1;
        owner_lut[outer] = static_cast<std::uint8_t>(owner);
        owner_states[owner] += group_size[r];
    }
}

int owner_of_support(Mask support, bool reverse) {
    const Mask outer = reverse
        ? ((support >> 15) & 8191u)
        : (support & 8191u);
    return owner_lut[outer];
}

struct Stats {
    Rank scratch_words[NGPU][BATCHES]{};
    Rank original_segments[NGPU][BATCHES]{};
    Rank compressed_entries[NGPU][BATCHES]{};
    Rank class_entries[NGPU][BATCHES][W + 1][MAX_SEGMENTS_PER_OWNER + 1]{};
    Rank local_entries[NGPU]{};
    Rank main_cycles = 0;
    Rank blocked_cycles = 0;
    int max_segments_per_owner_cycle = 0;
};

void process_main_cycle(Stats& stats, int period, bool reverse) {
    int occupied = 0;
    Mask support = 0;
    for (int i = 1; i <= W; ++i) {
        occupied += necklace_bits[i];
        if (necklace_bits[i]) support |= Mask(1) << (i - 1);
    }
    if (!(occupied & 1)) return;
    ++stats.main_cycles;
    if (period <= 1) return;

    const int step = reverse ? 15 : 13;
    int route_owner[W]{};
    Mask cur = support;
    for (int hop = 0; hop < period; ++hop) {
        route_owner[hop] = owner_of_support(cur, reverse);
        cur = rotate_bits(cur, W, step);
    }

    bool same = true;
    for (int hop = 1; hop < period; ++hop)
        same = same && route_owner[hop] == route_owner[0];
    if (same) {
        ++stats.local_entries[route_owner[0]];
        return;
    }

    int segments_per_owner[NGPU]{};
    int hop = 0;
    while (hop < period) {
        const int owner = route_owner[hop];
        int len = 1;
        while (hop + len < period && route_owner[hop + len] == owner) ++len;
        ++segments_per_owner[owner];
        hop += len;
    }
    if (route_owner[0] == route_owner[period - 1])
        --segments_per_owner[route_owner[0]];

    const Rank pc = primitive[occupied][1];
    const int batch = main_batch(support);
    for (int gpu = 0; gpu < NGPU; ++gpu) {
        const int segments = segments_per_owner[gpu];
        if (!segments) continue;
        if (segments < 1 || segments > MAX_SEGMENTS_PER_OWNER) {
            std::cerr << "main segment multiplicity out of range\n";
            std::exit(3);
        }
        stats.max_segments_per_owner_cycle =
            std::max(stats.max_segments_per_owner_cycle, segments);
        ++stats.compressed_entries[gpu][batch];
        stats.original_segments[gpu][batch] += segments;
        stats.scratch_words[gpu][batch] += Rank(segments) * pc;
        ++stats.class_entries[gpu][batch][occupied][segments];
    }
}

void generate_binary_necklaces(
    int t,
    int period,
    Stats& forward,
    Stats& reverse
) {
    if (t > W) {
        if (W % period == 0) {
            process_main_cycle(forward, period, false);
            process_main_cycle(reverse, period, true);
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

void enumerate_blocked_cycles(Stats& forward, Stats& reverse) {
    for (Mask a = 0; a < (Mask(1) << K); ++a) {
        for (Mask b = 0; b <= a; ++b) {
            const int free_occupied =
                __builtin_popcount(a) + __builtin_popcount(b);
            if (free_occupied & 1) continue;
            ++forward.blocked_cycles;
            ++reverse.blocked_cycles;
            if (a == b) continue;

            const int owner_a = owner_lut[a];
            const int owner_b = owner_lut[b];
            const int occupied = free_occupied + 1;
            const Rank pc = primitive[occupied][1];
            const int batch = blocked_batch(a, b, occupied);

            for (Stats* stats : {&forward, &reverse}) {
                if (owner_a == owner_b) {
                    ++stats->local_entries[owner_a];
                    continue;
                }
                for (const int owner : {owner_a, owner_b}) {
                    ++stats->compressed_entries[owner][batch];
                    ++stats->original_segments[owner][batch];
                    stats->scratch_words[owner][batch] += pc;
                    ++stats->class_entries[owner][batch][occupied][1];
                }
            }
        }
    }
}

void report(const char* direction, const Stats& stats) {
    Rank words = 0;
    Rank original_segments = 0;
    Rank compressed_entries = 0;
    Rank local_entries = 0;
    for (int gpu = 0; gpu < NGPU; ++gpu) {
        local_entries += stats.local_entries[gpu];
        for (int batch = 0; batch < BATCHES; ++batch) {
            words += stats.scratch_words[gpu][batch];
            original_segments += stats.original_segments[gpu][batch];
            compressed_entries += stats.compressed_entries[gpu][batch];
        }
    }

    if (words != 409769189454ULL ||
        original_segments != 117118478ULL ||
        local_entries != 5910700ULL)
        std::exit(4);

    const Rank total_list_entries = compressed_entries + local_entries;
    std::cout << "cycle-owner-compression-direction"
              << " direction=" << direction
              << " logical_peer_values=" << words
              << " original_cross_segments=" << original_segments
              << " compressed_cross_entries=" << compressed_entries
              << " local_entries=" << local_entries
              << " total_list_entries=" << total_list_entries
              << " compression="
              << double(original_segments + local_entries) /
                     double(total_list_entries)
              << " max_segments_per_owner_cycle="
              << stats.max_segments_per_owner_cycle
              << '\n';

    constexpr double GiB = double(1ULL << 30);
    constexpr double B300_GiB = 288e9 / GiB;
    double worst_peak = 0.0;
    int worst_gpu = -1;

    for (int gpu = 0; gpu < NGPU; ++gpu) {
        Rank max_scratch_words = 0;
        Rank owner_list_entries = stats.local_entries[gpu];
        for (int batch = 0; batch < BATCHES; ++batch) {
            max_scratch_words =
                std::max(max_scratch_words, stats.scratch_words[gpu][batch]);
            owner_list_entries += stats.compressed_entries[gpu][batch];
        }

        const double state_gib = double(owner_states[gpu]) * 4.0 / GiB;
        // Forward/reverse list distributions are symmetric for W=28/K=13.
        const double dual_list_gib = double(owner_list_entries) * 8.0 / GiB;
        const double scratch_gib = double(max_scratch_words) * 4.0 / GiB;
        // Descriptor-free execution needs only tiny (batch, occupied,
        // segment-count) prefix metadata.  Count two uint64 arrays.
        const double metadata_gib =
            double(BATCHES * (W + 1) *
                   (MAX_SEGMENTS_PER_OWNER + 1) * 2 * sizeof(Rank)) /
            GiB;
        const double peak_gib =
            state_gib + dual_list_gib + scratch_gib + metadata_gib;
        if (peak_gib > worst_peak) {
            worst_peak = peak_gib;
            worst_gpu = gpu;
        }

        std::cout << "cycle-owner-compression-owner"
                  << " direction=" << direction
                  << " gpu=" << gpu
                  << " list_entries=" << owner_list_entries
                  << " dual_list_MiB=" << dual_list_gib * 1024.0
                  << " max_scratch_GiB=" << scratch_gib
                  << " descriptor_GiB=0"
                  << " peak_GiB=" << peak_gib
                  << " B300_headroom_GiB=" << (B300_GiB - peak_gib)
                  << '\n';
    }

    std::cout << "cycle-owner-compression-peak"
              << " direction=" << direction
              << " worst_gpu=" << worst_gpu
              << " peak_GiB=" << worst_peak
              << " B300_headroom_GiB=" << (B300_GiB - worst_peak)
              << " descriptor_bytes=0"
              << " scratch_allocator_atomics=0"
              << '\n';
}

} // namespace

int main() {
    initialize_tables();
    Stats forward{};
    Stats reverse{};
    necklace_bits[0] = 0;
    generate_binary_necklaces(1, 1, forward, reverse);
    enumerate_blocked_cycles(forward, reverse);

    std::cout << std::fixed << std::setprecision(9);
    report("forward", forward);
    report("reverse", reverse);

    if (forward.max_segments_per_owner_cycle != 13 ||
        reverse.max_segments_per_owner_cycle != 13)
        return 5;

    std::cout << "ALL_OK production_cycle_owner_compression=1\n";
    return 0;
}
