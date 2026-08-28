#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {

std::vector<int> direct_rotate(const std::vector<int>& input) {
    const int n = static_cast<int>(input.size());
    std::vector<int> output(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        output[static_cast<std::size_t>((i + 1) % n)] =
            input[static_cast<std::size_t>(i)];
    return output;
}

struct Stats {
    int crossings = 0;
    int selected_segments = 0;
    int local_cycles = 0;
};

// CPU model of the CUDA owner-local selection rule.  A slab is selected as a
// cross-owner segment start iff its predecessor belongs to another owner.
// Therefore every maximal equal-owner segment is selected exactly once.  If
// there is no crossing, one canonical cycle leader handles the whole orbit.
std::vector<int> owner_local_rotate(
    const std::vector<int>& input,
    const std::vector<int>& owner,
    Stats& stats
) {
    const int n = static_cast<int>(input.size());
    std::vector<int> output = input;
    std::vector<int> outgoing(static_cast<std::size_t>(n), -1);
    std::vector<unsigned char> selected(static_cast<std::size_t>(n), 0);

    for (int i = 0; i < n; ++i) {
        const int prev = (i + n - 1) % n;
        if (owner[static_cast<std::size_t>(prev)] !=
            owner[static_cast<std::size_t>(i)]) {
            ++stats.crossings;
        }
    }

    if (!stats.crossings) {
        ++stats.local_cycles;
        const int tail = output.back();
        for (int i = n - 1; i > 0; --i)
            output[static_cast<std::size_t>(i)] =
                output[static_cast<std::size_t>(i - 1)];
        output[0] = tail;
        return output;
    }

    // Phase A.  Each selected start owns one maximal segment.  Save its tail
    // and shift only the interior locally, leaving the start as the hole.
    for (int start = 0; start < n; ++start) {
        const int prev = (start + n - 1) % n;
        if (owner[static_cast<std::size_t>(prev)] ==
            owner[static_cast<std::size_t>(start)]) continue;
        ++stats.selected_segments;
        selected[static_cast<std::size_t>(start)] = 1;

        int tail = start;
        while (owner[static_cast<std::size_t>((tail + 1) % n)] ==
               owner[static_cast<std::size_t>(start)]) {
            tail = (tail + 1) % n;
            if (tail == start) {
                std::cerr << "wrapped cross-owner segment\n";
                std::exit(10);
            }
        }
        outgoing[static_cast<std::size_t>(start)] =
            input[static_cast<std::size_t>(tail)];

        int cur = tail;
        while (cur != start) {
            const int p = (cur + n - 1) % n;
            output[static_cast<std::size_t>(cur)] =
                input[static_cast<std::size_t>(p)];
            cur = p;
        }
    }

    // Phase B.  A segment's saved tail goes to the next segment start.
    for (int start = 0; start < n; ++start) {
        if (!selected[static_cast<std::size_t>(start)]) continue;
        int tail = start;
        while (owner[static_cast<std::size_t>((tail + 1) % n)] ==
               owner[static_cast<std::size_t>(start)])
            tail = (tail + 1) % n;
        const int dst = (tail + 1) % n;
        output[static_cast<std::size_t>(dst)] =
            outgoing[static_cast<std::size_t>(start)];
    }
    return output;
}

bool check_case(const std::vector<int>& owner, std::uint64_t id) {
    const int n = static_cast<int>(owner.size());
    std::vector<int> input(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        input[static_cast<std::size_t>(i)] =
            static_cast<int>(1000 + 17 * i + (id & 255));

    Stats stats;
    const auto got = owner_local_rotate(input, owner, stats);
    const auto expected = direct_rotate(input);
    if (got != expected ||
        (stats.crossings && stats.selected_segments != stats.crossings) ||
        (!stats.crossings && stats.local_cycles != 1)) {
        std::cerr << "FAIL id=" << id
                  << " n=" << n
                  << " crossings=" << stats.crossings
                  << " selected=" << stats.selected_segments
                  << " local_cycles=" << stats.local_cycles << '\n';
        return false;
    }
    return true;
}

} // namespace

int main() {
    std::uint64_t cases = 0;

    for (int ngpu = 2; ngpu <= 4; ++ngpu) {
        for (int n = 2; n <= 9; ++n) {
            std::uint64_t total = 1;
            for (int i = 0; i < n; ++i)
                total *= static_cast<std::uint64_t>(ngpu);
            for (std::uint64_t code = 0; code < total; ++code) {
                std::uint64_t z = code;
                std::vector<int> owner(static_cast<std::size_t>(n));
                for (int i = 0; i < n; ++i) {
                    owner[static_cast<std::size_t>(i)] =
                        static_cast<int>(z % static_cast<std::uint64_t>(ngpu));
                    z /= static_cast<std::uint64_t>(ngpu);
                }
                if (!check_case(owner, code)) return 1;
                ++cases;
            }
        }
    }

    std::mt19937_64 rng(0x1618);
    for (int trial = 0; trial < 250000; ++trial) {
        const int n = 2 + static_cast<int>(rng() % 27);
        const int ngpu = 2 + static_cast<int>(rng() % 7);
        std::vector<int> owner(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i)
            owner[static_cast<std::size_t>(i)] =
                static_cast<int>(rng() % static_cast<std::uint64_t>(ngpu));
        if (!check_case(owner, static_cast<std::uint64_t>(trial))) return 2;
        ++cases;
    }

    std::cout << "ALL_OK"
              << " owner_local_segment_selection=1"
              << " cases=" << cases
              << " max_cycle_len=28"
              << " max_ngpu=8"
              << " one_start_per_maximal_segment=1"
              << " all_local_one_leader=1"
              << " phase_a_phase_b_exact=1\n";
    return 0;
}
