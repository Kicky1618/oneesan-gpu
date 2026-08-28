#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

namespace {

std::vector<int> direct_rotate(const std::vector<int>& input) {
    const int n = static_cast<int>(input.size());
    std::vector<int> output(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        output[static_cast<std::size_t>((i + 1) % n)] = input[static_cast<std::size_t>(i)];
    return output;
}

struct Segment {
    std::vector<int> positions;
    int owner = -1;
    int outgoing = 0;
};

// Rotate one scalar lane of a support cycle by maximal equal-owner segments.
// For a primitive slab, every lane executes this same permutation
// independently.  Each segment saves only its final value, shifts its internal
// values locally, then receives the previous segment's saved value at its
// first position.  Thus exactly one cross-owner write is required per owner
// boundary and no remote read is necessary.
std::vector<int> segmented_rotate(
    const std::vector<int>& input,
    const std::vector<int>& owner,
    int& cross_edges,
    int& remote_writes
) {
    const int n = static_cast<int>(input.size());
    std::vector<int> output = input;
    cross_edges = 0;
    for (int i = 0; i < n; ++i)
        cross_edges += owner[static_cast<std::size_t>(i)] !=
                       owner[static_cast<std::size_t>((i + 1) % n)];

    if (!cross_edges) {
        const int tail = output.back();
        for (int i = n - 1; i > 0; --i)
            output[static_cast<std::size_t>(i)] =
                output[static_cast<std::size_t>(i - 1)];
        output[0] = tail;
        remote_writes = 0;
        return output;
    }

    // Start exactly at an owner boundary so no maximal segment wraps around
    // the end of the linearized route.
    int start = 0;
    while (owner[static_cast<std::size_t>(start)] ==
           owner[static_cast<std::size_t>((start + n - 1) % n)]) {
        ++start;
    }

    std::vector<Segment> segments;
    int cur = start;
    do {
        Segment segment;
        segment.owner = owner[static_cast<std::size_t>(cur)];
        do {
            segment.positions.push_back(cur);
            cur = (cur + 1) % n;
        } while (cur != start &&
                 owner[static_cast<std::size_t>(cur)] == segment.owner);
        segments.push_back(std::move(segment));
    } while (cur != start);

    // Phase A: every owner can perform these operations locally.  Saving all
    // outgoing values before overwriting anything makes Phase B independent
    // of the order in which peer writes are issued.
    for (auto& segment : segments)
        segment.outgoing = output[static_cast<std::size_t>(segment.positions.back())];
    for (const auto& segment : segments) {
        for (int j = static_cast<int>(segment.positions.size()) - 1; j > 0; --j) {
            output[static_cast<std::size_t>(segment.positions[static_cast<std::size_t>(j)])] =
                output[static_cast<std::size_t>(segment.positions[static_cast<std::size_t>(j - 1)])];
        }
    }

    // Phase B: previous-owner -> next-owner first position.  These are the
    // only cross-device writes.  Because adjacent segments have different
    // owners, their count is exactly the cyclic owner-boundary count.
    for (int j = 0; j < static_cast<int>(segments.size()); ++j) {
        const int prev =
            (j + static_cast<int>(segments.size()) - 1) %
            static_cast<int>(segments.size());
        output[static_cast<std::size_t>(segments[static_cast<std::size_t>(j)].positions.front())] =
            segments[static_cast<std::size_t>(prev)].outgoing;
    }

    remote_writes = static_cast<int>(segments.size());
    return output;
}

} // namespace

int main() {
    std::uint64_t cases = 0;

    // Exhaust every owner sequence for small cycles.  This includes repeated
    // owners, one-owner cycles, alternating owners, and wrapped segments.
    for (int ngpu = 2; ngpu <= 4; ++ngpu) {
        for (int n = 2; n <= 8; ++n) {
            std::uint64_t total = 1;
            for (int i = 0; i < n; ++i) total *= static_cast<std::uint64_t>(ngpu);
            for (std::uint64_t code = 0; code < total; ++code) {
                std::uint64_t z = code;
                std::vector<int> owner(static_cast<std::size_t>(n));
                std::vector<int> input(static_cast<std::size_t>(n));
                for (int i = 0; i < n; ++i) {
                    owner[static_cast<std::size_t>(i)] = static_cast<int>(z % ngpu);
                    z /= static_cast<std::uint64_t>(ngpu);
                    input[static_cast<std::size_t>(i)] = i + 100;
                }

                int cross_edges = 0, remote_writes = 0;
                const auto direct = direct_rotate(input);
                const auto segmented = segmented_rotate(
                    input, owner, cross_edges, remote_writes);
                if (direct != segmented || remote_writes != cross_edges) {
                    std::cerr << "FAIL exhaustive"
                              << " n=" << n
                              << " ngpu=" << ngpu
                              << " code=" << code
                              << " cross_edges=" << cross_edges
                              << " remote_writes=" << remote_writes << '\n';
                    return 1;
                }
                ++cases;
            }
        }
    }

    // Cover the production maximum cycle length and all supported GPU counts.
    std::mt19937_64 rng(1);
    for (int trial = 0; trial < 200000; ++trial) {
        const int n = 2 + static_cast<int>(rng() % 27);
        const int ngpu = 2 + static_cast<int>(rng() % 7);
        std::vector<int> owner(static_cast<std::size_t>(n));
        std::vector<int> input(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            owner[static_cast<std::size_t>(i)] = static_cast<int>(rng() % ngpu);
            input[static_cast<std::size_t>(i)] = static_cast<int>(rng());
        }

        int cross_edges = 0, remote_writes = 0;
        if (direct_rotate(input) !=
                segmented_rotate(input, owner, cross_edges, remote_writes) ||
            remote_writes != cross_edges) {
            std::cerr << "FAIL random trial=" << trial << '\n';
            return 2;
        }
        ++cases;
    }

    std::cout << "ALL_OK"
              << " cases=" << cases
              << " max_cycle_len=28"
              << " max_ngpu=8"
              << " remote_writes_equals_owner_crossings=1"
              << " remote_reads_required=0"
              << " logical_peer_lower_bound_attained=1\n";
    return 0;
}
