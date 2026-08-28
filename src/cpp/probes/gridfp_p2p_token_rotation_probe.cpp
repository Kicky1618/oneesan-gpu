#include <cstdint>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

namespace {

struct Segment {
    std::vector<int> positions;
    int owner = -1;
};

std::vector<int> direct_rotate(const std::vector<int>& input) {
    const int n = static_cast<int>(input.size());
    std::vector<int> output(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        output[static_cast<std::size_t>((i + 1) % n)] = input[static_cast<std::size_t>(i)];
    return output;
}

std::vector<Segment> owner_segments(const std::vector<int>& owner) {
    const int n = static_cast<int>(owner.size());
    int crossings = 0;
    for (int i = 0; i < n; ++i)
        crossings += owner[static_cast<std::size_t>(i)] !=
                     owner[static_cast<std::size_t>((i + 1) % n)];
    if (!crossings) return {Segment{{}, owner[0]}};

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
    return segments;
}

// One-token / one-hole cyclic rotation.
//
// For a primitive slab every primitive-rank lane executes this permutation
// independently, so `token` becomes a chunk of contiguous u32 values in the
// GPU implementation.  Segment 0 extracts its outgoing tail into the token,
// shifts its local values, and leaves its first position as a hole.  The token
// then migrates through each owner segment.  A receiving owner extracts a new
// outgoing token, performs only local shifts, fills its first position from the
// incoming token, and forwards the new token.  The final owner returns the
// token to segment 0's hole.
//
// Therefore state memory itself never needs a remote read.  Exactly one token
// transfer is made per cyclic owner boundary, attaining the cross-owner lower
// bound with one token slot per active cycle/chunk.
std::vector<int> token_rotate(
    const std::vector<int>& input,
    const std::vector<int>& owner,
    int& peer_hops
) {
    const int n = static_cast<int>(input.size());
    std::vector<int> output = input;
    int crossings = 0;
    for (int i = 0; i < n; ++i)
        crossings += owner[static_cast<std::size_t>(i)] !=
                     owner[static_cast<std::size_t>((i + 1) % n)];

    if (!crossings) {
        const int tail = output.back();
        for (int i = n - 1; i > 0; --i)
            output[static_cast<std::size_t>(i)] =
                output[static_cast<std::size_t>(i - 1)];
        output[0] = tail;
        peer_hops = 0;
        return output;
    }

    const auto segments = owner_segments(owner);
    int token = output[static_cast<std::size_t>(segments[0].positions.back())];

    // Open the hole in the first segment.  Its first value has already been
    // consumed by the backward local shift before that position is left open.
    for (int k = static_cast<int>(segments[0].positions.size()) - 1; k > 0; --k) {
        output[static_cast<std::size_t>(segments[0].positions[static_cast<std::size_t>(k)])] =
            output[static_cast<std::size_t>(segments[0].positions[static_cast<std::size_t>(k - 1)])];
    }

    for (int j = 1; j < static_cast<int>(segments.size()); ++j) {
        const auto& segment = segments[static_cast<std::size_t>(j)];
        const int next_token =
            output[static_cast<std::size_t>(segment.positions.back())];
        for (int k = static_cast<int>(segment.positions.size()) - 1; k > 0; --k) {
            output[static_cast<std::size_t>(segment.positions[static_cast<std::size_t>(k)])] =
                output[static_cast<std::size_t>(segment.positions[static_cast<std::size_t>(k - 1)])];
        }
        output[static_cast<std::size_t>(segment.positions.front())] = token;
        token = next_token;
    }

    output[static_cast<std::size_t>(segments[0].positions.front())] = token;
    peer_hops = static_cast<int>(segments.size());
    return output;
}

int cyclic_crossings(const std::vector<int>& owner) {
    int z = 0;
    for (int i = 0; i < static_cast<int>(owner.size()); ++i)
        z += owner[static_cast<std::size_t>(i)] !=
             owner[static_cast<std::size_t>((i + 1) % owner.size())];
    return z;
}

} // namespace

int main() {
    std::uint64_t cases = 0;

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
                    input[static_cast<std::size_t>(i)] = 100 + i;
                }
                int peer_hops = 0;
                if (token_rotate(input, owner, peer_hops) != direct_rotate(input) ||
                    peer_hops != cyclic_crossings(owner)) {
                    std::cerr << "FAIL exhaustive"
                              << " n=" << n
                              << " ngpu=" << ngpu
                              << " code=" << code << '\n';
                    return 1;
                }
                ++cases;
            }
        }
    }

    std::mt19937_64 rng(2);
    for (int trial = 0; trial < 200000; ++trial) {
        const int n = 2 + static_cast<int>(rng() % 27);
        const int ngpu = 2 + static_cast<int>(rng() % 7);
        std::vector<int> owner(static_cast<std::size_t>(n));
        std::vector<int> input(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            owner[static_cast<std::size_t>(i)] = static_cast<int>(rng() % ngpu);
            input[static_cast<std::size_t>(i)] = static_cast<int>(rng());
        }
        int peer_hops = 0;
        if (token_rotate(input, owner, peer_hops) != direct_rotate(input) ||
            peer_hops != cyclic_crossings(owner)) {
            std::cerr << "FAIL random trial=" << trial << '\n';
            return 2;
        }
        ++cases;
    }

    std::cout << "ALL_OK"
              << " cases=" << cases
              << " max_cycle_len=28"
              << " max_ngpu=8"
              << " token_slots_per_active_cycle=1"
              << " peer_hops_equals_owner_crossings=1"
              << " remote_state_reads_required=0"
              << " logical_peer_lower_bound_attained=1\n";
    return 0;
}
