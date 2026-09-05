#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <utility>

namespace {

struct State {
    std::uint32_t support = 0;
    std::uint32_t spin = 0;

    friend bool operator==(const State&, const State&) = default;
};

struct StateHash {
    std::size_t operator()(const State& s) const noexcept {
        const std::uint64_t x = (std::uint64_t{s.spin} << 32) | s.support;
        std::uint64_t z = x + 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return static_cast<std::size_t>(z ^ (z >> 31));
    }
};

using Count = std::uint64_t;
using Table = std::unordered_map<State, Count, StateHash>;

constexpr std::uint32_t low_mask(unsigned bits) {
    return bits == 0 ? 0u : ((std::uint32_t{1} << bits) - 1u);
}

struct SpinOutputs {
    std::uint32_t value[2]{};
    int count = 0;
};

// spin bit j is the gap between active endpoints j and j+1, ordered left-to-right.
// 0 = up, 1 = down in the free-fermion embedding.
SpinOutputs insert_pair(std::uint32_t spin, unsigned endpoints, unsigned rank) {
    SpinOutputs out;
    if (rank > endpoints) return out;

    // Pair inserted to the far left: prepend [up, down] = [0, 1].
    if (rank == 0) {
        out.value[0] = (spin << 2) | (1u << 1);
        out.count = 1;
        return out;
    }

    // Pair inserted to the far right: append [down, up] = [1, 0].
    if (rank == endpoints) {
        out.value[0] = spin | (1u << (endpoints - 1));
        out.count = 1;
        return out;
    }

    // Interior insertion replaces one old gap by three new gaps.
    const unsigned g = rank - 1;
    const std::uint32_t below = spin & low_mask(g);
    const std::uint32_t above = spin >> (g + 1);
    const unsigned old = (spin >> g) & 1u;

    auto build = [&](unsigned triple) {
        return below | (std::uint32_t{triple} << g) | (above << (g + 3));
    };

    if (old == 0) {
        // up -> down,up,up + up,up,down
        // left-to-right bits [1,0,0] and [0,0,1].
        out.value[0] = build(0b001); // bit 0 is the leftmost gap.
        out.value[1] = build(0b100);
        out.count = 2;
    } else {
        // down -> down,up,down = [1,0,1].
        out.value[0] = build(0b101);
        out.count = 1;
    }
    return out;
}

SpinOutputs remove_pair(std::uint32_t spin, unsigned endpoints, unsigned rank) {
    SpinOutputs out;
    if (endpoints < 3 || rank + 1 >= endpoints) return out;

    // Remove the far-left pair: require [up, down] = [0, 1].
    if (rank == 0) {
        if ((spin & 0b11u) == 0b10u) {
            out.value[0] = spin >> 2;
            out.count = 1;
        }
        return out;
    }

    // Remove the far-right pair: require [down, up] = [1, 0].
    if (rank + 2 == endpoints) {
        const unsigned g = endpoints - 3;
        if (((spin >> g) & 0b11u) == 0b01u) {
            out.value[0] = spin & low_mask(g);
            out.count = 1;
        }
        return out;
    }

    // Interior removal acts on three adjacent gap spins.
    const unsigned g = rank - 1;
    const unsigned triple = (spin >> g) & 0b111u;
    int merged = -1;
    switch (triple) {
        case 0b010: merged = 0; break; // up,down,up -> up
        case 0b011: merged = 1; break; // down,down,up -> down
        case 0b110: merged = 1; break; // up,down,down -> down
        default: return out;
    }

    const std::uint32_t below = spin & low_mask(g);
    const std::uint32_t above = spin >> (g + 3);
    out.value[0] = below | (std::uint32_t{merged} << g) | (above << (g + 1));
    out.count = 1;
    return out;
}

void add(Table& table, State state, Count count) {
    table[state] += count;
}

Count solve(unsigned n, bool verbose) {
    if (n < 2 || n > 15) {
        std::cerr << "n must be in [2, 15] for this probe\n";
        std::exit(2);
    }

    const unsigned slots = n + 1;
    const std::uint32_t slots_mask = low_mask(slots);

    Table cur, next;
    cur.reserve(1 << 12);
    cur[{0, 0}] = 1;

    std::size_t peak_states = 1;

    for (unsigned y = 0; y < n; ++y) {
        for (unsigned x = 0; x < n; ++x) {
            next.clear();
            next.reserve(cur.size() * 2 + 64);

            const unsigned p = x;
            const unsigned q = x + 1;
            const std::uint32_t local_mask = (1u << p) | (1u << q);
            const bool source = (x == 0 && y == 0);
            const bool target = (x + 1 == n && y + 1 == n);
            const bool allow_down = (y + 1 < n);
            const bool allow_right = (x + 1 < n);

            for (const auto& [state, count] : cur) {
                const unsigned endpoints = std::popcount(state.support);
                const unsigned in = std::popcount(state.support & local_mask);

                if (source) {
                    if (state.support != 0 || state.spin != 0 || in != 0) continue;
                    if (allow_down) add(next, {1u << p, 0}, count);
                    if (allow_right) add(next, {1u << q, 0}, count);
                    continue;
                }

                if (target) {
                    if (endpoints == 1 && in == 1) add(next, {0, 0}, count);
                    continue;
                }

                // All non-terminal states are in the one-defect sector.
                if ((endpoints & 1u) == 0 || endpoints == 0) continue;
                const unsigned r = (endpoints - 1) / 2;
                if (std::popcount(state.spin) != r) continue;
                if ((state.spin & ~low_mask(endpoints - 1)) != 0) continue;

                const std::uint32_t base_support = state.support & ~local_mask;

                if (in == 0) {
                    // degree 0
                    add(next, {base_support, state.spin}, count);

                    // degree 2: create a cup, if both outgoing edges exist.
                    if (allow_down && allow_right) {
                        const unsigned rank = std::popcount(state.support & low_mask(p));
                        const auto spins = insert_pair(state.spin, endpoints, rank);
                        const std::uint32_t s2 = base_support | local_mask;
                        for (int i = 0; i < spins.count; ++i) {
                            add(next, {s2, spins.value[i]}, count);
                        }
                    }
                } else if (in == 1) {
                    // degree 2 requires exactly one outgoing edge.  The active
                    // endpoint keeps its rank, so the gap-spin word is unchanged.
                    if (allow_down) add(next, {base_support | (1u << p), state.spin}, count);
                    if (allow_right) add(next, {base_support | (1u << q), state.spin}, count);
                } else if (in == 2) {
                    // degree 2 with no outgoing edge: cap two adjacent endpoints.
                    const unsigned rank = std::popcount(state.support & low_mask(p));
                    const auto spins = remove_pair(state.spin, endpoints, rank);
                    for (int i = 0; i < spins.count; ++i) {
                        add(next, {base_support, spins.value[i]}, count);
                    }
                }
            }

            cur.swap(next);
            peak_states = std::max(peak_states, cur.size());
            if (verbose) {
                std::cerr << "n=" << n << " y=" << y << " x=" << x
                          << " states=" << cur.size() << "\n";
            }
        }

        // Move the kink from the right boundary back to the left boundary.
        // The active endpoint order is unchanged, hence spin is unchanged.
        next.clear();
        next.reserve(cur.size() + 16);
        for (const auto& [state, count] : cur) {
            if ((state.support >> n) & 1u) continue; // right boundary must be empty
            const State shifted{(state.support << 1) & slots_mask, state.spin};
            add(next, shifted, count);
        }
        cur.swap(next);
    }

    if (verbose) std::cerr << "peak_states=" << peak_states << "\n";
    const auto it = cur.find({0, 0});
    return it == cur.end() ? 0 : it->second;
}

} // namespace

int main(int argc, char** argv) {
    unsigned first = 2;
    unsigned last = 9;
    bool verbose = false;
    if (argc >= 2) first = static_cast<unsigned>(std::strtoul(argv[1], nullptr, 10));
    if (argc >= 3) last = static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10));
    if (argc >= 4) verbose = std::string_view(argv[3]) == "-v";

    for (unsigned n = first; n <= last; ++n) {
        const Count ans = solve(n, verbose);
        std::cout << n << ' ' << ans << '\n';
    }
}
