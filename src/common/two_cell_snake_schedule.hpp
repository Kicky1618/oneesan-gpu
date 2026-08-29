#pragma once

#include "two_cell_recoupling_rank.hpp"

#include <cstdint>

namespace oneesan::twocell {

enum class SnakePairKind : std::uint8_t {
    ForwardFusion2,
    RightBoundary,
    ReverseFusion2,
    LeftBoundary,
};

struct SnakePair {
    SnakePairKind kind{};
    int start = 0; // low active/support-window start
};

constexpr int kMaxSnakePairs = 2 * kMaxWidth;

struct SnakeSchedule {
    SnakePair pair[kMaxSnakePairs]{};
    int size = 0;
    bool valid = false;
};

// Schedule one complete forward-turn-reverse-turn snake cycle.  For even W,
// each row direction has W-3 reduced K steps.  All but the final boundary K are
// grouped as fusion2 pairs; the remaining K is fused with the physical turn.
//
// Forward W=28:
//   fusion2 starts 0,2,...,22; right boundary carries K_24 + turn.
// Reverse W=28:
//   fusion2 low starts 23,21,...,1; left boundary carries active 1->0 + turn.
inline SnakeSchedule make_snake_schedule(int W) {
    SnakeSchedule out{};
    if (W < 6 || W > kMaxWidth || (W & 1)) return out;

    // Forward pairs cover K_0 ... K_{W-5}; boundary covers K_{W-4}.
    for (int start = 0; start <= W - 6; start += 2)
        out.pair[out.size++] = SnakePair{SnakePairKind::ForwardFusion2, start};
    out.pair[out.size++] = SnakePair{SnakePairKind::RightBoundary, W - 4};

    // Reverse fusion2 low-window starts descend by two.  A reverse pair with
    // low start s covers active s+2 -> s+1 -> s.  The final active 1->0 step is
    // fused with the left physical turn.
    for (int start = W - 5; start >= 1; start -= 2)
        out.pair[out.size++] = SnakePair{SnakePairKind::ReverseFusion2, start};
    out.pair[out.size++] = SnakePair{SnakePairKind::LeftBoundary, 0};

    out.valid = true;
    return out;
}

inline const char* snake_pair_kind_name(SnakePairKind kind) {
    switch (kind) {
        case SnakePairKind::ForwardFusion2: return "forward2";
        case SnakePairKind::RightBoundary: return "right-boundary";
        case SnakePairKind::ReverseFusion2: return "reverse2";
        case SnakePairKind::LeftBoundary: return "left-boundary";
    }
    return "unknown";
}

} // namespace oneesan::twocell
