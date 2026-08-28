#pragma once

#include "two_cell_component_device.cuh"

#if defined(__CUDACC__)
#define ONEESAN_TC_TURN_HD __host__ __device__ __forceinline__
#else
#define ONEESAN_TC_TURN_HD inline
#endif

namespace oneesan::twocell {

constexpr int kMaxTurnStates = kMaxWidth / 2 + 1; // W=28 -> 15

struct ClosedTurnBlock {
    PackedKey state[kMaxTurnStates]{}; // alpha, beta, passive...
    int size = 0;
    bool singular = false;
    bool overflow = false;
};

ONEESAN_TC_TURN_HD std::uint32_t reverse_bits_len(std::uint32_t x, int len) {
    std::uint32_t y = 0;
    for (int p = 0; p < len; ++p)
        if ((x >> p) & 1u) y |= std::uint32_t(1) << (len - 1 - p);
    return y;
}

// Reflect the planar link state geometrically. Occupancy reverses. Every
// matched R endpoint becomes the L endpoint of the mirrored arc, while the
// distinguished unmatched root R stays an R after reflection.
ONEESAN_TC_TURN_HD PackedWord reflect_packed_word(PackedWord w) {
    const int root = root_position(w);
    if (root < 0) return PackedWord{};
    const std::uint32_t active = w.support & low_mask(w.len);
    const std::uint32_t rbits = active & ~w.left;
    const std::uint32_t matched_r = rbits & ~(std::uint32_t(1) << root);
    PackedWord z{};
    z.len = w.len;
    z.support = reverse_bits_len(active, w.len);
    z.left = reverse_bits_len(matched_r, w.len);
    return z;
}

ONEESAN_TC_TURN_HD PackedKey reflect_packed_key(PackedKey k, int W) {
    const PackedWord w = reflect_packed_word(state_word(k, W));
    return make_state(k.type, w);
}

ONEESAN_TC_TURN_HD bool turn_push(ClosedTurnBlock& b, PackedKey k) {
    if (b.size >= kMaxTurnStates) {
        b.overflow = true;
        return false;
    }
    b.state[b.size++] = k;
    return true;
}

// Exact right physical row-turn component for unrestricted label u in M_{W-2}.
// No graph traversal and no turn-edge generation are needed.
ONEESAN_TC_TURN_HD ClosedTurnBlock right_turn_closed_block(PackedWord label, int W) {
    ClosedTurnBlock out{};
    const int len = W - 2;
    if (label.len != len) {
        out.overflow = true;
        return out;
    }

    const Symbol last = symbol(label, len - 1);
    if (last == TC_R) {
        out.singular = true;
        turn_push(out, make_state(0, insert_symbol(label, len - 1, TC_N))); // alpha
        turn_push(out, make_state(1, label));                               // beta
        turn_push(out, make_state(0, insert_symbol(label, len, TC_N)));     // passive
        return out;
    }
    if (last != TC_N) {
        out.overflow = true;
        return out;
    }

    PackedWord v = remove_symbol(label, len - 1);
    PackedWord alpha = insert_symbol(v, v.len, TC_L);
    alpha = insert_symbol(alpha, alpha.len, TC_R);
    PackedWord beta = insert_symbol(v, v.len, TC_N);
    beta = insert_symbol(beta, beta.len, TC_N);
    turn_push(out, make_state(0, alpha));
    turn_push(out, make_state(0, beta));

    int h = 1;
    for (int q = 0; q < v.len; ++q) {
        const Symbol c = symbol(v, q);
        if (c == TC_L) {
            ++h;
        } else if (c == TC_R) {
            --h;
            if (h == 0) {
                PackedWord z = set_symbol(v, q, TC_L);
                z = insert_symbol(z, z.len, TC_R);
                z = insert_symbol(z, z.len, TC_R);
                turn_push(out, make_state(0, z));
            }
        }
        if (h < 0) {
            out.overflow = true;
            return out;
        }
    }
    if (h != 0 || out.size < 3) out.overflow = true;
    return out;
}

// The left physical turn is exact geometric reflection of the right one.
// Reflect the unrestricted label, construct the right block, then reflect each
// reduced coordinate back. The alpha/beta/passive ordering is preserved, so
// the same arithmetic executor applies in both row directions.
ONEESAN_TC_TURN_HD ClosedTurnBlock left_turn_closed_block(PackedWord label, int W) {
    const PackedWord mirrored_label = reflect_packed_word(label);
    ClosedTurnBlock out = right_turn_closed_block(mirrored_label, W);
    if (out.overflow) return out;
    for (int q = 0; q < out.size; ++q)
        out.state[q] = reflect_packed_key(out.state[q], W);
    return out;
}

} // namespace oneesan::twocell

#undef ONEESAN_TC_TURN_HD
