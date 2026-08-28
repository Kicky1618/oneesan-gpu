#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <set>
#include <vector>

namespace {

constexpr int CHUNK = 5;
constexpr int KEYN = 243; // 3^5
constexpr int STATES = 26;
constexpr int MAX_DEPTH = 15;
constexpr int MAX_FACTOR = 14;
constexpr uint8_t MASK_MASK = 0x1fu;
constexpr int HALT_SHIFT = 5;
constexpr uint8_t META_LCOUNT_MASK = 0x07u;
constexpr int META_DELTA_SHIFT = 3;
constexpr int META_DELTA_BIAS = 5;

enum Symbol : uint32_t { N = 0, R = 1, L = 2 };

constexpr uint32_t pow3(int n) {
    return n == 0 ? 1u : 3u * pow3(n - 1);
}

constexpr uint32_t digit(uint32_t key, int pos) {
    return (key / pow3(pos)) % 3u;
}

constexpr uint8_t popcount5(uint8_t x) {
    uint8_t n = 0;
    for (int i = 0; i < CHUNK; ++i)
        n = uint8_t(n + ((x >> i) & 1u));
    return n;
}

constexpr uint8_t lmask(uint32_t key) {
    uint8_t mask = 0;
    for (int pos = 0; pos < CHUNK; ++pos)
        if (digit(key, pos) == L)
            mask = uint8_t(mask | uint8_t(1u << pos));
    return mask;
}

constexpr int chunk_delta(uint32_t key) {
    int d = 0;
    for (int pos = 0; pos < CHUNK; ++pos) {
        const uint32_t v = digit(key, pos);
        if (v == R) --d;
        else if (v == L) ++d;
    }
    return d;
}

constexpr uint8_t rankstream_meta(uint32_t key) {
    const uint8_t lcount = popcount5(lmask(key));
    return uint8_t(
        lcount |
        (uint8_t(chunk_delta(key) + META_DELTA_BIAS) << META_DELTA_SHIFT));
}

struct Walk {
    uint8_t candidate_mask = 0;
    int state = 0;
    bool halt = false;
};

constexpr Walk walk5(uint32_t key, int input_state) {
    Walk out{};
    out.state = input_state;
    for (int pos = CHUNK - 1; pos >= 0; --pos) {
        const uint32_t v = digit(key, pos);
        if (v == R) {
            if (out.state == 1) {
                out.halt = true;
                break;
            }
            --out.state;
        } else if (v == L) {
            if (out.state == 1)
                out.candidate_mask = uint8_t(
                    out.candidate_mask | uint8_t(1u << pos));
            ++out.state;
        }
    }
    return out;
}

constexpr uint8_t direct_entry(uint32_t key, int input_state) {
    const Walk w = walk5(key, input_state);
    return uint8_t(w.candidate_mask | (uint8_t(w.halt) << HALT_SHIFT));
}

constexpr uint8_t rankstream_entry(uint32_t key, int input_state) {
    const uint8_t e = direct_entry(key, input_state);
    const uint8_t candidates = uint8_t(e & MASK_MASK);
    const uint8_t lm = lmask(key);
    uint8_t rankmask = 0;

    for (int pos = 0; pos < CHUNK; ++pos) {
        if (((candidates >> pos) & 1u) == 0u) continue;
        const uint8_t lower_bits = uint8_t((uint8_t(1u << (pos + 1))) - 1u);
        const uint8_t higher_l = uint8_t(lm & uint8_t(~lower_bits));
        const uint8_t ordinal = popcount5(higher_l);
        rankmask = uint8_t(rankmask | uint8_t(1u << ordinal));
    }

    return uint8_t(rankmask | (e & uint8_t(1u << HALT_SHIFT)));
}

uint8_t rankmask_to_position_mask(uint32_t key, uint8_t rankmask) {
    std::array<int, CHUNK> pos_of_ordinal{};
    int n = 0;
    for (int pos = CHUNK - 1; pos >= 0; --pos)
        if (digit(key, pos) == L)
            pos_of_ordinal[size_t(n++)] = pos;

    uint8_t mask = 0;
    while (rankmask) {
        const int ordinal = __builtin_ctz(unsigned(rankmask));
        rankmask = uint8_t(rankmask & uint8_t(rankmask - 1));
        if (ordinal >= n) {
            std::cerr << "rank ordinal outside L stream ordinal=" << ordinal
                      << " lcount=" << n << " key=" << key << '\n';
            std::exit(2);
        }
        mask = uint8_t(mask | uint8_t(1u << pos_of_ordinal[size_t(ordinal)]));
    }
    return mask;
}

std::set<int> advance_reachable(
    const std::set<int>& input, int len, int& max_input_state, uint64_t& cases
) {
    std::set<int> output;
    const uint32_t keys = pow3(len);
    for (int s : input) {
        if (s <= 0 || s >= STATES) {
            std::cerr << "table input state out of range state=" << s
                      << " len=" << len << '\n';
            std::exit(3);
        }
        if (s > max_input_state) max_input_state = s;
        for (uint32_t key = 0; key < keys; ++key) {
            // For len<5 the unused high ternary digits are N=0, exactly matching
            // the production lookup where chunk is in [0,3^len).
            const Walk w = walk5(key, s);
            ++cases;
            if (!w.halt) {
                if (w.state <= 0) {
                    std::cerr << "non-halt reached non-positive state state="
                              << w.state << " key=" << key << " len=" << len
                              << " input=" << s << '\n';
                    std::exit(4);
                }
                output.insert(w.state);
            }
        }
    }
    return output;
}

} // namespace

int main() {
    static_assert(pow3(CHUNK) == KEYN);
    static_assert(MAX_FACTOR == 2 * CHUNK + 4);
    static_assert(MAX_DEPTH + 2 * CHUNK == 25);
    static_assert(STATES == 26);

    uint64_t projection_cases = 0;
    uint64_t fused16_cases = 0;
    for (int state = 1; state < STATES; ++state) {
        for (uint32_t key = 0; key < KEYN; ++key) {
            const uint8_t direct = direct_entry(key, state);
            const uint8_t ranked = rankstream_entry(key, state);
            const bool direct_halt = ((direct >> HALT_SHIFT) & 1u) != 0;
            const bool ranked_halt = ((ranked >> HALT_SHIFT) & 1u) != 0;
            if (direct_halt != ranked_halt) {
                std::cerr << "halt projection mismatch state=" << state
                          << " key=" << key << '\n';
                return 5;
            }

            const uint8_t direct_mask = uint8_t(direct & MASK_MASK);
            const uint8_t rankmask = uint8_t(ranked & MASK_MASK);
            const uint8_t reconstructed = rankmask_to_position_mask(key, rankmask);
            if (direct_mask != reconstructed) {
                std::cerr << "rank projection mismatch state=" << state
                          << " key=" << key
                          << " direct_mask=" << unsigned(direct_mask)
                          << " reconstructed=" << unsigned(reconstructed)
                          << " rankmask=" << unsigned(rankmask) << '\n';
                return 6;
            }

            const uint8_t count = popcount5(lmask(key));
            if (count > CHUNK) return 7;

            const uint8_t meta = rankstream_meta(key);
            const uint16_t pair = uint16_t(ranked) | (uint16_t(meta) << 8);
            const uint8_t pair_entry = uint8_t(pair);
            const uint8_t pair_meta = uint8_t(pair >> 8);
            const uint8_t decoded_lcount = uint8_t(pair_meta & META_LCOUNT_MASK);
            const int decoded_delta =
                int(pair_meta >> META_DELTA_SHIFT) - META_DELTA_BIAS;
            if (pair_entry != ranked || pair_meta != meta ||
                decoded_lcount != count || decoded_delta != chunk_delta(key)) {
                std::cerr << "fused16 rankstream packing mismatch state=" << state
                          << " key=" << key << " pair=" << pair
                          << " entry=" << unsigned(pair_entry)
                          << " meta=" << unsigned(pair_meta) << '\n';
                return 8;
            }
            ++fused16_cases;
            ++projection_cases;
        }
    }

    // K=14 is the worst production split: 5 high digits, then 5, then 4.
    // Propagate every possible chunk key instead of sampling complete factors.
    // A down transition at state one halts before decrement, so every state that
    // reaches a later chunk remains >=1.
    std::set<int> reachable;
    for (int d = 1; d <= MAX_DEPTH; ++d) reachable.insert(d);

    std::array<int, 3> max_input{};
    uint64_t reachability_cases = 0;
    reachable = advance_reachable(reachable, 5, max_input[0], reachability_cases);
    reachable = advance_reachable(reachable, 5, max_input[1], reachability_cases);
    reachable = advance_reachable(reachable, 4, max_input[2], reachability_cases);

    if (max_input[0] != 15 || max_input[1] != 20 || max_input[2] != 25) {
        std::cerr << "unexpected CROSS5 state bounds got="
                  << max_input[0] << ',' << max_input[1] << ',' << max_input[2]
                  << " expected=15,20,25\n";
        return 9;
    }

    std::cout << "gridfp-cross5-rankstream-proof OK"
              << " projection_cases=" << projection_cases
              << " fused16_cases=" << fused16_cases
              << " reachability_cases=" << reachability_cases
              << " chunk_state_bounds=" << max_input[0] << ','
              << max_input[1] << ',' << max_input[2]
              << " states=" << STATES
              << " max_depth=" << MAX_DEPTH
              << " max_factor=" << MAX_FACTOR
              << " rank_projection_exact=1"
              << " halt_projection_exact=1"
              << " partial_chunk_zero_prefix_exact=1"
              << " fused16_entry_exact=1"
              << " fused16_meta_exact=1"
              << " fused16_byte_isolation_exact=1"
              << " fallback_structurally_unreachable=1\n";
    return 0;
}
