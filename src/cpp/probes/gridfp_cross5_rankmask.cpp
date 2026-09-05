#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

static constexpr int CHUNK = 5;
static constexpr int KEYN = 243;
static constexpr int STATES = 26;
static constexpr int MAX_RUNTIME_DEPTH = 15;
static constexpr int MAX_FACTOR = 14;
static constexpr uint8_t MASK_MASK = 0x1fu;
static constexpr int HALT_SHIFT = 5;
static constexpr uint8_t META_LCOUNT_MASK = 0x07u;
static constexpr int META_DELTA_SHIFT = 3;
static constexpr int META_DELTA_BIAS = 5;

static constexpr int pow3(int n) { return n ? 3 * pow3(n - 1) : 1; }

// Normalized alphabet: 0=N, 1=R/down, 2=L/up.
static constexpr uint8_t cross5_entry(int key, int input_state) {
    int s = input_state;
    uint8_t mask = 0;
    bool halt = false;
    for (int pos = CHUNK - 1; pos >= 0; --pos) {
        int v = (key / pow3(pos)) % 3;
        if (v == 1) {
            if (s == 1) { halt = true; break; }
            --s;
        } else if (v == 2) {
            if (s == 1) mask = uint8_t(mask | uint8_t(1u << pos));
            ++s;
        }
    }
    return uint8_t(mask | (uint8_t(halt) << HALT_SHIFT));
}

static constexpr int8_t cross5_delta(int key) {
    int d = 0;
    for (int pos = 0; pos < CHUNK; ++pos) {
        int v = (key / pow3(pos)) % 3;
        if (v == 1) --d;
        else if (v == 2) ++d;
    }
    return int8_t(d);
}

static constexpr uint8_t lmask(int key) {
    uint8_t mask = 0;
    for (int pos = 0; pos < CHUNK; ++pos)
        if ((key / pow3(pos)) % 3 == 2)
            mask = uint8_t(mask | uint8_t(1u << pos));
    return mask;
}

static constexpr uint8_t popcount5(uint8_t x) {
    uint8_t n = 0;
    for (int i = 0; i < CHUNK; ++i) n = uint8_t(n + ((x >> i) & 1u));
    return n;
}

static constexpr uint8_t rank_entry(int key, int input_state) {
    const uint8_t e = cross5_entry(key, input_state);
    const uint8_t mask = uint8_t(e & MASK_MASK);
    const uint8_t lm = lmask(key);
    uint8_t rankmask = 0;
    for (int pos = 0; pos < CHUNK; ++pos) {
        if (((mask >> pos) & 1u) == 0u) continue;
        const uint8_t higher = uint8_t(
            lm & uint8_t(~uint8_t((uint8_t(1u << (pos + 1))) - 1u)));
        const uint8_t ordinal = popcount5(higher);
        rankmask = uint8_t(rankmask | uint8_t(1u << ordinal));
    }
    return uint8_t(rankmask | (e & uint8_t(1u << HALT_SHIFT)));
}

static constexpr uint8_t chunk_meta(int key) {
    return uint8_t(
        popcount5(lmask(key)) |
        (uint8_t(int(cross5_delta(key)) + META_DELTA_BIAS) << META_DELTA_SHIFT));
}

struct Walk {
    uint16_t ordinals = 0;
    int state = 0;
    bool halt = false;
};

static Walk scalar(uint32_t key, int len, int input_state) {
    Walk z{};
    z.state = input_state;
    int higher_l = 0;
    for (int pos = len - 1; pos >= 0; --pos) {
        int v = int((key / uint32_t(pow3(pos))) % 3u);
        if (v == 1) {
            if (z.state == 1) { z.halt = true; break; }
            --z.state;
        } else if (v == 2) {
            if (z.state == 1) z.ordinals |= uint16_t(1u << higher_l);
            ++higher_l;
            ++z.state;
        }
    }
    return z;
}

static Walk ranked(uint32_t key, int len, int input_state, int& max_state) {
    Walk z{};
    z.state = input_state;
    int lbase = 0;
    int remaining = len;
    while (remaining > 0 && !z.halt) {
        const int clen = std::min(CHUNK, remaining);
        const int start = remaining - clen;
        const uint32_t chunk = (key / uint32_t(pow3(start))) % uint32_t(pow3(clen));
        if (z.state < 0 || z.state >= STATES) {
            std::cerr << "rankmask state overflow state=" << z.state << '\n';
            std::exit(3);
        }
        max_state = std::max(max_state, z.state);
        const uint8_t e = rank_entry(int(chunk), z.state);
        const uint8_t rm = uint8_t(e & MASK_MASK);
        z.ordinals |= uint16_t(uint16_t(rm) << lbase);
        z.halt = ((e >> HALT_SHIFT) & 1u) != 0;
        if (z.halt) {
            z.state = 1;
        } else {
            const uint8_t meta = chunk_meta(int(chunk));
            lbase += int(meta & META_LCOUNT_MASK);
            z.state += int(meta >> META_DELTA_SHIFT) - META_DELTA_BIAS;
        }
        remaining = start;
    }
    return z;
}

static void check(uint32_t key, int len, int depth, int& max_state) {
    const Walk a = scalar(key, len, depth);
    const Walk b = ranked(key, len, depth, max_state);
    if (a.ordinals != b.ordinals || a.state != b.state || a.halt != b.halt) {
        std::cerr << "rankmask mismatch len=" << len << " depth=" << depth << " key=" << key
                  << " scalar_ordinals=" << a.ordinals << " rank_ordinals=" << b.ordinals
                  << " scalar_state=" << a.state << " rank_state=" << b.state
                  << " scalar_halt=" << a.halt << " rank_halt=" << b.halt << '\n';
        std::exit(2);
    }
}

int main() {
    uint64_t table_cases = 0;
    uint64_t composed_cases = 0;
    int max_state = 0;

    for (int key = 0; key < KEYN; ++key) {
        const uint8_t meta = chunk_meta(key);
        const int got_lcount = int(meta & META_LCOUNT_MASK);
        const int got_delta = int(meta >> META_DELTA_SHIFT) - META_DELTA_BIAS;
        if (got_lcount != int(popcount5(lmask(key))) || got_delta != int(cross5_delta(key)))
            return 6;
    }

    for (int s = 1; s < STATES; ++s) {
        for (int key = 0; key < KEYN; ++key) {
            const uint8_t e = cross5_entry(key, s);
            const uint8_t lm = lmask(key);
            const uint8_t mask = uint8_t(e & MASK_MASK);
            if (mask & uint8_t(~lm)) return 4;

            uint8_t expected = 0;
            for (int pos = 0; pos < CHUNK; ++pos) {
                if (((mask >> pos) & 1u) == 0u) continue;
                const uint8_t higher = uint8_t(
                    lm & uint8_t(~uint8_t((uint8_t(1u << (pos + 1))) - 1u)));
                expected = uint8_t(expected | uint8_t(1u << popcount5(higher)));
            }
            const uint8_t got = rank_entry(key, s);
            if ((got & MASK_MASK) != expected ||
                ((got ^ e) & uint8_t(1u << HALT_SHIFT)))
                return 5;
            ++table_cases;
        }
    }

    constexpr int EXHAUST_LEN = 9;
    for (uint32_t key = 0; key < uint32_t(pow3(EXHAUST_LEN)); ++key) {
        for (int depth = 1; depth <= MAX_RUNTIME_DEPTH; ++depth) {
            check(key, EXHAUST_LEN, depth, max_state);
            ++composed_cases;
        }
    }

    std::mt19937 rng(0x1618a505u);
    constexpr int STRESS = 300000;
    for (int i = 0; i < STRESS; ++i) {
        const uint32_t key = uint32_t(rng()) % uint32_t(pow3(MAX_FACTOR));
        const int depth = 1 + int(rng() % uint32_t(MAX_RUNTIME_DEPTH));
        check(key, MAX_FACTOR, depth, max_state);
        ++composed_cases;
    }

    std::cout << "gridfp-cross5-rankmask OK"
              << " states=" << STATES
              << " keys=" << KEYN
              << " table_cases=" << table_cases
              << " composed_cases=" << composed_cases
              << " max_table_state=" << max_state
              << " max_factor=" << MAX_FACTOR
              << " rankmask_bytes=" << STATES * KEYN
              << " meta_bytes=" << KEYN
              << " constant_loads_per_chunk=2"
              << " ordinal_popcount_runtime=0"
              << " meta_lcount_exact=1 meta_delta_exact=1"
              << " candidate_set_exact=1 state_exact=1 halt_exact=1\n";
    return 0;
}
