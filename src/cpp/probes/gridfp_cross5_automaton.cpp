#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

static constexpr int CHUNK = 5;
static constexpr int KEYN = 243; // 3^5
// Runtime depth is four bits (<=15). Before the second/third 5-symbol chunk,
// at most 5/10 additional up symbols have been consumed, so the largest table
// input state is 15+10=25. States [0,25] are therefore sufficient.
static constexpr int STATES = 26;
static constexpr uint8_t MASK_MASK = 0x1fu;
static constexpr int HALT_SHIFT = 5;
static constexpr int MAX_RUNTIME_DEPTH = 15;
static constexpr int MAX_FACTOR = 14;
static constexpr int MAX_CHUNKS = 3;
static constexpr int MAX_TABLE_INPUT_STATE = MAX_RUNTIME_DEPTH + 2 * CHUNK;
static_assert(MAX_TABLE_INPUT_STATE == 25 && STATES == MAX_TABLE_INPUT_STATE + 1);

static constexpr int pow3(int n) { return n ? 3 * pow3(n - 1) : 1; }

// Normalized CROSS alphabet: 0=N, 1=down, 2=up.
// down at state 1 halts; up at state 1 contributes a preimage.
// The output state is deliberately not stored: if the chunk does not halt it
// is input_state + chunk_delta[key], while after halt no later chunk executes.
static constexpr uint8_t make_entry(int key, int input_state) {
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

static constexpr int8_t make_delta(int key) {
    int d = 0;
    for (int pos = 0; pos < CHUNK; ++pos) {
        int v = (key / pow3(pos)) % 3;
        if (v == 1) --d;
        else if (v == 2) ++d;
    }
    return int8_t(d);
}

static constexpr auto build_table() {
    std::array<uint8_t, STATES * KEYN> a{};
    for (int s = 0; s < STATES; ++s)
        for (int k = 0; k < KEYN; ++k)
            a[size_t(s) * KEYN + k] = make_entry(k, s);
    return a;
}
static constexpr auto build_delta() {
    std::array<int8_t, KEYN> a{};
    for (int k = 0; k < KEYN; ++k) a[size_t(k)] = make_delta(k);
    return a;
}
static constexpr auto TABLE = build_table();
static constexpr auto DELTA = build_delta();

struct Walk { uint16_t mask = 0; int state = 0; bool halt = false; };

static Walk scalar(uint32_t key, int len, int input_state) {
    Walk z{}; z.state = input_state;
    for (int pos = len - 1; pos >= 0; --pos) {
        int v = int((key / uint32_t(pow3(pos))) % 3u);
        if (v == 1) {
            if (z.state == 1) { z.halt = true; break; }
            --z.state;
        } else if (v == 2) {
            if (z.state == 1) z.mask |= uint16_t(1u << pos);
            ++z.state;
        }
    }
    return z;
}

static Walk cross5(uint32_t key, int len, int input_state, int& max_table_state) {
    Walk z{}; z.state = input_state;
    int remaining = len;
    while (remaining > 0 && !z.halt) {
        int clen = remaining > CHUNK ? CHUNK : remaining;
        int start = remaining - clen;
        uint32_t chunk = (key / uint32_t(pow3(start))) % uint32_t(pow3(clen));
        if (z.state < 0 || z.state >= STATES) {
            std::cerr << "state table overflow state=" << z.state << '\n'; std::exit(3);
        }
        max_table_state = std::max(max_table_state, z.state);
        uint8_t e = TABLE[size_t(z.state) * KEYN + chunk];
        uint8_t m = uint8_t(e & MASK_MASK);
        z.mask |= uint16_t(uint16_t(m) << start);
        z.halt = ((e >> HALT_SHIFT) & 1u) != 0;
        if (z.halt) {
            // A halt is precisely a down transition attempted from state 1;
            // scalar leaves the state at one because it breaks before --state.
            z.state = 1;
        } else {
            z.state += int(DELTA[chunk]);
        }
        remaining = start;
    }
    return z;
}

static void check_equal(uint32_t key, int len, int depth, int& max_table_state) {
    Walk a = scalar(key, len, depth), b = cross5(key, len, depth, max_table_state);
    if (a.mask != b.mask || a.state != b.state || a.halt != b.halt) {
        std::cerr << "cross5 mismatch len=" << len << " depth=" << depth << " key=" << key
                  << " scalar_mask=" << a.mask << " lut_mask=" << b.mask
                  << " scalar_state=" << a.state << " lut_state=" << b.state
                  << " scalar_halt=" << a.halt << " lut_halt=" << b.halt << '\n';
        std::exit(2);
    }
}

int main() {
    uint64_t table_cases = 0, composed_cases = 0;
    int max_table_state = 0;
    // Exhaust every compact LUT cell against the literal five-step state machine.
    for (int s = 1; s < STATES; ++s) {
        for (int key = 0; key < KEYN; ++key) {
            Walk a = scalar(uint32_t(key), CHUNK, s);
            uint8_t e = TABLE[size_t(s) * KEYN + key];
            bool halt = ((e >> HALT_SHIFT) & 1u) != 0;
            int next = halt ? 1 : s + int(DELTA[size_t(key)]);
            Walk b{uint16_t(e & MASK_MASK), next, halt};
            if (a.mask != b.mask || a.state != b.state || a.halt != b.halt) return 4;
            ++table_cases;
        }
    }

    // Exhaustive composition through 9 symbols, all runtime input depths.
    constexpr int EXHAUST_LEN = 9;
    for (uint32_t key = 0; key < uint32_t(pow3(EXHAUST_LEN)); ++key)
        for (int d = 1; d <= MAX_RUNTIME_DEPTH; ++d) {
            check_equal(key, EXHAUST_LEN, d, max_table_state); ++composed_cases;
        }

    // Deterministic stress for the production maximum K=14.
    std::mt19937 rng(0x1618c505u);
    constexpr int STRESS = 200000;
    for (int i = 0; i < STRESS; ++i) {
        uint32_t key = uint32_t(rng()) % uint32_t(pow3(MAX_FACTOR));
        int d = 1 + int(rng() % uint32_t(MAX_RUNTIME_DEPTH));
        check_equal(key, MAX_FACTOR, d, max_table_state);
        ++composed_cases;
    }

    const size_t mask_table_bytes = TABLE.size() * sizeof(uint8_t);
    const size_t delta_table_bytes = DELTA.size() * sizeof(int8_t);
    const size_t table_bytes = mask_table_bytes + delta_table_bytes;
    if (max_table_state > MAX_TABLE_INPUT_STATE) return 5;
    std::cout << "gridfp-cross5-automaton OK"
              << " chunk=" << CHUNK
              << " states=" << STATES
              << " mask_table_entries=" << TABLE.size()
              << " mask_table_bytes=" << mask_table_bytes
              << " delta_table_bytes=" << delta_table_bytes
              << " table_bytes=" << table_bytes
              << " old_table_bytes=15552"
              << " table_cases=" << table_cases
              << " composed_cases=" << composed_cases
              << " max_table_input_state=" << max_table_state
              << " proved_table_input_state_max=" << MAX_TABLE_INPUT_STATE
              << " max_factor=" << MAX_FACTOR << " max_chunks=" << MAX_CHUNKS
              << " candidate_mask_exact=1 state_exact=1 halt_exact=1 metadata_per_orbit=0 compact_state_delta=1\n";
    return 0;
}
