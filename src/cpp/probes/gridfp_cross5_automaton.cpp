#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>

static constexpr int CHUNK = 5;
static constexpr int KEYN = 243; // 3^5
static constexpr int STATES = 32;
static constexpr uint16_t MASK_MASK = 0x1fu;
static constexpr int STATE_SHIFT = 5;
static constexpr uint16_t STATE_MASK = 0x3fu;
static constexpr int HALT_SHIFT = 11;

static constexpr int pow3(int n) { return n ? 3 * pow3(n - 1) : 1; }

// Normalized CROSS alphabet: 0=N, 1=down, 2=up.
// down at state 1 halts; up at state 1 contributes a preimage.
static constexpr uint16_t make_entry(int key, int input_state) {
    int s = input_state;
    uint16_t mask = 0;
    bool halt = false;
    for (int pos = CHUNK - 1; pos >= 0; --pos) {
        int v = (key / pow3(pos)) % 3;
        if (v == 1) {
            if (s == 1) { halt = true; break; }
            --s;
        } else if (v == 2) {
            if (s == 1) mask |= uint16_t(1u << pos);
            ++s;
        }
    }
    return uint16_t(mask | (uint16_t(s) << STATE_SHIFT) | (uint16_t(halt) << HALT_SHIFT));
}

static constexpr auto build_table() {
    std::array<uint16_t, STATES * KEYN> a{};
    for (int s = 0; s < STATES; ++s)
        for (int k = 0; k < KEYN; ++k)
            a[size_t(s) * KEYN + k] = make_entry(k, s);
    return a;
}
static constexpr auto TABLE = build_table();

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

static Walk cross5(uint32_t key, int len, int input_state) {
    Walk z{}; z.state = input_state;
    int remaining = len;
    while (remaining > 0 && !z.halt) {
        int clen = remaining > CHUNK ? CHUNK : remaining;
        int start = remaining - clen;
        uint32_t chunk = (key / uint32_t(pow3(start))) % uint32_t(pow3(clen));
        if (z.state < 0 || z.state >= STATES) {
            std::cerr << "state table overflow state=" << z.state << '\n'; std::exit(3);
        }
        uint16_t e = TABLE[size_t(z.state) * KEYN + chunk];
        uint16_t m = e & MASK_MASK;
        z.mask |= uint16_t(m << start);
        z.state = int((e >> STATE_SHIFT) & STATE_MASK);
        z.halt = ((e >> HALT_SHIFT) & 1u) != 0;
        remaining = start;
    }
    return z;
}

static void check_equal(uint32_t key, int len, int depth) {
    Walk a = scalar(key, len, depth), b = cross5(key, len, depth);
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
    // Exhaust every LUT cell against the literal five-step state machine.
    for (int s = 1; s < STATES; ++s) {
        for (int key = 0; key < KEYN; ++key) {
            Walk a = scalar(uint32_t(key), CHUNK, s);
            uint16_t e = TABLE[size_t(s) * KEYN + key];
            Walk b{uint16_t(e & MASK_MASK), int((e >> STATE_SHIFT) & STATE_MASK), ((e >> HALT_SHIFT) & 1u) != 0};
            if (a.mask != b.mask || a.state != b.state || a.halt != b.halt) return 4;
            ++table_cases;
        }
    }

    // Exhaustive composition through 9 symbols, all runtime input depths.
    constexpr int EXHAUST_LEN = 9;
    for (uint32_t key = 0; key < uint32_t(pow3(EXHAUST_LEN)); ++key)
        for (int d = 1; d <= 15; ++d) { check_equal(key, EXHAUST_LEN, d); ++composed_cases; }

    // Deterministic stress for the production maximum K=14.
    std::mt19937 rng(0x1618c505u);
    constexpr int STRESS = 200000;
    for (int i = 0; i < STRESS; ++i) {
        uint32_t key = uint32_t(rng()) % uint32_t(pow3(14));
        int d = 1 + int(rng() % 15u);
        check_equal(key, 14, d);
        ++composed_cases;
    }

    std::cout << "gridfp-cross5-automaton OK"
              << " chunk=" << CHUNK
              << " table_entries=" << TABLE.size()
              << " table_bytes=" << TABLE.size() * sizeof(uint16_t)
              << " table_cases=" << table_cases
              << " composed_cases=" << composed_cases
              << " max_factor=14 max_chunks=3"
              << " candidate_mask_exact=1 state_exact=1 halt_exact=1 metadata_per_orbit=0\n";
    return 0;
}
