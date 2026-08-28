#include <cstdint>
#include <iostream>

static constexpr int CHUNK = 5;
static constexpr uint32_t KEYS = 243;
static constexpr uint32_t STATES = 26;

enum V : uint32_t { N = 0, R = 1, L = 2 };

static uint32_t pow3(int n) {
    uint32_t z = 1;
    while (n--) z *= 3u;
    return z;
}

struct Result {
    uint8_t rankmask = 0;
    uint8_t halt = 0;
    uint32_t state = 0;
};

// Reproduce ordinary CROSS5 positional-hit semantics first, then map selected
// positions to sparse-L ordinals exactly as rankstream_host_entry does.
static Result reference(uint32_t key, uint32_t input_state) {
    uint32_t s = input_state;
    uint8_t posmask = 0;
    bool halt = false;
    for (int pos = CHUNK - 1; pos >= 0; --pos) {
        const uint32_t v = (key / pow3(pos)) % 3u;
        if (v == R) {
            if (s == 1u) { halt = true; break; }
            --s;
        } else if (v == L) {
            if (s == 1u) posmask |= uint8_t(1u << pos);
            ++s;
        }
    }

    uint8_t lmask = 0;
    for (int pos = 0; pos < CHUNK; ++pos)
        if (((key / pow3(pos)) % 3u) == L) lmask |= uint8_t(1u << pos);
    uint8_t rankmask = 0;
    for (int pos = 0; pos < CHUNK; ++pos) {
        if (((posmask >> pos) & 1u) == 0u) continue;
        uint8_t ordinal = 0;
        for (int q = pos + 1; q < CHUNK; ++q)
            ordinal += uint8_t((lmask >> q) & 1u);
        rankmask |= uint8_t(1u << ordinal);
    }
    return Result{rankmask, uint8_t(halt), s};
}

// Single scan used by rankformula: sparse-L ordinal tracking, hit selection,
// halt detection, and state update happen together.
static Result inline_cross(uint32_t key, uint32_t input_state) {
    uint32_t s = input_state;
    uint8_t rankmask = 0, ordinal = 0;
    for (int pos = CHUNK - 1; pos >= 0; --pos) {
        const uint32_t v = (key / pow3(pos)) % 3u;
        if (v == R) {
            if (s == 1u) return Result{rankmask, 1u, s};
            --s;
        } else if (v == L) {
            if (s == 1u) rankmask |= uint8_t(1u << ordinal);
            ++s;
            ++ordinal;
        }
    }
    return Result{rankmask, 0u, s};
}

int main() {
    uint64_t cases = 0;
    uint32_t max_rankmask = 0;
    for (uint32_t state = 1; state < STATES; ++state) {
        for (uint32_t key = 0; key < KEYS; ++key) {
            const Result a = reference(key, state);
            const Result b = inline_cross(key, state);
            if (a.rankmask != b.rankmask || a.halt != b.halt || a.state != b.state) {
                std::cerr << "rankformula inline-cross mismatch state=" << state
                          << " key=" << key
                          << " ref_mask=" << unsigned(a.rankmask)
                          << " got_mask=" << unsigned(b.rankmask)
                          << " ref_halt=" << unsigned(a.halt)
                          << " got_halt=" << unsigned(b.halt)
                          << " ref_state=" << a.state
                          << " got_state=" << b.state << '\n';
                return 2;
            }
            if (!a.halt) {
                int delta = 0;
                for (int pos = 0; pos < CHUNK; ++pos) {
                    const uint32_t v = (key / pow3(pos)) % 3u;
                    delta += v == L ? 1 : v == R ? -1 : 0;
                }
                if (int(b.state) != int(state) + delta) return 3;
            }
            max_rankmask = max_rankmask > b.rankmask ? max_rankmask : b.rankmask;
            ++cases;
        }
    }
    std::cout << "gridfp-rankformula-inline-cross OK"
              << " states=25 keys=243 cases=" << cases
              << " max_rankmask=" << max_rankmask
              << " cross_lut_loads=0"
              << " single_symbol_scan=1\n";
    return 0;
}
