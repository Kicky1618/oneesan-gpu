#include <cstdint>
#include <iostream>

namespace {

constexpr int CHUNK = 5;
constexpr int MAX_K = 14;
constexpr int MAX_DEPTH = 15;
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
    for (int i = 0; i < CHUNK; ++i) n = uint8_t(n + ((x >> i) & 1u));
    return n;
}

constexpr uint8_t lmask(uint32_t key) {
    uint8_t out = 0;
    for (int pos = 0; pos < CHUNK; ++pos)
        if (digit(key, pos) == L) out = uint8_t(out | uint8_t(1u << pos));
    return out;
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

constexpr uint8_t direct_entry(uint32_t key, uint32_t input_state) {
    uint32_t state = input_state;
    uint8_t mask = 0;
    bool halt = false;
    for (int pos = CHUNK - 1; pos >= 0; --pos) {
        const uint32_t v = digit(key, pos);
        if (v == R) {
            if (state == 1u) { halt = true; break; }
            --state;
        } else if (v == L) {
            if (state == 1u) mask = uint8_t(mask | uint8_t(1u << pos));
            ++state;
        }
    }
    return uint8_t(mask | (uint8_t(halt) << HALT_SHIFT));
}

constexpr uint8_t rankstream_entry(uint32_t key, uint32_t input_state) {
    const uint8_t e = direct_entry(key, input_state);
    const uint8_t candidates = uint8_t(e & MASK_MASK);
    const uint8_t lm = lmask(key);
    uint8_t rankmask = 0;
    for (int pos = 0; pos < CHUNK; ++pos) {
        if (((candidates >> pos) & 1u) == 0u) continue;
        const uint8_t lower = uint8_t((uint8_t(1u << (pos + 1))) - 1u);
        const uint8_t higher_l = uint8_t(lm & uint8_t(~lower));
        const uint8_t ordinal = popcount5(higher_l);
        rankmask = uint8_t(rankmask | uint8_t(1u << ordinal));
    }
    return uint8_t(rankmask | (e & uint8_t(1u << HALT_SHIFT)));
}

constexpr uint8_t rankstream_meta(uint32_t key) {
    return uint8_t(
        popcount5(lmask(key)) |
        (uint8_t(chunk_delta(key) + META_DELTA_BIAS) << META_DELTA_SHIFT));
}

struct PackedChunks {
    uint32_t packed = 0;
    int nchunk = 0;
};

constexpr PackedChunks pack_chunks(uint32_t key, int k) {
    PackedChunks out{};
    int remain = k;
    int shift = 0;
    while (remain > 0) {
        const int len = remain >= CHUNK ? CHUNK : remain;
        remain -= len;
        const uint32_t chunk = (key / pow3(remain)) % pow3(len);
        out.packed |= chunk << shift;
        shift += 8;
        ++out.nchunk;
    }
    return out;
}

uint8_t composed_mask(uint32_t key, int k, uint32_t depth) {
    const PackedChunks pc = pack_chunks(key, k);
    uint32_t state = depth;
    uint32_t lbase = 0;
    uint8_t out = 0;
    for (int slot = 0; slot < pc.nchunk; ++slot) {
        const uint32_t chunk = (pc.packed >> (8 * slot)) & 0xffu;
        const uint8_t e = rankstream_entry(chunk, state);
        out = uint8_t(out | uint8_t(uint32_t(e & MASK_MASK) << lbase));
        if (((e >> HALT_SHIFT) & 1u) != 0u) return out;
        const uint8_t meta = rankstream_meta(chunk);
        lbase += uint32_t(meta & META_LCOUNT_MASK);
        state = uint32_t(
            int(state) + int(meta >> META_DELTA_SHIFT) - META_DELTA_BIAS);
    }
    return out;
}

uint8_t scalar_mask(uint32_t key, int k, uint32_t depth) {
    int ordinal_at[MAX_K]{};
    int ordinal = 0;
    for (int pos = k - 1; pos >= 0; --pos) {
        if (digit(key, pos) == L) ordinal_at[pos] = ordinal++;
        else ordinal_at[pos] = -1;
    }

    int state = int(depth);
    uint8_t out = 0;
    for (int pos = k - 1; pos >= 0; --pos) {
        const uint32_t v = digit(key, pos);
        if (v == R) {
            if (state == 1) break;
            --state;
        } else if (v == L) {
            if (state == 1)
                out = uint8_t(out | uint8_t(1u << ordinal_at[pos]));
            ++state;
        }
    }
    return out;
}

} // namespace

int main() {
    uint64_t cases = 0;
    uint8_t mask_or = 0;
    for (int k = 1; k <= MAX_K; ++k) {
        const uint32_t keys = pow3(k);
        for (uint32_t key = 0; key < keys; ++key) {
            for (uint32_t depth = 1; depth <= MAX_DEPTH; ++depth) {
                const uint8_t got = composed_mask(key, k, depth);
                const uint8_t want = scalar_mask(key, k, depth);
                if (got != want) {
                    std::cerr << "rankchunk32 directmask mismatch k=" << k
                              << " key=" << key << " depth=" << depth
                              << " got=" << unsigned(got)
                              << " want=" << unsigned(want) << '\n';
                    return 2;
                }
                mask_or = uint8_t(mask_or | got);
                ++cases;
            }
        }
    }
    if ((mask_or & 0x80u) != 0u) return 3;
    std::cout << "gridfp-rankchunk32-directmask-proof OK"
              << " cases=" << cases
              << " k_range=1..14"
              << " depth_range=1..15"
              << " mask_or=0x" << std::hex << unsigned(mask_or) << std::dec
              << " chunk_composition_exact=1"
              << " partial_chunk_exact=1"
              << " global_rank_ordinal_exact=1"
              << " max_mask_bits=7\n";
    return 0;
}
