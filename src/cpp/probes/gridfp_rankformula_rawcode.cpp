#include <cstdint>
#include <iostream>

static constexpr int K = 14;
static constexpr uint32_t POW3_K = 4782969u;

static uint32_t ternary_to_code(uint32_t key) {
    uint32_t code = 0;
    for (int pos = 0; pos < K; ++pos) {
        const uint32_t d = key % 3u;
        key /= 3u;
        code |= d << (2 * pos);
    }
    return code;
}

static uint32_t support_ref(uint32_t code) {
    uint32_t mask = 0;
    for (int pos = 0; pos < K; ++pos)
        if (((code >> (2 * pos)) & 3u) != 0u) mask |= 1u << pos;
    return mask;
}

static uint32_t support_broadword(uint32_t code) {
    uint32_t x = (code | (code >> 1)) & 0x55555555u;
    x = (x | (x >> 1)) & 0x33333333u;
    x = (x | (x >> 2)) & 0x0f0f0f0fu;
    x = (x | (x >> 4)) & 0x00ff00ffu;
    x = (x | (x >> 8)) & 0x0000ffffu;
    return x & ((1u << K) - 1u);
}

static uint32_t key_ref(uint32_t code10, int len) {
    uint32_t key = 0, w = 1;
    for (int pos = 0; pos < len; ++pos) {
        const uint32_t d = (code10 >> (2 * pos)) & 3u;
        key += d * w;
        w *= 3u;
    }
    return key;
}

// For two base-4 digits d0+4*d1, the base-3 value is d0+3*d1 = x-d1.
// Convert two pairs, then the fifth digit. Valid ternary digits are only 0..2.
static uint32_t key_fast5(uint32_t code10) {
    const uint32_t p0 = (code10 & 0x0fu) - ((code10 >> 2) & 3u);
    const uint32_t p1 = ((code10 >> 4) & 0x0fu) - ((code10 >> 6) & 3u);
    const uint32_t d4 = (code10 >> 8) & 3u;
    return p0 + 9u * p1 + 81u * d4;
}

static uint32_t mask_bits(int len) {
    return len == 16 ? 0xffffffffu : ((1u << (2 * len)) - 1u);
}

int main() {
    uint64_t support_exact = 0, chunks_exact = 0;
    uint32_t max_chunk_key = 0;
    for (uint32_t key = 0; key < POW3_K; ++key) {
        const uint32_t code = ternary_to_code(key);
        if (support_broadword(code) != support_ref(code)) {
            std::cerr << "rankformula rawcode support mismatch key=" << key << '\n';
            return 2;
        }
        ++support_exact;

        constexpr int L0 = 5, S0 = K - L0;
        constexpr int L1 = 5, S1 = S0 - L1;
        constexpr int L2 = S1;
        const uint32_t x0 = (code >> (2 * S0)) & mask_bits(L0);
        const uint32_t x1 = (code >> (2 * S1)) & mask_bits(L1);
        const uint32_t x2 = code & mask_bits(L2);
        const uint32_t k0 = key_fast5(x0), k1 = key_fast5(x1), k2 = key_fast5(x2);
        if (k0 != key_ref(x0, L0) || k1 != key_ref(x1, L1) || k2 != key_ref(x2, L2)) {
            std::cerr << "rankformula rawcode chunk mismatch key=" << key << '\n';
            return 3;
        }
        max_chunk_key = std::max(max_chunk_key, std::max(k0, std::max(k1, k2)));
        ++chunks_exact;
    }
    std::cout << "gridfp-rankformula-rawcode OK"
              << " K=" << K
              << " codes=" << POW3_K
              << " support_exact=" << support_exact
              << " chunks_exact=" << chunks_exact
              << " max_chunk_key=" << max_chunk_key
              << " metadata_bits=" << (2 * K)
              << " metadata_bytes=4"
              << " chunkinfo_loads=0"
              << " runtime_div=0 runtime_mod=0\n";
    return 0;
}
