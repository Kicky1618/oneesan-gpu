#include <cstdint>
#include <iostream>

static constexpr int K = 14;
static constexpr uint32_t BLOCK = 32;
static constexpr uint32_t POW3[] = {
    1u, 3u, 9u, 27u, 81u, 243u, 729u, 2187u,
    6561u, 19683u, 59049u, 177147u, 531441u, 1594323u, 4782969u
};

static uint32_t pack24(uint32_t key) {
    constexpr int l0 = 5, s0 = K - l0;
    constexpr int l1 = 5, s1 = s0 - l1;
    constexpr int l2 = s1;
    static_assert(l2 == 4);
    const uint32_t c0 = (key / POW3[s0]) % POW3[l0];
    const uint32_t c1 = (key / POW3[s1]) % POW3[l1];
    const uint32_t c2 = key % POW3[l2];
    return c0 | (c1 << 8) | (c2 << 16);
}

static uint32_t unpack24(uint32_t packed) {
    return (packed & 0xffu) * POW3[9]
         + ((packed >> 8) & 0xffu) * POW3[4]
         + ((packed >> 16) & 0xffu);
}

int main() {
    uint32_t max_l_legal = 0;
    uint32_t max_c2 = 0;
    uint64_t legal_balance_codes = 0;
    uint64_t roundtrip = 0;
    uint64_t bit23_zero = 0;

    for (uint32_t key = 0; key < POW3[K]; ++key) {
        uint32_t x = key;
        uint32_t nr = 0, nl = 0;
        for (int pos = 0; pos < K; ++pos) {
            const uint32_t d = x % 3u;
            x /= 3u;
            if (d == 1u) ++nr;
            if (d == 2u) ++nl;
        }
        // Any LOW code accepted by the factor builder starts at h0>=0 and
        // finishes at zero, hence #R-#L=h0>=0.  This necessary condition is
        // already enough to prove the per-code L bound used by bytepack.
        if (nr >= nl) {
            ++legal_balance_codes;
            if (nl > max_l_legal) max_l_legal = nl;
        }

        const uint32_t packed = pack24(key);
        const uint32_t c2 = (packed >> 16) & 0xffu;
        if (c2 > max_c2) max_c2 = c2;
        if (unpack24(packed) == key) ++roundtrip;
        if ((packed & 0x00800000u) == 0u) ++bit23_zero;
    }

    const uint32_t max_prefix = (BLOCK - 1u) * max_l_legal;
    if (max_l_legal != 7u) return 2;
    if (max_prefix != 217u || max_prefix >= 256u) return 3;
    if (max_c2 != 80u) return 4;
    if (roundtrip != POW3[K] || bit23_zero != POW3[K]) return 5;

    std::cout << "gridfp-rankchunk32-bytepack OK"
              << " K=" << K
              << " ternary_keys=" << POW3[K]
              << " legal_balance_codes=" << legal_balance_codes
              << " max_l_per_legal_code=" << max_l_legal
              << " block=" << BLOCK
              << " max_prefix=" << max_prefix
              << " prefix8_exact=1"
              << " chunk_bits=24 prefix_bits=8"
              << " byte_aligned_chunks=1"
              << " max_third_chunk=" << max_c2
              << " bit23_always_zero=1"
              << " pack_roundtrip_exact=1\n";
    return 0;
}
