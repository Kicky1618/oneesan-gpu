#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

static constexpr int W = 28;
static constexpr int L = W / 2;
static constexpr int H = W - L - 1;
static constexpr int S = W + 2;
static constexpr int NG = 8;
static constexpr uint32_t INVALID = 0xffffffffu;
static constexpr uint16_t RANK16_INVALID = 0xffffu;
static_assert(L == 14 && H == 13);

enum V : uint32_t { N = 0, R = 1, LL = 2 };

static uint32_t occ(uint32_t code, int len) {
    uint32_t m = 0;
    for (int p = 0; p < len; ++p)
        if (((code >> (2 * p)) & 3u) != 0) m |= 1u << p;
    return m;
}

static uint32_t ternary_key(uint32_t code, int len) {
    uint32_t key = 0, w = 1;
    for (int p = 0; p < len; ++p) {
        key += ((code >> (2 * p)) & 3u) * w;
        w *= 3u;
    }
    return key;
}

static uint32_t pow3(int n) {
    uint32_t z = 1;
    while (n--) z *= 3u;
    return z;
}

struct Factors {
    std::array<std::vector<uint32_t>, S> low_h;
    std::array<std::vector<uint32_t>, S> high_h;
    std::vector<std::vector<uint32_t>> low_mask_h;
};

static Factors build_factors() {
    Factors f;
    f.low_mask_h.resize(size_t(1u << L) * S);
    for (int h0 = 0; h0 <= L + 1; ++h0) {
        auto rec = [&](auto&& self, int pos, int h, uint32_t code) -> void {
            if (pos < 0) {
                if (h == 0) {
                    f.low_h[h0].push_back(code);
                    f.low_mask_h[size_t(occ(code, L)) * S + h0].push_back(code);
                }
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, code);
            if (h > 0) self(self, pos - 1, h - 1, code | (uint32_t(R) << (2 * pos)));
            self(self, pos - 1, h + 1, code | (uint32_t(LL) << (2 * pos)));
        };
        rec(rec, L - 1, h0, 0);
    }
    auto rech = [&](auto&& self, int pos, int h, uint32_t code) -> void {
        if (pos < 0) {
            if (h >= 0 && h < S) f.high_h[h].push_back(code);
            return;
        }
        self(self, pos - 1, h, code);
        if (h > 0) self(self, pos - 1, h - 1, code | (uint32_t(R) << (2 * pos)));
        self(self, pos - 1, h + 1, code | (uint32_t(LL) << (2 * pos)));
    };
    rech(rech, H - 1, 1, 0);
    return f;
}

static std::vector<uint8_t> low_mask_owners(const Factors& f) {
    const uint32_t LM = 1u << L;
    std::vector<uint64_t> weight(LM);
    for (uint32_t m = 0; m < LM; ++m) {
        uint64_t z = 0;
        for (int h = 0; h <= H + 1; ++h) {
            uint64_t ht = f.high_h[h].size();
            auto lc = [&](int x) -> uint64_t {
                if (x < 0 || x >= S) return 0;
                return f.low_mask_h[size_t(m) * S + size_t(x)].size();
            };
            z += ht * (2 * lc(h) + lc(h - 1) + lc(h + 1));
        }
        weight[m] = z;
    }
    std::vector<std::pair<uint64_t, uint32_t>> order;
    order.reserve(LM);
    for (uint32_t m = 0; m < LM; ++m) order.push_back({weight[m], m});
    std::sort(order.begin(), order.end(), [](auto a, auto b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    std::array<uint64_t, NG> load{};
    std::vector<uint8_t> owner(LM);
    for (auto [w, m] : order) {
        int g = 0;
        for (int j = 1; j < NG; ++j) if (load[j] < load[g]) g = j;
        owner[m] = uint8_t(g);
        load[g] += w;
    }
    return owner;
}

struct LocalCode {
    uint32_t code = 0;
    uint16_t rank = 0;
    uint8_t h = 0;
    uint8_t owner = 0;
};

int main() {
    Factors f = build_factors();
    auto mask_owner = low_mask_owners(f);
    std::array<std::vector<LocalCode>, NG> codes;
    std::vector<uint32_t> direct(pow3(L), INVALID);
    uint64_t total_codes = 0;
    uint32_t max_local_rank = 0;

    for (int h = 0; h <= L + 1; ++h) {
        std::array<uint32_t, NG> next{};
        for (uint32_t m = 0; m < (1u << L); ++m) {
            uint32_t g = mask_owner[m];
            const auto& v = f.low_mask_h[size_t(m) * S + size_t(h)];
            for (uint32_t code : v) {
                uint32_t r = next[g]++;
                if (r >= RANK16_INVALID) return 2;
                codes[g].push_back(LocalCode{code, uint16_t(r), uint8_t(h), uint8_t(g)});
                uint32_t key = ternary_key(code, L);
                if (direct[key] != INVALID) return 3;
                direct[key] = (uint32_t(g) << 16) | r;
                max_local_rank = std::max(max_local_rank, r);
                ++total_codes;
            }
        }
    }

    std::array<uint64_t, NG> valid{}, l_digits{}, owner_mismatch{};
    uint64_t total_valid = 0, total_l = 0;
    for (int g = 0; g < NG; ++g) {
        for (const LocalCode& z : codes[g]) {
            uint32_t key = ternary_key(z.code, L), weight = 1;
            for (int pos = 0; pos < L; ++pos) {
                if (((z.code >> (2 * pos)) & 3u) == LL) {
                    ++l_digits[g]; ++total_l;
                    uint32_t x = direct[key - weight];
                    if (x == INVALID) {
                        std::cerr << "LOW L->R legality invariant failed owner=" << g
                                  << " h=" << unsigned(z.h) << " pos=" << pos << '\n';
                        return 4;
                    }
                    int owner = int(x >> 16);
                    if (owner != g) ++owner_mismatch[g];
                    ++valid[g]; ++total_valid;
                }
                weight *= 3u;
            }
        }
    }

    uint64_t max_dense_bytes = 0, max_stream_bytes = 0, max_prekey_bytes = 0;
    uint64_t total_dense_bytes = 0, total_stream_bytes = 0, total_prekey_bytes = 0;
    for (int g = 0; g < NG; ++g) {
        uint64_t n = codes[g].size();
        uint64_t dense = n * uint64_t(L) * sizeof(uint16_t);
        // Every L->R transition is legal, and the ternary key already exposes
        // each chunk's L positions. A sparse stream therefore needs only one
        // uint32 offset per code plus one uint16 rank per L digit; no valid mask.
        uint64_t stream = n * sizeof(uint32_t) + l_digits[g] * sizeof(uint16_t);
        uint64_t prekey = n * sizeof(uint32_t);
        max_dense_bytes = std::max(max_dense_bytes, dense);
        max_stream_bytes = std::max(max_stream_bytes, stream);
        max_prekey_bytes = std::max(max_prekey_bytes, prekey);
        total_dense_bytes += dense;
        total_stream_bytes += stream;
        total_prekey_bytes += prekey;
        std::cout << "owner=" << g
                  << " low_codes=" << n
                  << " l_digits=" << l_digits[g]
                  << " valid_rank16=" << valid[g]
                  << " l_per_code=" << (n ? double(l_digits[g]) / double(n) : 0.0)
                  << " dense_rank16_mib=" << double(dense) / double(1 << 20)
                  << " rankstream_mib=" << double(stream) / double(1 << 20)
                  << " prekey_mib=" << double(prekey) / double(1 << 20)
                  << " owner_mismatch=" << owner_mismatch[g] << '\n';
    }

    uint64_t mismatches = std::accumulate(owner_mismatch.begin(), owner_mismatch.end(), uint64_t(0));
    if (mismatches || total_valid != total_l) return 5;
    std::cout << "gridfp-low-rank16-plan OK"
              << " W=" << W << " low_k=" << L << " high_k=" << H
              << " low_codes=" << total_codes
              << " max_local_rank=" << max_local_rank
              << " l_digits=" << total_l
              << " valid_rank16=" << total_valid
              << " l_per_code=" << double(total_l) / double(total_codes)
              << " valid_fraction_of_slots=" << double(total_valid) / double(total_codes * L)
              << " max_dense_rank16_mib=" << double(max_dense_bytes) / double(1 << 20)
              << " max_rankstream_mib=" << double(max_stream_bytes) / double(1 << 20)
              << " max_prekey_mib=" << double(max_prekey_bytes) / double(1 << 20)
              << " total_dense_rank16_mib=" << double(total_dense_bytes) / double(1 << 20)
              << " total_rankstream_mib=" << double(total_stream_bytes) / double(1 << 20)
              << " total_prekey_mib=" << double(total_prekey_bytes) / double(1 << 20)
              << " owner_invariant=1 all_L_flips_legal=1"
              << " dense_bytes_per_code=" << (L * sizeof(uint16_t))
              << " rankstream_model=offset32+rank16_per_L\n";
    return 0;
}
