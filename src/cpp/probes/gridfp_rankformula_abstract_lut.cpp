#define main gridfp_rankformula_plan_main_unused
#include "gridfp_rankformula_plan.cpp"
#undef main

#include <limits>

static constexpr uint32_t LP_BITS = L;
static constexpr uint32_t LP_MASK = (1u << LP_BITS) - 1u;

static uint32_t abstract_lpattern(int n, int h, uint32_t local) {
    uint32_t lp = 0;
    int s = h, rem = n;
    for (int ord = 0; ord < n; ++ord) {
        const uint32_t rc = s > 0 ? ballot_suffix(rem - 1, s - 1) : 0u;
        if (s > 0 && local < rc) {
            --s;
        } else {
            if (local < rc) return INVALID;
            local -= rc;
            lp |= 1u << ord;
            ++s;
        }
        --rem;
    }
    return (s == 0 && local == 0) ? lp : INVALID;
}

static uint32_t abstract_rank(int n, int h, uint32_t lp) {
    uint32_t rank = 0;
    int s = h, rem = n;
    for (int ord = 0; ord < n; ++ord) {
        const bool is_l = ((lp >> ord) & 1u) != 0u;
        const uint32_t rc = s > 0 ? ballot_suffix(rem - 1, s - 1) : 0u;
        if (is_l) {
            rank += rc;
            ++s;
        } else {
            if (s <= 0) return INVALID;
            --s;
        }
        --rem;
    }
    return s == 0 ? rank : INVALID;
}

int main() {
    Factors f = build_factors();
    const auto owner = low_mask_owners(f);
    const uint32_t LM = 1u << L;

    std::array<uint16_t, (L + 1) * 16> off{};
    std::vector<uint32_t> desc;
    std::vector<uint16_t> src_local;
    uint32_t max_abstract_rank = 0, max_stream_off = 0;
    for (int n = 0; n <= L; ++n) {
        for (int h = 0; h < 16; ++h) {
            if (desc.size() >= 65536u) return 2;
            off[size_t(n) * 16u + size_t(h)] = uint16_t(desc.size());
            const uint32_t cnt = ballot_suffix(n, h);
            for (uint32_t j = 0; j < cnt; ++j) {
                const uint32_t lp = abstract_lpattern(n, h, j);
                if (lp == INVALID || lp > LP_MASK || src_local.size() >= (1u << 15)) return 3;
                const uint32_t stream_off = uint32_t(src_local.size());
                max_stream_off = std::max(max_stream_off, stream_off);
                desc.push_back(lp | (stream_off << LP_BITS));
                for (int ord = 0; ord < n; ++ord) {
                    if (((lp >> ord) & 1u) == 0u) continue;
                    const uint32_t src_lp = lp & ~(1u << ord);
                    const uint32_t sr = abstract_rank(n, h + 2, src_lp);
                    if (sr == INVALID || sr >= 65536u) return 4;
                    src_local.push_back(uint16_t(sr));
                    max_abstract_rank = std::max(max_abstract_rank, sr);
                }
            }
        }
    }

    std::vector<int32_t> base(size_t(NG) * S * LM, -1);
    auto bref = [&](int g, int h, uint32_t m) -> int32_t& {
        return base[(size_t(g) * S + size_t(h)) * LM + m];
    };
    std::vector<uint32_t> direct(pow3(L), INVALID);
    uint64_t total_codes = 0;
    for (int h = 0; h <= L + 1; ++h) {
        std::array<uint32_t, NG> next{};
        for (uint32_t m = 0; m < LM; ++m) {
            const int g = int(owner[m]);
            const auto& v = f.low_mask_h[size_t(m) * S + size_t(h)];
            if (!v.empty()) bref(g, h, m) = int32_t(next[g]);
            for (uint32_t code : v) {
                const uint32_t r = next[g]++;
                const uint32_t key = ternary_key(code, L);
                if (r >= RANK16_INVALID || direct[key] != INVALID) return 5;
                direct[key] = (uint32_t(g) << 16) | r;
                ++total_codes;
            }
        }
    }

    uint64_t exact_codes = 0, exact_transitions = 0;
    for (int h = 0; h <= L + 1; ++h) {
        for (uint32_t m = 0; m < LM; ++m) {
            const int g = int(owner[m]);
            const auto& v = f.low_mask_h[size_t(m) * S + size_t(h)];
            if (v.empty()) continue;
            const int n = __builtin_popcount(m);
            const uint32_t ao = uint32_t(off[size_t(n) * 16u + size_t(h)]);
            const int32_t sb = h + 2 <= L + 1 ? bref(g, h + 2, m) : -1;
            for (uint32_t j = 0; j < v.size(); ++j) {
                const uint32_t code = v[j];
                const uint32_t d = desc[ao + j];
                const uint32_t want_lp = d & LP_MASK;
                uint32_t got_lp = 0, ord = 0;
                for (int pos = L - 1; pos >= 0; --pos) {
                    if (((m >> pos) & 1u) == 0u) continue;
                    if (((code >> (2 * pos)) & 3u) == LL) got_lp |= 1u << ord;
                    ++ord;
                }
                if (got_lp != want_lp) {
                    std::cerr << "abstract lpattern mismatch g=" << g << " h=" << h
                              << " mask=" << m << " local=" << j << '\n';
                    return 6;
                }
                uint32_t rp = d >> LP_BITS;
                uint32_t weight = pow3(L - 1), seen = 0;
                const uint32_t key = ternary_key(code, L);
                for (int pos = L - 1; pos >= 0; --pos) {
                    if (((m >> pos) & 1u) != 0u && ((got_lp >> seen) & 1u)) {
                        if (sb < 0 || rp >= src_local.size()) return 7;
                        const uint32_t x = direct[key - weight];
                        if (x == INVALID || int(x >> 16) != g) return 8;
                        const uint32_t actual_local = uint32_t(int(x & 0xffffu) - sb);
                        const uint32_t want_local = uint32_t(src_local[rp++]);
                        if (actual_local != want_local) {
                            std::cerr << "abstract source mismatch g=" << g << " h=" << h
                                      << " mask=" << m << " local=" << j
                                      << " pos=" << pos << " got=" << actual_local
                                      << " want=" << want_local << '\n';
                            return 9;
                        }
                        ++exact_transitions;
                    }
                    if ((m >> pos) & 1u) ++seen;
                    weight /= 3u;
                }
                ++exact_codes;
            }
        }
    }

    const size_t desc_bytes = desc.size() * sizeof(uint32_t);
    const size_t rank_bytes = src_local.size() * sizeof(uint16_t);
    const size_t off_bytes = off.size() * sizeof(uint16_t);
    if (total_codes != 1201917ull || exact_codes != total_codes ||
        exact_transitions != 3720805ull || desc.size() != 7060u ||
        src_local.size() != 32743u || max_abstract_rank != 1000u) return 10;
    std::cout << "gridfp-rankformula-abstract-lut OK"
              << " production_codes=" << total_codes
              << " production_transitions=" << exact_transitions
              << " abstract_states=" << desc.size()
              << " abstract_transitions=" << src_local.size()
              << " descriptor_bytes=" << desc_bytes
              << " source_rank_bytes=" << rank_bytes
              << " offset_bytes=" << off_bytes
              << " total_lut_bytes=" << (desc_bytes + rank_bytes + off_bytes)
              << " max_source_local_rank=" << max_abstract_rank
              << " max_stream_offset=" << max_stream_off
              << " descriptor_bits=29"
              << " mask_position_independent=1"
              << " all_production_codes_exact=1"
              << " all_production_transitions_exact=1\n";
    return 0;
}
