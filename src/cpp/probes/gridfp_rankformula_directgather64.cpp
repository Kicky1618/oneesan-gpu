#include <array>
#include <cstdint>
#include <iostream>

static uint32_t choose_u32(int n, int k) {
    if (k < 0 || k > n) return 0;
    uint64_t r = 1;
    for (int i = 1; i <= k; ++i) r = r * uint64_t(n - k + i) / uint64_t(i);
    return uint32_t(r);
}
static uint32_t ballot(int n, int h) {
    if (n < 0 || h < 0 || ((n - h) & 1)) return 0;
    int u = (n - h) / 2;
    if (u < 0 || u > n) return 0;
    return choose_u32(n, u) - choose_u32(n, u - 1);
}
static uint32_t lpattern(int n, int h, uint32_t local) {
    uint32_t lp = 0; int s = h, rem = n;
    for (int ord = 0; ord < n; ++ord) {
        const uint32_t rc = s > 0 ? ballot(rem - 1, s - 1) : 0u;
        if (s > 0 && local < rc) --s;
        else { local -= rc; lp |= 1u << ord; ++s; }
        --rem;
    }
    return (s == 0 && local == 0) ? lp : 0xffffffffu;
}
static uint32_t rank_pattern(int n, int h, uint32_t lp) {
    uint32_t rank = 0; int s = h, rem = n;
    for (int ord = 0; ord < n; ++ord) {
        const bool is_l = ((lp >> ord) & 1u) != 0u;
        const uint32_t rc = s > 0 ? ballot(rem - 1, s - 1) : 0u;
        if (is_l) { rank += rc; ++s; }
        else { if (s <= 0) return 0xffffffffu; --s; }
        --rem;
    }
    return s == 0 ? rank : 0xffffffffu;
}
static uint32_t select_mask(uint32_t lp, int n, uint32_t depth) {
    uint32_t state = depth, li = 0, select = 0;
    for (int ord = 0; ord < n; ++ord) {
        if ((lp >> ord) & 1u) {
            if (state == 1u) select |= 1u << li;
            ++li; ++state;
        } else {
            if (state == 1u) break;
            --state;
        }
    }
    return select;
}

int main() {
    constexpr int K = 14;
    constexpr uint32_t TEST_SOURCE_BASE = 29113u; // W28 production max GROUP61 base.
    std::array<uint64_t, 8> dist{};
    uint64_t abstract_states = 0, abstract_depths = 0;
    uint64_t universal_selected_sources = 0, pack_mismatch = 0;
    uint32_t max_source_local = 0, max_absolute_source = 0;
    uint64_t descriptor_ordinal = 0;

    for (int n = 0; n <= K; ++n) {
        std::array<uint64_t, 8> by_n{};
        for (int h = 0; h < 16; ++h) {
            const uint32_t cnt = ballot(n, h);
            abstract_states += cnt;
            for (uint32_t j = 0; j < cnt; ++j) {
                const uint32_t lp = lpattern(n, h, j);
                if (lp == 0xffffffffu) return 2;
                for (uint32_t d = 1; d <= 13; ++d, ++descriptor_ordinal) {
                    const uint32_t select = select_mask(lp, n, d);
                    const uint32_t c = uint32_t(__builtin_popcount(select));
                    if (c > 7) return 3;
                    ++by_n[c]; ++abstract_depths;

                    std::array<uint32_t, 7> rr{};
                    uint32_t li = 0, nr = 0;
                    for (int ord = 0; ord < n; ++ord) {
                        if (!((lp >> ord) & 1u)) continue;
                        if ((select >> li) & 1u) {
                            const uint32_t sr = rank_pattern(n, h + 2, lp & ~(1u << ord));
                            if (sr == 0xffffffffu || sr > 1000u || nr >= 7u) return 8;
                            max_source_local = std::max(max_source_local, sr);
                            rr[nr] = TEST_SOURCE_BASE + sr;
                            if (rr[nr] >= 32768u) return 9;
                            max_absolute_source = std::max(max_absolute_source, rr[nr]);
                            ++nr; ++universal_selected_sources;
                        }
                        ++li;
                    }
                    if (nr != c) return 10;

                    const uint32_t fake_rare_ix = c > 3u
                        ? uint32_t((descriptor_ordinal * 17u + 11u) & 0xffffu)
                        : 0u;
                    const uint64_t p = uint64_t(rr[0]) |
                        (uint64_t(rr[1]) << 15) |
                        (uint64_t(rr[2]) << 30) |
                        (uint64_t(c) << 45) |
                        (uint64_t(fake_rare_ix) << 48);
                    const uint64_t q = uint64_t(rr[3]) |
                        (uint64_t(rr[4]) << 15) |
                        (uint64_t(rr[5]) << 30) |
                        (uint64_t(rr[6]) << 45);

                    std::array<uint32_t, 7> got{};
                    const uint32_t got_count = uint32_t((p >> 45) & 7u);
                    const uint32_t got_rare_ix = uint32_t(p >> 48);
                    got[0] = uint32_t(p & 0x7fffu);
                    got[1] = uint32_t((p >> 15) & 0x7fffu);
                    got[2] = uint32_t((p >> 30) & 0x7fffu);
                    got[3] = uint32_t(q & 0x7fffu);
                    got[4] = uint32_t((q >> 15) & 0x7fffu);
                    got[5] = uint32_t((q >> 30) & 0x7fffu);
                    got[6] = uint32_t((q >> 45) & 0x7fffu);
                    if (got_count != c || got_rare_ix != fake_rare_ix) ++pack_mismatch;
                    for (uint32_t z = 0; z < c; ++z) if (got[z] != rr[z]) ++pack_mismatch;
                }
            }
        }
        const uint64_t mult = choose_u32(K, n);
        for (int c = 0; c <= 7; ++c) dist[c] += mult * by_n[c];
    }

    uint64_t total = 0, selected_total = 0, rare = 0;
    for (int c = 0; c <= 7; ++c) {
        total += dist[c]; selected_total += uint64_t(c) * dist[c];
        if (c >= 4) rare += dist[c];
    }
    const uint64_t old_bytes = total * 16u;
    const uint64_t primary_bytes = total * 8u;
    const uint64_t rare_bytes = rare * 8u;
    const uint64_t compact_bytes = primary_bytes + rare_bytes;

    std::cout << "rankformula-directgather64-proof"
              << " abstract_states=" << abstract_states
              << " abstract_depths=" << abstract_depths
              << " universal_selected_sources=" << universal_selected_sources
              << " max_source_local=" << max_source_local
              << " test_source_base=" << TEST_SOURCE_BASE
              << " max_absolute_source=" << max_absolute_source
              << " pack_mismatch=" << pack_mismatch
              << " production_descriptors=" << total
              << " selected_sources=" << selected_total
              << " count0=" << dist[0]
              << " count1=" << dist[1]
              << " count2=" << dist[2]
              << " count3=" << dist[3]
              << " count4=" << dist[4]
              << " count5=" << dist[5]
              << " count6=" << dist[6]
              << " count7=" << dist[7]
              << " rare_ge4=" << rare
              << " old_uint4_bytes=" << old_bytes
              << " primary64_bytes=" << primary_bytes
              << " rare64_bytes=" << rare_bytes
              << " compact_bytes=" << compact_bytes
              << " reduction_pct=" << (100.0 * double(old_bytes - compact_bytes) / double(old_bytes))
              << " source_rank_bits=15 rare_index_bits=16"
              << " exact=1\n";
    if (abstract_states != 7060 || abstract_depths != 91780) return 4;
    if (universal_selected_sources != 19273 || max_source_local != 1000u ||
        max_absolute_source != 30113u || pack_mismatch != 0) return 11;
    if (total != 15624921 || selected_total != 2492769) return 5;
    if (dist != std::array<uint64_t,8>{13867748,1195735,416811,118082,23689,2727,128,1}) return 6;
    if (rare != 26545) return 7;
    return 0;
}
