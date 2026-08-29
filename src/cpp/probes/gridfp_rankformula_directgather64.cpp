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
static uint32_t selected(uint32_t lp, int n, uint32_t depth) {
    uint32_t state = depth, count = 0;
    for (int ord = 0; ord < n; ++ord) {
        if ((lp >> ord) & 1u) {
            if (state == 1u) ++count;
            ++state;
        } else {
            if (state == 1u) break;
            --state;
        }
    }
    return count;
}
int main() {
    constexpr int K = 14;
    std::array<uint64_t, 8> dist{};
    uint64_t abstract_states = 0, abstract_depths = 0;
    for (int n = 0; n <= K; ++n) {
        std::array<uint64_t, 8> by_n{};
        for (int h = 0; h < 16; ++h) {
            const uint32_t cnt = ballot(n, h);
            abstract_states += cnt;
            for (uint32_t j = 0; j < cnt; ++j) {
                const uint32_t lp = lpattern(n, h, j);
                if (lp == 0xffffffffu) return 2;
                for (uint32_t d = 1; d <= 13; ++d) {
                    const uint32_t c = selected(lp, n, d);
                    if (c > 7) return 3;
                    ++by_n[c]; ++abstract_depths;
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
    if (total != 15624921 || selected_total != 2492769) return 5;
    if (dist != std::array<uint64_t,8>{13867748,1195735,416811,118082,23689,2727,128,1}) return 6;
    if (rare != 26545) return 7;
    return 0;
}
