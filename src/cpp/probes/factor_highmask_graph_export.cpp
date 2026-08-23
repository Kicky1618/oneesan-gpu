#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../common/gridfp_transition.hpp"

using oneesan::gridfp::MateID;
using oneesan::gridfp::MateValue;
using oneesan::gridfp::N;
using oneesan::gridfp::R;
using oneesan::gridfp::L;
using U64 = std::uint64_t;

static std::uint32_t occ(std::uint32_t code, int len) {
    std::uint32_t z = 0;
    for (int p = 0; p < len; ++p)
        if ((code >> (2 * p)) & 3u) z |= 1u << p;
    return z;
}

struct LowClass {
    U64 count = 0;
    std::uint32_t representative = 0xffffffffu;
};

static std::vector<LowClass> enumerate_low_classes(int len) {
    std::vector<LowClass> out(len + 2);
    for (int h0 = 0; h0 <= len + 1; ++h0) {
        auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
            if (pos < 0) {
                if (h == 0) {
                    ++out[h0].count;
                    if (out[h0].representative == 0xffffffffu)
                        out[h0].representative = code;
                }
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, code);
            if (h > 0) self(self, pos - 1, h - 1,
                            code | (std::uint32_t(R) << (2 * pos)));
            self(self, pos - 1, h + 1,
                 code | (std::uint32_t(L) << (2 * pos)));
        };
        rec(rec, len - 1, h0, 0);
    }
    return out;
}

static std::vector<std::vector<std::uint32_t>> enumerate_high(int len) {
    std::vector<std::vector<std::uint32_t>> by_h(len + 2);
    auto rec = [&](auto&& self, int pos, int h, std::uint32_t code) -> void {
        if (pos < 0) {
            if (h >= 0 && h < int(by_h.size())) by_h[h].push_back(code);
            return;
        }
        self(self, pos - 1, h, code);
        if (h > 0) self(self, pos - 1, h - 1,
                        code | (std::uint32_t(R) << (2 * pos)));
        self(self, pos - 1, h + 1,
             code | (std::uint32_t(L) << (2 * pos)));
    };
    rec(rec, len - 1, 1, 0);
    return by_h;
}

static U64 edge_key(std::uint32_t a, std::uint32_t b) {
    if (a > b) std::swap(a, b);
    return (U64(a) << 32) | U64(b);
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 28;
    const int low = argc > 2 ? std::atoi(argv[2]) : 14;
    const std::string prefix = argc > 3 ? argv[3] : "";
    const int high = W - 1 - low;
    if (W < 3 || W > 28 || low < 1 || high < 1 || high > 15) {
        std::cerr << "usage: factor_highmask_graph_export [W<=28] [LOW] [output-prefix]\n";
        return 1;
    }

    const auto lows = enumerate_low_classes(low);
    const auto highs = enumerate_high(high);
    const std::uint32_t nm = 1u << high;
    const MateID high_code_mask = (MateID(1) << (2 * high)) - 1;

    std::vector<U64> node_weight(nm);
    std::unordered_map<U64, U64> edge_weight;
    edge_weight.reserve(size_t(nm) * 24);
    U64 total_states = 0;
    U64 total_transition_updates = 0;
    U64 local_mask_updates = 0;

    auto add_transition = [&](std::uint32_t sm, std::uint32_t tm, U64 weight) {
        total_transition_updates += weight;
        if (sm == tm) {
            local_mask_updates += weight;
            return;
        }
        edge_weight[edge_key(sm, tm)] += weight;
    };

    for (int he = 0; he < int(highs.size()); ++he) {
        for (std::uint32_t hc : highs[he]) {
            const std::uint32_t sm = occ(hc, high);
            for (int cv = 0; cv < 3; ++cv) {
                const int hs = he + (cv == int(L) ? 1 : cv == int(R) ? -1 : 0);
                if (hs < 0 || hs >= int(lows.size()) || !lows[hs].count) continue;
                const U64 weight = lows[hs].count;
                node_weight[sm] += weight;
                total_states += weight;
                const std::uint32_t lc = lows[hs].representative;
                const MateID m = MateID(lc)
                    | (MateID(cv) << (2 * low))
                    | (MateID(hc) << (2 * (low + 1)));
                for (int p = W - 1; p >= low + 1; --p) {
                    const auto z = oneesan::gridfp::include_horizontal(m, W, p);
                    if (!z.valid) continue;
                    const int shift = z.blocked ? low : low + 1;
                    const std::uint32_t hc2 = std::uint32_t(
                        (z.mate >> (2 * shift)) & high_code_mask);
                    add_transition(sm, occ(hc2, high), weight);
                }
            }
        }
    }

    for (int he = 0; he < int(highs.size()) && he < int(lows.size()); ++he) {
        if (!lows[he].count) continue;
        const U64 weight = lows[he].count;
        const std::uint32_t lc = lows[he].representative;
        for (std::uint32_t hc : highs[he]) {
            const std::uint32_t sm = occ(hc, high);
            node_weight[sm] += weight;
            total_states += weight;
            const MateID m = MateID(lc) | (MateID(hc) << (2 * low));
            for (int p = W - 1; p >= low + 1; --p) {
                const MateID z = oneesan::gridfp::blocked_exclude(m, p);
                const std::uint32_t hc2 = std::uint32_t(
                    (z >> (2 * (low + 1))) & high_code_mask);
                add_transition(sm, occ(hc2, high), weight);
            }
        }
    }

    U64 cuttable_update_weight = 0;
    for (const auto& kv : edge_weight) cuttable_update_weight += kv.second;

    if (W == 28 && low == 14) {
        if (nm != 8192u
            || total_states != 520735012027ULL
            || total_transition_updates != 6154161750113ULL
            || local_mask_updates != 73007659168ULL
            || edge_weight.size() != 139267u
            || cuttable_update_weight != 6081154090945ULL) {
            std::cerr << "n=27 HIGH-mask graph regression mismatch\n";
            return 2;
        }
    }

    std::cout << "factor-highmask-graph W=" << W
              << " low=" << low << " high=" << high
              << " vertices=" << nm
              << " edges=" << edge_weight.size()
              << " states=" << total_states
              << " transition_updates=" << total_transition_updates
              << " same_mask_updates=" << local_mask_updates
              << " cuttable_updates=" << cuttable_update_weight << '\n';

    if (prefix.empty()) return 0;

    std::ofstream nodes(prefix + ".nodes.tsv");
    std::ofstream edges(prefix + ".edges.tsv");
    std::ofstream meta(prefix + ".meta.tsv");
    if (!nodes || !edges || !meta) {
        std::cerr << "cannot open output prefix: " << prefix << '\n';
        return 3;
    }

    meta << "key\tvalue\n"
         << "width\t" << W << '\n'
         << "low\t" << low << '\n'
         << "high\t" << high << '\n'
         << "vertices\t" << nm << '\n'
         << "edges\t" << edge_weight.size() << '\n'
         << "states\t" << total_states << '\n'
         << "transition_updates\t" << total_transition_updates << '\n'
         << "same_mask_updates\t" << local_mask_updates << '\n'
         << "cuttable_updates\t" << cuttable_update_weight << '\n';

    nodes << "mask\tstate_weight\n";
    for (std::uint32_t m = 0; m < nm; ++m)
        nodes << m << '\t' << node_weight[m] << '\n';

    std::vector<std::pair<U64, U64>> sorted(edge_weight.begin(), edge_weight.end());
    std::sort(sorted.begin(), sorted.end(), [](const auto& x, const auto& y) {
        return x.first < y.first;
    });
    edges << "mask_a\tmask_b\ttransition_weight\n";
    for (const auto& kv : sorted) {
        const std::uint32_t a = std::uint32_t(kv.first >> 32);
        const std::uint32_t b = std::uint32_t(kv.first);
        edges << a << '\t' << b << '\t' << kv.second << '\n';
    }

    std::cout << "meta_tsv=" << prefix << ".meta.tsv"
              << " nodes_tsv=" << prefix << ".nodes.tsv"
              << " edges_tsv=" << prefix << ".edges.tsv\n";
    return 0;
}
