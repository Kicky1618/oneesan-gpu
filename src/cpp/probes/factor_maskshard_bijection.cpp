#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <vector>

using U64 = std::uint64_t;
using U32 = std::uint32_t;

static constexpr int R = 1;
static constexpr int L = 2;

struct Tables {
    int W = 0;
    int low = 0;
    int high = 0;
    std::vector<std::vector<U32>> low_all, high_all;
    std::vector<std::vector<std::vector<U32>>> low_mask, high_mask;
    std::unordered_map<U32, U32> low_pack, high_pack;
};

static U32 occupancy(U32 code, int n) {
    U32 z = 0;
    for (int p = 0; p < n; ++p)
        if ((code >> (2 * p)) & 3u) z |= 1u << p;
    return z;
}

static Tables build_tables(int W, int low) {
    const int high = W - 1 - low;
    Tables t;
    t.W = W;
    t.low = low;
    t.high = high;
    t.low_all.resize(W + 3);
    t.high_all.resize(W + 3);
    t.low_mask.assign(1u << low, std::vector<std::vector<U32>>(W + 3));
    t.high_mask.assign(1u << high, std::vector<std::vector<U32>>(W + 3));

    for (int h0 = 0; h0 <= low + 1; ++h0) {
        auto rec = [&](auto&& self, int pos, int h, U32 code) -> void {
            if (pos < 0) {
                if (h == 0) {
                    t.low_all[h0].push_back(code);
                    t.low_mask[occupancy(code, low)][h0].push_back(code);
                }
                return;
            }
            if (h < 0 || h > pos + 1) return;
            self(self, pos - 1, h, code);
            if (h > 0) self(self, pos - 1, h - 1, code | (U32(R) << (2 * pos)));
            self(self, pos - 1, h + 1, code | (U32(L) << (2 * pos)));
        };
        rec(rec, low - 1, h0, 0);
    }

    auto rec_high = [&](auto&& self, int pos, int h, U32 code) -> void {
        if (pos < 0) {
            t.high_all[h].push_back(code);
            t.high_mask[occupancy(code, high)][h].push_back(code);
            return;
        }
        self(self, pos - 1, h, code);
        if (h > 0) self(self, pos - 1, h - 1, code | (U32(R) << (2 * pos)));
        self(self, pos - 1, h + 1, code | (U32(L) << (2 * pos)));
    };
    rec_high(rec_high, high - 1, 1, 0);

    // Rebuild the authoritative storage ranks independently: for each height,
    // occupancy masks are contiguous and mask-local order is preserved.
    for (int h = 0; h < W + 3; ++h) {
        U32 storage_rank = 0;
        for (U32 mask = 0; mask < (1u << low); ++mask) {
            const auto& v = t.low_mask[mask][h];
            for (U32 r = 0; r < v.size(); ++r)
                t.low_pack[v[r]] = (storage_rank++ << low) | r;
        }
        storage_rank = 0;
        for (U32 mask = 0; mask < (1u << high); ++mask) {
            const auto& v = t.high_mask[mask][h];
            for (U32 r = 0; r < v.size(); ++r)
                t.high_pack[v[r]] = (storage_rank++ << high) | r;
        }
    }
    return t;
}

struct Block {
    int he = 0, hs = 0, center = 0;
    U32 rows = 0, cols = 0;
};

struct Layout {
    std::vector<Block> main_blocks, block_blocks;
    std::vector<U64> main_group, block_group;
    std::vector<std::vector<U64>> main_block_off, block_block_off;
    std::vector<int> owner;
    std::vector<U64> main_base, block_base;
    std::array<U64, 8> gpu_main{}, gpu_block{};
};

static Layout build_layout(const Tables& t, int ngpu) {
    Layout s;
    const int H = t.high, Lw = t.low;
    for (int he = 0; he <= H + 1; ++he) {
        for (int cv = 0; cv < 3; ++cv) {
            const int hs = he + (cv == L ? 1 : cv == R ? -1 : 0);
            const U32 cols = (0 <= hs && hs <= Lw + 1) ? U32(t.low_all[hs].size()) : 0;
            const U32 rows = cols ? U32(t.high_all[he].size()) : 0;
            s.main_blocks.push_back({he, hs, cv, rows, cols});
        }
    }
    for (int h = 0; h <= H + 1; ++h)
        s.block_blocks.push_back({h, h, 0, U32(t.high_all[h].size()), U32(t.low_all[h].size())});

    const U32 nmasks = 1u << H;
    s.main_group.resize(nmasks);
    s.block_group.resize(nmasks);
    s.main_block_off.resize(nmasks);
    s.block_block_off.resize(nmasks);
    for (U32 mask = 0; mask < nmasks; ++mask) {
        U64 off = 0;
        for (const Block& b : s.main_blocks) {
            s.main_block_off[mask].push_back(off);
            off += U64(t.high_mask[mask][b.he].size()) * b.cols;
        }
        s.main_group[mask] = off;
        off = 0;
        for (const Block& b : s.block_blocks) {
            s.block_block_off[mask].push_back(off);
            off += U64(t.high_mask[mask][b.he].size()) * b.cols;
        }
        s.block_group[mask] = off;
    }

    std::vector<U32> order(nmasks);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](U32 a, U32 b) {
        const U64 wa = s.main_group[a] + s.block_group[a];
        const U64 wb = s.main_group[b] + s.block_group[b];
        return wa != wb ? wa > wb : a < b;
    });
    std::array<U64, 8> load{};
    s.owner.resize(nmasks);
    for (U32 mask : order) {
        int d = 0;
        for (int q = 1; q < ngpu; ++q) if (load[q] < load[d]) d = q;
        s.owner[mask] = d;
        load[d] += s.main_group[mask] + s.block_group[mask];
    }

    s.main_base.resize(nmasks);
    s.block_base.resize(nmasks);
    for (U32 mask = 0; mask < nmasks; ++mask) {
        const int d = s.owner[mask];
        s.main_base[mask] = s.gpu_main[d];
        s.block_base[mask] = s.gpu_block[d];
        s.gpu_main[d] += s.main_group[mask];
        s.gpu_block[d] += s.block_group[mask];
    }
    return s;
}

static U32 low_storage_code(const Tables& t, int h, U32 storage_rank) {
    U32 left = storage_rank;
    for (U32 mask = 0; mask < (1u << t.low); ++mask) {
        const auto& v = t.low_mask[mask][h];
        if (left < v.size()) return v[left];
        left -= U32(v.size());
    }
    std::cerr << "low storage rank out of range\n";
    std::exit(2);
}

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int low = argc > 2 ? std::atoi(argv[2]) : 5;
    const int ngpu = argc > 3 ? std::atoi(argv[3]) : 8;
    const int high = W - 1 - low;
    if (W < 4 || W > 14 || low < 1 || high < 1 || ngpu < 1 || ngpu > 8) return 1;

    const Tables t = build_tables(W, low);
    const Layout s = build_layout(t, ngpu);
    const U32 HR = (1u << high) - 1u;
    U64 low_main = 0, low_block = 0;

    // LOW window: the local group rank must already be owner-local authoritative rank.
    for (U32 mask = 0; mask < (1u << high); ++mask) {
        for (size_t bid = 0; bid < s.main_blocks.size(); ++bid) {
            const Block b = s.main_blocks[bid];
            if (!b.cols) continue;
            const auto& hv = t.high_mask[mask][b.he];
            for (U32 hr = 0; hr < hv.size(); ++hr) {
                const U32 hc = hv[hr];
                if ((t.high_pack.at(hc) & HR) != hr) return 2;
                for (U32 lr = 0; lr < b.cols; ++lr) {
                    const U32 lc = low_storage_code(t, b.hs, lr);
                    const U32 mr = t.high_pack.at(hc) & HR;
                    const U32 lar = t.low_pack.at(lc) >> low;
                    const U64 direct = s.main_base[mask] + s.main_block_off[mask][bid]
                                     + U64(hr) * b.cols + lr;
                    const U64 routed = s.main_base[mask] + s.main_block_off[mask][bid]
                                     + U64(mr) * b.cols + lar;
                    if (direct != routed) return 3;
                    ++low_main;
                }
            }
        }
        for (size_t bid = 0; bid < s.block_blocks.size(); ++bid) {
            const Block b = s.block_blocks[bid];
            if (!b.cols) continue;
            const auto& hv = t.high_mask[mask][b.he];
            for (U32 hr = 0; hr < hv.size(); ++hr) {
                const U32 hc = hv[hr];
                for (U32 lr = 0; lr < b.cols; ++lr) {
                    const U32 lc = low_storage_code(t, b.hs, lr);
                    const U64 direct = s.block_base[mask] + s.block_block_off[mask][bid]
                                     + U64(hr) * b.cols + lr;
                    const U64 routed = s.block_base[mask] + s.block_block_off[mask][bid]
                                     + U64(t.high_pack.at(hc) & HR) * b.cols
                                     + (t.low_pack.at(lc) >> low);
                    if (direct != routed) return 4;
                    ++low_block;
                }
            }
        }
    }

    // HIGH window: reconstruct the exact device route from all-rank/mask-rank pairs.
    U64 high_main = 0, high_block = 0;
    for (U32 low_mask = 0; low_mask < (1u << low); ++low_mask) {
        for (size_t bid = 0; bid < s.main_blocks.size(); ++bid) {
            const Block b = s.main_blocks[bid];
            if (!b.cols) continue;
            for (U32 hc : t.high_all[b.he]) {
                const U32 hm = occupancy(hc, high);
                const U32 mr = t.high_pack.at(hc) & HR;
                for (U32 lc : t.low_mask[low_mask][b.hs]) {
                    const U32 lar = t.low_pack.at(lc) >> low;
                    const U64 off = s.main_base[hm] + s.main_block_off[hm][bid]
                                  + U64(mr) * b.cols + lar;
                    if (off >= s.gpu_main[s.owner[hm]]) return 5;
                    ++high_main;
                }
            }
        }
        for (size_t bid = 0; bid < s.block_blocks.size(); ++bid) {
            const Block b = s.block_blocks[bid];
            if (!b.cols) continue;
            for (U32 hc : t.high_all[b.he]) {
                const U32 hm = occupancy(hc, high);
                const U32 mr = t.high_pack.at(hc) & HR;
                for (U32 lc : t.low_mask[low_mask][b.hs]) {
                    const U32 lar = t.low_pack.at(lc) >> low;
                    const U64 off = s.block_base[hm] + s.block_block_off[hm][bid]
                                  + U64(mr) * b.cols + lar;
                    if (off >= s.gpu_block[s.owner[hm]]) return 6;
                    ++high_block;
                }
            }
        }
    }

    if (low_main != high_main || low_block != high_block) return 7;
    std::cout << "maskshard-independent-bijection OK W=" << W
              << " low=" << low << " high=" << high
              << " main=" << low_main << " block=" << low_block << '\n';
    return 0;
}
