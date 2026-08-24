#pragma once

#include "ramstream32_b300_direct_maskshard.cuh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <unordered_map>
#include <vector>

// Communication-aware partitioner for the preferred HIGH-occupancy sharding.
// LOW-window transitions preserve HIGH occupancy, so LOW P2P stays identically
// zero for every assignment produced here.  Only HIGH-window edges matter.
struct B300MaskPartitionStats {
    uint64_t orbit_total = 0;
    uint64_t closure_total = 0;
    uint64_t orbit_cut_before = 0;
    uint64_t closure_cut_before = 0;
    uint64_t orbit_cut_after = 0;
    uint64_t closure_cut_after = 0;
    uint64_t moves = 0;
    uint64_t swaps = 0;
    double auth_min_gib = 0.0;
    double auth_max_gib = 0.0;
    double work_max_over_avg = 0.0;
};

struct B300MaskPartitionEdgeWeight { uint64_t orbit = 0, closure = 0; };
struct B300MaskPartitionAdj { uint32_t v = 0; uint64_t orbit = 0, closure = 0; };

static inline uint64_t b300_mask_partition_pair_key(uint32_t a, uint32_t b) {
    if (a > b) std::swap(a, b);
    return (uint64_t(a) << 32) | b;
}

static B300DirectMaskShardHost b300_build_mask_shard_from_owner(
    const StorageFactorHost& storage, const StorageLayout& layout, int ngpu,
    const std::vector<uint8_t>& owner
) {
    constexpr uint32_t NM = 1u << HIGH_LUT_K;
    if (ngpu < 1 || ngpu > MAXGPU || owner.size() != NM) std::exit(480);

    B300DirectMaskShardHost z;
    z.ngpu = ngpu;
    z.mask_owner = owner;
    z.high_owner.resize(storage.high_all_codes.size());
    z.high_local.resize(storage.high_all_codes.size());
    std::array<std::array<uint32_t, MAXW + 2>, MAXGPU> rows_per_h{};

    for (int h = 0; h <= MAXW; ++h) {
        std::array<uint32_t, MAXGPU> local{};
        uint32_t n = storage.high_all_off[h + 1] - storage.high_all_off[h];
        for (uint32_t hr = 0; hr < n; ++hr) {
            uint32_t ai = storage.high_all_off[h] + hr;
            uint32_t mask = seg_occ(storage.high_all_codes[ai], HIGH_LUT_K);
            int g = owner[mask];
            if (g < 0 || g >= ngpu) std::exit(481);
            z.high_owner[ai] = uint8_t(g);
            z.high_local[ai] = local[g]++;
        }
        for (int g = 0; g < ngpu; ++g) rows_per_h[g][h] = local[g];
    }

    for (int g = 0; g < ngpu; ++g) {
        for (int h = 0; h <= MAXW; ++h) {
            z.owned_off[g][h] = uint32_t(z.owned_rows[g].size());
            uint32_t n = storage.high_all_off[h + 1] - storage.high_all_off[h];
            for (uint32_t hr = 0; hr < n; ++hr) {
                uint32_t ai = storage.high_all_off[h] + hr;
                if (z.high_owner[ai] == g) z.owned_rows[g].push_back(hr);
            }
        }
        z.owned_off[g][MAXW + 1] = uint32_t(z.owned_rows[g].size());

        Code off = 0;
        for (size_t bid = 0; bid < layout.main_blocks.size(); ++bid) {
            z.main_off[g][bid] = off;
            const auto& b = layout.main_blocks[bid];
            off += Code(rows_per_h[g][b.he]) * b.cols;
        }
        z.main_count[g] = off;
        off = 0;
        for (size_t bid = 0; bid < layout.block_blocks.size(); ++bid) {
            z.block_off[g][bid] = off;
            const auto& b = layout.block_blocks[bid];
            off += Code(rows_per_h[g][b.he]) * b.cols;
        }
        z.block_count[g] = off;
    }
    return z;
}

static B300DirectMaskShardHost build_b300_direct_mask_shards_optimized(
    const StorageFactorHost& storage, const StorageLayout& layout,
    const B300SparseActionsHost& sparse, int ngpu,
    int closure_lambda = 16, double auth_slack_gib = 4.0,
    int pair_candidate_limit = 512,
    B300MaskPartitionStats* out_stats = nullptr
) {
    constexpr int S = MAXW + 2;
    constexpr uint32_t NM = 1u << HIGH_LUT_K;
    if (ngpu < 1 || ngpu > MAXGPU || closure_lambda < 1) std::exit(482);

    std::vector<uint64_t> bytes(NM, 0), work(NM, 0);
    auto add_auth = [&](const StorageBlock& b) {
        if (!b.valid || !b.cols) return;
        for (uint32_t mask = 0; mask < NM; ++mask) {
            size_t ix = size_t(mask) * S + b.he;
            uint64_t nr = G_FACTOR.high_mask_off[ix + 1] - G_FACTOR.high_mask_off[ix];
            bytes[mask] += nr * uint64_t(b.cols) * sizeof(Count);
        }
    };
    for (const auto& b : layout.main_blocks) add_auth(b);
    for (const auto& b : layout.block_blocks) add_auth(b);

    auto mask_of = [&](const StorageBlock& b, uint32_t hr) -> uint32_t {
        return seg_occ(storage.high_all_codes[storage.high_all_off[b.he] + hr], HIGH_LUT_K);
    };

    std::unordered_map<uint64_t, B300MaskPartitionEdgeWeight> em;
    em.reserve(1u << 20);
    uint64_t orbit_total = 0, closure_total = 0;
    auto add_orbit = [&](uint32_t a, uint32_t b, uint64_t w) {
        orbit_total += w;
        if (a != b) em[b300_mask_partition_pair_key(a, b)].orbit += w;
    };
    auto add_closure = [&](uint32_t a, uint32_t b, uint64_t w) {
        closure_total += w;
        if (a != b) em[b300_mask_partition_pair_key(a, b)].closure += w;
    };

    for (const auto& op : sparse.high_orbit) {
        const auto& x = layout.main_blocks[b300_sparse_sblock(op)];
        const auto& j = layout.main_blocks[b300_sparse_jblock(op)];
        const auto& d = layout.block_blocks[b300_sparse_dblock(op)];
        uint32_t s = mask_of(x, b300_sparse_src(op));
        work[s] += x.cols;
        add_orbit(s, mask_of(j, b300_sparse_jrank(op)), x.cols);
        add_orbit(s, mask_of(d, b300_sparse_drank(op)), x.cols);
    }
    for (uint64_t op : sparse.high_closure) {
        const auto& x = layout.main_blocks[b300_sparse_closure_sblock(op)];
        uint32_t desc = b300_sparse_closure_desc(op);
        const auto& d = layout.block_blocks[highdesc_block(desc)];
        uint32_t s = mask_of(x, b300_sparse_closure_src(op));
        work[s] += x.cols;
        add_closure(s, mask_of(d, highdesc_rank(desc)), x.cols);
    }

    std::vector<std::vector<B300MaskPartitionAdj>> adj(NM);
    for (const auto& kv : em) {
        uint32_t a = uint32_t(kv.first >> 32), b = uint32_t(kv.first);
        adj[a].push_back({b, kv.second.orbit, kv.second.closure});
        adj[b].push_back({a, kv.second.orbit, kv.second.closure});
    }

    std::vector<uint32_t> order(NM);
    std::iota(order.begin(), order.end(), 0u);
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        return bytes[a] != bytes[b] ? bytes[a] > bytes[b] : a < b;
    });

    std::vector<uint8_t> owner(NM, 0);
    std::array<uint64_t, MAXGPU> gb{}, gw{};
    for (uint32_t m : order) {
        int g = 0;
        for (int q = 1; q < ngpu; ++q) if (gb[q] < gb[g]) g = q;
        owner[m] = uint8_t(g); gb[g] += bytes[m]; gw[g] += work[m];
    }

    auto cuts = [&]() {
        std::pair<uint64_t,uint64_t> z{0,0};
        for (const auto& kv : em) {
            uint32_t a = uint32_t(kv.first >> 32), b = uint32_t(kv.first);
            if (owner[a] != owner[b]) { z.first += kv.second.orbit; z.second += kv.second.closure; }
        }
        return z;
    };
    auto before = cuts();

    long double total_b = std::accumulate(bytes.begin(), bytes.end(), (long double)0);
    long double total_w = std::accumulate(work.begin(), work.end(), (long double)0);
    long double avg_b = total_b / ngpu, avg_w = total_w / ngpu;
    uint64_t slack = uint64_t(auth_slack_gib * double(1ull << 30));
    uint64_t bmin = uint64_t(std::max<long double>(0, avg_b - slack));
    uint64_t bmax = uint64_t(avg_b + slack);
    uint64_t initial_wmax = 0;
    for (int g = 0; g < ngpu; ++g) initial_wmax = std::max(initial_wmax, gw[g]);
    long double wlimit = std::max<long double>(avg_w * 1.08L, initial_wmax);

    std::vector<uint64_t> degree(NM, 0);
    for (uint32_t u = 0; u < NM; ++u)
        for (const auto& e : adj[u]) degree[u] += e.orbit + uint64_t(closure_lambda) * e.closure;
    std::vector<uint32_t> cand(NM);
    std::iota(cand.begin(), cand.end(), 0u);
    std::sort(cand.begin(), cand.end(), [&](uint32_t a, uint32_t b) { return degree[a] > degree[b]; });

    uint64_t moves = 0;
    for (int pass = 0; pass < 30; ++pass) {
        bool changed = false;
        for (uint32_t u : cand) {
            int a = owner[u], best = a;
            int64_t best_gain = 0;
            std::array<uint64_t, MAXGPU> con{};
            for (const auto& e : adj[u]) con[owner[e.v]] += e.orbit + uint64_t(closure_lambda) * e.closure;
            for (int b = 0; b < ngpu; ++b) if (b != a) {
                if (gb[b] + bytes[u] > bmax || gb[a] - bytes[u] < bmin) continue;
                if ((long double)(gw[b] + work[u]) > wlimit) continue;
                int64_t gain = int64_t(con[b]) - int64_t(con[a]);
                if (gain > best_gain) { best_gain = gain; best = b; }
            }
            if (best != a) {
                owner[u] = uint8_t(best);
                gb[a] -= bytes[u]; gb[best] += bytes[u];
                gw[a] -= work[u]; gw[best] += work[u];
                ++moves; changed = true;
            }
        }
        if (!changed) break;
    }

    auto edge_weight = [&](uint32_t u, uint32_t v) -> uint64_t {
        for (const auto& e : adj[u]) if (e.v == v)
            return e.orbit + uint64_t(closure_lambda) * e.closure;
        return 0;
    };

    uint64_t swaps = 0;
    int lim = std::max(0, pair_candidate_limit);
    for (int pass = 0; pass < 100 && lim; ++pass) {
        bool changed = false;
        std::array<std::vector<uint32_t>, MAXGPU> bc;
        for (uint32_t u = 0; u < NM; ++u) {
            int a = owner[u]; bool boundary = false;
            for (const auto& e : adj[u]) if (owner[e.v] != a) { boundary = true; break; }
            if (boundary) bc[a].push_back(u);
        }
        for (int g = 0; g < ngpu; ++g) {
            std::sort(bc[g].begin(), bc[g].end(), [&](uint32_t a, uint32_t b) { return degree[a] > degree[b]; });
            if ((int)bc[g].size() > lim) bc[g].resize(size_t(lim));
        }
        for (int A = 0; A < ngpu; ++A) for (int B = A + 1; B < ngpu; ++B) {
            int64_t best_gain = 0; uint32_t bu = 0xffffffffu, bv = 0xffffffffu;
            for (uint32_t u : bc[A]) {
                std::array<uint64_t, MAXGPU> cu{};
                for (const auto& e : adj[u]) cu[owner[e.v]] += e.orbit + uint64_t(closure_lambda) * e.closure;
                int64_t gu = int64_t(cu[B]) - int64_t(cu[A]);
                for (uint32_t v : bc[B]) {
                    uint64_t nba = gb[A] - bytes[u] + bytes[v], nbb = gb[B] - bytes[v] + bytes[u];
                    if (nba < bmin || nba > bmax || nbb < bmin || nbb > bmax) continue;
                    uint64_t nwa = gw[A] - work[u] + work[v], nwb = gw[B] - work[v] + work[u];
                    if ((long double)nwa > wlimit || (long double)nwb > wlimit) continue;
                    std::array<uint64_t, MAXGPU> cv{};
                    for (const auto& e : adj[v]) cv[owner[e.v]] += e.orbit + uint64_t(closure_lambda) * e.closure;
                    int64_t gv = int64_t(cv[A]) - int64_t(cv[B]);
                    int64_t gain = gu + gv - 2 * int64_t(edge_weight(u, v));
                    if (gain > best_gain) { best_gain = gain; bu = u; bv = v; }
                }
            }
            if (bu != 0xffffffffu) {
                owner[bu] = uint8_t(B); owner[bv] = uint8_t(A);
                gb[A] = gb[A] - bytes[bu] + bytes[bv]; gb[B] = gb[B] - bytes[bv] + bytes[bu];
                gw[A] = gw[A] - work[bu] + work[bv]; gw[B] = gw[B] - work[bv] + work[bu];
                ++swaps; changed = true;
            }
        }
        if (!changed) break;
    }

    auto after = cuts();
    if (out_stats) {
        out_stats->orbit_total = orbit_total; out_stats->closure_total = closure_total;
        out_stats->orbit_cut_before = before.first; out_stats->closure_cut_before = before.second;
        out_stats->orbit_cut_after = after.first; out_stats->closure_cut_after = after.second;
        out_stats->moves = moves; out_stats->swaps = swaps;
        uint64_t mn = gb[0], mx = gb[0], wm = gw[0];
        for (int g = 1; g < ngpu; ++g) { mn = std::min(mn, gb[g]); mx = std::max(mx, gb[g]); wm = std::max(wm, gw[g]); }
        out_stats->auth_min_gib = double(mn) / double(1ull << 30);
        out_stats->auth_max_gib = double(mx) / double(1ull << 30);
        out_stats->work_max_over_avg = avg_w ? double((long double)wm / avg_w) : 0.0;
    }
    return b300_build_mask_shard_from_owner(storage, layout, ngpu, owner);
}
