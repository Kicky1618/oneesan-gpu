#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_persistent_pipeline_plan_probe_main_unused
#include "gridfp_reduced_production_persistent_pipeline_plan_probe.cpp"
#pragma pop_macro("main")

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace {

struct HashAuditStats {
    Rank a_values[NGPU][PIPE_BATCHES]{};
    Rank scratch_words[NGPU][PIPE_BATCHES]{};
    Rank entries[NGPU][PIPE_BATCHES]{};
    Rank local_entries[NGPU]{};
    Rank local_values[NGPU]{};
    Rank cross_total = 0;
    Rank local_total = 0;
};

int leader_main_batch(Mask support) {
    return int((mix32(support) >> 12) & (PIPE_BATCHES - 1));
}

int leader_blocked_batch(Mask a, Mask b) {
    const Mask hi = std::max(a, b);
    const Mask lo = std::min(a, b);
    const Mask key = hi | (lo << K);
    return int((mix32(key) >> 12) & (PIPE_BATCHES - 1));
}

void add_main_cycle(
    HashAuditStats& stats,
    int period,
    bool reverse,
    bool leader_hash
) {
    int occupied = 0;
    Mask support = 0;
    for (int i = 1; i <= W; ++i) {
        occupied += necklace_bits[i];
        if (necklace_bits[i]) support |= Mask(1) << (i - 1);
    }
    if (!(occupied & 1) || period <= 1) return;

    const Rank pc = primitive[occupied][1];
    const int step = reverse ? 15 : 13;
    int route_owner[W]{};
    Mask cur = support;
    for (int hop = 0; hop < period; ++hop) {
        route_owner[hop] = owner_of_support(cur, reverse);
        cur = rotate_bits(cur, W, step);
    }

    bool all_local = true;
    for (int hop = 1; hop < period; ++hop)
        all_local = all_local && route_owner[hop] == route_owner[0];
    if (all_local) {
        const Rank values = Rank(period) * pc;
        ++stats.local_entries[route_owner[0]];
        stats.local_values[route_owner[0]] += values;
        stats.local_total += values;
        return;
    }

    const int batch = leader_hash ? leader_main_batch(support) : main_batch16(support);
    int segments_per_owner[NGPU]{};
    for (int hop = 0; hop < period; ++hop)
        stats.a_values[route_owner[hop]][batch] += pc;

    int hop = 0;
    while (hop < period) {
        const int owner = route_owner[hop];
        int len = 1;
        while (hop + len < period && route_owner[hop + len] == owner) ++len;
        ++segments_per_owner[owner];
        hop += len;
    }
    if (route_owner[0] == route_owner[period - 1])
        --segments_per_owner[route_owner[0]];

    for (int gpu = 0; gpu < NGPU; ++gpu) {
        const int segments = segments_per_owner[gpu];
        if (!segments) continue;
        ++stats.entries[gpu][batch];
        stats.scratch_words[gpu][batch] += Rank(segments) * pc;
    }
    stats.cross_total += Rank(period) * pc;
}

void generate_hash_cycles(
    int t,
    int period,
    HashAuditStats& old_forward,
    HashAuditStats& old_reverse,
    HashAuditStats& new_forward,
    HashAuditStats& new_reverse
) {
    if (t > W) {
        if (W % period == 0) {
            add_main_cycle(old_forward, period, false, false);
            add_main_cycle(old_reverse, period, true, false);
            add_main_cycle(new_forward, period, false, true);
            add_main_cycle(new_reverse, period, true, true);
        }
        return;
    }
    necklace_bits[t] = necklace_bits[t - period];
    generate_hash_cycles(
        t + 1, period, old_forward, old_reverse, new_forward, new_reverse);
    for (int bit = necklace_bits[t - period] + 1; bit <= 1; ++bit) {
        necklace_bits[t] = bit;
        generate_hash_cycles(
            t + 1, t, old_forward, old_reverse, new_forward, new_reverse);
    }
}

void add_blocked(
    HashAuditStats& old_forward,
    HashAuditStats& old_reverse,
    HashAuditStats& new_forward,
    HashAuditStats& new_reverse
) {
    for (Mask a = 0; a < (Mask(1) << K); ++a) {
        for (Mask b = 0; b <= a; ++b) {
            const int free_occupied = __builtin_popcount(a) + __builtin_popcount(b);
            if (free_occupied & 1) continue;
            if (a == b) continue;

            const int owner_a = owner_lut[a];
            const int owner_b = owner_lut[b];
            const int occupied = free_occupied + 1;
            const Rank pc = primitive[occupied][1];
            const int old_batch = blocked_batch16(a, b, occupied);
            const int new_batch = leader_blocked_batch(a, b);

            for (HashAuditStats* s : {&old_forward, &old_reverse}) {
                if (owner_a == owner_b) {
                    ++s->local_entries[owner_a];
                    s->local_values[owner_a] += 2 * pc;
                    s->local_total += 2 * pc;
                } else {
                    ++s->entries[owner_a][old_batch];
                    ++s->entries[owner_b][old_batch];
                    s->scratch_words[owner_a][old_batch] += pc;
                    s->scratch_words[owner_b][old_batch] += pc;
                    s->a_values[owner_a][old_batch] += pc;
                    s->a_values[owner_b][old_batch] += pc;
                    s->cross_total += 2 * pc;
                }
            }
            for (HashAuditStats* s : {&new_forward, &new_reverse}) {
                if (owner_a == owner_b) {
                    ++s->local_entries[owner_a];
                    s->local_values[owner_a] += 2 * pc;
                    s->local_total += 2 * pc;
                } else {
                    ++s->entries[owner_a][new_batch];
                    ++s->entries[owner_b][new_batch];
                    s->scratch_words[owner_a][new_batch] += pc;
                    s->scratch_words[owner_b][new_batch] += pc;
                    s->a_values[owner_a][new_batch] += pc;
                    s->a_values[owner_b][new_batch] += pc;
                    s->cross_total += 2 * pc;
                }
            }
        }
    }
}

struct Summary {
    Rank staged_metric = 0;
    Rank eager_bound = 0;
    Rank entries = 0;
    Rank peer_words = 0;
    double worst_batch_imbalance = 0.0;
    int worst_batch = -1;
    double peak_gib = 0.0;
    int peak_gpu = -1;
};

Summary summarize(const HashAuditStats& s) {
    Summary out;
    std::array<Rank, NGPU> gpu_total{};
    for (int batch = 0; batch < PIPE_BATCHES; ++batch) {
        Rank batch_total = 0;
        Rank batch_max = 0;
        for (int gpu = 0; gpu < NGPU; ++gpu) {
            const Rank v = s.a_values[gpu][batch];
            batch_total += v;
            batch_max = std::max(batch_max, v);
            gpu_total[gpu] += v;
            out.entries += s.entries[gpu][batch];
            out.peer_words += s.scratch_words[gpu][batch];
        }
        out.staged_metric += batch_max;
        const double avg = double(batch_total) / NGPU;
        const double imbalance = avg ? double(batch_max) / avg : 0.0;
        if (imbalance > out.worst_batch_imbalance) {
            out.worst_batch_imbalance = imbalance;
            out.worst_batch = batch;
        }
    }
    for (Rank v : gpu_total) out.eager_bound = std::max(out.eager_bound, v);

    constexpr double GiB = double(1ULL << 30);
    for (int gpu = 0; gpu < NGPU; ++gpu) {
        Rank plane[2]{};
        Rank list_entries = s.local_entries[gpu];
        for (int batch = 0; batch < PIPE_BATCHES; ++batch) {
            plane[batch & 1] = std::max(plane[batch & 1], s.scratch_words[gpu][batch]);
            list_entries += s.entries[gpu][batch];
        }
        const double state = double(owner_states[gpu]) * 4.0 / GiB;
        // Forward and reverse lists are retained simultaneously; each packed
        // entry is one u32 and the W=28 direction distributions are identical.
        const double dual_list = double(list_entries) * 8.0 / GiB;
        const double scratch = double(plane[0] + plane[1]) * 4.0 / GiB;
        const double metadata =
            double(PIPE_BATCHES * (W + 1) *
                   (MAX_SEGMENTS_PER_OWNER + 1) * 2 * sizeof(Rank)) / GiB;
        const double peak = state + dual_list + scratch + metadata;
        if (peak > out.peak_gib) {
            out.peak_gib = peak;
            out.peak_gpu = gpu;
        }
    }
    return out;
}

bool same_distribution(const HashAuditStats& a, const HashAuditStats& b) {
    if (a.cross_total != b.cross_total || a.local_total != b.local_total) return false;
    for (int gpu = 0; gpu < NGPU; ++gpu) {
        if (a.local_entries[gpu] != b.local_entries[gpu] ||
            a.local_values[gpu] != b.local_values[gpu]) return false;
        for (int batch = 0; batch < PIPE_BATCHES; ++batch) {
            if (a.a_values[gpu][batch] != b.a_values[gpu][batch] ||
                a.scratch_words[gpu][batch] != b.scratch_words[gpu][batch] ||
                a.entries[gpu][batch] != b.entries[gpu][batch]) return false;
        }
    }
    return true;
}

void report(const char* name, const Summary& s) {
    constexpr double B300_GiB = 288e9 / double(1ULL << 30);
    std::cout << "leader-batch-hash"
              << " scheme=" << name
              << " staged_metric=" << s.staged_metric
              << " eager_bound=" << s.eager_bound
              << " staged_vs_eager=" << double(s.staged_metric) / s.eager_bound
              << " worst_batch=" << s.worst_batch
              << " worst_batch_imbalance=" << s.worst_batch_imbalance
              << " list_entries=" << s.entries
              << " peer_words=" << s.peer_words
              << " peak_gpu=" << s.peak_gpu
              << " peak_GiB=" << s.peak_gib
              << " B300_headroom_GiB=" << (B300_GiB - s.peak_gib)
              << '\n';
}

} // namespace

int main() {
    initialize_tables();
    HashAuditStats old_forward{};
    HashAuditStats old_reverse{};
    HashAuditStats new_forward{};
    HashAuditStats new_reverse{};
    necklace_bits[0] = 0;
    generate_hash_cycles(
        1, 1, old_forward, old_reverse, new_forward, new_reverse);
    add_blocked(old_forward, old_reverse, new_forward, new_reverse);

    if (!same_distribution(old_forward, old_reverse) ||
        !same_distribution(new_forward, new_reverse)) return 2;
    if (old_forward.cross_total != 454373526378ULL ||
        old_forward.local_total != 18956500538ULL ||
        new_forward.cross_total != old_forward.cross_total ||
        new_forward.local_total != old_forward.local_total) return 3;

    const Summary old_s = summarize(old_forward);
    const Summary new_s = summarize(new_forward);
    if (old_s.staged_metric != 62070698845ULL ||
        old_s.eager_bound != 57545699830ULL) return 4;
    if (new_s.staged_metric != 57610325500ULL ||
        new_s.eager_bound != 57545699830ULL) return 5;
    if (old_s.peer_words != 409769189454ULL ||
        new_s.peer_words != old_s.peer_words) return 6;
    if (old_s.entries != new_s.entries) return 7;
    if (!(new_s.peak_gib < old_s.peak_gib) ||
        !(new_s.worst_batch_imbalance < old_s.worst_batch_imbalance)) return 8;

    std::cout << std::fixed << std::setprecision(9);
    report("invariant", old_s);
    report("leader-shift12", new_s);
    std::cout << "leader-batch-hash-improvement"
              << " staged_metric_speedup="
              << double(old_s.staged_metric) / new_s.staged_metric
              << " eager_gap_old="
              << double(old_s.staged_metric) / old_s.eager_bound
              << " eager_gap_new="
              << double(new_s.staged_metric) / new_s.eager_bound
              << " peak_savings_GiB=" << (old_s.peak_gib - new_s.peak_gib)
              << " batch_hash_shift=12"
              << " canonical_cycle_leader=1\n";
    std::cout << "ALL_OK production_leader_batch_hash=1 exact_W28=1"
              << " forward_reverse_identical=1\n";
    return 0;
}
