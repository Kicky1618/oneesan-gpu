#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_persistent_segment_microprobe_main_unused
#include "gridfp_reduced_production_p2p_persistent_segment_microprobe.cu"
#pragma pop_macro("main")

namespace {

struct HostPersistentLists {
    std::vector<std::vector<std::vector<std::uint32_t>>> batch;
    std::vector<std::vector<std::uint32_t>> local;
    std::vector<std::vector<unsigned long long>> words;
};

std::uint32_t hostlist_rotate(std::uint32_t x, int len, int shift) {
    shift %= len;
    if (shift < 0) shift += len;
    if (!shift) return x;
    const std::uint32_t mask = (std::uint32_t(1) << len) - 1u;
    return ((x << shift) | (x >> (len - shift))) & mask;
}

Rank64 hostlist_support_rank(
    std::uint32_t mask, int len, int ones, const ProductionFactorTables& t
) {
    Rank64 rank = 0;
    int left = ones;
    for (int pos = 0; pos < len; ++pos) {
        if (((mask >> pos) & 1u) == 0) continue;
        rank += t.binom(len - pos - 1, left);
        --left;
    }
    return rank;
}

int hostlist_owner(
    std::uint32_t support,
    int W,
    int Kwin,
    bool reverse,
    int ngpu,
    const ProductionFactorTables& t
) {
    const int L = Kwin + 2;
    const int O = W - L;
    const int lo = reverse ? 0 : W - L;
    const int hi = reverse ? L - 1 : W - 1;
    std::uint32_t outer = 0;
    int cp = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit >= lo && bit <= hi) continue;
        if ((support >> bit) & 1u) outer |= std::uint32_t(1) << cp;
        ++cp;
    }

    const int r = __builtin_popcount(outer);
    const Rank64 group = host_group_size(t, L, r);
    const Rank64 sr = hostlist_support_rank(outer, O, r, t);
    Rank64 prefix = 0;
    Rank64 total = 0;
    for (int x = 0; x <= O; ++x) {
        const Rank64 g = host_group_size(t, L, x);
        total += t.binom(O, x) * g;
        if (x < r) prefix += t.binom(O, x) * g;
    }
    const __uint128_t midpoint =
        __uint128_t(prefix) + __uint128_t(sr) * group + group / 2;
    int owner = static_cast<int>(midpoint * Rank64(ngpu) / total);
    if (owner >= ngpu) owner = ngpu - 1;
    return owner;
}

std::uint32_t hostlist_next(
    std::uint32_t support,
    bool blocked,
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse
) {
    const int span = Kwin + S + 2;
    const int lo = reverse ? 0 : W - span;
    if (!blocked) {
        const std::uint32_t mask = (std::uint32_t(1) << span) - 1u;
        const std::uint32_t x = (support >> lo) & mask;
        const int shift = reverse ? span - S : S;
        return (support & ~(mask << lo)) |
               (hostlist_rotate(x, span, shift) << lo);
    }

    const int compact_len = span - 2;
    std::uint32_t compact = 0;
    int cp = 0;
    for (int bit = lo; bit < lo + span; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        if ((support >> bit) & 1u) compact |= std::uint32_t(1) << cp;
        ++cp;
    }
    const int shift = reverse ? compact_len - S : S;
    const std::uint32_t rotated = hostlist_rotate(compact, compact_len, shift);
    std::uint32_t out = support;
    cp = 0;
    for (int bit = lo; bit < lo + span; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        out &= ~(std::uint32_t(1) << bit);
        if ((rotated >> cp) & 1u) out |= std::uint32_t(1) << bit;
        ++cp;
    }
    return out;
}

std::uint32_t hostlist_prev(
    std::uint32_t support,
    bool blocked,
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse
) {
    const int span = Kwin + S + 2;
    const int lo = reverse ? 0 : W - span;
    if (!blocked) {
        const std::uint32_t mask = (std::uint32_t(1) << span) - 1u;
        const std::uint32_t x = (support >> lo) & mask;
        const int shift = reverse ? S : span - S;
        return (support & ~(mask << lo)) |
               (hostlist_rotate(x, span, shift) << lo);
    }

    const int compact_len = span - 2;
    std::uint32_t compact = 0;
    int cp = 0;
    for (int bit = lo; bit < lo + span; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        if ((support >> bit) & 1u) compact |= std::uint32_t(1) << cp;
        ++cp;
    }
    const int shift = reverse ? S : compact_len - S;
    const std::uint32_t rotated = hostlist_rotate(compact, compact_len, shift);
    std::uint32_t out = support;
    cp = 0;
    for (int bit = lo; bit < lo + span; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        out &= ~(std::uint32_t(1) << bit);
        if ((rotated >> cp) & 1u) out |= std::uint32_t(1) << bit;
        ++cp;
    }
    return out;
}

int hostlist_gcd(int a, int b) {
    while (b) {
        const int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

int hostlist_leader_length(
    std::uint32_t support,
    bool blocked,
    int W,
    int q,
    int Kwin,
    int S,
    bool reverse
) {
    const int len = blocked ? Kwin + S : Kwin + S + 2;
    const int order = len / hostlist_gcd(len, S);
    std::uint32_t cur = hostlist_next(
        support, blocked, W, q, Kwin, S, reverse);
    if (cur == support) return 1;
    std::uint32_t minimum = support;
    int count = 1;
    while (cur != support) {
        minimum = std::min(minimum, cur);
        cur = hostlist_next(cur, blocked, W, q, Kwin, S, reverse);
        if (++count > order) return -1;
    }
    return minimum == support ? count : 0;
}

std::uint32_t hostlist_mix32(std::uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

int hostlist_batch_id(
    std::uint32_t support,
    bool blocked,
    int W,
    int q,
    int Kwin,
    int batches
) {
    std::uint32_t h = 0;
    if (!blocked) {
        h = std::uint32_t(__builtin_popcount(support)) * 0x9e3779b1u;
        constexpr std::pair<int, std::uint32_t> terms[] = {
            {1, 0x85ebca6bu},
            {3, 0xc2b2ae35u},
            {5, 0x27d4eb2fu},
            {7, 0x165667b1u},
        };
        for (const auto [distance, coefficient] : terms) {
            h ^= std::uint32_t(__builtin_popcount(
                     support & hostlist_rotate(support, W, distance))) *
                 coefficient;
        }
    } else {
        std::uint32_t compact = 0;
        int cp = 0;
        for (int bit = 0; bit < W; ++bit) {
            if (bit == q - 1 || bit == q) continue;
            if ((support >> bit) & 1u) compact |= std::uint32_t(1) << cp;
            ++cp;
        }
        const std::uint32_t half_mask = (std::uint32_t(1) << Kwin) - 1u;
        const std::uint32_t a = compact & half_mask;
        const std::uint32_t b = (compact >> Kwin) & half_mask;
        const std::uint32_t lo = std::min(a, b);
        const std::uint32_t hi = std::max(a, b);
        h = lo * 0x9e3779b1u;
        h ^= hi * 0x85ebca6bu;
        h ^= std::uint32_t(__builtin_popcount(support)) * 0xc2b2ae35u;
    }
    return int(hostlist_mix32(h) & std::uint32_t(batches - 1));
}

void hostlist_expand_seeds(
    Rank64 compact,
    int W,
    int q,
    bool reverse,
    EqualTileRunSeed (&out)[3],
    int& count
) {
    std::uint32_t base = 0;
    int cp = 0;
    for (int bit = 0; bit < W; ++bit) {
        if (bit == q - 1 || bit == q) continue;
        if ((compact >> cp) & 1ULL) base |= std::uint32_t(1) << bit;
        ++cp;
    }
    const int fixed = reverse ? q : q - 1;
    const int missing = reverse ? q - 1 : q;
    const bool odd = (__builtin_popcountll(compact) & 1) != 0;
    if (odd) {
        out[0] = EqualTileRunSeed{base, 0, 1};
        out[1] = EqualTileRunSeed{
            base | (std::uint32_t(1) << (q - 1)) |
                   (std::uint32_t(1) << q), 0, 1};
        out[2] = {};
        count = 2;
        return;
    }
    const std::uint32_t fixed_support = base | (std::uint32_t(1) << fixed);
    out[0] = EqualTileRunSeed{fixed_support, 0, 1};
    out[1] = EqualTileRunSeed{fixed_support, 1, 1};
    out[2] = EqualTileRunSeed{base | (std::uint32_t(1) << missing), 0, 1};
    count = 3;
}

HostPersistentLists build_host_persistent_lists(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    int batches,
    const ProductionFactorTables& tables
) {
    HostPersistentLists out;
    out.batch.assign(
        static_cast<std::size_t>(ngpu),
        std::vector<std::vector<std::uint32_t>>(static_cast<std::size_t>(batches)));
    out.local.assign(static_cast<std::size_t>(ngpu), {});
    out.words.assign(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches)));

    const int tile_start = reverse ? 1 : W - 1;
    const int q = tile_start + (reverse ? S : -S);
    const Rank64 base_supports = Rank64(1) << (W - 2);
    for (Rank64 compact = 0; compact < base_supports; ++compact) {
        EqualTileRunSeed seeds[3]{};
        int nr = 0;
        hostlist_expand_seeds(compact, W, q, reverse, seeds, nr);
        for (int ri = 0; ri < nr; ++ri) {
            const auto seed = seeds[ri];
            const bool blocked = seed.blocked != 0;
            const int owner = hostlist_owner(
                seed.support, W, Kwin, reverse, ngpu, tables);
            const std::uint32_t prev = hostlist_prev(
                seed.support, blocked, W, q, Kwin, S, reverse);
            const int prev_owner = hostlist_owner(
                prev, W, Kwin, reverse, ngpu, tables);
            const Rank64 pc =
                tables.primitive[static_cast<std::size_t>(__builtin_popcount(seed.support))][1];
            if (!pc) fail("host list primitive count");
            const std::uint32_t packed =
                (seed.support & PERSISTENT_SUPPORT_MASK) |
                (blocked ? PERSISTENT_BLOCKED_BIT : 0u);

            if (prev_owner != owner) {
                const int batch = hostlist_batch_id(
                    seed.support, blocked, W, q, Kwin, batches);
                out.batch[static_cast<std::size_t>(owner)][static_cast<std::size_t>(batch)]
                    .push_back(packed);
                out.words[static_cast<std::size_t>(owner)][static_cast<std::size_t>(batch)] += pc;
                continue;
            }

            const int cycle_len = hostlist_leader_length(
                seed.support, blocked, W, q, Kwin, S, reverse);
            if (cycle_len < 0) fail("host list cycle length");
            if (cycle_len <= 1) continue;

            bool all_local = true;
            std::uint32_t cur = seed.support;
            for (int hop = 0; hop < cycle_len; ++hop) {
                if (hostlist_owner(cur, W, Kwin, reverse, ngpu, tables) != owner) {
                    all_local = false;
                    break;
                }
                cur = hostlist_next(cur, blocked, W, q, Kwin, S, reverse);
            }
            if (all_local)
                out.local[static_cast<std::size_t>(owner)].push_back(packed);
        }
    }
    return out;
}

void run_host_list_executor(
    int W,
    int Kwin,
    int S,
    bool reverse,
    int ngpu,
    int batches,
    unsigned requested_blocks
) {
    ProductionFactorTables tables(W);
    const HostTilePlan plan = make_host_tile_plan(tables, Kwin, ngpu);
    const int tile_start = reverse ? 1 : W - 1;
    const int q = tile_start + (reverse ? S : -S);

    std::vector<std::uint32_t> input;
    std::vector<std::uint32_t> expected;
    build_shift_boundary_vectors(
        W, Kwin, S, reverse, ngpu, tables, plan, input, expected);
    enable_scratch_full_peer_mesh(ngpu);

    const auto setup0 = std::chrono::steady_clock::now();
    HostPersistentLists lists = build_host_persistent_lists(
        W, Kwin, S, reverse, ngpu, batches, tables);

    std::vector<PersistentCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> state_ptr(static_cast<std::size_t>(ngpu), nullptr);
    std::vector<std::vector<unsigned long long>> batch_offset(
        static_cast<std::size_t>(ngpu),
        std::vector<unsigned long long>(static_cast<std::size_t>(batches + 1)));

    for (int g = 0; g < ngpu; ++g) {
        auto& offsets = batch_offset[static_cast<std::size_t>(g)];
        std::vector<std::uint32_t> packed;
        for (int b = 0; b < batches; ++b) {
            offsets[static_cast<std::size_t>(b)] = packed.size();
            const auto& part = lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
            packed.insert(packed.end(), part.begin(), part.end());
        }
        offsets[static_cast<std::size_t>(batches)] = packed.size();

        ck(cudaSetDevice(g), "host list set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.owner_begin, ngpu * sizeof(Rank64)),
           "host list alloc owner begin");
        ck(cudaMemcpy(c.owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64),
                      cudaMemcpyHostToDevice), "host list copy owner begin");

        ck(cudaMalloc(&c.batch_list,
                      std::max<std::size_t>(1, packed.size()) * sizeof(std::uint32_t)),
           "host list alloc cross list");
        if (!packed.empty())
            ck(cudaMemcpy(c.batch_list, packed.data(), packed.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "host list copy cross list");
        const auto& local = lists.local[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.local_list,
                      std::max<std::size_t>(1, local.size()) * sizeof(std::uint32_t)),
           "host list alloc local list");
        if (!local.empty())
            ck(cudaMemcpy(c.local_list, local.data(), local.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice), "host list copy local list");

        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, n * sizeof(std::uint32_t)), "host list alloc state");
        state_ptr[static_cast<std::size_t>(g)] = c.state;
        ck(cudaMemcpy(c.state,
                      input.data() + plan.shard_base[static_cast<std::size_t>(g)],
                      n * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "host list copy state");

        unsigned long long max_words = 1;
        unsigned long long max_desc = 1;
        for (int b = 0; b < batches; ++b) {
            max_words = std::max(
                max_words,
                lists.words[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)]);
            max_desc = std::max<unsigned long long>(
                max_desc,
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)].size());
        }
        ck(cudaMalloc(&c.scratch, max_words * sizeof(std::uint32_t)),
           "host list alloc scratch");
        ck(cudaMalloc(&c.descriptor, max_desc * sizeof(ScratchDescriptor)),
           "host list alloc descriptor");
        ck(cudaMalloc(&c.peer_state, ngpu * sizeof(std::uint32_t*)),
           "host list alloc peer table");
        ck(cudaMalloc(&c.head_words, sizeof(unsigned long long)),
           "host list alloc scratch head");
        ck(cudaMalloc(&c.head_desc, sizeof(unsigned long long)),
           "host list alloc descriptor head");
        ck(cudaMalloc(&c.peer_words, sizeof(unsigned long long)),
           "host list alloc peer words");
        ck(cudaMalloc(&c.error, sizeof(int)), "host list alloc error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "host list peer table set device");
        ck(cudaMemcpy(ctx[static_cast<std::size_t>(g)].peer_state,
                      state_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "host list copy peer table");
    }
    const double setup_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - setup0).count();

    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        const auto count = lists.local[static_cast<std::size_t>(g)].size();
        if (!count) continue;
        ck(cudaSetDevice(g), "host list local set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMemset(c.error, 0, sizeof(int)), "host list zero local error");
        const unsigned blocks = static_cast<unsigned>(
            std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
        persistent_local_cycle_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
            c.state, c.local_list, count,
            W, q, reverse, tile_start, Kwin, S, ngpu, g,
            c.owner_begin, c.error);
        ck(cudaGetLastError(), "host list local launch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "host list local sync set device");
        ck(cudaDeviceSynchronize(), "host list local sync");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "host list local copy error");
        if (error) fail("host list local device error=" + std::to_string(error));
    }

    unsigned long long total_peer_words = 0;
    for (int batch = 0; batch < batches; ++batch) {
        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            if (!count) continue;
            ck(cudaSetDevice(g), "host list phase A set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            ck(cudaMemset(c.head_words, 0, sizeof(unsigned long long)),
               "host list zero scratch head");
            ck(cudaMemset(c.head_desc, 0, sizeof(unsigned long long)),
               "host list zero descriptor head");
            ck(cudaMemset(c.peer_words, 0, sizeof(unsigned long long)),
               "host list zero peer words");
            ck(cudaMemset(c.error, 0, sizeof(int)), "host list zero batch error");
            const unsigned blocks = static_cast<unsigned>(
                std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
            const unsigned long long offset =
                batch_offset[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            persistent_phase_a_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
                c.state, c.scratch, c.descriptor,
                lists.words[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)],
                count, c.head_words, c.head_desc,
                c.batch_list + offset, count, batch, batches,
                W, q, reverse, tile_start, Kwin, S, ngpu, g,
                c.owner_begin, c.error);
            ck(cudaGetLastError(), "host list phase A launch");
        }
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "host list phase A sync set device");
            ck(cudaDeviceSynchronize(), "host list phase A sync");
        }

        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            if (!count) continue;
            ck(cudaSetDevice(g), "host list phase B set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            const unsigned blocks = static_cast<unsigned>(
                std::max<std::size_t>(1, std::min<std::size_t>(requested_blocks, count)));
            scratch_full_phase_b_kernel<<<blocks, SCRATCH_FULL_THREADS>>>(
                c.peer_state, c.scratch, c.descriptor,
                count, c.peer_words, c.error);
            ck(cudaGetLastError(), "host list phase B launch");
        }
        for (int g = 0; g < ngpu; ++g) {
            ck(cudaSetDevice(g), "host list phase B sync set device");
            ck(cudaDeviceSynchronize(), "host list phase B sync");
        }

        for (int g = 0; g < ngpu; ++g) {
            const auto count =
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)].size();
            if (!count) continue;
            ck(cudaSetDevice(g), "host list batch audit set device");
            auto& c = ctx[static_cast<std::size_t>(g)];
            int error = 0;
            unsigned long long head_words = 0, head_desc = 0, peer_words = 0;
            ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
               "host list copy batch error");
            ck(cudaMemcpy(&head_words, c.head_words, sizeof(head_words),
                          cudaMemcpyDeviceToHost), "host list copy scratch head");
            ck(cudaMemcpy(&head_desc, c.head_desc, sizeof(head_desc),
                          cudaMemcpyDeviceToHost), "host list copy descriptor head");
            ck(cudaMemcpy(&peer_words, c.peer_words, sizeof(peer_words),
                          cudaMemcpyDeviceToHost), "host list copy peer words");
            const auto expected_words =
                lists.words[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)];
            if (error) fail("host list batch device error=" + std::to_string(error));
            if (head_words != expected_words || head_desc != count ||
                peer_words != expected_words)
                fail("host list batch count/execute mismatch");
            total_peer_words += peer_words;
        }
    }
    const double runtime_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> output(input.size());
    unsigned long long list_entries = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "host list gather set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 n = plan.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(
               output.data() + plan.shard_base[static_cast<std::size_t>(g)],
               c.state, n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "host list gather state");
        list_entries += lists.local[static_cast<std::size_t>(g)].size();
        for (int b = 0; b < batches; ++b)
            list_entries +=
                lists.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)].size();
    }
    if (output != expected) fail("host persistent list redistribution mismatch");

    std::cout << "gridfp-p2p-host-persistent-list"
              << " W=" << W
              << " Kwin=" << Kwin
              << " shift=" << S
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " batches=" << batches
              << " states=" << tables.size()
              << " list_entries=" << list_entries
              << " logical_peer_KiB="
              << double(total_peer_words) * sizeof(std::uint32_t) / 1024.0
              << " startup_gpu_support_scan_passes=0"
              << " startup_gpu_count_passes=0"
              << " runtime_support_scan_passes=0"
              << " runtime_count_passes=0"
              << " setup_ms=" << setup_ms
              << " runtime_ms=" << runtime_ms
              << " native_peer_atomics_required=0"
              << " remote_state_reads=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "host list free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        cudaFree(c.error);
        cudaFree(c.peer_words);
        cudaFree(c.head_desc);
        cudaFree(c.head_words);
        cudaFree(c.peer_state);
        cudaFree(c.descriptor);
        cudaFree(c.scratch);
        cudaFree(c.state);
        cudaFree(c.local_list);
        cudaFree(c.batch_list);
        cudaFree(c.owner_begin);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int Kwin = argc > 2 ? std::atoi(argv[2]) : 4;
    const int S = argc > 3 ? std::atoi(argv[3]) : 4;
    const int batches = argc > 4 ? std::atoi(argv[4]) : 8;
    const unsigned blocks = argc > 5
        ? static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10)) : 256u;
    const int ngpu = argc > 6 ? std::atoi(argv[6]) : 8;
    if (W < 7 || W > 11 || Kwin < 1 || S != Kwin ||
        Kwin + S + 2 != W || batches < 2 || batches > PERSISTENT_BATCH_MAX ||
        (batches & (batches - 1)) != 0 || !blocks ||
        ngpu < 2 || ngpu > SCRATCH_FULL_MAX_GPU) return 2;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "host persistent list device count");
    if (visible < ngpu) return 3;

    run_host_list_executor(W, Kwin, S, false, ngpu, batches, blocks);
    run_host_list_executor(W, Kwin, S, true, ngpu, batches, blocks);
    std::cout << "ALL_OK gridfp_p2p_host_persistent_list=1\n";
    return 0;
}
