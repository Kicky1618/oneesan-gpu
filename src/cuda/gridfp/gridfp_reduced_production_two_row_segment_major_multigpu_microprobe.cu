#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_two_row_runtime_multigpu_microprobe_main_unused2
#include "gridfp_reduced_production_two_row_runtime_multigpu_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_p2p_segment_major_runtime.cuh"

#include <array>
#include <map>
#include <set>

namespace {

struct RtMajorRun {
    int owner = -1;
    Rank64 base = 0;
    Rank64 pc = 0;
};

struct RtMajorKey {
    std::uint32_t support = 0;
    bool blocked = false;
    bool operator<(const RtMajorKey& o) const {
        return support < o.support || (support == o.support && blocked < o.blocked);
    }
    bool operator==(const RtMajorKey& o) const {
        return support == o.support && blocked == o.blocked;
    }
};

struct RtMajorPacked {
    std::vector<std::uint32_t> low;
    std::vector<std::uint8_t> high;
};

struct RtMajorHostBatch {
    std::vector<std::uint32_t> run_begin;
    RtMajorPacked source;
    RtMajorPacked local;
    std::array<P2PMajorGroupMeta, RP_P2P_MAJOR_PC_CLASSES> group{};
    Rank64 scratch_states = 0;
};

struct RtMajorHostLocal {
    std::vector<std::uint32_t> run_begin;
    RtMajorPacked local;
    std::array<P2PMajorGroupMeta, RP_P2P_MAJOR_PC_CLASSES> group{};
};

struct RtMajorHostPlan {
    std::vector<std::vector<RtMajorHostBatch>> batch; // [gpu][batch]
    std::vector<RtMajorHostLocal> local;              // [gpu]
    std::vector<Rank64> scratch_states;
    Rank64 network_segments = 0;
    Rank64 network_states = 0;
    Rank64 local_cycles = 0;
    Rank64 local_runs = 0;
};

struct RtMajorTempSegment {
    RtMajorRun source;
    std::vector<RtMajorRun> local;
};

std::uint32_t rtmajor_rotate_bits(std::uint32_t x, int len, int shift) {
    if (len <= 0 || len >= 32) fail("rtmajor rotate width");
    shift %= len;
    if (shift < 0) shift += len;
    if (!shift) return x;
    const std::uint32_t mask = (std::uint32_t(1) << len) - 1u;
    return ((x << shift) | (x >> (len - shift))) & mask;
}

std::uint32_t rtmajor_next_support(
    std::uint32_t support, bool blocked, int W, int K, bool reverse
) {
    if (!blocked) {
        const int shift = reverse ? W - K : K;
        return rtmajor_rotate_bits(support, W, shift);
    }
    const std::uint32_t kmask = (std::uint32_t(1) << K) - 1u;
    const std::uint32_t low = support & kmask;
    const std::uint32_t middle = (support >> K) & 3u;
    const std::uint32_t high = (support >> (K + 2)) & kmask;
    return high | (middle << K) | (low << (K + 2));
}

std::uint32_t rtmajor_batch_hash(
    std::uint32_t support, bool blocked, bool reverse
) {
    std::uint32_t x = support ^ 0x9e3779b9u;
    x ^= blocked ? 0x27d4eb2du : 0u;
    x ^= reverse ? 0x165667b1u : 0u;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

int rtmajor_pc_class(Rank64 pc) {
    for (int c = 0; c < RP_P2P_MAJOR_PC_CLASSES; ++c)
        if (catalan(c + 1) == pc) return c;
    fail("rtmajor primitive class");
}

void rtmajor_pack(int owner, Rank64 local, RtMajorPacked& dst) {
    if (owner < 0 || owner >= 8 || local >= (Rank64(1) << 36))
        fail("rtmajor packed run range");
    dst.low.push_back(
        std::uint32_t(owner) |
        (std::uint32_t(local & ((Rank64(1) << 29) - 1)) << 3));
    dst.high.push_back(static_cast<std::uint8_t>((local >> 29) & 0x7fu));
}

std::map<RtMajorKey,RtMajorRun> rtmajor_runs(
    const ProductionFactorTables& tables,
    int W,
    int K,
    bool reverse,
    int ngpu,
    const OwnerPlan& owner_plan
) {
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);
    const auto main = gen_words(W);
    const auto block = gen_words(W - 1);
    std::map<RtMajorKey,RtMajorRun> out;
    for (Key key : layout(main, block, q)) {
        const MateID full = embed_full(key, W, q, reverse);
        const std::uint32_t support = occupancy_mask(full, W);
        const Rank64 pr = tables.primitive_rank(full, W);
        const Rank64 pc = catalan((__builtin_popcount(support) + 1) / 2);
        const GroupedRank gr = grouped_rank(
            key, tables, W, q, reverse, old_start, K, ngpu, owner_plan);
        if (gr.local < pr) fail("rtmajor run base underflow");
        const RtMajorKey rk{support, key.blocked};
        const RtMajorRun rr{gr.owner, gr.local - pr, pc};
        auto [it, inserted] = out.emplace(rk, rr);
        if (!inserted &&
            (it->second.owner != rr.owner || it->second.base != rr.base ||
             it->second.pc != rr.pc))
            fail("rtmajor run inconsistency");
    }
    return out;
}

RtMajorHostPlan make_rtmajor_host_plan(
    const ProductionFactorTables& tables,
    int W,
    int K,
    bool reverse,
    int ngpu,
    int batches,
    const HostTilePlan& tile
) {
    const OwnerPlan owner_plan{tile.owner_begin, tile.owner_size};
    const auto runs = rtmajor_runs(tables, W, K, reverse, ngpu, owner_plan);

    using LocalBucket = std::array<std::vector<std::vector<RtMajorRun>>,
                                   RP_P2P_MAJOR_PC_CLASSES>;
    using NetBucket = std::array<std::vector<RtMajorTempSegment>,
                                 RP_P2P_MAJOR_PC_CLASSES>;
    std::vector<LocalBucket> local_bucket(static_cast<std::size_t>(ngpu));
    std::vector<std::vector<NetBucket>> net_bucket(static_cast<std::size_t>(ngpu));
    for (auto& g : net_bucket) g.resize(static_cast<std::size_t>(batches));

    std::set<RtMajorKey> seen;
    for (const auto& [root, root_rec] : runs) {
        if (seen.count(root)) continue;
        std::vector<RtMajorKey> cycle;
        RtMajorKey cur = root;
        do {
            const auto it = runs.find(cur);
            if (it == runs.end()) fail("rtmajor cycle escaped run map");
            if (!seen.insert(cur).second) fail("rtmajor cycle overlap");
            cycle.push_back(cur);
            cur.support = rtmajor_next_support(cur.support, cur.blocked, W, K, reverse);
            if (cycle.size() > static_cast<std::size_t>(RP_MAX_W))
                fail("rtmajor cycle too long");
        } while (!(cur == root));
        if (cycle.size() <= 1) continue;

        const Rank64 pc = root_rec.pc;
        const int cls = rtmajor_pc_class(pc);
        std::vector<int> owner(cycle.size());
        for (std::size_t i = 0; i < cycle.size(); ++i) {
            const auto& r = runs.at(cycle[i]);
            if (r.pc != pc) fail("rtmajor pc changed in cycle");
            owner[i] = r.owner;
        }
        std::vector<int> starts;
        for (int i = 0; i < static_cast<int>(cycle.size()); ++i) {
            const int pred = (i + static_cast<int>(cycle.size()) - 1) %
                             static_cast<int>(cycle.size());
            if (owner[static_cast<std::size_t>(i)] !=
                owner[static_cast<std::size_t>(pred)])
                starts.push_back(i);
        }
        if (starts.empty()) {
            const int g = owner[0];
            std::vector<RtMajorRun> seq;
            for (const auto& k : cycle) seq.push_back(runs.at(k));
            local_bucket[static_cast<std::size_t>(g)][static_cast<std::size_t>(cls)]
                .push_back(std::move(seq));
            continue;
        }

        const int batch = int(rtmajor_batch_hash(root.support, root.blocked, reverse) %
                              std::uint32_t(batches));
        const int n = static_cast<int>(cycle.size());
        for (int start : starts) {
            const int pred = (start + n - 1) % n;
            const int g = owner[static_cast<std::size_t>(start)];
            RtMajorTempSegment seg;
            seg.source = runs.at(cycle[static_cast<std::size_t>(pred)]);
            int h = start;
            while (true) {
                const auto& rr = runs.at(cycle[static_cast<std::size_t>(h)]);
                if (rr.owner != g) fail("rtmajor segment owner mismatch");
                seg.local.push_back(rr);
                const int next = (h + 1) % n;
                if (owner[static_cast<std::size_t>(next)] != g) break;
                h = next;
            }
            net_bucket[static_cast<std::size_t>(g)][static_cast<std::size_t>(batch)]
                      [static_cast<std::size_t>(cls)].push_back(std::move(seg));
        }
    }
    if (seen.size() != runs.size()) fail("rtmajor run coverage");

    RtMajorHostPlan out;
    out.batch.resize(static_cast<std::size_t>(ngpu));
    for (auto& g : out.batch) g.resize(static_cast<std::size_t>(batches));
    out.local.resize(static_cast<std::size_t>(ngpu));
    out.scratch_states.assign(static_cast<std::size_t>(ngpu), 0);

    for (int g = 0; g < ngpu; ++g) {
        auto& lo = out.local[static_cast<std::size_t>(g)];
        std::uint32_t cycle_cursor = 0;
        for (int cls = 0; cls < RP_P2P_MAJOR_PC_CLASSES; ++cls) {
            auto& gm = lo.group[static_cast<std::size_t>(cls)];
            gm.begin = cycle_cursor;
            gm.pc = catalan(cls + 1);
            for (const auto& seq : local_bucket[static_cast<std::size_t>(g)]
                                                    [static_cast<std::size_t>(cls)]) {
                if (lo.local.low.size() > std::numeric_limits<std::uint32_t>::max())
                    fail("rtmajor local run offset overflow");
                lo.run_begin.push_back(static_cast<std::uint32_t>(lo.local.low.size()));
                for (const auto& rr : seq) {
                    if (rr.owner != g || rr.pc != gm.pc) fail("rtmajor local record");
                    rtmajor_pack(g, rr.base, lo.local);
                    ++out.local_runs;
                }
                ++cycle_cursor;
                ++out.local_cycles;
            }
            gm.end = cycle_cursor;
        }
        lo.run_begin.push_back(static_cast<std::uint32_t>(lo.local.low.size()));

        for (int b = 0; b < batches; ++b) {
            auto& hb = out.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
            std::uint32_t seg_cursor = 0;
            Rank64 scratch_base = 0;
            for (int cls = 0; cls < RP_P2P_MAJOR_PC_CLASSES; ++cls) {
                auto& gm = hb.group[static_cast<std::size_t>(cls)];
                gm.begin = seg_cursor;
                gm.scratch_base = scratch_base;
                gm.pc = catalan(cls + 1);
                for (const auto& seg : net_bucket[static_cast<std::size_t>(g)]
                                                       [static_cast<std::size_t>(b)]
                                                       [static_cast<std::size_t>(cls)]) {
                    if (hb.local.low.size() > std::numeric_limits<std::uint32_t>::max())
                        fail("rtmajor network run offset overflow");
                    hb.run_begin.push_back(static_cast<std::uint32_t>(hb.local.low.size()));
                    rtmajor_pack(seg.source.owner, seg.source.base, hb.source);
                    if (seg.source.owner == g || seg.source.pc != gm.pc)
                        fail("rtmajor source record");
                    for (const auto& rr : seg.local) {
                        if (rr.owner != g || rr.pc != gm.pc) fail("rtmajor network local record");
                        rtmajor_pack(g, rr.base, hb.local);
                        ++out.local_runs;
                    }
                    ++seg_cursor;
                    ++out.network_segments;
                    out.network_states += gm.pc;
                }
                gm.end = seg_cursor;
                scratch_base += Rank64(gm.end - gm.begin) * gm.pc;
            }
            hb.scratch_states = scratch_base;
            hb.run_begin.push_back(static_cast<std::uint32_t>(hb.local.low.size()));
            if (hb.source.low.size() != seg_cursor || hb.source.high.size() != seg_cursor ||
                hb.run_begin.size() != std::size_t(seg_cursor) + 1)
                fail("rtmajor network flattened shape");
            out.scratch_states[static_cast<std::size_t>(g)] = std::max(
                out.scratch_states[static_cast<std::size_t>(g)], scratch_base);
        }
    }
    return out;
}

struct RtMajorDeviceBatch {
    std::uint32_t* run_begin = nullptr;
    std::uint32_t* source_low = nullptr;
    std::uint8_t* source_high = nullptr;
    std::uint32_t* local_low = nullptr;
    std::uint8_t* local_high = nullptr;
    P2PMajorGroupMeta* group = nullptr;
    Rank64 segments = 0;
};

struct RtMajorDeviceLocal {
    std::uint32_t* run_begin = nullptr;
    std::uint32_t* local_low = nullptr;
    std::uint8_t* local_high = nullptr;
    P2PMajorGroupMeta* group = nullptr;
    Rank64 cycles = 0;
};

struct RtMajorDevicePlan {
    RtMajorDeviceLocal local;
    std::vector<RtMajorDeviceBatch> batch;
};

template<class T>
void rtmajor_upload_vec(T*& dst, const std::vector<T>& src, const char* what) {
    ck(cudaMalloc(&dst, std::max<std::size_t>(1, src.size()) * sizeof(T)), what);
    if (!src.empty())
        ck(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), what);
}

void rtmajor_upload_groups(
    P2PMajorGroupMeta*& dst,
    const std::array<P2PMajorGroupMeta,RP_P2P_MAJOR_PC_CLASSES>& src
) {
    ck(cudaMalloc(&dst, sizeof(P2PMajorGroupMeta) * RP_P2P_MAJOR_PC_CLASSES),
       "rtmajor alloc groups");
    ck(cudaMemcpy(dst, src.data(), sizeof(P2PMajorGroupMeta) * RP_P2P_MAJOR_PC_CLASSES,
                  cudaMemcpyHostToDevice), "rtmajor copy groups");
}

RtMajorDevicePlan rtmajor_upload_plan(
    const RtMajorHostPlan& hp, int g, int batches
) {
    RtMajorDevicePlan d;
    const auto& lo = hp.local[static_cast<std::size_t>(g)];
    d.local.cycles = lo.run_begin.empty() ? 0 : lo.run_begin.size() - 1;
    rtmajor_upload_vec(d.local.run_begin, lo.run_begin, "rtmajor local run_begin");
    rtmajor_upload_vec(d.local.local_low, lo.local.low, "rtmajor local low");
    rtmajor_upload_vec(d.local.local_high, lo.local.high, "rtmajor local high");
    rtmajor_upload_groups(d.local.group, lo.group);
    d.batch.resize(static_cast<std::size_t>(batches));
    for (int b = 0; b < batches; ++b) {
        const auto& hb = hp.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
        auto& db = d.batch[static_cast<std::size_t>(b)];
        db.segments = hb.source.low.size();
        rtmajor_upload_vec(db.run_begin, hb.run_begin, "rtmajor net run_begin");
        rtmajor_upload_vec(db.source_low, hb.source.low, "rtmajor net source low");
        rtmajor_upload_vec(db.source_high, hb.source.high, "rtmajor net source high");
        rtmajor_upload_vec(db.local_low, hb.local.low, "rtmajor net local low");
        rtmajor_upload_vec(db.local_high, hb.local.high, "rtmajor net local high");
        rtmajor_upload_groups(db.group, hb.group);
    }
    return d;
}

void rtmajor_free_plan(RtMajorDevicePlan& d) {
    for (auto& b : d.batch) {
        cudaFree(b.group); cudaFree(b.local_high); cudaFree(b.local_low);
        cudaFree(b.source_high); cudaFree(b.source_low); cudaFree(b.run_begin);
    }
    cudaFree(d.local.group); cudaFree(d.local.local_high);
    cudaFree(d.local.local_low); cudaFree(d.local.run_begin);
}

unsigned rtmajor_blocks(Rank64 n, unsigned cap) {
    const Rank64 one_pass =
        (n + RP_RUNTIME_WARPS_PER_BLOCK - 1) / RP_RUNTIME_WARPS_PER_BLOCK;
    return static_cast<unsigned>(std::max<Rank64>(1, std::min<Rank64>(cap, one_pass)));
}

void rtmajor_launch_redistribution(
    std::vector<RuntimeScheduleDevice>& dev,
    std::vector<RtMajorDevicePlan>& plan,
    std::vector<std::uint32_t*>& scratch,
    int ngpu,
    int batches,
    unsigned blocks
) {
    // Local-only cycles are disjoint from every network cycle.
    for (int g = 0; g < ngpu; ++g) {
        auto& p = plan[static_cast<std::size_t>(g)];
        if (!p.local.cycles) continue;
        ck(cudaSetDevice(g), "rtmajor local set device");
        p2p_major_local_cycle_kernel<<<
            rtmajor_blocks(p.local.cycles, blocks), RP_RUNTIME_THREADS>>>(
            dev[static_cast<std::size_t>(g)].state,
            p.local.run_begin, p.local.local_low, p.local.local_high,
            p.local.group, p.local.cycles, dev[static_cast<std::size_t>(g)].error);
        ck(cudaGetLastError(), "rtmajor local launch");
    }
    runtime_sync_all(ngpu);

    for (int b = 0; b < batches; ++b) {
        for (int g = 0; g < ngpu; ++g) {
            auto& p = plan[static_cast<std::size_t>(g)].batch[static_cast<std::size_t>(b)];
            if (!p.segments) continue;
            ck(cudaSetDevice(g), "rtmajor gather set device");
            p2p_major_gather_kernel<<<
                rtmajor_blocks(p.segments, blocks), RP_RUNTIME_THREADS>>>(
                dev[static_cast<std::size_t>(g)].peer,
                scratch[static_cast<std::size_t>(g)],
                p.source_low, p.source_high, p.group, p.segments,
                ngpu, g, dev[static_cast<std::size_t>(g)].error);
            ck(cudaGetLastError(), "rtmajor gather launch");
        }
        runtime_sync_all(ngpu);
        for (int g = 0; g < ngpu; ++g) {
            auto& p = plan[static_cast<std::size_t>(g)].batch[static_cast<std::size_t>(b)];
            if (!p.segments) continue;
            ck(cudaSetDevice(g), "rtmajor rotate set device");
            p2p_major_rotate_kernel<<<
                rtmajor_blocks(p.segments, blocks), RP_RUNTIME_THREADS>>>(
                dev[static_cast<std::size_t>(g)].state,
                scratch[static_cast<std::size_t>(g)],
                p.run_begin, p.local_low, p.local_high, p.group, p.segments,
                dev[static_cast<std::size_t>(g)].error);
            ck(cudaGetLastError(), "rtmajor rotate launch");
        }
        runtime_sync_all(ngpu);
    }
}

void run_two_row_segment_major_schedule(
    int W, int ngpu, int batches, unsigned blocks, std::uint32_t mod
) {
    if ((W & 1) || W < 8) fail("rtmajor two-row requires even W>=8");
    const int K = (W - 2) / 2;
    ProductionFactorTables tables(W);
    const HostTilePlan tile = make_host_tile_plan(tables, K, ngpu);
    const OwnerPlan owner_plan{tile.owner_begin, tile.owner_size};
    const auto main_words = gen_words(W);

    ModMap initial;
    Rank64 serial = 0;
    for (MateID m : main_words) {
        const std::uint32_t v = static_cast<std::uint32_t>(
            1 + (serial++ * 2654435761ULL) % (mod - 1ULL));
        initial.emplace(Key{false,m}, v);
    }
    ModMap ref = initial;
    for (int p = W - 1; p >= 1; --p) ref = schedule_raw_step(ref, W, p, false, mod);
    for (int p = 1; p < W; ++p) ref = schedule_raw_step(ref, W, p, true, mod);
    ref = schedule_projected_step(ref, W, W - 1, false, mod);

    std::vector<std::uint32_t> flat_input(static_cast<std::size_t>(tables.size()), 0);
    std::vector<std::uint32_t> flat_expected(static_cast<std::size_t>(tables.size()), 0);
    for (const auto& [k,v] : initial) {
        const GroupedRank gr = grouped_rank(
            k, tables, W, W - 1, false, W - 1, K, ngpu, owner_plan);
        flat_input[static_cast<std::size_t>(tile.shard_base[gr.owner] + gr.local)] = v;
    }
    for (const auto& [k,v] : ref) {
        const GroupedRank gr = grouped_rank(
            k, tables, W, W - 2, false, W - 1, K, ngpu, owner_plan);
        flat_expected[static_cast<std::size_t>(tile.shard_base[gr.owner] + gr.local)] = v;
    }

    const RtMajorHostPlan fwd = make_rtmajor_host_plan(
        tables, W, K, false, ngpu, batches, tile);
    const RtMajorHostPlan rev = make_rtmajor_host_plan(
        tables, W, K, true, ngpu, batches, tile);
    const RunTraffic tf = equal_run_traffic_model(W, K, ngpu);
    if (fwd.network_states != tf.moved_states || rev.network_states != tf.moved_states)
        fail("rtmajor network lower-bound mismatch");

    schedule_enable_peer(ngpu);
    std::vector<RuntimeScheduleDevice> dev(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> peer_ptr(static_cast<std::size_t>(ngpu));
    std::vector<HostOwnerComponentPlan> ip(static_cast<std::size_t>(ngpu));
    std::vector<HostOwnerComponentPlan> cp(static_cast<std::size_t>(ngpu));
    std::vector<RtMajorDevicePlan> fdev(static_cast<std::size_t>(ngpu));
    std::vector<RtMajorDevicePlan> rdev(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> scratch(static_cast<std::size_t>(ngpu), nullptr);

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "rtmajor alloc set device");
        install_tables(tables);
        ip[static_cast<std::size_t>(g)] =
            make_host_owner_component_plan(tables, K, g, ngpu);
        cp[static_cast<std::size_t>(g)] =
            make_host_turn_compress_plan(tables, K, g, ngpu);
        auto& d = dev[static_cast<std::size_t>(g)];
        d.interior_components = ip[static_cast<std::size_t>(g)].prefix.back();
        d.compress_components = cp[static_cast<std::size_t>(g)].prefix.back();
        const Rank64 nstate = tile.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&d.state, nstate * sizeof(std::uint32_t)), "rtmajor alloc state");
        peer_ptr[static_cast<std::size_t>(g)] = d.state;
        const Rank64 base = tile.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(d.state, flat_input.data() + base,
                      nstate * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "rtmajor copy state");
        ck(cudaMalloc(&d.owner_begin, ngpu * sizeof(Rank64)), "rtmajor alloc owner begin");
        ck(cudaMemcpy(d.owner_begin, tile.owner_begin.data(),
                      ngpu * sizeof(Rank64), cudaMemcpyHostToDevice),
           "rtmajor copy owner begin");
        runtime_upload_plan(ip[static_cast<std::size_t>(g)], d.iprefix, d.isr, d.icg);
        runtime_upload_plan(cp[static_cast<std::size_t>(g)], d.cprefix, d.csr, d.ccg);
        ck(cudaMalloc(&d.error, sizeof(int)), "rtmajor alloc error");
        ck(cudaMemset(d.error, 0, sizeof(int)), "rtmajor zero error");

        fdev[static_cast<std::size_t>(g)] = rtmajor_upload_plan(fwd, g, batches);
        rdev[static_cast<std::size_t>(g)] = rtmajor_upload_plan(rev, g, batches);
        const Rank64 nscratch = std::max<Rank64>(
            1, std::max(fwd.scratch_states[static_cast<std::size_t>(g)],
                        rev.scratch_states[static_cast<std::size_t>(g)]));
        ck(cudaMalloc(&scratch[static_cast<std::size_t>(g)],
                      nscratch * sizeof(std::uint32_t)), "rtmajor alloc scratch");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "rtmajor peer table set device");
        auto& d = dev[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&d.peer, ngpu * sizeof(std::uint32_t*)), "rtmajor alloc peer table");
        ck(cudaMemcpy(d.peer, peer_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "rtmajor copy peer table");
    }

    const auto t0 = std::chrono::steady_clock::now();
    runtime_launch_turn_all(dev, W, K, true, true, ngpu, blocks, mod);
    for (int p = W - 2; p >= K + 2; --p)
        runtime_launch_interior_all(dev, W, p, false, W - 1, K, ngpu, blocks, mod);
    runtime_sync_all(ngpu);

    rtmajor_launch_redistribution(dev, fdev, scratch, ngpu, batches, blocks);
    for (int p = K + 1; p >= 2; --p)
        runtime_launch_interior_all(dev, W, p, false, K + 1, K, ngpu, blocks, mod);

    runtime_launch_turn_all(dev, W, K, false, false, ngpu, blocks, mod);
    runtime_launch_turn_all(dev, W, K, false, true, ngpu, blocks, mod);
    for (int p = 2; p <= K; ++p)
        runtime_launch_interior_all(dev, W, p, true, 1, K, ngpu, blocks, mod);
    runtime_sync_all(ngpu);

    rtmajor_launch_redistribution(dev, rdev, scratch, ngpu, batches, blocks);
    for (int p = K + 1; p <= W - 2; ++p)
        runtime_launch_interior_all(dev, W, p, true, K + 1, K, ngpu, blocks, mod);

    runtime_launch_turn_all(dev, W, K, true, false, ngpu, blocks, mod);
    runtime_launch_turn_all(dev, W, K, true, true, ngpu, blocks, mod);
    runtime_sync_all(ngpu);
    const double wall_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> flat_output(flat_expected.size());
    Rank64 max_scratch = 0, metadata_bytes = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "rtmajor gather set device");
        auto& d = dev[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(&error, d.error, sizeof(error), cudaMemcpyDeviceToHost),
           "rtmajor copy error");
        if (error)
            fail("rtmajor two-row device error=" + std::to_string(error) +
                 " gpu=" + std::to_string(g));
        const Rank64 nstate = tile.owner_size[static_cast<std::size_t>(g)];
        const Rank64 base = tile.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(flat_output.data() + base, d.state,
                      nstate * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "rtmajor gather state");
        max_scratch = std::max(max_scratch,
            std::max(fwd.scratch_states[static_cast<std::size_t>(g)],
                     rev.scratch_states[static_cast<std::size_t>(g)]));
        for (const RtMajorHostPlan* hp : {&fwd, &rev}) {
            const auto& lo = hp->local[static_cast<std::size_t>(g)];
            metadata_bytes += lo.run_begin.size() * 4ULL + lo.local.low.size() * 5ULL +
                              sizeof(P2PMajorGroupMeta) * RP_P2P_MAJOR_PC_CLASSES;
            for (const auto& hb : hp->batch[static_cast<std::size_t>(g)])
                metadata_bytes += hb.run_begin.size() * 4ULL + hb.source.low.size() * 5ULL +
                                  hb.local.low.size() * 5ULL +
                                  sizeof(P2PMajorGroupMeta) * RP_P2P_MAJOR_PC_CLASSES;
        }
    }
    if (flat_output != flat_expected)
        fail("segment-major two-row runtime mismatch");

    std::cout << "gridfp-reduced-two-row-segment-major-multigpu"
              << " W=" << W << " K=" << K << " ngpu=" << ngpu
              << " batches=" << batches
              << " states=" << tables.size()
              << " two_rows=1 next_row_entry=1 redistributions=2"
              << " exact_network_u32_per_redistribution=" << fwd.network_states
              << " network_lower_bound_exact=1"
              << " max_scratch_MiB_per_gpu="
              << double(max_scratch) * 4.0 / double(1ULL << 20)
              << " forward_reverse_metadata_bytes=" << metadata_bytes
              << " state_streams_per_gpu=1 second_full_state_buffer_bytes=0"
              << " segment_scratch_offset_bytes=0 cycle_id_bytes=0 len_bytes=0"
              << " wall_ms=" << wall_ms << " exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "rtmajor free device");
        rtmajor_free_plan(fdev[static_cast<std::size_t>(g)]);
        rtmajor_free_plan(rdev[static_cast<std::size_t>(g)]);
        cudaFree(scratch[static_cast<std::size_t>(g)]);
        auto& d = dev[static_cast<std::size_t>(g)];
        cudaFree(d.error);
        cudaFree(d.ccg); cudaFree(d.csr); cudaFree(d.cprefix);
        cudaFree(d.icg); cudaFree(d.isr); cudaFree(d.iprefix);
        cudaFree(d.owner_begin); cudaFree(d.peer); cudaFree(d.state);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int ngpu = argc > 2 ? std::atoi(argv[2]) : 2;
    const int batches = argc > 3 ? std::atoi(argv[3]) : 4;
    const unsigned blocks = argc > 4
        ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)) : 256u;
    const std::uint32_t mod = argc > 5
        ? static_cast<std::uint32_t>(std::strtoul(argv[5], nullptr, 10)) : 4294967291u;
    if (W < 8 || W > 11 || (W & 1) || ngpu < 2 || ngpu > SCHEDULE_MAX_GPU ||
        batches < 1 || batches > 64 || !blocks || mod < 3)
        return 2;
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "rtmajor device count");
    if (visible < ngpu) return 3;
    run_two_row_segment_major_schedule(W, ngpu, batches, blocks, mod);
    std::cout << "ALL_OK production_two_row_segment_major_runtime=1\n";
    return 0;
}
