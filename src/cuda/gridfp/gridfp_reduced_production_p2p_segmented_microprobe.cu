#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_packed_schedule_microprobe_main_unused
#include "gridfp_reduced_production_p2p_packed_schedule_microprobe.cu"
#pragma pop_macro("main")

#include <array>
#include <map>
#include <set>

namespace {

struct SegmentMeta {
    Rank64 source_base = 0;
    Rank64 scratch_offset = 0;
    Rank64 local_base[RP_MAX_W]{};
    Rank64 pc = 0;
    int source_owner = -1;
    int len = 0;
};

struct LocalCycleMeta {
    Rank64 local_base[RP_MAX_W]{};
    Rank64 pc = 0;
    int len = 0;
};

struct HostSegmentRun {
    int owner = -1;
    Rank64 base = 0;
    Rank64 pc = 0;
};

std::uint32_t segment_batch_hash(
    std::uint32_t support,
    bool blocked,
    bool reverse
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

struct HostSegmentPlan {
    std::vector<std::vector<std::vector<SegmentMeta>>> segment; // [batch][gpu]
    std::vector<std::vector<LocalCycleMeta>> local;             // [gpu]
    std::vector<Rank64> scratch_states;                         // max over batches/gpu
    Rank64 network_states = 0;
    Rank64 network_segments = 0;
    Rank64 local_cycles = 0;
};

HostSegmentPlan make_segment_plan(
    int W,
    int K,
    bool reverse,
    int ngpu,
    int batches,
    const ProductionFactorTables& tables,
    const HostTilePlan& tile
) {
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);
    const OwnerPlan owner_plan{tile.owner_begin, tile.owner_size};
    const auto main = gen_words(W);
    const auto block = gen_words(W - 1);

    std::map<RunKey, HostSegmentRun> runs;
    for (Key key : layout(main, block, q)) {
        const MateID full = embed_full(key, W, q, reverse);
        const std::uint32_t support = occupancy_mask(full, W);
        const Rank64 pr = tables.primitive_rank(full, W);
        const Rank64 pc = catalan((__builtin_popcount(support) + 1) / 2);
        const GroupedRank gr = grouped_rank(
            key, tables, W, q, reverse, old_start, K, ngpu, owner_plan);
        if (gr.local < pr) fail("segmented host run base underflow");
        const RunKey rk{support, key.blocked};
        const HostSegmentRun rec{gr.owner, gr.local - pr, pc};
        auto [it, inserted] = runs.emplace(rk, rec);
        if (!inserted &&
            (it->second.owner != rec.owner || it->second.base != rec.base || it->second.pc != rec.pc))
            fail("segmented host run inconsistency");
    }

    HostSegmentPlan out;
    out.segment.resize(static_cast<std::size_t>(batches));
    for (auto& b : out.segment) b.resize(static_cast<std::size_t>(ngpu));
    out.local.resize(static_cast<std::size_t>(ngpu));
    out.scratch_states.assign(static_cast<std::size_t>(ngpu), 0);

    std::set<RunKey> seen;
    for (const auto& [root, rec0] : runs) {
        if (seen.count(root)) continue;
        std::vector<RunKey> cycle;
        RunKey cur = root;
        do {
            if (!runs.count(cur)) fail("segmented host cycle escaped run set");
            if (!seen.insert(cur).second) fail("segmented host cycle overlap");
            cycle.push_back(cur);
            cur = shifted_next_key(cur, W, q, K, K, reverse);
            if (cycle.size() > static_cast<std::size_t>(RP_MAX_W))
                fail("segmented host cycle too long");
        } while (!(cur.support == root.support && cur.blocked == root.blocked));
        if (cycle.size() <= 1) continue;

        const Rank64 pc = rec0.pc;
        std::vector<int> owners(cycle.size());
        for (std::size_t i = 0; i < cycle.size(); ++i) {
            const auto& rec = runs.at(cycle[i]);
            if (rec.pc != pc) fail("segmented host pc changed");
            owners[i] = rec.owner;
        }
        std::vector<int> starts;
        for (int i = 0; i < static_cast<int>(cycle.size()); ++i) {
            const int p = (i + static_cast<int>(cycle.size()) - 1) %
                          static_cast<int>(cycle.size());
            if (owners[static_cast<std::size_t>(i)] != owners[static_cast<std::size_t>(p)])
                starts.push_back(i);
        }
        if (starts.empty()) {
            LocalCycleMeta m{};
            m.pc = pc;
            m.len = static_cast<int>(cycle.size());
            const int owner = owners[0];
            for (int i = 0; i < m.len; ++i)
                m.local_base[i] = runs.at(cycle[static_cast<std::size_t>(i)]).base;
            out.local[static_cast<std::size_t>(owner)].push_back(m);
            ++out.local_cycles;
            continue;
        }

        const int batch = int(segment_batch_hash(
            root.support, root.blocked, reverse) % std::uint32_t(batches));
        for (std::size_t si = 0; si < starts.size(); ++si) {
            const int s = starts[si];
            const int next_s = starts[(si + 1) % starts.size()];
            const int n = static_cast<int>(cycle.size());
            const int pred = (s + n - 1) % n;
            const int dest_owner = owners[static_cast<std::size_t>(s)];
            const int source_owner = owners[static_cast<std::size_t>(pred)];
            if (dest_owner == source_owner) fail("segmented boundary owner equality");

            SegmentMeta m{};
            m.source_owner = source_owner;
            m.source_base = runs.at(cycle[static_cast<std::size_t>(pred)]).base;
            m.pc = pc;
            int h = s;
            while (true) {
                if (m.len >= RP_MAX_W) fail("segmented segment length overflow");
                const auto& rec = runs.at(cycle[static_cast<std::size_t>(h)]);
                if (rec.owner != dest_owner) fail("segmented local segment owner mismatch");
                m.local_base[m.len++] = rec.base;
                const int next = (h + 1) % n;
                if (next == next_s) break;
                h = next;
            }
            out.segment[static_cast<std::size_t>(batch)]
                       [static_cast<std::size_t>(dest_owner)].push_back(m);
            out.network_states += pc;
            ++out.network_segments;
        }
    }
    if (seen.size() != runs.size()) fail("segmented host run coverage");

    // Scratch offsets are local to one (batch,destination) and can be reused by
    // every other batch after its gather+rotate barrier pair completes.
    for (int b = 0; b < batches; ++b) {
        for (int g = 0; g < ngpu; ++g) {
            Rank64 off = 0;
            auto& v = out.segment[static_cast<std::size_t>(b)][static_cast<std::size_t>(g)];
            for (auto& m : v) {
                m.scratch_offset = off;
                off += m.pc;
            }
            out.scratch_states[static_cast<std::size_t>(g)] =
                std::max(out.scratch_states[static_cast<std::size_t>(g)], off);
        }
    }

    const RunTraffic traffic = equal_run_traffic_model(W, K, ngpu);
    Rank64 moved = 0;
    for (int s = 0; s < ngpu; ++s)
        for (int d = 0; d < ngpu; ++d)
            if (s != d) moved += traffic.pair_states[s][d];
    if (out.network_states != moved)
        fail("segmented host network lower-bound mismatch");
    return out;
}

__global__ void segmented_local_cycle_kernel(
    std::uint32_t* __restrict__ state,
    const LocalCycleMeta* __restrict__ cycle,
    Rank64 count
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    for (Rank64 ci = first; ci < count; ci += stride) {
        const LocalCycleMeta m = cycle[ci];
        for (Rank64 i = Rank64(lane); i < m.pc; i += 32) {
            std::uint32_t temp = state[m.local_base[m.len - 1] + i];
            for (int h = m.len - 1; h > 0; --h)
                state[m.local_base[h] + i] = state[m.local_base[h - 1] + i];
            state[m.local_base[0] + i] = temp;
        }
    }
}

__global__ void segmented_gather_kernel(
    std::uint32_t** __restrict__ peer_state,
    std::uint32_t* __restrict__ scratch,
    const SegmentMeta* __restrict__ segment,
    Rank64 count
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    for (Rank64 si = first; si < count; si += stride) {
        const SegmentMeta m = segment[si];
        const std::uint32_t* src = peer_state[m.source_owner] + m.source_base;
        std::uint32_t* dst = scratch + m.scratch_offset;
        for (Rank64 i = Rank64(lane); i < m.pc; i += 32)
            dst[i] = src[i];
    }
}

__global__ void segmented_rotate_kernel(
    std::uint32_t* __restrict__ state,
    const std::uint32_t* __restrict__ scratch,
    const SegmentMeta* __restrict__ segment,
    Rank64 count
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    for (Rank64 si = first; si < count; si += stride) {
        const SegmentMeta m = segment[si];
        for (Rank64 i = Rank64(lane); i < m.pc; i += 32) {
            for (int h = m.len - 1; h > 0; --h)
                state[m.local_base[h] + i] = state[m.local_base[h - 1] + i];
            state[m.local_base[0] + i] = scratch[m.scratch_offset + i];
        }
    }
}

struct DeviceSegmentCtx : DevicePeerCtx {
    std::uint32_t* scratch = nullptr;
    LocalCycleMeta* local = nullptr;
    Rank64 local_count = 0;
    std::vector<SegmentMeta*> segment;
    std::vector<Rank64> segment_count;
};

unsigned segment_blocks(Rank64 n, unsigned cap) {
    const Rank64 one_pass = (n + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    return static_cast<unsigned>(std::max<Rank64>(1, std::min<Rank64>(cap, one_pass)));
}

void sync_segment_devices(int ngpu) {
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "segmented sync set device");
        ck(cudaDeviceSynchronize(), "segmented sync");
    }
}

void run_segmented_p2p_probe(
    int W,
    int K,
    bool reverse,
    int ngpu,
    int batches,
    unsigned blocks
) {
    ProductionFactorTables tables(W);
    const HostTilePlan tile = make_host_tile_plan(tables, K, ngpu);
    const HostSegmentPlan hp = make_segment_plan(
        W, K, reverse, ngpu, batches, tables, tile);
    std::vector<std::uint32_t> flat_input, flat_expected;
    build_shift_boundary_vectors(
        W, K, K, reverse, ngpu, tables, tile, flat_input, flat_expected);

    enable_all_peer_access(ngpu);
    std::vector<DeviceSegmentCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> peer_ptr(static_cast<std::size_t>(ngpu));
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "segmented alloc set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 nstate = tile.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, nstate * sizeof(std::uint32_t)), "segmented alloc state");
        peer_ptr[static_cast<std::size_t>(g)] = c.state;
        const Rank64 base = tile.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(c.state, flat_input.data() + base,
                      nstate * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
           "segmented copy state");
        const Rank64 scratch_n = std::max<Rank64>(1, hp.scratch_states[static_cast<std::size_t>(g)]);
        ck(cudaMalloc(&c.scratch, scratch_n * sizeof(std::uint32_t)), "segmented alloc scratch");
        c.local_count = hp.local[static_cast<std::size_t>(g)].size();
        ck(cudaMalloc(&c.local,
                      std::max<Rank64>(1, c.local_count) * sizeof(LocalCycleMeta)),
           "segmented alloc local cycles");
        if (c.local_count)
            ck(cudaMemcpy(c.local, hp.local[static_cast<std::size_t>(g)].data(),
                          c.local_count * sizeof(LocalCycleMeta), cudaMemcpyHostToDevice),
               "segmented copy local cycles");
        c.segment.resize(static_cast<std::size_t>(batches), nullptr);
        c.segment_count.resize(static_cast<std::size_t>(batches), 0);
        for (int b = 0; b < batches; ++b) {
            const auto& hs = hp.segment[static_cast<std::size_t>(b)][static_cast<std::size_t>(g)];
            c.segment_count[static_cast<std::size_t>(b)] = hs.size();
            ck(cudaMalloc(&c.segment[static_cast<std::size_t>(b)],
                          std::max<std::size_t>(1, hs.size()) * sizeof(SegmentMeta)),
               "segmented alloc segment metadata");
            if (!hs.empty())
                ck(cudaMemcpy(c.segment[static_cast<std::size_t>(b)], hs.data(),
                              hs.size() * sizeof(SegmentMeta), cudaMemcpyHostToDevice),
                   "segmented copy segment metadata");
        }
        ck(cudaMalloc(&c.error, sizeof(int)), "segmented alloc error");
        ck(cudaMemset(c.error, 0, sizeof(int)), "segmented zero error");
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "segmented peer table set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.peer, ngpu * sizeof(std::uint32_t*)), "segmented alloc peer table");
        ck(cudaMemcpy(c.peer, peer_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "segmented copy peer table");
    }

    const auto t0 = std::chrono::steady_clock::now();
    // Local-only cycles are independent from all network cycles.
    for (int g = 0; g < ngpu; ++g) {
        auto& c = ctx[static_cast<std::size_t>(g)];
        if (!c.local_count) continue;
        ck(cudaSetDevice(g), "segmented local set device");
        segmented_local_cycle_kernel<<<segment_blocks(c.local_count, blocks), THREADS>>>(
            c.state, c.local, c.local_count);
        ck(cudaGetLastError(), "segmented local launch");
    }
    sync_segment_devices(ngpu);

    for (int b = 0; b < batches; ++b) {
        // Phase 1: capture every cross-owner predecessor for every cycle in the
        // batch BEFORE any state in those cycles is modified.
        for (int g = 0; g < ngpu; ++g) {
            auto& c = ctx[static_cast<std::size_t>(g)];
            const Rank64 n = c.segment_count[static_cast<std::size_t>(b)];
            if (!n) continue;
            ck(cudaSetDevice(g), "segmented gather set device");
            segmented_gather_kernel<<<segment_blocks(n, blocks), THREADS>>>(
                c.peer, c.scratch, c.segment[static_cast<std::size_t>(b)], n);
            ck(cudaGetLastError(), "segmented gather launch");
        }
        sync_segment_devices(ngpu);

        // Phase 2: all remaining moves are owner-local.  Backward traversal
        // preserves old source values within each segment.
        for (int g = 0; g < ngpu; ++g) {
            auto& c = ctx[static_cast<std::size_t>(g)];
            const Rank64 n = c.segment_count[static_cast<std::size_t>(b)];
            if (!n) continue;
            ck(cudaSetDevice(g), "segmented rotate set device");
            segmented_rotate_kernel<<<segment_blocks(n, blocks), THREADS>>>(
                c.state, c.scratch, c.segment[static_cast<std::size_t>(b)], n);
            ck(cudaGetLastError(), "segmented rotate launch");
        }
        sync_segment_devices(ngpu);
    }
    const double wall_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> flat_output(flat_expected.size());
    Rank64 max_scratch = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "segmented gather output set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost),
           "segmented copy error");
        if (error) fail("segmented device error=" + std::to_string(error));
        const Rank64 n = tile.owner_size[static_cast<std::size_t>(g)];
        const Rank64 base = tile.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(flat_output.data() + base, c.state,
                      n * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
           "segmented gather state");
        max_scratch = std::max(max_scratch, hp.scratch_states[static_cast<std::size_t>(g)]);
    }
    if (flat_output != flat_expected)
        fail("segmented P2P redistribution mismatch");

    std::cout << "gridfp-reduced-production-p2p-segmented"
              << " W=" << W << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu
              << " batches=" << batches
              << " network_segments=" << hp.network_segments
              << " local_cycles=" << hp.local_cycles
              << " exact_network_u32_values=" << hp.network_states
              << " exact_network_GiB="
              << double(hp.network_states) * 4.0 / double(1ULL << 30)
              << " max_scratch_MiB_per_gpu="
              << double(max_scratch) * 4.0 / double(1ULL << 20)
              << " wall_ms=" << wall_ms
              << " one_peer_read_per_owner_boundary_value=1"
              << " network_lower_bound_exact=1"
              << " cycle_atomic_batches=1 staging_state_buffer=0 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "segmented free set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        for (auto* p : c.segment) cudaFree(p);
        cudaFree(c.local); cudaFree(c.scratch);
        cudaFree(c.error); cudaFree(c.peer); cudaFree(c.state);
    }
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int K = argc > 2 ? std::atoi(argv[2]) : (W - 2) / 2;
    const int batches = argc > 3 ? std::atoi(argv[3]) : 4;
    const unsigned blocks = argc > 4
        ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)) : 256u;
    const int ngpu = argc > 5 ? std::atoi(argv[5]) : 2;
    if (W < 8 || W > 11 || (W & 1) || K != (W - 2) / 2 ||
        batches < 1 || batches > 64 || !blocks || ngpu < 2 || ngpu > P2P_MAX_GPU)
        return 2;
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "segmented device count");
    if (visible < ngpu) return 3;

    run_segmented_p2p_probe(W, K, false, ngpu, batches, blocks);
    run_segmented_p2p_probe(W, K, true, ngpu, batches, blocks);
    std::cout << "ALL_OK production_p2p_segmented_cuda=1\n";
    return 0;
}
