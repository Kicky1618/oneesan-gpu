#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_segmented_microprobe_main_unused
#include "gridfp_reduced_production_p2p_segmented_microprobe.cu"
#pragma pop_macro("main")

#include <array>

namespace {

static constexpr int MAJOR_PC_CLASSES = 14;

int major_pc_class_host(Rank64 pc) {
    for (int c = 0; c < MAJOR_PC_CLASSES; ++c)
        if (catalan(c + 1) == pc) return c;
    fail("segment-major host primitive class");
}

void major_pack39_host(int owner, Rank64 local, std::uint32_t& low, std::uint8_t& high) {
    if (owner < 0 || owner >= 8 || local >= (Rank64(1) << 36))
        fail("segment-major host packed run range");
    low = std::uint32_t(owner) |
          (std::uint32_t(local & ((Rank64(1) << 29) - 1)) << 3);
    high = static_cast<std::uint8_t>((local >> 29) & 0x7fu);
}

struct MajorGroupMeta {
    std::uint32_t begin = 0;
    std::uint32_t end = 0;
    Rank64 scratch_base = 0;
    Rank64 pc = 0;
};
static_assert(sizeof(MajorGroupMeta) == 24);

struct HostMajorBatch {
    std::vector<std::uint32_t> run_begin;
    std::vector<std::uint32_t> source_low;
    std::vector<std::uint8_t> source_high;
    std::vector<std::uint32_t> local_low;
    std::vector<std::uint8_t> local_high;
    std::array<MajorGroupMeta, MAJOR_PC_CLASSES> group{};
    Rank64 scratch_states = 0;
};

struct HostMajorLocal {
    std::vector<std::uint32_t> run_begin;
    std::vector<std::uint32_t> local_low;
    std::vector<std::uint8_t> local_high;
    std::array<MajorGroupMeta, MAJOR_PC_CLASSES> group{};
};

struct HostMajorPlan {
    std::vector<std::vector<HostMajorBatch>> batch; // [gpu][batch]
    std::vector<HostMajorLocal> local;              // [gpu]
    std::vector<Rank64> scratch_states;
    Rank64 network_segments = 0;
    Rank64 local_cycles = 0;
    Rank64 local_run_records = 0;
};

HostMajorPlan make_major_plan(const HostSegmentPlan& hp, int ngpu, int batches) {
    HostMajorPlan out;
    out.batch.resize(static_cast<std::size_t>(ngpu));
    for (auto& g : out.batch) g.resize(static_cast<std::size_t>(batches));
    out.local.resize(static_cast<std::size_t>(ngpu));
    out.scratch_states.assign(static_cast<std::size_t>(ngpu), 0);

    for (int g = 0; g < ngpu; ++g) {
        // Local-only cycles grouped by primitive multiplicity.
        std::array<std::vector<const LocalCycleMeta*>, MAJOR_PC_CLASSES> lbucket;
        for (const auto& m : hp.local[static_cast<std::size_t>(g)])
            lbucket[static_cast<std::size_t>(major_pc_class_host(m.pc))].push_back(&m);
        auto& lo = out.local[static_cast<std::size_t>(g)];
        std::uint32_t cycle_cursor = 0;
        for (int cls = 0; cls < MAJOR_PC_CLASSES; ++cls) {
            auto& gm = lo.group[static_cast<std::size_t>(cls)];
            gm.begin = cycle_cursor;
            gm.pc = catalan(cls + 1);
            for (const LocalCycleMeta* pm : lbucket[static_cast<std::size_t>(cls)]) {
                const auto& m = *pm;
                if (lo.local_low.size() > std::numeric_limits<std::uint32_t>::max())
                    fail("segment-major local run offset overflow");
                lo.run_begin.push_back(static_cast<std::uint32_t>(lo.local_low.size()));
                for (int h = 0; h < m.len; ++h) {
                    std::uint32_t low = 0; std::uint8_t high = 0;
                    major_pack39_host(g, m.local_base[h], low, high);
                    lo.local_low.push_back(low); lo.local_high.push_back(high);
                    ++out.local_run_records;
                }
                ++cycle_cursor;
                ++out.local_cycles;
            }
            gm.end = cycle_cursor;
        }
        lo.run_begin.push_back(static_cast<std::uint32_t>(lo.local_low.size()));

        for (int b = 0; b < batches; ++b) {
            auto& dst = out.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
            std::array<std::vector<const SegmentMeta*>, MAJOR_PC_CLASSES> bucket;
            for (const auto& m : hp.segment[static_cast<std::size_t>(b)][static_cast<std::size_t>(g)])
                bucket[static_cast<std::size_t>(major_pc_class_host(m.pc))].push_back(&m);
            std::uint32_t seg_cursor = 0;
            Rank64 scratch_base = 0;
            for (int cls = 0; cls < MAJOR_PC_CLASSES; ++cls) {
                auto& gm = dst.group[static_cast<std::size_t>(cls)];
                gm.begin = seg_cursor;
                gm.scratch_base = scratch_base;
                gm.pc = catalan(cls + 1);
                for (const SegmentMeta* pm : bucket[static_cast<std::size_t>(cls)]) {
                    const auto& m = *pm;
                    if (dst.local_low.size() > std::numeric_limits<std::uint32_t>::max())
                        fail("segment-major network run offset overflow");
                    dst.run_begin.push_back(static_cast<std::uint32_t>(dst.local_low.size()));
                    std::uint32_t low = 0; std::uint8_t high = 0;
                    major_pack39_host(m.source_owner, m.source_base, low, high);
                    dst.source_low.push_back(low); dst.source_high.push_back(high);
                    for (int h = 0; h < m.len; ++h) {
                        major_pack39_host(g, m.local_base[h], low, high);
                        dst.local_low.push_back(low); dst.local_high.push_back(high);
                        ++out.local_run_records;
                    }
                    ++seg_cursor;
                    ++out.network_segments;
                }
                gm.end = seg_cursor;
                scratch_base += Rank64(gm.end - gm.begin) * gm.pc;
            }
            dst.scratch_states = scratch_base;
            dst.run_begin.push_back(static_cast<std::uint32_t>(dst.local_low.size()));
            if (dst.source_low.size() != seg_cursor || dst.source_high.size() != seg_cursor ||
                dst.run_begin.size() != std::size_t(seg_cursor) + 1)
                fail("segment-major network shape");
            out.scratch_states[static_cast<std::size_t>(g)] = std::max(
                out.scratch_states[static_cast<std::size_t>(g)], scratch_base);
        }
    }
    if (out.network_segments != hp.network_segments || out.local_cycles != hp.local_cycles)
        fail("segment-major host count mismatch");
    return out;
}

__device__ __forceinline__ Rank64 major_unpack_local(
    std::uint32_t low, std::uint8_t high
) {
    return Rank64(low >> 3) | (Rank64(high & 0x7fu) << 29);
}

__device__ __forceinline__ int major_find_class(
    Rank64 index, const MajorGroupMeta* group
) {
#pragma unroll
    for (int cls = 0; cls < MAJOR_PC_CLASSES; ++cls)
        if (index < group[cls].end) return cls;
    return -1;
}

__global__ void major_local_cycle_kernel(
    std::uint32_t* __restrict__ state,
    const std::uint32_t* __restrict__ run_begin,
    const std::uint32_t* __restrict__ local_low,
    const std::uint8_t* __restrict__ local_high,
    const MajorGroupMeta* __restrict__ group,
    Rank64 cycles,
    int* error
) {
    __shared__ Rank64 sh_base[WARPS_PER_BLOCK][RP_MAX_W];
    __shared__ int sh_len[WARPS_PER_BLOCK];
    __shared__ Rank64 sh_pc[WARPS_PER_BLOCK];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    for (Rank64 ci = first; ci < cycles; ci += stride) {
        if (lane == 0) {
            sh_len[warp] = 0;
            const int cls = major_find_class(ci, group);
            const std::uint32_t begin = run_begin[ci], end = run_begin[ci + 1];
            if (cls < 0 || end <= begin || end - begin > RP_MAX_W) {
                atomicCAS(error, 0, 421);
            } else {
                const int len = int(end - begin);
                for (int h = 0; h < len; ++h)
                    sh_base[warp][h] = major_unpack_local(local_low[begin + h], local_high[begin + h]);
                sh_len[warp] = len;
                sh_pc[warp] = group[cls].pc;
            }
        }
        __syncwarp();
        const int len = sh_len[warp];
        if (len > 1) {
            const Rank64 pc = sh_pc[warp];
            for (Rank64 i = Rank64(lane); i < pc; i += 32) {
                std::uint32_t temp = state[sh_base[warp][len - 1] + i];
                for (int h = len - 1; h > 0; --h)
                    state[sh_base[warp][h] + i] = state[sh_base[warp][h - 1] + i];
                state[sh_base[warp][0] + i] = temp;
            }
        }
        __syncwarp();
    }
}

__global__ void major_gather_kernel(
    std::uint32_t** __restrict__ peer_state,
    std::uint32_t* __restrict__ scratch,
    const std::uint32_t* __restrict__ source_low,
    const std::uint8_t* __restrict__ source_high,
    const MajorGroupMeta* __restrict__ group,
    Rank64 segments,
    int* error
) {
    __shared__ Rank64 sh_source[WARPS_PER_BLOCK];
    __shared__ Rank64 sh_scratch[WARPS_PER_BLOCK];
    __shared__ Rank64 sh_pc[WARPS_PER_BLOCK];
    __shared__ int sh_owner[WARPS_PER_BLOCK];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    for (Rank64 si = first; si < segments; si += stride) {
        if (lane == 0) {
            const int cls = major_find_class(si, group);
            if (cls < 0) {
                atomicCAS(error, 0, 431);
                sh_pc[warp] = 0;
            } else {
                const MajorGroupMeta gm = group[cls];
                const std::uint32_t low = source_low[si];
                const int owner = int(low & 7u);
                sh_owner[warp] = owner;
                sh_source[warp] = major_unpack_local(low, source_high[si]);
                sh_pc[warp] = gm.pc;
                sh_scratch[warp] = gm.scratch_base + (si - gm.begin) * gm.pc;
            }
        }
        __syncwarp();
        const Rank64 pc = sh_pc[warp];
        if (pc) {
            const std::uint32_t* src = peer_state[sh_owner[warp]] + sh_source[warp];
            std::uint32_t* dst = scratch + sh_scratch[warp];
            for (Rank64 i = Rank64(lane); i < pc; i += 32) dst[i] = src[i];
        }
        __syncwarp();
    }
}

__global__ void major_rotate_kernel(
    std::uint32_t* __restrict__ state,
    const std::uint32_t* __restrict__ scratch,
    const std::uint32_t* __restrict__ run_begin,
    const std::uint32_t* __restrict__ local_low,
    const std::uint8_t* __restrict__ local_high,
    const MajorGroupMeta* __restrict__ group,
    Rank64 segments,
    int* error
) {
    __shared__ Rank64 sh_base[WARPS_PER_BLOCK][RP_MAX_W];
    __shared__ int sh_len[WARPS_PER_BLOCK];
    __shared__ Rank64 sh_pc[WARPS_PER_BLOCK];
    __shared__ Rank64 sh_scratch[WARPS_PER_BLOCK];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    for (Rank64 si = first; si < segments; si += stride) {
        if (lane == 0) {
            sh_len[warp] = 0;
            const int cls = major_find_class(si, group);
            const std::uint32_t begin = run_begin[si], end = run_begin[si + 1];
            if (cls < 0 || end <= begin || end - begin > RP_MAX_W) {
                atomicCAS(error, 0, 441);
            } else {
                const MajorGroupMeta gm = group[cls];
                const int len = int(end - begin);
                for (int h = 0; h < len; ++h)
                    sh_base[warp][h] = major_unpack_local(local_low[begin + h], local_high[begin + h]);
                sh_len[warp] = len;
                sh_pc[warp] = gm.pc;
                sh_scratch[warp] = gm.scratch_base + (si - gm.begin) * gm.pc;
            }
        }
        __syncwarp();
        const int len = sh_len[warp];
        if (len > 0) {
            const Rank64 pc = sh_pc[warp];
            for (Rank64 i = Rank64(lane); i < pc; i += 32) {
                for (int h = len - 1; h > 0; --h)
                    state[sh_base[warp][h] + i] = state[sh_base[warp][h - 1] + i];
                state[sh_base[warp][0] + i] = scratch[sh_scratch[warp] + i];
            }
        }
        __syncwarp();
    }
}

struct DeviceMajorBatch {
    std::uint32_t* run_begin = nullptr;
    std::uint32_t* source_low = nullptr;
    std::uint8_t* source_high = nullptr;
    std::uint32_t* local_low = nullptr;
    std::uint8_t* local_high = nullptr;
    MajorGroupMeta* group = nullptr;
    Rank64 segments = 0;
};

struct DeviceMajorLocal {
    std::uint32_t* run_begin = nullptr;
    std::uint32_t* local_low = nullptr;
    std::uint8_t* local_high = nullptr;
    MajorGroupMeta* group = nullptr;
    Rank64 cycles = 0;
};

struct DeviceMajorCtx : DevicePeerCtx {
    std::uint32_t* scratch = nullptr;
    DeviceMajorLocal local_major;
    std::vector<DeviceMajorBatch> batch_major;
};

template<class T>
void major_copy_alloc(T*& dst, const std::vector<T>& src, const char* what) {
    ck(cudaMalloc(&dst, std::max<std::size_t>(1, src.size()) * sizeof(T)), what);
    if (!src.empty()) ck(cudaMemcpy(dst, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice), what);
}

void major_copy_groups(MajorGroupMeta*& dst, const std::array<MajorGroupMeta,MAJOR_PC_CLASSES>& src) {
    ck(cudaMalloc(&dst, sizeof(MajorGroupMeta) * MAJOR_PC_CLASSES), "major alloc groups");
    ck(cudaMemcpy(dst, src.data(), sizeof(MajorGroupMeta) * MAJOR_PC_CLASSES,
                  cudaMemcpyHostToDevice), "major copy groups");
}

void run_segment_major_probe(
    int W, int K, bool reverse, int ngpu, int batches, unsigned blocks
) {
    ProductionFactorTables tables(W);
    const HostTilePlan tile = make_host_tile_plan(tables, K, ngpu);
    const HostSegmentPlan hp = make_segment_plan(W, K, reverse, ngpu, batches, tables, tile);
    const HostMajorPlan major = make_major_plan(hp, ngpu, batches);
    std::vector<std::uint32_t> flat_input, flat_expected;
    build_shift_boundary_vectors(W, K, K, reverse, ngpu, tables, tile, flat_input, flat_expected);

    enable_all_peer_access(ngpu);
    std::vector<DeviceMajorCtx> ctx(static_cast<std::size_t>(ngpu));
    std::vector<std::uint32_t*> peer_ptr(static_cast<std::size_t>(ngpu));
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "major alloc set device");
        install_tables(tables);
        auto& c = ctx[static_cast<std::size_t>(g)];
        const Rank64 nstate = tile.owner_size[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.state, nstate * sizeof(std::uint32_t)), "major alloc state");
        peer_ptr[static_cast<std::size_t>(g)] = c.state;
        const Rank64 base = tile.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(c.state, flat_input.data() + base, nstate * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice), "major copy state");
        const Rank64 scratch_n = std::max<Rank64>(1, major.scratch_states[static_cast<std::size_t>(g)]);
        ck(cudaMalloc(&c.scratch, scratch_n * sizeof(std::uint32_t)), "major alloc scratch");
        ck(cudaMalloc(&c.error, sizeof(int)), "major alloc error");
        ck(cudaMemset(c.error, 0, sizeof(int)), "major zero error");

        const auto& hl = major.local[static_cast<std::size_t>(g)];
        c.local_major.cycles = hl.run_begin.empty() ? 0 : hl.run_begin.size() - 1;
        major_copy_alloc(c.local_major.run_begin, hl.run_begin, "major local run_begin");
        major_copy_alloc(c.local_major.local_low, hl.local_low, "major local low");
        major_copy_alloc(c.local_major.local_high, hl.local_high, "major local high");
        major_copy_groups(c.local_major.group, hl.group);

        c.batch_major.resize(static_cast<std::size_t>(batches));
        for (int b = 0; b < batches; ++b) {
            const auto& hb = major.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
            auto& db = c.batch_major[static_cast<std::size_t>(b)];
            db.segments = hb.source_low.size();
            major_copy_alloc(db.run_begin, hb.run_begin, "major network run_begin");
            major_copy_alloc(db.source_low, hb.source_low, "major source low");
            major_copy_alloc(db.source_high, hb.source_high, "major source high");
            major_copy_alloc(db.local_low, hb.local_low, "major local low");
            major_copy_alloc(db.local_high, hb.local_high, "major local high");
            major_copy_groups(db.group, hb.group);
        }
    }
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "major peer table set device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        ck(cudaMalloc(&c.peer, ngpu * sizeof(std::uint32_t*)), "major alloc peer table");
        ck(cudaMemcpy(c.peer, peer_ptr.data(), ngpu * sizeof(std::uint32_t*),
                      cudaMemcpyHostToDevice), "major copy peer table");
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int g = 0; g < ngpu; ++g) {
        auto& c = ctx[static_cast<std::size_t>(g)];
        if (!c.local_major.cycles) continue;
        ck(cudaSetDevice(g), "major local set device");
        major_local_cycle_kernel<<<segment_blocks(c.local_major.cycles, blocks), THREADS>>>(
            c.state, c.local_major.run_begin, c.local_major.local_low,
            c.local_major.local_high, c.local_major.group, c.local_major.cycles, c.error);
        ck(cudaGetLastError(), "major local launch");
    }
    sync_segment_devices(ngpu);

    for (int b = 0; b < batches; ++b) {
        for (int g = 0; g < ngpu; ++g) {
            auto& c = ctx[static_cast<std::size_t>(g)];
            auto& db = c.batch_major[static_cast<std::size_t>(b)];
            if (!db.segments) continue;
            ck(cudaSetDevice(g), "major gather set device");
            major_gather_kernel<<<segment_blocks(db.segments, blocks), THREADS>>>(
                c.peer, c.scratch, db.source_low, db.source_high, db.group,
                db.segments, c.error);
            ck(cudaGetLastError(), "major gather launch");
        }
        sync_segment_devices(ngpu);
        for (int g = 0; g < ngpu; ++g) {
            auto& c = ctx[static_cast<std::size_t>(g)];
            auto& db = c.batch_major[static_cast<std::size_t>(b)];
            if (!db.segments) continue;
            ck(cudaSetDevice(g), "major rotate set device");
            major_rotate_kernel<<<segment_blocks(db.segments, blocks), THREADS>>>(
                c.state, c.scratch, db.run_begin, db.local_low, db.local_high,
                db.group, db.segments, c.error);
            ck(cudaGetLastError(), "major rotate launch");
        }
        sync_segment_devices(ngpu);
    }
    const double wall_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now() - t0).count();

    std::vector<std::uint32_t> flat_output(flat_expected.size());
    Rank64 metadata_bytes = 0, max_scratch = 0;
    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "major gather output device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        int error = 0;
        ck(cudaMemcpy(&error, c.error, sizeof(error), cudaMemcpyDeviceToHost), "major copy error");
        if (error) fail("segment-major device error=" + std::to_string(error));
        const Rank64 nstate = tile.owner_size[static_cast<std::size_t>(g)];
        const Rank64 base = tile.shard_base[static_cast<std::size_t>(g)];
        ck(cudaMemcpy(flat_output.data() + base, c.state, nstate * sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost), "major gather state");
        max_scratch = std::max(max_scratch, major.scratch_states[static_cast<std::size_t>(g)]);
        const auto& hl = major.local[static_cast<std::size_t>(g)];
        metadata_bytes += hl.run_begin.size() * 4ULL + hl.local_low.size() * 5ULL +
                          sizeof(MajorGroupMeta) * MAJOR_PC_CLASSES;
        for (const auto& hb : major.batch[static_cast<std::size_t>(g)])
            metadata_bytes += hb.run_begin.size() * 4ULL + hb.source_low.size() * 5ULL +
                              hb.local_low.size() * 5ULL + sizeof(MajorGroupMeta) * MAJOR_PC_CLASSES;
    }
    if (flat_output != flat_expected) fail("segment-major P2P redistribution mismatch");

    std::cout << "gridfp-reduced-production-p2p-segment-major"
              << " W=" << W << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu << " batches=" << batches
              << " network_segments=" << major.network_segments
              << " local_cycles=" << major.local_cycles
              << " local_run_records=" << major.local_run_records
              << " metadata_bytes=" << metadata_bytes
              << " exact_network_u32_values=" << hp.network_states
              << " max_scratch_MiB_per_gpu=" << double(max_scratch) * 4.0 / double(1ULL << 20)
              << " wall_ms=" << wall_ms
              << " per_segment_scratch_offset_bytes=0 cycle_id_bytes=0 len_bytes=0"
              << " one_peer_read_per_owner_boundary_value=1 exact=OK\n";

    for (int g = 0; g < ngpu; ++g) {
        ck(cudaSetDevice(g), "major free device");
        auto& c = ctx[static_cast<std::size_t>(g)];
        for (auto& db : c.batch_major) {
            cudaFree(db.group); cudaFree(db.local_high); cudaFree(db.local_low);
            cudaFree(db.source_high); cudaFree(db.source_low); cudaFree(db.run_begin);
        }
        cudaFree(c.local_major.group); cudaFree(c.local_major.local_high);
        cudaFree(c.local_major.local_low); cudaFree(c.local_major.run_begin);
        cudaFree(c.error); cudaFree(c.peer); cudaFree(c.scratch); cudaFree(c.state);
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
    ck(cudaGetDeviceCount(&visible), "segment-major device count");
    if (visible < ngpu) return 3;
    run_segment_major_probe(W, K, false, ngpu, batches, blocks);
    run_segment_major_probe(W, K, true, ngpu, batches, blocks);
    std::cout << "ALL_OK production_p2p_segment_major_cuda=1\n";
    return 0;
}
