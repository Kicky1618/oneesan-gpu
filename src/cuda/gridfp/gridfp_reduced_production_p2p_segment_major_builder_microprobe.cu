#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_p2p_segment_major_microprobe_main_unused
#include "gridfp_reduced_production_p2p_segment_major_microprobe.cu"
#pragma pop_macro("main")

namespace {

__device__ __forceinline__ std::uint32_t major_batch_hash_device(
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

__device__ __forceinline__ int major_group_index(
    int gpu, int batch, int cls, int batches
) {
    return (gpu * batches + batch) * MAJOR_PC_CLASSES + cls;
}

__global__ void segment_major_count_kernel(
    Rank64 base_supports,
    int W,
    int K,
    bool reverse,
    int ngpu,
    int batches,
    unsigned long long* __restrict__ network_segments,
    unsigned long long* __restrict__ network_runs,
    unsigned long long* __restrict__ local_cycles,
    unsigned long long* __restrict__ local_runs,
    unsigned long long* __restrict__ network_states,
    int* error
) {
    const Rank64 first = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    const Rank64 stride = Rank64(gridDim.x) * blockDim.x;
    const int old_start = reverse ? 1 : W - 1;
    const int q = old_start + (reverse ? K : -K);
    for (Rank64 base_rank = first; base_rank < base_supports; base_rank += stride) {
        EqualTileRunSeed seeds[3]{};
        const int nr = equal_tile_run_seeds_device(base_rank, W, q, reverse, seeds);
        for (int ri = 0; ri < nr; ++ri) {
            const EqualTileRunSeed root = seeds[ri];
            const bool blocked = root.blocked != 0;
            const int len = shift_cycle_leader_length_device(
                root.support, blocked, W, q, K, K, reverse);
            if (len < 0 || len > RP_MAX_W) {
                atomicCAS(error, 0, 451);
                continue;
            }
            if (len <= 1) continue; // nonleader, fixed point, or trivial cycle

            const int occupied = __popc(root.support);
            if (!(occupied & 1)) {
                atomicCAS(error, 0, 452);
                continue;
            }
            const int cls = (occupied + 1) / 2 - 1;
            if (cls < 0 || cls >= MAJOR_PC_CLASSES) {
                atomicCAS(error, 0, 453);
                continue;
            }
            const Rank64 pc = RP_PRIMITIVE[occupied][1];

            int owner[RP_MAX_W]{};
            std::uint32_t cur = root.support;
            bool owner_ok = true;
            for (int h = 0; h < len; ++h) {
                owner[h] = p2p_support_owner_device(
                    cur, W, old_start, K, reverse, ngpu);
                if (owner[h] < 0 || owner[h] >= ngpu) {
                    owner_ok = false;
                    break;
                }
                cur = shift_next_support_device(cur, blocked, W, q, K, K, reverse);
            }
            if (!owner_ok) {
                atomicCAS(error, 0, 454);
                continue;
            }

            int boundaries = 0;
            for (int h = 0; h < len; ++h)
                boundaries += owner[h] != owner[(h + len - 1) % len];
            if (!boundaries) {
                const int g = owner[0];
                const int idx = g * MAJOR_PC_CLASSES + cls;
                atomicAdd(local_cycles + idx, 1ULL);
                atomicAdd(local_runs + idx, static_cast<unsigned long long>(len));
                continue;
            }

            const int batch = int(major_batch_hash_device(root.support, blocked, reverse) %
                                  std::uint32_t(batches));
            for (int start = 0; start < len; ++start) {
                const int pred = (start + len - 1) % len;
                if (owner[start] == owner[pred]) continue;
                const int g = owner[start];
                int seg_len = 1;
                while (seg_len < len && owner[(start + seg_len) % len] == g) ++seg_len;
                if (seg_len >= len) {
                    atomicCAS(error, 0, 455);
                    break;
                }
                const int idx = major_group_index(g, batch, cls, batches);
                atomicAdd(network_segments + idx, 1ULL);
                atomicAdd(network_runs + idx, static_cast<unsigned long long>(seg_len));
                atomicAdd(network_states, static_cast<unsigned long long>(pc));
            }
        }
    }
}

struct MajorCountHost {
    std::vector<unsigned long long> network_segments;
    std::vector<unsigned long long> network_runs;
    std::vector<unsigned long long> local_cycles;
    std::vector<unsigned long long> local_runs;
    unsigned long long network_states = 0;
};

MajorCountHost expected_major_counts(
    const HostMajorPlan& major, int ngpu, int batches
) {
    MajorCountHost out;
    out.network_segments.assign(
        static_cast<std::size_t>(ngpu * batches * MAJOR_PC_CLASSES), 0);
    out.network_runs.assign(out.network_segments.size(), 0);
    out.local_cycles.assign(static_cast<std::size_t>(ngpu * MAJOR_PC_CLASSES), 0);
    out.local_runs.assign(out.local_cycles.size(), 0);
    for (int g = 0; g < ngpu; ++g) {
        const auto& lo = major.local[static_cast<std::size_t>(g)];
        for (int cls = 0; cls < MAJOR_PC_CLASSES; ++cls) {
            const auto gm = lo.group[static_cast<std::size_t>(cls)];
            const int idx = g * MAJOR_PC_CLASSES + cls;
            out.local_cycles[static_cast<std::size_t>(idx)] = gm.end - gm.begin;
            Rank64 runs = 0;
            for (std::uint32_t i = gm.begin; i < gm.end; ++i)
                runs += lo.run_begin[static_cast<std::size_t>(i + 1)] -
                        lo.run_begin[static_cast<std::size_t>(i)];
            out.local_runs[static_cast<std::size_t>(idx)] = runs;
        }
        for (int b = 0; b < batches; ++b) {
            const auto& hb = major.batch[static_cast<std::size_t>(g)][static_cast<std::size_t>(b)];
            for (int cls = 0; cls < MAJOR_PC_CLASSES; ++cls) {
                const auto gm = hb.group[static_cast<std::size_t>(cls)];
                const int idx = major_group_index(g, b, cls, batches);
                const Rank64 nseg = gm.end - gm.begin;
                out.network_segments[static_cast<std::size_t>(idx)] = nseg;
                Rank64 runs = 0;
                for (std::uint32_t i = gm.begin; i < gm.end; ++i)
                    runs += hb.run_begin[static_cast<std::size_t>(i + 1)] -
                            hb.run_begin[static_cast<std::size_t>(i)];
                out.network_runs[static_cast<std::size_t>(idx)] = runs;
                out.network_states += nseg * gm.pc;
            }
        }
    }
    return out;
}

void run_segment_major_count_probe(
    int W, int K, bool reverse, int ngpu, int batches, unsigned blocks
) {
    ProductionFactorTables tables(W);
    const HostTilePlan tile = make_host_tile_plan(tables, K, ngpu);
    const HostSegmentPlan hp = make_segment_plan(W, K, reverse, ngpu, batches, tables, tile);
    const HostMajorPlan major = make_major_plan(hp, ngpu, batches);
    const MajorCountHost want = expected_major_counts(major, ngpu, batches);

    ck(cudaSetDevice(0), "major count set device");
    install_tables(tables);
    const int ngroups = ngpu * batches * MAJOR_PC_CLASSES;
    const int nlocal = ngpu * MAJOR_PC_CLASSES;
    unsigned long long *d_ns = nullptr, *d_nr = nullptr, *d_lc = nullptr, *d_lr = nullptr;
    unsigned long long* d_states = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_ns, ngroups * sizeof(unsigned long long)), "major count alloc ns");
    ck(cudaMalloc(&d_nr, ngroups * sizeof(unsigned long long)), "major count alloc nr");
    ck(cudaMalloc(&d_lc, nlocal * sizeof(unsigned long long)), "major count alloc lc");
    ck(cudaMalloc(&d_lr, nlocal * sizeof(unsigned long long)), "major count alloc lr");
    ck(cudaMalloc(&d_states, sizeof(unsigned long long)), "major count alloc states");
    ck(cudaMalloc(&d_error, sizeof(int)), "major count alloc error");
    ck(cudaMemset(d_ns, 0, ngroups * sizeof(unsigned long long)), "major count zero ns");
    ck(cudaMemset(d_nr, 0, ngroups * sizeof(unsigned long long)), "major count zero nr");
    ck(cudaMemset(d_lc, 0, nlocal * sizeof(unsigned long long)), "major count zero lc");
    ck(cudaMemset(d_lr, 0, nlocal * sizeof(unsigned long long)), "major count zero lr");
    ck(cudaMemset(d_states, 0, sizeof(unsigned long long)), "major count zero states");
    ck(cudaMemset(d_error, 0, sizeof(int)), "major count zero error");

    const Rank64 base_supports = Rank64(1) << (W - 2);
    const unsigned threads = 256;
    const Rank64 one_pass = (base_supports + threads - 1) / threads;
    const unsigned launch_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass)));
    cudaEvent_t a{}, b{};
    ck(cudaEventCreate(&a), "major count event a");
    ck(cudaEventCreate(&b), "major count event b");
    ck(cudaEventRecord(a), "major count record a");
    segment_major_count_kernel<<<launch_blocks, threads>>>(
        base_supports, W, K, reverse, ngpu, batches,
        d_ns, d_nr, d_lc, d_lr, d_states, d_error);
    ck(cudaGetLastError(), "major count launch");
    ck(cudaEventRecord(b), "major count record b");
    ck(cudaEventSynchronize(b), "major count sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, a, b), "major count elapsed");

    MajorCountHost got;
    got.network_segments.resize(static_cast<std::size_t>(ngroups));
    got.network_runs.resize(static_cast<std::size_t>(ngroups));
    got.local_cycles.resize(static_cast<std::size_t>(nlocal));
    got.local_runs.resize(static_cast<std::size_t>(nlocal));
    int error = 0;
    ck(cudaMemcpy(got.network_segments.data(), d_ns, ngroups * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost), "major count copy ns");
    ck(cudaMemcpy(got.network_runs.data(), d_nr, ngroups * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost), "major count copy nr");
    ck(cudaMemcpy(got.local_cycles.data(), d_lc, nlocal * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost), "major count copy lc");
    ck(cudaMemcpy(got.local_runs.data(), d_lr, nlocal * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost), "major count copy lr");
    ck(cudaMemcpy(&got.network_states, d_states, sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost), "major count copy states");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "major count copy error");
    if (error) fail("segment-major count device error=" + std::to_string(error));
    if (got.network_segments != want.network_segments || got.network_runs != want.network_runs ||
        got.local_cycles != want.local_cycles || got.local_runs != want.local_runs ||
        got.network_states != want.network_states)
        fail("segment-major count builder mismatch");

    unsigned long long total_segments = 0, total_network_runs = 0,
                       total_local_cycles = 0, total_local_runs = 0;
    for (auto z : got.network_segments) total_segments += z;
    for (auto z : got.network_runs) total_network_runs += z;
    for (auto z : got.local_cycles) total_local_cycles += z;
    for (auto z : got.local_runs) total_local_runs += z;
    std::cout << "gridfp-reduced-production-p2p-segment-major-count"
              << " W=" << W << " K=" << K
              << " direction=" << (reverse ? "reverse" : "forward")
              << " ngpu=" << ngpu << " batches=" << batches
              << " network_segments=" << total_segments
              << " network_local_runs=" << total_network_runs
              << " local_cycles=" << total_local_cycles
              << " local_cycle_runs=" << total_local_runs
              << " exact_network_u32_values=" << got.network_states
              << " count_ms=" << ms
              << " owner_only_count_pass=1 grouped_rank_calls=0 exact=OK\n";

    cudaFree(d_error); cudaFree(d_states); cudaFree(d_lr); cudaFree(d_lc);
    cudaFree(d_nr); cudaFree(d_ns); cudaEventDestroy(a); cudaEventDestroy(b);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int K = argc > 2 ? std::atoi(argv[2]) : (W - 2) / 2;
    const int batches = argc > 3 ? std::atoi(argv[3]) : 4;
    const unsigned blocks = argc > 4
        ? static_cast<unsigned>(std::strtoul(argv[4], nullptr, 10)) : 256u;
    const int ngpu = argc > 5 ? std::atoi(argv[5]) : 2;
    if (W < 8 || W > 12 || (W & 1) || K != (W - 2) / 2 ||
        batches < 1 || batches > 64 || !blocks || ngpu < 2 || ngpu > P2P_MAX_GPU)
        return 2;
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "segment-major count device count");
    if (visible < 1) return 3;
    run_segment_major_count_probe(W, K, false, ngpu, batches, blocks);
    run_segment_major_count_probe(W, K, true, ngpu, batches, blocks);
    std::cout << "ALL_OK production_p2p_segment_major_count_builder=1\n";
    return 0;
}
