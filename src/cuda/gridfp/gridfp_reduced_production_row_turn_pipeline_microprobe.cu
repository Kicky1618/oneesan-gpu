#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_turn_microprobe_main_unused
#include "gridfp_reduced_production_turn_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_row_turn_main_shift_device.cuh"

namespace {

__global__ void turn_main_shift_inplace_kernel(
    std::uint32_t* __restrict__ state,
    Rank64 support_runs,
    int W,
    int K,
    int ngpu,
    int gpu_id,
    const Rank64* __restrict__ owner_begin,
    const Rank64* __restrict__ shard_base,
    unsigned long long* __restrict__ cycles,
    unsigned long long* __restrict__ rotated_values,
    int* error
) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    const int old_start = K + 1; // B=[0,K+1]

    for (Rank64 rr = first; rr < support_runs; rr += stride) {
        const std::uint32_t support = turn_main_support_from_rank_device(rr, W);
        const int cycle_len = turn_main_shift_leader_length_device(support, W, K);
        if (cycle_len < 0) {
            if (lane == 0) set_error(error, 181);
            continue;
        }
        if (cycle_len <= 1) continue;

        const DeviceKey leader = turn_main_run_key0_device(support, W);
        const GroupedDeviceRank lr = grouped_rank_device(
            leader, W, 1, false, old_start, K, ngpu, owner_begin);
        if (lr.owner != gpu_id) continue;

        const int occupied = __popc(support);
        const Rank64 pc = RP_PRIMITIVE[occupied][1];
        const Rank64 leader_pos = shard_base[lr.owner] + lr.local;
        for (Rank64 i = Rank64(lane); i < pc; i += 32) {
            std::uint32_t temp = state[leader_pos + i];
            std::uint32_t cur = turn_main_shift_next_support_device(support, W, K);
            int hops = 1;
            while (cur != support) {
                const DeviceKey ck = turn_main_run_key0_device(cur, W);
                const GroupedDeviceRank cr = grouped_rank_device(
                    ck, W, 1, false, old_start, K, ngpu, owner_begin);
                const Rank64 pos = shard_base[cr.owner] + cr.local + i;
                const std::uint32_t next = state[pos];
                state[pos] = temp;
                temp = next;
                cur = turn_main_shift_next_support_device(cur, W, K);
                ++hops;
                if (hops > cycle_len) {
                    set_error(error, 182);
                    break;
                }
            }
            state[leader_pos + i] = temp;
        }
        __syncwarp();
        if (lane == 0) {
            atomicAdd(cycles, 1ULL);
            atomicAdd(rotated_values, static_cast<unsigned long long>(pc) * cycle_len);
        }
    }
}

// Same table-free expansion as turn_microprobe, but the source and destination
// use the first reverse tile A=[1,K+2] instead of the temporary low-edge B
// layout.  The main-only cycle above converts B -> A before this kernel runs.
__global__ void turn_expand_A_inplace_kernel(
    std::uint32_t* __restrict__ state,
    Rank64 components,
    int W,
    int K,
    int ngpu,
    const Rank64* __restrict__ owner_begin,
    const Rank64* __restrict__ shard_base,
    std::uint32_t mod,
    unsigned long long* __restrict__ written,
    int* error
) {
    __shared__ DeviceKey sh_src[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ DeviceKey sh_dst[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ std::uint32_t sh_value[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ int sh_ns[WARPS_PER_BLOCK];
    __shared__ int sh_nd[WARPS_PER_BLOCK];
    __shared__ int sh_owner[WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    const int tile_start = 2; // A=[1,K+2]

    for (Rank64 cr = first; cr < components; cr += stride) {
        if (lane == 0) {
            sh_ns[warp] = 0;
            sh_nd[warp] = 0;
            const MateID label = component_label_unrank_device(W, 1, true, cr);
            const DeviceKey seed = turn_expand_seed(label, W);
            const GroupedDeviceRank sgr = grouped_rank_device(
                seed, W, 1, true, tile_start, K, ngpu, owner_begin);
            sh_owner[warp] = sgr.owner;
            sh_src[warp][0] = seed;
            sh_ns[warp] = 1;
            int cursor = 0;
            while (cursor < sh_ns[warp]) {
                SmallTerms edge;
                if (!turn_small_step(sh_src[warp][cursor++], W, true, edge)) {
                    set_error(error, 183); break;
                }
                for (int ei = 0; ei < edge.n; ++ei) {
                    const DeviceKey d = edge.v[ei].key;
                    if (find_key(sh_dst[warp], sh_nd[warp], d) >= 0) continue;
                    if (sh_nd[warp] >= MAX_PAIRS) { set_error(error, 184); break; }
                    sh_dst[warp][sh_nd[warp]++] = d;
                    DeviceTerm pre[RP_MAX_TERMS]{};
                    const int np = turn_expand_inverse(d, W, pre);
                    if (np < 0) { set_error(error, 185); break; }
                    for (int pi = 0; pi < np; ++pi) {
                        if (!pre[pi].coef || pre[pi].key.blocked) continue;
                        if (find_key(sh_src[warp], sh_ns[warp], pre[pi].key) >= 0) continue;
                        if (sh_ns[warp] >= MAX_PAIRS) { set_error(error, 186); break; }
                        sh_src[warp][sh_ns[warp]++] = pre[pi].key;
                    }
                }
                if (*error) break;
            }
        }
        __syncwarp();

        const int ns = sh_ns[warp], nd = sh_nd[warp], owner = sh_owner[warp];
        if (lane < ns) {
            const GroupedDeviceRank r = grouped_rank_device(
                sh_src[warp][lane], W, 1, true, tile_start, K, ngpu, owner_begin);
            if (r.owner != owner) { set_error(error, 187); sh_value[warp][lane] = 0; }
            else sh_value[warp][lane] = state[shard_base[r.owner] + r.local];
        }
        __syncwarp();

        if (lane < nd) {
            const DeviceKey mine = sh_dst[warp][lane];
            const GroupedDeviceRank dr = grouped_rank_device(
                mine, W, 2, true, tile_start, K, ngpu, owner_begin);
            if (dr.owner != owner) set_error(error, 188);
            long long acc = 0;
            for (int si = 0; si < ns; ++si) {
                SmallTerms edge;
                if (!turn_small_step(sh_src[warp][si], W, true, edge)) {
                    set_error(error, 189); continue;
                }
                for (int ei = 0; ei < edge.n; ++ei)
                    if (key_equal(edge.v[ei].key, mine))
                        acc += static_cast<long long>(edge.v[ei].coef) * sh_value[warp][si];
            }
            long long z = acc % static_cast<long long>(mod);
            if (z < 0) z += mod;
            state[shard_base[dr.owner] + dr.local] = static_cast<std::uint32_t>(z);
            if (lane == 0) atomicAdd(written, static_cast<unsigned long long>(nd));
        }
        __syncwarp();
    }
}

void build_turn_pipeline_vectors(
    int W,
    int K,
    int ngpu,
    std::uint32_t mod,
    const ProductionFactorTables& tables,
    const HostTilePlan& plan,
    std::vector<std::uint32_t>& input,
    std::vector<std::uint32_t>& expected
) {
    const OwnerPlan owner_plan{plan.owner_begin, plan.owner_size};
    const auto main_words = gen_words(W);
    const auto block_words = gen_words(W - 1);
    const auto q1 = layout(main_words, block_words, 1);

    input.assign(static_cast<std::size_t>(tables.size()), 0);
    expected.assign(static_cast<std::size_t>(tables.size()), 0);
    std::map<Key, std::uint32_t> main_values;
    Rank64 serial = 0;
    for (Key k : q1) {
        const std::uint32_t value = static_cast<std::uint32_t>(
            1 + (serial++ * 2654435761ULL) % (mod - 1ULL));
        const GroupedRank sr = grouped_rank(
            k, tables, W, 1, false, K + 1, K, ngpu, owner_plan);
        input[static_cast<std::size_t>(plan.shard_base[sr.owner] + sr.local)] = value;
        for (const auto& [d, c] : step_basis(k, W, 1, false))
            add_mod_signed(main_values[d], value, int(c), mod);
    }
    if (serial != tables.size()) fail("turn pipeline Q1 dimension");

    for (const auto& [s, value] : main_values) {
        const Vec col = project_vec(step_basis(s, W, 1, true), W, 2, true);
        for (const auto& [d, c] : col) {
            const GroupedRank dr = grouped_rank(
                d, tables, W, 2, true, 2, K, ngpu, owner_plan);
            add_mod_signed(expected[static_cast<std::size_t>(plan.shard_base[dr.owner] + dr.local)],
                           value, int(c), mod);
        }
    }
}

void run_turn_pipeline(int W, int K, int ngpu, unsigned blocks, std::uint32_t mod) {
    ProductionFactorTables tables(W);
    install_tables(tables);
    const HostTilePlan plan = make_host_tile_plan(tables, K, ngpu);
    std::vector<std::uint32_t> input, expected;
    build_turn_pipeline_vectors(W, K, ngpu, mod, tables, plan, input, expected);

    std::uint32_t* d_state = nullptr;
    Rank64* d_owner_begin = nullptr;
    Rank64* d_shard_base = nullptr;
    unsigned long long *d_compress_written = nullptr, *d_expand_written = nullptr;
    unsigned long long *d_cycles = nullptr, *d_rotated = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_state, tables.size() * sizeof(std::uint32_t)), "turnpipe alloc state");
    ck(cudaMalloc(&d_owner_begin, ngpu * sizeof(Rank64)), "turnpipe alloc owner begin");
    ck(cudaMalloc(&d_shard_base, ngpu * sizeof(Rank64)), "turnpipe alloc shard base");
    ck(cudaMalloc(&d_compress_written, sizeof(unsigned long long)), "turnpipe alloc compress count");
    ck(cudaMalloc(&d_expand_written, sizeof(unsigned long long)), "turnpipe alloc expand count");
    ck(cudaMalloc(&d_cycles, sizeof(unsigned long long)), "turnpipe alloc cycles");
    ck(cudaMalloc(&d_rotated, sizeof(unsigned long long)), "turnpipe alloc rotated");
    ck(cudaMalloc(&d_error, sizeof(int)), "turnpipe alloc error");
    ck(cudaMemcpy(d_state, input.data(), tables.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "turnpipe copy state");
    ck(cudaMemcpy(d_owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "turnpipe copy owner begin");
    ck(cudaMemcpy(d_shard_base, plan.shard_base.data(), ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "turnpipe copy shard base");
    ck(cudaMemset(d_compress_written, 0, sizeof(unsigned long long)), "turnpipe zero compress");
    ck(cudaMemset(d_expand_written, 0, sizeof(unsigned long long)), "turnpipe zero expand");
    ck(cudaMemset(d_cycles, 0, sizeof(unsigned long long)), "turnpipe zero cycles");
    ck(cudaMemset(d_rotated, 0, sizeof(unsigned long long)), "turnpipe zero rotated");
    ck(cudaMemset(d_error, 0, sizeof(int)), "turnpipe zero error");

    const Rank64 compress_components = motzkin_count(W - 1);
    const Rank64 expand_components = compress_components - motzkin_count(W - 3);
    const Rank64 support_runs = Rank64(1) << (W - 1);
    const Rank64 work = std::max({compress_components, expand_components, support_runs});
    const Rank64 one_pass_blocks = (work + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned launch_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass_blocks)));

    cudaEvent_t a{}, b{}, c{}, d{};
    ck(cudaEventCreate(&a), "turnpipe event a");
    ck(cudaEventCreate(&b), "turnpipe event b");
    ck(cudaEventCreate(&c), "turnpipe event c");
    ck(cudaEventCreate(&d), "turnpipe event d");
    ck(cudaEventRecord(a), "turnpipe record a");
    turn_compress_inplace_kernel<<<launch_blocks, THREADS>>>(
        d_state, compress_components, W, K, ngpu, d_owner_begin, d_shard_base,
        mod, d_compress_written, d_error);
    ck(cudaGetLastError(), "turnpipe compress launch");
    ck(cudaEventRecord(b), "turnpipe record b");
    for (int g = 0; g < ngpu; ++g) {
        turn_main_shift_inplace_kernel<<<launch_blocks, THREADS>>>(
            d_state, support_runs, W, K, ngpu, g, d_owner_begin, d_shard_base,
            d_cycles, d_rotated, d_error);
        ck(cudaGetLastError(), "turnpipe main shift launch");
    }
    ck(cudaEventRecord(c), "turnpipe record c");
    turn_expand_A_inplace_kernel<<<launch_blocks, THREADS>>>(
        d_state, expand_components, W, K, ngpu, d_owner_begin, d_shard_base,
        mod, d_expand_written, d_error);
    ck(cudaGetLastError(), "turnpipe expand launch");
    ck(cudaEventRecord(d), "turnpipe record d");
    ck(cudaEventSynchronize(d), "turnpipe sync");

    int error = 0;
    unsigned long long cw = 0, ew = 0, cycles = 0, rotated = 0;
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "turnpipe copy error");
    ck(cudaMemcpy(&cw, d_compress_written, sizeof(cw), cudaMemcpyDeviceToHost), "turnpipe copy compress");
    ck(cudaMemcpy(&ew, d_expand_written, sizeof(ew), cudaMemcpyDeviceToHost), "turnpipe copy expand");
    ck(cudaMemcpy(&cycles, d_cycles, sizeof(cycles), cudaMemcpyDeviceToHost), "turnpipe copy cycles");
    ck(cudaMemcpy(&rotated, d_rotated, sizeof(rotated), cudaMemcpyDeviceToHost), "turnpipe copy rotated");
    if (error) fail("turn pipeline CUDA device error=" + std::to_string(error));

    std::vector<std::uint32_t> output(static_cast<std::size_t>(tables.size()));
    ck(cudaMemcpy(output.data(), d_state, tables.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "turnpipe copy output");
    if (output != expected) fail("turn pipeline CUDA mismatch");

    float compress_ms = 0, shift_ms = 0, expand_ms = 0;
    ck(cudaEventElapsedTime(&compress_ms, a, b), "turnpipe elapsed compress");
    ck(cudaEventElapsedTime(&shift_ms, b, c), "turnpipe elapsed shift");
    ck(cudaEventElapsedTime(&expand_ms, c, d), "turnpipe elapsed expand");
    std::cout << "W=" << W
              << " K=" << K
              << " states=" << tables.size()
              << " compress_components=" << compress_components
              << " expand_components=" << expand_components
              << " main_support_runs=" << support_runs
              << " compress_written=" << cw
              << " expand_written=" << ew
              << " main_cycles=" << cycles
              << " rotated_values=" << rotated
              << " compress_ms=" << compress_ms
              << " main_shift_ms=" << shift_ms
              << " expand_ms=" << expand_ms
              << " state_buffers=1 second_state_buffer_bytes=0"
              << " component_table_bytes=0 run_table_bytes=0 visited_bytes=0"
              << " exact=OK\n";

    cudaEventDestroy(d); cudaEventDestroy(c); cudaEventDestroy(b); cudaEventDestroy(a);
    cudaFree(d_error); cudaFree(d_rotated); cudaFree(d_cycles);
    cudaFree(d_expand_written); cudaFree(d_compress_written);
    cudaFree(d_shard_base); cudaFree(d_owner_begin); cudaFree(d_state);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 9;
    const int K = argc > 2 ? std::atoi(argv[2]) : 3;
    const unsigned blocks = argc > 3 ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10)) : 256u;
    const int ngpu = argc > 4 ? std::atoi(argv[4]) : 8;
    const std::uint32_t mod = argc > 5 ? static_cast<std::uint32_t>(std::strtoul(argv[5], nullptr, 10)) : 4294967291u;
    if (W < 6 || W > 11 || K < 2 || K + 3 > W || blocks == 0 || ngpu != 8 || mod < 3) return 2;

    run_turn_pipeline(W, K, ngpu, blocks, mod);
    std::cout << "ALL_OK production_row_turn_single_stream_cuda=1\n";
    return 0;
}
