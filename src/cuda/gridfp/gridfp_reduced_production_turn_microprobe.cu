#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_grouped_inplace_microprobe_main_unused
#include "gridfp_reduced_production_grouped_inplace_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_turn_device.cuh"

namespace {

__device__ __forceinline__ bool turn_small_step(
    DeviceKey src, int W, bool expand, SmallTerms& z
) {
    DeviceTerm tmp[RP_MAX_TERMS]{};
    const int n = expand ? turn_expand_step(src, W, tmp)
                         : turn_compress_step(src, W, tmp);
    if (n < 0) return false;
    for (int i = 0; i < n; ++i)
        if (!small_add(z, tmp[i].key, tmp[i].coef)) return false;
    return true;
}

__device__ __forceinline__ int turn_inverse_direction(
    DeviceKey d, int W, bool expand, DeviceTerm* pre
) {
    return expand ? turn_expand_inverse(d, W, pre)
                  : turn_compress_inverse(d, W, pre);
}

__global__ void turn_compress_inplace_kernel(
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
    const int tile_start = K + 1; // low physical window [0,K+1]

    for (Rank64 cr = first; cr < components; cr += stride) {
        if (lane == 0) {
            sh_ns[warp] = 0;
            sh_nd[warp] = 0;
            const MateID label = motzkin_unrank_device(W - 1, cr);
            const DeviceKey seed = turn_compress_seed(label, W);
            const GroupedDeviceRank sgr = grouped_rank_device(
                seed, W, 1, false, tile_start, K, ngpu, owner_begin);
            sh_owner[warp] = sgr.owner;
            sh_src[warp][0] = seed;
            sh_ns[warp] = 1;
            int cursor = 0;
            while (cursor < sh_ns[warp]) {
                SmallTerms edge;
                if (!turn_small_step(sh_src[warp][cursor++], W, false, edge)) {
                    set_error(error, 161); break;
                }
                for (int ei = 0; ei < edge.n; ++ei) {
                    const DeviceKey d = edge.v[ei].key;
                    if (find_key(sh_dst[warp], sh_nd[warp], d) >= 0) continue;
                    if (sh_nd[warp] >= MAX_PAIRS) { set_error(error, 162); break; }
                    sh_dst[warp][sh_nd[warp]++] = d;
                    DeviceTerm pre[RP_MAX_TERMS]{};
                    const int np = turn_compress_inverse(d, W, pre);
                    if (np < 0) { set_error(error, 163); break; }
                    for (int pi = 0; pi < np; ++pi) {
                        if (!pre[pi].coef) continue;
                        if (find_key(sh_src[warp], sh_ns[warp], pre[pi].key) >= 0) continue;
                        if (sh_ns[warp] >= MAX_PAIRS) { set_error(error, 164); break; }
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
                sh_src[warp][lane], W, 1, false, tile_start, K, ngpu, owner_begin);
            if (r.owner != owner) { set_error(error, 165); sh_value[warp][lane] = 0; }
            else sh_value[warp][lane] = state[shard_base[r.owner] + r.local];
        }
        __syncwarp();

        if (lane < nd) {
            const DeviceKey mine = sh_dst[warp][lane];
            const GroupedDeviceRank dr = grouped_rank_device(
                mine, W, 1, false, tile_start, K, ngpu, owner_begin);
            if (dr.owner != owner) set_error(error, 166);
            long long acc = 0;
            for (int si = 0; si < ns; ++si) {
                SmallTerms edge;
                if (!turn_small_step(sh_src[warp][si], W, false, edge)) {
                    set_error(error, 167); continue;
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

__global__ void turn_expand_inplace_kernel(
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
    const int tile_start = 1; // same low physical window [0,K+1]

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
                    set_error(error, 171); break;
                }
                for (int ei = 0; ei < edge.n; ++ei) {
                    const DeviceKey d = edge.v[ei].key;
                    if (find_key(sh_dst[warp], sh_nd[warp], d) >= 0) continue;
                    if (sh_nd[warp] >= MAX_PAIRS) { set_error(error, 172); break; }
                    sh_dst[warp][sh_nd[warp]++] = d;
                    DeviceTerm pre[RP_MAX_TERMS]{};
                    const int np = turn_expand_inverse(d, W, pre);
                    if (np < 0) { set_error(error, 173); break; }
                    for (int pi = 0; pi < np; ++pi) {
                        if (!pre[pi].coef) continue;
                        if (find_key(sh_src[warp], sh_ns[warp], pre[pi].key) >= 0) continue;
                        if (sh_ns[warp] >= MAX_PAIRS) { set_error(error, 174); break; }
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
            if (r.owner != owner) { set_error(error, 175); sh_value[warp][lane] = 0; }
            else sh_value[warp][lane] = state[shard_base[r.owner] + r.local];
        }
        __syncwarp();

        if (lane < nd) {
            const DeviceKey mine = sh_dst[warp][lane];
            const GroupedDeviceRank dr = grouped_rank_device(
                mine, W, 2, true, tile_start, K, ngpu, owner_begin);
            if (dr.owner != owner) set_error(error, 176);
            long long acc = 0;
            for (int si = 0; si < ns; ++si) {
                SmallTerms edge;
                if (!turn_small_step(sh_src[warp][si], W, true, edge)) {
                    set_error(error, 177); continue;
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

void build_turn_vectors(
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
    const auto q2 = layout(main_words, block_words, 2);

    input.assign(static_cast<std::size_t>(tables.size()), 0);
    expected.assign(static_cast<std::size_t>(tables.size()), 0);
    std::map<Key, std::uint32_t> main_values;

    Rank64 serial = 0;
    for (Key k : q1) {
        const std::uint32_t value = static_cast<std::uint32_t>(
            1 + (serial++ * 2654435761ULL) % (mod - 1ULL));
        const GroupedRank gr = grouped_rank(
            k, tables, W, 1, false, K + 1, K, ngpu, owner_plan);
        input[static_cast<std::size_t>(plan.shard_base[gr.owner] + gr.local)] = value;
        for (const auto& [d,c] : step_basis(k, W, 1, false))
            add_mod_signed(main_values[d], value, int(c), mod);
    }

    for (const auto& [s,value] : main_values) {
        const Vec col = project_vec(step_basis(s, W, 1, true), W, 2, true);
        for (const auto& [d,c] : col) {
            const GroupedRank gr = grouped_rank(
                d, tables, W, 2, true, 1, K, ngpu, owner_plan);
            add_mod_signed(expected[static_cast<std::size_t>(plan.shard_base[gr.owner] + gr.local)],
                           value, int(c), mod);
        }
    }
    if (serial != tables.size()) fail("turn input dimension");
}

void run_turn_probe(int W, int K, int ngpu, unsigned blocks, std::uint32_t mod) {
    ProductionFactorTables tables(W);
    install_tables(tables);
    const HostTilePlan plan = make_host_tile_plan(tables, K, ngpu);
    std::vector<std::uint32_t> input, expected;
    build_turn_vectors(W, K, ngpu, mod, tables, plan, input, expected);

    std::uint32_t* d_state = nullptr;
    Rank64* d_owner_begin = nullptr;
    Rank64* d_shard_base = nullptr;
    unsigned long long* d_written = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_state, tables.size() * sizeof(std::uint32_t)), "turn alloc state");
    ck(cudaMalloc(&d_owner_begin, ngpu * sizeof(Rank64)), "turn alloc owner begin");
    ck(cudaMalloc(&d_shard_base, ngpu * sizeof(Rank64)), "turn alloc shard base");
    ck(cudaMalloc(&d_written, sizeof(unsigned long long)), "turn alloc written");
    ck(cudaMalloc(&d_error, sizeof(int)), "turn alloc error");
    ck(cudaMemcpy(d_state, input.data(), tables.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "turn copy state");
    ck(cudaMemcpy(d_owner_begin, plan.owner_begin.data(), ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "turn copy owner begin");
    ck(cudaMemcpy(d_shard_base, plan.shard_base.data(), ngpu * sizeof(Rank64), cudaMemcpyHostToDevice), "turn copy shard base");
    ck(cudaMemset(d_written, 0, sizeof(unsigned long long)), "turn zero written");
    ck(cudaMemset(d_error, 0, sizeof(int)), "turn zero error");

    const Rank64 compress_components = motzkin_count(W - 1);
    const Rank64 expand_components = compress_components - motzkin_count(W - 3);
    const Rank64 max_components = std::max(compress_components, expand_components);
    const Rank64 one_pass_blocks = (max_components + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned launch_blocks = static_cast<unsigned>(
        std::max<Rank64>(1, std::min<Rank64>(blocks, one_pass_blocks)));

    cudaEvent_t a{}, b{}, c{};
    ck(cudaEventCreate(&a), "turn event a");
    ck(cudaEventCreate(&b), "turn event b");
    ck(cudaEventCreate(&c), "turn event c");
    ck(cudaEventRecord(a), "turn record a");
    turn_compress_inplace_kernel<<<launch_blocks, THREADS>>>(
        d_state, compress_components, W, K, ngpu, d_owner_begin, d_shard_base,
        mod, d_written, d_error);
    ck(cudaGetLastError(), "turn compress launch");
    ck(cudaEventRecord(b), "turn record b");
    turn_expand_inplace_kernel<<<launch_blocks, THREADS>>>(
        d_state, expand_components, W, K, ngpu, d_owner_begin, d_shard_base,
        mod, d_written, d_error);
    ck(cudaGetLastError(), "turn expand launch");
    ck(cudaEventRecord(c), "turn record c");
    ck(cudaEventSynchronize(c), "turn sync");

    float compress_ms = 0, expand_ms = 0;
    ck(cudaEventElapsedTime(&compress_ms, a, b), "turn compress elapsed");
    ck(cudaEventElapsedTime(&expand_ms, b, c), "turn expand elapsed");
    int error = 0;
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "turn copy error");
    if (error) fail("turn CUDA device error=" + std::to_string(error));
    std::vector<std::uint32_t> output(expected.size());
    ck(cudaMemcpy(output.data(), d_state, output.size() * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "turn copy output");
    if (output != expected) fail("turn CUDA exact mismatch");

    std::cout << "gridfp-reduced-production-turn-microprobe"
              << " W=" << W << " K=" << K
              << " states=" << tables.size()
              << " compress_components=" << compress_components
              << " expand_components=" << expand_components
              << " compress_ms=" << compress_ms
              << " expand_ms=" << expand_ms
              << " single_state_stream=1"
              << " component_table_bytes=0 inverse_table_bytes=0"
              << " second_state_buffer_bytes=0 exact=OK\n";

    cudaEventDestroy(a); cudaEventDestroy(b); cudaEventDestroy(c);
    cudaFree(d_error); cudaFree(d_written); cudaFree(d_shard_base);
    cudaFree(d_owner_begin); cudaFree(d_state);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const int K = argc > 2 ? std::atoi(argv[2]) : (W - 2) / 2;
    const unsigned blocks = argc > 3 ? static_cast<unsigned>(std::strtoul(argv[3], nullptr, 10)) : 256u;
    const int ngpu = argc > 4 ? std::atoi(argv[4]) : 8;
    const std::uint32_t mod = argc > 5 ? static_cast<std::uint32_t>(std::strtoul(argv[5], nullptr, 10)) : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > RP_MAX_W || K < 2 || K + 2 > W || !blocks || ngpu != 8 || mod < 3) return 2;

    ProductionFactorTables tables(W);
    if (plan_only) {
        std::cout << "gridfp-reduced-production-turn-plan"
                  << " W=" << W << " K=" << K
                  << " states=" << tables.size()
                  << " compress_components=" << motzkin_count(W - 1)
                  << " expand_components=" << (motzkin_count(W - 1) - motzkin_count(W - 3))
                  << " max_pairs_candidate=" << (W / 2 + 4)
                  << " single_state_stream=1 second_state_buffer_bytes=0"
                  << " component_table_bytes=0 inverse_table_bytes=0\n";
        return 0;
    }
    if (W > 12) {
        std::cerr << "execution mode intentionally limited to W<=12; use --plan-only for production width\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "turn device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "turn set device");
    run_turn_probe(W, K, ngpu, blocks, mod);
    std::cout << "ALL_OK gridfp_reduced_production_turn_cuda=1\n";
    return 0;
}
