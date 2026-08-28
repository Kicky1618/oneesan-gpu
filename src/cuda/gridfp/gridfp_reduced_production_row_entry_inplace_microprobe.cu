#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_component_dense_microprobe_main_unused
#include "gridfp_reduced_production_component_dense_microprobe.cu"
#pragma pop_macro("main")

namespace {

__device__ __forceinline__ DeviceKey row_entry_seed(
    MateID label, int W, int p, bool reverse
) {
    const MateID m = reverse ? blocked_exclude_reverse(label, W, p)
                             : blocked_exclude(label, p);
    return DeviceKey{m, 0};
}

__global__ void row_entry_inplace_kernel(
    std::uint32_t* state,
    unsigned long long* owner,
    Rank64 components,
    Rank64 state_count,
    int W,
    int p,
    bool reverse,
    std::uint32_t mod,
    unsigned long long* processed,
    int* error
) {
    __shared__ DeviceKey sh_src[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ DeviceKey sh_dst[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ std::uint32_t sh_value[WARPS_PER_BLOCK][MAX_PAIRS];
    __shared__ int sh_ns[WARPS_PER_BLOCK];
    __shared__ int sh_nd[WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank64 first = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);
    const Rank64 stride = Rank64(gridDim.x) * WARPS_PER_BLOCK;
    const int dst_fixed = reverse ? p : p - 2;

    for (Rank64 component_rank = first; component_rank < components; component_rank += stride) {
        if (lane == 0) {
            sh_ns[warp] = 0;
            sh_nd[warp] = 0;
            const MateID label = component_label_unrank_device(W, p, reverse, component_rank);
            const DeviceKey seed = row_entry_seed(label, W, p, reverse);
            if (!valid_mate_device(seed.mate, W)) {
                set_error(error, 111);
            } else {
                sh_src[warp][0] = seed;
                sh_ns[warp] = 1;
                int cursor = 0;
                while (cursor < sh_ns[warp]) {
                    SmallTerms edge;
                    if (!small_step(sh_src[warp][cursor++], W, p, reverse, edge)) {
                        set_error(error, 112);
                        break;
                    }
                    for (int ei = 0; ei < edge.n; ++ei) {
                        if (!edge.v[ei].coef) continue;
                        const DeviceKey d = edge.v[ei].key;
                        if (find_key(sh_dst[warp], sh_nd[warp], d) < 0) {
                            if (sh_nd[warp] >= MAX_PAIRS) {
                                set_error(error, 113);
                                break;
                            }
                            sh_dst[warp][sh_nd[warp]++] = d;
                        }
                        DeviceTerm pre[RP_MAX_TERMS]{};
                        const int np = inverse_direction(d, W, p, reverse, pre);
                        if (np < 0) {
                            set_error(error, 114);
                            break;
                        }
                        for (int pi = 0; pi < np; ++pi) {
                            if (!pre[pi].coef || pre[pi].key.blocked) continue;
                            if (find_key(sh_src[warp], sh_ns[warp], pre[pi].key) >= 0) continue;
                            if (sh_ns[warp] >= MAX_PAIRS) {
                                set_error(error, 115);
                                break;
                            }
                            sh_src[warp][sh_ns[warp]++] = pre[pi].key;
                        }
                    }
                    if (*error) break;
                }
                if (!(sh_nd[warp] == sh_ns[warp] || sh_nd[warp] == sh_ns[warp] + 1))
                    set_error(error, 116);
                if (sh_ns[warp] > W / 2 + 3 || sh_nd[warp] > W / 2 + 4)
                    set_error(error, 117);
                atomicAdd(processed, 1ULL);
            }
        }
        __syncwarp();

        const int ns = sh_ns[warp];
        const int nd = sh_nd[warp];
        if (lane < ns) {
            const DeviceKey k = sh_src[warp][lane];
            if (k.blocked) {
                set_error(error, 118);
                sh_value[warp][lane] = 0;
            } else {
                const Rank64 r = factor_rank_device(k, W, dst_fixed);
                if (r >= state_count) {
                    set_error(error, 119);
                    sh_value[warp][lane] = 0;
                } else {
                    sh_value[warp][lane] = state[r];
                }
            }
        }
        __syncwarp();

        if (lane < nd) {
            const DeviceKey mine = sh_dst[warp][lane];
            long long acc = 0;
            for (int si = 0; si < ns; ++si) {
                SmallTerms edge;
                if (!small_step(sh_src[warp][si], W, p, reverse, edge)) {
                    set_error(error, 120);
                    continue;
                }
                for (int ei = 0; ei < edge.n; ++ei) {
                    if (key_equal(edge.v[ei].key, mine)) {
                        acc += static_cast<long long>(edge.v[ei].coef) *
                               static_cast<long long>(sh_value[warp][si]);
                    }
                }
            }
            long long z = acc % static_cast<long long>(mod);
            if (z < 0) z += mod;
            const Rank64 dr = factor_rank_device(mine, W, dst_fixed);
            if (dr >= state_count) {
                set_error(error, 121);
            } else {
                const unsigned long long empty = ~0ULL;
                const unsigned long long prev = atomicCAS(
                    owner + dr, empty, static_cast<unsigned long long>(component_rank));
                if (prev != empty && prev != static_cast<unsigned long long>(component_rank))
                    set_error(error, 122);
                state[dr] = static_cast<std::uint32_t>(z);
            }
        }
        __syncwarp();
    }
}

void run_row_entry_inplace(
    int W,
    bool reverse,
    const ProductionFactorTables& tables,
    std::uint32_t mod,
    unsigned requested_blocks
) {
    const int p = reverse ? 1 : W - 1;
    const int dst_fixed = reverse ? p : p - 2;
    ProductionFactorCodec codec(tables, dst_fixed);
    const Rank64 states = tables.size();
    const Rank64 components = motzkin_count(W - 1) - motzkin_count(W - 3);

    std::vector<std::uint32_t> initial(static_cast<std::size_t>(states), 0);
    std::vector<std::uint32_t> reference(static_cast<std::size_t>(states), 0);
    const auto main_words = gen_words(W);
    for (Rank64 q = 0; q < main_words.size(); ++q) {
        const Key src{false, main_words[static_cast<std::size_t>(q)]};
        const Rank64 sr = codec.rank(src);
        const std::uint32_t value = static_cast<std::uint32_t>(
            (1 + (q * 2654435761ULL) % (mod - 1ULL)) % mod);
        initial[static_cast<std::size_t>(sr)] = value;
        for (const auto& [d, c] : reduced_step_basis(src, W, p, reverse))
            add_mod_signed(reference[static_cast<std::size_t>(codec.rank(d))], value, int(c), mod);
    }

    std::uint32_t* d_state = nullptr;
    unsigned long long* d_owner = nullptr;
    unsigned long long* d_processed = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_state, states * sizeof(std::uint32_t)), "entry inplace alloc state");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "entry inplace alloc owner");
    ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "entry inplace alloc processed");
    ck(cudaMalloc(&d_error, sizeof(int)), "entry inplace alloc error");
    ck(cudaMemcpy(d_state, initial.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "entry inplace copy state");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "entry inplace clear owner");
    ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "entry inplace zero processed");
    ck(cudaMemset(d_error, 0, sizeof(int)), "entry inplace zero error");

    const Rank64 one_pass_blocks = (components + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned blocks = static_cast<unsigned>(std::max<Rank64>(
        1, std::min<Rank64>(requested_blocks, one_pass_blocks)));
    cudaEvent_t a{}, b{};
    ck(cudaEventCreate(&a), "entry inplace event a");
    ck(cudaEventCreate(&b), "entry inplace event b");
    ck(cudaEventRecord(a), "entry inplace record a");
    row_entry_inplace_kernel<<<blocks, THREADS>>>(
        d_state, d_owner, components, states, W, p, reverse, mod, d_processed, d_error);
    ck(cudaGetLastError(), "entry inplace launch");
    ck(cudaEventRecord(b), "entry inplace record b");
    ck(cudaEventSynchronize(b), "entry inplace sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, a, b), "entry inplace elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    std::vector<unsigned long long> owner(static_cast<std::size_t>(states));
    unsigned long long processed = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_state, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "entry inplace copy output");
    ck(cudaMemcpy(owner.data(), d_owner, states * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "entry inplace copy owner");
    ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost), "entry inplace copy processed");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "entry inplace copy error");
    if (error || processed != components) {
        std::cerr << "FAIL row entry inplace error=" << error
                  << " processed=" << processed << " want=" << components << '\n';
        std::exit(121);
    }
    for (Rank64 r = 0; r < states; ++r) {
        if (owner[static_cast<std::size_t>(r)] == std::numeric_limits<unsigned long long>::max() ||
            output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL row entry inplace arithmetic W=" << W
                      << " direction=" << (reverse ? "reverse" : "forward")
                      << " rank=" << r << '\n';
            std::exit(122);
        }
    }

    std::cout << "gridfp-reduced-row-entry-inplace-microprobe"
              << " W=" << W
              << " direction=" << (reverse ? "reverse" : "forward")
              << " states=" << states
              << " components=" << components
              << " blocks=" << blocks
              << " kernel_ms=" << ms
              << " state_buffers=1"
              << " blocked_slots_reused=1"
              << " arithmetic=OK\n";

    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaFree(d_state);
    cudaFree(d_owner);
    cudaFree(d_processed);
    cudaFree(d_error);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const unsigned blocks = argc > 2 ? static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10)) : 4096u;
    const std::uint32_t mod = argc > 3 ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 5 || W > RP_MAX_W || blocks == 0 || mod < 3) return 2;

    ProductionFactorTables tables(W);
    if (plan_only) {
        std::cout << "gridfp-reduced-row-entry-inplace-plan"
                  << " W=" << W
                  << " states=" << tables.size()
                  << " components=" << (motzkin_count(W - 1) - motzkin_count(W - 3))
                  << " max_source_slots_bound=" << (W / 2 + 3)
                  << " max_destination_slots_bound=" << (W / 2 + 4)
                  << " state_buffers=1"
                  << " full_stream_scratch_bytes=0\n";
        return 0;
    }
    if (W > 12) {
        std::cerr << "execution mode is intentionally limited to W<=12; use --plan-only for production widths\n";
        return 3;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "entry inplace device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "entry inplace set device");
    install_tables(tables);
    run_row_entry_inplace(W, false, tables, mod, blocks);
    run_row_entry_inplace(W, true, tables, mod, blocks);
    std::cout << "ALL_OK gridfp_reduced_production_cuda_row_entry_inplace=1 W=" << W << '\n';
    return 0;
}
