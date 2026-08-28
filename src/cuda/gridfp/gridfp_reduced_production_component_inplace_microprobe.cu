#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_component_dense_microprobe_main_unused
#include "gridfp_reduced_production_component_dense_microprobe.cu"
#pragma pop_macro("main")

#include <iomanip>

namespace {

template<bool INPLACE>
__global__ void component_ab_kernel(
    const std::uint32_t* input,
    std::uint32_t* output,
    Rank64 components,
    Rank64 state_count,
    int W,
    int p,
    bool reverse,
    std::uint32_t mod,
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
    const int src_fixed = p - 1;
    const int dst_fixed = reverse ? p : p - 2;

    for (Rank64 component_rank = first; component_rank < components; component_rank += stride) {
        if (lane == 0) {
            sh_ns[warp] = 0;
            sh_nd[warp] = 0;
            const MateID label = component_label_unrank_device(W, p, reverse, component_rank);
            bool eligible = false;
            const DeviceKey seed = component_seed_direction(label, W, p, reverse, eligible);
            if (!eligible) {
                set_error(error, INPLACE ? 101 : 91);
            } else {
                sh_src[warp][0] = seed;
                sh_ns[warp] = 1;
                int cursor = 0;
                while (cursor < sh_ns[warp]) {
                    SmallTerms edge;
                    if (!small_step(sh_src[warp][cursor++], W, p, reverse, edge)) {
                        set_error(error, INPLACE ? 102 : 92);
                        break;
                    }
                    for (int ei = 0; ei < edge.n; ++ei) {
                        if (!edge.v[ei].coef) continue;
                        const DeviceKey d = edge.v[ei].key;
                        if (find_key(sh_dst[warp], sh_nd[warp], d) >= 0) continue;
                        if (sh_nd[warp] >= MAX_PAIRS) {
                            set_error(error, INPLACE ? 103 : 93);
                            break;
                        }
                        sh_dst[warp][sh_nd[warp]++] = d;
                        DeviceTerm pre[RP_MAX_TERMS]{};
                        const int np = inverse_direction(d, W, p, reverse, pre);
                        if (np < 0) {
                            set_error(error, INPLACE ? 104 : 94);
                            break;
                        }
                        for (int pi = 0; pi < np; ++pi) {
                            if (!pre[pi].coef) continue;
                            if (find_key(sh_src[warp], sh_ns[warp], pre[pi].key) >= 0) continue;
                            if (sh_ns[warp] >= MAX_PAIRS) {
                                set_error(error, INPLACE ? 105 : 95);
                                break;
                            }
                            sh_src[warp][sh_ns[warp]++] = pre[pi].key;
                        }
                    }
                }
                if (sh_ns[warp] != sh_nd[warp]) set_error(error, INPLACE ? 106 : 96);
            }
        }
        __syncwarp();

        const int ns = sh_ns[warp];
        const int nd = sh_nd[warp];

        // This barrier is the key to safe in-place execution: every source
        // value in the component is copied to warp-local storage before any
        // destination rank is overwritten. Factorized component alignment
        // guarantees that no other component owns any of these numeric ranks.
        if (lane < ns) {
            const Rank64 r = factor_rank_device(sh_src[warp][lane], W, src_fixed);
            if (r >= state_count) {
                set_error(error, INPLACE ? 107 : 97);
                sh_value[warp][lane] = 0;
            } else {
                sh_value[warp][lane] = input[r];
            }
        }
        __syncwarp();

        if (lane < nd) {
            const DeviceKey mine = sh_dst[warp][lane];
            long long acc = 0;
            for (int si = 0; si < ns; ++si) {
                SmallTerms edge;
                if (!small_step(sh_src[warp][si], W, p, reverse, edge)) {
                    set_error(error, INPLACE ? 108 : 98);
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
                set_error(error, INPLACE ? 109 : 99);
            } else {
                output[dr] = static_cast<std::uint32_t>(z);
            }
        }
        __syncwarp();
    }
}

float launch_timed(
    bool inplace,
    const std::uint32_t* input,
    std::uint32_t* output,
    Rank64 components,
    Rank64 states,
    int W,
    int p,
    bool reverse,
    std::uint32_t mod,
    unsigned blocks,
    int* error
) {
    cudaEvent_t a{}, b{};
    ck(cudaEventCreate(&a), "inplace ab event a");
    ck(cudaEventCreate(&b), "inplace ab event b");
    ck(cudaEventRecord(a), "inplace ab record a");
    if (inplace) {
        component_ab_kernel<true><<<blocks, THREADS>>>(
            input, output, components, states, W, p, reverse, mod, error);
    } else {
        component_ab_kernel<false><<<blocks, THREADS>>>(
            input, output, components, states, W, p, reverse, mod, error);
    }
    ck(cudaGetLastError(), "inplace ab launch");
    ck(cudaEventRecord(b), "inplace ab record b");
    ck(cudaEventSynchronize(b), "inplace ab sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, a, b), "inplace ab elapsed");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms;
}

void verify_result(
    const std::vector<std::uint32_t>& got,
    const std::vector<std::uint32_t>& reference,
    const char* mode,
    int W,
    int p,
    bool reverse
) {
    if (got.size() != reference.size()) fail("inplace ab result size");
    for (std::size_t r = 0; r < got.size(); ++r) {
        if (got[r] != reference[r]) {
            std::cerr << "FAIL component " << mode
                      << " W=" << W << " p=" << p
                      << " reverse=" << reverse
                      << " rank=" << r
                      << " got=" << got[r]
                      << " want=" << reference[r] << '\n';
            std::exit(117);
        }
    }
}

void run_inplace_ab_position(
    int W,
    int p,
    bool reverse,
    const ProductionFactorTables& tables,
    std::uint32_t mod,
    unsigned requested_blocks
) {
    const int src_fixed = p - 1;
    const int dst_fixed = reverse ? p : p - 2;
    ProductionFactorCodec src(tables, src_fixed);
    ProductionFactorCodec dst(tables, dst_fixed);
    const Rank64 states = tables.size();
    const Rank64 components = motzkin_count(W - 1) - motzkin_count(W - 3);

    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    std::vector<std::uint32_t> reference(static_cast<std::size_t>(states));
    for (Rank64 r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            (1 + (r * 2654435761ULL) % (mod - 1ULL)) % mod);
    for (Rank64 s = 0; s < states; ++s) {
        const Key k = src.unrank(s);
        const std::uint32_t v = input[static_cast<std::size_t>(s)];
        for (const auto& [d, c] : reduced_step_basis(k, W, p, reverse))
            add_mod_signed(reference[static_cast<std::size_t>(dst.rank(d))], v, int(c), mod);
    }

    std::uint32_t* d_input = nullptr;
    std::uint32_t* d_output = nullptr;
    std::uint32_t* d_state = nullptr;
    int* d_error = nullptr;
    const std::size_t bytes = static_cast<std::size_t>(states) * sizeof(std::uint32_t);
    ck(cudaMalloc(&d_input, bytes), "inplace ab alloc input");
    ck(cudaMalloc(&d_output, bytes), "inplace ab alloc output");
    ck(cudaMalloc(&d_state, bytes), "inplace ab alloc state");
    ck(cudaMalloc(&d_error, sizeof(int)), "inplace ab alloc error");
    ck(cudaMemcpy(d_input, input.data(), bytes, cudaMemcpyHostToDevice), "inplace ab copy input");
    ck(cudaMemcpy(d_state, input.data(), bytes, cudaMemcpyHostToDevice), "inplace ab copy state");
    ck(cudaMemset(d_output, 0, bytes), "inplace ab zero output");
    ck(cudaMemset(d_error, 0, sizeof(int)), "inplace ab zero error");

    const Rank64 one_pass_blocks = (components + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned blocks = static_cast<unsigned>(std::max<Rank64>(
        1, std::min<Rank64>(requested_blocks, one_pass_blocks)));

    const float twobuffer_ms = launch_timed(
        false, d_input, d_output, components, states, W, p, reverse, mod, blocks, d_error);
    int error = 0;
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "inplace ab copy two error");
    if (error) {
        std::cerr << "FAIL two-buffer kernel error=" << error << '\n';
        std::exit(115);
    }

    ck(cudaMemset(d_error, 0, sizeof(int)), "inplace ab reset error");
    const float inplace_ms = launch_timed(
        true, d_state, d_state, components, states, W, p, reverse, mod, blocks, d_error);
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "inplace ab copy in error");
    if (error) {
        std::cerr << "FAIL in-place kernel error=" << error << '\n';
        std::exit(116);
    }

    std::vector<std::uint32_t> two(static_cast<std::size_t>(states));
    std::vector<std::uint32_t> one(static_cast<std::size_t>(states));
    ck(cudaMemcpy(two.data(), d_output, bytes, cudaMemcpyDeviceToHost), "inplace ab copy two output");
    ck(cudaMemcpy(one.data(), d_state, bytes, cudaMemcpyDeviceToHost), "inplace ab copy one output");
    verify_result(two, reference, "two-buffer", W, p, reverse);
    verify_result(one, reference, "in-place", W, p, reverse);

    const double state_gib = double(bytes) / double(1ULL << 30);
    std::cout << std::fixed << std::setprecision(6)
              << "gridfp-reduced-component-inplace-ab"
              << " W=" << W << " p=" << p
              << " direction=" << (reverse ? "reverse" : "forward")
              << " states=" << states
              << " components=" << components
              << " blocks=" << blocks
              << " twobuffer_ms=" << twobuffer_ms
              << " inplace_ms=" << inplace_ms
              << " inplace_speedup=" << (inplace_ms > 0.0f ? twobuffer_ms / inplace_ms : 0.0f)
              << " one_state_GiB=" << state_gib
              << " twobuffer_state_GiB=" << 2.0 * state_gib
              << " inplace_state_GiB=" << state_gib
              << " component_rank_alignment=required"
              << " arithmetic=OK\n";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_state);
    cudaFree(d_error);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const unsigned blocks = argc > 2
        ? static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10))
        : 4096u;
    const std::uint32_t mod = argc > 3
        ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10))
        : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 5 || W > RP_MAX_W || blocks == 0 || mod < 3) return 2;

    ProductionFactorTables tables(W);
    const Rank64 states = tables.size();
    const Rank64 components = motzkin_count(W - 1) - motzkin_count(W - 3);
    const double one_gib = double(states) * 4.0 / double(1ULL << 30);
    if (plan_only) {
        const Rank64 launched_warps = Rank64(blocks) * WARPS_PER_BLOCK;
        std::cout << std::fixed << std::setprecision(6)
                  << "gridfp-reduced-component-inplace-ab-plan"
                  << " W=" << W
                  << " states=" << states
                  << " components=" << components
                  << " blocks=" << blocks
                  << " threads=" << THREADS
                  << " launched_warps=" << launched_warps
                  << " components_per_launched_warp="
                  << double(components) / double(launched_warps)
                  << " one_state_GiB=" << one_gib
                  << " twobuffer_state_GiB=" << 2.0 * one_gib
                  << " inplace_state_GiB=" << one_gib
                  << " per_8gpu_inplace_GiB=" << one_gib / 8.0
                  << " component_table_bytes=0"
                  << " second_state_buffer_bytes=0\n";
        return 0;
    }
    if (W > 12) {
        std::cerr << "execution mode is intentionally limited to W<=12; use --plan-only for production widths\n";
        return 3;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "inplace ab device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "inplace ab set device");
    install_tables(tables);

    for (int p = W - 1; p >= 3; --p)
        run_inplace_ab_position(W, p, false, tables, mod, blocks);
    for (int p = 1; p <= W - 3; ++p)
        run_inplace_ab_position(W, p, true, tables, mod, blocks);

    std::cout << "ALL_OK gridfp_reduced_production_cuda_component_inplace=1 W=" << W << '\n';
    return 0;
}
