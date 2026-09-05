#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_component_persistent_microprobe_main_unused
#include "gridfp_reduced_production_component_persistent_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_component_codec_device.cuh"

namespace {

__global__ void component_dense_kernel(
    const std::uint32_t* __restrict__ input,
    std::uint32_t* __restrict__ output,
    unsigned long long* __restrict__ owner,
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
                set_error(error, 81);
            } else {
                atomicAdd(processed, 1ULL);
                sh_src[warp][0] = seed;
                sh_ns[warp] = 1;
                int cursor = 0;
                while (cursor < sh_ns[warp]) {
                    SmallTerms edge;
                    if (!small_step(sh_src[warp][cursor++], W, p, reverse, edge)) {
                        set_error(error, 82);
                        break;
                    }
                    for (int ei = 0; ei < edge.n; ++ei) {
                        if (!edge.v[ei].coef) continue;
                        const DeviceKey d = edge.v[ei].key;
                        if (find_key(sh_dst[warp], sh_nd[warp], d) >= 0) continue;
                        if (sh_nd[warp] >= MAX_PAIRS) {
                            set_error(error, 83);
                            break;
                        }
                        sh_dst[warp][sh_nd[warp]++] = d;
                        DeviceTerm pre[RP_MAX_TERMS]{};
                        const int np = inverse_direction(d, W, p, reverse, pre);
                        if (np < 0) {
                            set_error(error, 84);
                            break;
                        }
                        for (int pi = 0; pi < np; ++pi) {
                            if (!pre[pi].coef) continue;
                            if (find_key(sh_src[warp], sh_ns[warp], pre[pi].key) >= 0) continue;
                            if (sh_ns[warp] >= MAX_PAIRS) {
                                set_error(error, 85);
                                break;
                            }
                            sh_src[warp][sh_ns[warp]++] = pre[pi].key;
                        }
                    }
                    if (*error) break;
                }
                if (sh_ns[warp] != sh_nd[warp]) set_error(error, 86);
            }
        }
        __syncwarp();

        const int ns = sh_ns[warp];
        const int nd = sh_nd[warp];
        if (lane < ns) {
            const Rank64 r = factor_rank_device(sh_src[warp][lane], W, src_fixed);
            if (r >= state_count) {
                set_error(error, 87);
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
                    set_error(error, 88);
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
                set_error(error, 89);
            } else {
                const unsigned long long empty = ~0ULL;
                const unsigned long long prev = atomicCAS(
                    owner + dr, empty, static_cast<unsigned long long>(component_rank));
                if (prev != empty && prev != static_cast<unsigned long long>(component_rank))
                    set_error(error, 90);
                output[dr] = static_cast<std::uint32_t>(z);
            }
        }
        __syncwarp();
    }
}

void run_dense_position(
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
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>((1 + (r * 2654435761ULL) % (mod - 1ULL)) % mod);
    for (Rank64 s = 0; s < states; ++s) {
        const Key k = src.unrank(s);
        const std::uint32_t v = input[static_cast<std::size_t>(s)];
        for (const auto& [d, c] : reduced_step_basis(k, W, p, reverse))
            add_mod_signed(reference[static_cast<std::size_t>(dst.rank(d))], v, int(c), mod);
    }

    std::uint32_t* d_input = nullptr;
    std::uint32_t* d_output = nullptr;
    unsigned long long* d_owner = nullptr;
    unsigned long long* d_processed = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_input, states * sizeof(std::uint32_t)), "dense alloc input");
    ck(cudaMalloc(&d_output, states * sizeof(std::uint32_t)), "dense alloc output");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "dense alloc owner");
    ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "dense alloc processed");
    ck(cudaMalloc(&d_error, sizeof(int)), "dense alloc error");
    ck(cudaMemcpy(d_input, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "dense copy input");
    ck(cudaMemset(d_output, 0, states * sizeof(std::uint32_t)), "dense zero output");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "dense clear owner");
    ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "dense zero processed");
    ck(cudaMemset(d_error, 0, sizeof(int)), "dense zero error");

    const Rank64 one_pass_blocks = (components + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned blocks = static_cast<unsigned>(std::max<Rank64>(1, std::min<Rank64>(requested_blocks, one_pass_blocks)));
    cudaEvent_t a{}, b{};
    ck(cudaEventCreate(&a), "dense event a");
    ck(cudaEventCreate(&b), "dense event b");
    ck(cudaEventRecord(a), "dense record a");
    component_dense_kernel<<<blocks, THREADS>>>(
        d_input, d_output, d_owner, components, states, W, p, reverse, mod, d_processed, d_error);
    ck(cudaGetLastError(), "dense launch");
    ck(cudaEventRecord(b), "dense record b");
    ck(cudaEventSynchronize(b), "dense sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, a, b), "dense elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    std::vector<unsigned long long> owner(static_cast<std::size_t>(states));
    unsigned long long processed = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_output, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "dense copy output");
    ck(cudaMemcpy(owner.data(), d_owner, states * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "dense copy owner");
    ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost), "dense copy processed");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "dense copy error");
    if (error || processed != components) {
        std::cerr << "FAIL dense component error=" << error << " processed=" << processed
                  << " want=" << components << '\n';
        std::exit(106);
    }
    for (Rank64 r = 0; r < states; ++r) {
        if (owner[static_cast<std::size_t>(r)] == std::numeric_limits<unsigned long long>::max() ||
            output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL dense arithmetic W=" << W << " p=" << p
                      << " reverse=" << reverse << " rank=" << r << '\n';
            std::exit(107);
        }
    }

    const double components_per_warp = double(components) / double(Rank64(blocks) * WARPS_PER_BLOCK);
    std::cout << "gridfp-reduced-component-dense-microprobe"
              << " W=" << W << " p=" << p
              << " direction=" << (reverse ? "reverse" : "forward")
              << " states=" << states
              << " components=" << components
              << " blocks=" << blocks
              << " components_per_launched_warp=" << components_per_warp
              << " kernel_ms=" << ms
              << " skipped_labels=0 dense_component_codec=1"
              << " destination_inverse_buffer=0 arithmetic=OK\n";

    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaFree(d_input);
    cudaFree(d_output);
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
    const Rank64 components = motzkin_count(W - 1) - motzkin_count(W - 3);
    if (plan_only) {
        const double components_per_warp = double(components) / double(Rank64(blocks) * WARPS_PER_BLOCK);
        std::cout << "gridfp-reduced-component-dense-microprobe-plan"
                  << " W=" << W
                  << " states=" << tables.size()
                  << " components=" << components
                  << " blocks=" << blocks
                  << " threads=" << THREADS
                  << " launched_warps=" << (Rank64(blocks) * WARPS_PER_BLOCK)
                  << " components_per_launched_warp=" << components_per_warp
                  << " skipped_labels=0"
                  << " saved_raw_label_unranks=" << motzkin_count(W - 3)
                  << " component_table_bytes=0\n";
        return 0;
    }
    if (W > 12) {
        std::cerr << "execution mode is intentionally limited to W<=12; use --plan-only for production widths\n";
        return 3;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "dense device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "dense set device");
    install_tables(tables);
    for (int p = W - 1; p >= 3; --p) run_dense_position(W, p, false, tables, mod, blocks);
    for (int p = 1; p <= W - 3; ++p) run_dense_position(W, p, true, tables, mod, blocks);
    std::cout << "ALL_OK gridfp_reduced_production_cuda_component_dense=1 W=" << W << '\n';
    return 0;
}
