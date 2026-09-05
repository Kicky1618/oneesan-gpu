#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_component_register_microprobe_main_unused
#include "gridfp_reduced_production_component_register_microprobe.cu"
#pragma pop_macro("main")

namespace {

__global__ void component_persistent_kernel(
    const std::uint32_t* __restrict__ input,
    std::uint32_t* __restrict__ output,
    unsigned long long* __restrict__ owner,
    Rank64 raw_labels,
    Rank64 state_count,
    int W,
    int p,
    bool reverse,
    std::uint32_t mod,
    unsigned long long* component_count,
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

    for (Rank64 label_rank = first; label_rank < raw_labels; label_rank += stride) {
        if (lane == 0) {
            sh_ns[warp] = 0;
            sh_nd[warp] = 0;
            const MateID label = motzkin_unrank_device(W - 1, label_rank);
            bool eligible = false;
            const DeviceKey seed = component_seed_direction(label, W, p, reverse, eligible);
            if (eligible) {
                atomicAdd(component_count, 1ULL);
                sh_src[warp][0] = seed;
                sh_ns[warp] = 1;
                int cursor = 0;
                while (cursor < sh_ns[warp]) {
                    SmallTerms edge;
                    if (!small_step(sh_src[warp][cursor++], W, p, reverse, edge)) {
                        set_error(error, 71);
                        break;
                    }
                    for (int ei = 0; ei < edge.n; ++ei) {
                        if (!edge.v[ei].coef) continue;
                        const DeviceKey d = edge.v[ei].key;
                        if (find_key(sh_dst[warp], sh_nd[warp], d) >= 0) continue;
                        if (sh_nd[warp] >= MAX_PAIRS) {
                            set_error(error, 72);
                            break;
                        }
                        sh_dst[warp][sh_nd[warp]++] = d;
                        DeviceTerm pre[RP_MAX_TERMS]{};
                        const int np = inverse_direction(d, W, p, reverse, pre);
                        if (np < 0) {
                            set_error(error, 73);
                            break;
                        }
                        for (int pi = 0; pi < np; ++pi) {
                            if (!pre[pi].coef) continue;
                            if (find_key(sh_src[warp], sh_ns[warp], pre[pi].key) >= 0) continue;
                            if (sh_ns[warp] >= MAX_PAIRS) {
                                set_error(error, 74);
                                break;
                            }
                            sh_src[warp][sh_ns[warp]++] = pre[pi].key;
                        }
                    }
                    if (*error) break;
                }
                if (sh_ns[warp] != sh_nd[warp]) set_error(error, 75);
            }
        }
        __syncwarp();

        const int ns = sh_ns[warp];
        const int nd = sh_nd[warp];
        if (lane < ns) {
            const Rank64 r = factor_rank_device(sh_src[warp][lane], W, src_fixed);
            if (r >= state_count) {
                set_error(error, 76);
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
                    set_error(error, 77);
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
                set_error(error, 78);
            } else {
                const unsigned long long empty = ~0ULL;
                const unsigned long long prev = atomicCAS(
                    owner + dr, empty, static_cast<unsigned long long>(label_rank));
                if (prev != empty && prev != static_cast<unsigned long long>(label_rank))
                    set_error(error, 79);
                output[dr] = static_cast<std::uint32_t>(z);
            }
        }
        __syncwarp();
    }
}

void run_persistent_position(
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
    const Rank64 raw_labels = motzkin_count(W - 1);
    const Rank64 expected_components = raw_labels - motzkin_count(W - 3);

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
    unsigned long long* d_components = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_input, states * sizeof(std::uint32_t)), "persistent alloc input");
    ck(cudaMalloc(&d_output, states * sizeof(std::uint32_t)), "persistent alloc output");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "persistent alloc owner");
    ck(cudaMalloc(&d_components, sizeof(unsigned long long)), "persistent alloc components");
    ck(cudaMalloc(&d_error, sizeof(int)), "persistent alloc error");
    ck(cudaMemcpy(d_input, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "persistent copy input");
    ck(cudaMemset(d_output, 0, states * sizeof(std::uint32_t)), "persistent zero output");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "persistent clear owner");
    ck(cudaMemset(d_components, 0, sizeof(unsigned long long)), "persistent zero components");
    ck(cudaMemset(d_error, 0, sizeof(int)), "persistent zero error");

    const Rank64 one_pass_blocks = (raw_labels + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned blocks = static_cast<unsigned>(std::max<Rank64>(1, std::min<Rank64>(requested_blocks, one_pass_blocks)));
    cudaEvent_t a{}, b{};
    ck(cudaEventCreate(&a), "persistent event a");
    ck(cudaEventCreate(&b), "persistent event b");
    ck(cudaEventRecord(a), "persistent record a");
    component_persistent_kernel<<<blocks, THREADS>>>(
        d_input, d_output, d_owner, raw_labels, states, W, p, reverse, mod, d_components, d_error);
    ck(cudaGetLastError(), "persistent launch");
    ck(cudaEventRecord(b), "persistent record b");
    ck(cudaEventSynchronize(b), "persistent sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, a, b), "persistent elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    std::vector<unsigned long long> owner(static_cast<std::size_t>(states));
    unsigned long long components = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_output, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "persistent copy output");
    ck(cudaMemcpy(owner.data(), d_owner, states * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "persistent copy owner");
    ck(cudaMemcpy(&components, d_components, sizeof(components), cudaMemcpyDeviceToHost), "persistent copy components");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "persistent copy error");
    if (error || components != expected_components) {
        std::cerr << "FAIL persistent component error=" << error << " components=" << components
                  << " want=" << expected_components << '\n';
        std::exit(104);
    }
    for (Rank64 r = 0; r < states; ++r) {
        if (owner[static_cast<std::size_t>(r)] == std::numeric_limits<unsigned long long>::max() ||
            output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL persistent arithmetic W=" << W << " p=" << p
                      << " reverse=" << reverse << " rank=" << r << '\n';
            std::exit(105);
        }
    }

    const double labels_per_warp = double(raw_labels) / double(Rank64(blocks) * WARPS_PER_BLOCK);
    std::cout << "gridfp-reduced-component-persistent-microprobe"
              << " W=" << W << " p=" << p
              << " direction=" << (reverse ? "reverse" : "forward")
              << " states=" << states
              << " raw_labels=" << raw_labels
              << " components=" << components
              << " blocks=" << blocks
              << " labels_per_launched_warp=" << labels_per_warp
              << " kernel_ms=" << ms
              << " grid_stride=1 destination_inverse_buffer=0 arithmetic=OK\n";

    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_owner);
    cudaFree(d_components);
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
    const Rank64 raw_labels = motzkin_count(W - 1);
    const Rank64 components = raw_labels - motzkin_count(W - 3);
    if (plan_only) {
        const double labels_per_warp = double(raw_labels) / double(Rank64(blocks) * WARPS_PER_BLOCK);
        std::cout << "gridfp-reduced-component-persistent-microprobe-plan"
                  << " W=" << W
                  << " states=" << tables.size()
                  << " raw_labels=" << raw_labels
                  << " components=" << components
                  << " blocks=" << blocks
                  << " threads=" << THREADS
                  << " launched_warps=" << (Rank64(blocks) * WARPS_PER_BLOCK)
                  << " labels_per_launched_warp=" << labels_per_warp
                  << " grid_stride=1"
                  << " one_label_per_warp_launch_required=0\n";
        return 0;
    }
    if (W > 12) {
        std::cerr << "execution mode is intentionally limited to W<=12; use --plan-only for production widths\n";
        return 3;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "persistent device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "persistent set device");
    install_tables(tables);
    for (int p = W - 1; p >= 3; --p) run_persistent_position(W, p, false, tables, mod, blocks);
    for (int p = 1; p <= W - 3; ++p) run_persistent_position(W, p, true, tables, mod, blocks);
    std::cout << "ALL_OK gridfp_reduced_production_cuda_component_persistent=1 W=" << W << '\n';
    return 0;
}
