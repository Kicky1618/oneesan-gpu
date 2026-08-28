#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_component_microprobe_main_unused
#include "gridfp_reduced_production_component_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_reverse_device.cuh"

namespace {

__global__ void component_kernel_reverse(
    const std::uint32_t* __restrict__ input,
    std::uint32_t* __restrict__ output,
    unsigned long long* __restrict__ owner,
    Rank64 raw_labels,
    Rank64 state_count,
    int W,
    int p,
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
    const Rank64 label_rank = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);

    if (lane == 0) {
        sh_ns[warp] = 0;
        sh_nd[warp] = 0;
        if (label_rank < raw_labels) {
            const MateID label = motzkin_unrank_device(W - 1, label_rank);
            bool eligible = false;
            const DeviceKey seed = reverse_component_seed(label, W, p, eligible);
            if (eligible) {
                atomicAdd(component_count, 1ULL);
                sh_src[warp][0] = seed;
                sh_ns[warp] = 1;
                int cursor = 0;
                while (cursor < sh_ns[warp]) {
                    DeviceTerm edge[RP_MAX_TERMS]{};
                    const int ne = reduced_step_reverse(sh_src[warp][cursor++], W, p, edge);
                    if (ne < 0) {
                        set_error(error, 21);
                        break;
                    }
                    for (int ei = 0; ei < ne; ++ei) {
                        if (!edge[ei].coef) continue;
                        const DeviceKey d = edge[ei].key;
                        if (find_key(sh_dst[warp], sh_nd[warp], d) >= 0) continue;
                        if (sh_nd[warp] >= MAX_PAIRS) {
                            set_error(error, 22);
                            break;
                        }
                        sh_dst[warp][sh_nd[warp]++] = d;

                        DeviceTerm pre[RP_MAX_TERMS]{};
                        const int np = inverse_reduced_reverse(d, W, p, pre);
                        if (np < 0) {
                            set_error(error, 23);
                            break;
                        }
                        for (int pi = 0; pi < np; ++pi) {
                            if (!pre[pi].coef) continue;
                            if (find_key(sh_src[warp], sh_ns[warp], pre[pi].key) >= 0) continue;
                            if (sh_ns[warp] >= MAX_PAIRS) {
                                set_error(error, 24);
                                break;
                            }
                            sh_src[warp][sh_ns[warp]++] = pre[pi].key;
                        }
                    }
                    if (*error) break;
                }
                if (sh_ns[warp] != sh_nd[warp]) set_error(error, 25);
            }
        }
    }
    __syncwarp();

    const int ns = sh_ns[warp];
    const int nd = sh_nd[warp];
    if (lane < ns) {
        const Rank64 r = factor_rank_device(sh_src[warp][lane], W, p - 1);
        if (r >= state_count) {
            set_error(error, 26);
            sh_value[warp][lane] = 0;
        } else {
            sh_value[warp][lane] = input[r];
        }
    }
    __syncwarp();

    if (lane < nd) {
        DeviceTerm pre[RP_MAX_TERMS]{};
        const int np = inverse_reduced_reverse(sh_dst[warp][lane], W, p, pre);
        if (np < 0) {
            set_error(error, 27);
            return;
        }
        long long acc = 0;
        for (int i = 0; i < np; ++i) {
            if (!pre[i].coef) continue;
            const int si = find_key(sh_src[warp], ns, pre[i].key);
            if (si < 0) {
                set_error(error, 28);
                continue;
            }
            acc += static_cast<long long>(pre[i].coef) * static_cast<long long>(sh_value[warp][si]);
        }
        long long z = acc % static_cast<long long>(mod);
        if (z < 0) z += mod;
        const Rank64 dr = factor_rank_device(sh_dst[warp][lane], W, p);
        if (dr >= state_count) {
            set_error(error, 29);
            return;
        }
        const unsigned long long empty = ~0ULL;
        const unsigned long long prev = atomicCAS(owner + dr, empty, static_cast<unsigned long long>(label_rank));
        if (prev != empty && prev != static_cast<unsigned long long>(label_rank)) set_error(error, 30);
        output[dr] = static_cast<std::uint32_t>(z);
    }
}

void run_reverse_position(int W, int p, const ProductionFactorTables& tables, std::uint32_t mod) {
    ProductionFactorCodec src(tables, p - 1);
    ProductionFactorCodec dst(tables, p);
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
        for (const auto& [d, c] : reduced_step_basis(k, W, p, true))
            add_mod_signed(reference[static_cast<std::size_t>(dst.rank(d))], v, int(c), mod);
    }

    std::uint32_t* d_input = nullptr;
    std::uint32_t* d_output = nullptr;
    unsigned long long* d_owner = nullptr;
    unsigned long long* d_components = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_input, states * sizeof(std::uint32_t)), "alloc reverse input");
    ck(cudaMalloc(&d_output, states * sizeof(std::uint32_t)), "alloc reverse output");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "alloc reverse owner");
    ck(cudaMalloc(&d_components, sizeof(unsigned long long)), "alloc reverse components");
    ck(cudaMalloc(&d_error, sizeof(int)), "alloc reverse error");
    ck(cudaMemcpy(d_input, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "copy reverse input");
    ck(cudaMemset(d_output, 0, states * sizeof(std::uint32_t)), "zero reverse output");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "clear reverse owner");
    ck(cudaMemset(d_components, 0, sizeof(unsigned long long)), "zero reverse components");
    ck(cudaMemset(d_error, 0, sizeof(int)), "zero reverse error");

    const unsigned blocks = static_cast<unsigned>((raw_labels + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    cudaEvent_t a{}, b{};
    ck(cudaEventCreate(&a), "reverse event a");
    ck(cudaEventCreate(&b), "reverse event b");
    ck(cudaEventRecord(a), "record reverse a");
    component_kernel_reverse<<<blocks, THREADS>>>(
        d_input, d_output, d_owner, raw_labels, states, W, p, mod, d_components, d_error);
    ck(cudaGetLastError(), "launch reverse component microprobe");
    ck(cudaEventRecord(b), "record reverse b");
    ck(cudaEventSynchronize(b), "sync reverse b");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, a, b), "reverse elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    std::vector<unsigned long long> owner(static_cast<std::size_t>(states));
    unsigned long long components = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_output, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "copy reverse output");
    ck(cudaMemcpy(owner.data(), d_owner, states * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "copy reverse owner");
    ck(cudaMemcpy(&components, d_components, sizeof(components), cudaMemcpyDeviceToHost), "copy reverse components");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "copy reverse error");

    if (error) {
        std::cerr << "FAIL reverse device error=" << error << " W=" << W << " p=" << p << '\n';
        std::exit(95);
    }
    if (components != expected_components) {
        std::cerr << "FAIL reverse component count got=" << components << " want=" << expected_components << '\n';
        std::exit(96);
    }
    for (Rank64 r = 0; r < states; ++r) {
        if (owner[static_cast<std::size_t>(r)] == std::numeric_limits<unsigned long long>::max()) {
            std::cerr << "FAIL reverse uncovered destination rank=" << r << '\n';
            std::exit(97);
        }
        if (output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL reverse arithmetic W=" << W << " p=" << p << " rank=" << r
                      << " gpu=" << output[static_cast<std::size_t>(r)]
                      << " cpu=" << reference[static_cast<std::size_t>(r)] << '\n';
            std::exit(98);
        }
    }

    const double values_per_s = ms > 0 ? double(states) / (double(ms) * 1e-3) : 0.0;
    std::cout << "gridfp-reduced-component-reverse-microprobe"
              << " W=" << W << " p=" << p
              << " states=" << states
              << " raw_label_warps=" << raw_labels
              << " components=" << components
              << " skipped_warps=" << (raw_labels - components)
              << " kernel_ms=" << ms
              << " state_pairs_per_s=" << values_per_s
              << " source_load_once=1 destination_store_once=1"
              << " component_table_bytes=0 inverse_table_bytes=0"
              << " arithmetic=OK\n";

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
    const int requested_p = argc > 2 ? std::atoi(argv[2]) : -1;
    const std::uint32_t mod = argc > 3 ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 5 || W > RP_MAX_W || mod < 3) return 2;
    if (requested_p >= 0 && (requested_p < 1 || requested_p > W - 3)) return 3;

    ProductionFactorTables tables(W);
    const Rank64 raw_labels = motzkin_count(W - 1);
    const Rank64 components = raw_labels - motzkin_count(W - 3);
    const Rank64 states = tables.size();
    if (plan_only) {
        std::cout << "gridfp-reduced-component-reverse-microprobe-plan"
                  << " W=" << W
                  << " states=" << states
                  << " raw_label_warps=" << raw_labels
                  << " components=" << components
                  << " skipped_fraction=" << double(raw_labels - components) / double(raw_labels)
                  << " blocks_per_position=" << ((raw_labels + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK)
                  << " threads=" << THREADS
                  << " max_pairs_storage=" << MAX_PAIRS
                  << " component_table_bytes=0 inverse_table_bytes=0\n";
        return 0;
    }
    if (W > 12) {
        std::cerr << "execution mode is intentionally limited to W<=12; use --plan-only for production widths\n";
        return 4;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "reverse device count");
    if (visible < 1) return 5;
    ck(cudaSetDevice(0), "reverse set device");
    install_tables(tables);

    if (requested_p >= 0) {
        run_reverse_position(W, requested_p, tables, mod);
    } else {
        for (int p = 1; p <= W - 3; ++p) run_reverse_position(W, p, tables, mod);
    }
    std::cout << "ALL_OK gridfp_reduced_production_cuda_component_reverse=1 W=" << W << '\n';
    return 0;
}
