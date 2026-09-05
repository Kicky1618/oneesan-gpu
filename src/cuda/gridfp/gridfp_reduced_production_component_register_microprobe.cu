#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_component_microprobe_main_unused
#include "gridfp_reduced_production_component_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_reverse_device.cuh"

namespace {

struct SmallTerms {
    DeviceTerm v[3]{};
    int n = 0;
};

__device__ __forceinline__ bool small_add(SmallTerms& z, DeviceKey k, int c) {
    if (!c) return true;
    for (int i = 0; i < z.n; ++i) {
        if (!key_equal(z.v[i].key, k)) continue;
        const int x = int(z.v[i].coef) + c;
        z.v[i].coef = static_cast<std::int8_t>(x);
        if (!x) z.v[i] = z.v[--z.n];
        return true;
    }
    if (z.n >= 3) return false;
    z.v[z.n++] = DeviceTerm{k, static_cast<std::int8_t>(c)};
    return true;
}

__device__ __forceinline__ bool small_project_forward(DeviceKey k, int W, int q, SmallTerms& z) {
    if (!k.blocked || mget(k.mate, q - 1) != N) return small_add(z, k, 1);
    const MateID nn = blocked_exclude(k.mate, q);
    return small_add(z, DeviceKey{nn, 0}, 1) &&
           small_add(z, DeviceKey{msetpair(nn, q, LR), 0}, -1);
}

__device__ __forceinline__ bool small_step_forward(DeviceKey src, int W, int p, SmallTerms& z) {
    if (!src.blocked) {
        if (!small_add(z, src, 1)) return false;
        const IncludeResult x = include_horizontal(src.mate, W, p);
        if (!x.valid) return true;
        return small_project_forward(DeviceKey{x.mate, std::uint8_t(x.blocked)}, W, p - 1, z);
    }
    return small_add(z, DeviceKey{blocked_exclude(src.mate, p), 0}, 1);
}

__device__ __forceinline__ bool small_step(DeviceKey src, int W, int p, bool reverse, SmallTerms& z) {
    if (!reverse) return small_step_forward(src, W, p, z);
    const int fp = W - p;
    SmallTerms tmp;
    if (!small_step_forward(mirror_key_device(src, W), W, fp, tmp)) return false;
    for (int i = 0; i < tmp.n; ++i) {
        if (!small_add(z, mirror_key_device(tmp.v[i].key, W), tmp.v[i].coef)) return false;
    }
    return true;
}

__device__ __forceinline__ DeviceKey component_seed_direction(
    MateID label, int W, int p, bool reverse, bool& eligible
) {
    return reverse ? reverse_component_seed(label, W, p, eligible)
                   : forward_component_seed(label, W, p, eligible);
}

__device__ __forceinline__ int inverse_direction(
    DeviceKey d, int W, int p, bool reverse, DeviceTerm* pre
) {
    return reverse ? inverse_reduced_reverse(d, W, p, pre)
                   : inverse_reduced_forward(d, W, p, pre);
}

__global__ void component_register_kernel(
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
    const Rank64 label_rank = Rank64(blockIdx.x) * WARPS_PER_BLOCK + Rank64(warp);

    if (lane == 0) {
        sh_ns[warp] = 0;
        sh_nd[warp] = 0;
        if (label_rank < raw_labels) {
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
                        set_error(error, 61);
                        break;
                    }
                    for (int ei = 0; ei < edge.n; ++ei) {
                        if (!edge.v[ei].coef) continue;
                        const DeviceKey d = edge.v[ei].key;
                        if (find_key(sh_dst[warp], sh_nd[warp], d) >= 0) continue;
                        if (sh_nd[warp] >= MAX_PAIRS) {
                            set_error(error, 62);
                            break;
                        }
                        sh_dst[warp][sh_nd[warp]++] = d;
                        DeviceTerm pre[RP_MAX_TERMS]{};
                        const int np = inverse_direction(d, W, p, reverse, pre);
                        if (np < 0) {
                            set_error(error, 63);
                            break;
                        }
                        for (int pi = 0; pi < np; ++pi) {
                            if (!pre[pi].coef) continue;
                            if (find_key(sh_src[warp], sh_ns[warp], pre[pi].key) >= 0) continue;
                            if (sh_ns[warp] >= MAX_PAIRS) {
                                set_error(error, 64);
                                break;
                            }
                            sh_src[warp][sh_ns[warp]++] = pre[pi].key;
                        }
                    }
                    if (*error) break;
                }
                if (sh_ns[warp] != sh_nd[warp]) set_error(error, 65);
            }
        }
    }
    __syncwarp();

    const int ns = sh_ns[warp];
    const int nd = sh_nd[warp];
    const int src_fixed = p - 1;
    const int dst_fixed = reverse ? p : p - 2;

    if (lane < ns) {
        const Rank64 r = factor_rank_device(sh_src[warp][lane], W, src_fixed);
        if (r >= state_count) {
            set_error(error, 66);
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
                set_error(error, 67);
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
            set_error(error, 68);
            return;
        }
        const unsigned long long empty = ~0ULL;
        const unsigned long long prev = atomicCAS(owner + dr, empty, static_cast<unsigned long long>(label_rank));
        if (prev != empty && prev != static_cast<unsigned long long>(label_rank)) set_error(error, 69);
        output[dr] = static_cast<std::uint32_t>(z);
    }
}

void run_register_position(
    int W, int p, bool reverse, const ProductionFactorTables& tables, std::uint32_t mod
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
    ck(cudaMalloc(&d_input, states * sizeof(std::uint32_t)), "register alloc input");
    ck(cudaMalloc(&d_output, states * sizeof(std::uint32_t)), "register alloc output");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "register alloc owner");
    ck(cudaMalloc(&d_components, sizeof(unsigned long long)), "register alloc components");
    ck(cudaMalloc(&d_error, sizeof(int)), "register alloc error");
    ck(cudaMemcpy(d_input, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "register copy input");
    ck(cudaMemset(d_output, 0, states * sizeof(std::uint32_t)), "register zero output");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "register clear owner");
    ck(cudaMemset(d_components, 0, sizeof(unsigned long long)), "register zero components");
    ck(cudaMemset(d_error, 0, sizeof(int)), "register zero error");

    const unsigned blocks = static_cast<unsigned>((raw_labels + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    cudaEvent_t a{}, b{};
    ck(cudaEventCreate(&a), "register event a");
    ck(cudaEventCreate(&b), "register event b");
    ck(cudaEventRecord(a), "register record a");
    component_register_kernel<<<blocks, THREADS>>>(
        d_input, d_output, d_owner, raw_labels, states, W, p, reverse, mod, d_components, d_error);
    ck(cudaGetLastError(), "register launch");
    ck(cudaEventRecord(b), "register record b");
    ck(cudaEventSynchronize(b), "register sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, a, b), "register elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    std::vector<unsigned long long> owner(static_cast<std::size_t>(states));
    unsigned long long components = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_output, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "register copy output");
    ck(cudaMemcpy(owner.data(), d_owner, states * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "register copy owner");
    ck(cudaMemcpy(&components, d_components, sizeof(components), cudaMemcpyDeviceToHost), "register copy components");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "register copy error");

    if (error || components != expected_components) {
        std::cerr << "FAIL register component error=" << error << " components=" << components
                  << " want=" << expected_components << '\n';
        std::exit(102);
    }
    for (Rank64 r = 0; r < states; ++r) {
        if (owner[static_cast<std::size_t>(r)] == std::numeric_limits<unsigned long long>::max() ||
            output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL register arithmetic W=" << W << " p=" << p
                      << " reverse=" << reverse << " rank=" << r << '\n';
            std::exit(103);
        }
    }

    std::cout << "gridfp-reduced-component-register-microprobe"
              << " W=" << W << " p=" << p
              << " direction=" << (reverse ? "reverse" : "forward")
              << " states=" << states
              << " components=" << components
              << " kernel_ms=" << ms
              << " lane_term_capacity=3"
              << " destination_inverse_buffer=0"
              << " source_load_once=1 destination_store_once=1 arithmetic=OK\n";

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
    const std::uint32_t mod = argc > 2 ? static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10)) : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 5 || W > RP_MAX_W || mod < 3) return 2;

    ProductionFactorTables tables(W);
    if (plan_only) {
        std::cout << "gridfp-reduced-component-register-microprobe-plan"
                  << " W=" << W
                  << " states=" << tables.size()
                  << " components=" << (motzkin_count(W - 1) - motzkin_count(W - 3))
                  << " warps_per_block=" << WARPS_PER_BLOCK
                  << " lane_term_capacity=3"
                  << " destination_inverse_buffer=0"
                  << " shared_component_keys=1"
                  << " component_table_bytes=0\n";
        return 0;
    }
    if (W > 12) {
        std::cerr << "execution mode is intentionally limited to W<=12; use --plan-only for production widths\n";
        return 3;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "register device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "register set device");
    install_tables(tables);
    for (int p = W - 1; p >= 3; --p) run_register_position(W, p, false, tables, mod);
    for (int p = 1; p <= W - 3; ++p) run_register_position(W, p, true, tables, mod);
    std::cout << "ALL_OK gridfp_reduced_production_cuda_component_register=1 W=" << W << '\n';
    return 0;
}
