#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_component_microprobe_main_unused
#include "gridfp_reduced_production_component_microprobe.cu"
#pragma pop_macro("main")

#include "gridfp_reduced_production_codec_device.cuh"

namespace {

__device__ __forceinline__ int raw_step_device(
    DeviceKey src, int W, int p, bool reverse, DeviceTerm* out
) {
    int n = 0;
    if (!src.blocked) {
        n = add_term(out, n, src, 1);
        if (n < 0) return n;
        const IncludeResult z = reverse
            ? include_horizontal_reverse(src.mate, W, p)
            : include_horizontal(src.mate, W, p);
        if (z.valid) n = add_term(out, n, DeviceKey{z.mate, std::uint8_t(z.blocked)}, 1);
        return n;
    }
    const MateID m = reverse
        ? blocked_exclude_reverse(src.mate, W, p)
        : blocked_exclude(src.mate, p);
    return add_term(out, n, DeviceKey{m, 0}, 1);
}

__device__ __forceinline__ void atomic_add_mod(
    std::uint32_t* ptr, std::uint32_t value, int coef, std::uint32_t mod
) {
    std::uint32_t old = *ptr;
    while (true) {
        long long z = static_cast<long long>(old) +
                      static_cast<long long>(coef) * static_cast<long long>(value);
        z %= static_cast<long long>(mod);
        if (z < 0) z += mod;
        const std::uint32_t desired = static_cast<std::uint32_t>(z);
        const std::uint32_t seen = atomicCAS(ptr, old, desired);
        if (seen == old) return;
        old = seen;
    }
}

__global__ void fused_row_edge_kernel(
    const std::uint32_t* __restrict__ input,
    std::uint32_t* __restrict__ output,
    Rank64 states,
    int W,
    bool reverse,
    std::uint32_t mod,
    unsigned long long* edge_ops,
    int* error
) {
    const Rank64 s = Rank64(blockIdx.x) * blockDim.x + threadIdx.x;
    if (s >= states) return;

    const int src_fixed = reverse ? W - 3 : 1;
    const int dst_fixed = reverse ? W - 2 : 0;
    const int p0 = reverse ? W - 2 : 2;
    const int p1 = reverse ? W - 1 : 1;
    const DeviceKey src = factor_unrank_device(s, W, src_fixed);
    const std::uint32_t value = input[s];

    DeviceTerm a[RP_MAX_TERMS]{};
    const int na = raw_step_device(src, W, p0, reverse, a);
    if (na < 0) {
        set_error(error, 41);
        return;
    }

    unsigned local_edges = 0;
    for (int i = 0; i < na; ++i) {
        DeviceTerm b[RP_MAX_TERMS]{};
        const int nb = raw_step_device(a[i].key, W, p1, reverse, b);
        if (nb < 0) {
            set_error(error, 42);
            return;
        }
        for (int j = 0; j < nb; ++j) {
            if (b[j].key.blocked) {
                set_error(error, 43);
                continue;
            }
            const int coef = int(a[i].coef) * int(b[j].coef);
            if (!coef) continue;
            const Rank64 d = factor_rank_device(b[j].key, W, dst_fixed);
            if (d >= states) {
                set_error(error, 44);
                continue;
            }
            atomic_add_mod(output + d, value, coef, mod);
            ++local_edges;
        }
    }
    atomicAdd(edge_ops, static_cast<unsigned long long>(local_edges));
}

void run_edge_direction(int W, bool reverse, const ProductionFactorTables& tables, std::uint32_t mod) {
    const int src_fixed = reverse ? W - 3 : 1;
    const int dst_fixed = reverse ? W - 2 : 0;
    const int p0 = reverse ? W - 2 : 2;
    const int p1 = reverse ? W - 1 : 1;
    ProductionFactorCodec src(tables, src_fixed);
    ProductionFactorCodec dst(tables, dst_fixed);
    const Rank64 states = tables.size();

    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    std::vector<std::uint32_t> reference(static_cast<std::size_t>(states));
    unsigned long long reference_edges = 0;
    for (Rank64 r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>((1 + (r * 2654435761ULL) % (mod - 1ULL)) % mod);

    for (Rank64 s = 0; s < states; ++s) {
        const Key k = src.unrank(s);
        const std::uint32_t value = input[static_cast<std::size_t>(s)];
        const Vec a = step_basis(k, W, p0, reverse);
        for (const auto& [mid, ca] : a) {
            const Vec b = step_basis(mid, W, p1, reverse);
            for (const auto& [out, cb] : b) {
                if (out.blocked) fail("row edge left blocked output");
                add_mod_signed(reference[static_cast<std::size_t>(dst.rank(out))], value, int(ca * cb), mod);
                ++reference_edges;
            }
        }
    }

    std::uint32_t* d_input = nullptr;
    std::uint32_t* d_output = nullptr;
    unsigned long long* d_edges = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_input, states * sizeof(std::uint32_t)), "edge alloc input");
    ck(cudaMalloc(&d_output, states * sizeof(std::uint32_t)), "edge alloc output");
    ck(cudaMalloc(&d_edges, sizeof(unsigned long long)), "edge alloc counter");
    ck(cudaMalloc(&d_error, sizeof(int)), "edge alloc error");
    ck(cudaMemcpy(d_input, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "edge copy input");
    ck(cudaMemset(d_output, 0, states * sizeof(std::uint32_t)), "edge zero output");
    ck(cudaMemset(d_edges, 0, sizeof(unsigned long long)), "edge zero counter");
    ck(cudaMemset(d_error, 0, sizeof(int)), "edge zero error");

    const int threads = 256;
    const unsigned blocks = static_cast<unsigned>((states + threads - 1) / threads);
    cudaEvent_t aev{}, bev{};
    ck(cudaEventCreate(&aev), "edge event a");
    ck(cudaEventCreate(&bev), "edge event b");
    ck(cudaEventRecord(aev), "edge record a");
    fused_row_edge_kernel<<<blocks, threads>>>(d_input, d_output, states, W, reverse, mod, d_edges, d_error);
    ck(cudaGetLastError(), "edge launch");
    ck(cudaEventRecord(bev), "edge record b");
    ck(cudaEventSynchronize(bev), "edge sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, aev, bev), "edge elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    unsigned long long edges = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_output, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "edge copy output");
    ck(cudaMemcpy(&edges, d_edges, sizeof(edges), cudaMemcpyDeviceToHost), "edge copy counter");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "edge copy error");
    if (error) {
        std::cerr << "FAIL row-edge device error=" << error << " reverse=" << reverse << '\n';
        std::exit(99);
    }
    if (edges != reference_edges) {
        std::cerr << "FAIL row-edge count gpu=" << edges << " cpu=" << reference_edges << '\n';
        std::exit(100);
    }
    if (output != reference) {
        for (Rank64 r = 0; r < states; ++r) {
            if (output[static_cast<std::size_t>(r)] == reference[static_cast<std::size_t>(r)]) continue;
            std::cerr << "FAIL row-edge arithmetic reverse=" << reverse << " rank=" << r
                      << " gpu=" << output[static_cast<std::size_t>(r)]
                      << " cpu=" << reference[static_cast<std::size_t>(r)] << '\n';
            break;
        }
        std::exit(101);
    }

    std::cout << "gridfp-reduced-row-edge-microprobe"
              << " W=" << W
              << " direction=" << (reverse ? "reverse" : "forward")
              << " states=" << states
              << " fused_edge_ops=" << edges
              << " avg_ops_per_source=" << double(edges) / double(states)
              << " kernel_ms=" << ms
              << " output_blocked_zero=1"
              << " boundary_atomic=1 arithmetic=OK\n";

    cudaEventDestroy(aev);
    cudaEventDestroy(bev);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_edges);
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
        const double stream_gib = double(tables.size()) * 4.0 / double(1ULL << 30);
        std::cout << "gridfp-reduced-row-edge-microprobe-plan"
                  << " W=" << W
                  << " states=" << tables.size()
                  << " u32_stream_GiB=" << stream_gib
                  << " source_unrank=device"
                  << " output_layout=reduced-main-slots"
                  << " boundary_atomic=1"
                  << " blocked_output=0\n";
        return 0;
    }
    if (W > 12) {
        std::cerr << "execution mode is intentionally limited to W<=12; use --plan-only for production widths\n";
        return 3;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "row-edge device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "row-edge set device");
    install_tables(tables);
    run_edge_direction(W, false, tables, mod);
    run_edge_direction(W, true, tables, mod);
    std::cout << "ALL_OK gridfp_reduced_production_cuda_row_edge=1 W=" << W << '\n';
    return 0;
}
