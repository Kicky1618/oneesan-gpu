#pragma push_macro("main")
#undef main
#define main two_cell_fusion2_shared_microprobe_main_unused
#include "two_cell_fusion2_shared_microprobe.cu"
#pragma pop_macro("main")

namespace {

// Enumerate fixed-popcount masks in the same lexicographic support order used
// by support_unrank().  One CTA owns one outer mask and therefore one fused
// block.  Launching one occupied-count bucket at a time lets each bucket opt in
// to exactly 2*F_2(o)*sizeof(u32) dynamic shared bytes instead of reserving the
// global maximum.
__global__ void two_cell_fusion2_bucket_kernel(
    const std::uint32_t* __restrict__ input,
    std::uint32_t* __restrict__ output,
    unsigned long long* __restrict__ owner,
    int W,
    int start,
    int outer_ones,
    Rank support_count,
    Rank state_count,
    std::uint32_t mod,
    unsigned long long* processed_blocks,
    unsigned long long* global_loads,
    unsigned long long* global_stores,
    int* error
) {
    extern __shared__ std::uint32_t shared_values[];
    const int outer_bits = W - FUSION_STEPS - 3;
    const Rank n = oneesan::twocell::fusion_block_size(
        FUSION_STEPS, outer_ones, TC_RANK_TABLES);
    std::uint32_t* cur = shared_values;
    std::uint32_t* next = shared_values + n;

    for (Rank support_rank = blockIdx.x; support_rank < support_count;
         support_rank += gridDim.x) {
        const std::uint32_t outer = oneesan::twocell::support_unrank(
            outer_bits, outer_ones, support_rank, TC_RANK_TABLES);

        for (Rank r = threadIdx.x; r < n; r += blockDim.x) {
            const auto d = oneesan::twocell::fusion_local_unrank_at(
                r, outer, W, start, FUSION_STEPS, start, TC_RANK_TABLES);
            if (!d.valid) {
                set_error(error, 231);
                continue;
            }
            const Rank gr = oneesan::twocell::stationary_rank_with_primitive(
                d.key, W, start, d.primitive,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
            if (gr >= state_count) {
                set_error(error, 232);
                continue;
            }
            cur[r] = input[gr];
            atomicAdd(global_loads, 1ULL);
        }
        __syncthreads();

        for (int phase = 0; phase < FUSION_STEPS; ++phase) {
            const int active = start + phase;
            for (Rank r = threadIdx.x; r < n; r += blockDim.x) {
                const auto dst = oneesan::twocell::fusion_local_unrank_at(
                    r, outer, W, start, FUSION_STEPS, active + 1,
                    TC_RANK_TABLES);
                if (!dst.valid) {
                    set_error(error, 233);
                    continue;
                }
                const auto pre = oneesan::twocell::inverse_K(dst.key, W, active);
                if (pre.overflow) {
                    set_error(error, 234);
                    continue;
                }
                unsigned long long sum = 0;
                for (int q = 0; q < pre.size; ++q) {
                    const PackedKey src = pre.value[q];
                    if (oneesan::twocell::fusion_outer_mask_at(
                            src, start, FUSION_STEPS, active) != outer) {
                        set_error(error, 235);
                        continue;
                    }
                    const Rank sr = oneesan::twocell::fusion_local_rank_at(
                        src, W, start, FUSION_STEPS, active, outer_ones,
                        TC_RANK_TABLES);
                    if (sr >= n) {
                        set_error(error, 236);
                        continue;
                    }
                    sum += cur[sr];
                }
                next[r] = static_cast<std::uint32_t>(sum % mod);
            }
            __syncthreads();
            std::uint32_t* tmp = cur;
            cur = next;
            next = tmp;
            __syncthreads();
        }

        for (Rank r = threadIdx.x; r < n; r += blockDim.x) {
            const auto d = oneesan::twocell::fusion_local_unrank_at(
                r, outer, W, start, FUSION_STEPS, start + FUSION_STEPS,
                TC_RANK_TABLES);
            if (!d.valid) {
                set_error(error, 237);
                continue;
            }
            const Rank gr = oneesan::twocell::stationary_rank_with_primitive(
                d.key, W, start + FUSION_STEPS, d.primitive,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
            if (gr >= state_count) {
                set_error(error, 238);
                continue;
            }
            const unsigned long long component_id =
                (static_cast<unsigned long long>(outer_ones) << 56) |
                static_cast<unsigned long long>(support_rank);
            const unsigned long long previous = atomicCAS(
                owner + gr, ~0ULL, component_id);
            if (previous != ~0ULL && previous != component_id)
                set_error(error, 239);
            output[gr] = cur[r];
            atomicAdd(global_stores, 1ULL);
        }
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(processed_blocks, 1ULL);
        __syncthreads();
    }
}

struct Fusion2BucketPlan {
    Rank fitted_blocks = 0;
    Rank fitted_states = 0;
    Rank fallback_blocks = 0;
    Rank fallback_states = 0;
    int max_fused_outer_ones = -1;
};

Fusion2BucketPlan make_bucket_plan(
    int W,
    Rank shared_limit_bytes,
    const RankTables& rt
) {
    Fusion2BucketPlan p{};
    const int outer_bits = W - 5;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank blocks = rt.choose[outer_bits][o];
        const Rank n = oneesan::twocell::fusion_block_size(2, o, rt);
        const bool fit = 2 * n * sizeof(std::uint32_t) <= shared_limit_bytes;
        if (fit) {
            p.fitted_blocks += blocks;
            p.fitted_states += blocks * n;
            p.max_fused_outer_ones = o;
        } else {
            p.fallback_blocks += blocks;
            p.fallback_states += blocks * n;
        }
    }
    return p;
}

void print_bucket_plan(int W, Rank shared_limit, const RankTables& rt) {
    const auto p = make_bucket_plan(W, shared_limit, rt);
    const Rank total_states = p.fitted_states + p.fallback_states;
    const Rank total_blocks = p.fitted_blocks + p.fallback_blocks;
    const double sf = total_states ? double(p.fitted_states) / double(total_states) : 0.0;
    std::cout << "fusion2_bucket_plan"
              << " W=" << W
              << " shared_limit_bytes=" << shared_limit
              << " max_fused_outer_ones=" << p.max_fused_outer_ones
              << " fused_blocks=" << p.fitted_blocks
              << " fallback_blocks=" << p.fallback_blocks
              << " fused_block_fraction="
              << (total_blocks ? double(p.fitted_blocks) / double(total_blocks) : 0.0)
              << " fused_state_fraction=" << sf
              << " ideal_HBM_reduction_vs_two_pass=" << 0.5 * sf
              << "\n";
}

void run_bucketed_fused_only(
    int W,
    int start,
    int threads,
    Rank shared_limit,
    const RankTables& rt,
    const StationaryRankTables& st,
    std::uint32_t mod
) {
    const Rank states = st.total[W];
    const int outer_bits = W - 5;
    const auto plan = make_bucket_plan(W, shared_limit, rt);

    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + Rank(start) * 131ULL) % (mod - 1ULL)));
    const auto full_reference = fusion2_reference(input, W, start, rt, st, mod);

    std::uint32_t* d_input = nullptr;
    std::uint32_t* d_output = nullptr;
    unsigned long long* d_owner = nullptr;
    unsigned long long* d_blocks = nullptr;
    unsigned long long* d_loads = nullptr;
    unsigned long long* d_stores = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_input, states * sizeof(std::uint32_t)), "bucket alloc input");
    ck(cudaMalloc(&d_output, states * sizeof(std::uint32_t)), "bucket alloc output");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "bucket alloc owner");
    ck(cudaMalloc(&d_blocks, sizeof(unsigned long long)), "bucket alloc blocks");
    ck(cudaMalloc(&d_loads, sizeof(unsigned long long)), "bucket alloc loads");
    ck(cudaMalloc(&d_stores, sizeof(unsigned long long)), "bucket alloc stores");
    ck(cudaMalloc(&d_error, sizeof(int)), "bucket alloc error");
    ck(cudaMemcpy(d_input, input.data(), states * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice), "bucket copy input");
    ck(cudaMemset(d_output, 0, states * sizeof(std::uint32_t)), "bucket zero output");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "bucket clear owner");
    ck(cudaMemset(d_blocks, 0, sizeof(unsigned long long)), "bucket zero blocks");
    ck(cudaMemset(d_loads, 0, sizeof(unsigned long long)), "bucket zero loads");
    ck(cudaMemset(d_stores, 0, sizeof(unsigned long long)), "bucket zero stores");
    ck(cudaMemset(d_error, 0, sizeof(int)), "bucket zero error");

    for (int o = 0; o <= outer_bits; ++o) {
        const Rank n = oneesan::twocell::fusion_block_size(2, o, rt);
        const Rank shared_bytes = 2 * n * sizeof(std::uint32_t);
        if (shared_bytes > shared_limit) continue;
        const Rank support_count = rt.choose[outer_bits][o];
        if (!support_count) continue;
        const unsigned grid = static_cast<unsigned>(std::min<Rank>(support_count, 65535));
        ck(cudaFuncSetAttribute(
               two_cell_fusion2_bucket_kernel,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               static_cast<int>(shared_bytes)),
           "bucket optin shared");
        two_cell_fusion2_bucket_kernel<<<grid, threads, shared_bytes>>>(
            d_input, d_output, d_owner, W, start, o, support_count, states, mod,
            d_blocks, d_loads, d_stores, d_error);
        ck(cudaGetLastError(), "bucket launch");
    }
    ck(cudaDeviceSynchronize(), "bucket sync");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    std::vector<unsigned long long> owner(static_cast<std::size_t>(states));
    unsigned long long processed = 0, loads = 0, stores = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_output, states * sizeof(std::uint32_t),
                  cudaMemcpyDeviceToHost), "bucket copy output");
    ck(cudaMemcpy(owner.data(), d_owner, states * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost), "bucket copy owner");
    ck(cudaMemcpy(&processed, d_blocks, sizeof(processed), cudaMemcpyDeviceToHost),
       "bucket copy blocks");
    ck(cudaMemcpy(&loads, d_loads, sizeof(loads), cudaMemcpyDeviceToHost),
       "bucket copy loads");
    ck(cudaMemcpy(&stores, d_stores, sizeof(stores), cudaMemcpyDeviceToHost),
       "bucket copy stores");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "bucket copy error");

    if (error || processed != plan.fitted_blocks ||
        loads != plan.fitted_states || stores != plan.fitted_states) {
        std::cerr << "FAIL bucket counters W=" << W << " start=" << start
                  << " error=" << error
                  << " processed=" << processed << "/" << plan.fitted_blocks
                  << " loads=" << loads << "/" << plan.fitted_states
                  << " stores=" << stores << "/" << plan.fitted_states << '\n';
        std::exit(240);
    }
    for (Rank r = 0; r < states; ++r) {
        const bool fused = owner[static_cast<std::size_t>(r)] != ~0ULL;
        if (fused && output[static_cast<std::size_t>(r)] !=
                         full_reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL bucket arithmetic W=" << W << " start=" << start
                      << " rank=" << r << '\n';
            std::exit(241);
        }
    }

    std::cout << "two-cell-fusion2-bucketed-fused-only"
              << " W=" << W
              << " start=" << start
              << " shared_limit=" << shared_limit
              << " fused_states=" << plan.fitted_states
              << " fallback_states=" << plan.fallback_states
              << " global_loads=" << loads
              << " global_stores=" << stores
              << " fused_arithmetic=OK\n";

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_owner);
    cudaFree(d_blocks);
    cudaFree(d_loads);
    cudaFree(d_stores);
    cudaFree(d_error);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 8;
    const Rank requested_kib = argc > 2
        ? static_cast<Rank>(std::strtoull(argv[2], nullptr, 10)) : 96ULL;
    const int threads = argc > 3 ? std::atoi(argv[3]) : 256;
    const std::uint32_t mod = argc > 4
        ? static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 10))
        : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth || threads < 32 || threads > 1024 || mod < 3)
        return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    const Rank requested_bytes = requested_kib * 1024ULL;
    if (plan_only) {
        print_bucket_plan(W, requested_bytes, rt);
        if (W == 28) {
            for (Rank kib : {64ULL, 96ULL, 128ULL, 160ULL, 192ULL, 228ULL, 256ULL})
                print_bucket_plan(W, kib * 1024ULL, rt);
        }
        return 0;
    }

    if (W > 10) {
        std::cerr << "execution mode intentionally limited to W<=10; use --plan-only above that\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "bucket device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "bucket set device");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "bucket device props");
    const Rank shared_limit = std::min<Rank>(
        requested_bytes, static_cast<Rank>(prop.sharedMemPerBlockOptin));
    install_tables(rt);
    install_stationary_tables(st);

    for (int start = 0; start + 2 <= W - 3; ++start)
        run_bucketed_fused_only(W, start, threads, shared_limit, rt, st, mod);
    std::cout << "ALL_OK two_cell_fusion2_bucketed=1 W=" << W << '\n';
    return 0;
}
