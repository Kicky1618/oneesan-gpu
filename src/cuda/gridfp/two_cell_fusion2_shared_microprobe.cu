#pragma push_macro("main")
#undef main
#define main two_cell_component_stationary_lifting_microprobe_main_unused
#include "two_cell_component_stationary_lifting_microprobe.cu"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_unrank.hpp"

namespace {

constexpr int FUSION_STEPS = 2;

__global__ void two_cell_fusion2_shared_kernel(
    const std::uint32_t* __restrict__ input,
    std::uint32_t* __restrict__ output,
    unsigned long long* __restrict__ owner,
    int W,
    int start,
    Rank state_count,
    Rank shared_stride,
    std::uint32_t mod,
    unsigned long long* processed_blocks,
    unsigned long long* global_loads,
    unsigned long long* global_stores,
    int* error
) {
    extern __shared__ std::uint32_t shared_values[];
    std::uint32_t* cur = shared_values;
    std::uint32_t* next = shared_values + shared_stride;

    const int outer_bits = W - FUSION_STEPS - 3;
    const std::uint32_t outer = static_cast<std::uint32_t>(blockIdx.x);
    if (outer >= (std::uint32_t(1) << outer_bits)) return;
    const int outer_ones = oneesan::twocell::popcount32(outer);
    const Rank n = oneesan::twocell::fusion_block_size(
        FUSION_STEPS, outer_ones, TC_RANK_TABLES);
    if (n > shared_stride) {
        if (threadIdx.x == 0) set_error(error, 211);
        return;
    }

    for (Rank r = threadIdx.x; r < n; r += blockDim.x) {
        const auto d = oneesan::twocell::fusion_local_unrank_at(
            r, outer, W, start, FUSION_STEPS, start, TC_RANK_TABLES);
        if (!d.valid) {
            set_error(error, 212);
            continue;
        }
        const Rank gr = oneesan::twocell::stationary_rank_with_primitive(
            d.key, W, start, d.primitive,
            TC_RANK_TABLES, TC_STATIONARY_TABLES);
        if (gr >= state_count) {
            set_error(error, 213);
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
                set_error(error, 214);
                continue;
            }

            const auto pre = oneesan::twocell::inverse_K(dst.key, W, active);
            if (pre.overflow) {
                set_error(error, 215);
                continue;
            }
            unsigned long long sum = 0;
            for (int q = 0; q < pre.size; ++q) {
                const PackedKey src = pre.value[q];
                const std::uint32_t src_outer = oneesan::twocell::fusion_outer_mask_at(
                    src, start, FUSION_STEPS, active);
                if (src_outer != outer) {
                    set_error(error, 216);
                    continue;
                }
                const Rank sr = oneesan::twocell::fusion_local_rank_at(
                    src, W, start, FUSION_STEPS, active, outer_ones,
                    TC_RANK_TABLES);
                if (sr >= n) {
                    set_error(error, 217);
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
            set_error(error, 218);
            continue;
        }
        const Rank gr = oneesan::twocell::stationary_rank_with_primitive(
            d.key, W, start + FUSION_STEPS, d.primitive,
            TC_RANK_TABLES, TC_STATIONARY_TABLES);
        if (gr >= state_count) {
            set_error(error, 219);
            continue;
        }
        const unsigned long long previous = atomicCAS(
            owner + gr, ~0ULL, static_cast<unsigned long long>(outer));
        if (previous != ~0ULL && previous != static_cast<unsigned long long>(outer))
            set_error(error, 220);
        output[gr] = cur[r];
        atomicAdd(global_stores, 1ULL);
    }
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(processed_blocks, 1ULL);
}

std::uint32_t add_ref_mod(
    std::uint32_t a, std::uint32_t b, std::uint32_t mod
) {
    return static_cast<std::uint32_t>(
        (static_cast<unsigned long long>(a) + b) % mod);
}

std::vector<std::uint32_t> fusion2_reference(
    const std::vector<std::uint32_t>& input,
    int W,
    int start,
    const RankTables& rt,
    const StationaryRankTables& st,
    std::uint32_t mod
) {
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(W + 1));
    for (int n = 1; n <= W; ++n) words[n] = gen_words(n);
    std::vector<std::uint32_t> cur = input;
    std::vector<std::uint32_t> next(input.size());
    for (int active = start; active < start + FUSION_STEPS; ++active) {
        std::fill(next.begin(), next.end(), 0);
        for (const Key& s : q_basis(W, active, words)) {
            const Rank sr = oneesan::twocell::stationary_rank(
                device_key(s), W, active, rt, st);
            const std::uint32_t x = cur[static_cast<std::size_t>(sr)];
            for (const auto& [d, c] : K_basis(s, W, active)) {
                if (c != 1) std::exit(221);
                const Rank dr = oneesan::twocell::stationary_rank(
                    device_key(d), W, active + 1, rt, st);
                next[static_cast<std::size_t>(dr)] = add_ref_mod(
                    next[static_cast<std::size_t>(dr)], x, mod);
            }
        }
        cur.swap(next);
    }
    return cur;
}

Rank max_fusion2_block(int W, const RankTables& rt) {
    const int outer_bits = W - 5;
    Rank z = 0;
    for (int o = 0; o <= outer_bits; ++o)
        z = std::max(z, oneesan::twocell::fusion_block_size(2, o, rt));
    return z;
}

void print_capacity_plan(int W, Rank bytes, const RankTables& rt) {
    const int outer_bits = W - 5;
    Rank fit_states = 0, total_states = 0, fit_blocks = 0, total_blocks = 0;
    int max_o = -1;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank blocks = rt.choose[outer_bits][o];
        const Rank n = oneesan::twocell::fusion_block_size(2, o, rt);
        total_blocks += blocks;
        total_states += blocks * n;
        if (2 * n * sizeof(std::uint32_t) <= bytes) {
            fit_blocks += blocks;
            fit_states += blocks * n;
            max_o = o;
        }
    }
    const double f = total_states ? double(fit_states) / double(total_states) : 0.0;
    std::cout << "fusion2_double_shared_capacity bytes=" << bytes
              << " max_outer_ones=" << max_o
              << " fit_block_fraction=" << double(fit_blocks) / double(total_blocks)
              << " fit_state_fraction=" << f
              << " HBM_reduction_vs_two_pass=" << 0.5 * f
              << "\n";
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 8;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 256;
    const std::uint32_t mod = argc > 3
        ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10))
        : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth || threads < 32 || threads > 1024 || mod < 3)
        return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    const Rank states = st.total[W];
    const int outer_bits = W - 5;
    const Rank blocks = Rank(1) << outer_bits;
    const Rank max_block = max_fusion2_block(W, rt);
    const Rank shared_bytes = 2 * max_block * sizeof(std::uint32_t);

    if (plan_only) {
        std::cout << "two-cell-fusion2-shared-plan"
                  << " W=" << W
                  << " states=" << states
                  << " outer_blocks=" << blocks
                  << " max_block_states=" << max_block
                  << " double_shared_bytes=" << shared_bytes
                  << " ideal_all_fused_HBM_ratio=0.5"
                  << " global_intermediate_values=0"
                  << "\n";
        if (W == 28) {
            for (Rank kib : {64ULL, 128ULL, 192ULL, 228ULL, 256ULL})
                print_capacity_plan(W, kib * 1024ULL, rt);
        }
        return 0;
    }

    if (W > 10) {
        std::cerr << "execution mode intentionally limited to W<=10; use --plan-only above that\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "fusion2 device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "fusion2 set device");
    install_tables(rt);
    install_stationary_tables(st);

    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "fusion2 device props");
    if (shared_bytes > static_cast<Rank>(prop.sharedMemPerBlockOptin)) {
        std::cerr << "fusion2 requires " << shared_bytes
                  << " shared bytes, device opt-in limit=" << prop.sharedMemPerBlockOptin << '\n';
        return 5;
    }
    ck(cudaFuncSetAttribute(
           two_cell_fusion2_shared_kernel,
           cudaFuncAttributeMaxDynamicSharedMemorySize,
           static_cast<int>(shared_bytes)),
       "fusion2 optin shared");

    for (int start = 0; start + 2 <= W - 3; ++start) {
        std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
        for (Rank r = 0; r < states; ++r)
            input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
                1 + ((r * 2654435761ULL + Rank(start) * 101ULL) % (mod - 1ULL)));
        const auto reference = fusion2_reference(input, W, start, rt, st, mod);

        std::uint32_t* d_input = nullptr;
        std::uint32_t* d_output = nullptr;
        unsigned long long* d_owner = nullptr;
        unsigned long long* d_blocks = nullptr;
        unsigned long long* d_loads = nullptr;
        unsigned long long* d_stores = nullptr;
        int* d_error = nullptr;
        ck(cudaMalloc(&d_input, states * sizeof(std::uint32_t)), "fusion2 alloc input");
        ck(cudaMalloc(&d_output, states * sizeof(std::uint32_t)), "fusion2 alloc output");
        ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "fusion2 alloc owner");
        ck(cudaMalloc(&d_blocks, sizeof(unsigned long long)), "fusion2 alloc blocks");
        ck(cudaMalloc(&d_loads, sizeof(unsigned long long)), "fusion2 alloc loads");
        ck(cudaMalloc(&d_stores, sizeof(unsigned long long)), "fusion2 alloc stores");
        ck(cudaMalloc(&d_error, sizeof(int)), "fusion2 alloc error");
        ck(cudaMemcpy(d_input, input.data(), states * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice), "fusion2 copy input");
        ck(cudaMemset(d_output, 0, states * sizeof(std::uint32_t)), "fusion2 zero output");
        ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "fusion2 clear owner");
        ck(cudaMemset(d_blocks, 0, sizeof(unsigned long long)), "fusion2 zero blocks");
        ck(cudaMemset(d_loads, 0, sizeof(unsigned long long)), "fusion2 zero loads");
        ck(cudaMemset(d_stores, 0, sizeof(unsigned long long)), "fusion2 zero stores");
        ck(cudaMemset(d_error, 0, sizeof(int)), "fusion2 zero error");

        cudaEvent_t begin{}, end{};
        ck(cudaEventCreate(&begin), "fusion2 event begin");
        ck(cudaEventCreate(&end), "fusion2 event end");
        ck(cudaEventRecord(begin), "fusion2 record begin");
        two_cell_fusion2_shared_kernel<<<static_cast<unsigned>(blocks), threads, shared_bytes>>>(
            d_input, d_output, d_owner, W, start, states, max_block, mod,
            d_blocks, d_loads, d_stores, d_error);
        ck(cudaGetLastError(), "fusion2 launch");
        ck(cudaEventRecord(end), "fusion2 record end");
        ck(cudaEventSynchronize(end), "fusion2 sync");
        float ms = 0.0f;
        ck(cudaEventElapsedTime(&ms, begin, end), "fusion2 elapsed");

        std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
        unsigned long long processed = 0, loads = 0, stores = 0;
        int error = 0;
        ck(cudaMemcpy(output.data(), d_output, states * sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost), "fusion2 copy output");
        ck(cudaMemcpy(&processed, d_blocks, sizeof(processed), cudaMemcpyDeviceToHost),
           "fusion2 copy blocks");
        ck(cudaMemcpy(&loads, d_loads, sizeof(loads), cudaMemcpyDeviceToHost),
           "fusion2 copy loads");
        ck(cudaMemcpy(&stores, d_stores, sizeof(stores), cudaMemcpyDeviceToHost),
           "fusion2 copy stores");
        ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
           "fusion2 copy error");

        if (error || processed != blocks || loads != states || stores != states ||
            output != reference) {
            std::cerr << "FAIL fusion2 W=" << W << " start=" << start
                      << " error=" << error << " blocks=" << processed
                      << " loads=" << loads << " stores=" << stores << '\n';
            return 6;
        }

        std::cout << "two-cell-fusion2-shared-microprobe"
                  << " W=" << W
                  << " start=" << start
                  << " states=" << states
                  << " outer_blocks=" << blocks
                  << " max_block_states=" << max_block
                  << " shared_bytes=" << shared_bytes
                  << " global_loads=" << loads
                  << " global_stores=" << stores
                  << " baseline_two_pass_global_values=" << (4 * states)
                  << " fused_global_values=" << (2 * states)
                  << " HBM_value_traffic_ratio=0.5"
                  << " kernel_ms=" << ms
                  << " arithmetic=OK\n";

        cudaEventDestroy(begin);
        cudaEventDestroy(end);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_owner);
        cudaFree(d_blocks);
        cudaFree(d_loads);
        cudaFree(d_stores);
        cudaFree(d_error);
    }
    std::cout << "ALL_OK two_cell_fusion2_shared_cuda=1 W=" << W << '\n';
    return 0;
}
