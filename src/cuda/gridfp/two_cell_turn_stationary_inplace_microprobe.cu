#pragma push_macro("main")
#undef main
#define main two_cell_component_stationary_lifting_microprobe_main_unused
#include "two_cell_component_stationary_lifting_microprobe.cu"
#pragma pop_macro("main")

#include "../../common/two_cell_turn_closed_device.cuh"

namespace {

__device__ __forceinline__ std::uint32_t double_mod_u32(
    std::uint32_t x, std::uint32_t mod
) {
    const unsigned long long z = 2ULL * static_cast<unsigned long long>(x);
    return static_cast<std::uint32_t>(z >= mod ? z - mod : z);
}

__global__ void two_cell_turn_stationary_kernel(
    std::uint32_t* __restrict__ values,
    unsigned long long* __restrict__ owner,
    const std::uint32_t* __restrict__ primitive_lut,
    Rank supports,
    Rank primitive_count,
    Rank state_count,
    int occupied_count,
    int W,
    int direction, // +1 right turn, -1 left turn
    std::uint32_t mod,
    unsigned long long* processed,
    unsigned long long* primitive_scans,
    unsigned long long* add_ops,
    int* error
) {
    __shared__ PackedKey sh_state[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_value[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ Rank sh_rank[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_support[WARPS_PER_BLOCK];
    __shared__ int sh_ns[WARPS_PER_BLOCK];
    __shared__ int sh_singular[WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank chunks = (primitive_count + PRIMITIVE_CHUNK - 1) / PRIMITIVE_CHUNK;
    const Rank tiles = supports * chunks;
    const Rank first = Rank(blockIdx.x) * WARPS_PER_BLOCK + Rank(warp);
    const Rank stride = Rank(gridDim.x) * WARPS_PER_BLOCK;
    const int label_len = W - 2;
    const int active = direction > 0 ? W - 3 : 0;

    for (Rank tile = first; tile < tiles; tile += stride) {
        const Rank support_rank = tile / chunks;
        const Rank chunk = tile - support_rank * chunks;
        const Rank primitive_begin = chunk * PRIMITIVE_CHUNK;
        Rank primitive_end = primitive_begin + PRIMITIVE_CHUNK;
        if (primitive_end > primitive_count) primitive_end = primitive_count;

        if (lane == 0) {
            sh_support[warp] = oneesan::twocell::support_unrank(
                label_len, occupied_count, support_rank, TC_RANK_TABLES);
        }
        __syncwarp();

        const std::uint32_t support = sh_support[warp];
        for (Rank pr = primitive_begin; pr < primitive_end; ++pr) {
            std::uint32_t compact_left = lane == 0 ? primitive_lut[pr] : 0;
            compact_left = __shfl_sync(0xffffffffu, compact_left, 0);
            const std::uint32_t left = deposit_left_warp(support, compact_left, label_len);

            if (lane == 0) {
                const PackedWord label{support, left, static_cast<std::uint8_t>(label_len)};
                const auto block = direction > 0
                    ? oneesan::twocell::right_turn_closed_block(label, W)
                    : oneesan::twocell::left_turn_closed_block(label, W);
                sh_ns[warp] = 0;
                sh_singular[warp] = block.singular;
                if (block.overflow || block.size < 3 || block.size > MAX_STATES) {
                    set_error(error, 201);
                } else {
                    sh_ns[warp] = block.size;
                    for (int q = 0; q < block.size; ++q)
                        sh_state[warp][q] = block.state[q];
                    atomicAdd(processed, 1ULL);
                }
            }
            __syncwarp();

            const int ns = sh_ns[warp];
            if (lane < ns) {
                const PackedKey s = sh_state[warp][lane];
                Rank primitive = pr;
                bool reuse = false;
                if (direction > 0) {
                    // R-terminated singular components only insert vacancies,
                    // so all three coordinates retain the label primitive word.
                    // For N-terminated components beta=vNN retains it as well.
                    reuse = sh_singular[warp] || (!sh_singular[warp] && lane == 1);
                }
                if (!reuse) {
                    const int len = s.type ? W - 2 : W - 1;
                    primitive = oneesan::twocell::primitive_rank(
                        s.support, s.left, len, TC_RANK_TABLES);
                    atomicAdd(primitive_scans, 1ULL);
                }
                const Rank r = oneesan::twocell::stationary_rank_with_primitive(
                    s, W, active, primitive, TC_RANK_TABLES, TC_STATIONARY_TABLES);
                sh_rank[warp][lane] = r;
                if (r >= state_count) {
                    sh_value[warp][lane] = 0;
                    set_error(error, 202);
                } else {
                    sh_value[warp][lane] = values[r];
                }
            }
            __syncwarp();

            if (lane == 0 && ns > 0) {
                if (sh_singular[warp]) {
                    if (ns != 3) {
                        set_error(error, 203);
                    } else {
                        const std::uint32_t t0 = add_mod_u32(
                            sh_value[warp][0], sh_value[warp][2], mod);
                        const std::uint32_t t1 = add_mod_u32(
                            sh_value[warp][1], sh_value[warp][2], mod);
                        sh_value[warp][0] = double_mod_u32(t0, mod);
                        sh_value[warp][1] = t1;
                        sh_value[warp][2] = t1;
                        atomicAdd(add_ops, 2ULL);
                    }
                } else {
                    std::uint32_t t = sh_value[warp][1];
                    for (int q = 2; q < ns; ++q)
                        t = add_mod_u32(t, sh_value[warp][q], mod);
                    sh_value[warp][0] = add_mod_u32(
                        double_mod_u32(sh_value[warp][0], mod), t, mod);
                    sh_value[warp][1] = t;
                    for (int q = 2; q < ns; ++q)
                        sh_value[warp][q] = double_mod_u32(sh_value[warp][q], mod);
                    atomicAdd(add_ops, static_cast<unsigned long long>(ns - 1));
                }
            }
            __syncwarp();

            if (lane < ns) {
                const Rank r = sh_rank[warp][lane];
                const unsigned long long component_id =
                    (static_cast<unsigned long long>(support_rank) << 32) |
                    static_cast<unsigned long long>(pr);
                const unsigned long long empty = ~0ULL;
                const unsigned long long previous = atomicCAS(owner + r, empty, component_id);
                if (previous != empty && previous != component_id)
                    set_error(error, 204);
                values[r] = sh_value[warp][lane];
            }
            __syncwarp();
        }
    }
}

void run_turn_direction(
    int W,
    int direction,
    const RankTables& rt,
    const StationaryRankTables& st,
    const PrimitiveLut& host_lut,
    std::uint32_t mod,
    unsigned requested_blocks
) {
    const Rank states = st.total[W];
    const Rank components = oneesan::twocell::component_label_count(W, rt);
    const int active = direction > 0 ? W - 3 : 0;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(W + 1));
    for (int n = 1; n <= W; ++n) words[n] = gen_words(n);
    const auto basis = direction > 0
        ? q_basis(W, W - 3, words)
        : reverse_q_basis(W, 1, words);
    if (basis.size() != states) std::exit(205);

    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    std::vector<std::uint32_t> reference(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + Rank(direction > 0 ? 31 : 47)) % (mod - 1ULL)));

    for (const Key& s : basis) {
        const Rank sr = oneesan::twocell::stationary_rank(
            device_key(s), W, active, rt, st);
        const std::uint32_t x = input[static_cast<std::size_t>(sr)];
        const CVec col = direction > 0 ? turn_right_basis(s, W) : turn_left_basis(s, W);
        for (const auto& [d, c] : col) {
            const Rank dr = oneesan::twocell::stationary_rank(
                device_key(d), W, active, rt, st);
            const unsigned long long z = static_cast<unsigned long long>(reference[dr]) +
                                         static_cast<unsigned long long>(c) * x;
            reference[dr] = static_cast<std::uint32_t>(z % mod);
        }
    }

    std::uint32_t* d_values = nullptr;
    std::uint32_t* d_lut = nullptr;
    unsigned long long* d_owner = nullptr;
    unsigned long long* d_processed = nullptr;
    unsigned long long* d_primitive_scans = nullptr;
    unsigned long long* d_add_ops = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)), "turn alloc values");
    ck(cudaMalloc(&d_lut, host_lut.value.size() * sizeof(std::uint32_t)), "turn alloc lut");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "turn alloc owner");
    ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "turn alloc processed");
    ck(cudaMalloc(&d_primitive_scans, sizeof(unsigned long long)), "turn alloc primitive scans");
    ck(cudaMalloc(&d_add_ops, sizeof(unsigned long long)), "turn alloc adds");
    ck(cudaMalloc(&d_error, sizeof(int)), "turn alloc error");
    ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "turn copy input");
    ck(cudaMemcpy(d_lut, host_lut.value.data(),
                  host_lut.value.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "turn copy lut");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "turn clear owner");
    ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "turn zero processed");
    ck(cudaMemset(d_primitive_scans, 0, sizeof(unsigned long long)), "turn zero primitive scans");
    ck(cudaMemset(d_add_ops, 0, sizeof(unsigned long long)), "turn zero adds");
    ck(cudaMemset(d_error, 0, sizeof(int)), "turn zero error");

    cudaEvent_t begin{}, end{};
    ck(cudaEventCreate(&begin), "turn event begin");
    ck(cudaEventCreate(&end), "turn event end");
    ck(cudaEventRecord(begin), "turn record begin");
    for (int occupied = 1; occupied <= W - 2; occupied += 2) {
        const Rank supports = rt.choose[W - 2][occupied];
        const Rank pc = rt.primitive[occupied][1];
        const Rank tiles = sector_tiles(W, occupied, rt);
        const Rank one_pass_blocks = (tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned blocks = static_cast<unsigned>(std::max<Rank>(
            1, std::min<Rank>(requested_blocks, one_pass_blocks)));
        two_cell_turn_stationary_kernel<<<blocks, THREADS>>>(
            d_values, d_owner, d_lut + host_lut.offset[occupied],
            supports, pc, states, occupied, W, direction, mod,
            d_processed, d_primitive_scans, d_add_ops, d_error);
        ck(cudaGetLastError(), "turn launch sector");
    }
    ck(cudaEventRecord(end), "turn record end");
    ck(cudaEventSynchronize(end), "turn sync end");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, begin, end), "turn elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    std::vector<unsigned long long> owner(static_cast<std::size_t>(states));
    unsigned long long processed = 0, primitive_scans = 0, add_ops = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
       "turn copy output");
    ck(cudaMemcpy(owner.data(), d_owner, states * sizeof(unsigned long long), cudaMemcpyDeviceToHost),
       "turn copy owner");
    ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost),
       "turn copy processed");
    ck(cudaMemcpy(&primitive_scans, d_primitive_scans, sizeof(primitive_scans), cudaMemcpyDeviceToHost),
       "turn copy primitive scans");
    ck(cudaMemcpy(&add_ops, d_add_ops, sizeof(add_ops), cudaMemcpyDeviceToHost),
       "turn copy adds");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "turn copy error");

    if (error || processed != components || add_ops != states - components) {
        std::cerr << "FAIL turn stationary error=" << error
                  << " processed=" << processed << " expected=" << components
                  << " add_ops=" << add_ops << " expected_adds=" << states - components << '\n';
        std::exit(206);
    }
    for (Rank r = 0; r < states; ++r) {
        if (owner[static_cast<std::size_t>(r)] == std::numeric_limits<unsigned long long>::max() ||
            output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL turn stationary arithmetic W=" << W
                      << " direction=" << direction << " rank=" << r << '\n';
            std::exit(207);
        }
    }

    std::cout << "two-cell-turn-stationary-inplace-microprobe"
              << " W=" << W
              << " direction=" << (direction > 0 ? "right" : "left")
              << " states=" << states
              << " components=" << components
              << " add_ops=" << add_ops
              << " primitive_scans=" << primitive_scans
              << " one_buffer=1 destination_rank_change=0 component_graph=0"
              << " kernel_ms=" << ms
              << " arithmetic=OK\n";

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(d_values);
    cudaFree(d_lut);
    cudaFree(d_owner);
    cudaFree(d_processed);
    cudaFree(d_primitive_scans);
    cudaFree(d_add_ops);
    cudaFree(d_error);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 9;
    const unsigned blocks = argc > 2
        ? static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10)) : 4096u;
    const std::uint32_t mod = argc > 3
        ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 5 || W > oneesan::twocell::kMaxWidth || blocks == 0 || mod < 3) return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    const Rank states = st.total[W];
    const Rank components = oneesan::twocell::component_label_count(W, rt);
    if (plan_only) {
        std::cout << "two-cell-turn-stationary-inplace-plan"
                  << " W=" << W
                  << " states=" << states
                  << " components=" << components
                  << " add_ops=" << states - components
                  << " one_u32_vector_GiB="
                  << double(states * 4ULL) / double(1ULL << 30)
                  << " max_turn_states=" << oneesan::twocell::kMaxTurnStates
                  << " right_left_same_executor=1"
                  << " destination_vector_bytes=0 component_graph_bytes=0"
                  << "\n";
        return 0;
    }

    if (W > 10) {
        std::cerr << "execution mode intentionally limited to W<=10; use --plan-only above that\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "turn device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "turn set device");
    install_tables(rt);
    install_stationary_tables(st);
    const PrimitiveLut lut = build_primitive_lut(W, rt);

    run_turn_direction(W, +1, rt, st, lut, mod, blocks);
    run_turn_direction(W, -1, rt, st, lut, mod, blocks);
    std::cout << "ALL_OK two_cell_turn_stationary_inplace_cuda=1 W=" << W << '\n';
    return 0;
}
