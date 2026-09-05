#pragma push_macro("main")
#undef main
#define main two_cell_component_stationary_tiledbase_microprobe_main_unused
#include "two_cell_component_stationary_tiledbase_microprobe.cu"
#pragma pop_macro("main")

#include "../../common/two_cell_component_matching.cuh"

namespace {

__device__ __forceinline__ std::uint32_t add_mod_u32(
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t mod
) {
    const unsigned long long z =
        static_cast<unsigned long long>(a) + static_cast<unsigned long long>(b);
    return static_cast<std::uint32_t>(z >= mod ? z - mod : z);
}

__global__ void two_cell_stationary_lifting_kernel(
    std::uint32_t* __restrict__ values,
    unsigned long long* __restrict__ owner,
    const std::uint32_t* __restrict__ primitive_lut,
    Rank supports,
    Rank primitive_count,
    Rank state_count,
    int occupied_count,
    int W,
    int i,
    std::uint32_t mod,
    unsigned long long* processed,
    unsigned long long* primitive_scans,
    unsigned long long* support_base_builds,
    unsigned long long* residual_adds,
    unsigned long long* nonidentity_matchings,
    int* error
) {
    __shared__ PackedKey sh_src[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_value[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_output[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ Rank sh_rank[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_support[WARPS_PER_BLOCK];
    __shared__ oneesan::twocell::StationaryComponentBases sh_bases[WARPS_PER_BLOCK];
    __shared__ oneesan::twocell::ComponentMatching sh_matching[WARPS_PER_BLOCK];
    __shared__ int sh_ns[WARPS_PER_BLOCK];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const Rank chunks = (primitive_count + PRIMITIVE_CHUNK - 1) / PRIMITIVE_CHUNK;
    const Rank tiles = supports * chunks;
    const Rank first = Rank(blockIdx.x) * WARPS_PER_BLOCK + Rank(warp);
    const Rank stride = Rank(gridDim.x) * WARPS_PER_BLOCK;
    const int label_len = W - 2;

    for (Rank tile = first; tile < tiles; tile += stride) {
        const Rank support_rank = tile / chunks;
        const Rank chunk = tile - support_rank * chunks;
        const Rank primitive_begin = chunk * PRIMITIVE_CHUNK;
        Rank primitive_end = primitive_begin + PRIMITIVE_CHUNK;
        if (primitive_end > primitive_count) primitive_end = primitive_count;

        if (lane == 0) {
            const std::uint32_t support = oneesan::twocell::support_unrank(
                label_len, occupied_count, support_rank, TC_RANK_TABLES);
            sh_support[warp] = support;
            sh_bases[warp] = oneesan::twocell::stationary_component_bases(
                support, W, i, TC_RANK_TABLES, TC_STATIONARY_TABLES);
            atomicAdd(support_base_builds,
                      static_cast<unsigned long long>(sh_bases[warp].count));
        }
        __syncwarp();

        const std::uint32_t support = sh_support[warp];
        const auto bases = sh_bases[warp];
        for (Rank pr = primitive_begin; pr < primitive_end; ++pr) {
            std::uint32_t compact_left = lane == 0 ? primitive_lut[pr] : 0;
            compact_left = __shfl_sync(0xffffffffu, compact_left, 0);
            const std::uint32_t left = deposit_left_warp(support, compact_left, label_len);

            if (lane == 0) {
                const PackedWord label{support, left, static_cast<std::uint8_t>(label_len)};
                const auto src = oneesan::twocell::direct_component_sources(label, W, i);
                sh_ns[warp] = 0;
                if (src.overflow || src.size <= 0 || src.size > MAX_STATES) {
                    set_error(error, 181);
                } else {
                    sh_ns[warp] = src.size;
                    for (int s = 0; s < src.size; ++s) sh_src[warp][s] = src.value[s];
                    const auto matching = oneesan::twocell::build_component_matching(
                        sh_src[warp], src.size, W, i);
                    if (!matching.ok) {
                        set_error(error, 183);
                    } else {
                        sh_matching[warp] = matching;
                        bool nonidentity = false;
                        for (int s = 0; s < src.size; ++s)
                            nonidentity |= matching.src_to_dst[s] != s;
                        if (nonidentity) atomicAdd(nonidentity_matchings, 1ULL);
                        atomicAdd(processed, 1ULL);
                    }
                }
            }
            __syncwarp();

            const int ns = sh_ns[warp];
            if (lane < ns) {
                const PackedKey source = sh_src[warp][lane];
                Rank primitive = pr;
                const bool reuse_label = bases.retained &&
                    (lane < 3 || (lane == 3 && bases.xN_deep));
                if (!reuse_label) {
                    const int len = source.type ? W - 2 : W - 1;
                    primitive = oneesan::twocell::primitive_rank(
                        source.support, source.left, len, TC_RANK_TABLES);
                    atomicAdd(primitive_scans, 1ULL);
                }

                const Rank r = oneesan::twocell::stationary_component_source_base(
                    bases, lane) + primitive;
                sh_rank[warp][lane] = r;
                if (r >= state_count) {
                    sh_value[warp][lane] = 0;
                    set_error(error, 182);
                } else {
                    sh_value[warp][lane] = values[r];
                }
            }
            __syncwarp();

            if (lane == 0 && ns > 0) {
                const auto matching = sh_matching[warp];
                if (!oneesan::twocell::apply_component_matching(
                        matching, sh_value[warp], sh_output[warp], mod)) {
                    set_error(error, 184);
                } else {
                    atomicAdd(residual_adds,
                              static_cast<unsigned long long>(matching.residual_edges));
                }
            }
            __syncwarp();

            // Destination coordinate index t is recouple(src[t]); stationary
            // source and destination ranks are identical, so the result for
            // destination index t is written to the already-computed source
            // address sh_rank[t]. The matrix matching itself may be nonidentity.
            if (lane < ns) {
                const Rank r = sh_rank[warp][lane];
                const unsigned long long component_id =
                    (static_cast<unsigned long long>(support_rank) << 32) |
                    static_cast<unsigned long long>(pr);
                const unsigned long long empty = ~0ULL;
                const unsigned long long previous = atomicCAS(
                    owner + r, empty, component_id);
                if (previous != empty && previous != component_id)
                    set_error(error, 185);
                values[r] = sh_output[warp][lane];
            }
            __syncwarp();
        }
    }
}

void run_stationary_lifting_position(
    int W,
    int i,
    const RankTables& rt,
    const StationaryRankTables& st,
    const PrimitiveLut& host_lut,
    std::uint32_t mod,
    unsigned requested_blocks
) {
    const Rank states = st.total[W];
    const Rank components = oneesan::twocell::component_label_count(W, rt);

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(W + 1));
    for (int n = 1; n <= W; ++n) words[n] = gen_words(n);
    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    std::vector<std::uint32_t> reference(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + Rank(i) * 17ULL) % (mod - 1ULL)));
    for (const Key& s : q_basis(W, i, words)) {
        const Rank sr = oneesan::twocell::stationary_rank(
            device_key(s), W, i, rt, st);
        const std::uint32_t value = input[static_cast<std::size_t>(sr)];
        for (const auto& [d, c] : K_basis(s, W, i)) {
            if (c != 1) std::exit(189);
            const Rank dr = oneesan::twocell::stationary_rank(
                device_key(d), W, i + 1, rt, st);
            reference[static_cast<std::size_t>(dr)] =
                add_mod(reference[static_cast<std::size_t>(dr)], value, mod);
        }
    }

    std::uint32_t* d_values = nullptr;
    std::uint32_t* d_lut = nullptr;
    unsigned long long* d_owner = nullptr;
    unsigned long long* d_processed = nullptr;
    unsigned long long* d_primitive_scans = nullptr;
    unsigned long long* d_support_base_builds = nullptr;
    unsigned long long* d_residual_adds = nullptr;
    unsigned long long* d_nonidentity = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)), "lifting alloc values");
    ck(cudaMalloc(&d_lut, host_lut.value.size() * sizeof(std::uint32_t)), "lifting alloc lut");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "lifting alloc owner");
    ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "lifting alloc processed");
    ck(cudaMalloc(&d_primitive_scans, sizeof(unsigned long long)), "lifting alloc primitive scans");
    ck(cudaMalloc(&d_support_base_builds, sizeof(unsigned long long)), "lifting alloc support builds");
    ck(cudaMalloc(&d_residual_adds, sizeof(unsigned long long)), "lifting alloc residual adds");
    ck(cudaMalloc(&d_nonidentity, sizeof(unsigned long long)), "lifting alloc nonidentity");
    ck(cudaMalloc(&d_error, sizeof(int)), "lifting alloc error");
    ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "lifting copy input");
    ck(cudaMemcpy(d_lut, host_lut.value.data(),
                  host_lut.value.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "lifting copy lut");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "lifting clear owner");
    ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "lifting zero processed");
    ck(cudaMemset(d_primitive_scans, 0, sizeof(unsigned long long)), "lifting zero primitive scans");
    ck(cudaMemset(d_support_base_builds, 0, sizeof(unsigned long long)), "lifting zero support builds");
    ck(cudaMemset(d_residual_adds, 0, sizeof(unsigned long long)), "lifting zero residual adds");
    ck(cudaMemset(d_nonidentity, 0, sizeof(unsigned long long)), "lifting zero nonidentity");
    ck(cudaMemset(d_error, 0, sizeof(int)), "lifting zero error");

    cudaEvent_t begin{}, end{};
    ck(cudaEventCreate(&begin), "lifting event begin");
    ck(cudaEventCreate(&end), "lifting event end");
    ck(cudaEventRecord(begin), "lifting record begin");
    for (int occupied = 1; occupied <= W - 2; occupied += 2) {
        const Rank supports = rt.choose[W - 2][occupied];
        const Rank pc = rt.primitive[occupied][1];
        const Rank tiles = sector_tiles(W, occupied, rt);
        const Rank one_pass_blocks = (tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned blocks = static_cast<unsigned>(std::max<Rank>(
            1, std::min<Rank>(requested_blocks, one_pass_blocks)));
        two_cell_stationary_lifting_kernel<<<blocks, THREADS>>>(
            d_values, d_owner, d_lut + host_lut.offset[occupied],
            supports, pc, states, occupied, W, i, mod,
            d_processed, d_primitive_scans, d_support_base_builds,
            d_residual_adds, d_nonidentity, d_error);
        ck(cudaGetLastError(), "lifting launch sector");
    }
    ck(cudaEventRecord(end), "lifting record end");
    ck(cudaEventSynchronize(end), "lifting sync end");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, begin, end), "lifting elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    unsigned long long processed = 0;
    unsigned long long primitive_scans = 0;
    unsigned long long support_base_builds = 0;
    unsigned long long residual_adds = 0;
    unsigned long long nonidentity = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
       "lifting copy output");
    ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost),
       "lifting copy processed");
    ck(cudaMemcpy(&primitive_scans, d_primitive_scans, sizeof(primitive_scans), cudaMemcpyDeviceToHost),
       "lifting copy primitive scans");
    ck(cudaMemcpy(&support_base_builds, d_support_base_builds, sizeof(support_base_builds), cudaMemcpyDeviceToHost),
       "lifting copy support builds");
    ck(cudaMemcpy(&residual_adds, d_residual_adds, sizeof(residual_adds), cudaMemcpyDeviceToHost),
       "lifting copy residual adds");
    ck(cudaMemcpy(&nonidentity, d_nonidentity, sizeof(nonidentity), cudaMemcpyDeviceToHost),
       "lifting copy nonidentity");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "lifting copy error");
    if (error || processed != components || residual_adds != states - components) {
        std::cerr << "FAIL lifting error=" << error
                  << " processed=" << processed << " expected=" << components
                  << " residual_adds=" << residual_adds
                  << " expected_adds=" << (states - components) << '\n';
        std::exit(190);
    }
    for (Rank r = 0; r < states; ++r) {
        if (output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL lifting arithmetic W=" << W << " i=" << i
                      << " rank=" << r << '\n';
            std::exit(191);
        }
    }

    std::cout << "two-cell-stationary-matching-microprobe"
              << " W=" << W
              << " i=" << i
              << " states=" << states
              << " components=" << components
              << " residual_adds=" << residual_adds
              << " sparse_nnz_baseline=" << (states + residual_adds)
              << " explicit_add_fraction="
              << double(residual_adds) / double(states + residual_adds)
              << " nonidentity_matchings=" << nonidentity
              << " primitive_scans=" << primitive_scans
              << " support_base_builds=" << support_base_builds
              << " kernel_ms=" << ms
              << " value_vectors=1 matching_table_bytes=0"
              << " destination_rank_calls=0 arithmetic=OK\n";

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(d_values);
    cudaFree(d_lut);
    cudaFree(d_owner);
    cudaFree(d_processed);
    cudaFree(d_primitive_scans);
    cudaFree(d_support_base_builds);
    cudaFree(d_residual_adds);
    cudaFree(d_nonidentity);
    cudaFree(d_error);
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 10;
    const unsigned blocks = argc > 2
        ? static_cast<unsigned>(std::strtoul(argv[2], nullptr, 10)) : 4096u;
    const std::uint32_t mod = argc > 3
        ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10)) : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth || blocks == 0 || mod < 3) return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    const Rank states = st.total[W];
    const Rank components = oneesan::twocell::component_label_count(W, rt);
    const Rank residual = states - components;
    if (plan_only) {
        Rank lut_entries = 0;
        for (int occupied = 1; occupied <= W - 2; occupied += 2)
            lut_entries += rt.primitive[occupied][1];
        std::cout << "two-cell-stationary-matching-microprobe-plan"
                  << " W=" << W
                  << " states=" << states
                  << " components=" << components
                  << " residual_adds=" << residual
                  << " sparse_nnz_baseline=" << (states + residual)
                  << " value_vectors=1 matching_table_bytes=0"
                  << " destination_rank_calls=0"
                  << " primitive_lut_MiB="
                  << double(lut_entries * sizeof(std::uint32_t)) / double(1ULL << 20)
                  << "\n";
        if (W == 28) {
            std::cout << "W=28 one_vector_GiB="
                      << double(states * 4ULL) / double(1ULL << 30)
                      << " matching_permutation_edges=" << states
                      << " residual_adds=" << residual
                      << "\n";
        }
        return 0;
    }

    if (W > 11) {
        std::cerr << "execution mode is intentionally limited to W<=11; use --plan-only above that\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "lifting device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "lifting set device");
    install_tables(rt);
    install_stationary_tables(st);
    const PrimitiveLut lut = build_primitive_lut(W, rt);
    for (int i = 1; i <= W - 5; ++i)
        run_stationary_lifting_position(W, i, rt, st, lut, mod, blocks);
    std::cout << "ALL_OK two_cell_stationary_matching_cuda=1 W=" << W << '\n';
    return 0;
}
