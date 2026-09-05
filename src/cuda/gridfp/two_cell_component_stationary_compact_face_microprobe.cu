#pragma push_macro("main")
#undef main
#define main two_cell_component_stationary_closed_face_microprobe_main_unused
#include "two_cell_component_stationary_closed_face_microprobe.cu"
#pragma pop_macro("main")

namespace {

__global__ void two_cell_stationary_compact_face_kernel(
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
    unsigned long long* deep_components,
    unsigned long long* primitive_scans,
    unsigned long long* support_base_builds,
    unsigned long long* residual_adds,
    int* error
) {
    __shared__ PackedKey sh_src[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_value[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_output[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ Rank sh_rank[WARPS_PER_BLOCK][MAX_STATES];
    __shared__ std::uint32_t sh_support[WARPS_PER_BLOCK];
    __shared__ PackedWord sh_label[WARPS_PER_BLOCK];
    __shared__ oneesan::twocell::StationaryComponentBases sh_bases[WARPS_PER_BLOCK];
    __shared__ int sh_ns[WARPS_PER_BLOCK];
    __shared__ int sh_deep[WARPS_PER_BLOCK];
    __shared__ int sh_partner_rounds[WARPS_PER_BLOCK];

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
                sh_label[warp] = PackedWord{
                    support, left, static_cast<std::uint8_t>(label_len)};
                PackedWord collapsed{};
                sh_deep[warp] = oneesan::twocell::deep_collapse(
                    sh_label[warp], i, collapsed) ? 1 : 0;
                sh_ns[warp] = 0;
                sh_partner_rounds[warp] = 0;
            }
            __syncwarp();

            if (sh_deep[warp]) {
                oneesan::twocell::cuda_face::deep_component_sources_compact(
                    sh_label[warp], W, i,
                    sh_src[warp], &sh_ns[warp],
                    &sh_partner_rounds[warp], error);
                __syncwarp();
                if (lane == 0) atomicAdd(deep_components, 1ULL);
            } else if (lane == 0) {
                const auto src = oneesan::twocell::direct_component_sources(
                    sh_label[warp], W, i);
                if (src.overflow || src.size <= 0 || src.size > MAX_STATES) {
                    set_error(error, 381);
                } else {
                    sh_ns[warp] = src.size;
                    for (int s = 0; s < src.size; ++s)
                        sh_src[warp][s] = src.value[s];
                }
            }
            __syncwarp();

            const int ns = sh_ns[warp];
            if (ns <= 0 || ns > MAX_STATES) {
                if (lane == 0) set_error(error, 382);
                __syncwarp();
                continue;
            }
            if (lane == 0) atomicAdd(processed, 1ULL);

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
                    set_error(error, 383);
                } else {
                    sh_value[warp][lane] = values[r];
                }
            }
            __syncwarp();

            if (lane == 0) {
                if (!oneesan::twocell::apply_component_fastpath(
                        sh_src[warp], ns, W, i,
                        sh_value[warp], sh_output[warp], mod)) {
                    set_error(error, 384);
                } else {
                    atomicAdd(residual_adds,
                              static_cast<unsigned long long>(ns - 1));
                }
            }
            __syncwarp();

            if (lane < ns) {
                const Rank r = sh_rank[warp][lane];
                const unsigned long long component_id =
                    (static_cast<unsigned long long>(support_rank) << 32) |
                    static_cast<unsigned long long>(pr);
                const unsigned long long previous = atomicCAS(
                    owner + r, ~0ULL, component_id);
                if (previous != ~0ULL && previous != component_id)
                    set_error(error, 385);
                values[r] = sh_output[warp][lane];
            }
            __syncwarp();
        }
    }
}

void run_stationary_compact_face_position(
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
    const Rank expected_deep = oneesan::twocell::component_label_count(W - 1, rt);

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(W + 1));
    for (int n = 1; n <= W; ++n) words[n] = gen_words(n);
    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    std::vector<std::uint32_t> reference(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + Rank(i) * 263ULL) % (mod - 1ULL)));
    for (const Key& s : q_basis(W, i, words)) {
        const Rank sr = oneesan::twocell::stationary_rank(
            device_key(s), W, i, rt, st);
        const std::uint32_t value = input[static_cast<std::size_t>(sr)];
        for (const auto& [d, c] : K_basis(s, W, i)) {
            if (c != 1) std::exit(386);
            const Rank dr = oneesan::twocell::stationary_rank(
                device_key(d), W, i + 1, rt, st);
            reference[static_cast<std::size_t>(dr)] =
                add_mod(reference[static_cast<std::size_t>(dr)], value, mod);
        }
    }

    std::uint32_t* d_values = nullptr;
    std::uint32_t* d_lut = nullptr;
    unsigned long long *d_owner = nullptr, *d_processed = nullptr, *d_deep = nullptr;
    unsigned long long *d_scans = nullptr, *d_bases = nullptr, *d_adds = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)), "compact face alloc values");
    ck(cudaMalloc(&d_lut, host_lut.value.size() * sizeof(std::uint32_t)), "compact face alloc lut");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "compact face alloc owner");
    ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "compact face alloc processed");
    ck(cudaMalloc(&d_deep, sizeof(unsigned long long)), "compact face alloc deep");
    ck(cudaMalloc(&d_scans, sizeof(unsigned long long)), "compact face alloc scans");
    ck(cudaMalloc(&d_bases, sizeof(unsigned long long)), "compact face alloc bases");
    ck(cudaMalloc(&d_adds, sizeof(unsigned long long)), "compact face alloc adds");
    ck(cudaMalloc(&d_error, sizeof(int)), "compact face alloc error");
    ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "compact face copy values");
    ck(cudaMemcpy(d_lut, host_lut.value.data(),
                  host_lut.value.size() * sizeof(std::uint32_t), cudaMemcpyHostToDevice),
       "compact face copy lut");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "compact face clear owner");
    ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "compact face zero processed");
    ck(cudaMemset(d_deep, 0, sizeof(unsigned long long)), "compact face zero deep");
    ck(cudaMemset(d_scans, 0, sizeof(unsigned long long)), "compact face zero scans");
    ck(cudaMemset(d_bases, 0, sizeof(unsigned long long)), "compact face zero bases");
    ck(cudaMemset(d_adds, 0, sizeof(unsigned long long)), "compact face zero adds");
    ck(cudaMemset(d_error, 0, sizeof(int)), "compact face zero error");

    cudaEvent_t begin{}, end{};
    ck(cudaEventCreate(&begin), "compact face event begin");
    ck(cudaEventCreate(&end), "compact face event end");
    ck(cudaEventRecord(begin), "compact face record begin");
    for (int occupied = 1; occupied <= W - 2; occupied += 2) {
        const Rank supports = rt.choose[W - 2][occupied];
        const Rank pc = rt.primitive[occupied][1];
        const Rank tiles = sector_tiles(W, occupied, rt);
        const Rank one_pass_blocks = (tiles + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const unsigned blocks = static_cast<unsigned>(std::max<Rank>(
            1, std::min<Rank>(requested_blocks, one_pass_blocks)));
        two_cell_stationary_compact_face_kernel<<<blocks, THREADS>>>(
            d_values, d_owner, d_lut + host_lut.offset[occupied],
            supports, pc, states, occupied, W, i, mod,
            d_processed, d_deep, d_scans, d_bases, d_adds, d_error);
        ck(cudaGetLastError(), "compact face launch sector");
    }
    ck(cudaEventRecord(end), "compact face record end");
    ck(cudaEventSynchronize(end), "compact face sync end");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, begin, end), "compact face elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    unsigned long long processed = 0, deep = 0, scans = 0, bases = 0, adds = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
       "compact face copy output");
    ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost),
       "compact face copy processed");
    ck(cudaMemcpy(&deep, d_deep, sizeof(deep), cudaMemcpyDeviceToHost),
       "compact face copy deep");
    ck(cudaMemcpy(&scans, d_scans, sizeof(scans), cudaMemcpyDeviceToHost),
       "compact face copy scans");
    ck(cudaMemcpy(&bases, d_bases, sizeof(bases), cudaMemcpyDeviceToHost),
       "compact face copy bases");
    ck(cudaMemcpy(&adds, d_adds, sizeof(adds), cudaMemcpyDeviceToHost),
       "compact face copy adds");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "compact face copy error");

    if (error || processed != components || deep != expected_deep ||
        adds != states - components || output != reference) {
        std::cerr << "FAIL stationary compact face W=" << W << " i=" << i
                  << " error=" << error
                  << " processed=" << processed << "/" << components
                  << " deep=" << deep << "/" << expected_deep
                  << " adds=" << adds << "/" << (states - components) << '\n';
        std::exit(387);
    }

    cudaFuncAttributes compact_attr{}, old_attr{};
    ck(cudaFuncGetAttributes(&compact_attr, two_cell_stationary_compact_face_kernel),
       "compact face attributes");
    ck(cudaFuncGetAttributes(&old_attr, two_cell_stationary_closed_face_kernel),
       "old closed face attributes");
    std::cout << "two-cell-stationary-compact-face"
              << " W=" << W
              << " i=" << i
              << " states=" << states
              << " components=" << components
              << " deep_components=" << deep
              << " residual_adds=" << adds
              << " primitive_scans=" << scans
              << " support_base_builds=" << bases
              << " static_shared_bytes=" << compact_attr.sharedSizeBytes
              << " old_face_static_shared_bytes=" << old_attr.sharedSizeBytes
              << " candidate_shared_scratch_bytes=0"
              << " partner_scan_rounds=0"
              << " validity_scans=0"
              << " candidate_dedup=0"
              << " matching_K_step_calls=0"
              << " kernel_ms=" << ms
              << " arithmetic=OK\n";

    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    cudaFree(d_values);
    cudaFree(d_lut);
    cudaFree(d_owner);
    cudaFree(d_processed);
    cudaFree(d_deep);
    cudaFree(d_scans);
    cudaFree(d_bases);
    cudaFree(d_adds);
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
    if (plan_only) {
        const Rank states = st.total[W];
        const Rank components = oneesan::twocell::component_label_count(W, rt);
        std::cout << "two-cell-stationary-compact-face-plan"
                  << " W=" << W
                  << " states=" << states
                  << " components=" << components
                  << " deep_components="
                  << oneesan::twocell::component_label_count(W - 1, rt)
                  << " max_component_sources=17"
                  << " candidate_shared_scratch_bytes=0"
                  << " partner_scan_rounds=0"
                  << " validity_scans=0 candidate_dedup=0"
                  << " matching_K_step_calls=0"
                  << " global_value_vectors=1\n";
        return 0;
    }
    if (W > 11) {
        std::cerr << "execution mode intentionally limited to W<=11; use --plan-only above that\n";
        return 3;
    }
    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "compact face device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "compact face set device");
    install_tables(rt);
    install_stationary_tables(st);
    const PrimitiveLut lut = build_primitive_lut(W, rt);
    for (int i = 1; i <= W - 5; ++i)
        run_stationary_compact_face_position(W, i, rt, st, lut, mod, blocks);
    std::cout << "ALL_OK two_cell_stationary_compact_face_cuda=1 W=" << W << '\n';
    return 0;
}
