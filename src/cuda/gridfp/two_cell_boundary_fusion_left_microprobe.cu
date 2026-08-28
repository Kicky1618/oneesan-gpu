#pragma push_macro("main")
#undef main
#define main two_cell_boundary_fusion_singlebuffer_microprobe_main_unused
#include "two_cell_boundary_fusion_singlebuffer_microprobe.cu"
#pragma pop_macro("main")

namespace {

__global__ void two_cell_left_boundary_fusion_kernel(
    std::uint32_t* __restrict__ values,
    int W,
    int outer_ones,
    Rank support_count,
    Rank state_count,
    std::uint32_t mod,
    unsigned long long* processed_blocks,
    unsigned long long* processed_components,
    unsigned long long* global_loads,
    unsigned long long* global_stores,
    unsigned long long* local_adds,
    int* error
) {
    extern __shared__ std::uint32_t block_values[];
    __shared__ oneesan::twocell::PackedKey sh_state[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ oneesan::twocell::PackedKey sh_forward[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ std::uint32_t sh_value[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ std::uint32_t sh_output[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ Rank sh_rank[FUSION_WARPS][FUSION_MAX_COMPONENT];
    __shared__ oneesan::twocell::ComponentMatching sh_matching[FUSION_WARPS];
    __shared__ int sh_ns[FUSION_WARPS];
    __shared__ int sh_singular[FUSION_WARPS];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (warp >= FUSION_WARPS) return;

    const int start = 0;
    const int source_active = 1; // Q^rev_2 fixes C position 1
    const int edge_active = 0;   // Q^rev_1 and Q^fwd_0 fix C position 0
    const int outer_bits = W - 4;
    const Rank n = oneesan::twocell::fusion_block_size(
        BOUNDARY_STEPS, outer_ones, TC_RANK_TABLES);
    const Rank nc = oneesan::twocell::fusion_component_count(
        BOUNDARY_STEPS, outer_ones, TC_RANK_TABLES);

    for (Rank support_rank = blockIdx.x; support_rank < support_count;
         support_rank += gridDim.x) {
        const std::uint32_t outer = oneesan::twocell::support_unrank(
            outer_bits, outer_ones, support_rank, TC_RANK_TABLES);

        for (Rank r = threadIdx.x; r < n; r += blockDim.x) {
            const auto d = oneesan::twocell::fusion_local_unrank_at(
                r, outer, W, start, BOUNDARY_STEPS, source_active,
                TC_RANK_TABLES);
            if (!d.valid) {
                set_error(error, 301);
                continue;
            }
            const Rank gr = oneesan::twocell::stationary_rank_with_primitive(
                d.key, W, source_active, d.primitive,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
            if (gr >= state_count) {
                set_error(error, 302);
                continue;
            }
            block_values[r] = values[gr];
            atomicAdd(global_loads, 1ULL);
        }
        __syncthreads();

        // Reverse final interior transfer is J K_{W-4} J. Recover the unique
        // matching in the forward reflected component and apply the same source
        // index -> destination-coordinate index permutation to reflected values.
        for (Rank cr = warp; cr < nc; cr += FUSION_WARPS) {
            if (lane == 0) {
                sh_ns[warp] = 0;
                const auto label = oneesan::twocell::fusion_component_unrank(
                    cr, outer, W, start, BOUNDARY_STEPS, TC_RANK_TABLES);
                if (!label.valid) {
                    set_error(error, 303);
                } else {
                    const auto mirrored_label = oneesan::twocell::reflect_packed_word(label.word);
                    const auto forward = oneesan::twocell::direct_component_sources(
                        mirrored_label, W, W - 4);
                    if (forward.overflow || forward.size <= 0 ||
                        forward.size > FUSION_MAX_COMPONENT) {
                        set_error(error, 304);
                    } else {
                        sh_ns[warp] = forward.size;
                        for (int q = 0; q < forward.size; ++q) {
                            sh_forward[warp][q] = forward.value[q];
                            sh_state[warp][q] = oneesan::twocell::reflect_packed_key(
                                forward.value[q], W);
                        }
                        const auto m = oneesan::twocell::build_component_matching(
                            sh_forward[warp], forward.size, W, W - 4);
                        if (!m.ok) {
                            set_error(error, 307);
                        } else {
                            sh_matching[warp] = m;
                            atomicAdd(processed_components, 1ULL);
                        }
                    }
                }
            }
            __syncwarp();

            const int ns = sh_ns[warp];
            if (lane < ns) {
                const auto source = sh_state[warp][lane];
                if (oneesan::twocell::fusion_outer_mask_at(
                        source, start, BOUNDARY_STEPS, source_active) != outer)
                    set_error(error, 305);
                const int len = source.type ? W - 2 : W - 1;
                const Rank primitive = oneesan::twocell::primitive_rank(
                    source.support, source.left, len, TC_RANK_TABLES);
                const Rank lr = oneesan::twocell::fusion_local_rank_at_with_primitive(
                    source, start, BOUNDARY_STEPS, source_active, outer_ones,
                    primitive, TC_RANK_TABLES);
                sh_rank[warp][lane] = lr;
                if (lr >= n) {
                    sh_value[warp][lane] = 0;
                    set_error(error, 306);
                } else {
                    sh_value[warp][lane] = block_values[lr];
                }
            }
            __syncwarp();

            if (lane == 0 && ns > 0) {
                const auto m = sh_matching[warp];
                if (!oneesan::twocell::apply_component_matching(
                        m, sh_value[warp], sh_output[warp], mod)) {
                    set_error(error, 308);
                } else {
                    atomicAdd(local_adds,
                              static_cast<unsigned long long>(m.residual_edges));
                }
            }
            __syncwarp();
            if (lane < ns && sh_rank[warp][lane] < n)
                block_values[sh_rank[warp][lane]] = sh_output[warp][lane];
            __syncwarp();
        }
        __syncthreads();

        // Physical left turn uses the directly reflected closed block.
        for (Rank cr = warp; cr < nc; cr += FUSION_WARPS) {
            if (lane == 0) {
                sh_ns[warp] = 0;
                const auto label = oneesan::twocell::fusion_component_unrank(
                    cr, outer, W, start, BOUNDARY_STEPS, TC_RANK_TABLES);
                if (!label.valid) {
                    set_error(error, 312);
                } else {
                    const auto b = oneesan::twocell::left_turn_closed_block(label.word, W);
                    sh_singular[warp] = b.singular;
                    if (b.overflow || b.size < 3 || b.size > FUSION_MAX_COMPONENT) {
                        set_error(error, 313);
                    } else {
                        sh_ns[warp] = b.size;
                        for (int q = 0; q < b.size; ++q) sh_state[warp][q] = b.state[q];
                        atomicAdd(processed_components, 1ULL);
                    }
                }
            }
            __syncwarp();

            const int ns = sh_ns[warp];
            if (lane < ns) {
                const auto s = sh_state[warp][lane];
                if (oneesan::twocell::fusion_outer_mask_at(
                        s, start, BOUNDARY_STEPS, edge_active) != outer)
                    set_error(error, 314);
                const int len = s.type ? W - 2 : W - 1;
                const Rank primitive = oneesan::twocell::primitive_rank(
                    s.support, s.left, len, TC_RANK_TABLES);
                const Rank lr = oneesan::twocell::fusion_local_rank_at_with_primitive(
                    s, start, BOUNDARY_STEPS, edge_active, outer_ones,
                    primitive, TC_RANK_TABLES);
                sh_rank[warp][lane] = lr;
                if (lr >= n) {
                    sh_value[warp][lane] = 0;
                    set_error(error, 315);
                } else {
                    sh_value[warp][lane] = block_values[lr];
                }
            }
            __syncwarp();

            if (lane == 0 && ns > 0) {
                if (sh_singular[warp]) {
                    if (ns != 3) {
                        set_error(error, 316);
                    } else {
                        const std::uint32_t t0 = fusion_add_mod(
                            sh_value[warp][0], sh_value[warp][2], mod);
                        const std::uint32_t t1 = fusion_add_mod(
                            sh_value[warp][1], sh_value[warp][2], mod);
                        sh_value[warp][0] = boundary_double_mod(t0, mod);
                        sh_value[warp][1] = t1;
                        sh_value[warp][2] = t1;
                        atomicAdd(local_adds, 2ULL);
                    }
                } else {
                    std::uint32_t t = sh_value[warp][1];
                    for (int q = 2; q < ns; ++q)
                        t = fusion_add_mod(t, sh_value[warp][q], mod);
                    sh_value[warp][0] = fusion_add_mod(
                        boundary_double_mod(sh_value[warp][0], mod), t, mod);
                    sh_value[warp][1] = t;
                    for (int q = 2; q < ns; ++q)
                        sh_value[warp][q] = boundary_double_mod(sh_value[warp][q], mod);
                    atomicAdd(local_adds, static_cast<unsigned long long>(ns - 1));
                }
            }
            __syncwarp();
            if (lane < ns && sh_rank[warp][lane] < n)
                block_values[sh_rank[warp][lane]] = sh_value[warp][lane];
            __syncwarp();
        }
        __syncthreads();

        for (Rank r = threadIdx.x; r < n; r += blockDim.x) {
            const auto d = oneesan::twocell::fusion_local_unrank_at(
                r, outer, W, start, BOUNDARY_STEPS, edge_active, TC_RANK_TABLES);
            if (!d.valid) {
                set_error(error, 317);
                continue;
            }
            const Rank gr = oneesan::twocell::stationary_rank_with_primitive(
                d.key, W, edge_active, d.primitive,
                TC_RANK_TABLES, TC_STATIONARY_TABLES);
            if (gr >= state_count) {
                set_error(error, 318);
                continue;
            }
            values[gr] = block_values[r];
            atomicAdd(global_stores, 1ULL);
        }
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(processed_blocks, 1ULL);
        __syncthreads();
    }
}

std::vector<std::uint32_t> left_boundary_reference(
    const std::vector<std::uint32_t>& input,
    int W,
    const RankTables& rt,
    const StationaryRankTables& st,
    std::uint32_t mod
) {
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(W + 1));
    for (int n = 1; n <= W; ++n) words[n] = gen_words(n);
    std::vector<std::uint32_t> mid(input.size()), out(input.size());
    for (const Key& s : reverse_q_basis(W, 2, words)) {
        const Rank sr = oneesan::twocell::stationary_rank(
            device_key(s), W, 1, rt, st);
        const std::uint32_t x = input[static_cast<std::size_t>(sr)];
        for (const auto& [d, c] : K_reverse_basis(s, W, 2)) {
            if (c != 1) std::exit(319);
            const Rank dr = oneesan::twocell::stationary_rank(
                device_key(d), W, 0, rt, st);
            mid[static_cast<std::size_t>(dr)] = add_ref_mod(
                mid[static_cast<std::size_t>(dr)], x, mod);
        }
    }
    for (const Key& s : reverse_q_basis(W, 1, words)) {
        const Rank sr = oneesan::twocell::stationary_rank(
            device_key(s), W, 0, rt, st);
        const std::uint32_t x = mid[static_cast<std::size_t>(sr)];
        for (const auto& [d, c] : turn_left_basis(s, W)) {
            const Rank dr = oneesan::twocell::stationary_rank(
                device_key(d), W, 0, rt, st);
            const unsigned long long add = static_cast<unsigned long long>(x) *
                                           static_cast<unsigned long long>(c);
            out[static_cast<std::size_t>(dr)] = static_cast<std::uint32_t>(
                (out[static_cast<std::size_t>(dr)] + add) % mod);
        }
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    const int W = argc > 1 ? std::atoi(argv[1]) : 8;
    const Rank shared_kib = argc > 2
        ? static_cast<Rank>(std::strtoull(argv[2], nullptr, 10)) : 228ULL;
    const std::uint32_t mod = argc > 3
        ? static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10))
        : 4294967291u;
    const bool plan_only = has_arg(argc, argv, "--plan-only");
    if (W < 6 || W > oneesan::twocell::kMaxWidth || mod < 3) return 2;

    const RankTables rt = oneesan::twocell::make_rank_tables();
    const StationaryRankTables st = oneesan::twocell::make_stationary_rank_tables(rt);
    if (plan_only) {
        print_boundary_singlebuffer_plan(
            W, shared_kib * 1024ULL, singlebuffer_workspace_estimate(), rt);
        return 0;
    }
    if (W > 10) return 3;

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "left boundary device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "left boundary set device");
    install_tables(rt);
    install_stationary_tables(st);

    cudaFuncAttributes attr{};
    ck(cudaFuncGetAttributes(&attr, two_cell_left_boundary_fusion_kernel),
       "left boundary attributes");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0), "left boundary props");
    const Rank limit = std::min<Rank>(
        shared_kib * 1024ULL, static_cast<Rank>(prop.sharedMemPerBlockOptin));
    const Rank static_shared = attr.sharedSizeBytes;
    const Rank states = st.total[W];
    const int outer_bits = W - 4;

    std::vector<std::uint32_t> input(static_cast<std::size_t>(states));
    for (Rank r = 0; r < states; ++r)
        input[static_cast<std::size_t>(r)] = static_cast<std::uint32_t>(
            1 + ((r * 2654435761ULL + 197ULL) % (mod - 1ULL)));
    const auto reference = left_boundary_reference(input, W, rt, st, mod);

    std::uint32_t* d_values = nullptr;
    unsigned long long *d_blocks = nullptr, *d_components = nullptr;
    unsigned long long *d_loads = nullptr, *d_stores = nullptr, *d_adds = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_values, states * sizeof(std::uint32_t)), "left boundary alloc values");
    ck(cudaMalloc(&d_blocks, sizeof(unsigned long long)), "left boundary alloc blocks");
    ck(cudaMalloc(&d_components, sizeof(unsigned long long)), "left boundary alloc components");
    ck(cudaMalloc(&d_loads, sizeof(unsigned long long)), "left boundary alloc loads");
    ck(cudaMalloc(&d_stores, sizeof(unsigned long long)), "left boundary alloc stores");
    ck(cudaMalloc(&d_adds, sizeof(unsigned long long)), "left boundary alloc adds");
    ck(cudaMalloc(&d_error, sizeof(int)), "left boundary alloc error");
    ck(cudaMemcpy(d_values, input.data(), states * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice), "left boundary copy values");
    ck(cudaMemset(d_blocks, 0, sizeof(unsigned long long)), "left boundary zero blocks");
    ck(cudaMemset(d_components, 0, sizeof(unsigned long long)), "left boundary zero components");
    ck(cudaMemset(d_loads, 0, sizeof(unsigned long long)), "left boundary zero loads");
    ck(cudaMemset(d_stores, 0, sizeof(unsigned long long)), "left boundary zero stores");
    ck(cudaMemset(d_adds, 0, sizeof(unsigned long long)), "left boundary zero adds");
    ck(cudaMemset(d_error, 0, sizeof(int)), "left boundary zero error");

    Rank fit_states = 0, fit_blocks = 0, fit_components = 0;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank n = oneesan::twocell::fusion_block_size(BOUNDARY_STEPS, o, rt);
        if (static_shared + n * sizeof(std::uint32_t) > limit) continue;
        const Rank supports = rt.choose[outer_bits][o];
        const Rank nc = oneesan::twocell::fusion_component_count(BOUNDARY_STEPS, o, rt);
        fit_states += supports * n;
        fit_blocks += supports;
        fit_components += 2 * supports * nc;
        const Rank dynamic_bytes = n * sizeof(std::uint32_t);
        ck(cudaFuncSetAttribute(
               two_cell_left_boundary_fusion_kernel,
               cudaFuncAttributeMaxDynamicSharedMemorySize,
               static_cast<int>(dynamic_bytes)),
           "left boundary optin shared");
        const unsigned grid = static_cast<unsigned>(std::min<Rank>(supports, 65535));
        two_cell_left_boundary_fusion_kernel<<<grid, FUSION_THREADS, dynamic_bytes>>>(
            d_values, W, o, supports, states, mod,
            d_blocks, d_components, d_loads, d_stores, d_adds, d_error);
        ck(cudaGetLastError(), "left boundary launch");
    }
    ck(cudaDeviceSynchronize(), "left boundary sync");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    unsigned long long blocks = 0, components = 0, loads = 0, stores = 0, adds = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_values, states * sizeof(std::uint32_t),
                  cudaMemcpyDeviceToHost), "left boundary copy output");
    ck(cudaMemcpy(&blocks, d_blocks, sizeof(blocks), cudaMemcpyDeviceToHost),
       "left boundary copy blocks");
    ck(cudaMemcpy(&components, d_components, sizeof(components), cudaMemcpyDeviceToHost),
       "left boundary copy components");
    ck(cudaMemcpy(&loads, d_loads, sizeof(loads), cudaMemcpyDeviceToHost),
       "left boundary copy loads");
    ck(cudaMemcpy(&stores, d_stores, sizeof(stores), cudaMemcpyDeviceToHost),
       "left boundary copy stores");
    ck(cudaMemcpy(&adds, d_adds, sizeof(adds), cudaMemcpyDeviceToHost),
       "left boundary copy adds");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost),
       "left boundary copy error");

    if (error || blocks != fit_blocks || components != fit_components ||
        loads != fit_states || stores != fit_states) {
        std::cerr << "FAIL left boundary counters error=" << error << '\n';
        return 5;
    }
    if (fit_states == states && output != reference) {
        std::cerr << "FAIL left boundary arithmetic W=" << W << '\n';
        return 6;
    }
    std::cout << "two-cell-left-boundary-fusion"
              << " W=" << W
              << " fused_states=" << fit_states
              << " global_loads=" << loads
              << " global_stores=" << stores
              << " local_adds=" << adds
              << " matching=leaf_peeling reflection_reused=1 arithmetic="
              << (fit_states == states ? "OK" : "PARTIAL") << '\n';

    cudaFree(d_values);
    cudaFree(d_blocks);
    cudaFree(d_components);
    cudaFree(d_loads);
    cudaFree(d_stores);
    cudaFree(d_adds);
    cudaFree(d_error);
    return 0;
}
