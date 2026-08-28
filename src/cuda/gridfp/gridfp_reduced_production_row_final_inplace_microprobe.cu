#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_component_dense_microprobe_main_unused
#include "gridfp_reduced_production_component_dense_microprobe.cu"
#pragma pop_macro("main")

namespace {

__device__ __forceinline__ bool final_small_step(
    DeviceKey src, int W, int p, bool reverse, SmallTerms& out
) {
    if (!src.blocked) {
        if (!small_add(out, src, 1)) return false;
        const IncludeResult z = reverse
            ? include_horizontal_reverse(src.mate, W, p)
            : include_horizontal(src.mate, W, p);
        if (!z.valid) return true;
        if (z.blocked) return false;
        return small_add(out, DeviceKey{z.mate, 0}, 1);
    }
    const MateID m = reverse
        ? blocked_exclude_reverse(src.mate, W, p)
        : blocked_exclude(src.mate, p);
    return small_add(out, DeviceKey{m, 0}, 1);
}

__device__ __forceinline__ int final_try_main_preimage_forward(
    MateID x, MateID dest, int W, DeviceTerm* out, int n
) {
    if (!valid_mate_device(x, W)) return n;
    const IncludeResult z = include_horizontal(x, W, 1);
    if (z.valid && !z.blocked && z.mate == dest)
        return add_term(out, n, DeviceKey{x, 0}, 1);
    return n;
}

__device__ __forceinline__ int final_inverse_forward(
    DeviceKey dest, int W, DeviceTerm* out
) {
    if (dest.blocked) return -1;
    int n = 0;
    const MateID d = dest.mate;
    n = add_term(out, n, DeviceKey{d, 0}, 1); // excluded identity
    if (n < 0) return n;

    const MateValuePair w = mpair(d, 1);
    if (w == LR) n = final_try_main_preimage_forward(msetpair(d, 1, NN), d, W, out, n);
    if (n < 0) return n;
    if (w == NR) n = final_try_main_preimage_forward(msetpair(d, 1, RN), d, W, out, n);
    if (n < 0) return n;
    if (w == RN) n = final_try_main_preimage_forward(msetpair(d, 1, NR), d, W, out, n);
    if (n < 0) return n;
    if (w == NL) n = final_try_main_preimage_forward(msetpair(d, 1, LN), d, W, out, n);
    if (n < 0) return n;
    if (w == LN) n = final_try_main_preimage_forward(msetpair(d, 1, NL), d, W, out, n);
    if (n < 0) return n;

    MateID closure[RP_MAX_TERMS]{};
    const int nc = ordinary_closure_preimages_partial(d, W, 1, closure);
    for (int i = 0; i < nc; ++i) {
        n = final_try_main_preimage_forward(closure[i], d, W, out, n);
        if (n < 0) return n;
    }

    if (mget(d, 1) == N && is_endpoint(mget(d, 0))) {
        const MateID b = mshrink(d, 1);
        if (valid_mate_device(b, W - 1) && mget(b, 0) != N && blocked_exclude(b, 1) == d) {
            n = add_term(out, n, DeviceKey{b, 1}, 1);
            if (n < 0) return n;
        }
    }
    return n;
}

__device__ __forceinline__ int final_inverse_direction(
    DeviceKey dest, int W, bool reverse, DeviceTerm* out
) {
    if (!reverse) return final_inverse_forward(dest, W, out);
    const DeviceKey md = mirror_key_device(dest, W);
    DeviceTerm tmp[RP_MAX_TERMS]{};
    const int nt = final_inverse_forward(md, W, tmp);
    if (nt < 0) return nt;
    int n = 0;
    for (int i = 0; i < nt; ++i) {
        n = add_term(out, n, mirror_key_device(tmp[i].key, W), tmp[i].coef);
        if (n < 0) return n;
    }
    return n;
}

__device__ __forceinline__ DeviceKey final_component_seed(
    MateID label, int W, int p, bool reverse
) {
    const MateID m = reverse ? blocked_exclude_reverse(label, W, p)
                             : blocked_exclude(label, p);
    return DeviceKey{m, 0};
}

__global__ void row_final_inplace_kernel(
    std::uint32_t* state,
    unsigned long long* owner,
    Rank64 raw_labels,
    Rank64 state_count,
    int W,
    int p,
    bool reverse,
    std::uint32_t mod,
    unsigned long long* processed,
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

    for (Rank64 label_rank = first; label_rank < raw_labels; label_rank += stride) {
        if (lane == 0) {
            sh_ns[warp] = 0;
            sh_nd[warp] = 0;
            const MateID label = motzkin_unrank_device(W - 1, label_rank);
            const DeviceKey seed = final_component_seed(label, W, p, reverse);
            if (!valid_mate_device(seed.mate, W)) {
                set_error(error, 131);
            } else {
                sh_src[warp][0] = seed;
                sh_ns[warp] = 1;
                int cursor = 0;
                while (cursor < sh_ns[warp]) {
                    SmallTerms edge;
                    if (!final_small_step(sh_src[warp][cursor++], W, p, reverse, edge)) {
                        set_error(error, 132);
                        break;
                    }
                    for (int ei = 0; ei < edge.n; ++ei) {
                        if (!edge.v[ei].coef) continue;
                        const DeviceKey d = edge.v[ei].key;
                        if (d.blocked) {
                            set_error(error, 133);
                            break;
                        }
                        if (find_key(sh_dst[warp], sh_nd[warp], d) < 0) {
                            if (sh_nd[warp] >= MAX_PAIRS) {
                                set_error(error, 134);
                                break;
                            }
                            sh_dst[warp][sh_nd[warp]++] = d;
                        }
                        DeviceTerm pre[RP_MAX_TERMS]{};
                        const int np = final_inverse_direction(d, W, reverse, pre);
                        if (np < 0) {
                            set_error(error, 135);
                            break;
                        }
                        for (int pi = 0; pi < np; ++pi) {
                            if (!pre[pi].coef) continue;
                            if (find_key(sh_src[warp], sh_ns[warp], pre[pi].key) >= 0) continue;
                            if (sh_ns[warp] >= MAX_PAIRS) {
                                set_error(error, 136);
                                break;
                            }
                            sh_src[warp][sh_ns[warp]++] = pre[pi].key;
                        }
                    }
                    if (*error) break;
                }
                if (!(sh_ns[warp] == sh_nd[warp] ||
                      (sh_ns[warp] == 3 && sh_nd[warp] == 2)))
                    set_error(error, 137);
                if (sh_ns[warp] > (W + 1) / 2 + 1)
                    set_error(error, 138);
                atomicAdd(processed, 1ULL);
            }
        }
        __syncwarp();

        const int ns = sh_ns[warp];
        const int nd = sh_nd[warp];
        if (lane < ns) {
            const Rank64 r = factor_rank_device(sh_src[warp][lane], W, src_fixed);
            if (r >= state_count) {
                set_error(error, 139);
                sh_value[warp][lane] = 0;
            } else {
                sh_value[warp][lane] = state[r];
            }
        }
        __syncwarp();

        if (lane < nd) {
            const DeviceKey mine = sh_dst[warp][lane];
            long long acc = 0;
            for (int si = 0; si < ns; ++si) {
                SmallTerms edge;
                if (!final_small_step(sh_src[warp][si], W, p, reverse, edge)) {
                    set_error(error, 140);
                    continue;
                }
                for (int ei = 0; ei < edge.n; ++ei) {
                    if (key_equal(edge.v[ei].key, mine))
                        acc += static_cast<long long>(sh_value[warp][si]);
                }
            }
            long long z = acc % static_cast<long long>(mod);
            if (z < 0) z += mod;
            const Rank64 dr = factor_rank_device(mine, W, src_fixed);
            if (dr >= state_count) {
                set_error(error, 141);
            } else {
                const unsigned long long empty = ~0ULL;
                const unsigned long long prev = atomicCAS(
                    owner + dr, empty, static_cast<unsigned long long>(label_rank));
                if (prev != empty && prev != static_cast<unsigned long long>(label_rank))
                    set_error(error, 142);
                state[dr] = static_cast<std::uint32_t>(z);
            }
        }
        __syncwarp();
    }
}

void run_row_final_inplace(
    int W,
    bool reverse,
    const ProductionFactorTables& tables,
    std::uint32_t mod,
    unsigned requested_blocks
) {
    const int p = reverse ? W - 1 : 1;
    const int src_fixed = p - 1;
    ProductionFactorCodec codec(tables, src_fixed);
    const Rank64 states = tables.size();
    const Rank64 raw_labels = motzkin_count(W - 1);

    std::vector<std::uint32_t> initial(static_cast<std::size_t>(states));
    std::vector<std::uint32_t> reference(static_cast<std::size_t>(states));
    std::vector<std::uint8_t> is_main(static_cast<std::size_t>(states));
    for (Rank64 r = 0; r < states; ++r) {
        const Key src = codec.unrank(r);
        const std::uint32_t value = static_cast<std::uint32_t>(
            (1 + (r * 2654435761ULL) % (mod - 1ULL)) % mod);
        initial[static_cast<std::size_t>(r)] = value;
        if (!src.blocked) is_main[static_cast<std::size_t>(r)] = 1;
        for (const auto& [d, c] : step_basis(src, W, p, reverse)) {
            if (c != 1 || d.blocked) fail("row final host reference non-main/nonunit");
            add_mod_signed(reference[static_cast<std::size_t>(codec.rank(d))], value, 1, mod);
        }
    }

    std::uint32_t* d_state = nullptr;
    unsigned long long* d_owner = nullptr;
    unsigned long long* d_processed = nullptr;
    int* d_error = nullptr;
    ck(cudaMalloc(&d_state, states * sizeof(std::uint32_t)), "final inplace alloc state");
    ck(cudaMalloc(&d_owner, states * sizeof(unsigned long long)), "final inplace alloc owner");
    ck(cudaMalloc(&d_processed, sizeof(unsigned long long)), "final inplace alloc processed");
    ck(cudaMalloc(&d_error, sizeof(int)), "final inplace alloc error");
    ck(cudaMemcpy(d_state, initial.data(), states * sizeof(std::uint32_t), cudaMemcpyHostToDevice), "final inplace copy state");
    ck(cudaMemset(d_owner, 0xff, states * sizeof(unsigned long long)), "final inplace clear owner");
    ck(cudaMemset(d_processed, 0, sizeof(unsigned long long)), "final inplace zero processed");
    ck(cudaMemset(d_error, 0, sizeof(int)), "final inplace zero error");

    const Rank64 one_pass_blocks = (raw_labels + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    const unsigned blocks = static_cast<unsigned>(std::max<Rank64>(
        1, std::min<Rank64>(requested_blocks, one_pass_blocks)));
    cudaEvent_t a{}, b{};
    ck(cudaEventCreate(&a), "final inplace event a");
    ck(cudaEventCreate(&b), "final inplace event b");
    ck(cudaEventRecord(a), "final inplace record a");
    row_final_inplace_kernel<<<blocks, THREADS>>>(
        d_state, d_owner, raw_labels, states, W, p, reverse, mod, d_processed, d_error);
    ck(cudaGetLastError(), "final inplace launch");
    ck(cudaEventRecord(b), "final inplace record b");
    ck(cudaEventSynchronize(b), "final inplace sync");
    float ms = 0.0f;
    ck(cudaEventElapsedTime(&ms, a, b), "final inplace elapsed");

    std::vector<std::uint32_t> output(static_cast<std::size_t>(states));
    std::vector<unsigned long long> owner(static_cast<std::size_t>(states));
    unsigned long long processed = 0;
    int error = 0;
    ck(cudaMemcpy(output.data(), d_state, states * sizeof(std::uint32_t), cudaMemcpyDeviceToHost), "final inplace copy output");
    ck(cudaMemcpy(owner.data(), d_owner, states * sizeof(unsigned long long), cudaMemcpyDeviceToHost), "final inplace copy owner");
    ck(cudaMemcpy(&processed, d_processed, sizeof(processed), cudaMemcpyDeviceToHost), "final inplace copy processed");
    ck(cudaMemcpy(&error, d_error, sizeof(error), cudaMemcpyDeviceToHost), "final inplace copy error");
    if (error || processed != raw_labels) {
        std::cerr << "FAIL row final inplace error=" << error
                  << " processed=" << processed << " want=" << raw_labels << '\n';
        std::exit(131);
    }
    for (Rank64 r = 0; r < states; ++r) {
        if (!is_main[static_cast<std::size_t>(r)]) continue;
        if (owner[static_cast<std::size_t>(r)] == std::numeric_limits<unsigned long long>::max() ||
            output[static_cast<std::size_t>(r)] != reference[static_cast<std::size_t>(r)]) {
            std::cerr << "FAIL row final inplace arithmetic W=" << W
                      << " direction=" << (reverse ? "reverse" : "forward")
                      << " rank=" << r << '\n';
            std::exit(132);
        }
    }

    std::cout << "gridfp-reduced-row-final-inplace-microprobe"
              << " W=" << W
              << " direction=" << (reverse ? "reverse" : "forward")
              << " states=" << states
              << " components=" << raw_labels
              << " blocks=" << blocks
              << " kernel_ms=" << ms
              << " state_buffers=1"
              << " blocked_slots_become_dead=1"
              << " arithmetic=OK\n";

    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaFree(d_state);
    cudaFree(d_owner);
    cudaFree(d_processed);
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
    if (plan_only) {
        std::cout << "gridfp-reduced-row-final-inplace-plan"
                  << " W=" << W
                  << " states=" << tables.size()
                  << " components=" << motzkin_count(W - 1)
                  << " max_source_slots_bound=" << ((W + 1) / 2 + 1)
                  << " state_buffers=1"
                  << " full_stream_scratch_bytes=0\n";
        return 0;
    }
    if (W > 12) {
        std::cerr << "execution mode is intentionally limited to W<=12; use --plan-only for production widths\n";
        return 3;
    }

    int visible = 0;
    ck(cudaGetDeviceCount(&visible), "final inplace device count");
    if (visible < 1) return 4;
    ck(cudaSetDevice(0), "final inplace set device");
    install_tables(tables);
    run_row_final_inplace(W, false, tables, mod, blocks);
    run_row_final_inplace(W, true, tables, mod, blocks);
    std::cout << "ALL_OK gridfp_reduced_production_cuda_row_final_inplace=1 W=" << W << '\n';
    return 0;
}
