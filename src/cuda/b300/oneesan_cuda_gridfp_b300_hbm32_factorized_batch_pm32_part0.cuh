
static constexpr Count PM32_MOD = 4294967291u;
static constexpr Count PM32_FOLD = 5u;
static_assert(sizeof(Count) == sizeof(unsigned int), "pm32 requires 32-bit unsigned Count");

// For p = 2^32 - 5, a uint32 wrap loses 2^32 == 5 (mod p).
// Compensate every hardware wrap with +5. The stored representative need not
// be canonical; every incoming Count is reduced before it is added.
__device__ __forceinline__ void pm32_atomic_add_mod(Count* p, Count v) {
    Count mod = D_MOD;
    if (mod != PM32_MOD) {
        atomic_add_mod(p, v);
        return;
    }
    if (v >= PM32_MOD) v -= PM32_MOD;
    if (!v) return;
    Count add = v;
    do {
        Count old = atomicAdd(reinterpret_cast<unsigned int*>(p),
                              static_cast<unsigned int>(add));
        Count neu = old + add;
        add = neu < old ? PM32_FOLD : 0u;
    } while (add);
}

__global__ void pm32_blocked_group_kernel(
    const Count* in, Code n, Count* out_main, int p
) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code stride = Code(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        Count c = in[i];
        if (!c) continue;
        MateID sm = factor_unrank_block(i);
        MateID t = oneesan::gridfp::blocked_exclude(sm, p);
        Code j = factor_rank_main(t);
        pm32_atomic_add_mod(out_main + j, c);
    }
}

__global__ void pm32_main_group_kernel(
    const Count* in, const MateID* mates, Code n,
    Count* out_main, Count* out_block, int p
) {
    Code i = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    Code stride = Code(gridDim.x) * blockDim.x;
    for (; i < n; i += stride) {
        Count c = in[i];
        if (!c) continue;
        MateID m = mates ? mates[i] : factor_unrank_main(i);
        auto z = oneesan::gridfp::include_horizontal(m, TARGET_W, p);
        if (!z.valid) continue;
        if (z.blocked)
            pm32_atomic_add_mod(out_block + factor_rank_block(z.mate), c);
        else
            pm32_atomic_add_mod(out_main + factor_rank_main(z.mate), c);
    }
}

static void process_group_pm32(
    DeviceCtx& c, int W, const WindowPlan& wp, const PreparedGroup& pg,
    int threads, size_t target
) {
    auto t0 = std::chrono::steady_clock::now();
    ck(cudaSetDevice(c.dev), "set worker");
    auto const& ms = pg.ms;
    auto const& ds = pg.ds;
    if (!ms.size && !ds.size) return;

    bool fixLow = wp.p_hi > LOW_LUT_K;
    uint32_t fmask = fixLow
        ? (pg.mo & ((1u << LOW_LUT_K) - 1u))
        : ((pg.mo >> (LOW_LUT_K + 1)) & ((1u << HIGH_LUT_K) - 1u));
    auto fmb = make_factor_main_blocks(fixLow, fmask);
    auto fdb = make_factor_block_blocks(fixLow, fmask);
    if (fmb.back().end != ms.size || fdb.back().end != ds.size) {
        std::cerr << "factor size mismatch main=" << fmb.back().end << "/" << ms.size
                  << " block=" << fdb.back().end << "/" << ds.size
                  << " fixLow=" << fixLow << " mask=" << fmask << "\n";
        std::exit(20);
    }

    int fm = int(fmb.size()), fd = int(fdb.size()), fl = fixLow ? 1 : 0;
    ck(cudaMemcpyToSymbol(D_F_MAIN_BLOCKS, fmb.data(), fmb.size()*sizeof(FBlock)), "factor main blocks");
