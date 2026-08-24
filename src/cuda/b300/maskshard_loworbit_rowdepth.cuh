#pragma once

#ifndef MASKSHARD_LOW_ORBIT_ROW_DEPTH
#error "maskshard_loworbit_rowdepth.cuh requires MASKSHARD_LOW_ORBIT_ROW_DEPTH"
#endif
#ifndef MASKSHARD_BLOCK_ORBIT
#error "LOW orbit row-depth pruning currently requires BLOCKED-domain orbit"
#endif
#ifndef MASKSHARD_BLOCK_ORBIT_AUX
#error "LOW orbit row-depth pruning currently requires compact BLOCKED aux"
#endif
#ifndef MASKSHARD_LOW_CLOSURE_ROW_DEPTH
#error "LOW orbit row-depth pruning reuses v0.24 LOW-all peak metadata"
#endif
#ifndef MASKSHARD_ROW_DEPTH_EXACT_IO
#error "LOW orbit row-depth pruning reuses exact HIGH peak metadata"
#endif

// v0.28: the BLOCKED-domain LOW orbit is safe to skip as a whole when the
// compressed BLOCKED state has exact frontier depth above the current row cap.
// The corresponding representative MAIN state and its orbit companion are then
// also structurally unreachable (exhaustively checked by
// factor_loworbit_rowdepth_semantics.cpp through W=12, including p=1).
//
// LOW groups fix a HIGH occupancy mask.  dhr is therefore a mask-local HIGH
// rank; convert it through D_F_HIGH_PACKED_RANK to the occupancy-major storage
// rank before indexing D_MS_ROW_DEPTH_HIGH_PEAK.  dlr is already a LOW-all
// storage rank and uses the v0.24 LOW-all peak table directly.
__global__ void maskshard_main_block_loworbit_rowdepth_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    constexpr int S = MAXW + 2;
    constexpr int H = HIGH_LUT_K;
    constexpr int FULL_CAP = TARGET_W / 2;
    const Code step = Code(gridDim.x) * blockDim.x;
    const std::uint32_t pi = std::uint32_t(LOW_LUT_K - p);
    const int last_bid = D_F_BLOCK_NBLOCKS - 1;
    if (last_bid < 0) return;
    const Code block_n = D_F_BLOCK_BLOCKS[last_bid].end;
    const int cap = min(D_MS_ROW_DEPTH_INDEX + 1, FULL_CAP);
    const bool saturated = cap >= FULL_CAP;

    Code di = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; di < block_n; di += step) {
        const int dbid = f_find_block(di);
        const FBlock dx = D_F_BLOCK_BLOCKS[dbid];
        std::uint32_t dhr = 0, dlr = 0;
        maskshard_split_rank(di, dx, dhr, dlr);

        if (!saturated) {
            if (int(dx.he) > cap) continue;
            const std::uint32_t ha = D_F_HIGH_MASK_OFF[
                std::size_t(D_F_MASK) * S + dx.he];
            const std::uint32_t hc = D_F_HIGH_MASK_CODES[ha + dhr];
            const std::uint32_t packed = D_F_HIGH_PACKED_RANK[hc];
            if (packed == 0xffffffffu) continue;
            const std::uint32_t high_storage_rank = packed >> H;
            const std::uint32_t hi = D_F_HIGH_ALL_OFF[dx.he] + high_storage_rank;
            const std::uint32_t lo = D_F_LOW_ALL_OFF[dx.he] + dlr;
            const int hp = int(D_MS_ROW_DEPTH_HIGH_PEAK[hi]);
            const int lp = int(D_MS_LOW_CLOSURE_ROW_DEPTH_LOW_ALL_PEAK[lo]);
            if ((hp > lp ? hp : lp) > cap) continue;
        }

        const std::size_t bdi = std::size_t(pi) * D_LOWDESC_BLOCK_TOTAL
                              + D_LOWDESC_BLOCK_BASE[dbid] + dlr;
        const std::uint32_t bdesc = D_LOWDESC_BLOCK[bdi];
        if (lowdesc_kind(bdesc) != LOWDESC_MAIN) continue;
        const std::uint32_t sbid = lowdesc_block(bdesc);
        const std::uint32_t slr = lowdesc_lr(bdesc);
        const FBlock sx = D_F_MAIN_BLOCKS[sbid];
        const Code i = sx.off + Code(dhr) * sx.stride + slr;

        const std::size_t sdi = std::size_t(pi) * D_LOWDESC_MAIN_TOTAL
                              + D_LOWDESC_MAIN_BASE[sbid] + slr;
        const std::uint32_t aux = D_MS_LOW_ORBIT_AUX[bdi];
        const std::uint32_t ak = maskshard_orbit_aux_kind(aux);
        if (ak == MS_ORBIT_AUX_INVALID) continue;

        const Count c = mainv[i];
        const Count d = blockv[di];
        if (ak == MS_ORBIT_AUX_NN || p == 1) {
            const std::uint32_t desc = D_LOWDESC_MAIN[sdi];
            if (lowdesc_kind(desc) != LOWDESC_MAIN) continue;
            const FBlock y = D_F_MAIN_BLOCKS[lowdesc_block(desc)];
            const Code j = y.off + Code(dhr) * y.stride + lowdesc_lr(desc);
            if (ak == MS_ORBIT_AUX_NN) {
                mainv[j] = maskshard_add_mod_plain(mainv[j], c);
                mainv[i] = maskshard_add_mod_plain(c, d);
                blockv[di] = 0;
            } else {
                const Count cc = mainv[j];
                mainv[i] = maskshard_add_mod_plain(
                    maskshard_add_mod_plain(c, cc), d);
                mainv[j] = maskshard_add_mod_plain(c, cc);
                blockv[di] = 0;
            }
        } else {
            const FBlock y = D_F_MAIN_BLOCKS[maskshard_orbit_aux_block(aux)];
            const Code j = y.off + Code(dhr) * y.stride
                         + maskshard_orbit_aux_rank(aux);
            const Count cc = mainv[j];
            mainv[i] = maskshard_add_mod_plain(
                maskshard_add_mod_plain(c, cc), d);
            blockv[di] = c;
        }
    }
    (void)n;
}

#ifdef maskshard_main_block_loworbit_kernel
#undef maskshard_main_block_loworbit_kernel
#endif
#define maskshard_main_block_loworbit_kernel \
        maskshard_main_block_loworbit_rowdepth_kernel
