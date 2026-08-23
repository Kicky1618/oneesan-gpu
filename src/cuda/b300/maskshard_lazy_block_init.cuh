#pragma once

#ifndef MASKSHARD_LAZY_ZERO_BLOCK_INIT
#error "maskshard_lazy_block_init.cuh requires MASKSHARD_LAZY_ZERO_BLOCK_INIT"
#endif
#ifndef MASKSHARD_BLOCK_ORBIT
#error "lazy BLOCKED init requires MASKSHARD_BLOCK_ORBIT"
#endif
#ifndef MASKSHARD_ORBIT_AUX
#error "lazy BLOCKED init requires MASKSHARD_ORBIT_AUX"
#endif

// The row-boundary BLOCKED input is known zero. At the first HIGH position
// p=TARGET_W-1, the blocked-domain orbit visits every local BLOCKED coordinate.
// It can therefore use d=0 directly and define blockv[di] as part of the orbit,
// avoiding both a remote gather and a local zero-fill pass.
__global__ void maskshard_main_block_highorbit_lazyzero_kernel(
    Count* mainv, Count* blockv, Code n, int p
) {
    const Code step = Code(gridDim.x) * blockDim.x;
    const uint32_t pi = uint32_t((TARGET_W - 1) - p);
    const bool first_high = p == TARGET_W - 1;

    const int last_bid = D_F_BLOCK_NBLOCKS - 1;
    if (last_bid < 0) return;
    const Code block_n = D_F_BLOCK_BLOCKS[last_bid].end;
    Code di = Code(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; di < block_n; di += step) {
        const int dbid = f_find_block(di);
        const FBlock dx = D_F_BLOCK_BLOCKS[dbid];
        uint32_t dhr = 0, dlr = 0;
        maskshard_split_rank(di, dx, dhr, dlr);

        const size_t bdi = size_t(pi) * D_HIGHDESC_BLOCK_TOTAL
                         + D_HIGHDESC_BLOCK_BASE[dbid] + dhr;
        const uint32_t bdesc = D_HIGHDESC_BLOCK[bdi];
        if (highdesc_kind(bdesc) != HIGHDESC_MAIN) {
            if (first_high) blockv[di] = 0;
            continue;
        }
        const uint32_t sbid = highdesc_block(bdesc);
        const uint32_t shr = highdesc_rank(bdesc);
        const FBlock sx = D_F_MAIN_BLOCKS[sbid];
        const Code i = sx.off + Code(shr) * sx.stride + dlr;

        const size_t sdi = size_t(pi) * D_HIGHDESC_MAIN_TOTAL
                         + D_HIGHDESC_MAIN_BASE[sbid] + shr;
#ifdef MASKSHARD_BLOCK_ORBIT_AUX
        const uint32_t aux = D_MS_HIGH_ORBIT_AUX[bdi];
#else
        const uint32_t aux = D_MS_HIGH_ORBIT_AUX[sdi];
#endif
        const uint32_t ak = maskshard_orbit_aux_kind(aux);
        if (ak == MS_ORBIT_AUX_INVALID) {
            if (first_high) blockv[di] = 0;
            continue;
        }

        const Count c = mainv[i];
        const Count d = first_high ? Count(0) : blockv[di];
        if (ak == MS_ORBIT_AUX_NN) {
            const uint32_t desc = D_HIGHDESC_MAIN[sdi];
            if (highdesc_kind(desc) != HIGHDESC_MAIN) {
                if (first_high) blockv[di] = 0;
                continue;
            }
            const FBlock y = D_F_MAIN_BLOCKS[highdesc_block(desc)];
            const Code j = y.off + Code(highdesc_rank(desc)) * y.stride + dlr;
            mainv[j] = maskshard_add_mod_plain(mainv[j], c);
            mainv[i] = maskshard_add_mod_plain(c, d);
            blockv[di] = 0;
        } else {
            const FBlock y = D_F_MAIN_BLOCKS[maskshard_orbit_aux_block(aux)];
            const Code j = y.off + Code(maskshard_orbit_aux_rank(aux)) * y.stride + dlr;
            const Count cc = mainv[j];
            mainv[i] = maskshard_add_mod_plain(maskshard_add_mod_plain(c, cc), d);
            blockv[di] = c;
        }
    }
    (void)n;
}

#define maskshard_main_block_highorbit_kernel \
        maskshard_main_block_highorbit_lazyzero_kernel
