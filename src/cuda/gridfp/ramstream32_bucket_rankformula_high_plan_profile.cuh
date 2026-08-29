#pragma once

#ifndef P10DC_RANKFORMULA_HIGH_PLAN_PROFILE
#define P10DC_RANKFORMULA_HIGH_PLAN_PROFILE 0
#endif
static_assert(P10DC_RANKFORMULA_HIGH_PLAN_PROFILE == 0 ||
              P10DC_RANKFORMULA_HIGH_PLAN_PROFILE == 1,
              "P10DC_RANKFORMULA_HIGH_PLAN_PROFILE must be 0 or 1");

#if P10DC_RANKFORMULA_HIGH_PLAN_PROFILE
// phase 0=forward, 1=reverse. Count only blockIdx.x==0 so an orbit context is
// represented once even though the column stripes are replicated across gx.
__device__ unsigned long long D_P10DC_RF_PLAN_LOCAL_CTX[2][9];
__device__ unsigned long long D_P10DC_RF_PLAN_LOCAL_COLS[2][9];
__device__ unsigned long long D_P10DC_RF_PLAN_CROSS_CTX[2][16];
__device__ unsigned long long D_P10DC_RF_PLAN_CROSS_COLS[2][16];
// Joint histogram keeps the correlation lost by the marginal arrays above.
// It is profiling-only global memory, so the extra atomic does not affect
// production builds and lets adaptive local/cross dispatch thresholds be chosen
// from measured column traffic rather than guesses.
__device__ unsigned long long D_P10DC_RF_PLAN_JOINT_CTX[2][9][16];
__device__ unsigned long long D_P10DC_RF_PLAN_JOINT_COLS[2][9][16];

__device__ __forceinline__ void p10dc_rankformula_profile_high_plan(
    const P10DCDirectHighResolvedCtx& c, uint32_t phase
) {
    if (blockIdx.x != 0 || (uint32_t(threadIdx.x) & 31u) != 0u) return;
    phase &= 1u;
    const uint32_t ln = c.local_n <= 8u ? uint32_t(c.local_n) : 8u;
    const uint32_t cd = c.cross_depth <= 15u ? c.cross_depth : 15u;
    const unsigned long long cols = static_cast<unsigned long long>(c.xb.cols);
    atomicAdd(&D_P10DC_RF_PLAN_LOCAL_CTX[phase][ln], 1ull);
    atomicAdd(&D_P10DC_RF_PLAN_LOCAL_COLS[phase][ln], cols);
    atomicAdd(&D_P10DC_RF_PLAN_CROSS_CTX[phase][cd], 1ull);
    atomicAdd(&D_P10DC_RF_PLAN_CROSS_COLS[phase][cd], cols);
    atomicAdd(&D_P10DC_RF_PLAN_JOINT_CTX[phase][ln][cd], 1ull);
    atomicAdd(&D_P10DC_RF_PLAN_JOINT_COLS[phase][ln][cd], cols);
}

static inline void p10dc_rankformula_high_plan_profile_reset() {
    unsigned long long zlocal[2][9]{};
    unsigned long long zcross[2][16]{};
    unsigned long long zjoint[2][9][16]{};
    ck(cudaMemcpyToSymbol(D_P10DC_RF_PLAN_LOCAL_CTX, zlocal, sizeof(zlocal)),
       "rankformula profile reset local ctx");
    ck(cudaMemcpyToSymbol(D_P10DC_RF_PLAN_LOCAL_COLS, zlocal, sizeof(zlocal)),
       "rankformula profile reset local cols");
    ck(cudaMemcpyToSymbol(D_P10DC_RF_PLAN_CROSS_CTX, zcross, sizeof(zcross)),
       "rankformula profile reset cross ctx");
    ck(cudaMemcpyToSymbol(D_P10DC_RF_PLAN_CROSS_COLS, zcross, sizeof(zcross)),
       "rankformula profile reset cross cols");
    ck(cudaMemcpyToSymbol(D_P10DC_RF_PLAN_JOINT_CTX, zjoint, sizeof(zjoint)),
       "rankformula profile reset joint ctx");
    ck(cudaMemcpyToSymbol(D_P10DC_RF_PLAN_JOINT_COLS, zjoint, sizeof(zjoint)),
       "rankformula profile reset joint cols");
}

static inline void p10dc_rankformula_high_plan_profile_report(int device) {
    unsigned long long lctx[2][9]{}, lcols[2][9]{};
    unsigned long long cctx[2][16]{}, ccols[2][16]{};
    unsigned long long jctx[2][9][16]{}, jcols[2][9][16]{};
    ck(cudaSetDevice(device), "rankformula profile report device");
    ck(cudaMemcpyFromSymbol(lctx, D_P10DC_RF_PLAN_LOCAL_CTX, sizeof(lctx)),
       "rankformula profile local ctx");
    ck(cudaMemcpyFromSymbol(lcols, D_P10DC_RF_PLAN_LOCAL_COLS, sizeof(lcols)),
       "rankformula profile local cols");
    ck(cudaMemcpyFromSymbol(cctx, D_P10DC_RF_PLAN_CROSS_CTX, sizeof(cctx)),
       "rankformula profile cross ctx");
    ck(cudaMemcpyFromSymbol(ccols, D_P10DC_RF_PLAN_CROSS_COLS, sizeof(ccols)),
       "rankformula profile cross cols");
    ck(cudaMemcpyFromSymbol(jctx, D_P10DC_RF_PLAN_JOINT_CTX, sizeof(jctx)),
       "rankformula profile joint ctx");
    ck(cudaMemcpyFromSymbol(jcols, D_P10DC_RF_PLAN_JOINT_COLS, sizeof(jcols)),
       "rankformula profile joint cols");
    for (uint32_t phase = 0; phase < 2; ++phase) {
        const char* name = phase ? "reverse" : "forward";
        unsigned long long total_ctx = 0, total_cols = 0, local_reads = 0;
        unsigned long long cross_ctx = 0, cross_cols = 0;
        for (uint32_t n = 0; n <= 8; ++n) {
            total_ctx += lctx[phase][n];
            total_cols += lcols[phase][n];
            local_reads += lcols[phase][n] * n;
            if (lctx[phase][n] || lcols[phase][n])
                std::cerr << "rankformula_plan_profile device=" << device
                          << " phase=" << name << " local_n=" << n
                          << " contexts=" << lctx[phase][n]
                          << " columns=" << lcols[phase][n] << '\n';
        }
        for (uint32_t d = 0; d < 16; ++d) {
            if (d) { cross_ctx += cctx[phase][d]; cross_cols += ccols[phase][d]; }
            if (cctx[phase][d] || ccols[phase][d])
                std::cerr << "rankformula_plan_profile device=" << device
                          << " phase=" << name << " cross_depth=" << d
                          << " contexts=" << cctx[phase][d]
                          << " columns=" << ccols[phase][d] << '\n';
        }
        unsigned long long no_cross_small_cols[9]{};
        for (uint32_t n = 0; n <= 8; ++n) {
            for (uint32_t d = 0; d < 16; ++d) {
                if (jctx[phase][n][d] || jcols[phase][n][d])
                    std::cerr << "rankformula_plan_profile_joint device=" << device
                              << " phase=" << name
                              << " local_n=" << n
                              << " cross_depth=" << d
                              << " contexts=" << jctx[phase][n][d]
                              << " columns=" << jcols[phase][n][d] << '\n';
            }
            no_cross_small_cols[n] = jcols[phase][n][0];
        }
        const double avg_local = total_cols
            ? double(local_reads) / double(total_cols) : 0.0;
        const double cross_col_frac = total_cols
            ? double(cross_cols) / double(total_cols) : 0.0;
        const double cross_ctx_frac = total_ctx
            ? double(cross_ctx) / double(total_ctx) : 0.0;
        unsigned long long no_cross_le2 = 0, no_cross_le4 = 0, no_cross_le6 = 0;
        for (uint32_t n = 0; n <= 8; ++n) {
            if (n <= 2) no_cross_le2 += no_cross_small_cols[n];
            if (n <= 4) no_cross_le4 += no_cross_small_cols[n];
            if (n <= 6) no_cross_le6 += no_cross_small_cols[n];
        }
        auto frac = [total_cols](unsigned long long x) {
            return total_cols ? double(x) / double(total_cols) : 0.0;
        };
        std::cerr << "rankformula_plan_profile_summary device=" << device
                  << " phase=" << name
                  << " contexts=" << total_ctx
                  << " columns=" << total_cols
                  << " local_source_reads=" << local_reads
                  << " avg_local_sources_per_column=" << avg_local
                  << " cross_context_fraction=" << cross_ctx_frac
                  << " cross_column_fraction=" << cross_col_frac
                  << " no_cross_local_le2_column_fraction=" << frac(no_cross_le2)
                  << " no_cross_local_le4_column_fraction=" << frac(no_cross_le4)
                  << " no_cross_local_le6_column_fraction=" << frac(no_cross_le6)
                  << '\n';
    }
}
#else
__device__ __forceinline__ void p10dc_rankformula_profile_high_plan(
    const P10DCDirectHighResolvedCtx&, uint32_t) {}
static inline void p10dc_rankformula_high_plan_profile_reset() {}
static inline void p10dc_rankformula_high_plan_profile_report(int) {}
#endif
