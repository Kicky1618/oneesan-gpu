#pragma once
// A tile contains independent groups with identical factor-block geometry.
// The address of group lane g, logical state i is base + (i << loglanes) + g.
// All factor descriptors come from the tile leader; only the inactive mask
// differs by lane. No coefficients are merged. See docs/research/groupbatch-interleaved.md.
struct BatchFactorIoCfg {
  FBlock mainb[64];
  FBlock blockb[32];
  Code main_n, block_n, main_off, block_off;
  uint32_t mask;
  int main_nb, block_nb, fix_low;
  int loglanes;
};
struct BatchFactorIoTask {
  uint32_t cfg;
  uint16_t block, bid;
  Code begin;
};
static_assert(sizeof(BatchFactorIoTask) == 16, "compact IO task metadata");
#ifndef GROUPBATCH_CHUNK_ELEMS
#define GROUPBATCH_CHUNK_ELEMS (TARGET_W >= 28 ? 65536ULL : 8192ULL)
#endif
static constexpr Code GROUPBATCH_CHUNK = Code(GROUPBATCH_CHUNK_ELEMS);
static_assert(GROUPBATCH_CHUNK >= 32, "groupbatch chunks must hold one warp");
__device__ __forceinline__ int batch_factor_find(Code i, const FBlock *b, int nb) {
  int lo = 0, hi = nb;
  while (lo < hi) {
    int m = (lo + hi) >> 1;
    if (i < b[m].end)
      hi = m;
    else
      lo = m + 1;
  }
  return lo;
}
__device__ __forceinline__ Code batch_factor_global_main(Code i, const BatchFactorIoCfg *cp) {
  const BatchFactorIoCfg &c = *cp;
  constexpr int L = LOW_LUT_K, H = HIGH_LUT_K, S = MAXW + 2;
  int bi = batch_factor_find(i, c.mainb, c.main_nb);
  FBlock x = c.mainb[bi];
  Code r = i - x.off;
  uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0,
           lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
  uint32_t har, lar;
  if (c.fix_low) {
    har = hr;
    uint32_t lc = D_F_LOW_MASK_CODES[D_F_LOW_MASK_OFF[size_t(c.mask) * S + x.hs] + lr];
    lar = D_F_LOW_DENSE_PACKED_RANK[lc] >> L;
  } else {
    uint32_t hc = D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(c.mask) * S + x.he] + hr];
    har = D_F_HIGH_PACKED_RANK[hc] >> H;
    lar = lr;
  }
  Code rank = D_F_HIGH_MAIN_BASE[D_F_HIGH_ALL_OFF[x.he] + har];
  if (x.c > N)
    rank += D_FULL_DP[L][x.he];
  if (x.c > R && x.he > 0)
    rank += D_FULL_DP[L][x.he - 1];
  return rank + lar;
}
__device__ __forceinline__ Code batch_factor_global_block(Code i, const BatchFactorIoCfg *cp) {
  const BatchFactorIoCfg &c = *cp;
  constexpr int L = LOW_LUT_K, H = HIGH_LUT_K, S = MAXW + 2;
  int bi = batch_factor_find(i, c.blockb, c.block_nb);
  FBlock x = c.blockb[bi];
  Code r = i - x.off;
  uint32_t hr = x.stride ? uint32_t(r / x.stride) : 0,
           lr = x.stride ? uint32_t(r - Code(hr) * x.stride) : 0;
  uint32_t har, lar;
  if (c.fix_low) {
    har = hr;
    uint32_t lc = D_F_LOW_MASK_CODES[D_F_LOW_MASK_OFF[size_t(c.mask) * S + x.hs] + lr];
    lar = D_F_LOW_DENSE_PACKED_RANK[lc] >> L;
  } else {
    uint32_t hc = D_F_HIGH_MASK_CODES[D_F_HIGH_MASK_OFF[size_t(c.mask) * S + x.he] + hr];
    har = D_F_HIGH_PACKED_RANK[hc] >> H;
    lar = lr;
  }
  return D_F_HIGH_BLOCK_BASE[D_F_HIGH_ALL_OFF[x.he] + har] + lar;
}
// For p=2^32-c, a wrapped u32 atomicAdd loses 2^32 == c (mod p).
// Feed c back with another native atomicAdd.  A compensation can itself wrap
// after interleaving with another thread, so keep feeding c back until one add
// does not wrap.  The mode guard p>2^31 makes final canonicalization one subtract.
__device__ __forceinline__ Count groupbatch_wrap32_normalize(Count x) {
  Count mod = D_MOD;
  return x >= mod ? x - mod : x;
}
__device__ __forceinline__ void groupbatch_wrap32_atomic_add(Count *p, Count v) {
  if (!v)
    return;
  unsigned int add = unsigned(v), gap = 0u - unsigned(D_MOD);
  for (;;) {
    unsigned int old = atomicAdd(reinterpret_cast<unsigned int *>(p), add);
    if (old <= 0xffffffffu - add)
      return;
    add = gap;
  }
}
template <bool SCATTER, bool NORMALIZE_BLOCK>
__global__ void batch_factor_io_kernel(Count *arena, const BatchFactorIoCfg *cfgs,
                                       const BatchFactorIoTask *tasks, Code nt) {
  Code ti = blockIdx.x;
  if (ti >= nt)
    return;
  auto t = tasks[ti];
  const BatchFactorIoCfg *c = cfgs + t.cfg;
  int lg = c->loglanes;
  unsigned lane = threadIdx.x & ((1u << lg) - 1);
  constexpr Code CH = GROUPBATCH_CHUNK;
  FBlock x = t.block ? c->blockb[t.bid] : c->mainb[t.bid];
  Code end = min(x.end - x.off, t.begin + (CH >> lg)), bo = t.block ? c->block_off : c->main_off;
  constexpr int L = LOW_LUT_K, H = HIGH_LUT_K, S = MAXW + 2;
  for (Code z = t.begin + (threadIdx.x >> lg); z < end; z += (blockDim.x >> lg)) {
    Code i = x.off + z;
    auto qr = oneesan::invariant_divmod(z, x.stride, x.reciprocal);
    uint32_t hr = qr.quotient, lr = qr.remainder;
    uint32_t har, lar;
    if (c->fix_low) {
      har = hr;
      lar = D_F_LOW_MASK_ALL_RANK
          [D_F_LOW_MASK_OFF[size_t(cfgs[t.cfg + lane].mask) * S + (t.block ? x.he : x.hs)] + lr];
    } else {
      har = D_F_HIGH_MASK_ALL_RANK[D_F_HIGH_MASK_OFF[size_t(cfgs[t.cfg + lane].mask) * S + x.he] +
                                   hr];
      lar = lr;
    }
    Code g;
    if (t.block)
      g = D_F_HIGH_BLOCK_BASE[D_F_HIGH_ALL_OFF[x.he] + har] + lar;
    else {
      g = D_F_HIGH_MAIN_BASE[D_F_HIGH_ALL_OFF[x.he] + har];
      if (x.c > N)
        g += D_FULL_DP[L][x.he];
      if (x.c > R && x.he > 0)
        g += D_FULL_DP[L][x.he - 1];
      g += lar;
    }
    Count *bp = arena + bo + lane + (i << lg);
    if constexpr (SCATTER) {
      Count v = *bp;
      if constexpr (NORMALIZE_BLOCK) {
        if (t.block)
          v = groupbatch_wrap32_normalize(v);
      }
      if (t.block)
        global_store_block(g, v);
      else
        global_store_main(g, v);
    } else
      *bp = t.block ? global_load_block(g) : global_load_main(g);
  }
}

__device__ __forceinline__ Count add_mod_plain(Count a, Count b);
struct BatchOwnerTask {
  uint32_t cfg, bid;
  Code begin;
};
struct BatchCrossTask {
  uint32_t cfg, bid, sel_off, sel_count;
  Code begin;
  uint64_t reciprocal;
};
struct BatchP1DestTask {
  uint32_t cfg, rec_off, rec_count, he;
  Code begin;
  uint64_t reciprocal;
};

template <bool WRAP32>
__global__ void batch_low_owner_kernel(Count *arena, const BatchFactorIoCfg *cfgs,
                                       const BatchOwnerTask *tasks, Code nt, int p) {
  Code ti = blockIdx.x;
  if (ti >= nt)
    return;
  auto t = tasks[ti];
  const BatchFactorIoCfg *c = cfgs + t.cfg;
  int lg = c->loglanes;
  unsigned lane = threadIdx.x & ((1u << lg) - 1);
  FBlock bx = c->blockb[t.bid];
  constexpr Code CH = GROUPBATCH_CHUNK;
  Code len = bx.end - bx.off, end = min(len, t.begin + (CH >> lg));
  constexpr uint32_t B = 20, M = (1u << B) - 1u;
  Code lowTotal = D_F_LOW_ALL_OFF[MAXW + 1];
  Count *mainv = arena + c->main_off + lane, *blockv = arena + c->block_off + lane;
  for (Code z = t.begin + (threadIdx.x >> lg); z < end; z += (blockDim.x >> lg)) {
    auto qr = oneesan::invariant_divmod(z, bx.stride, bx.reciprocal);
    uint32_t hr = qr.quotient, blr = qr.remainder;
    Code obi = Code(p - 1) * lowTotal + D_F_LOW_ALL_OFF[bx.he] + blr;
    unsigned long long r = D_PR_OWNER_BLOCK_REC[obi];
    uint32_t lr = uint32_t(r) & M, mlr = uint32_t(r >> B) & M, cvv = uint32_t(r >> (2 * B)) & 3u,
             ty = uint32_t(r >> (2 * B + 2)) & 3u;
    FBlock x = c->mainb[3 * int(bx.he) + int(cvv)], dx = x;
    if (p == LOW_LUT_K) {
      int dc = (ty == 1 ? int(R) : int(::L));
      dx = c->mainb[3 * int(bx.he) + dc];
    }
    Code i = x.off + Code(hr) * x.stride + lr, j = dx.off + Code(hr) * dx.stride + mlr,
         q = bx.off + z;
    Count cv = mainv[(i) << lg], d = blockv[(q) << lg], ca = 0;
    if constexpr (WRAP32)
      d = groupbatch_wrap32_normalize(d);
    uint32_t meta = D_PR_OWNER_BLOCK_CLOSURE_META[obi], co = meta >> 4, cn = meta & 15u;
    for (uint32_t a = 0; a < cn; ++a) {
      uint32_t sr = D_PR_OWNER_BLOCK_CLOSURE_SRC[co + a], cc = sr >> 18,
               sl = sr & ((1u << 18) - 1u);
      FBlock sx = c->mainb[3 * int(bx.he) + int(cc)];
      Count v = mainv[(sx.off + Code(hr) * sx.stride + sl) << lg];
      if (v)
        ca = add_mod_plain(ca, v);
    }
    if (ty == 0) {
      mainv[(j) << lg] = add_mod_plain(mainv[(j) << lg], cv);
      mainv[(i) << lg] = add_mod_plain(cv, d);
      blockv[(q) << lg] = ca;
    } else {
      Count cc = mainv[(j) << lg], pair = add_mod_plain(cv, cc);
      mainv[(i) << lg] = add_mod_plain(pair, d);
      if (p == 1) {
        mainv[(j) << lg] = pair;
        blockv[(q) << lg] = 0;
      } else
        blockv[(q) << lg] = add_mod_plain(cv, ca);
    }
  }
}

template <bool WRAP32>
__global__ void batch_high_owner_kernel(Count *arena, const BatchFactorIoCfg *cfgs,
                                        const BatchOwnerTask *tasks, Code nt, int p) {
  Code ti = blockIdx.x;
  if (ti >= nt)
    return;
  auto t = tasks[ti];
  const BatchFactorIoCfg *c = cfgs + t.cfg;
  int lg = c->loglanes;
  unsigned lane = threadIdx.x & ((1u << lg) - 1);
  FBlock bx = c->blockb[t.bid];
  constexpr Code CH = GROUPBATCH_CHUNK;
  Code len = bx.end - bx.off, end = min(len, t.begin + (CH >> lg));
  constexpr uint32_t B = 20, M = (1u << B) - 1u;
  Code highTotal = D_F_HIGH_ALL_OFF[MAXW + 1];
  Count *mainv = arena + c->main_off + lane, *blockv = arena + c->block_off + lane;
  for (Code z = t.begin + (threadIdx.x >> lg); z < end; z += (blockDim.x >> lg)) {
    auto qr = oneesan::invariant_divmod(z, bx.stride, bx.reciprocal);
    uint32_t bhr = qr.quotient, lr = qr.remainder;
    Code obi = Code(p - LOW_LUT_K - 1) * highTotal + D_F_HIGH_ALL_OFF[bx.he] + bhr;
    unsigned long long r = D_HP_OWNER_BLOCK_REC[obi];
    uint32_t hr = uint32_t(r) & M, mhr = uint32_t(r >> B) & M, cvv = uint32_t(r >> (2 * B)) & 3u,
             ty = uint32_t(r >> (2 * B + 2)) & 3u, tc = uint32_t(r >> (2 * B + 4)) & 3u;
    int sd = (cvv == uint32_t(::L) ? 1 : cvv == uint32_t(R) ? -1 : 0), he = int(bx.he) - sd;
    FBlock x = c->mainb[3 * he + int(cvv)];
    int td = (tc == uint32_t(::L) ? 1 : tc == uint32_t(R) ? -1 : 0), the = int(bx.he) - td;
    FBlock dx = c->mainb[3 * the + int(tc)];
    Code i = x.off + Code(hr) * x.stride + lr, j = dx.off + Code(mhr) * dx.stride + lr,
         q = bx.off + z;
    Count cv = mainv[(i) << lg], d = blockv[(q) << lg], ca = 0;
    if constexpr (WRAP32)
      d = groupbatch_wrap32_normalize(d);
    uint32_t meta = D_HP_OWNER_BLOCK_CLOSURE_META[obi], co = meta >> 4, cn = meta & 15u;
    for (uint32_t a = 0; a < cn; ++a) {
      uint32_t sr = D_HP_OWNER_BLOCK_CLOSURE_SRC[co + a], cc = sr >> 18,
               sh = sr & ((1u << 18) - 1u);
      int sdc = (cc == uint32_t(::L) ? 1 : cc == uint32_t(R) ? -1 : 0), she = int(bx.he) - sdc;
      FBlock sx = c->mainb[3 * she + int(cc)];
      Count v = mainv[(sx.off + Code(sh) * sx.stride + lr) << lg];
      if (v)
        ca = add_mod_plain(ca, v);
    }
    if (ty == 0) {
      mainv[(j) << lg] = add_mod_plain(mainv[(j) << lg], cv);
      mainv[(i) << lg] = add_mod_plain(cv, d);
      blockv[(q) << lg] = ca;
    } else {
      Count cc = mainv[(j) << lg];
      mainv[(i) << lg] = add_mod_plain(add_mod_plain(cv, cc), d);
      blockv[(q) << lg] = add_mod_plain(cv, ca);
    }
  }
}

template <bool WRAP32, bool P1, bool MATCHLUT>
__global__ void batch_low_highrr_kernel(Count *arena, const BatchFactorIoCfg *cfgs,
                                        const BatchCrossTask *tasks, Code nt) {
  Code ti = blockIdx.x;
  if (ti >= nt)
    return;
  auto t = tasks[ti];
  const BatchFactorIoCfg *c = cfgs + t.cfg;
  int lg = c->loglanes;
  unsigned lane = threadIdx.x & ((1u << lg) - 1);
  FBlock x = c->mainb[t.bid];
  constexpr Code CH = GROUPBATCH_CHUNK;
  Code rows =
           x.stride ? oneesan::invariant_divmod(x.end - x.off, x.stride, x.reciprocal).quotient : 0,
       total = rows * Code(t.sel_count), end = min(total, t.begin + (CH >> lg));
  constexpr uint32_t B = 20, M = (1u << B) - 1u;
  constexpr int H = HIGH_LUT_K, S = MAXW + 2;
  constexpr uint32_t HM = (1u << H) - 1u;
  Count *mainv = arena + c->main_off + lane, *blockv = arena + c->block_off + lane;
  for (Code z = t.begin + (threadIdx.x >> lg); z < end; z += (blockDim.x >> lg)) {
    auto qr = oneesan::invariant_divmod(z, uint32_t(t.sel_count), t.reciprocal);
    uint32_t hr = qr.quotient, k = qr.remainder;
    unsigned long long rec = D_PR_HIGHRR_REC[t.sel_off + k];
    uint32_t lr = uint32_t(rec) & M, dlr = uint32_t(rec >> B) & M;
    int depth = int((rec >> (2 * B)) & 63u);
    Count v = mainv[(x.off + Code(hr) * x.stride + lr) << lg];
    if (!v)
      continue;
    size_t hmo = D_F_HIGH_MASK_OFF[size_t(cfgs[t.cfg + lane].mask) * S + x.he];
    uint32_t hc = D_F_HIGH_MASK_CODES[hmo + hr];
    int match = -1;
    if constexpr (MATCHLUT) {
      uint32_t har = D_F_HIGH_MASK_ALL_RANK[hmo + hr];
      unsigned long long mw = D_CROSS_HIGH_MATCH[D_F_HIGH_ALL_OFF[x.he] + har];
      match = (depth >= 1 && depth <= H) ? int((mw >> (4 * (depth - 1))) & 15u) - 1 : -1;
    } else {
      int ss = depth;
#pragma unroll
      for (int a = 0; a < H; ++a) {
        auto mv = MateValue((hc >> (2 * a)) & 3u);
        if (mv == L) {
          if (--ss == 0) {
            match = a;
            break;
          }
        } else if (mv == R)
          ++ss;
      }
    }
    if (match < 0)
      continue;
    uint32_t nhc = (hc & ~(3u << (2 * match))) | (uint32_t(R) << (2 * match));
    uint32_t hp = D_F_HIGH_PACKED_RANK[nhc], nhr = hp & HM;
    int nhe = int(x.he) - 2;
    if constexpr (P1) {
      FBlock dx = c->mainb[3 * nhe + int(x.c)];
      atomic_add_mod(mainv + ((dx.off + Code(nhr) * dx.stride + dlr) << lg), v);
    } else {
      FBlock bx = c->blockb[nhe];
      Code di = bx.off + Code(nhr) * bx.stride + dlr;
      if constexpr (WRAP32)
        groupbatch_wrap32_atomic_add(blockv + (di << lg), v);
      else
        atomic_add_mod(blockv + (di << lg), v);
    }
  }
}

__global__ void batch_low_inv_p1_kernel(Count *arena, const BatchFactorIoCfg *cfgs,
                                        const BatchCrossTask *tasks, Code nt) {
  Code ti = blockIdx.x;
  if (ti >= nt)
    return;
  auto t = tasks[ti];
  const BatchFactorIoCfg *c = cfgs + t.cfg;
  int lg = c->loglanes;
  unsigned lane = threadIdx.x & ((1u << lg) - 1);
  FBlock x = c->mainb[t.bid];
  constexpr Code CH = GROUPBATCH_CHUNK;
  Code rows =
           x.stride ? oneesan::invariant_divmod(x.end - x.off, x.stride, x.reciprocal).quotient : 0,
       total = rows * Code(t.sel_count), end = min(total, t.begin + (CH >> lg));
  constexpr uint32_t IB = 18, IM = (1u << IB) - 1u, P = 36;
  constexpr unsigned long long PM = (1ULL << P) - 1ULL;
  Count *mainv = arena + c->main_off + lane;
  for (Code z = t.begin + (threadIdx.x >> lg); z < end; z += (blockDim.x >> lg)) {
    auto qr = oneesan::invariant_divmod(z, uint32_t(t.sel_count), t.reciprocal);
    uint32_t hr = qr.quotient, k = qr.remainder;
    unsigned long long r = D_PR_CLOSURE_INV_REC[t.sel_off + k];
    uint32_t dlr = uint32_t(r) & IM;
    unsigned long long payload = (r >> IB) & PM;
    uint32_t cnt = uint32_t(r >> (IB + P)) & 7u, dc = uint32_t(r >> (IB + P + 3)) & 3u;
    Count acc = 0;
    for (uint32_t j = 0; j < cnt; ++j) {
      uint32_t lr = cnt <= 2 ? uint32_t(payload >> (18 * j)) & IM
                             : D_PR_CLOSURE_INV_SRC[uint32_t(payload) + j];
      Count v = mainv[(x.off + Code(hr) * x.stride + lr) << lg];
      if (v)
        acc = add_mod_plain(acc, v);
    }
    if (acc) {
      FBlock dx = c->mainb[3 * x.he + int(dc)];
      atomic_add_mod(mainv + ((dx.off + Code(hr) * dx.stride + dlr) << lg), acc);
    }
  }
}
__global__ void batch_low_p1_dest_kernel(Count *arena, const BatchFactorIoCfg *cfgs,
                                         const BatchP1DestTask *tasks, Code nt) {
  Code ti = blockIdx.x;
  if (ti >= nt)
    return;
  auto t = tasks[ti];
  const BatchFactorIoCfg *c = cfgs + t.cfg;
  int lg = c->loglanes;
  unsigned lane = threadIdx.x & ((1u << lg) - 1);
  constexpr Code CH = GROUPBATCH_CHUNK;
  Code total = Code(t.rec_count) *
               (D_F_HIGH_MASK_OFF[size_t(cfgs[t.cfg + lane].mask) * (MAXW + 2) + int(t.he) + 1] -
                D_F_HIGH_MASK_OFF[size_t(cfgs[t.cfg + lane].mask) * (MAXW + 2) + int(t.he)]),
       end = min(total, t.begin + (CH >> lg));
  constexpr uint32_t IM = (1u << 18) - 1u;
  Count *mainv = arena + c->main_off + lane;
  for (Code z = t.begin + (threadIdx.x >> lg); z < end; z += (blockDim.x >> lg)) {
    auto qr = oneesan::invariant_divmod(z, uint32_t(t.rec_count), t.reciprocal);
    uint32_t hr = qr.quotient, k = qr.remainder;
    unsigned long long r = D_P1_DEST_REC[t.rec_off + k];
    uint32_t dlr = uint32_t(r) & IM, dc = uint32_t(r >> 18) & 3u,
             so = uint32_t(r >> 20) & 0x0fffffffu, cnt = uint32_t(r >> 48) & 7u;
    Count acc = 0;
    for (uint32_t j = 0; j < cnt; ++j) {
      uint32_t sr = D_P1_DEST_SRC[so + j], lr = sr & IM, sc = sr >> 18;
      FBlock sx = c->mainb[3 * int(t.he) + int(sc)];
      Count v = mainv[(sx.off + Code(hr) * sx.stride + lr) << lg];
      if (v)
        acc = add_mod_plain(acc, v);
    }
    if (acc) {
      FBlock dx = c->mainb[3 * int(t.he) + int(dc)];
      Count *dp = mainv + ((dx.off + Code(hr) * dx.stride + dlr) << lg);
      *dp = add_mod_plain(*dp, acc);
    }
  }
}

template <bool WRAP32, bool MATCHLUT>
__global__ void batch_high_crossll_kernel(Count *arena, const BatchFactorIoCfg *cfgs,
                                          const BatchCrossTask *tasks, Code nt) {
  Code ti = blockIdx.x;
  if (ti >= nt)
    return;
  auto t = tasks[ti];
  const BatchFactorIoCfg *c = cfgs + t.cfg;
  int lg = c->loglanes;
  unsigned lane = threadIdx.x & ((1u << lg) - 1);
  FBlock x = c->mainb[t.bid];
  constexpr Code CH = GROUPBATCH_CHUNK;
  Code lc = x.stride, total = lc * Code(t.sel_count), end = min(total, t.begin + (CH >> lg));
  constexpr uint32_t B = 20, M = (1u << B) - 1u;
  constexpr int LOWK = LOW_LUT_K, S = MAXW + 2;
  constexpr uint32_t LM = (1u << LOWK) - 1u;
  Count *mainv = arena + c->main_off + lane, *blockv = arena + c->block_off + lane;
  for (Code z = t.begin + (threadIdx.x >> lg); z < end; z += (blockDim.x >> lg)) {
    auto qr = oneesan::invariant_divmod(z, uint32_t(lc), t.reciprocal);
    uint32_t k = qr.quotient, lr = qr.remainder;
    unsigned long long r = D_HP_CROSSLL_REC[t.sel_off + k];
    uint32_t hr = uint32_t(r) & M, bhr = uint32_t(r >> B) & M;
    int depth = int((r >> (2 * B)) & 63u);
    Count v = mainv[(x.off + Code(hr) * x.stride + lr) << lg];
    if (!v)
      continue;
    size_t lmo = D_F_LOW_MASK_OFF[size_t(cfgs[t.cfg + lane].mask) * S + x.hs];
    uint32_t code = D_F_LOW_MASK_CODES[lmo + lr];
    int match = -1;
    if constexpr (MATCHLUT) {
      uint32_t lar = D_F_LOW_MASK_ALL_RANK[lmo + lr];
      unsigned long long mw = D_CROSS_LOW_MATCH[D_F_LOW_ALL_OFF[x.hs] + lar];
      match = (depth >= 1 && depth <= LOWK) ? int((mw >> (4 * (depth - 1))) & 15u) - 1 : -1;
    } else {
      int ss = depth;
#pragma unroll
      for (int a = LOWK - 1; a >= 0; --a) {
        auto mv = MateValue((code >> (2 * a)) & 3u);
        if (mv == R) {
          if (--ss == 0) {
            match = a;
            break;
          }
        } else if (mv == oneesan::gridfp::L)
          ++ss;
      }
    }
    if (match < 0)
      continue;
    uint32_t nc = (code & ~(3u << (2 * match))) | (uint32_t(::L) << (2 * match));
    uint32_t nlr = D_F_LOW_DENSE_PACKED_RANK[nc] & LM;
    int nh = int(x.hs) - 2;
    FBlock bx = c->blockb[nh];
    Code di = bx.off + Code(bhr) * bx.stride + nlr;
    if constexpr (WRAP32)
      groupbatch_wrap32_atomic_add(blockv + (di << lg), v);
    else
      atomic_add_mod(blockv + (di << lg), v);
  }
}
