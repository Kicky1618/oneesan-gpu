#pragma once
// Equal-work groups are ordered by the actual occupancy mask, not its reversed
// enumeration index. This orders global ranks monotonically in every tile row.
static bool groupbatch_work_order(const PreparedGroup &a, const PreparedGroup &b) {
  return a.work != b.work ? a.work > b.work : a.mo < b.mo;
}

// Read at schedule construction; the chosen layout is immutable during graph replay.
static int groupbatch_lane_log() {
  int lanes = 16;
  if (const char *e = std::getenv("GRIDFP_GROUPBATCH_LANES")) {
    char *end = nullptr;
    long value = std::strtol(e, &end, 10);
    if (end == e || *end || value < 1 || value > 32 || (value & (value - 1)))
      throw std::runtime_error("GRIDFP_GROUPBATCH_LANES must be 1,2,4,8,16,32");
    lanes = int(value);
  }
  return __builtin_ctz(unsigned(lanes));
}

static bool same_batch_geometry(const BatchFactorIoCfg &a, const BatchFactorIoCfg &b) {
  if (a.fix_low != b.fix_low || a.main_n != b.main_n || a.block_n != b.block_n ||
      a.main_nb != b.main_nb || a.block_nb != b.block_nb ||
      __builtin_popcount(a.mask) != __builtin_popcount(b.mask))
    return false;
  auto equal = [](const FBlock &x, const FBlock &y) {
    return x.off == y.off && x.end == y.end && x.stride == y.stride && x.he == y.he &&
           x.hs == y.hs && x.c == y.c;
  };
  for (int i = 0; i < a.main_nb; ++i)
    if (!equal(a.mainb[i], b.mainb[i]))
      return false;
  for (int i = 0; i < a.block_nb; ++i)
    if (!equal(a.blockb[i], b.blockb[i]))
      return false;
  return true;
}

// All admissions and allocations use this one-copy, 64-element-aligned layout.
static Code groupbatch_aligned_states(Code n) { return (n + 63) & ~Code(63); }
static size_t groupbatch_group_bytes(Code main, Code blocked) {
  return size_t(groupbatch_aligned_states(main) + groupbatch_aligned_states(blocked)) * sizeof(Count);
}

static void prepare_window_batches(int W, PreparedWindow &pw, size_t target) {
  auto al = groupbatch_aligned_states;
  pw.batches.clear();
  size_t q0 = 0;
  while (q0 < pw.groups.size()) {
    PreparedBatchHost b;
    Code used = 0;
    size_t q = q0;
    bool fixLow = pw.wp.p_hi > LOW_LUT_K;
    for (; q < pw.groups.size(); ++q) {
      auto const &pg = pw.groups[q];
      if (!pg.ms.size && !pg.ds.size)
        continue;
      Code need = groupbatch_group_bytes(pg.ms.size, pg.ds.size) / sizeof(Count);
      uint64_t nextBytes = uint64_t(used + need) * sizeof(Count),
               oneBytes = uint64_t(need) * sizeof(Count);
      if (!b.cfg.empty() && nextBytes > target)
        break;
      if (oneBytes > target)
        throw std::runtime_error("groupbatch single group exceeds target");
      b.work += pg.work;
      uint32_t fmask = fixLow ? (pg.mo & ((1u << LOW_LUT_K) - 1u))
                              : ((pg.mo >> (LOW_LUT_K + 1)) & ((1u << HIGH_LUT_K) - 1u));
      auto fmb = make_factor_main_blocks(fixLow, fmask),
           fdb = make_factor_block_blocks(fixLow, fmask);
      if (fmb.back().end != pg.ms.size || fdb.back().end != pg.ds.size)
        throw std::runtime_error("groupbatch factor size mismatch");
      BatchFactorIoCfg io{};
      std::memcpy(io.mainb, fmb.data(), fmb.size() * sizeof(FBlock));
      std::memcpy(io.blockb, fdb.data(), fdb.size() * sizeof(FBlock));
      io.main_n = pg.ms.size;
      io.block_n = pg.ds.size;
      io.main_off = used;
      io.block_off = used + al(pg.ms.size);
      io.mask = fmask;
      io.main_nb = fmb.size();
      io.block_nb = fdb.size();
      io.fix_low = fixLow ? 1 : 0;
      b.cfg.push_back(io);
      used += need;
    }
    if (b.cfg.empty()) {
      q0 = q + 1;
      continue;
    }
    // Independent, same-shape groups become adjacent columns of one tile.
    used = 0;
    int maxlg = groupbatch_lane_log();
    for (size_t gi = 0; gi < b.cfg.size();) {
      size_t last = gi + 1;
      while (last < b.cfg.size() && same_batch_geometry(b.cfg[gi], b.cfg[last]))
        ++last;
      int lg = maxlg;
      while ((1u << lg) > last - gi)
        --lg;
      uint32_t lanes = 1u << lg;
      Code mo = used, bo = mo + al(b.cfg[gi].main_n * lanes);
      used = bo + al(b.cfg[gi].block_n * lanes);
      for (uint32_t lane = 0; lane < lanes; ++lane) {
        auto &io = b.cfg[gi + lane];
        io.main_off = mo + lane;
        io.block_off = bo + lane;
        io.loglanes = lg;
      }
      gi += lanes;
    }
    if (used * sizeof(Count) > target)
      throw std::runtime_error("groupbatch tile exceeds scratch target");
    b.used = used;
    b.fix_low = fixLow;
    for (uint32_t gi = 0; gi < b.cfg.size(); gi += (1u << b.cfg[gi].loglanes)) {
      auto const &io = b.cfg[gi];
      Code chunk = GROUPBATCH_CHUNK >> io.loglanes;
      for (uint32_t block = 0; block < 2; ++block)
        for (uint32_t bid = 0; bid < uint32_t(block ? io.block_nb : io.main_nb); ++bid) {
          auto x = block ? io.blockb[bid] : io.mainb[bid];
          for (Code z = 0; z < x.end - x.off; z += chunk)
            b.io.push_back({gi, uint16_t(block), uint16_t(bid), z});
        }
      for (uint32_t bid = 0; bid < io.block_nb; ++bid) {
        auto bx = io.blockb[bid];
        Code len = bx.end - bx.off;
        for (Code z = 0; z < len; z += chunk)
          b.owner.push_back({gi, bid, z});
      }
    }
    for (int pp = pw.wp.p_hi; pp >= pw.wp.p_lo; --pp) {
      b.cross_off[pp] = b.cross.size();
      b.inv_off[pp] = b.inv.size();
      for (uint32_t gi = 0; gi < b.cfg.size(); gi += (1u << b.cfg[gi].loglanes)) {
        auto const &io = b.cfg[gi];
        Code chunk = GROUPBATCH_CHUNK >> io.loglanes;
        for (uint32_t bid = 0; bid < io.main_nb; ++bid) {
          auto x = io.mainb[bid];
          if (!x.stride || x.end <= x.off)
            continue;
          if (fixLow) {
            auto sel = G_HPR.crossll_sel[high_pr_key(pp, x.he, x.c)];
            Code total = Code(x.stride) * sel.count;
            for (Code z = 0; z < total; z += chunk)
              b.cross.push_back(
                  {gi, bid, sel.off, sel.count, z, oneesan::division_reciprocal(x.stride)});
          } else {
            auto key = sparse_key(pp, x.hs, x.c);
            auto sel = G_PRERANK.highrr_sel[key];
            Code rows = (x.end - x.off) / x.stride, total = rows * Code(sel.count);
            for (Code z = 0; z < total; z += chunk)
              b.cross.push_back(
                  {gi, bid, sel.off, sel.count, z, oneesan::division_reciprocal(sel.count)});
            if (pp == 1 && !groupbatch_p1_dest_mode()) {
              auto iv = G_PRERANK.closure_inv_sel[key];
              Code itotal = rows * Code(iv.count);
              for (Code z = 0; z < itotal; z += chunk)
                b.inv.push_back(
                    {gi, bid, iv.off, iv.count, z, oneesan::division_reciprocal(iv.count)});
            }
          }
        }
      }
      b.cross_count[pp] = b.cross.size() - b.cross_off[pp];
      b.inv_count[pp] = b.inv.size() - b.inv_off[pp];
    }
    if (!fixLow && pw.wp.p_lo <= 1 && pw.wp.p_hi >= 1 && groupbatch_p1_dest_mode()) {
      for (uint32_t gi = 0; gi < b.cfg.size(); gi += (1u << b.cfg[gi].loglanes)) {
        auto const &io = b.cfg[gi];
        Code chunk = GROUPBATCH_CHUNK >> io.loglanes;
        for (int he = 0; he <= HIGH_LUT_K + 1; ++he) {
          uint32_t ro = G_P1_DEST.off[he], rc = G_P1_DEST.off[he + 1] - ro;
          if (!rc)
            continue;
          uint32_t rows = factor_count(G_FACTOR.high_mask_off, io.mask, he);
          Code total = Code(rows) * rc;
          for (Code z = 0; z < total; z += chunk)
            b.p1.push_back({gi, ro, rc, uint32_t(he), z, oneesan::division_reciprocal(rc)});
        }
      }
    }
    pw.batches.push_back(std::move(b));
    q0 = q;
  }
}
