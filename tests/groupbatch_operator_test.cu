// Independent forward recurrence on arbitrary main and blocked coefficients.
#define main oneesan_solver_main
#ifndef GROUPBATCH_TEST_SOURCE
// clang-format off
#define GROUPBATCH_TEST_SOURCE "../src/cuda/b300/oneesan_cuda_gridfp_b300_hbm32_factorized_fullprerank_orbit_blockfused_groupbatch_batch.cu"
// clang-format on
#endif
#include GROUPBATCH_TEST_SOURCE
#undef main
#include <random>
static void legal_words(int pos, int height, MateID word, std::vector<MateID> &words) {
  if (pos < 0) {
    if (!height)
      words.push_back(word);
    return;
  }
  legal_words(pos - 1, height, word, words);
  if (height)
    legal_words(pos - 1, height - 1, word | (MateID(R) << (2 * pos)), words);
  if (height < pos + 1)
    legal_words(pos - 1, height + 1, word | (MateID(L) << (2 * pos)), words);
}

int main() {
  static_assert(TARGET_W == 10 || TARGET_W == 14);
  build_full_dp();
  G_FACTOR = build_factor_tables();
  G_PRERANK = build_prerank_orbit_tables();
  G_HPR = build_high_prerank_tables();
  G_P1_DEST = build_p1_dest_tables();
  build_cross_match_words();
  std::vector<uint32_t> low_dense(size_t(1) << (2 * LOW_LUT_K), 0xffffffffu);
  std::vector<uint32_t> high_dense(size_t(1) << (2 * HIGH_LUT_K), 0xffffffffu);
  for (size_t i = 0; i < G_FACTOR.low_all_codes.size(); ++i)
    low_dense[G_FACTOR.low_all_codes[i]] = G_FACTOR.low_packed_values[i];
  for (size_t i = 0; i < G_FACTOR.high_all_codes.size(); ++i)
    high_dense[G_FACTOR.high_all_codes[i]] = G_FACTOR.high_packed_values[i];
  std::vector<void *> allocations;
#define UPLOAD(symbol, vec)                                                                        \
  do {                                                                                             \
    using T = typename std::decay_t<decltype(vec)>::value_type;                                    \
    T *ptr = nullptr;                                                                              \
    if (!(vec).empty()) {                                                                          \
      ck(cudaMalloc(&ptr, (vec).size() * sizeof(T)), "table alloc");                               \
      allocations.push_back(ptr);                                                                  \
      ck(cudaMemcpy(ptr, (vec).data(), (vec).size() * sizeof(T), cudaMemcpyHostToDevice),          \
         "table copy");                                                                            \
    }                                                                                              \
    ck(cudaMemcpyToSymbol(symbol, &ptr, sizeof(ptr)), "table symbol");                             \
  } while (0)
  UPLOAD(D_F_LOW_ALL_CODES, G_FACTOR.low_all_codes);
  UPLOAD(D_F_LOW_MASK_CODES, G_FACTOR.low_mask_codes);
  UPLOAD(D_F_LOW_MASK_OFF, G_FACTOR.low_mask_off);
  UPLOAD(D_F_LOW_MASK_ALL_RANK, G_FACTOR.low_mask_all_rank);
  UPLOAD(D_F_LOW_DENSE_PACKED_RANK, low_dense);
  UPLOAD(D_F_HIGH_ALL_CODES, G_FACTOR.high_all_codes);
  UPLOAD(D_F_HIGH_MASK_CODES, G_FACTOR.high_mask_codes);
  UPLOAD(D_F_HIGH_MASK_OFF, G_FACTOR.high_mask_off);
  UPLOAD(D_F_HIGH_MASK_ALL_RANK, G_FACTOR.high_mask_all_rank);
  UPLOAD(D_F_HIGH_PACKED_RANK, high_dense);
  UPLOAD(D_F_HIGH_MAIN_BASE, G_FACTOR.high_main_base);
  UPLOAD(D_F_HIGH_BLOCK_BASE, G_FACTOR.high_block_base);
  UPLOAD(D_PR_OWNER_BLOCK_REC, G_PRERANK.owner_block_rec);
  UPLOAD(D_PR_OWNER_BLOCK_CLOSURE_META, G_PRERANK.owner_block_closure_meta);
  UPLOAD(D_PR_OWNER_BLOCK_CLOSURE_SRC, G_PRERANK.owner_block_closure_src);
  UPLOAD(D_PR_CLOSURE_INV_REC, G_PRERANK.closure_inv_rec);
  UPLOAD(D_PR_CLOSURE_INV_SRC, G_PRERANK.closure_inv_src);
  UPLOAD(D_PR_HIGHRR_REC, G_PRERANK.highrr_rec);
  UPLOAD(D_HP_OWNER_BLOCK_REC, G_HPR.owner_block_rec);
  UPLOAD(D_HP_OWNER_BLOCK_CLOSURE_META, G_HPR.owner_block_closure_meta);
  UPLOAD(D_HP_OWNER_BLOCK_CLOSURE_SRC, G_HPR.owner_block_closure_src);
  UPLOAD(D_HP_CLOSURE_INV_REC, G_HPR.closure_inv_rec);
  UPLOAD(D_HP_CLOSURE_INV_SRC, G_HPR.closure_inv_src);
  UPLOAD(D_HP_CROSSLL_REC, G_HPR.crossll_rec);
  UPLOAD(D_P1_DEST_REC, G_P1_DEST.rec);
  UPLOAD(D_P1_DEST_SRC, G_P1_DEST.src);
  UPLOAD(D_CROSS_LOW_MATCH, G_CROSS_LOW_MATCH);
  UPLOAD(D_CROSS_HIGH_MATCH, G_CROSS_HIGH_MATCH);
#undef UPLOAD
  ck(cudaMemcpyToSymbol(D_F_LOW_ALL_OFF, G_FACTOR.low_all_off.data(),
                        sizeof(uint32_t) * (MAXW + 2)),
     "low off");
  ck(cudaMemcpyToSymbol(D_F_HIGH_ALL_OFF, G_FACTOR.high_all_off.data(),
                        sizeof(uint32_t) * (MAXW + 2)),
     "high off");
  std::vector<MateID> main_words, block_words;
  legal_words(TARGET_W - 1, 1, 0, main_words);
  legal_words(TARGET_W - 2, 1, 0, block_words);
  Code n = H_DP[TARGET_W][1], dn = H_DP[TARGET_W - 1][1];
  if (main_words.size() != n || block_words.size() != dn)
    throw std::runtime_error("CPU states");
  Count *mp[MAXGPU]{}, *bp[MAXGPU]{};
  ck(cudaMalloc(&mp[0], n * sizeof(Count)), "main alloc");
  ck(cudaMalloc(&bp[0], dn * sizeof(Count)), "block alloc");
  DeviceCtx c;
  c.init(0, 4294967291u, mp, bp, n, dn, 1);
  c.capArena = 2ull << 20;
  ck(cudaMalloc(&c.arena, c.capArena), "arena");
  std::mt19937_64 rng(0x625f1309);
  uint64_t cases = 0, coefficients = 0, lane_coverage = 0;
  for (Count mod : {2u, 4294967291u, 4294966997u}) {
    ck(cudaMemcpyToSymbol(D_MOD, &mod, sizeof(mod)), "test modulus");
    G_GROUPBATCH_WRAP32_ACTIVE = groupbatch_wrap32_mode() && groupbatch_wrap32_mod_safe(mod);
    for (bool fixed_low : {false, true}) {
      const int hi = fixed_low ? TARGET_W - 1 : LOW_LUT_K, lo = fixed_low ? LOW_LUT_K + 1 : 1;
      // Whole windows and each individual cell, including p=1 and the split.
      for (int cell = lo - 1; cell <= hi; ++cell) {
        if (TARGET_W == 14 && cell != lo - 1)
          continue;
        PreparedWindow pw;
        pw.wp.p_hi = cell == lo - 1 ? hi : cell;
        pw.wp.p_lo = cell == lo - 1 ? lo : cell;
        // Keep the full-window partition, even for a single-cell oracle.
        pw.wp.fixed_pos = window_candidates(TARGET_W, hi, lo);
        auto partition = pw.wp;
        partition.p_hi = hi;
        partition.p_lo = lo;
        for (int g = 0; g < (1 << pw.wp.fixed_pos.size()); ++g)
          pw.groups.push_back(prepare_group(TARGET_W, partition, g, n, dn, 1));
        std::sort(pw.groups.begin(), pw.groups.end(), groupbatch_work_order);
        std::vector<Count> input(n), dinput(dn);
        for (auto &v : input)
          v = Count(rng() % mod);
        for (auto &v : dinput)
          v = Count(rng() % mod);
        auto expected = input, dexpected = dinput;
        for (int p = pw.wp.p_hi; p >= pw.wp.p_lo; --p) {
          auto next = expected;
          std::vector<Count> dnext(dn, 0);
          auto add = [&](Count &a, Count b) { a = Count((uint64_t(a) + b) % mod); };
          for (auto word : main_words) {
            auto edge = oneesan::gridfp::include_horizontal(word, TARGET_W, p);
            if (!edge.valid)
              continue;
            Count v = expected[rank_full(word, TARGET_W)];
            if (edge.blocked)
              add(dnext[rank_full(edge.mate, TARGET_W - 1)], v);
            else
              add(next[rank_full(edge.mate, TARGET_W)], v);
          }
          for (auto word : block_words)
            add(next[rank_full(oneesan::gridfp::blocked_exclude(word, p), TARGET_W)],
                dexpected[rank_full(word, TARGET_W - 1)]);
          expected.swap(next);
          dexpected.swap(dnext);
        }
        for (int lg = 0; lg <= 5; ++lg)
          for (size_t target : {(TARGET_W == 14 ? 128ull : 8ull) << 10, 2ull << 20}) {
            setenv("GRIDFP_GROUPBATCH_LANES", std::to_string(1u << lg).c_str(), 1);
            prepare_window_batches(TARGET_W, pw, target);
            for (auto const &batch : pw.batches) {
              if (batch.used * sizeof(Count) > target)
                throw std::runtime_error("scratch bound");
              for (auto const &cfg : batch.cfg)
                lane_coverage |= 1ull << cfg.loglanes;
            }
            for (int replay = 0; replay < 2; ++replay) {
              ck(cudaMemcpy(mp[0], input.data(), n * sizeof(Count), cudaMemcpyHostToDevice),
                 "initial main");
              ck(cudaMemcpy(bp[0], dinput.data(), dn * sizeof(Count), cudaMemcpyHostToDevice),
                 "initial block");
              // Pageable H2D cudaMemcpy may return before its device transfer finishes.
              // The solver uses a nonblocking stream, so complete the fixture upload
              // explicitly; cached graphs no longer trigger metadata-allocation syncs.
              ck(cudaDeviceSynchronize(), "input upload complete");
              process_window_batched_io(c, TARGET_W, pw, 64, target);
              std::vector<Count> got(n), dgot(dn);
              ck(cudaMemcpy(got.data(), mp[0], n * sizeof(Count), cudaMemcpyDeviceToHost),
                 "result main");
              ck(cudaMemcpy(dgot.data(), bp[0], dn * sizeof(Count), cudaMemcpyDeviceToHost),
                 "result block");
              if (got != expected || dgot != dexpected) {
                std::cerr << "FAIL mod=" << mod << " fixed_low=" << fixed_low << " cell=" << cell
                          << " lg=" << lg << " target=" << target << " replay=" << replay << '\n';
                for(Code i=0;i<n;++i)if(got[i]!=expected[i]){std::cerr<<"main index="<<i<<" got="<<got[i]<<" expected="<<expected[i]<<'\n';break;}
                for(Code i=0;i<dn;++i)if(dgot[i]!=dexpected[i]){std::cerr<<"block index="<<i<<" got="<<dgot[i]<<" expected="<<dexpected[i]<<'\n';break;}
                return 1;
              }
              ++cases;
              coefficients += n + dn;
            }
            destroy_groupbatch_device_meta(pw, 1);
          }
      }
    }
  }
  if (TARGET_W == 14 && lane_coverage != 63)
    throw std::runtime_error("missing lane coverage");
  c.destroy();
  cudaFree(mp[0]);
  cudaFree(bp[0]);
  for (auto ptr : allocations)
    cudaFree(ptr);
  std::cout << "PASS " << cases << " full-vector CPU/GPU cases, " << coefficients
            << " coefficients, lane_coverage=" << lane_coverage << '\n';
}
