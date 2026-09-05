#pragma push_macro("main")
#undef main
#define main two_cell_recoupling_support_probe_main_unused
#include "two_cell_recoupling_support_probe.cpp"
#pragma pop_macro("main")

namespace {

struct PrimitiveCodec {
    int max_len = 0;
    std::vector<std::vector<Rank>> dp;

    explicit PrimitiveCodec(int n)
        : max_len(n),
          dp(static_cast<std::size_t>(n + 1),
             std::vector<Rank>(static_cast<std::size_t>(n + 2), 0)) {
        dp[0][0] = 1;
        for (int rem = 1; rem <= n; ++rem) {
            for (int h = 0; h <= n; ++h) {
                Rank z = dp[rem - 1][h + 1];
                if (h > 0) z += dp[rem - 1][h - 1];
                dp[rem][h] = z;
            }
        }
    }

    Rank count(int occupied) const {
        if (occupied < 0 || occupied > max_len) return 0;
        return dp[occupied][1];
    }

    Rank rank(const Word& w) const {
        int occupied = 0;
        for (char c : w) occupied += c != N;
        Rank r = 0;
        int h = 1;
        int seen = 0;
        for (char c : w) {
            if (c == N) continue;
            const int rem = occupied - (++seen);
            if (c == L) {
                if (h > 0) r += dp[rem][h - 1];
                ++h;
            } else {
                if (c != R || h <= 0) fail("recoupling primitive rank invalid word");
                --h;
            }
        }
        if (h != 0 || r >= count(occupied)) fail("recoupling primitive rank range");
        return r;
    }

    Word unrank_primitive(int occupied, Rank r) const {
        if (r >= count(occupied)) fail("recoupling primitive unrank range");
        Word out;
        out.reserve(static_cast<std::size_t>(occupied));
        int h = 1;
        for (int pos = 0; pos < occupied; ++pos) {
            const int rem = occupied - pos - 1;
            const Rank rc = h > 0 ? dp[rem][h - 1] : 0;
            if (r < rc) {
                out.push_back(R);
                --h;
            } else {
                r -= rc;
                out.push_back(L);
                ++h;
            }
        }
        if (h != 0 || r != 0) fail("recoupling primitive unrank terminal");
        return out;
    }
};

Rank suffix_weight(int remaining_bits, int ones_prefix) {
    Rank z = 0;
    for (int r = 0; r <= remaining_bits; ++r)
        z += binom_small(remaining_bits, r) * state_block_size(ones_prefix + r);
    return z;
}

Rank outer_block_offset(std::uint64_t mask, int q) {
    Rank offset = 0;
    int ones = 0;
    for (int bit = q - 1; bit >= 0; --bit) {
        if ((mask >> bit) & 1ULL) {
            offset += suffix_weight(bit, ones);
            ++ones;
        }
    }
    return offset;
}

std::pair<std::uint64_t, Rank> outer_block_unrank(Rank rank, int q) {
    std::uint64_t mask = 0;
    int ones = 0;
    for (int bit = q - 1; bit >= 0; --bit) {
        const Rank zero = suffix_weight(bit, ones);
        if (rank < zero) continue;
        rank -= zero;
        mask |= std::uint64_t(1) << bit;
        ++ones;
    }
    const Rank block = state_block_size(ones);
    if (rank >= block) fail("recoupling outer block residual");
    return {mask, rank};
}

int support_popcount(const Word& w) {
    int z = 0;
    for (char c : w) z += c != N;
    return z;
}

struct SlidingRankCodec {
    int W = 0;
    int window = 0; // recoupling window i,i+1,i+2 between K_i and K_{i+1}
    PrimitiveCodec primitive;

    SlidingRankCodec(int W_, int window_)
        : W(W_), window(window_), primitive(W_) {
        if (window < 0 || window + 2 >= W - 2)
            fail("recoupling codec window");
    }

    int outer_bits() const { return W - 5; }

    std::uint64_t outer_mask(const Key& key) const {
        std::uint64_t mask = 0;
        int q = 0;
        if (key.type == 'A') {
            if (static_cast<int>(key.w.size()) != W - 1) fail("recoupling A length");
            for (int p = 0; p < W - 1; ++p) {
                if (p >= window && p <= window + 3) continue;
                if (key.w[p] != N) mask |= std::uint64_t(1) << q;
                ++q;
            }
        } else {
            if (key.type != 'C' || static_cast<int>(key.w.size()) != W - 2)
                fail("recoupling C length");
            if (key.w[window + 1] == N) fail("recoupling C fixed bit");
            for (int p = 0; p < W - 2; ++p) {
                if (p >= window && p <= window + 2) continue;
                if (key.w[p] != N) mask |= std::uint64_t(1) << q;
                ++q;
            }
        }
        if (q != outer_bits()) fail("recoupling outer bit count");
        return mask;
    }

    Rank rank(const Key& key) const {
        const std::uint64_t mask = outer_mask(key);
        const int k = popcount64(mask);
        Rank local = outer_block_offset(mask, outer_bits());

        if (key.type == 'A') {
            int code = 0;
            for (int t = 0; t < 4; ++t)
                if (key.w[window + t] != N) code |= 1 << t;
            for (int c = 0; c < code; ++c)
                local += primitive_count(k + __builtin_popcount(static_cast<unsigned>(c)));
            const int occupied = k + __builtin_popcount(static_cast<unsigned>(code));
            local += primitive.rank(key.w);
            if (primitive.count(occupied) != primitive_count(occupied))
                fail("recoupling primitive count A");
            return local;
        }

        local += state_block_size(k) -
                 [&]() {
                     Rank c = 0;
                     for (int r = 0; r <= 2; ++r)
                         c += binom_small(2, r) * primitive_count(k + 1 + r);
                     return c;
                 }();
        int code = 0;
        if (key.w[window] != N) code |= 1;
        if (key.w[window + 2] != N) code |= 2;
        for (int c = 0; c < code; ++c)
            local += primitive_count(k + 1 + __builtin_popcount(static_cast<unsigned>(c)));
        const int occupied = k + 1 + __builtin_popcount(static_cast<unsigned>(code));
        local += primitive.rank(key.w);
        if (primitive.count(occupied) != primitive_count(occupied))
            fail("recoupling primitive count C");
        return local;
    }

    Key unrank(Rank rank_value) const {
        const auto [mask, residual0] = outer_block_unrank(rank_value, outer_bits());
        Rank residual = residual0;
        const int k = popcount64(mask);

        Rank a_count = 0;
        for (int l = 0; l <= 4; ++l)
            a_count += binom_small(4, l) * primitive_count(k + l);

        char type = 'A';
        int code = -1;
        int local_width = 4;
        int occupied = 0;
        if (residual < a_count) {
            for (int c = 0; c < 16; ++c) {
                const int occ = k + __builtin_popcount(static_cast<unsigned>(c));
                const Rank n = primitive_count(occ);
                if (residual < n) {
                    code = c;
                    occupied = occ;
                    break;
                }
                residual -= n;
            }
        } else {
            type = 'C';
            residual -= a_count;
            local_width = 2;
            for (int c = 0; c < 4; ++c) {
                const int occ = k + 1 + __builtin_popcount(static_cast<unsigned>(c));
                const Rank n = primitive_count(occ);
                if (residual < n) {
                    code = c;
                    occupied = occ;
                    break;
                }
                residual -= n;
            }
        }
        if (code < 0 || residual >= primitive_count(occupied))
            fail("recoupling local block unrank");

        const Word prim = primitive.unrank_primitive(occupied, residual);
        Word w(static_cast<std::size_t>(type == 'A' ? W - 1 : W - 2), N);
        int outer_q = 0;
        if (type == 'A') {
            for (int t = 0; t < 4; ++t)
                if ((code >> t) & 1) w[window + t] = '?';
            for (int p = 0; p < W - 1; ++p) {
                if (p >= window && p <= window + 3) continue;
                if ((mask >> outer_q) & 1ULL) w[p] = '?';
                ++outer_q;
            }
        } else {
            if (code & 1) w[window] = '?';
            w[window + 1] = '?';
            if (code & 2) w[window + 2] = '?';
            for (int p = 0; p < W - 2; ++p) {
                if (p >= window && p <= window + 2) continue;
                if ((mask >> outer_q) & 1ULL) w[p] = '?';
                ++outer_q;
            }
        }
        if (outer_q != outer_bits()) fail("recoupling unrank outer bits");

        int z = 0;
        for (char& c : w)
            if (c == '?') c = prim[z++];
        if (z != occupied || !valid_word(w)) fail("recoupling unrank word");
        return Key{type, w};
    }
};

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    if (maxW < 6 || maxW > 14) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 6; W <= maxW; ++W) {
        const Rank dim = Rank(words[W - 1].size()) + Rank(words[W - 2].size()) -
                         Rank(words[W - 3].size());
        Rank checked_states = 0;
        Rank checked_steps = 0;

        for (int window = 0; window <= W - 5; ++window) {
            SlidingRankCodec codec(W, window);
            const auto q = q_basis(W, window + 1, words);
            if (q.size() != dim) fail("recoupling codec dimension");
            std::vector<std::uint8_t> seen(static_cast<std::size_t>(dim));
            for (const Key& key : q) {
                const Rank r = codec.rank(key);
                if (r >= dim || seen[static_cast<std::size_t>(r)]++)
                    fail("recoupling codec collision");
                if (!(codec.unrank(r) == key))
                    fail("recoupling codec roundtrip W=" + std::to_string(W));
                ++checked_states;
            }
            for (std::uint8_t x : seen)
                if (x != 1) fail("recoupling codec hole");
        }

        // Interior reduced step K_i reads Q_i in the previous recoupling
        // window (i-1) and writes Q_{i+1} directly in window i. No explicit
        // permutation between component-major views is required.
        for (int i = 1; i <= W - 5; ++i) {
            SlidingRankCodec in_codec(W, i - 1);
            SlidingRankCodec out_codec(W, i);
            std::vector<std::uint64_t> got(static_cast<std::size_t>(dim));
            std::vector<std::uint64_t> ref(static_cast<std::size_t>(dim));
            std::vector<std::uint8_t> source_seen(static_cast<std::size_t>(dim));

            for (const Word& u : words[W - 2]) {
                const auto src = packed_direct_component_sources(pack_word(u), W, i);
                for (int q = 0; q < src.size; ++q) {
                    const Key s = unpack_key(src.value[q]);
                    const Rank sr = in_codec.rank(s);
                    if (source_seen[static_cast<std::size_t>(sr)]++)
                        fail("recoupling component source overlap");
                    const std::uint64_t value =
                        1 + ((sr * 0x9e3779b97f4a7c15ULL) ^ (Rank(W) << 32) ^ Rank(i));
                    for (const auto& [d, c] : K_basis(s, W, i)) {
                        if (c != 1) fail("recoupling component nonunit");
                        got[static_cast<std::size_t>(out_codec.rank(d))] += value;
                    }
                }
            }
            for (std::uint8_t x : source_seen)
                if (x != 1) fail("recoupling component source coverage");

            for (const Key& s : q_basis(W, i, words)) {
                const Rank sr = in_codec.rank(s);
                const std::uint64_t value =
                    1 + ((sr * 0x9e3779b97f4a7c15ULL) ^ (Rank(W) << 32) ^ Rank(i));
                for (const auto& [d, c] : K_basis(s, W, i))
                    ref[static_cast<std::size_t>(out_codec.rank(d))] += c * value;
            }
            if (got != ref) fail("recoupling sliding-layout arithmetic");
            ++checked_steps;
        }

        std::cout << "W=" << W
                  << " reduced=" << dim
                  << " checked_state_roundtrips=" << checked_states
                  << " checked_sliding_steps=" << checked_steps
                  << " outer_offset_table_bytes=0"
                  << " permutation_table_bytes=0"
                  << " layout_transition=direct_store"
                  << " primitive_dp=O(W^2)"
                  << " OK\n";
    }

    const int W = 28;
    const int q = W - 5;
    Rank total = 0;
    for (int k = 0; k <= q; ++k)
        total += binom_small(q, k) * state_block_size(k);
    std::cout << "W=28_theory outer_bits=" << q
              << " blocks=" << (Rank(1) << q)
              << " reduced=" << total
              << " rank_tables_bytes=O(W^2)"
              << " global_permutation_bytes=0"
              << "\n";
    std::cout << "ALL_OK sliding_recoupling_rank=1\n";
    return 0;
}
