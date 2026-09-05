#pragma push_macro("main")
#undef main
#define main two_cell_inverse_gather_probe_main_unused
#include "two_cell_inverse_gather_probe.cpp"
#pragma pop_macro("main")

#include <numeric>

namespace {

std::vector<Key> factor_q_basis(
    int W, int fixed, const std::vector<std::vector<Word>>& words
) {
    std::vector<Key> out;
    for (const Word& w : words[W - 1]) out.push_back(Key{'A', w});
    for (const Word& w : words[W - 2])
        if (w[fixed] != N) out.push_back(Key{'C', w});
    return out;
}

struct FactorTables {
    int W;
    std::vector<std::vector<Rank>> choose;
    std::vector<std::vector<Rank>> primitive;
    std::vector<Rank> sector_offset;
    std::vector<Rank> sector_a;
    std::vector<Rank> sector_c;
    std::vector<Rank> sector_primitive;

    explicit FactorTables(int W_)
        : W(W_),
          choose(static_cast<std::size_t>(W_ + 1),
                 std::vector<Rank>(static_cast<std::size_t>(W_ + 1), 0)),
          primitive(static_cast<std::size_t>(W_ + 1),
                    std::vector<Rank>(static_cast<std::size_t>(W_ + 2), 0)) {
        for (int n = 0; n <= W; ++n) {
            choose[n][0] = choose[n][n] = 1;
            for (int k = 1; k < n; ++k)
                choose[n][k] = choose[n - 1][k - 1] + choose[n - 1][k];
        }

        primitive[0][0] = 1;
        for (int rem = 1; rem <= W; ++rem) {
            for (int h = 0; h <= W; ++h) {
                Rank z = primitive[rem - 1][h + 1];
                if (h > 0) z += primitive[rem - 1][h - 1];
                primitive[rem][h] = z;
            }
        }

        sector_offset.push_back(0);
        for (int p = 0; 2 * p + 1 <= W - 1; ++p) {
            const int k = 2 * p + 1;
            const Rank pc = primitive[k][1];
            const Rank ac = choose[W - 1][k];
            const Rank cc = (k - 1 <= W - 3) ? choose[W - 3][k - 1] : 0;
            sector_a.push_back(ac * pc);
            sector_c.push_back(cc * pc);
            sector_primitive.push_back(pc);
            sector_offset.push_back(sector_offset.back() + (ac + cc) * pc);
        }
    }

    Rank size() const { return sector_offset.back(); }

    std::size_t logical_bytes() const {
        return std::size_t(W + 1) * std::size_t(W + 1) * sizeof(Rank) +
               std::size_t(W + 1) * std::size_t(W + 2) * sizeof(Rank) +
               (sector_offset.size() + sector_a.size() + sector_c.size() +
                sector_primitive.size()) * sizeof(Rank);
    }

    Rank binom(int n, int k) const {
        if (k < 0 || k > n) return 0;
        return choose[n][k];
    }

    int sector_of(Rank rank) const {
        auto it = std::upper_bound(sector_offset.begin(), sector_offset.end(), rank);
        assert(it != sector_offset.begin() && it != sector_offset.end());
        return static_cast<int>(it - sector_offset.begin() - 1);
    }

    Rank support_rank(const std::vector<uint8_t>& bits, int ones) const {
        Rank rank = 0;
        int left = ones;
        for (int pos = 0; pos < static_cast<int>(bits.size()); ++pos) {
            if (!bits[pos]) continue;
            const int rem = static_cast<int>(bits.size()) - pos - 1;
            rank += binom(rem, left);
            --left;
        }
        assert(left == 0);
        return rank;
    }

    std::vector<uint8_t> support_unrank(int len, int ones, Rank rank) const {
        assert(rank < binom(len, ones));
        std::vector<uint8_t> bits(static_cast<std::size_t>(len), 0);
        int left = ones;
        for (int pos = 0; pos < len; ++pos) {
            const int rem = len - pos - 1;
            const Rank zero_count = binom(rem, left);
            if (rank < zero_count) continue;
            rank -= zero_count;
            bits[pos] = 1;
            --left;
        }
        assert(left == 0 && rank == 0);
        return bits;
    }

    Rank primitive_rank(const Word& word) const {
        int occupied = 0;
        for (char c : word) occupied += c != N;
        int h = 1;
        int seen = 0;
        Rank rank = 0;
        for (char c : word) {
            if (c == N) continue;
            const int rem = occupied - (++seen);
            if (c == L) {
                if (h > 0) rank += primitive[rem][h - 1];
                ++h;
            } else {
                assert(c == R && h > 0);
                --h;
            }
        }
        assert(h == 0);
        return rank;
    }

    Word primitive_unrank(int occupied, Rank rank) const {
        assert(rank < primitive[occupied][1]);
        Word out;
        out.reserve(static_cast<std::size_t>(occupied));
        int h = 1;
        for (int pos = 0; pos < occupied; ++pos) {
            const int rem = occupied - pos - 1;
            const Rank r_count = h > 0 ? primitive[rem][h - 1] : 0;
            if (rank < r_count) {
                out.push_back(R);
                --h;
            } else {
                rank -= r_count;
                out.push_back(L);
                ++h;
            }
        }
        assert(h == 0 && rank == 0);
        return out;
    }
};

struct FactorizedCodec {
    const FactorTables& t;
    int fixed;

    FactorizedCodec(const FactorTables& t_, int fixed_) : t(t_), fixed(fixed_) {}

    Rank size() const { return t.size(); }

    Rank rank(const Key& key) const {
        const int len = static_cast<int>(key.w.size());
        int occupied = 0;
        std::vector<uint8_t> bits;
        bits.reserve(static_cast<std::size_t>(len));
        for (char c : key.w) {
            const uint8_t bit = c != N;
            bits.push_back(bit);
            occupied += bit;
        }
        assert((occupied & 1) && occupied > 0);
        const int p = (occupied - 1) / 2;
        const Rank pc = t.sector_primitive[p];
        const Rank pr = t.primitive_rank(key.w);
        Rank local = t.sector_offset[p];

        if (key.type == 'A') {
            assert(len == t.W - 1);
            const Rank sr = t.support_rank(bits, occupied);
            return local + sr * pc + pr;
        }

        assert(key.type == 'C' && len == t.W - 2);
        assert(fixed >= 0 && fixed < len && bits[fixed]);
        std::vector<uint8_t> compact;
        compact.reserve(static_cast<std::size_t>(len - 1));
        for (int pos = 0; pos < len; ++pos)
            if (pos != fixed) compact.push_back(bits[pos]);
        const Rank sr = t.support_rank(compact, occupied - 1);
        return local + t.sector_a[p] + sr * pc + pr;
    }

    Key unrank(Rank rank) const {
        assert(rank < size());
        const int p = t.sector_of(rank);
        const int occupied = 2 * p + 1;
        const Rank pc = t.sector_primitive[p];
        Rank local = rank - t.sector_offset[p];
        std::vector<uint8_t> bits;
        Rank primitive_rank = 0;
        char type = 'A';

        if (local < t.sector_a[p]) {
            const Rank support_rank = local / pc;
            primitive_rank = local % pc;
            bits = t.support_unrank(t.W - 1, occupied, support_rank);
        } else {
            type = 'C';
            local -= t.sector_a[p];
            const Rank support_rank = local / pc;
            primitive_rank = local % pc;
            const auto compact = t.support_unrank(t.W - 3, occupied - 1, support_rank);
            bits.reserve(static_cast<std::size_t>(t.W - 2));
            int q = 0;
            for (int pos = 0; pos < t.W - 2; ++pos) {
                if (pos == fixed) bits.push_back(1);
                else bits.push_back(compact[q++]);
            }
        }

        const Word primitive = t.primitive_unrank(occupied, primitive_rank);
        Word w(bits.size(), N);
        int q = 0;
        for (int pos = 0; pos < static_cast<int>(bits.size()); ++pos)
            if (bits[pos]) w[pos] = primitive[q++];
        assert(q == occupied && valid_word(w));
        return Key{type, w};
    }
};

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 12;
    if (maxW < 4 || maxW > 15) return 2;

    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 4; W <= maxW; ++W) {
        FactorTables tables(W);
        Rank max_indeg = 0;
        int largest_p = 0;
        Rank largest_sector = 0;
        for (int p = 0; p < static_cast<int>(tables.sector_a.size()); ++p) {
            const Rank n = tables.sector_a[p] + tables.sector_c[p];
            if (n > largest_sector) {
                largest_sector = n;
                largest_p = p;
            }
        }

        for (int i = 0; i <= W - 4; ++i) {
            FactorizedCodec src(tables, i), dst(tables, i + 1);
            const auto src_keys = factor_q_basis(W, i, words);
            const auto dst_keys = factor_q_basis(W, i + 1, words);
            if (src.size() != src_keys.size() || dst.size() != dst_keys.size())
                fail("factorized dimension");

            std::vector<uint8_t> seen_src(static_cast<std::size_t>(src.size()));
            std::vector<uint8_t> seen_dst(static_cast<std::size_t>(dst.size()));
            for (const Key& k : src_keys) {
                const Rank r = src.rank(k);
                if (r >= src.size() || seen_src[r] || !(src.unrank(r) == k))
                    fail("factorized source codec");
                seen_src[r] = 1;
            }
            for (const Key& k : dst_keys) {
                const Rank r = dst.rank(k);
                if (r >= dst.size() || seen_dst[r] || !(dst.unrank(r) == k))
                    fail("factorized destination codec");
                seen_dst[r] = 1;
            }

            std::vector<std::vector<Rank>> incoming(static_cast<std::size_t>(dst.size()));
            for (const Key& k : src_keys) {
                const Rank s = src.rank(k);
                for (const auto& [d, c] : K_basis(k, W, i)) {
                    if (c != 1) fail("nonunit");
                    incoming[dst.rank(d)].push_back(s);
                }
            }

            std::vector<std::uint64_t> values(static_cast<std::size_t>(src.size()));
            std::vector<std::uint64_t> scatter(static_cast<std::size_t>(dst.size()));
            std::vector<std::uint64_t> gather(static_cast<std::size_t>(dst.size()));
            for (Rank s = 0; s < src.size(); ++s)
                values[s] = 1 + ((s * 0x9e3779b97f4a7c15ULL) ^
                                 (Rank(W) << 32) ^ Rank(i));

            for (Rank d = 0; d < dst.size(); ++d) {
                std::vector<Rank> got;
                for (const Key& s : inverse_K(dst.unrank(d), W, i))
                    got.push_back(src.rank(s));
                std::sort(got.begin(), got.end());
                auto expected = incoming[d];
                std::sort(expected.begin(), expected.end());
                if (got != expected)
                    fail("factorized inverse W=" + std::to_string(W) +
                         " i=" + std::to_string(i) + " d=" + std::to_string(d));
                max_indeg = std::max<Rank>(max_indeg, got.size());
                for (Rank s : expected) scatter[d] += values[s];
                for (Rank s : got) gather[d] += values[s];
            }
            if (scatter != gather) fail("factorized gather");
        }

        std::cout << "W=" << W
                  << " reduced=" << tables.size()
                  << " max_indeg=" << max_indeg
                  << " largest_sector_p=" << largest_p
                  << " largest_sector=" << largest_sector
                  << " factor_table_kib=" << double(tables.logical_bytes()) / 1024.0
                  << " OK\n";
    }

    FactorTables w28(28);
    Rank largest = 0;
    int largest_p = -1;
    for (int p = 0; p < static_cast<int>(w28.sector_a.size()); ++p) {
        const Rank n = w28.sector_a[p] + w28.sector_c[p];
        if (n > largest) {
            largest = n;
            largest_p = p;
        }
    }
    std::cout << "W=28_theory reduced=" << w28.size()
              << " largest_sector_p=" << largest_p
              << " largest_sector=" << largest
              << " factor_table_kib=" << double(w28.logical_bytes()) / 1024.0
              << "\n";
}
