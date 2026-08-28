#pragma push_macro("main")
#undef main
#define main gridfp_reduced_production_inverse_probe_main_unused
#include "gridfp_reduced_production_inverse_probe.cpp"
#pragma pop_macro("main")

namespace {

struct ProductionFactorTables {
    int W;
    std::vector<std::vector<Rank>> choose;
    std::vector<std::vector<Rank>> primitive;
    std::vector<Rank> sector_offset;
    std::vector<Rank> sector_main;
    std::vector<Rank> sector_block;
    std::vector<Rank> sector_primitive;

    explicit ProductionFactorTables(int W_)
        : W(W_),
          choose(static_cast<std::size_t>(W_ + 1),
                 std::vector<Rank>(static_cast<std::size_t>(W_ + 1))),
          primitive(static_cast<std::size_t>(W_ + 1),
                    std::vector<Rank>(static_cast<std::size_t>(W_ + 2))) {
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
        for (int p = 0; 2 * p + 1 <= W; ++p) {
            const int k = 2 * p + 1;
            const Rank pc = primitive[k][1];
            const Rank mc = binom(W, k);
            const Rank bc = binom(W - 2, k - 1);
            sector_main.push_back(mc * pc);
            sector_block.push_back(bc * pc);
            sector_primitive.push_back(pc);
            sector_offset.push_back(sector_offset.back() + (mc + bc) * pc);
        }
    }

    Rank binom(int n, int k) const {
        if (n < 0 || k < 0 || k > n) return 0;
        return choose[static_cast<std::size_t>(n)][static_cast<std::size_t>(k)];
    }
    Rank size() const { return sector_offset.back(); }

    std::size_t logical_bytes() const {
        std::size_t z = 0;
        for (const auto& r : choose) z += r.size() * sizeof(Rank);
        for (const auto& r : primitive) z += r.size() * sizeof(Rank);
        z += (sector_offset.size() + sector_main.size() + sector_block.size() +
              sector_primitive.size()) * sizeof(Rank);
        return z;
    }

    int sector_of(Rank rank) const {
        const auto it = std::upper_bound(sector_offset.begin(), sector_offset.end(), rank);
        if (it == sector_offset.begin() || it == sector_offset.end()) fail("factor sector rank");
        return static_cast<int>(it - sector_offset.begin() - 1);
    }

    Rank support_rank(const std::vector<std::uint8_t>& bits, int ones) const {
        Rank rank = 0;
        int left = ones;
        for (int pos = 0; pos < static_cast<int>(bits.size()); ++pos) {
            if (!bits[static_cast<std::size_t>(pos)]) continue;
            const int rem = static_cast<int>(bits.size()) - pos - 1;
            rank += binom(rem, left);
            --left;
        }
        if (left != 0) fail("support rank ones");
        return rank;
    }

    std::vector<std::uint8_t> support_unrank(int len, int ones, Rank rank) const {
        if (rank >= binom(len, ones)) fail("support unrank range");
        std::vector<std::uint8_t> bits(static_cast<std::size_t>(len));
        int left = ones;
        for (int pos = 0; pos < len; ++pos) {
            const int rem = len - pos - 1;
            const Rank zero_count = binom(rem, left);
            if (rank < zero_count) continue;
            rank -= zero_count;
            bits[static_cast<std::size_t>(pos)] = 1;
            --left;
        }
        if (left != 0 || rank != 0) fail("support unrank final");
        return bits;
    }

    Rank primitive_rank(MateID m, int len) const {
        int occupied = 0;
        for (int bit = 0; bit < len; ++bit) occupied += mget(m, bit) != N;
        int h = 1, seen = 0;
        Rank rank = 0;
        for (int pos = 0; pos < len; ++pos) {
            const MateValue c = mget(m, len - 1 - pos);
            if (c == N) continue;
            if (c == X) fail("primitive rank X");
            const int rem = occupied - (++seen);
            if (c == L) {
                if (h > 0) rank += primitive[rem][h - 1];
                ++h;
            } else {
                if (h <= 0) fail("primitive rank height");
                --h;
            }
        }
        if (h != 0) fail("primitive rank final");
        return rank;
    }

    std::vector<MateValue> primitive_unrank(int occupied, Rank rank) const {
        if (rank >= primitive[occupied][1]) fail("primitive unrank range");
        std::vector<MateValue> out;
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
        if (h != 0 || rank != 0) fail("primitive unrank final");
        return out;
    }
};

struct ProductionFactorCodec {
    const ProductionFactorTables& t;
    int fixed_bit; // compressed blocked bit that must be occupied

    ProductionFactorCodec(const ProductionFactorTables& t_, int fixed_bit_)
        : t(t_), fixed_bit(fixed_bit_) {}

    Rank size() const { return t.size(); }

    Rank rank(Key k) const {
        const int len = k.blocked ? t.W - 1 : t.W;
        if (k.blocked && mget(k.mate, fixed_bit) == N) fail("factor blocked fixed N");
        int occupied = 0;
        std::vector<std::uint8_t> bits;
        bits.reserve(static_cast<std::size_t>(len));
        for (int pos = 0; pos < len; ++pos) {
            const bool bit = mget(k.mate, len - 1 - pos) != N;
            bits.push_back(static_cast<std::uint8_t>(bit));
            occupied += bit;
        }
        if (!(occupied & 1)) fail("factor occupied parity");
        const int p = (occupied - 1) / 2;
        const Rank pc = t.sector_primitive[static_cast<std::size_t>(p)];
        const Rank pr = t.primitive_rank(k.mate, len);
        const Rank base = t.sector_offset[static_cast<std::size_t>(p)];

        if (!k.blocked) {
            const Rank sr = t.support_rank(bits, occupied);
            return base + sr * pc + pr;
        }

        const int fixed_pos = len - 1 - fixed_bit;
        std::vector<std::uint8_t> compact;
        compact.reserve(static_cast<std::size_t>(len - 1));
        for (int pos = 0; pos < len; ++pos)
            if (pos != fixed_pos) compact.push_back(bits[static_cast<std::size_t>(pos)]);
        const Rank sr = t.support_rank(compact, occupied - 1);
        return base + t.sector_main[static_cast<std::size_t>(p)] + sr * pc + pr;
    }

    Key unrank(Rank rank) const {
        if (rank >= size()) fail("factor unrank range");
        const int p = t.sector_of(rank);
        const int occupied = 2 * p + 1;
        const Rank pc = t.sector_primitive[static_cast<std::size_t>(p)];
        Rank local = rank - t.sector_offset[static_cast<std::size_t>(p)];
        bool blocked = false;
        int len = t.W;
        std::vector<std::uint8_t> bits;
        Rank pr = 0;

        if (local < t.sector_main[static_cast<std::size_t>(p)]) {
            const Rank sr = local / pc;
            pr = local % pc;
            bits = t.support_unrank(t.W, occupied, sr);
        } else {
            blocked = true;
            len = t.W - 1;
            local -= t.sector_main[static_cast<std::size_t>(p)];
            const Rank sr = local / pc;
            pr = local % pc;
            const auto compact = t.support_unrank(t.W - 2, occupied - 1, sr);
            const int fixed_pos = len - 1 - fixed_bit;
            bits.reserve(static_cast<std::size_t>(len));
            int q = 0;
            for (int pos = 0; pos < len; ++pos) {
                if (pos == fixed_pos) bits.push_back(1);
                else bits.push_back(compact[static_cast<std::size_t>(q++)]);
            }
        }

        const auto primitive = t.primitive_unrank(occupied, pr);
        MateID m = 0;
        int q = 0;
        for (int pos = 0; pos < len; ++pos) {
            if (!bits[static_cast<std::size_t>(pos)]) continue;
            const int bit = len - 1 - pos;
            m |= MateID(primitive[static_cast<std::size_t>(q++)]) << (2 * bit);
        }
        if (q != occupied || !valid_mate(m, len)) fail("factor unrank mate");
        return Key{blocked, m};
    }
};

void verify_factor_position(
    const std::vector<MateID>& main,
    const std::vector<MateID>& block,
    const ProductionFactorTables& tables,
    int W,
    int p,
    bool reverse,
    Rank& max_indegree
) {
    const int next = reverse ? p + 1 : p - 1;
    ProductionFactorCodec src(tables, p - 1), dst(tables, next - 1);
    const auto src_keys = layout(main, block, p);
    const auto dst_keys = layout(main, block, next);
    if (src.size() != src_keys.size() || dst.size() != dst_keys.size())
        fail("factor layout dimension");

    std::vector<std::uint8_t> seen_s(static_cast<std::size_t>(src.size()));
    std::vector<std::uint8_t> seen_d(static_cast<std::size_t>(dst.size()));
    for (Key k : src_keys) {
        const Rank r = src.rank(k);
        if (r >= src.size() || seen_s[static_cast<std::size_t>(r)]++ || !(src.unrank(r) == k))
            fail("factor source codec");
    }
    for (Key k : dst_keys) {
        const Rank r = dst.rank(k);
        if (r >= dst.size() || seen_d[static_cast<std::size_t>(r)]++ || !(dst.unrank(r) == k))
            fail("factor destination codec");
    }

    std::vector<Coef> values(static_cast<std::size_t>(src.size()));
    std::vector<Coef> scatter(static_cast<std::size_t>(dst.size()));
    std::vector<Coef> gather(static_cast<std::size_t>(dst.size()));
    for (Rank s = 0; s < src.size(); ++s) {
        values[static_cast<std::size_t>(s)] = Coef(1 + (s % 1000003));
        const Key k = src.unrank(s);
        for (const auto& [d, c] : reduced_step_basis(k, W, p, reverse))
            scatter[static_cast<std::size_t>(dst.rank(d))] += c * values[static_cast<std::size_t>(s)];
    }
    for (Rank d = 0; d < dst.size(); ++d) {
        const Key k = dst.unrank(d);
        const Vec pre = inverse_reduced(k, W, p, reverse);
        max_indegree = std::max<Rank>(max_indegree, pre.size());
        for (const auto& [s, c] : pre)
            gather[static_cast<std::size_t>(d)] += c * values[static_cast<std::size_t>(src.rank(s))];
    }
    if (scatter != gather) fail("factorized signed gather");
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    if (maxW < 5 || maxW > 12) return 2;

    std::vector<std::vector<MateID>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        ProductionFactorTables tables(W);
        Rank max_indegree = 0;
        Rank largest_sector = 0;
        int largest_p = -1;
        for (int p = 0; p < static_cast<int>(tables.sector_main.size()); ++p) {
            const Rank n = tables.sector_main[static_cast<std::size_t>(p)] +
                           tables.sector_block[static_cast<std::size_t>(p)];
            if (n > largest_sector) { largest_sector = n; largest_p = p; }
        }
        for (int p = W - 1; p >= 3; --p)
            verify_factor_position(words[W], words[W - 1], tables, W, p, false, max_indegree);
        for (int p = 1; p <= W - 3; ++p)
            verify_factor_position(words[W], words[W - 1], tables, W, p, true, max_indegree);

        const Rank want = words[W].size() + words[W - 1].size() - words[W - 2].size();
        if (tables.size() != want) fail("factor total dimension");
        std::cout << "W=" << W
                  << " reduced=" << tables.size()
                  << " max_indegree=" << max_indegree
                  << " largest_sector_p=" << largest_p
                  << " largest_sector=" << largest_sector
                  << " factor_table_kib=" << double(tables.logical_bytes()) / 1024.0
                  << " rank_roundtrip=OK signed_gather=OK forward=OK reverse=OK\n";
    }

    ProductionFactorTables w28(28);
    Rank largest = 0;
    int largest_p = -1;
    for (int p = 0; p < static_cast<int>(w28.sector_main.size()); ++p) {
        const Rank n = w28.sector_main[static_cast<std::size_t>(p)] +
                       w28.sector_block[static_cast<std::size_t>(p)];
        if (n > largest) { largest = n; largest_p = p; }
    }
    std::cout << "W=28_theory reduced=" << w28.size()
              << " largest_sector_p=" << largest_p
              << " largest_sector=" << largest
              << " factor_table_kib=" << double(w28.logical_bytes()) / 1024.0
              << "\n";
    std::cout << "ALL_OK production_factorized_layout=1\n";
    return 0;
}
