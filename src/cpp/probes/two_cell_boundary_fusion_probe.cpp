#pragma push_macro("main")
#undef main
#define main two_cell_stationary_snake_cycle_probe_main_unused
#include "two_cell_stationary_snake_cycle_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_fusion_rank.hpp"

namespace {

struct Dsu {
    std::vector<Rank> p, sz;
    explicit Dsu(Rank n) : p(static_cast<std::size_t>(n)), sz(static_cast<std::size_t>(n), 1) {
        for (Rank i = 0; i < n; ++i) p[static_cast<std::size_t>(i)] = i;
    }
    Rank find(Rank x) {
        Rank r = x;
        while (p[static_cast<std::size_t>(r)] != r)
            r = p[static_cast<std::size_t>(r)];
        while (p[static_cast<std::size_t>(x)] != x) {
            Rank q = p[static_cast<std::size_t>(x)];
            p[static_cast<std::size_t>(x)] = r;
            x = q;
        }
        return r;
    }
    void unite(Rank a, Rank b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (sz[static_cast<std::size_t>(a)] < sz[static_cast<std::size_t>(b)])
            std::swap(a, b);
        p[static_cast<std::size_t>(b)] = a;
        sz[static_cast<std::size_t>(a)] += sz[static_cast<std::size_t>(b)];
    }
    Rank size(Rank x) { return sz[static_cast<std::size_t>(find(x))]; }
};

Rank boundary_block_size(int outer_ones, const oneesan::twocell::RankTables& t) {
    Rank z = 0;
    for (std::uint32_t code = 0; code < 8; ++code)
        z += oneesan::twocell::primitive_count_for_occupied(
            outer_ones + oneesan::twocell::popcount32(code), t);
    for (std::uint32_t code = 0; code < 2; ++code)
        z += oneesan::twocell::primitive_count_for_occupied(
            outer_ones + 1 + oneesan::twocell::popcount32(code), t);
    return z;
}

std::uint32_t boundary_outer_mask(
    const Key& k,
    int W,
    int active
) {
    const auto p = snake_pack(k);
    const int start = W - 4;
    if (!p.type)
        return oneesan::twocell::remove_support_window(p.support, start, 3);
    const std::uint32_t support = oneesan::twocell::stationary_c_rebase_support(
        p.support, active, start);
    return oneesan::twocell::remove_support_window(support, start, 2);
}

void print_boundary_capacity(
    int W,
    Rank shared_bytes,
    Rank workspace,
    const oneesan::twocell::RankTables& rt
) {
    const int outer_bits = W - 4;
    Rank fit = 0, total = 0;
    int max_o = -1;
    for (int o = 0; o <= outer_bits; ++o) {
        const Rank blocks = rt.choose[outer_bits][o];
        const Rank n = boundary_block_size(o, rt);
        total += blocks * n;
        if (workspace + n * sizeof(std::uint32_t) <= shared_bytes) {
            fit += blocks * n;
            max_o = o;
        }
    }
    const double f = double(fit) / double(total);
    std::cout << "boundary_fusion_capacity"
              << " shared_bytes=" << shared_bytes
              << " max_outer_ones=" << max_o
              << " fused_state_fraction=" << f
              << " two_op_HBM_reduction=" << 0.5 * f
              << "\n";
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 11;
    if (maxW < 6 || maxW > 13) return 2;

    const auto rt = oneesan::twocell::make_rank_tables();
    const auto st = oneesan::twocell::make_stationary_rank_tables(rt);
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int n = 1; n <= maxW; ++n) words[n] = gen_words(n);

    for (int W = 6; W <= maxW; ++W) {
        const Rank n = st.total[W];
        Dsu dsu(n);
        const int last = W - 4;
        const int edge_active = W - 3;

        for (const Key& s : q_basis(W, last, words)) {
            const Rank sr = snake_rank(s, W, last, rt, st);
            for (const auto& [d, c] : K_basis(s, W, last)) {
                if (c != 1) fail("boundary fusion nonunit interior");
                dsu.unite(sr, snake_rank(d, W, edge_active, rt, st));
            }
        }
        for (const Key& s : q_basis(W, edge_active, words)) {
            const Rank sr = snake_rank(s, W, edge_active, rt, st);
            for (const auto& [d, c] : turn_right_basis(s, W)) {
                if (c < 1 || c > 2) fail("boundary fusion turn coefficient");
                dsu.unite(sr, snake_rank(d, W, edge_active, rt, st));
            }
        }

        std::map<Rank, std::uint32_t> root_outer;
        std::map<std::uint32_t, Rank> outer_root;
        for (const Key& k : q_basis(W, last, words)) {
            const Rank r = snake_rank(k, W, last, rt, st);
            const Rank root = dsu.find(r);
            const std::uint32_t outer = boundary_outer_mask(k, W, last);
            const auto a = root_outer.emplace(root, outer);
            if (!a.second && a.first->second != outer)
                fail("boundary fusion outer not invariant W=" + std::to_string(W));
            const auto b = outer_root.emplace(outer, root);
            if (!b.second && b.first->second != root)
                fail("boundary fusion outer split W=" + std::to_string(W));
        }

        const Rank expected_blocks = Rank(1) << (W - 4);
        if (root_outer.size() != expected_blocks || outer_root.size() != expected_blocks)
            fail("boundary fusion component count W=" + std::to_string(W));

        Rank max_block = 0;
        for (const auto& [root, outer] : root_outer) {
            const int o = oneesan::twocell::popcount32(outer);
            const Rank expected = boundary_block_size(o, rt);
            const Rank actual = dsu.size(root);
            if (actual != expected)
                fail("boundary fusion block formula W=" + std::to_string(W));
            max_block = std::max(max_block, actual);
        }

        std::cout << "W=" << W
                  << " states=" << n
                  << " boundary_union_blocks=" << root_outer.size()
                  << " expected_2pow=" << expected_blocks
                  << " max_block_states=" << max_block
                  << " invariant=outer_support_Wm4"
                  << " formula=A_local3+C_local1 OK\n";
    }

    const int W = 28;
    const int outer_bits = W - 4;
    Rank total = 0;
    for (int o = 0; o <= outer_bits; ++o)
        total += rt.choose[outer_bits][o] * boundary_block_size(o, rt);
    if (total != st.total[W]) fail("W28 boundary block total");
    std::cout << "W=28_theory states=" << total
              << " boundary_union_blocks=" << (Rank(1) << outer_bits)
              << " max_block_states=" << boundary_block_size(outer_bits, rt)
              << "\n";
    for (Rank kib : {64ULL, 96ULL, 128ULL, 160ULL, 192ULL, 228ULL, 256ULL})
        print_boundary_capacity(W, kib * 1024ULL, 4096ULL, rt);
    std::cout << "ALL_OK boundary_interior_turn_fusion=1\n";
    return 0;
}
