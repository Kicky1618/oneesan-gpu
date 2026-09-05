#pragma push_macro("main")
#undef main
#define main two_cell_turn_closed_block_probe_main_unused
#include "two_cell_turn_closed_block_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_stationary_rank.hpp"

namespace {

using Value = std::uint64_t;
constexpr Value kMod = 1000000007ULL;

Value madd(Value a, Value b) { return (a + b) % kMod; }
Value mmul(Value a, std::int64_t b) {
    return (a * static_cast<Value>(b % static_cast<std::int64_t>(kMod))) % kMod;
}

oneesan::twocell::PackedKey snake_pack(const Key& k) {
    oneesan::twocell::PackedKey z{};
    z.type = static_cast<std::uint8_t>(k.type == 'C');
    for (int p = 0; p < static_cast<int>(k.w.size()); ++p) {
        const std::uint32_t bit = std::uint32_t(1) << p;
        if (k.w[p] != N) z.support |= bit;
        if (k.w[p] == L) z.left |= bit;
    }
    return z;
}

Rank snake_rank(
    const Key& k,
    int W,
    int active,
    const oneesan::twocell::RankTables& rt,
    const oneesan::twocell::StationaryRankTables& st
) {
    return oneesan::twocell::stationary_rank(snake_pack(k), W, active, rt, st);
}

struct Dsu {
    std::vector<int> p;
    explicit Dsu(int n) : p(static_cast<std::size_t>(n), -1) {}
    int find(int x) {
        if (p[static_cast<std::size_t>(x)] < 0) return x;
        return p[static_cast<std::size_t>(x)] = find(p[static_cast<std::size_t>(x)]);
    }
    void unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return;
        if (p[static_cast<std::size_t>(a)] > p[static_cast<std::size_t>(b)]) std::swap(a, b);
        p[static_cast<std::size_t>(a)] += p[static_cast<std::size_t>(b)];
        p[static_cast<std::size_t>(b)] = a;
    }
};

std::map<Key, Value> exact_forward(
    const std::map<Key, Value>& in, int W, int i
) {
    std::map<Key, Value> out;
    for (const auto& [s, x] : in)
        for (const auto& [d, c] : K_basis(s, W, i))
            out[d] = madd(out[d], mmul(x, c));
    return out;
}

std::map<Key, Value> exact_reverse(
    const std::map<Key, Value>& in, int W, int pair
) {
    std::map<Key, Value> out;
    for (const auto& [s, x] : in)
        for (const auto& [d, c] : K_reverse_basis(s, W, pair))
            out[d] = madd(out[d], mmul(x, c));
    return out;
}

std::map<Key, Value> exact_turn(
    const std::map<Key, Value>& in, int W, bool right
) {
    std::map<Key, Value> out;
    for (const auto& [s, x] : in) {
        const CVec col = right ? turn_right_basis(s, W) : turn_left_basis(s, W);
        for (const auto& [d, c] : col) out[d] = madd(out[d], mmul(x, c));
    }
    return out;
}

template <class ColumnFn>
Rank generic_component_inplace(
    std::vector<Value>& values,
    const std::vector<Key>& source_basis,
    int W,
    int source_active,
    int destination_active,
    const oneesan::twocell::RankTables& rt,
    const oneesan::twocell::StationaryRankTables& st,
    ColumnFn column
) {
    const int n = static_cast<int>(values.size());
    if (static_cast<int>(source_basis.size()) != n)
        fail("snake generic source dimension");

    std::vector<Key> source_at_rank(static_cast<std::size_t>(n));
    std::vector<std::uint8_t> seen(static_cast<std::size_t>(n));
    Dsu dsu(n);
    for (const Key& s : source_basis) {
        const Rank sr64 = snake_rank(s, W, source_active, rt, st);
        if (sr64 >= static_cast<Rank>(n)) fail("snake generic source rank");
        const int sr = static_cast<int>(sr64);
        if (seen[static_cast<std::size_t>(sr)]++) fail("snake generic rank collision");
        source_at_rank[static_cast<std::size_t>(sr)] = s;
        for (const auto& [d, c] : column(s)) {
            if (!c) continue;
            const Rank dr64 = snake_rank(d, W, destination_active, rt, st);
            if (dr64 >= static_cast<Rank>(n)) fail("snake generic destination rank");
            dsu.unite(sr, static_cast<int>(dr64));
        }
    }

    std::map<int, std::vector<int>> component;
    for (int r = 0; r < n; ++r) component[dsu.find(r)].push_back(r);

    for (const auto& [root, ranks] : component) {
        (void)root;
        std::map<int, int> local;
        std::vector<Value> x(ranks.size()), y(ranks.size());
        for (int q = 0; q < static_cast<int>(ranks.size()); ++q) {
            local.emplace(ranks[static_cast<std::size_t>(q)], q);
            x[static_cast<std::size_t>(q)] =
                values[static_cast<std::size_t>(ranks[static_cast<std::size_t>(q)])];
        }
        for (int q = 0; q < static_cast<int>(ranks.size()); ++q) {
            const int sr = ranks[static_cast<std::size_t>(q)];
            const Key& s = source_at_rank[static_cast<std::size_t>(sr)];
            for (const auto& [d, c] : column(s)) {
                const int dr = static_cast<int>(snake_rank(
                    d, W, destination_active, rt, st));
                const auto it = local.find(dr);
                if (it == local.end()) fail("snake generic edge leaves component");
                y[static_cast<std::size_t>(it->second)] = madd(
                    y[static_cast<std::size_t>(it->second)], mmul(x[static_cast<std::size_t>(q)], c));
            }
        }
        for (int q = 0; q < static_cast<int>(ranks.size()); ++q)
            values[static_cast<std::size_t>(ranks[static_cast<std::size_t>(q)])] =
                y[static_cast<std::size_t>(q)];
    }
    return component.size();
}

void local_turn_inplace(
    std::vector<Value>& v,
    int W,
    bool right,
    const std::vector<std::vector<Word>>& words,
    const oneesan::twocell::RankTables& rt,
    const oneesan::twocell::StationaryRankTables& st
) {
    const int active = right ? W - 3 : 0;
    for (const Word& u : words[W - 2]) {
        ClosedTurnComponent c;
        if (right) {
            c = closed_right_turn_component(u, W);
        } else {
            const ClosedTurnComponent r = closed_right_turn_component(reflect_word(u), W);
            c.singular = r.singular;
            c.alpha = reflect_key(r.alpha);
            c.beta = reflect_key(r.beta);
            for (const Key& p : r.passive) c.passive.push_back(reflect_key(p));
        }
        const auto states = c.states();
        std::vector<Value> x(states.size()), y(states.size());
        std::vector<Rank> rank(states.size());
        for (int q = 0; q < static_cast<int>(states.size()); ++q) {
            rank[static_cast<std::size_t>(q)] = snake_rank(states[static_cast<std::size_t>(q)], W, active, rt, st);
            x[static_cast<std::size_t>(q)] = v[static_cast<std::size_t>(rank[static_cast<std::size_t>(q)])];
        }

        if (c.singular) {
            const Value t0 = madd(x[0], x[2]);
            const Value t1 = madd(x[1], x[2]);
            y[0] = madd(t0, t0);
            y[1] = y[2] = t1;
        } else {
            Value t = x[1];
            for (std::size_t q = 2; q < x.size(); ++q) t = madd(t, x[q]);
            y[0] = madd(madd(x[0], x[0]), t);
            y[1] = t;
            for (std::size_t q = 2; q < x.size(); ++q) y[q] = madd(x[q], x[q]);
        }
        for (int q = 0; q < static_cast<int>(states.size()); ++q)
            v[static_cast<std::size_t>(rank[static_cast<std::size_t>(q)])] = y[static_cast<std::size_t>(q)];
    }
}

} // namespace

int main(int argc, char** argv) {
    const int maxW = argc > 1 ? std::atoi(argv[1]) : 10;
    if (maxW < 5 || maxW > 12) return 2;

    const auto rt = oneesan::twocell::make_rank_tables();
    const auto st = oneesan::twocell::make_stationary_rank_tables(rt);
    std::vector<std::vector<Word>> words(static_cast<std::size_t>(maxW + 1));
    for (int W = 1; W <= maxW; ++W) words[W] = gen_words(W);

    for (int W = 5; W <= maxW; ++W) {
        const auto start_basis = q_basis(W, 0, words);
        std::map<Key, Value> exact;
        std::vector<Value> stationary(static_cast<std::size_t>(st.total[W]));
        for (Rank q = 0; q < start_basis.size(); ++q) {
            const Value x = 1 + ((q * 2654435761ULL + 97ULL) % (kMod - 1));
            exact[start_basis[static_cast<std::size_t>(q)]] = x;
            const Rank r = snake_rank(start_basis[static_cast<std::size_t>(q)], W, 0, rt, st);
            stationary[static_cast<std::size_t>(r)] = x;
        }

        Rank forward_steps = 0, reverse_steps = 0;
        Rank component_partitions = 0;
        for (int i = 0; i <= W - 4; ++i) {
            exact = exact_forward(exact, W, i);
            const auto basis = q_basis(W, i, words);
            component_partitions += generic_component_inplace(
                stationary, basis, W, i, i + 1, rt, st,
                [&](const Key& s) { return K_basis(s, W, i); });
            ++forward_steps;
        }
        exact = exact_turn(exact, W, true);
        local_turn_inplace(stationary, W, true, words, rt, st);

        for (int pair = W - 2; pair >= 2; --pair) {
            exact = exact_reverse(exact, W, pair);
            const auto basis = reverse_q_basis(W, pair, words);
            component_partitions += generic_component_inplace(
                stationary, basis, W, pair - 1, pair - 2, rt, st,
                [&](const Key& s) { return K_reverse_basis(s, W, pair); });
            ++reverse_steps;
        }
        exact = exact_turn(exact, W, false);
        local_turn_inplace(stationary, W, false, words, rt, st);

        const auto end_basis = q_basis(W, 0, words);
        for (const Key& k : end_basis) {
            const Rank r = snake_rank(k, W, 0, rt, st);
            const Value a = exact.count(k) ? exact[k] : 0;
            const Value b = stationary[static_cast<std::size_t>(r)];
            if (a != b)
                fail("stationary snake cycle mismatch W=" + std::to_string(W));
        }

        const Rank transfers = forward_steps + reverse_steps + 2;
        std::cout << "W=" << W
                  << " states=" << st.total[W]
                  << " forward_steps=" << forward_steps
                  << " reverse_steps=" << reverse_steps
                  << " turns=2"
                  << " transfer_steps=" << transfers
                  << " component_partitions=" << component_partitions
                  << " layout_conversions=0"
                  << " global_vectors=1"
                  << " stationary_cycle=OK\n";
    }

    std::cout << "W=28_theory two_row_cycle_transfers=" << (2 * (28 - 3) + 2)
              << " layout_conversions=0"
              << " global_u32_vectors=1"
              << " vector_GiB="
              << double(st.total[28] * 4ULL) / double(1ULL << 30)
              << "\n";
    std::cout << "ALL_OK stationary_two_row_snake_cycle=1\n";
    return 0;
}
