#pragma push_macro("main")
#undef main
#define main two_cell_reverse_stationary_probe_main_unused
#include "two_cell_reverse_stationary_probe.cpp"
#pragma pop_macro("main")

#pragma push_macro("main")
#undef main
#define main two_cell_turn_closed_device_probe_main_unused
#include "two_cell_turn_closed_device_probe.cpp"
#pragma pop_macro("main")

#include "../../common/two_cell_stationary_rank.hpp"

namespace {

using Value = std::uint64_t;
constexpr Value kMod = 1000000007ULL;

Value madd(Value a, Value b) { return (a + b) % kMod; }
Value mmul(Value a, std::int64_t b) {
    return (a * static_cast<Value>(b % static_cast<std::int64_t>(kMod))) % kMod;
}

Rank snake_rank(
    const Key& k,
    int W,
    int active,
    const oneesan::twocell::RankTables& rt,
    const oneesan::twocell::StationaryRankTables& st
) {
    return oneesan::twocell::stationary_rank(
        reverse_stationary_pack(k), W, active, rt, st);
}

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

void local_forward_inplace(
    std::vector<Value>& v,
    int W,
    int i,
    const std::vector<std::vector<Word>>& words,
    const oneesan::twocell::RankTables& rt,
    const oneesan::twocell::StationaryRankTables& st
) {
    for (const Word& u : words[W - 2]) {
        const auto packed = packed_direct_component_sources(pack_word(u), W, i);
        std::vector<Key> src;
        for (int q = 0; q < packed.size; ++q) src.push_back(unpack_key(packed.value[q]));
        std::vector<Rank> rank(src.size());
        std::map<Rank, int> slot;
        std::vector<Value> x(src.size()), y(src.size());
        for (int q = 0; q < static_cast<int>(src.size()); ++q) {
            rank[q] = snake_rank(src[q], W, i, rt, st);
            if (!slot.emplace(rank[q], q).second) fail("snake forward local rank collision");
            x[q] = v[static_cast<std::size_t>(rank[q])];
        }
        for (int q = 0; q < static_cast<int>(src.size()); ++q) {
            for (const auto& [d, c] : K_basis(src[q], W, i)) {
                const Rank dr = snake_rank(d, W, i + 1, rt, st);
                const auto it = slot.find(dr);
                if (it == slot.end()) fail("snake forward destination leaves component");
                y[it->second] = madd(y[it->second], mmul(x[q], c));
            }
        }
        for (int q = 0; q < static_cast<int>(src.size()); ++q)
            v[static_cast<std::size_t>(rank[q])] = y[q];
    }
}

void local_reverse_inplace(
    std::vector<Value>& v,
    int W,
    int pair,
    const std::vector<std::vector<Word>>& words,
    const oneesan::twocell::RankTables& rt,
    const oneesan::twocell::StationaryRankTables& st
) {
    const int i = W - 2 - pair;
    const int src_active = pair - 1;
    const int dst_active = pair - 2;
    for (const Word& u : words[W - 2]) {
        const auto packed = packed_direct_component_sources(pack_word(u), W, i);
        std::vector<Key> src;
        for (int q = 0; q < packed.size; ++q)
            src.push_back(reflect_key(unpack_key(packed.value[q])));
        std::vector<Rank> rank(src.size());
        std::map<Rank, int> slot;
        std::vector<Value> x(src.size()), y(src.size());
        for (int q = 0; q < static_cast<int>(src.size()); ++q) {
            rank[q] = snake_rank(src[q], W, src_active, rt, st);
            if (!slot.emplace(rank[q], q).second) fail("snake reverse local rank collision");
            x[q] = v[static_cast<std::size_t>(rank[q])];
        }
        for (int q = 0; q < static_cast<int>(src.size()); ++q) {
            for (const auto& [d, c] : K_reverse_basis(src[q], W, pair)) {
                const Rank dr = snake_rank(d, W, dst_active, rt, st);
                const auto it = slot.find(dr);
                if (it == slot.end()) fail("snake reverse destination leaves component");
                y[it->second] = madd(y[it->second], mmul(x[q], c));
            }
        }
        for (int q = 0; q < static_cast<int>(src.size()); ++q)
            v[static_cast<std::size_t>(rank[q])] = y[q];
    }
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
        std::vector<Value> x(states.size());
        std::vector<Rank> rank(states.size());
        for (int q = 0; q < static_cast<int>(states.size()); ++q) {
            rank[q] = snake_rank(states[q], W, active, rt, st);
            x[q] = v[static_cast<std::size_t>(rank[q])];
        }

        std::vector<Value> y(states.size());
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
            v[static_cast<std::size_t>(rank[q])] = y[q];
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
        for (int i = 0; i <= W - 4; ++i) {
            exact = exact_forward(exact, W, i);
            local_forward_inplace(stationary, W, i, words, rt, st);
            ++forward_steps;
        }
        exact = exact_turn(exact, W, true);
        local_turn_inplace(stationary, W, true, words, rt, st);

        for (int pair = W - 2; pair >= 2; --pair) {
            exact = exact_reverse(exact, W, pair);
            local_reverse_inplace(stationary, W, pair, words, rt, st);
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
