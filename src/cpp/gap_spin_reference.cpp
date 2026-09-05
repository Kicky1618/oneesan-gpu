#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>

namespace {

// 4 bits/slot: 0..13 = partner slot, 14 = source defect, 15 = empty.
constexpr std::uint8_t DEF = 14;
constexpr std::uint8_t EMP = 15;

std::uint8_t get(std::uint64_t s, unsigned i) { return (s >> (4 * i)) & 15u; }
std::uint64_t set(std::uint64_t s, unsigned i, std::uint8_t v) {
    const auto sh = 4 * i;
    return (s & ~(std::uint64_t{15} << sh)) | (std::uint64_t{v} << sh);
}

std::uint64_t empty_state(unsigned slots) {
    std::uint64_t s = 0;
    for (unsigned i = 0; i < slots; ++i) s = set(s, i, EMP);
    return s;
}

unsigned active_count(std::uint64_t s, unsigned slots) {
    unsigned n = 0;
    for (unsigned i = 0; i < slots; ++i) n += get(s, i) != EMP;
    return n;
}

using Table = std::unordered_map<std::uint64_t, std::uint64_t>;
void add(Table& t, std::uint64_t s, std::uint64_t v) { t[s] += v; }

std::uint64_t solve(unsigned n, bool verbose) {
    const unsigned slots = n + 1;
    if (slots > 14) std::abort();
    Table cur, nxt;
    cur[empty_state(slots)] = 1;
    std::size_t peak = 1;

    for (unsigned y = 0; y < n; ++y) {
        for (unsigned x = 0; x < n; ++x) {
            nxt.clear();
            nxt.reserve(cur.size() * 2 + 32);
            const unsigned p = x, q = x + 1;
            const bool source = x == 0 && y == 0;
            const bool target = x + 1 == n && y + 1 == n;
            const bool down = y + 1 < n;
            const bool right = x + 1 < n;

            for (const auto& kv : cur) {
                auto s = kv.first;
                const auto cnt = kv.second;
                const auto a = get(s, p), b = get(s, q);
                const unsigned in = (a != EMP) + (b != EMP);

                if (source) {
                    if (active_count(s, slots) != 0) continue;
                    if (down) add(nxt, set(s, p, DEF), cnt);
                    if (right) add(nxt, set(s, q, DEF), cnt);
                    continue;
                }

                if (target) {
                    if (active_count(s, slots) != 1 || in != 1) continue;
                    const auto v = a != EMP ? a : b;
                    if (v != DEF) continue;
                    s = set(set(s, p, EMP), q, EMP);
                    add(nxt, s, cnt);
                    continue;
                }

                auto base = set(set(s, p, EMP), q, EMP);
                if (in == 0) {
                    // degree 0
                    add(nxt, base, cnt);
                    // cup: pair p,q
                    if (down && right) {
                        auto z = set(set(base, p, static_cast<std::uint8_t>(q)), q,
                                     static_cast<std::uint8_t>(p));
                        add(nxt, z, cnt);
                    }
                } else if (in == 1) {
                    const auto label = a != EMP ? a : b;
                    auto move_to = [&](unsigned pos) {
                        auto z = base;
                        if (label == DEF) {
                            z = set(z, pos, DEF);
                        } else {
                            // Rewire the partner to the new slot.
                            z = set(z, pos, label);
                            z = set(z, label, static_cast<std::uint8_t>(pos));
                        }
                        add(nxt, z, cnt);
                    };
                    if (down) move_to(p);
                    if (right) move_to(q);
                } else {
                    // cap: connect endpoints at p,q.  If already paired, this closes
                    // a contractible loop, whose beta=0 weight is zero.
                    if (a == q && b == p) continue;

                    if (a == DEF || b == DEF) {
                        const auto other = a == DEF ? b : a;
                        if (other == EMP || other == DEF) continue;
                        auto z = base;
                        z = set(z, other, DEF);
                        add(nxt, z, cnt);
                    } else {
                        if (a == EMP || b == EMP || a == DEF || b == DEF) continue;
                        auto z = base;
                        z = set(z, a, b);
                        z = set(z, b, a);
                        add(nxt, z, cnt);
                    }
                }
            }
            cur.swap(nxt);
            peak = std::max(peak, cur.size());
            if (verbose) std::cerr << "ref n=" << n << " y=" << y << " x=" << x << " states=" << cur.size() << '\n';
        }

        // Reindex slots [0..n-1] -> [1..n], with new slot 0 empty.
        nxt.clear();
        nxt.reserve(cur.size() + 8);
        for (const auto& kv : cur) {
            const auto s = kv.first;
            const auto cnt = kv.second;
            if (get(s, n) != EMP) continue;
            auto z = empty_state(slots);
            for (unsigned i = 0; i < n; ++i) {
                const auto v = get(s, i);
                if (v == EMP) continue;
                if (v == DEF) z = set(z, i + 1, DEF);
                else z = set(z, i + 1, static_cast<std::uint8_t>(v + 1));
            }
            add(nxt, z, cnt);
        }
        cur.swap(nxt);
    }

    if (verbose) std::cerr << "ref peak=" << peak << '\n';
    const auto e = empty_state(slots);
    auto it = cur.find(e);
    return it == cur.end() ? 0 : it->second;
}

} // namespace

int main(int argc, char** argv) {
    unsigned a = argc > 1 ? std::strtoul(argv[1], nullptr, 10) : 2;
    unsigned b = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 9;
    bool v = argc > 3;
    for (unsigned n = a; n <= b; ++n) std::cout << n << ' ' << solve(n, v) << '\n';
}
