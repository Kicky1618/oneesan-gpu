#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::cpp_int;

namespace {

constexpr int MAXH = 9;

struct Key {
    uint64_t a = 0;
    uint64_t b = 0;
    bool operator==(const Key& o) const { return a == o.a && b == o.b; }
};
struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
        uint64_t x = k.a ^ (k.b + 0x9e3779b97f4a7c15ULL + (k.a << 6) + (k.a >> 2));
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        return size_t(x);
    }
};

struct State {
    uint16_t color = 0;
    std::array<uint8_t, MAXH> label{};
    std::array<uint8_t, MAXH> safe{};
    uint8_t nlabel = 0;
    uint8_t old_upper_left = 0;
};

Key pack(const State& s, int h) {
    __uint128_t z = s.color;
    int shift = h;
    for (int r = 0; r < h; ++r) {
        z |= __uint128_t(s.label[r]) << shift;
        shift += 4;
    }
    uint16_t sm = 0;
    for (int i = 0; i < s.nlabel; ++i) if (s.safe[i]) sm |= uint16_t(1u << i);
    z |= __uint128_t(sm) << shift;
    shift += h;
    z |= __uint128_t(s.old_upper_left) << shift;
    return {uint64_t(z), uint64_t(z >> 64)};
}

State unpack(Key k, int h) {
    __uint128_t z = __uint128_t(k.a) | (__uint128_t(k.b) << 64);
    State s;
    s.color = uint16_t(z & ((__uint128_t(1) << h) - 1));
    int shift = h;
    uint8_t mx = 0;
    for (int r = 0; r < h; ++r) {
        s.label[r] = uint8_t((z >> shift) & 15);
        mx = std::max(mx, s.label[r]);
        shift += 4;
    }
    s.nlabel = uint8_t(mx + 1);
    uint16_t sm = uint16_t((z >> shift) & ((1u << h) - 1));
    shift += h;
    for (int i = 0; i < s.nlabel; ++i) s.safe[i] = uint8_t((sm >> i) & 1);
    s.old_upper_left = uint8_t((z >> shift) & 1);
    return s;
}

// P0 is the top+right outer s-t arc; Q0 is left+bottom.
// On a real top boundary only color 1 has reached P0.  On a real bottom
// boundary only color 0 has reached Q0.  An artificial strip boundary allows
// either color to defer its required connection to the neighboring strip.
bool horizontal_boundary_safe(int color, bool artificial, bool top) {
    if (artificial) return true;
    return top ? color == 1 : color == 0;
}

State initial_column(int pattern, int h, bool top_artificial, bool bottom_artificial) {
    State s;
    s.color = uint16_t(pattern);
    int lab = -1, prev = -1;
    for (int r = 0; r < h; ++r) {
        int c = (pattern >> r) & 1;
        if (r == 0 || c != prev) {
            ++lab;
            s.safe[lab] = 0;
        }
        s.label[r] = uint8_t(lab);
        // Left boundary belongs to Q0, so only a 0-component is certified here.
        if (c == 0) s.safe[lab] = 1;
        if (r == 0 && horizontal_boundary_safe(c, top_artificial, true)) s.safe[lab] = 1;
        if (r == h - 1 && horizontal_boundary_safe(c, bottom_artificial, false)) s.safe[lab] = 1;
        prev = c;
    }
    s.nlabel = uint8_t(lab + 1);
    return s;
}

bool add_cell(
    const State& old, int h, int row, int value,
    bool top_artificial, bool bottom_artificial, State& out
) {
    int left_color = (old.color >> row) & 1;
    int up_color = row ? ((old.color >> (row - 1)) & 1) : 0;

    // The four cells of a completed 2x2 face block may not be checkerboard:
    // that would give degree four at the enclosed grid vertex.
    if (row > 0) {
        int ul = old.old_upper_left;
        if (ul != left_color && up_color != value && ul != up_color) return false;
    }

    const int k = old.nlabel;
    int parent[MAXH + 1];
    uint8_t safe[MAXH + 1]{};
    for (int i = 0; i <= k; ++i) {
        parent[i] = i;
        if (i < k) safe[i] = old.safe[i];
    }
    auto find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) {
            parent[b] = a;
            safe[a] |= safe[b];
        }
    };

    int fresh = k;
    if (left_color == value) unite(fresh, old.label[row]);
    if (row > 0 && up_color == value) unite(fresh, old.label[row - 1]);
    if (row == 0 && horizontal_boundary_safe(value, top_artificial, true)) safe[find(fresh)] = 1;
    if (row == h - 1 && horizontal_boundary_safe(value, bottom_artificial, false)) safe[find(fresh)] = 1;

    bool present[MAXH + 1]{};
    for (int r = 0; r < h; ++r) {
        int node = (r == row) ? fresh : old.label[r];
        present[find(node)] = true;
    }
    // If a component disappears from the strip frontier without reaching its
    // required real boundary or an artificial strip boundary, it can never be
    // repaired by unprocessed cells.
    for (int i = 0; i < k; ++i) {
        if (find(i) == i && !present[i] && !safe[i]) return false;
    }

    out = {};
    out.color = uint16_t((old.color & ~(1u << row)) | (uint16_t(value) << row));
    out.old_upper_left = uint8_t(left_color);
    int remap[MAXH + 1];
    std::fill(remap, remap + MAXH + 1, -1);
    int nk = 0;
    for (int r = 0; r < h; ++r) {
        int node = (r == row) ? fresh : old.label[r];
        int root = find(node);
        if (remap[root] < 0) {
            remap[root] = nk;
            out.safe[nk] = safe[root];
            ++nk;
        }
        out.label[r] = uint8_t(remap[root]);
    }
    out.nlabel = uint8_t(nk);
    if (row == h - 1) out.old_upper_left = 0;
    return true;
}

struct Phase {
    std::unordered_map<Key, int, KeyHash> ids;
    std::vector<Key> keys;
    std::vector<std::array<int, 2>> transition;

    int intern(Key k) {
        auto it = ids.find(k);
        if (it != ids.end()) return it->second;
        int id = int(keys.size());
        ids.emplace(k, id);
        keys.push_back(k);
        transition.push_back({-2, -2});
        return id;
    }
};

cpp_int strip_count(int h, int width, bool top_artificial, bool bottom_artificial) {
    if (h < 1 || h > MAXH || width < 1) std::exit(600);
    std::vector<Phase> phase(h);
    std::vector<cpp_int> dp;
    for (int x = 0; x < (1 << h); ++x) {
        int id = phase[0].intern(pack(initial_column(x, h, top_artificial, bottom_artificial), h));
        if (int(dp.size()) <= id) dp.resize(id + 1);
        dp[id] += 1;
    }

    for (int col = 1; col < width; ++col) {
        for (int row = 0; row < h; ++row) {
            int next_phase = (row + 1) % h;
            std::vector<cpp_int> ndp(phase[next_phase].keys.size());
            for (int id = 0; id < int(dp.size()); ++id) {
                if (dp[id] == 0) continue;
                for (int value = 0; value < 2; ++value) {
                    int& to = phase[row].transition[id][value];
                    if (to == -2) {
                        State old = unpack(phase[row].keys[id], h), next;
                        if (!add_cell(old, h, row, value, top_artificial, bottom_artificial, next)) {
                            to = -1;
                        } else {
                            to = phase[next_phase].intern(pack(next, h));
                        }
                    }
                    if (to >= 0) {
                        if (int(ndp.size()) <= to) ndp.resize(to + 1);
                        ndp[to] += dp[id];
                    }
                }
            }
            dp.swap(ndp);
        }
    }

    cpp_int total = 0;
    for (int id = 0; id < int(dp.size()); ++id) {
        if (dp[id] == 0) continue;
        State s = unpack(phase[0].keys[id], h);
        std::array<int, MAXH> color_of_label{};
        color_of_label.fill(-1);
        for (int r = 0; r < h; ++r) color_of_label[s.label[r]] = (s.color >> r) & 1;
        bool ok = true;
        for (int l = 0; l < s.nlabel; ++l) {
            // Right boundary belongs to P0, so an otherwise-unresolved
            // 1-component is certified at the final column; a 0-component is not.
            if (!s.safe[l] && color_of_label[l] != 1) {
                ok = false;
                break;
            }
        }
        if (ok) total += dp[id];
    }
    return total;
}

bool checkerboard_free(uint32_t bits, int h, int w) {
    auto get = [&](int r, int c) { return int((bits >> (c * h + r)) & 1u); };
    for (int c = 0; c + 1 < w; ++c) for (int r = 0; r + 1 < h; ++r) {
        int a = get(r, c), b = get(r + 1, c), x = get(r, c + 1), y = get(r + 1, c + 1);
        if (a != b && x != y && a != x) return false;
    }
    return true;
}

bool brute_component_condition(uint32_t bits, int h, int w, bool top_artificial, bool bottom_artificial) {
    if (!checkerboard_free(bits, h, w)) return false;
    std::vector<uint8_t> seen(h * w, 0);
    auto get = [&](int r, int c) { return int((bits >> (c * h + r)) & 1u); };
    constexpr int dr[4] = {-1, 1, 0, 0};
    constexpr int dc[4] = {0, 0, -1, 1};
    for (int sr = 0; sr < h; ++sr) for (int sc = 0; sc < w; ++sc) {
        int si = sc * h + sr;
        if (seen[si]) continue;
        int color = get(sr, sc);
        bool safe = false;
        std::queue<std::pair<int, int>> q;
        q.push({sr, sc});
        seen[si] = 1;
        while (!q.empty()) {
            auto [r, c] = q.front(); q.pop();
            if (c == 0 && color == 0) safe = true; // Q0 left
            if (c == w - 1 && color == 1) safe = true; // P0 right
            if (r == 0 && horizontal_boundary_safe(color, top_artificial, true)) safe = true;
            if (r == h - 1 && horizontal_boundary_safe(color, bottom_artificial, false)) safe = true;
            for (int d = 0; d < 4; ++d) {
                int rr = r + dr[d], cc = c + dc[d];
                if (rr < 0 || rr >= h || cc < 0 || cc >= w || get(rr, cc) != color) continue;
                int ii = cc * h + rr;
                if (!seen[ii]) {
                    seen[ii] = 1;
                    q.push({rr, cc});
                }
            }
        }
        if (!safe) return false;
    }
    return true;
}

uint64_t brute_count(int h, int w, bool top_artificial, bool bottom_artificial) {
    int cells = h * w;
    if (cells > 20) std::exit(601);
    uint64_t total = 0;
    for (uint32_t bits = 0; bits < (1u << cells); ++bits)
        total += brute_component_condition(bits, h, w, top_artificial, bottom_artificial);
    return total;
}

} // namespace

int main() {
    // Independent small-instance check of the frontier implementation.
    for (int ta = 0; ta < 2; ++ta) for (int ba = 0; ba < 2; ++ba) {
        cpp_int dp = strip_count(3, 4, ta, ba);
        uint64_t brute = brute_count(3, 4, ta, ba);
        if (dp != brute) {
            std::cerr << "small brute mismatch top_art=" << ta << " bottom_art=" << ba
                      << " dp=" << dp << " brute=" << brute << '\n';
            return 602;
        }
    }

    const cpp_int expected_top("1439363966680482394681847048772970007433626003156790462009370");
    const cpp_int expected_middle("22942552281959548690313451479726513472161304029234933083393982");

    cpp_int top = strip_count(9, 27, false, true);
    cpp_int middle = strip_count(9, 27, true, true);
    cpp_int bottom = strip_count(9, 27, true, false);
    if (top != expected_top || bottom != expected_top || middle != expected_middle) {
        std::cerr << "n27 P0/Q0 strip constant mismatch\n"
                  << "top=" << top << "\nmiddle=" << middle << "\nbottom=" << bottom << '\n';
        return 603;
    }

    cpp_int bound = top * middle * bottom;
    const uint32_t primes[] = {
        4294967291u, 4294967279u, 4294967231u, 4294967197u,
        4294967189u, 4294967161u, 4294967143u, 4294967111u,
        4294967087u, 4294967029u, 4294966997u, 4294966981u,
        4294966943u, 4294966927u, 4294966909u, 4294966877u,
        4294966829u, 4294966813u, 4294966769u
    };
    cpp_int m18 = 1;
    for (int i = 0; i < 18; ++i) m18 *= primes[i];
    cpp_int m19 = m18 * primes[18];
    if (!(m18 <= bound && bound < m19)) {
        std::cerr << "CRT threshold mismatch\n";
        return 604;
    }

    std::cout << "pq-component-strip-bound OK"
              << " top=" << top
              << " middle=" << middle
              << " bottom=" << bottom
              << " bound=" << bound
              << " bound_bits=" << (boost::multiprecision::msb(bound) + 1)
              << " crt_primes=19\n";
    return 0;
}
