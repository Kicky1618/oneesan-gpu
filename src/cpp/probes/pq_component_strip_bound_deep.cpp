#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <boost/multiprecision/cpp_int.hpp>

using boost::multiprecision::cpp_int;

namespace {

constexpr int MAXH = 12;

struct Key {
    uint64_t a = 0, b = 0;
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
    uint16_t safe_mask = 0;
    for (int i = 0; i < s.nlabel; ++i)
        if (s.safe[i]) safe_mask |= uint16_t(1u << i);
    z |= __uint128_t(safe_mask) << shift;
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
    uint16_t safe_mask = uint16_t((z >> shift) & ((1u << h) - 1));
    shift += h;
    for (int i = 0; i < s.nlabel; ++i) s.safe[i] = uint8_t((safe_mask >> i) & 1);
    s.old_upper_left = uint8_t((z >> shift) & 1);
    return s;
}

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
        if (c == 0) s.safe[lab] = 1; // Q0 left boundary
        if (r == 0 && horizontal_boundary_safe(c, top_artificial, true)) s.safe[lab] = 1;
        if (r == h - 1 && horizontal_boundary_safe(c, bottom_artificial, false)) s.safe[lab] = 1;
        prev = c;
    }
    s.nlabel = uint8_t(lab + 1);
    return s;
}

bool add_cell(const State& old, int h, int row, int value,
              bool top_artificial, bool bottom_artificial, State& out) {
    int left_color = (old.color >> row) & 1;
    int up_color = row ? ((old.color >> (row - 1)) & 1) : 0;
    if (row > 0) {
        int upper_left = old.old_upper_left;
        if (upper_left != left_color && up_color != value && upper_left != up_color)
            return false; // checkerboard 2x2 => degree four
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
        a = find(a); b = find(b);
        if (a != b) {
            parent[b] = a;
            safe[a] |= safe[b];
        }
    };

    int fresh = k;
    if (left_color == value) unite(fresh, old.label[row]);
    if (row > 0 && up_color == value) unite(fresh, old.label[row - 1]);
    if (row == 0 && horizontal_boundary_safe(value, top_artificial, true))
        safe[find(fresh)] = 1;
    if (row == h - 1 && horizontal_boundary_safe(value, bottom_artificial, false))
        safe[find(fresh)] = 1;

    bool present[MAXH + 1]{};
    for (int r = 0; r < h; ++r) {
        int node = (r == row) ? fresh : old.label[r];
        present[find(node)] = true;
    }
    for (int i = 0; i < k; ++i)
        if (find(i) == i && !present[i] && !safe[i]) return false;

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
    if (h < 1 || h > MAXH || width < 1) std::exit(620);
    std::vector<Phase> phase(h);
    std::vector<cpp_int> dp;
    for (int pattern = 0; pattern < (1 << h); ++pattern) {
        int id = phase[0].intern(pack(initial_column(pattern, h, top_artificial, bottom_artificial), h));
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
                        State next;
                        if (!add_cell(unpack(phase[row].keys[id], h), h, row, value,
                                      top_artificial, bottom_artificial, next))
                            to = -1;
                        else
                            to = phase[next_phase].intern(pack(next, h));
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
        for (int r = 0; r < h; ++r)
            color_of_label[s.label[r]] = (s.color >> r) & 1;
        bool ok = true;
        for (int lab = 0; lab < s.nlabel; ++lab) {
            // Right boundary is P0, so unresolved 1-components become safe here.
            if (!s.safe[lab] && color_of_label[lab] != 1) {
                ok = false;
                break;
            }
        }
        if (ok) total += dp[id];
    }
    return total;
}

} // namespace

int main() {
    const cpp_int expected_top9(
        "1439363966680482394681847048772970007433626003156790462009370");
    const cpp_int expected_top12(
        "52999285085137477335762761439368203729124254768255645682708867086111555458359993");
    const cpp_int expected_middle3("5560340541250024201342");

    // Cross-check the deeper implementation against the independently
    // brute-validated height-9 probe before trusting the height-12 result.
    cpp_int top9 = strip_count(9, 27, false, true);
    if (top9 != expected_top9) {
        std::cerr << "deep probe height-9 cross-check failed\n";
        return 621;
    }
    cpp_int top12 = strip_count(12, 27, false, true);
    cpp_int middle3 = strip_count(3, 27, true, true);
    if (top12 != expected_top12 || middle3 != expected_middle3) {
        std::cerr << "deep probe exact constant mismatch\n"
                  << "top12=" << top12 << "\nmiddle3=" << middle3 << '\n';
        return 622;
    }

    cpp_int bound = top12 * middle3 * top12;
    const cpp_int expected_bound(
        "15618575215183301705063439765068401724388506597738859931455510208310320299436859660966963881375212405652267800049353132341876865432031342825300421387932561298182015870844774718185758");
    if (bound != expected_bound) return 623;

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
    if (!(m18 <= bound && bound < m19)) return 624;

    std::cout << "pq-component-strip-bound-deep OK"
              << " strips=12,3,12"
              << " top12=" << top12
              << " middle3=" << middle3
              << " bound=" << bound
              << " bound_bits=" << (boost::multiprecision::msb(bound) + 1)
              << " crt_primes=19\n";
    return 0;
}
