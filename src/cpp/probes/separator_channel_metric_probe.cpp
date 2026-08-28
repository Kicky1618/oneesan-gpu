#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using u64 = std::uint64_t;

// Candidate renewal-channel labels.  A word has symbols
//   '0', '+', '-', '|'
// and every gap between bars has zero +/- charge.  Bars are self-dual; the
// local qutrit metric sends + <-> -, 0 <-> 0.
static bool valid(const std::string& s, int want_bars) {
    int q = 0, bars = 0;
    for (char c : s) {
        if (c == '|') {
            if (q) return false;
            ++bars;
        } else if (c == '+') ++q;
        else if (c == '-') --q;
        else if (c != '0') return false;
    }
    return q == 0 && bars == want_bars;
}

static std::string dual(std::string s) {
    for (char& c : s) {
        if (c == '+') c = '-';
        else if (c == '-') c = '+';
    }
    return s;
}

static void gen_rec(int n, std::string& s, std::vector<std::string>& out) {
    if (int(s.size()) == n) { out.push_back(s); return; }
    for (char c : std::string("0+-|")) {
        s.push_back(c);
        gen_rec(n, s, out);
        s.pop_back();
    }
}

static std::vector<u64> central_trinomial(int n) {
    std::vector<u64> t(n + 1);
    for (int m = 0; m <= n; ++m) {
        std::vector<u64> dp(2 * m + 1), ndp(2 * m + 1);
        dp[m] = 1;
        for (int k = 0; k < m; ++k) {
            std::fill(ndp.begin(), ndp.end(), 0);
            for (int q = -k; q <= k; ++q) {
                u64 x = dp[m + q];
                ndp[m + q] += x;
                ndp[m + q + 1] += x;
                ndp[m + q - 1] += x;
            }
            dp.swap(ndp);
        }
        t[m] = dp[m];
    }
    return t;
}

static u64 channel_formula(int r, int h) {
    int d = r - h;
    if (d < 0) return 0;
    auto t = central_trinomial(d);
    std::vector<u64> p(d + 1), q(d + 1);
    p[0] = 1;
    for (int f = 0; f < h + 1; ++f) {
        std::fill(q.begin(), q.end(), 0);
        for (int i = 0; i <= d; ++i) if (p[i])
            for (int j = 0; i + j <= d; ++j)
                q[i + j] += p[i] * t[j];
        p.swap(q);
    }
    return p[d];
}

int main(int argc, char** argv) {
    int max_r = argc > 1 ? std::atoi(argv[1]) : 10;
    if (max_r > 11) max_r = 11; // explicit 4^r enumeration is intentional here.

    for (int r = 0; r <= max_r; ++r) {
        std::vector<std::string> all;
        std::string s;
        gen_rec(r, s, all);
        u64 row = 0;
        for (int h = 0; h <= r; ++h) {
            std::vector<std::string> ch;
            for (auto const& w : all) if (valid(w, h)) ch.push_back(w);
            u64 want = channel_formula(r, h);
            if (ch.size() != want) {
                std::cerr << "count mismatch r=" << r << " h=" << h
                          << " got=" << ch.size() << " want=" << want << "\n";
                return 1;
            }
            std::unordered_map<std::string, int> id;
            for (int i = 0; i < int(ch.size()); ++i) id.emplace(ch[i], i);
            for (int i = 0; i < int(ch.size()); ++i) {
                auto d = dual(ch[i]);
                auto it = id.find(d);
                if (it == id.end() || dual(d) != ch[i]) {
                    std::cerr << "dual mismatch r=" << r << " h=" << h << "\n";
                    return 2;
                }
            }
            row += ch.size();
            std::cout << "r=" << r << " h=" << h
                      << " channels=" << ch.size()
                      << " metric_nnz=" << ch.size() << "\n";
        }
        std::cout << "ROW r=" << r << " total=" << row
                  << " candidate_metric_nnz=" << row << "\n";
    }

    // Number of unnormalised tail words if the final gap is allowed to carry a
    // pending charge.  From one completed channel this is the maximum number of
    // raw extensions before projecting back to the renewal basis.
    std::unordered_map<int, u64> dp{{0,1}}, ndp;
    u64 total = 1;
    std::cout << "tail=0 raw_extensions=1\n";
    for (int t = 1; t <= 6; ++t) {
        ndp.clear();
        for (auto [q, x] : dp) {
            ndp[q] += x;       // 0
            ndp[q + 1] += x;   // +
            ndp[q - 1] += x;   // -
            if (q == 0) ndp[q] += x; // |
        }
        dp.swap(ndp);
        total = 0;
        for (auto [q, x] : dp) total += x;
        std::cout << "tail=" << t << " raw_extensions=" << total << "\n";
    }
    return 0;
}
