#include <bits/stdc++.h>
using namespace std;

static constexpr uint32_t P = 1000000007u;

static uint32_t mod_pow(uint32_t a, uint32_t e) {
    uint64_t r = 1, x = a;
    while (e) {
        if (e & 1) r = r * x % P;
        x = x * x % P;
        e >>= 1;
    }
    return static_cast<uint32_t>(r);
}

static int encode(string_view s) {
    int x = 0;
    for (char c : s) x = x * 3 + (c == 'N' ? 0 : c == 'R' ? 1 : 2);
    return x;
}

static string decode(int code, int len) {
    string s(len, 'N');
    for (int i = len - 1; i >= 0; --i) {
        const int d = code % 3;
        code /= 3;
        s[i] = d == 0 ? 'N' : d == 1 ? 'R' : 'L';
    }
    return s;
}

static int prefix_height(string_view s) {
    int h = 1;
    for (char c : s) {
        if (c == 'R') --h;
        else if (c == 'L') ++h;
    }
    return h;
}

struct Row {
    int prefix_code;
    unordered_map<int, uint32_t> values;
};

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "usage: row_hankel_pivots DUMP W K\n";
        return 2;
    }

    const string path = argv[1];
    const int width = atoi(argv[2]);
    const int cut = atoi(argv[3]);

    vector<unordered_map<int, unordered_map<int, uint32_t>>> blocks(width + 2);
    ifstream in(path);
    int w;
    string word;
    uint32_t value;
    while (in >> w >> word >> value) {
        if (w != width || value == 0) continue;
        const string_view sv(word);
        const auto left = sv.substr(0, cut);
        const int h = prefix_height(left);
        if (h < 0 || h >= static_cast<int>(blocks.size())) continue;

        auto& slot = blocks[h][encode(left)][encode(sv.substr(cut))];
        slot += value;
        if (slot >= P) slot -= P;
    }

    for (int h = 0; h < static_cast<int>(blocks.size()); ++h) {
        vector<Row> rows;
        rows.reserve(blocks[h].size());
        for (auto& [prefix, values] : blocks[h]) {
            rows.push_back({prefix, move(values)});
        }
        sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) {
            return a.prefix_code < b.prefix_code;
        });

        unordered_map<int, unordered_map<int, uint32_t>> basis;
        vector<pair<int, int>> pivots;
        for (auto& row : rows) {
            auto values = move(row.values);
            while (!values.empty()) {
                int pivot = INT_MAX;
                for (const auto& [column, x] : values) {
                    if (x != 0 && column < pivot) pivot = column;
                }
                if (pivot == INT_MAX) break;

                const auto it = basis.find(pivot);
                if (it == basis.end()) {
                    const uint32_t inv = mod_pow(values[pivot], P - 2);
                    for (auto& [column, x] : values) {
                        x = static_cast<uint64_t>(x) * inv % P;
                    }
                    basis.emplace(pivot, move(values));
                    pivots.emplace_back(row.prefix_code, pivot);
                    break;
                }

                const uint32_t scale = values[pivot];
                for (const auto& [column, x] : it->second) {
                    const uint32_t sub = static_cast<uint64_t>(scale) * x % P;
                    const auto jt = values.find(column);
                    const uint32_t old = jt == values.end() ? 0 : jt->second;
                    const uint32_t next = old >= sub ? old - sub : old + P - sub;
                    if (next != 0) values[column] = next;
                    else if (jt != values.end()) values.erase(jt);
                }
            }
        }

        if (pivots.empty()) continue;
        cout << "h=" << h << " rank=" << pivots.size() << '\n';
        for (const auto& [prefix, suffix] : pivots) {
            cout << "  " << decode(prefix, cut)
                 << " -> " << decode(suffix, width - cut) << '\n';
        }
    }
}
