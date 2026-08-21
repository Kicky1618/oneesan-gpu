#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <vector>

using Count = unsigned long long;

struct State {
    std::vector<uint8_t> degree;
    std::vector<uint8_t> comp;
    std::vector<uint8_t> flags; // bit0=start, bit1=target
    bool done = false;
};

struct Key {
    std::vector<uint8_t> bytes;
    bool operator==(const Key& o) const { return bytes == o.bytes; }
};

struct KeyHash {
    size_t operator()(const Key& k) const noexcept {
        uint64_t h = 1469598103934665603ULL;
        for (uint8_t x : k.bytes) {
            h ^= x;
            h *= 1099511628211ULL;
        }
        return static_cast<size_t>(h);
    }
};

static void canonicalize(State& s) {
    std::array<uint8_t, 64> remap{};
    std::array<uint8_t, 64> new_flags{};
    uint8_t next = 1;
    for (size_t i = 0; i < s.comp.size(); ++i) {
        uint8_t c = s.comp[i];
        if (!c) continue;
        if (!remap[c]) {
            remap[c] = next;
            new_flags[next] = s.flags[c];
            ++next;
        }
        s.comp[i] = remap[c];
    }
    s.flags.assign(next, 0);
    for (uint8_t c = 1; c < next; ++c) s.flags[c] = new_flags[c];
}

static Key make_key(State s) {
    canonicalize(s);
    Key k;
    k.bytes.reserve(2 + 2 * s.degree.size() + s.flags.size());
    k.bytes.push_back(s.done ? 1 : 0);
    k.bytes.push_back(static_cast<uint8_t>(s.degree.size()));
    for (size_t i = 0; i < s.degree.size(); ++i) {
        k.bytes.push_back(s.degree[i]);
        k.bytes.push_back(s.comp[i]);
    }
    k.bytes.push_back(0xff);
    for (size_t i = 1; i < s.flags.size(); ++i) k.bytes.push_back(s.flags[i]);
    return k;
}

static State decode(const Key& k) {
    State s;
    size_t p = 0;
    s.done = k.bytes[p++] != 0;
    const size_t n = k.bytes[p++];
    s.degree.resize(n);
    s.comp.resize(n);
    uint8_t max_comp = 0;
    for (size_t i = 0; i < n; ++i) {
        s.degree[i] = k.bytes[p++];
        s.comp[i] = k.bytes[p++];
        max_comp = std::max(max_comp, s.comp[i]);
    }
    ++p; // 0xff
    s.flags.assign(max_comp + 1, 0);
    for (uint8_t c = 1; c <= max_comp; ++c) s.flags[c] = k.bytes[p++];
    return s;
}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::stoi(argv[1]) : 4;
    const int w = n + 1;
    const int V = w * w;
    const int start = 0, target = V - 1;

    std::vector<std::pair<int,int>> edges;
    for (int r = 0; r < w; ++r) {
        for (int c = 0; c < w; ++c) {
            int u = r * w + c;
            if (c + 1 < w) edges.push_back({u, u + 1});
            if (r + 1 < w) edges.push_back({u, u + w});
        }
    }

    std::vector<int> first(V, 1e9), last(V, -1);
    for (int i = 0; i < (int)edges.size(); ++i) {
        auto [u,v] = edges[i];
        first[u] = std::min(first[u], i); first[v] = std::min(first[v], i);
        last[u] = std::max(last[u], i);  last[v] = std::max(last[v], i);
    }

    // active_before[e] = vertices whose state is needed immediately before edge e.
    std::vector<std::vector<int>> active_before(edges.size() + 1);
    for (int e = 0; e <= (int)edges.size(); ++e) {
        for (int v = 0; v < V; ++v) {
            if (first[v] < e && last[v] >= e) active_before[e].push_back(v);
        }
    }

    std::unordered_map<Key, Count, KeyHash> cur, nxt;
    State initial;
    initial.degree.assign(active_before[0].size(), 0);
    initial.comp.assign(active_before[0].size(), 0);
    initial.flags.assign(1, 0);
    cur[make_key(initial)] = 1;

    size_t peak = cur.size();
    size_t peak_bytes = 0;

    for (int ei = 0; ei < (int)edges.size(); ++ei) {
        const auto& before = active_before[ei];
        const auto& after  = active_before[ei + 1];
        auto [u,v] = edges[ei];

        // Work slots = before plus endpoints first appearing on this edge.
        std::vector<int> work = before;
        auto ensure = [&](int x) {
            if (std::find(work.begin(), work.end(), x) == work.end()) work.push_back(x);
        };
        ensure(u); ensure(v);

        std::vector<int> before_pos(V, -1), work_pos(V, -1), after_pos(V, -1);
        for (int i = 0; i < (int)before.size(); ++i) before_pos[before[i]] = i;
        for (int i = 0; i < (int)work.size(); ++i) work_pos[work[i]] = i;
        for (int i = 0; i < (int)after.size(); ++i) after_pos[after[i]] = i;

        nxt.clear();
        nxt.reserve(cur.size() * 2 + 16);

        for (const auto& [kk, ways] : cur) {
            State old = decode(kk);
            State base;
            base.done = old.done;
            base.degree.assign(work.size(), 0);
            base.comp.assign(work.size(), 0);
            base.flags = old.flags;
            for (int i = 0; i < (int)before.size(); ++i) {
                int j = work_pos[before[i]];
                base.degree[j] = old.degree[i];
                base.comp[j] = old.comp[i];
            }

            for (int take = 0; take < 2; ++take) {
                State s = base;
                bool ok = true;
                int iu = work_pos[u], iv = work_pos[v];

                if (take) {
                    if (s.done) continue;
                    const uint8_t max_u = (u == start || u == target) ? 1 : 2;
                    const uint8_t max_v = (v == start || v == target) ? 1 : 2;
                    if (s.degree[iu] >= max_u || s.degree[iv] >= max_v) continue;
                    ++s.degree[iu]; ++s.degree[iv];

                    uint8_t a = s.comp[iu], b = s.comp[iv];
                    if (!a && !b) {
                        uint8_t q = static_cast<uint8_t>(s.flags.size());
                        s.flags.push_back(0);
                        if (u == start || v == start) s.flags[q] |= 1;
                        if (u == target || v == target) s.flags[q] |= 2;
                        s.comp[iu] = s.comp[iv] = q;
                    } else if (!a || !b) {
                        uint8_t q = a ? a : b;
                        s.comp[iu] = s.comp[iv] = q;
                        if (u == start || v == start) s.flags[q] |= 1;
                        if (u == target || v == target) s.flags[q] |= 2;
                    } else {
                        if (a == b) continue; // cycle
                        uint8_t keep = std::min(a,b), kill = std::max(a,b);
                        s.flags[keep] |= s.flags[kill];
                        for (auto& c : s.comp) if (c == kill) c = keep;
                    }
                }

                // Vertices not in the next frontier are forgotten now.
                for (int x : work) {
                    if (after_pos[x] != -1) continue;
                    int ix = work_pos[x];
                    bool terminal = (x == start || x == target);
                    if ((terminal && s.degree[ix] != 1) ||
                        (!terminal && s.degree[ix] != 0 && s.degree[ix] != 2)) {
                        ok = false; break;
                    }
                    uint8_t q = s.comp[ix];
                    s.comp[ix] = 0;
                    if (q) {
                        bool alive = false;
                        for (uint8_t c : s.comp) if (c == q) { alive = true; break; }
                        if (!alive) {
                            if (s.flags[q] == 3) s.done = true;
                            else { ok = false; break; }
                        }
                    }
                }
                if (!ok) continue;

                State out;
                out.done = s.done;
                out.degree.resize(after.size());
                out.comp.resize(after.size());
                out.flags = s.flags;
                for (int i = 0; i < (int)after.size(); ++i) {
                    int j = work_pos[after[i]];
                    out.degree[i] = s.degree[j];
                    out.comp[i] = s.comp[j];
                }
                canonicalize(out);
                Key nk = make_key(out);
                peak_bytes = std::max(peak_bytes, nk.bytes.size());
                nxt[nk] += ways;
            }
        }

        cur.swap(nxt);
        peak = std::max(peak, cur.size());
        if ((ei + 1) % w == 0 || ei + 1 == (int)edges.size()) {
            std::cerr << "edge " << (ei + 1) << "/" << edges.size()
                      << " frontier=" << after.size()
                      << " states=" << cur.size() << "\n";
        }
    }

    Count ans = 0;
    for (const auto& [k, ways] : cur) if (decode(k).done) ans += ways;
    std::cout << "n=" << n << " paths=" << ans
              << " peak_states=" << peak
              << " max_key_bytes=" << peak_bytes << "\n";
}
