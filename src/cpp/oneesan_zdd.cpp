#include <rapidd/rapidd.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using rapidd::Zdd;
using rapidd::ZddManager;

struct Edge { int u, v; };

struct State {
    std::vector<uint8_t> deg;
    std::vector<int16_t> label;
    std::vector<uint8_t> flags;
    bool done = false;
};

struct Builder {
    int n = 0;
    int side = 0;
    int V = 0;
    int E = 0;
    int s = 0;
    int t = 0;
    std::vector<Edge> edges;
    std::vector<int> first_edge, last_edge;
    std::vector<std::vector<int>> forget_at;
    std::vector<std::vector<int>> active_at;
    ZddManager manager;
    std::vector<std::unordered_map<std::string, Zdd>> memo;
    uint64_t calls = 0, memo_hits = 0, pruned = 0;

    Builder(int n_, uint32_t max_nodes)
        : n(n_), side(n_ + 1), V(side * side), s(0), t(V - 1),
          manager(ZddManager::Config{.max_nodes=max_nodes, .unique_shards=256,
                                    .thread_safe=false,
                                    .apply_cache_slots=1u<<18,
                                    .unary_cache_slots=1u<<16}) {
        if (n < 1) throw std::invalid_argument("n must be >= 1");
        build_edges();
        E = (int)edges.size();
        manager.ensure_variables(E);
        first_edge.assign(V, E);
        last_edge.assign(V, -1);
        for (int i = 0; i < E; ++i) {
            for (int v : {edges[i].u, edges[i].v}) {
                first_edge[v] = std::min(first_edge[v], i);
                last_edge[v] = std::max(last_edge[v], i);
            }
        }
        forget_at.assign(E, {});
        for (int v = 0; v < V; ++v) {
            if (last_edge[v] >= 0) forget_at[last_edge[v]].push_back(v);
        }
        active_at.assign(E + 1, {});
        for (int i = 0; i <= E; ++i) {
            for (int v = 0; v < V; ++v) {
                if (first_edge[v] < i && i <= last_edge[v]) active_at[i].push_back(v);
            }
        }
        memo.resize(E + 1);
    }

    int id(int r, int c) const { return r * side + c; }

    void build_edges() {
        // Vertex-major order. At (r,c), process right then down. This keeps the
        // frontier O(n) while assigning one ZDD variable to every grid edge.
        for (int r = 0; r < side; ++r) {
            for (int c = 0; c < side; ++c) {
                if (c + 1 < side) edges.push_back({id(r,c), id(r,c+1)});
                if (r + 1 < side) edges.push_back({id(r,c), id(r+1,c)});
            }
        }
    }

    uint8_t terminal_flag(int v) const {
        return uint8_t((v == s ? 1 : 0) | (v == t ? 2 : 0));
    }

    bool degree_ok_final(int v, uint8_t d) const {
        if (v == s || v == t) return d == 1;
        return d == 0 || d == 2;
    }

    void canonicalize(State& st, int next_i) const {
        std::array<int16_t, 256> map{};
        map.fill(-1);
        std::array<uint8_t, 256> oldflags{};
        for (size_t i = 0; i < st.flags.size() && i < oldflags.size(); ++i) oldflags[i] = st.flags[i];
        std::vector<uint8_t> nf;
        int16_t next = 0;
        for (int v : active_at[next_i]) {
            int16_t x = st.label[v];
            if (x < 0) continue;
            if (x >= (int16_t)map.size()) throw std::runtime_error("too many component labels");
            if (map[x] < 0) {
                map[x] = next++;
                nf.push_back(oldflags[x]);
            }
            st.label[v] = map[x];
        }
        st.flags = std::move(nf);
    }

    std::string key(const State& st, int i) const {
        std::string k;
        k.reserve(2 * active_at[i].size() + st.flags.size() + 2);
        k.push_back(char(st.done ? 1 : 0));
        for (int v : active_at[i]) {
            k.push_back(char(st.deg[v]));
            k.push_back(char(st.label[v] + 1));
        }
        k.push_back(char(st.flags.size()));
        for (uint8_t f : st.flags) k.push_back(char(f));
        return k;
    }

    int new_label(State& st, uint8_t fl) const {
        int x = (int)st.flags.size();
        if (x >= 255) throw std::runtime_error("component-label overflow");
        st.flags.push_back(fl);
        return x;
    }

    bool include_edge(State& st, int u, int v) {
        if (st.done) return false;
        const uint8_t lim_u = (u == s || u == t) ? 1 : 2;
        const uint8_t lim_v = (v == s || v == t) ? 1 : 2;
        if (st.deg[u] >= lim_u || st.deg[v] >= lim_v) return false;

        int lu = st.label[u], lv = st.label[v];
        if (lu >= 0 && lv >= 0 && lu == lv) {
            // Adding this edge closes a cycle.
            return false;
        }

        if (lu < 0 && lv < 0) {
            int z = new_label(st, terminal_flag(u) | terminal_flag(v));
            st.label[u] = st.label[v] = z;
        } else if (lu < 0) {
            st.label[u] = lv;
            st.flags[lv] |= terminal_flag(u);
        } else if (lv < 0) {
            st.label[v] = lu;
            st.flags[lu] |= terminal_flag(v);
        } else {
            // Merge lv into lu.
            st.flags[lu] |= st.flags[lv];
            for (int x = 0; x < V; ++x) if (st.label[x] == lv) st.label[x] = lu;
            st.flags[lv] = 0;
        }
        ++st.deg[u];
        ++st.deg[v];
        return true;
    }

    bool finalize_after_edge(State& st, int i) {
        // Validate vertices whose last incident edge has just been processed.
        std::vector<int16_t> touched;
        for (int v : forget_at[i]) {
            if (!degree_ok_final(v, st.deg[v])) return false;
            if (st.label[v] >= 0) touched.push_back(st.label[v]);
        }

        for (int v : forget_at[i]) {
            st.deg[v] = 0;
            st.label[v] = -1;
        }

        std::sort(touched.begin(), touched.end());
        touched.erase(std::unique(touched.begin(), touched.end()), touched.end());
        for (int16_t lab : touched) {
            bool alive = false;
            for (int v : active_at[i + 1]) {
                if (st.label[v] == lab) { alive = true; break; }
            }
            if (alive) continue;
            uint8_t fl = (lab >= 0 && lab < (int)st.flags.size()) ? st.flags[lab] : 0;
            if (fl != 3) return false;
            // A completed s-t path cannot coexist with another selected component.
            for (int v : active_at[i + 1]) if (st.label[v] >= 0) return false;
            st.done = true;
        }

        canonicalize(st, i + 1);
        return true;
    }

    Zdd solve(int i, const State& in) {
        ++calls;
        if (i == E) return in.done ? manager.unit() : manager.empty();
        std::string k = key(in, i);
        auto& mm = memo[i];
        auto it = mm.find(k);
        if (it != mm.end()) { ++memo_hits; return it->second; }

        State lo_state = in;
        Zdd lo = manager.empty();
        if (finalize_after_edge(lo_state, i)) lo = solve(i + 1, lo_state);
        else ++pruned;

        State hi_state = in;
        Zdd hi = manager.empty();
        const Edge e = edges[i];
        if (include_edge(hi_state, e.u, e.v) && finalize_after_edge(hi_state, i)) {
            hi = solve(i + 1, hi_state);
        } else {
            ++pruned;
        }

        // RAPiDD requires child levels < parent levels. Reverse the natural
        // edge index so later recursion uses lower levels.
        uint32_t level = uint32_t(E - i);
        Zdd out = manager.make_node(level, lo, hi);
        mm.emplace(std::move(k), out);
        return out;
    }

    Zdd build() {
        State init;
        init.deg.assign(V, 0);
        init.label.assign(V, -1);
        return solve(0, init);
    }

    std::vector<std::array<uint32_t,4>> reachable_rows(Zdd root) const {
        std::unordered_set<uint32_t> seen;
        std::vector<Zdd> stack{root};
        std::vector<std::array<uint32_t,4>> rows;
        while (!stack.empty()) {
            Zdd z = stack.back(); stack.pop_back();
            if (z.raw() <= 1 || !seen.insert(z.raw()).second) continue;
            uint32_t lev = manager.top_level(z);
            auto [low, high] = manager.split(z);
            rows.push_back({z.raw(), lev, low.raw(), high.raw()});
            stack.push_back(low); stack.push_back(high);
        }
        return rows;
    }

    void write_text(const std::string& path, Zdd root) const {
        std::ofstream os(path, std::ios::binary);
        if (!os) throw std::runtime_error("cannot open output: " + path);
        os << "ONEESAN_ZDD_V1\n";
        os << "grid_n " << n << "\n";
        os << "vertices " << V << "\n";
        os << "variables " << E << "\n";
        os << "source " << s << "\n";
        os << "target " << t << "\n";
        os << "root " << root.raw() << "\n";
        os << "# level edge_index u v\n";
        for (int i = 0; i < E; ++i) {
            os << "var " << (E - i) << ' ' << i << ' ' << edges[i].u << ' ' << edges[i].v << "\n";
        }
        os << "# id level low high\n";
        auto rows = reachable_rows(root);
        std::sort(rows.begin(), rows.end(), [](auto const& a, auto const& b){ return a[0] < b[0]; });
        for (auto const& r : rows) os << "node " << r[0] << ' ' << r[1] << ' ' << r[2] << ' ' << r[3] << "\n";
        os << "end\n";
    }

    void write_sapporo(const std::string& path, Zdd root) const {
        // SAPPOROBDD bddexport/bddimportz compatible text format. External
        // node IDs are deliberately even because SAPPOROBDD reserves bit 0
        // for complemented-edge encoding.
        std::ofstream os(path, std::ios::binary);
        if (!os) throw std::runtime_error("cannot open SAPPOROBDD output: " + path);
        auto rows = reachable_rows(root);
        std::unordered_map<uint32_t,uint64_t> ext;
        ext.reserve(rows.size() * 2 + 1);
        uint64_t next = 2;
        // RAPiDD nodes are allocated after their children, but sort by level and
        // then raw ID to make the required child-before-parent order explicit.
        std::sort(rows.begin(), rows.end(), [](auto const& a, auto const& b){
            if (a[1] != b[1]) return a[1] < b[1];
            return a[0] < b[0];
        });
        for (auto const& r : rows) { ext.emplace(r[0], next); next += 2; }
        auto ref = [&](uint32_t raw) -> std::string {
            if (raw == 0) return "F";
            if (raw == 1) return "T";
            return std::to_string(ext.at(raw));
        };
        os << "_i " << E << "\n_o 1\n_n " << rows.size() << "\n";
        for (auto const& r : rows) {
            os << ext.at(r[0]) << ' ' << r[1] << ' ' << ref(r[2]) << ' ' << ref(r[3]) << "\n";
        }
        os << ref(root.raw()) << "\n";
    }
};

int main(int argc, char** argv) {
    try {
        if (argc < 3) {
            std::cerr << "usage: " << argv[0]
                      << " <n> <output.zdd> [max_nodes] [--sapporo <output.zbdd>]\n";
            return 2;
        }
        int n = std::stoi(argv[1]);
        std::string out = argv[2];
        uint32_t max_nodes = 32u * 1024u * 1024u;
        std::string sapporo;
        int argi = 3;
        if (argi < argc && argv[argi][0] != '-') max_nodes = (uint32_t)std::stoul(argv[argi++]);
        while (argi < argc) {
            std::string a = argv[argi++];
            if (a == "--sapporo" && argi < argc) sapporo = argv[argi++];
            else throw std::invalid_argument("unknown/incomplete argument: " + a);
        }
        Builder b(n, max_nodes);
        Zdd root = b.build();
        b.write_text(out, root);
        if (!sapporo.empty()) b.write_sapporo(sapporo, root);
        std::cout << "n=" << n
                  << " variables=" << b.E
                  << " cardinality_u64=" << root.cardinality()
                  << " reachable_nodes=" << root.node_count()
                  << " allocated_nodes=" << b.manager.allocated_nodes()
                  << " calls=" << b.calls
                  << " memo_hits=" << b.memo_hits
                  << " pruned=" << b.pruned
                  << " output=" << out;
        if (!sapporo.empty()) std::cout << " sapporo_output=" << sapporo;
        std::cout << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
