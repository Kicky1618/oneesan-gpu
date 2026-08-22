#include <rapidd/rapidd.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <tuple>
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
    std::string order_name;
    int max_frontier = 0;
    uint64_t sum_frontier = 0;

    Builder(int n_, uint32_t max_nodes, std::string order_)
        : n(n_), side(n_ + 1), V(side * side), s(0), t(V - 1), order_name(std::move(order_)),
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
        for (auto const& a : active_at) { max_frontier = std::max(max_frontier, (int)a.size()); sum_frontier += a.size(); }
        memo.resize(E + 1);
    }

    int id(int r, int c) const { return r * side + c; }

    void build_edges() {
        if(order_name=="wave2"||order_name.rfind("wave2m",0)==0){
            int N=side-1;if(N<4)throw std::invalid_argument("wave2 n>=4");
            auto H=[&](int r,int c){edges.push_back({id(r,c),id(r,c+1)});};
            auto Vv=[&](int r,int c){edges.push_back({id(r,c),id(r+1,c)});};
            H(0,N-1);Vv(0,N);
            for(int c=N-2;c>=1;--c){H(0,c);Vv(0,c+1);H(1,c+1);Vv(1,c+2);}
            H(0,0);Vv(0,1);Vv(0,0);H(1,1);Vv(1,2);H(1,0);Vv(1,1);Vv(1,0);
            int mid=(N-1)/2;if(order_name.rfind("wave2m",0)==0)mid=std::stoi(order_name.substr(6));if(mid<0||mid>N)throw std::invalid_argument("wave2 mid");
            for(int r=2;r<=N-3;++r){for(int c=N-1;c>=0;--c)H(r,c);for(int c=mid;c<=N;++c)Vv(r,c);for(int c=mid-1;c>=0;--c)Vv(r,c);}
            int A=N-2,B=N-1,C=N;
            for(int c=N-1;c>=0;--c)H(A,c);
            Vv(A,N-1);Vv(A,N);H(B,N-1);Vv(A,N-2);H(B,N-2);Vv(B,N);H(C,N-1);Vv(B,N-1);
            for(int c=N-3;c>=0;--c){Vv(A,c);H(B,c);H(C,c+1);Vv(B,c+1);}
            H(C,0);Vv(B,0);
            if((int)edges.size()!=2*side*(side-1))throw std::runtime_error("wave2 edge count");return;
        }
        if(order_name=="custom"){
            std::vector<Edge> canon;canon.reserve(2*side*(side-1));
            for(int r=0;r<side;++r)for(int c=0;c<side;++c){if(c+1<side)canon.push_back({id(r,c),id(r,c+1)});if(r+1<side)canon.push_back({id(r,c),id(r+1,c)});}
            const char* fn=std::getenv("ZDD_ORDER_FILE");if(!fn)throw std::invalid_argument("ZDD_ORDER_FILE not set");std::ifstream f(fn);int x;std::vector<char>seen(canon.size());while(f>>x){if(x<0||x>=(int)canon.size()||seen[x])throw std::runtime_error("bad custom edge id");seen[x]=1;edges.push_back(canon[x]);}if(edges.size()!=canon.size())throw std::runtime_error("custom order size mismatch");return;
        }
        bool rev=false;
        if(order_name.size()>4 && order_name.substr(order_name.size()-4)=="-rev"){rev=true;order_name.resize(order_name.size()-4);}
        auto finish=[&](){if(rev)std::reverse(edges.begin(),edges.end());};
        if(order_name=="row-out-hv" || order_name=="row-out-vh") {
            bool vh=order_name=="row-out-vh";
            for(int r=0;r<side;++r)for(int c=0;c<side;++c){
                if(vh){if(r+1<side)edges.push_back({id(r,c),id(r+1,c)});if(c+1<side)edges.push_back({id(r,c),id(r,c+1)});}
                else {if(c+1<side)edges.push_back({id(r,c),id(r,c+1)});if(r+1<side)edges.push_back({id(r,c),id(r+1,c)});}
            }finish();return;
        }
        if(order_name=="row-in-vh" || order_name=="row-in-hv") {
            bool vh=order_name=="row-in-vh";
            for(int r=0;r<side;++r){
                if(r==0){for(int c=0;c+1<side;++c)edges.push_back({id(r,c),id(r,c+1)});continue;}
                if(vh){
                    for(int c=0;c+1<side;++c){edges.push_back({id(r-1,c),id(r,c)});edges.push_back({id(r,c),id(r,c+1)});}
                    edges.push_back({id(r-1,side-1),id(r,side-1)});
                }else{
                    for(int c=0;c+1<side;++c){edges.push_back({id(r,c),id(r,c+1)});edges.push_back({id(r-1,c),id(r,c)});}
                    edges.push_back({id(r-1,side-1),id(r,side-1)});
                }
            }finish();return;
        }
        if(order_name=="row-out-HVband" || order_name=="row-out-VHband") {
            bool vh=order_name=="row-out-VHband";
            for(int r=0;r<side;++r){
                if(vh){if(r+1<side)for(int c=0;c<side;++c)edges.push_back({id(r,c),id(r+1,c)});for(int c=0;c+1<side;++c)edges.push_back({id(r,c),id(r,c+1)});}
                else {for(int c=0;c+1<side;++c)edges.push_back({id(r,c),id(r,c+1)});if(r+1<side)for(int c=0;c<side;++c)edges.push_back({id(r,c),id(r+1,c)});}
            }finish();return;
        }
        if(order_name=="row-in-VHband" || order_name=="row-in-HVband") {
            bool vh=order_name=="row-in-VHband";
            for(int r=0;r<side;++r){
                if(vh && r>0)for(int c=0;c<side;++c)edges.push_back({id(r-1,c),id(r,c)});
                for(int c=0;c+1<side;++c)edges.push_back({id(r,c),id(r,c+1)});
                if(!vh && r>0)for(int c=0;c<side;++c)edges.push_back({id(r-1,c),id(r,c)});
            }finish();return;
        }
        struct VE { Edge e; int orient; };
        std::vector<VE> all; all.reserve(2*side*(side-1));
        for(int r=0;r<side;++r) for(int c=0;c<side;++c){
            if(c+1<side) all.push_back({{id(r,c),id(r,c+1)},0}); // H
            if(r+1<side) all.push_back({{id(r,c),id(r+1,c)},1}); // V
        }
        auto rc=[&](int v){return std::pair<int,int>{v/side,v%side};};
        std::vector<int> vrank(V);
        std::vector<int> verts(V); for(int v=0;v<V;++v)verts[v]=v;
        bool tie_v_first=false;
        bool cfg_mode=false; unsigned long long cfg_dir=0,cfg_tie=0;
        bool interval_mode=false; int interval_tie=0; bool interval_slot_early=false;
        if(order_name.rfind("interval-",0)==0){
            interval_mode=true;
            bool bu=order_name.find("BU")!=std::string::npos; bool rl=order_name.find("RL")!=std::string::npos;
            if(order_name.find("deadline")!=std::string::npos) interval_tie=1;
            if(order_name.find("edgelate")!=std::string::npos) interval_tie=2;
            if(order_name.find("slotearly")!=std::string::npos) interval_slot_early=true;
            std::sort(verts.begin(),verts.end(),[&](int a,int b){auto [ar,ac]=rc(a);auto [br,bc]=rc(b);int ra=bu?-ar:ar,rb=bu?-br:br;if(ra!=rb)return ra<rb;int ca=rl?-ac:ac,cb=rl?-bc:bc;return ca<cb;});
        } else if(order_name.rfind("corner-",0)==0){
            // corner-TD-LR, corner-TD-RL, corner-BU-LR, corner-BU-RL
            bool bu=order_name.find("BU")!=std::string::npos; bool rl=order_name.find("RL")!=std::string::npos;
            std::sort(verts.begin(),verts.end(),[&](int a,int b){auto [ar,ac]=rc(a);auto [br,bc]=rc(b);int ra=bu?-ar:ar,rb=bu?-br:br;if(ra!=rb)return ra<rb;int ca=rl?-ac:ac,cb=rl?-bc:bc;return ca<cb;});
        } else if(order_name.rfind("cfg-",0)==0){
            cfg_mode=true; auto q=order_name.substr(4); auto z=q.find('-'); if(z==std::string::npos)throw std::invalid_argument("bad cfg");
            cfg_dir=std::stoull(q.substr(0,z),nullptr,16);cfg_tie=std::stoull(q.substr(z+1),nullptr,16);
            std::sort(verts.begin(),verts.end(),[&](int a,int b){auto [ar,ac]=rc(a);auto [br,bc]=rc(b);if(ar!=br)return ar<br;bool ra=(cfg_dir>>ar)&1,rb=(cfg_dir>>br)&1;int aa=ra?-ac:ac,bb=rb?-bc:bc;return aa<bb;});
        } else if(order_name=="row-rd"||order_name=="row-dr"){
            tie_v_first=(order_name=="row-dr");
            std::sort(verts.begin(),verts.end(),[&](int a,int b){auto [ar,ac]=rc(a);auto [br,bc]=rc(b);return std::tie(ar,ac)<std::tie(br,bc);});
        } else if(order_name=="row-snake" || order_name.rfind("rows-",0)==0){
            unsigned long long mask=0; bool custom=order_name.rfind("rows-",0)==0;
            if(custom) mask=std::stoull(order_name.substr(5),nullptr,16);
            std::sort(verts.begin(),verts.end(),[&](int a,int b){auto [ar,ac]=rc(a);auto [br,bc]=rc(b);if(ar!=br)return ar<br;bool ra=custom?((mask>>ar)&1):(ar&1);bool rb=custom?((mask>>br)&1):(br&1);int aa=ra?-ac:ac,bb=rb?-bc:bc;return aa<bb;});
        } else if(order_name=="col-dr"||order_name=="col-rd"){
            tie_v_first=(order_name=="col-rd");
            std::sort(verts.begin(),verts.end(),[&](int a,int b){auto [ar,ac]=rc(a);auto [br,bc]=rc(b);return std::tie(ac,ar)<std::tie(bc,br);});
        } else if(order_name=="col-snake"){
            std::sort(verts.begin(),verts.end(),[&](int a,int b){auto [ar,ac]=rc(a);auto [br,bc]=rc(b);if(ac!=bc)return ac<bc;int aa=(ac&1)?-ar:ar,bb=(bc&1)?-br:br;return aa<bb;});
        } else if(order_name.rfind("diag",0)==0){
            bool rev=order_name=="diag-rev";
            std::sort(verts.begin(),verts.end(),[&](int a,int b){auto [ar,ac]=rc(a);auto [br,bc]=rc(b);int sa=ar+ac,sb=br+bc;if(sa!=sb)return sa<sb;return rev?ac>bc:ac<bc;});
        } else if(order_name.rfind("proj-",0)==0){
            // proj-A-B orders vertices by A*r+B*c. Positive integer weights.
            int A=1,B=1; sscanf(order_name.c_str(),"proj-%d-%d",&A,&B);
            std::sort(verts.begin(),verts.end(),[&](int a,int b){auto [ar,ac]=rc(a);auto [br,bc]=rc(b);long long ka=1LL*A*ar+1LL*B*ac,kb=1LL*A*br+1LL*B*bc;if(ka!=kb)return ka<kb;return std::tie(ar,ac)<std::tie(br,bc);});
        } else throw std::invalid_argument("unknown order: "+order_name);
        for(int i=0;i<V;++i) vrank[verts[i]]=i;
        if(interval_mode){
            std::vector<int> last(V);for(int v=0;v<V;++v)last[v]=vrank[v];
            for(auto const&x:all){last[x.e.u]=std::max(last[x.e.u],vrank[x.e.v]);last[x.e.v]=std::max(last[x.e.v],vrank[x.e.u]);}
            std::vector<int> bag(V);
            for(int k=0;k<V;++k){int z=0;for(int v=0;v<V;++v)if(vrank[v]<=k&&k<=last[v])++z;bag[k]=z;}
            struct IE{VE x;int slot,deadline;};std::vector<IE> q;q.reserve(all.size());
            for(auto const&x:all){int a=vrank[x.e.u],b=vrank[x.e.v];int lo=std::max(a,b),hi=std::min(last[x.e.u],last[x.e.v]);if(hi<lo)hi=lo;int slot=lo;for(int k=lo+1;k<=hi;++k)if(bag[k]<bag[slot]||(bag[k]==bag[slot]&&!interval_slot_early&&k>slot))slot=k;q.push_back({x,slot,hi});}
            std::sort(q.begin(),q.end(),[&](IE const&a,IE const&b){if(a.slot!=b.slot)return a.slot<b.slot;if(interval_tie==1&&a.deadline!=b.deadline)return a.deadline<b.deadline;if(interval_tie==2&&a.deadline!=b.deadline)return a.deadline>b.deadline;if(a.x.orient!=b.x.orient)return a.x.orient<b.x.orient;int aa=std::max(vrank[a.x.e.u],vrank[a.x.e.v]),bb=std::max(vrank[b.x.e.u],vrank[b.x.e.v]);return aa<bb;});
            for(auto const&x:q)edges.push_back(x.x.e);finish();return;
        }
        std::sort(all.begin(),all.end(),[&](VE const&a,VE const&b){
            int ae=std::min(vrank[a.e.u],vrank[a.e.v]),be=std::min(vrank[b.e.u],vrank[b.e.v]);
            if(ae!=be)return ae<be;
            int al=std::max(vrank[a.e.u],vrank[a.e.v]),bl=std::max(vrank[b.e.u],vrank[b.e.v]);
            auto orient_key=[&](VE const&x,int early){
                bool vf=tie_v_first;
                if(cfg_mode){int ev=(vrank[x.e.u]==early)?x.e.u:x.e.v;int er=ev/side;vf=(cfg_tie>>er)&1;}
                return vf?(1-x.orient):x.orient;
            };
            if(a.orient!=b.orient){int ao=orient_key(a,ae),bo=orient_key(b,be);if(ao!=bo)return ao<bo;}
            if(al!=bl)return al<bl;
            return a.orient<b.orient;
        });
        for(auto const&x:all)edges.push_back(x.e);
        finish();
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
        auto pk=[](int a,int b){if(a>b)std::swap(a,b);return (std::uint64_t(std::uint32_t(a))<<32)|std::uint32_t(b);};
        std::unordered_map<std::uint64_t,int> ce; int cid=0;
        for(int r=0;r<side;++r)for(int c=0;c<side;++c){if(c+1<side)ce[pk(id(r,c),id(r,c+1))]=cid++;if(r+1<side)ce[pk(id(r,c),id(r+1,c))]=cid++;}
        for (int i = 0; i < E; ++i) {
            os << "var " << (E - i) << ' ' << ce.at(pk(edges[i].u,edges[i].v)) << ' ' << edges[i].u << ' ' << edges[i].v << "\n";
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
        if(argc<3){std::cerr<<"usage: "<<argv[0]<<" <n> <order> [max_nodes]\n";return 2;}
        int n=std::stoi(argv[1]); std::string order=argv[2];
        uint32_t cap=argc>=4?(uint32_t)std::stoul(argv[3]):64u*1024u*1024u;
        auto t0=std::chrono::steady_clock::now();
        Builder b(n,cap,order);
        if(const char* fn=std::getenv("ZDD_DUMP_ORDER")){std::ofstream f(fn);std::unordered_map<unsigned long long,int> ids;int q=0;auto pk=[](int a,int b){if(a>b)std::swap(a,b);return (unsigned long long)(unsigned)a<<32|(unsigned)b;};for(int r=0;r<b.side;++r)for(int c=0;c<b.side;++c){if(c+1<b.side)ids[pk(b.id(r,c),b.id(r,c+1))]=q++;if(r+1<b.side)ids[pk(b.id(r,c),b.id(r+1,c))]=q++;}for(auto e:b.edges)f<<ids.at(pk(e.u,e.v))<<' ';f<<'\n';}
        auto root=b.build();
        if(const char* outp=std::getenv("ZDD_ORDER_OUT")) b.write_text(outp,root);
        double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        std::cout<<"n="<<n<<" order="<<order<<" nodes="<<root.node_count()
                 <<" allocated="<<b.manager.allocated_nodes()<<" calls="<<b.calls<<" hits="<<b.memo_hits
                 <<" max_frontier="<<b.max_frontier<<" avg_frontier="<<double(b.sum_frontier)/(b.E+1)
                 <<" cardinality_u64="<<root.cardinality()<<" sec="<<sec;
        if(std::getenv("ZDD_PROJECT_HORIZONTAL")||std::getenv("ZDD_PROJECT_COTREE")){
            const bool cotree=std::getenv("ZDD_PROJECT_COTREE")!=nullptr;
            const int H=cotree?n*n:b.side*(b.side-1);
            rapidd::ZddManager pm(rapidd::ZddManager::Config{.max_nodes=cap,.unique_shards=256,.thread_safe=false,
                .apply_cache_slots=1u<<20,.unary_cache_slots=1u<<18});
            pm.ensure_variables(H);
            std::vector<int> newlev(b.E+1,0);int hc=0;
            for(int i=0;i<b.E;++i){auto e=b.edges[i];bool keep=e.u/b.side==e.v/b.side; if(cotree&&keep) keep=(e.u/b.side)!=0; if(keep)newlev[b.E-i]=H-hc++;}
            if(hc!=H)throw std::runtime_error("projection variable count mismatch");
            std::unordered_map<uint32_t,rapidd::Zdd> mm;mm.reserve(root.node_count()*2+1);
            mm.emplace(0u,pm.empty());mm.emplace(1u,pm.unit());
            std::function<rapidd::Zdd(rapidd::Zdd)> proj=[&](rapidd::Zdd z)->rapidd::Zdd{
                if(auto it=mm.find(z.raw());it!=mm.end())return it->second;
                auto lev=b.manager.top_level(z);auto [lo0,hi0]=b.manager.split(z);
                auto lo=proj(lo0),hi=proj(hi0);rapidd::Zdd q;
                if(newlev[lev])q=pm.make_node((uint32_t)newlev[lev],lo,hi);
                else q=pm.apply(rapidd::ZddManager::Operation::Union,lo,hi);
                mm.emplace(z.raw(),q);return q;
            };
            auto tp=std::chrono::steady_clock::now();auto pr=proj(root);
            double psec=std::chrono::duration<double>(std::chrono::steady_clock::now()-tp).count();
            std::cout<<" projected_nodes="<<pr.node_count()<<" projected_allocated="<<pm.allocated_nodes()
                     <<" projected_cardinality_u64="<<pr.cardinality()<<" project_sec="<<psec;
            if(const char* outp=std::getenv("ZDD_PROJECT_OUT")){
                auto pk=[](int a,int b){if(a>b)std::swap(a,b);return (unsigned long long)(unsigned)a<<32|(unsigned)b;};
                std::unordered_map<unsigned long long,int> canon;int ce=0;
                for(int r=0;r<b.side;++r)for(int c=0;c<b.side;++c){if(c+1<b.side)canon[pk(b.id(r,c),b.id(r,c+1))]=ce++;if(r+1<b.side)canon[pk(b.id(r,c),b.id(r+1,c))]=ce++;}
                std::vector<int> edge_by_new(H+1,-1);
                for(int i=0;i<b.E;++i)if(newlev[b.E-i])edge_by_new[newlev[b.E-i]]=canon.at(pk(b.edges[i].u,b.edges[i].v));
                std::ofstream os(outp);if(!os)throw std::runtime_error("cannot write projected ZDD");int V=b.side*b.side;
                os<<"ONEESAN_ZDD_V1\n"<<"grid_n "<<n<<"\nvertices "<<V<<"\nvariables "<<H<<"\nsource 0\ntarget "<<(V-1)<<"\nroot "<<pr.raw()<<"\nprojection_horizontal 1\n";
                std::vector<std::pair<int,int>> endpoints(ce);for(auto const& kv:canon){int u=int(kv.first>>32),v=int((unsigned)kv.first);endpoints[kv.second]={u,v};}
                os<<"# level edge_index u v\n";for(int lev=1;lev<=H;++lev){int eid=edge_by_new[lev];auto [u,v]=endpoints.at(eid);os<<"var "<<lev<<' '<<eid<<' '<<u<<' '<<v<<"\n";}
                std::unordered_set<uint32_t> seen;std::vector<rapidd::Zdd> st{pr};std::vector<std::array<uint32_t,4>> rr;
                while(!st.empty()){auto q=st.back();st.pop_back();if(q.raw()<=1||!seen.insert(q.raw()).second)continue;auto [lo,hi]=pm.split(q);rr.push_back({q.raw(),pm.top_level(q),lo.raw(),hi.raw()});st.push_back(lo);st.push_back(hi);}
                std::sort(rr.begin(),rr.end(),[](auto const&a,auto const&b){return a[0]<b[0];});os<<"# id level low high\n";for(auto const&r:rr)os<<"node "<<r[0]<<' '<<r[1]<<' '<<r[2]<<' '<<r[3]<<"\n";os<<"end\n";
            }
        }
        std::cout<<"\n";
    }catch(std::exception const&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
