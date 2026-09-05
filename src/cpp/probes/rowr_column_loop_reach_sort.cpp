#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <vector>

// Reachability-only version of rowr_column_loop_transducer.cpp.  It keeps the
// same virtual-edge semantics but uses fixed storage and no coefficients.
// Intended to enumerate the width-independent raw column WFA for r <= 8.

static constexpr int MAXR = 8;
static constexpr int MAXV = 2 * MAXR + 1;
static constexpr int MAXC = MAXV + 2;

struct State {
    std::array<std::uint8_t, MAXV> deg{};
    std::array<std::uint8_t, MAXV> comp{};
    std::array<std::uint8_t, MAXC> status{}; // bit0 virtual edge, bit1 closed
    std::array<std::uint8_t, MAXR + 1> stack{};
    std::uint8_t n = 0;
    std::uint8_t ns = 1;
    std::uint8_t sp = 0;

    bool operator==(State const& o) const noexcept {
        if (n != o.n || ns != o.ns || sp != o.sp) return false;
        for (int i = 0; i < n; ++i)
            if (deg[i] != o.deg[i] || comp[i] != o.comp[i]) return false;
        for (int i = 0; i < sp; ++i) if (stack[i] != o.stack[i]) return false;
        for (int i = 1; i < ns; ++i) if (status[i] != o.status[i]) return false;
        return true;
    }
    bool operator<(State const& o) const noexcept {
        if (n != o.n) return n < o.n;
        if (ns != o.ns) return ns < o.ns;
        if (sp != o.sp) return sp < o.sp;
        for (int i = 0; i < n; ++i) {
            if (deg[i] != o.deg[i]) return deg[i] < o.deg[i];
            if (comp[i] != o.comp[i]) return comp[i] < o.comp[i];
        }
        for (int i = 0; i < sp; ++i) if (stack[i] != o.stack[i]) return stack[i] < o.stack[i];
        for (int i = 1; i < ns; ++i) if (status[i] != o.status[i]) return status[i] < o.status[i];
        return false;
    }
};

static bool stack_has(State const& s, std::uint8_t q) {
    for (int i = 0; i < s.sp; ++i) if (s.stack[i] == q) return true;
    return false;
}

static void canon(State& s) {
    std::array<std::uint8_t, MAXC> rm{}, st{};
    std::uint8_t nx = 1;
    auto take = [&](std::uint8_t c) {
        if (c && !rm[c]) { rm[c] = nx; st[nx] = s.status[c]; ++nx; }
    };
    for (int i = 0; i < s.n; ++i) take(s.comp[i]);
    for (int i = 0; i < s.sp; ++i) take(s.stack[i]);
    for (int i = 0; i < s.n; ++i) if (s.comp[i]) s.comp[i] = rm[s.comp[i]];
    for (int i = 0; i < s.sp; ++i) if (s.stack[i]) s.stack[i] = rm[s.stack[i]];
    s.status.fill(0);
    for (int q = 1; q < nx; ++q) s.status[q] = st[q];
    s.ns = nx;
    for (int i = s.n; i < MAXV; ++i) { s.deg[i] = 0; s.comp[i] = 0; }
    for (int i = s.sp; i <= MAXR; ++i) s.stack[i] = 0;
}

static bool closed_consistent(State const& s, std::uint8_t q) {
    if (!(s.status[q] & 2)) return true;
    if (stack_has(s, q)) return false;
    for (int i = 0; i < s.n; ++i)
        if (s.comp[i] == q && s.deg[i] != 2) return false;
    return true;
}

static bool merge_components(State& s, std::uint8_t a, std::uint8_t b, bool virt) {
    if (!a || !b) return false;
    if ((s.status[a] & 2) || (s.status[b] & 2)) return false;
    if (a == b) {
        if (virt) {
            if (s.status[a] & 1) return false;
            s.status[a] |= 3;
            return true;
        }
        if (!(s.status[a] & 1)) return false;
        s.status[a] |= 2;
        return true;
    }
    int vc = (s.status[a] & 1 ? 1 : 0) + (s.status[b] & 1 ? 1 : 0) + (virt ? 1 : 0);
    if (vc > 1) return false;
    std::uint8_t keep = std::min(a,b), kill = std::max(a,b);
    for (int i = 0; i < s.n; ++i) if (s.comp[i] == kill) s.comp[i] = keep;
    for (int i = 0; i < s.sp; ++i) if (s.stack[i] == kill) s.stack[i] = keep;
    s.status[keep] = vc ? 1 : 0;
    s.status[kill] = 0;
    return true;
}

static bool add_physical(State& s, int i, int j, int maxi, int maxj) {
    if (s.deg[i] >= maxi || s.deg[j] >= maxj) return false;
    std::uint8_t a = s.comp[i], b = s.comp[j];
    if ((a && (s.status[a] & 2)) || (b && (s.status[b] & 2))) return false;
    ++s.deg[i]; ++s.deg[j];
    if (!a && !b) {
        if (s.ns >= MAXC) return false;
        std::uint8_t q = s.ns++;
        s.status[q] = 0;
        s.comp[i] = s.comp[j] = q;
        return true;
    }
    if (!a || !b) {
        std::uint8_t q = a ? a : b;
        s.comp[i] = s.comp[j] = q;
        return true;
    }
    return merge_components(s, a, b, false);
}

using StateVec = std::vector<State>;

static inline void insert_state(StateVec& s, State x) {
    canon(x);
    s.push_back(std::move(x));
}
static inline void normalize(StateVec& s) {
    std::sort(s.begin(), s.end());
    s.erase(std::unique(s.begin(), s.end()), s.end());
}

struct Engine {
    int r, W, V, E, start = 0;
    std::vector<std::pair<int,int>> edges;
    std::vector<int> first, last;
    std::vector<std::vector<int>> active;

    Engine(int rr, int ww) : r(rr), W(ww), V((r+1)*W) {
        for (int c = 0; c < W; ++c) {
            if (c+1 < W) for (int y = 0; y < r; ++y)
                edges.push_back({y*W+c, y*W+c+1});
            for (int y = 0; y < r; ++y)
                edges.push_back({y*W+c, (y+1)*W+c});
        }
        E = (int)edges.size();
        first.assign(V, 1e9); last.assign(V, -1);
        for (int e = 0; e < E; ++e) {
            auto [u,v] = edges[e];
            first[u] = std::min(first[u], e); first[v] = std::min(first[v], e);
            last[u] = std::max(last[u], e); last[v] = std::max(last[v], e);
        }
        active.resize(E+1);
        for (int e = 0; e <= E; ++e)
            for (int v = 0; v < V; ++v)
                if (first[v] < e && last[v] >= e) active[e].push_back(v);
    }

    std::vector<size_t> run() {
        StateVec cur, nxt;
        State z; z.ns = 1; insert_state(cur, z); normalize(cur);
        std::vector<size_t> perColumn;
        for (int ei = 0; ei < E; ++ei) {
            auto const& before = active[ei];
            auto const& after = active[ei+1];
            auto [u,v] = edges[ei];
            std::vector<int> work = before;
            auto ensure = [&](int x){ if (std::find(work.begin(),work.end(),x)==work.end()) work.push_back(x); };
            ensure(u); ensure(v);
            std::vector<int> wp(V,-1), ap(V,-1);
            for (int i=0;i<(int)work.size();++i) wp[work[i]]=i;
            for (int i=0;i<(int)after.size();++i) ap[after[i]]=i;

            nxt.clear();
            nxt.reserve(cur.size()*2 + 32);
            bool emitted = false;
            for (State const& old : cur) {
                State base{}; base.n = (std::uint8_t)work.size(); base.ns=old.ns; base.sp=old.sp;
                base.status = old.status; base.stack = old.stack;
                for (int i=0;i<(int)before.size();++i) {
                    int j=wp[before[i]]; base.deg[j]=old.deg[i]; base.comp[j]=old.comp[i];
                }
                for (int take=0; take<2; ++take) {
                    State s=base; int iu=wp[u], iv=wp[v];
                    if (take) {
                        int mu=u==start?1:2, mv=v==start?1:2;
                        if (!add_physical(s,iu,iv,mu,mv)) continue;
                    }
                    std::array<int,MAXV> forgotten{}; int nf=0, bottomX=-1;
                    for(int x:work) if(ap[x]==-1) {
                        forgotten[nf++]=x;
                        if(x/W==r) bottomX=x;
                    }
                    int symLo=bottomX>=0?0:-1, symHi=bottomX>=0?2:-1;
                    for(int sym=symLo;sym<=symHi;++sym) {
                        State t=s; bool ok=true;
                        for(int fi=0;fi<nf;++fi) {
                            int x=forgotten[fi], ix=wp[x], y=x/W;
                            std::uint8_t q=t.comp[ix];
                            if(y==r) {
                                emitted=true;
                                if(sym==0) {
                                    if(t.deg[ix]!=0){ok=false;break;}
                                } else {
                                    if(t.deg[ix]!=1||!q){ok=false;break;}
                                    if(sym==2) {
                                        if(t.sp>=r){ok=false;break;}
                                        t.stack[t.sp++]=q;
                                    } else {
                                        if(!t.sp){ok=false;break;}
                                        std::uint8_t q2=t.stack[--t.sp];
                                        if(!merge_components(t,q,q2,true)){ok=false;break;}
                                        q=t.comp[ix];
                                    }
                                }
                            } else {
                                bool source=x==start;
                                if((source&&t.deg[ix]!=1)||(!source&&t.deg[ix]!=0&&t.deg[ix]!=2)){ok=false;break;}
                                if(source) {
                                    if(!q || t.sp>=r){ok=false;break;}
                                    for(int j=t.sp;j>0;--j)t.stack[j]=t.stack[j-1];
                                    t.stack[0]=q; ++t.sp;
                                }
                            }
                            t.comp[ix]=0;
                            if(q) {
                                bool alive=stack_has(t,q);
                                if(!alive) for(int j=0;j<t.n;++j) if(t.comp[j]==q){alive=true;break;}
                                if(!alive && !(t.status[q]&2)){ok=false;break;}
                            }
                        }
                        if(!ok) continue;
                        for(int q=1;q<t.ns;++q) if((t.status[q]&2)&&!closed_consistent(t,q)){ok=false;break;}
                        if(!ok) continue;
                        State out{}; out.n=(std::uint8_t)after.size(); out.ns=t.ns; out.sp=t.sp;
                        out.status=t.status; out.stack=t.stack;
                        for(int i=0;i<(int)after.size();++i){int j=wp[after[i]];out.deg[i]=t.deg[j];out.comp[i]=t.comp[j];}
                        insert_state(nxt,out);
                    }
                }
            }
            normalize(nxt);
            cur.swap(nxt);
            if(emitted) {
                perColumn.push_back(cur.size());
                std::cerr << "r="<<r<<" col="<<perColumn.size()<<" states="<<cur.size()<<"\n";
            }
        }
        std::cout << "sort-loop r="<<r<<" W="<<W<<" per_column=";
        for(size_t i=0;i<perColumn.size();++i)std::cout<<(i?",":"")<<perColumn[i];
        std::cout<<" final="<<cur.size()<<"\n";
        return perColumn;
    }
};

int main(int argc,char**argv){
    int r=argc>1?std::atoi(argv[1]):8;
    int W=argc>2?std::atoi(argv[2]):20;
    if(r<1||r>MAXR||W<2){std::cerr<<"usage: r=1..8 W>=2\n";return 2;}
    Engine(r,W).run();
}
