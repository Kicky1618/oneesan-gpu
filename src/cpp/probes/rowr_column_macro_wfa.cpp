#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using Count = std::uint64_t;
using MateID = oneesan::gridfp::MateID;

// At a column boundary there are r live physical vertices. Components may also
// occur in the virtual matching stack. status bit0 = component already contains
// one virtual edge, bit1 = its augmented cycle is closed.
struct State {
    std::vector<std::uint8_t> deg, comp, status, stack;
};
struct Key {
    std::vector<std::uint8_t> b;
    bool operator==(Key const& o) const { return b == o.b; }
    bool operator<(Key const& o) const { return b < o.b; }
};
struct KH {
    size_t operator()(Key const& k) const noexcept {
        std::uint64_t h=1469598103934665603ULL;
        for(auto x:k.b){ h^=x; h*=1099511628211ULL; }
        return (size_t)h;
    }
};
static bool stack_has(State const&s,std::uint8_t q){
    return std::find(s.stack.begin(),s.stack.end(),q)!=s.stack.end();
}
static void canon(State&s){
    std::array<std::uint8_t,96> rm{}, ns{}; std::uint8_t nx=1;
    auto take=[&](std::uint8_t c){ if(c&&!rm[c]){rm[c]=nx;ns[nx]=s.status[c];++nx;} };
    for(auto c:s.comp)take(c); for(auto c:s.stack)take(c);
    for(auto&c:s.comp)if(c)c=rm[c]; for(auto&c:s.stack)c=rm[c];
    s.status.assign(nx,0); for(std::uint8_t q=1;q<nx;++q)s.status[q]=ns[q];
}
static Key key(State s){
    canon(s); Key k; k.b.reserve(4+2*s.deg.size()+s.stack.size()+s.status.size());
    k.b.push_back((std::uint8_t)s.deg.size());
    for(size_t i=0;i<s.deg.size();++i){k.b.push_back(s.deg[i]);k.b.push_back(s.comp[i]);}
    k.b.push_back(0xfe); k.b.push_back((std::uint8_t)s.stack.size());
    for(auto q:s.stack)k.b.push_back(q); k.b.push_back(0xfd);
    for(size_t q=1;q<s.status.size();++q)k.b.push_back(s.status[q]);
    return k;
}
static State dec(Key const&k){
    State s; size_t p=0,n=k.b[p++]; s.deg.resize(n);s.comp.resize(n);std::uint8_t mx=0;
    for(size_t i=0;i<n;++i){s.deg[i]=k.b[p++];s.comp[i]=k.b[p++];mx=std::max(mx,s.comp[i]);}
    if(k.b[p++]!=0xfe)throw std::runtime_error("bad key");
    size_t sn=k.b[p++];s.stack.resize(sn);for(size_t i=0;i<sn;++i){s.stack[i]=k.b[p++];mx=std::max(mx,s.stack[i]);}
    if(k.b[p++]!=0xfd)throw std::runtime_error("bad key2");
    s.status.assign(mx+1,0);for(std::uint8_t q=1;q<=mx;++q)s.status[q]=k.b[p++];
    return s;
}
static bool closed_consistent(State const&s,std::uint8_t q){
    if(!(s.status[q]&2))return true; if(stack_has(s,q))return false;
    for(size_t i=0;i<s.comp.size();++i)if(s.comp[i]==q&&s.deg[i]!=2)return false;
    return true;
}
static bool merge_components(State&s,std::uint8_t a,std::uint8_t b,bool virt){
    if(!a||!b)return false; if((s.status[a]&2)||(s.status[b]&2))return false;
    if(a==b){
        if(virt){if(s.status[a]&1)return false;s.status[a]|=3;return true;}
        if(!(s.status[a]&1))return false;s.status[a]|=2;return true;
    }
    int vc=(s.status[a]&1)+(s.status[b]&1)+(virt?1:0);if(vc>1)return false;
    auto keep=std::min(a,b),kill=std::max(a,b);for(auto&c:s.comp)if(c==kill)c=keep;for(auto&c:s.stack)if(c==kill)c=keep;
    s.status[keep]=vc?1:0;s.status[kill]=0;return true;
}
static bool add_physical(State&s,int i,int j,int maxi=2,int maxj=2){
    if(s.deg[i]>=maxi||s.deg[j]>=maxj)return false;
    auto a=s.comp[i],b=s.comp[j]; if((a&&(s.status[a]&2))||(b&&(s.status[b]&2)))return false;
    ++s.deg[i];++s.deg[j];
    if(!a&&!b){std::uint8_t q=(std::uint8_t)s.status.size();s.status.push_back(0);s.comp[i]=s.comp[j]=q;return true;}
    if(!a||!b){auto q=a?a:b;s.comp[i]=s.comp[j]=q;return true;}
    return merge_components(s,a,b,false);
}

struct Out { Key k; int sym=-1; Count mult=0; };
struct OutKey { Key k; std::uint8_t sym=0; bool operator==(OutKey const&o)const{return sym==o.sym&&k==o.k;} };
struct OutKH { size_t operator()(OutKey const&o)const noexcept{return KH{}(o.k)^(0x9e3779b9u*o.sym);} };

class Macro {
    int r;
    using M=std::unordered_map<Key,Count,KH>;
    static void push(M& m,State const&s,Count c){m[key(s)]+=c;}
    void physical_step(M const&cur,M&nxt,int i,int j,int maxi=2,int maxj=2) const {
        nxt.clear();nxt.reserve(cur.size()*2+8);
        for(auto const&kv:cur){State s=dec(kv.first);push(nxt,s,kv.second);State t=s;if(add_physical(t,i,j,maxi,maxj))push(nxt,t,kv.second);}
    }
    std::vector<Out> finish_column(M const&cur,bool source,bool keep_new) const {
        std::unordered_map<OutKey,Count,OutKH> acc;acc.reserve(cur.size()*2+8);
        const int old0=0, bottom=keep_new?2*r:r;
        const int nold=r, new0=r;
        for(auto const&kv:cur){
            State base=dec(kv.first); bool ok=true;
            // All old column vertices are now fully resolved.
            for(int y=0;y<r;++y){int i=old0+y;bool isSource=source&&y==0;
                if((isSource&&base.deg[i]!=1)||(!isSource&&base.deg[i]!=0&&base.deg[i]!=2)){ok=false;break;}
            }
            if(!ok)continue;
            if(source){auto q=base.comp[0];if(!q)continue;base.stack.insert(base.stack.begin(),q);}
            for(int sym=0;sym<3;++sym){State s=base;auto q=s.comp[bottom];
                if(sym==0){if(s.deg[bottom]!=0)continue;}
                else {if(s.deg[bottom]!=1||!q)continue;
                    if(sym==2){if((int)s.stack.size()>=r)continue;s.stack.push_back(q);}
                    else {if(s.stack.empty())continue;auto q2=s.stack.back();s.stack.pop_back();if(!merge_components(s,q,q2,true))continue;}
                }
                // Forget old vertices and bottom, then ensure vanished components are complete.
                std::vector<std::uint8_t> oldComp;
                oldComp.reserve(r+1);for(int y=0;y<r;++y)oldComp.push_back(s.comp[y]);oldComp.push_back(s.comp[bottom]);
                std::vector<std::uint8_t> nd,nc;if(keep_new){nd.resize(r);nc.resize(r);for(int y=0;y<r;++y){nd[y]=s.deg[new0+y];nc[y]=s.comp[new0+y];}}
                s.deg.swap(nd);s.comp.swap(nc);
                for(auto qq:oldComp)if(qq){bool alive=stack_has(s,qq);if(!alive)for(auto cc:s.comp)if(cc==qq){alive=true;break;}if(!alive&&!(s.status[qq]&2)){ok=false;break;}}
                if(!ok){ok=true;continue;}
                for(std::uint8_t qq=1;qq<s.status.size();++qq)if((s.status[qq]&2)&&!closed_consistent(s,qq)){ok=false;break;}
                if(!ok){ok=true;continue;}
                if(!keep_new && (!s.stack.empty()||!s.comp.empty()))continue;
                OutKey z{key(s),(std::uint8_t)sym};acc[z]+=kv.second;
            }
        }
        std::vector<Out> out;out.reserve(acc.size());for(auto&kv:acc)out.push_back({std::move(kv.first.k),(int)kv.first.sym,kv.second});return out;
    }
public:
    explicit Macro(int rr):r(rr){}
    // First/interior column: input contains r old vertices. Process r outgoing
    // horizontals to r new vertices, then r verticals in the old column.
    std::vector<Out> step(Key const&in,bool source=false) const {
        State z=dec(in);if((int)z.deg.size()!=r)throw std::runtime_error("macro input width");
        z.deg.resize(2*r+1,0);z.comp.resize(2*r+1,0);
        M a,b;a[key(z)]=1;
        for(int y=0;y<r;++y){physical_step(a,b,y,r+y,(source&&y==0)?1:2,2);a.swap(b);}
        for(int y=0;y<r-1;++y){physical_step(a,b,y,y+1,(source&&y==0)?1:2,2);a.swap(b);}
        physical_step(a,b,r-1,2*r,2,2);a.swap(b);
        return finish_column(a,source,true);
    }
    // Last column: no outgoing horizontals; only verticals, then all vertices disappear.
    std::array<Count,3> finish(Key const&in) const {
        State z=dec(in);if((int)z.deg.size()!=r)throw std::runtime_error("finish input width");z.deg.resize(r+1,0);z.comp.resize(r+1,0);
        M a,b;a[key(z)]=1;for(int y=0;y<r-1;++y){physical_step(a,b,y,y+1);a.swap(b);}physical_step(a,b,r-1,r);a.swap(b);
        auto v=finish_column(a,false,false);std::array<Count,3> ans{};for(auto const&o:v){State s=dec(o.k);if(s.deg.empty()&&s.stack.empty())ans[o.sym]+=o.mult;}return ans;
    }
    Key empty_input() const {State z;z.deg.assign(r,0);z.comp.assign(r,0);z.status.assign(1,0);return key(z);}
};

static std::map<MateID,Count> gridfp(int rows,int W){using namespace oneesan::gridfp;std::unordered_map<MateID,Count>M,D,nM,nD;M[MateID(R)<<(2*(W-1))]=1;for(int row=0;row<rows;++row)for(int p=W-1;p>=1;--p){nM.clear();nD.clear();for(auto const&kv:M){auto m=kv.first,c=kv.second;nM[m]+=c;auto z=include_horizontal(m,W,p);if(z.valid){if(z.blocked)nD[z.mate]+=c;else nM[z.mate]+=c;}}for(auto const&kv:D)nM[blocked_exclude(kv.first,p)]+=kv.second;M.swap(nM);D.swap(nD);}return {M.begin(),M.end()};}
static std::map<MateID,Count> eval_wfa(int r,int W){Macro M(r);using V=std::unordered_map<Key,Count,KH>;std::array<V,3> first;for(auto const&o:M.step(M.empty_input(),true))first[o.sym][o.k]+=o.mult;
    // carry a word map only for differential testing.
    struct WK{Key k;MateID w;bool operator==(WK const&o)const{return w==o.w&&k==o.k;}};struct H{size_t operator()(WK const&x)const noexcept{return KH{}(x.k)^size_t(x.w*0x9e3779b97f4a7c15ULL);}};
    std::unordered_map<WK,Count,H> cur,nxt;for(int s=0;s<3;++s)for(auto const&kv:first[s])cur[{kv.first,MateID(s)<<(2*(W-1))}]+=kv.second;
    for(int c=1;c<W-1;++c){nxt.clear();for(auto const&kv:cur)for(auto const&o:M.step(kv.first.k,false)){MateID w=kv.first.w|MateID(o.sym)<<(2*(W-1-c));nxt[{o.k,w}]+=kv.second*o.mult;}cur.swap(nxt);}
    std::map<MateID,Count> out;for(auto const&kv:cur){auto b=M.finish(kv.first.k);for(int s=0;s<3;++s)if(b[s])out[kv.first.w|MateID(s)]+=kv.second*b[s];}return out;
}
static std::string show(MateID m,int W){std::string s;for(int p=W-1;p>=0;--p){auto v=oneesan::gridfp::mget(m,p);s+=v==oneesan::gridfp::N?'N':v==oneesan::gridfp::R?'R':'L';}return s;}

int main(int ac,char**av){int r=ac>1?std::atoi(av[1]):4;int W=ac>2?std::atoi(av[2]):8;int mode=ac>3?std::atoi(av[3]):0;Macro M(r);
    if(mode){auto a=eval_wfa(r,W),b=gridfp(r,W);std::map<MateID,Count> all=a;for(auto const&kv:b)all.try_emplace(kv.first,0);std::uint64_t bad=0;for(auto const&kv:all){auto x=a.count(kv.first)?a.at(kv.first):0,y=b.count(kv.first)?b.at(kv.first):0;if(x!=y){if(bad<8)std::cerr<<"bad "<<show(kv.first,W)<<" macro="<<x<<" fp="<<y<<"\n";++bad;}}std::cout<<"compare r="<<r<<" W="<<W<<" macro="<<a.size()<<" gridfp="<<b.size()<<" bad="<<bad<<"\n";if(bad)return 1;if(mode>=2)return 0;}
    std::unordered_map<Key,int,KH> id;std::deque<Key>q;auto first=M.step(M.empty_input(),true);for(auto const&o:first)if(!id.count(o.k)){int z=id.size();id[o.k]=z;q.push_back(o.k);}std::uint64_t edges=0,maxout=0;while(!q.empty()){Key s=std::move(q.front());q.pop_front();auto tr=M.step(s,false);edges+=tr.size();maxout=std::max<std::uint64_t>(maxout,tr.size());for(auto const&o:tr)if(!id.count(o.k)){int z=id.size();id[o.k]=z;q.push_back(o.k);}if((id.size()&1023)==0)std::cerr<<"states="<<id.size()<<" queue="<<q.size()<<" edges="<<edges<<"\n";}
    std::uint64_t finals=0;for(auto const&kv:id){auto b=M.finish(kv.first);for(auto x:b)if(x)++finals;}std::cout<<"macro r="<<r<<" states="<<id.size()<<" edges="<<edges<<" maxout="<<maxout<<" finals="<<finals<<" first="<<first.size()<<"\n";
}
