#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
using Count=std::uint64_t;using MateID=oneesan::gridfp::MateID;

// status bit0: component contains one virtual matching edge.
// status bit1: its augmented cycle is already closed; then no new edge may touch it.
struct State{std::vector<uint8_t>deg,comp,status,stack;};
struct Key{std::vector<uint8_t>b;bool operator==(Key const&o)const{return b==o.b;}};
struct KH{size_t operator()(Key const&k)const noexcept{uint64_t h=1469598103934665603ULL;for(auto x:k.b){h^=x;h*=1099511628211ULL;}return h;}};
struct WK{Key k;MateID w=0;bool operator==(WK const&o)const{return w==o.w&&k==o.k;}};
struct WKH{size_t operator()(WK const&x)const noexcept{return KH{}(x.k)^size_t(x.w*0x9e3779b97f4a7c15ULL);}};

static bool stack_has(State const&s,uint8_t q){return std::find(s.stack.begin(),s.stack.end(),q)!=s.stack.end();}
static void canon(State&s){
 std::array<uint8_t,64> rm{},ns{};uint8_t nx=1;
 auto take=[&](uint8_t c){if(c&&!rm[c]){rm[c]=nx;ns[nx]=s.status[c];++nx;}};
 for(auto c:s.comp)take(c);for(auto c:s.stack)take(c);
 for(auto&c:s.comp)if(c)c=rm[c];for(auto&c:s.stack)c=rm[c];s.status.assign(nx,0);for(uint8_t q=1;q<nx;++q)s.status[q]=ns[q];
}
static Key key(State s){canon(s);Key k;k.b.push_back((uint8_t)s.deg.size());for(size_t i=0;i<s.deg.size();++i){k.b.push_back(s.deg[i]);k.b.push_back(s.comp[i]);}k.b.push_back(0xfe);k.b.push_back((uint8_t)s.stack.size());for(auto q:s.stack)k.b.push_back(q);k.b.push_back(0xfd);for(size_t q=1;q<s.status.size();++q)k.b.push_back(s.status[q]);return k;}
static State dec(Key const&k){State s;size_t p=0,n=k.b[p++];s.deg.resize(n);s.comp.resize(n);uint8_t mx=0;for(size_t i=0;i<n;++i){s.deg[i]=k.b[p++];s.comp[i]=k.b[p++];mx=std::max(mx,s.comp[i]);}if(k.b[p++]!=0xfe)throw std::runtime_error("key");size_t sn=k.b[p++];s.stack.resize(sn);for(size_t i=0;i<sn;++i){s.stack[i]=k.b[p++];mx=std::max(mx,s.stack[i]);}if(k.b[p++]!=0xfd)throw std::runtime_error("key2");s.status.assign(mx+1,0);for(uint8_t q=1;q<=mx;++q)s.status[q]=k.b[p++];return s;}

static bool closed_consistent(State const&s,uint8_t q,int except=-1){if(!(s.status[q]&2))return true;if(stack_has(s,q))return false;for(int i=0;i<(int)s.comp.size();++i)if(i!=except&&s.comp[i]==q&&s.deg[i]!=2)return false;return true;}
static bool merge_components(State&s,uint8_t a,uint8_t b,bool virtualEdge,int ia=-1,int ib=-1){
 if(!a||!b)return false;if((s.status[a]&2)||(s.status[b]&2))return false;
 if(a==b){if(virtualEdge){if(s.status[a]&1)return false;s.status[a]|=3;return true;}else{if(!(s.status[a]&1))return false;s.status[a]|=2;return true;}}
 int add=virtualEdge?1:0;int vc=(s.status[a]&1)+(s.status[b]&1)+add;if(vc>1)return false;uint8_t keep=std::min(a,b),kill=std::max(a,b);uint8_t st=vc?1:0;for(auto&c:s.comp)if(c==kill)c=keep;for(auto&c:s.stack)if(c==kill)c=keep;s.status[keep]=st;s.status[kill]=0;return true;
}

struct Engine{
 int r,W,V,E,start=0;std::vector<std::pair<int,int>>edges;std::vector<int>first,last;std::vector<std::vector<int>>active;
 Engine(int rr,int ww):r(rr),W(ww),V((r+1)*W){
   // Outgoing horizontals first, then verticals in this column. This makes the
   // source vertex disappear before the first bottom symbol is consumed.
   for(int c=0;c<W;++c){if(c+1<W)for(int y=0;y<r;++y)edges.push_back({y*W+c,y*W+c+1});for(int y=0;y<r;++y)edges.push_back({y*W+c,(y+1)*W+c});}
   E=edges.size();first.assign(V,1e9);last.assign(V,-1);for(int e=0;e<E;++e){auto[u,v]=edges[e];first[u]=std::min(first[u],e);first[v]=std::min(first[v],e);last[u]=std::max(last[u],e);last[v]=std::max(last[v],e);}active.resize(E+1);for(int e=0;e<=E;++e)for(int v=0;v<V;++v)if(first[v]<e&&last[v]>=e)active[e].push_back(v);
 }
 template<class Map> void run(Map&cur,std::vector<size_t>*statesPerSymbol=nullptr){
   for(int ei=0;ei<E;++ei){auto const&before=active[ei];auto const&after=active[ei+1];auto[u,v]=edges[ei];std::vector<int>work=before;auto ensure=[&](int x){if(std::find(work.begin(),work.end(),x)==work.end())work.push_back(x);};ensure(u);ensure(v);std::vector<int>wp(V,-1),ap(V,-1);for(int i=0;i<(int)work.size();++i)wp[work[i]]=i;for(int i=0;i<(int)after.size();++i)ap[after[i]]=i;
     Map nxt;nxt.reserve(cur.size()*3+16);bool emitted=false;int emitCol=-1;
     for(auto const&kv:cur){Key kk;if constexpr(std::is_same_v<typename Map::key_type,WK>)kk=kv.first.k;else kk=kv.first;State old=dec(kk),base;base.deg.assign(work.size(),0);base.comp.assign(work.size(),0);base.status=old.status;base.stack=old.stack;for(int i=0;i<(int)before.size();++i){int j=wp[before[i]];base.deg[j]=old.deg[i];base.comp[j]=old.comp[i];}
       for(int take=0;take<2;++take){State s=base;int iu=wp[u],iv=wp[v];if(take){uint8_t mu=u==start?1:2,mv=v==start?1:2;if(s.deg[iu]>=mu||s.deg[iv]>=mv)continue;uint8_t a=s.comp[iu],b=s.comp[iv];if((a&&(s.status[a]&2))||(b&&(s.status[b]&2)))continue;++s.deg[iu];++s.deg[iv];if(!a&&!b){uint8_t q=s.status.size();s.status.push_back(0);s.comp[iu]=s.comp[iv]=q;}else if(!a||!b){uint8_t q=a?a:b;s.comp[iu]=s.comp[iv]=q;}else if(!merge_components(s,a,b,false,iu,iv))continue;}
         // There is at most one forgotten bottom vertex at an edge boundary.
         std::vector<int> forgotten;for(int x:work)if(ap[x]==-1)forgotten.push_back(x);
         int bottomX=-1;for(int x:forgotten)if(x/W==r)bottomX=x;
         int symLo=bottomX>=0?0:-1,symHi=bottomX>=0?2:-1;
         for(int sym=symLo;sym<=symHi;++sym){State t=s;bool ok=true;int outSym=bottomX>=0?sym:-1;
           for(int x:forgotten){int ix=wp[x],y=x/W,c=x%W;uint8_t q=t.comp[ix];if(y==r){emitted=true;emitCol=c;if(sym==0){if(t.deg[ix]!=0){ok=false;break;}}else{if(t.deg[ix]!=1||!q){ok=false;break;}if(sym==2){ // L: remember this component as an unmatched bottom endpoint.
                       if((int)t.stack.size()>=r){ok=false;break;}
                       t.stack.push_back(q);
                     }else{ // R: add the virtual matching edge to the most recent L/source marker.
                       if(t.stack.empty()){ok=false;break;}uint8_t q2=t.stack.back();t.stack.pop_back();if(!merge_components(t,q,q2,true,ix,-1)){ok=false;break;}q=t.comp[ix];
                     }}
             }else{bool source=x==start;if((source&&t.deg[ix]!=1)||(!source&&t.deg[ix]!=0&&t.deg[ix]!=2)){ok=false;break;}if(source){if(!q){ok=false;break;}t.stack.insert(t.stack.begin(),q);}}
             t.comp[ix]=0;if(q){bool alive=false;for(auto cc:t.comp)if(cc==q){alive=true;break;}if(!alive&&!stack_has(t,q)){if(!(t.status[q]&2)){ok=false;break;}}}
           }
           if(!ok)continue; // Closed components may remain as degree-2 frontier vertices only.
           for(uint8_t q=1;q<t.status.size();++q)if((t.status[q]&2)&&!closed_consistent(t,q)){ok=false;break;}if(!ok)continue;
           State out;out.deg.resize(after.size());out.comp.resize(after.size());out.status=t.status;out.stack=t.stack;for(int i=0;i<(int)after.size();++i){int j=wp[after[i]];out.deg[i]=t.deg[j];out.comp[i]=t.comp[j];}canon(out);Key nk=key(out);Count ways=kv.second;if constexpr(std::is_same_v<typename Map::key_type,WK>){MateID ww=kv.first.w;if(outSym>=0){int p=W-1-emitCol;auto mv=outSym==0?oneesan::gridfp::N:outSym==1?oneesan::gridfp::R:oneesan::gridfp::L;ww|=MateID(mv)<<(2*p);}nxt[WK{std::move(nk),ww}]+=ways;}else nxt[std::move(nk)]+=ways;
         }
       }
     }cur.swap(nxt);if(emitted&&statesPerSymbol)statesPerSymbol->push_back(cur.size());
   }
 }
};
static std::map<MateID,Count> words(int r,int W){Engine e(r,W);std::unordered_map<WK,Count,WKH>m;State z;z.status.assign(1,0);m[WK{key(z),0}]=1;e.run(m);std::map<MateID,Count>out;for(auto const&kv:m){State s=dec(kv.first.k);if(s.deg.empty()&&s.stack.empty())out[kv.first.w]+=kv.second;}return out;}
static std::map<MateID,Count> gridfp(int rows,int W){using namespace oneesan::gridfp;std::unordered_map<MateID,Count>M,D,nM,nD;M[MateID(R)<<(2*(W-1))]=1;for(int row=0;row<rows;++row)for(int p=W-1;p>=1;--p){nM.clear();nD.clear();for(auto const&kv:M){auto m=kv.first,c=kv.second;nM[m]+=c;auto z=include_horizontal(m,W,p);if(z.valid){if(z.blocked)nD[z.mate]+=c;else nM[z.mate]+=c;}}for(auto const&kv:D)nM[blocked_exclude(kv.first,p)]+=kv.second;M.swap(nM);D.swap(nD);}return {M.begin(),M.end()};}
static std::string show(MateID m,int W){std::string s;for(int p=W-1;p>=0;--p){auto v=oneesan::gridfp::mget(m,p);s+=v==oneesan::gridfp::N?'N':v==oneesan::gridfp::R?'R':'L';}return s;}
int main(int ac,char**av){int r=ac>1?atoi(av[1]):3,W=ac>2?atoi(av[2]):7;bool cmp=ac<=3||atoi(av[3]);if(cmp){auto a=words(r,W),b=gridfp(r,W);uint64_t bad=0;std::map<MateID,Count>keys=a;for(auto const&kv:b)keys.try_emplace(kv.first,0);for(auto const&kv:keys){Count x=a.count(kv.first)?a.at(kv.first):0,y=b.count(kv.first)?b.at(kv.first):0;if(x!=y){if(bad<8)std::cerr<<"bad "<<show(kv.first,W)<<" loop="<<x<<" fp="<<y<<"\n";++bad;}}std::cout<<"r="<<r<<" W="<<W<<" loop_words="<<a.size()<<" gridfp="<<b.size()<<" bad="<<bad<<"\n";if(bad)return 1;}
 Engine e(r,W);std::unordered_map<Key,Count,KH>m;State z;z.status.assign(1,0);m[key(z)]=1;std::vector<size_t>ss;e.run(m,&ss);std::cout<<"raw-loop r="<<r<<" W="<<W<<" per_symbol=";for(size_t i=0;i<ss.size();++i)std::cout<<(i?",":"")<<ss[i];std::cout<<" final="<<m.size()<<"\n";
}
