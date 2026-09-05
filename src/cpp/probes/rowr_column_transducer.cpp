#include "../../common/gridfp_transition.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>
using Count=std::uint64_t;using MateID=oneesan::gridfp::MateID;

struct State{std::vector<uint8_t>deg,comp,flags;}; // endpoint status: 0 none, 1 source, 2 bottom-open, 3 closed(two fixed endpoints)
struct Key{std::vector<uint8_t>b;bool operator==(Key const&o)const{return b==o.b;}};
struct KH{size_t operator()(Key const&k)const noexcept{uint64_t h=1469598103934665603ULL;for(auto x:k.b){h^=x;h*=1099511628211ULL;}return h;}};
static void canon(State&s){std::array<uint8_t,64>rm{},nf{};uint8_t nx=1;for(auto&c:s.comp)if(c){if(!rm[c]){rm[c]=nx;nf[nx]=s.flags[c];++nx;}c=rm[c];}s.flags.assign(nx,0);for(uint8_t c=1;c<nx;++c)s.flags[c]=nf[c];}
static Key key(State s){canon(s);Key k;k.b.push_back((uint8_t)s.deg.size());for(size_t i=0;i<s.deg.size();++i){k.b.push_back(s.deg[i]);k.b.push_back(s.comp[i]);}k.b.push_back(0xff);for(size_t c=1;c<s.flags.size();++c)k.b.push_back(s.flags[c]);return k;}
static State dec(Key const&k){State s;size_t p=0,n=k.b[p++];s.deg.resize(n);s.comp.resize(n);uint8_t mx=0;for(size_t i=0;i<n;++i){s.deg[i]=k.b[p++];s.comp[i]=k.b[p++];mx=std::max(mx,s.comp[i]);}++p;s.flags.assign(mx+1,0);for(uint8_t c=1;c<=mx;++c)s.flags[c]=k.b[p++];return s;}
struct WK{Key k;MateID w=0;bool operator==(WK const&o)const{return w==o.w&&k==o.k;}};struct WKH{size_t operator()(WK const&x)const noexcept{return KH{}(x.k)^(x.w*0x9e3779b97f4a7c15ULL);}};

struct Engine{
 int r,W,V,E,start=0;std::vector<std::pair<int,int>>edges;std::vector<int>first,last;std::vector<std::vector<int>>active;std::vector<int>bottom_col;
 Engine(int rr,int ww):r(rr),W(ww),V((r+1)*W){
   // Column-major: verticals in c, then horizontals c->c+1 on the processed r rows.
   for(int c=0;c<W;++c){for(int y=0;y<r;++y)edges.push_back({y*W+c,(y+1)*W+c});if(c+1<W)for(int y=0;y<r;++y)edges.push_back({y*W+c,y*W+c+1});}
   E=edges.size();first.assign(V,1e9);last.assign(V,-1);for(int e=0;e<E;++e){auto[u,v]=edges[e];first[u]=std::min(first[u],e);first[v]=std::min(first[v],e);last[u]=std::max(last[u],e);last[v]=std::max(last[v],e);}active.resize(E+1);for(int e=0;e<=E;++e)for(int v=0;v<V;++v)if(first[v]<e&&last[v]>=e)active[e].push_back(v);
 }
 template<class Map> void run(Map&cur,bool keepWord,std::vector<size_t>*topoByCol=nullptr){
   int emitted=0;for(int ei=0;ei<E;++ei){auto const&before=active[ei];auto const&after=active[ei+1];auto[u,v]=edges[ei];std::vector<int>work=before;auto ensure=[&](int x){if(std::find(work.begin(),work.end(),x)==work.end())work.push_back(x);};ensure(u);ensure(v);std::vector<int>wp(V,-1),ap(V,-1);for(int i=0;i<(int)work.size();++i)wp[work[i]]=i;for(int i=0;i<(int)after.size();++i)ap[after[i]]=i;
     Map nxt;nxt.reserve(cur.size()*2+16);bool emittedThis=false;int emitCol=-1;
     for(auto const&kv:cur){Key kk;if constexpr(std::is_same_v<typename Map::key_type,WK>)kk=kv.first.k;else kk=kv.first;State old=dec(kk),base;base.deg.assign(work.size(),0);base.comp.assign(work.size(),0);base.flags=old.flags;for(int i=0;i<(int)before.size();++i){int j=wp[before[i]];base.deg[j]=old.deg[i];base.comp[j]=old.comp[i];}
       for(int take=0;take<2;++take){State s=base;bool ok=true;int iu=wp[u],iv=wp[v];if(take){uint8_t mu=u==start?1:2,mv=v==start?1:2;if(s.deg[iu]>=mu||s.deg[iv]>=mv)continue;uint8_t a=s.comp[iu],b=s.comp[iv];if((a&&s.flags[a]==3)||(b&&s.flags[b]==3))continue;++s.deg[iu];++s.deg[iv];auto mergeStatus=[](uint8_t x,uint8_t y)->uint8_t{if(!x)return y;if(!y)return x;return 3;};if(!a&&!b){uint8_t q=s.flags.size();s.flags.push_back((u==start||v==start)?1:0);s.comp[iu]=s.comp[iv]=q;}else if(!a||!b){uint8_t q=a?a:b;s.comp[iu]=s.comp[iv]=q;if(u==start||v==start)s.flags[q]=mergeStatus(s.flags[q],1);}else{if(a==b)continue;uint8_t keep=std::min(a,b),kill=std::max(a,b);uint8_t fs=mergeStatus(s.flags[keep],s.flags[kill]);s.flags[keep]=fs;for(auto&c:s.comp)if(c==kill)c=keep;}}
         int outSym=-1;
         for(int x:work){if(ap[x]!=-1)continue;int ix=wp[x],y=x/W,c=x%W;bool bottom=(y==r),source=(x==start);uint8_t q=s.comp[ix];if(bottom){if(s.deg[ix]==0){outSym=0;}else if(s.deg[ix]==1&&q){uint8_t f=s.flags[q];if(f==0){outSym=2;s.flags[q]=2;}else if(f==1||f==2){outSym=1;s.flags[q]=3;}else{ok=false;break;}}else{ok=false;break;}emittedThis=true;emitCol=c;
           }else{if((source&&s.deg[ix]!=1)||(!source&&s.deg[ix]!=0&&s.deg[ix]!=2)){ok=false;break;}}
           s.comp[ix]=0;if(q){bool alive=false;for(auto cc:s.comp)if(cc==q){alive=true;break;}if(!alive&&s.flags[q]!=3){ok=false;break;}}
         }
         if(!ok)continue;State out;out.deg.resize(after.size());out.comp.resize(after.size());out.flags=s.flags;for(int i=0;i<(int)after.size();++i){int j=wp[after[i]];out.deg[i]=s.deg[j];out.comp[i]=s.comp[j];}canon(out);Key nk=key(out);Count ways=kv.second;if constexpr(std::is_same_v<typename Map::key_type,WK>){MateID ww0=kv.first.w;if(outSym>=0){int p=W-1-emitCol;auto mv=outSym==0?oneesan::gridfp::N:outSym==1?oneesan::gridfp::R:oneesan::gridfp::L;ww0|=MateID(mv)<<(2*p);}nxt[WK{std::move(nk),ww0}]+=ways;}else nxt[std::move(nk)]+=ways;
       }
     }cur.swap(nxt);if(emittedThis){++emitted;if(topoByCol)topoByCol->push_back(cur.size());}
   }
 }
};
static std::map<MateID,Count> columnWords(int r,int W){Engine e(r,W);std::unordered_map<WK,Count,WKH>m;State z;z.flags.assign(1,0);m[WK{key(z),0}]=1;e.run(m,true);std::map<MateID,Count>out;for(auto const&kv:m){State s=dec(kv.first.k);if(!s.deg.empty())continue;bool clean=true;for(size_t q=1;q<s.flags.size();++q)if(s.flags[q])clean=false;if(clean)out[kv.first.w]+=kv.second;}return out;}
static std::map<MateID,Count> gridfp(int rows,int W){using namespace oneesan::gridfp;std::unordered_map<MateID,Count>M,D,nM,nD;M[MateID(R)<<(2*(W-1))]=1;for(int row=0;row<rows;++row)for(int p=W-1;p>=1;--p){nM.clear();nD.clear();for(auto const&kv:M){auto m=kv.first,c=kv.second;nM[m]+=c;auto z=include_horizontal(m,W,p);if(z.valid){if(z.blocked)nD[z.mate]+=c;else nM[z.mate]+=c;}}for(auto const&kv:D)nM[blocked_exclude(kv.first,p)]+=kv.second;M.swap(nM);D.swap(nD);}return {M.begin(),M.end()};}
static std::string show(MateID m,int W){std::string s;for(int p=W-1;p>=0;--p){auto v=oneesan::gridfp::mget(m,p);s+=v==oneesan::gridfp::N?'N':v==oneesan::gridfp::R?'R':'L';}return s;}
int main(int ac,char**av){int r=ac>1?atoi(av[1]):4,W=ac>2?atoi(av[2]):8;bool words=ac<=3||atoi(av[3]);if(words){auto a=columnWords(r,W),b=gridfp(r,W);uint64_t bad=0;for(auto const&kv:a)if(b[kv.first]!=kv.second){if(bad<8)std::cerr<<"bad "<<show(kv.first,W)<<" col="<<kv.second<<" fp="<<b[kv.first]<<"\n";++bad;}for(auto const&kv:b)if(a[kv.first]!=kv.second){if(bad<8)std::cerr<<"bad "<<show(kv.first,W)<<" col="<<a[kv.first]<<" fp="<<kv.second<<"\n";++bad;}std::cout<<"r="<<r<<" W="<<W<<" column_words="<<a.size()<<" gridfp="<<b.size()<<" bad="<<bad<<"\n";if(bad)return 1;}
 Engine e(r,W);std::unordered_map<Key,Count,KH>m;State z;z.flags.assign(1,0);m[key(z)]=1;std::vector<size_t>sz;e.run(m,false,&sz);std::cout<<"topology r="<<r<<" W="<<W<<" per_emitted_col=";for(size_t i=0;i<sz.size();++i)std::cout<<(i?",":"")<<sz[i];std::cout<<" final="<<m.size()<<"\n";
}
