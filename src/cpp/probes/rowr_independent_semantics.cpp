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

using oneesan::gridfp::MateID;
using oneesan::gridfp::N;
using oneesan::gridfp::R;
using oneesan::gridfp::L;
using Count = std::uint64_t;

struct State {
    std::vector<std::uint8_t> degree, comp;
    std::vector<std::uint8_t> source; // per canonical component id
};
struct Key { std::vector<std::uint8_t> b; bool operator==(Key const&o)const{return b==o.b;} };
struct Hash { size_t operator()(Key const&k)const noexcept{uint64_t h=1469598103934665603ULL;for(auto x:k.b){h^=x;h*=1099511628211ULL;}return h;} };

static void canonicalize(State&s){
    std::array<std::uint8_t,64> remap{},nf{};std::uint8_t nx=1;
    for(auto &c:s.comp)if(c){if(!remap[c]){remap[c]=nx;nf[nx]=s.source[c];++nx;}c=remap[c];}
    s.source.assign(nx,0);for(std::uint8_t c=1;c<nx;++c)s.source[c]=nf[c];
}
static Key key(State s){canonicalize(s);Key k;k.b.push_back((uint8_t)s.degree.size());for(size_t i=0;i<s.degree.size();++i){k.b.push_back(s.degree[i]);k.b.push_back(s.comp[i]);}k.b.push_back(0xff);for(size_t c=1;c<s.source.size();++c)k.b.push_back(s.source[c]);return k;}
static State decode(Key const&k){State s;size_t p=0,n=k.b[p++];s.degree.resize(n);s.comp.resize(n);uint8_t mx=0;for(size_t i=0;i<n;++i){s.degree[i]=k.b[p++];s.comp[i]=k.b[p++];mx=std::max(mx,s.comp[i]);}++p;s.source.assign(mx+1,0);for(uint8_t c=1;c<=mx;++c)s.source[c]=k.b[p++];return s;}

static std::map<MateID,Count> independent(int rows,int W){
    // r layers: row y horizontals plus verticals y -> y+1, y=0..rows-1.
    const int H=rows+1,V=H*W,start=0;
    std::vector<std::pair<int,int>> edges;
    for(int y=0;y<rows;++y){
        for(int c=0;c+1<W;++c)edges.push_back({y*W+c,y*W+c+1});
        for(int c=0;c<W;++c)edges.push_back({y*W+c,(y+1)*W+c});
    }
    const int E=edges.size();
    std::vector<int> first(V,1e9),last(V,-1);
    for(int e=0;e<E;++e){auto [u,v]=edges[e];first[u]=std::min(first[u],e);first[v]=std::min(first[v],e);last[u]=std::max(last[u],e);last[v]=std::max(last[v],e);}
    // Keep the final boundary row alive after all edges.
    for(int c=0;c<W;++c){int v=rows*W+c;first[v]=std::min(first[v],E-1);last[v]=E;}
    std::vector<std::vector<int>> active(E+1);
    for(int e=0;e<=E;++e)for(int v=0;v<V;++v)if(first[v]<e&&last[v]>=e)active[e].push_back(v);

    std::unordered_map<Key,Count,Hash> cur,nxt;State ini;ini.source.assign(1,0);cur[key(ini)]=1;
    for(int ei=0;ei<E;++ei){auto const&before=active[ei];auto const&after=active[ei+1];auto [u,v]=edges[ei];
        std::vector<int> work=before;auto ensure=[&](int x){if(std::find(work.begin(),work.end(),x)==work.end())work.push_back(x);};ensure(u);ensure(v);
        std::vector<int> bp(V,-1),wp(V,-1),ap(V,-1);for(int i=0;i<(int)before.size();++i)bp[before[i]]=i;for(int i=0;i<(int)work.size();++i)wp[work[i]]=i;for(int i=0;i<(int)after.size();++i)ap[after[i]]=i;
        nxt.clear();nxt.reserve(cur.size()*2+16);
        for(auto const&[kk,ways]:cur){State old=decode(kk),base;base.degree.assign(work.size(),0);base.comp.assign(work.size(),0);base.source=old.source;for(int i=0;i<(int)before.size();++i){int j=wp[before[i]];base.degree[j]=old.degree[i];base.comp[j]=old.comp[i];}
            for(int take=0;take<2;++take){State s=base;bool ok=true;int iu=wp[u],iv=wp[v];if(take){uint8_t mu=u==start?1:2,mv=v==start?1:2;if(s.degree[iu]>=mu||s.degree[iv]>=mv)continue;++s.degree[iu];++s.degree[iv];uint8_t a=s.comp[iu],b=s.comp[iv];if(!a&&!b){uint8_t q=s.source.size();s.source.push_back((u==start||v==start)?1:0);s.comp[iu]=s.comp[iv]=q;}else if(!a||!b){uint8_t q=a?a:b;s.comp[iu]=s.comp[iv]=q;if(u==start||v==start)s.source[q]=1;}else{if(a==b)continue;uint8_t keep=std::min(a,b),kill=std::max(a,b);s.source[keep]|=s.source[kill];for(auto&c:s.comp)if(c==kill)c=keep;}}
                for(int x:work){if(ap[x]!=-1)continue;int ix=wp[x];bool terminal=x==start;if((terminal&&s.degree[ix]!=1)||(!terminal&&s.degree[ix]!=0&&s.degree[ix]!=2)){ok=false;break;}uint8_t q=s.comp[ix];s.comp[ix]=0;if(q){bool alive=false;for(auto c:s.comp)if(c==q){alive=true;break;}if(!alive){ok=false;break;}}}
                if(!ok)continue;State out;out.degree.resize(after.size());out.comp.resize(after.size());out.source=s.source;for(int i=0;i<(int)after.size();++i){int j=wp[after[i]];out.degree[i]=s.degree[j];out.comp[i]=s.comp[j];}canonicalize(out);nxt[key(out)]+=ways;
            }
        }cur.swap(nxt);
    }
    std::map<MateID,Count> out;
    // active[E] is the boundary row in left-to-right column order.
    for(auto const&[kk,ways]:cur){State s=decode(kk);if((int)s.degree.size()!=W)throw std::runtime_error("boundary size");std::map<int,std::vector<int>> pos;int sourceComp=-1;bool ok=true;for(int c=0;c<W;++c){if(s.degree[c]>1){ok=false;break;}if(s.degree[c]==1){int q=s.comp[c];if(!q){ok=false;break;}pos[q].push_back(c);if(s.source[q])sourceComp=q;}}if(!ok||sourceComp<0||pos[sourceComp].size()!=1)continue;MateID m=0;for(auto const&[q,vv]:pos){if(q==sourceComp){int c=vv[0],p=W-1-c;m|=MateID(R)<<(2*p);}else{if(vv.size()!=2){ok=false;break;}int c0=vv[0],c1=vv[1];int p0=W-1-c0,p1=W-1-c1; // high p is leftmost c0
                m|=MateID(L)<<(2*p0);m|=MateID(R)<<(2*p1);}}
        if(ok)out[m]+=ways;
    }
    return out;
}

static std::map<MateID,Count> gridfp(int rows,int W){using namespace oneesan::gridfp;std::unordered_map<MateID,Count>M,D,nM,nD;M[MateID(R)<<(2*(W-1))]=1;for(int row=0;row<rows;++row)for(int p=W-1;p>=1;--p){nM.clear();nD.clear();for(auto const&kv:M){auto m=kv.first,c=kv.second;nM[m]+=c;auto z=include_horizontal(m,W,p);if(z.valid){if(z.blocked)nD[z.mate]+=c;else nM[z.mate]+=c;}}for(auto const&kv:D)nM[blocked_exclude(kv.first,p)]+=kv.second;M.swap(nM);D.swap(nD);}return {M.begin(),M.end()};}
static std::string show(MateID m,int W){std::string s;for(int p=W-1;p>=0;--p){auto v=oneesan::gridfp::mget(m,p);s+=v==N?'N':v==R?'R':'L';}return s;}

int main(int ac,char**av){int maxR=ac>1?std::atoi(av[1]):4,maxW=ac>2?std::atoi(av[2]):9;for(int r=1;r<=maxR;++r)for(int W=2;W<=maxW;++W){auto a=independent(r,W),b=gridfp(r,W);uint64_t bad=0;auto ia=a.begin(),ib=b.begin();while(ia!=a.end()||ib!=b.end()){MateID k;if(ib==b.end()||(ia!=a.end()&&ia->first<ib->first))k=ia->first;else k=ib->first;Count x=a.count(k)?a.at(k):0,y=b.count(k)?b.at(k):0;if(x!=y){if(bad<6)std::cerr<<"mismatch r="<<r<<" W="<<W<<" "<<show(k,W)<<" independent="<<x<<" gridfp="<<y<<"\n";++bad;}if(ia!=a.end()&&ia->first==k)++ia;if(ib!=b.end()&&ib->first==k)++ib;}std::cout<<"r="<<r<<" W="<<W<<" independent="<<a.size()<<" gridfp="<<b.size()<<" bad="<<bad<<"\n";if(bad)return 1;}}
