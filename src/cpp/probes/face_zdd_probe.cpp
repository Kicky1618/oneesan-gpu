#include <rapidd/rapidd.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using rapidd::Zdd;using rapidd::ZddManager;
struct Edge{int u,v,a=-1,b=-1;bool q=false;};
struct State{std::vector<uint8_t>deg,fval;std::vector<int16_t>lab;std::vector<uint8_t>flags;bool done=false;};
class B{public:int n,W,V,Q,s,t;std::string ord;std::vector<int> cellorder,stepof,lastface;std::vector<Edge> edges;std::vector<std::vector<int>> edgeat,finat,activev,activef;ZddManager m;std::vector<std::unordered_map<std::string,Zdd>> memo;uint64_t calls=0,hits=0,pruned=0;int maxv=0,maxfaces=0;
 int vid(int r,int c)const{return r*W+c;}int cid(int r,int c)const{return r*n+c;}
 B(int nn,std::string o,uint32_t cap):n(nn),W(n+1),V(W*W),Q(n*n),s(0),t(V-1),ord(std::move(o)),m(ZddManager::Config{.max_nodes=cap,.unique_shards=256,.thread_safe=false,.apply_cache_slots=1u<<18,.unary_cache_slots=1u<<16}){order();graph();schedule();m.ensure_variables(Q);memo.resize(Q+1);}
 void order(){std::vector<std::pair<int,int>> cells;for(int r=0;r<n;++r)for(int c=0;c<n;++c)cells.push_back({r,c});bool bu=ord.find("BU")!=std::string::npos,rl=ord.find("RL")!=std::string::npos,snake=ord.find("snake")!=std::string::npos;std::sort(cells.begin(),cells.end(),[&](auto a,auto b){int ar=bu?-a.first:a.first,br=bu?-b.first:b.first;if(ar!=br)return ar<br;bool rr=rl;if(snake)rr^=((bu?(n-1-a.first):a.first)&1);int ac=rr?-a.second:a.second,bc=rr?-b.second:b.second;return ac<bc;});for(auto [r,c]:cells)cellorder.push_back(cid(r,c));stepof.assign(Q,-1);for(int i=0;i<Q;++i)stepof[cellorder[i]]=i;}
 void graph(){ // Q reference path: top boundary then right boundary.
  for(int r=0;r<W;++r)for(int c=0;c<W-1;++c){Edge e{vid(r,c),vid(r,c+1)};if(r>0)e.a=cid(r-1,c);if(r<n)e.b=cid(r,c);e.q=(r==0);edges.push_back(e);}
  for(int r=0;r<W-1;++r)for(int c=0;c<W;++c){Edge e{vid(r,c),vid(r+1,c)};if(c>0)e.a=cid(r,c-1);if(c<n)e.b=cid(r,c);e.q=(c==n);edges.push_back(e);}
 }
 void schedule(){edgeat.assign(Q,{});finat.assign(Q,{});std::vector<int> firstv(V,Q),lastv(V,-1);lastface.assign(Q,-1);for(int ei=0;ei<(int)edges.size();++ei){auto&e=edges[ei];int st=-1;if(e.a>=0)st=std::max(st,stepof[e.a]);if(e.b>=0)st=std::max(st,stepof[e.b]);if(st<0)throw std::runtime_error("edge no face");edgeat[st].push_back(ei);firstv[e.u]=std::min(firstv[e.u],st);firstv[e.v]=std::min(firstv[e.v],st);lastv[e.u]=std::max(lastv[e.u],st);lastv[e.v]=std::max(lastv[e.v],st);if(e.a>=0)lastface[e.a]=std::max(lastface[e.a],st);if(e.b>=0)lastface[e.b]=std::max(lastface[e.b],st);}for(int v=0;v<V;++v)finat[lastv[v]].push_back(v);activev.assign(Q+1,{});activef.assign(Q+1,{});for(int i=0;i<=Q;++i){for(int v=0;v<V;++v)if(firstv[v]<i&&i<=lastv[v])activev[i].push_back(v);for(int c=0;c<Q;++c)if(stepof[c]<i&&i<=lastface[c])activef[i].push_back(c);maxv=std::max(maxv,(int)activev[i].size());maxfaces=std::max(maxfaces,(int)activef[i].size());}}
 uint8_t tf(int v)const{return(v==s?1:0)|(v==t?2:0);}bool okdeg(int v,int d)const{return v==s||v==t?d==1:(d==0||d==2);}
 bool add(State&z,int u,int v){if(z.done)return false;int du=(u==s||u==t)?1:2,dv=(v==s||v==t)?1:2;if(z.deg[u]>=du||z.deg[v]>=dv)return false;int a=z.lab[u],b=z.lab[v];if(a>=0&&b>=0&&a==b)return false;if(a<0&&b<0){int q=z.flags.size();z.flags.push_back(tf(u)|tf(v));z.lab[u]=z.lab[v]=q;}else if(a<0){z.lab[u]=b;z.flags[b]|=tf(u);}else if(b<0){z.lab[v]=a;z.flags[a]|=tf(v);}else{z.flags[a]|=z.flags[b];for(int x=0;x<V;++x)if(z.lab[x]==b)z.lab[x]=a;z.flags[b]=0;}++z.deg[u];++z.deg[v];return true;}
 bool fin(State&z,int v){if(!okdeg(v,z.deg[v]))return false;int q=z.lab[v];z.deg[v]=0;z.lab[v]=-1;if(q<0)return true;bool alive=false;for(int x=0;x<V;++x)if(z.lab[x]==q){alive=true;break;}if(alive)return true;if(z.flags[q]!=3)return false;for(int x=0;x<V;++x)if(z.lab[x]>=0)return false;z.done=true;return true;}
 void canon(State&z,int nx){std::vector<int16_t>mp(z.flags.size(),-1);std::vector<uint8_t>nf;int k=0;for(int v:activev[nx]){int q=z.lab[v];if(q<0)continue;if(mp[q]<0){mp[q]=k++;nf.push_back(z.flags[q]);}z.lab[v]=mp[q];}z.flags=std::move(nf);for(int c=0;c<Q;++c)if(lastface[c]<nx)z.fval[c]=2;}
 bool events(State&z,int st){for(int ei:edgeat[st]){auto&e=edges[ei];int x=e.q?1:0;if(e.a>=0){if(z.fval[e.a]>1)return false;x^=z.fval[e.a];}if(e.b>=0){if(z.fval[e.b]>1)return false;x^=z.fval[e.b];}if(x&&!add(z,e.u,e.v))return false;}for(int v:finat[st])if(!fin(z,v))return false;canon(z,st+1);return true;}
 std::string key(State const&z,int st)const{std::string k;k.reserve(activev[st].size()*2+activef[st].size()+z.flags.size()+2);k.push_back(char(z.done));for(int v:activev[st]){k.push_back(char(z.deg[v]));k.push_back(char(z.lab[v]+1));}for(int c:activef[st])k.push_back(char(z.fval[c]));k.push_back(char(z.flags.size()));for(auto f:z.flags)k.push_back(char(f));return k;}
 Zdd solve(int st,State const&in){++calls;if(st==Q)return in.done?m.unit():m.empty();auto k=key(in,st);auto&mm=memo[st];if(auto it=mm.find(k);it!=mm.end()){++hits;return it->second;}int cell=cellorder[st];State a=in;a.fval[cell]=0;Zdd lo=m.empty();if(events(a,st))lo=solve(st+1,a);else++pruned;State b=in;b.fval[cell]=1;Zdd hi=m.empty();if(events(b,st))hi=solve(st+1,b);else++pruned;Zdd z=m.make_node(Q-st,lo,hi);mm.emplace(std::move(k),z);return z;}
 Zdd build(){State z;z.deg.assign(V,0);z.lab.assign(V,-1);z.fval.assign(Q,2);return solve(0,z);}
};
int main(int ac,char**av){try{if(ac<3){std::cerr<<"usage n TD-LR|TD-RL|BU-LR|BU-RL|TD-snake|BU-snake [cap]\n";return 2;}int n=std::stoi(av[1]);uint32_t cap=ac>3?std::stoul(av[3]):32000000u;auto tt=std::chrono::steady_clock::now();B b(n,av[2],cap);auto r=b.build();double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-tt).count();std::cout<<"n="<<n<<" order="<<av[2]<<" vars="<<b.Q<<" nodes="<<r.node_count()<<" allocated="<<b.m.allocated_nodes()<<" calls="<<b.calls<<" hits="<<b.hits<<" pruned="<<b.pruned<<" max_vfront="<<b.maxv<<" max_facefront="<<b.maxfaces<<" card="<<r.cardinality()<<" sec="<<sec<<"\n";}catch(std::exception const&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}
