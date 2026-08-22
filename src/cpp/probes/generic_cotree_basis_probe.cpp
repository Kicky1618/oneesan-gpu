#include <rapidd/rapidd.hpp>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <array>
#include <unordered_set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
using rapidd::Zdd;using rapidd::ZddManager;
struct Edge{int u,v;};
struct S{std::vector<uint8_t>d;std::vector<int16_t>lab;std::vector<uint8_t>fl;bool done=false;};
struct B{
 int n,W,V,Q,s=0,t,root;std::string basis,ord;std::vector<Edge> all,vars;std::vector<char> is_tree;std::vector<int>par,depth,ev,first;std::vector<std::vector<int>>ch,at,active;ZddManager m;std::vector<std::unordered_map<std::string,Zdd>>memo;uint64_t calls=0,hits=0,pruned=0;int maxf=0;uint64_t sumf=0;
 int id(int r,int c)const{return r*W+c;} static uint64_t pk(int a,int b){if(a>b)std::swap(a,b);return(uint64_t(uint32_t(a))<<32)|uint32_t(b);}
 B(int nn,std::string ba,std::string o,uint32_t cap):n(nn),W(n+1),V(W*W),Q(n*n),t(V-1),basis(std::move(ba)),ord(std::move(o)),m(ZddManager::Config{.max_nodes=cap,.unique_shards=256,.thread_safe=false,.apply_cache_slots=1u<<18,.unary_cache_slots=1u<<16}){build_all();build_tree();root_tree();build_vars();schedule();m.ensure_variables(Q);memo.resize(Q+1);}
 void build_all(){for(int r=0;r<W;++r)for(int c=0;c<W;++c){if(c+1<W)all.push_back({id(r,c),id(r,c+1)});if(r+1<W)all.push_back({id(r,c),id(r+1,c)});}is_tree.assign(all.size(),0);}
 int edgeid(int a,int b)const{for(int i=0;i<(int)all.size();++i)if(pk(all[i].u,all[i].v)==pk(a,b))return i;throw std::runtime_error("edge");}
 void mark(int a,int b){is_tree[edgeid(a,b)]=1;}
 void build_tree(){
   if(basis.rfind("vcomb",0)==0){int sr=0;auto z=basis.find(':');if(z!=std::string::npos)sr=std::stoi(basis.substr(z+1));if(sr<0||sr>=W)throw std::runtime_error("spine");for(int r=0;r<W-1;++r)for(int c=0;c<W;++c)mark(id(r,c),id(r+1,c));for(int c=0;c<W-1;++c)mark(id(sr,c),id(sr,c+1));root=id(sr,0);
   }else if(basis=="hcomb"){for(int r=0;r<W;++r)for(int c=0;c<W-1;++c)mark(id(r,c),id(r,c+1));for(int r=0;r<W-1;++r)mark(id(r,0),id(r+1,0));root=0;
   }else if(basis=="snake"){for(int r=0;r<W;++r){for(int c=0;c<W-1;++c)mark(id(r,c),id(r,c+1));if(r+1<W){int c=(r&1)?0:W-1;mark(id(r,c),id(r+1,c));}}root=0;
   }else if(basis.rfind("diag",0)==0){int bias=0;auto z=basis.find(':');if(z!=std::string::npos)bias=std::stoi(basis.substr(z+1));root=0;for(int r=0;r<W;++r)for(int c=0;c<W;++c)if(r||c){if(r>0 && (c==0 || r-c>bias))mark(id(r,c),id(r-1,c));else mark(id(r,c),id(r,c-1));}}
   else throw std::invalid_argument("basis");
   int tc=0;for(char x:is_tree)tc+=x;if(tc!=V-1)throw std::runtime_error("not spanning tree edge count "+std::to_string(tc));
 }
 void root_tree(){std::vector<std::vector<int>>g(V);for(int i=0;i<(int)all.size();++i)if(is_tree[i]){auto e=all[i];g[e.u].push_back(e.v);g[e.v].push_back(e.u);}par.assign(V,-2);depth.assign(V,0);ch.assign(V,{});par[root]=-1;std::vector<int>q{root};for(size_t k=0;k<q.size();++k){int v=q[k];for(int u:g[v])if(par[u]==-2){par[u]=v;depth[u]=depth[v]+1;ch[v].push_back(u);q.push_back(u);}}if((int)q.size()!=V)throw std::runtime_error("tree disconnected");}
 void build_vars(){for(int i=0;i<(int)all.size();++i)if(!is_tree[i])vars.push_back(all[i]);if((int)vars.size()!=Q)throw std::runtime_error("Q");
   auto rc=[&](int v){return std::pair<int,int>{v/W,v%W};};
   if(ord=="physical"||ord=="physical-rev"){std::sort(vars.begin(),vars.end(),[&](Edge a,Edge b){auto[ar,ac]=rc(std::min(a.u,a.v));auto[br,bc]=rc(std::min(b.u,b.v));return std::tie(ar,ac,a.u,a.v)<std::tie(br,bc,b.u,b.v);});if(ord=="physical-rev")std::reverse(vars.begin(),vars.end());}
   else if(ord=="depth"){std::sort(vars.begin(),vars.end(),[&](Edge a,Edge b){int da=std::max(depth[a.u],depth[a.v]),db=std::max(depth[b.u],depth[b.v]);if(da!=db)return da>db;return pk(a.u,a.v)<pk(b.u,b.v);});}
   else if(ord=="tree-post"){std::sort(vars.begin(),vars.end(),[&](Edge a,Edge b){int da=std::min(depth[a.u],depth[a.v]),db=std::min(depth[b.u],depth[b.v]);if(da!=db)return da>db;return pk(a.u,a.v)<pk(b.u,b.v);});}
   else throw std::invalid_argument("order");
 }
 void schedule(){std::vector<int>nf(V,Q),nl(V,-1);for(int i=0;i<Q;++i)for(int v:{vars[i].u,vars[i].v}){nf[v]=std::min(nf[v],i);nl[v]=std::max(nl[v],i);}ev.assign(V,-1);first.assign(V,Q);at.assign(Q,{});std::vector<int>vs(V);for(int i=0;i<V;++i)vs[i]=i;std::sort(vs.begin(),vs.end(),[&](int a,int b){return depth[a]>depth[b];});for(int v:vs){if(v==root)continue;int e=nl[v],f=nf[v];for(int u:ch[v]){e=std::max(e,ev[u]);f=std::min(f,ev[u]);}if(e<0)e=0;ev[v]=e;first[v]=f;at[e].push_back(v);}int rf=nf[root];for(int u:ch[root])rf=std::min(rf,ev[u]);first[root]=rf;for(auto&a:at)std::sort(a.begin(),a.end(),[&](int x,int y){return depth[x]>depth[y];});active.assign(Q+1,{});for(int st=0;st<=Q;++st){for(int v=0;v<V;++v)if(first[v]<st && (v==root?st<=Q:st<=ev[v]))active[st].push_back(v);maxf=std::max(maxf,(int)active[st].size());sumf+=active[st].size();}}
 uint8_t tf(int v)const{return(v==s?1:0)|(v==t?2:0);}bool okdeg(int v,int d)const{return v==s||v==t?d==1:(d==0||d==2);}int nlbl(S&z,uint8_t f){int q=z.fl.size();z.fl.push_back(f);return q;}
 bool add(S&z,int u,int v){if(z.done)return false;int du=(u==s||u==t)?1:2,dv=(v==s||v==t)?1:2;if(z.d[u]>=du||z.d[v]>=dv)return false;int a=z.lab[u],b=z.lab[v];if(a>=0&&b>=0&&a==b)return false;if(a<0&&b<0){int q=nlbl(z,tf(u)|tf(v));z.lab[u]=z.lab[v]=q;}else if(a<0){z.lab[u]=b;z.fl[b]|=tf(u);}else if(b<0){z.lab[v]=a;z.fl[a]|=tf(v);}else{z.fl[a]|=z.fl[b];for(int x=0;x<V;++x)if(z.lab[x]==b)z.lab[x]=a;z.fl[b]=0;}++z.d[u];++z.d[v];return true;}
 bool fin(S&z,int v){if(!okdeg(v,z.d[v]))return false;int q=z.lab[v];z.d[v]=0;z.lab[v]=-1;if(q<0)return true;bool alive=false;for(int x=0;x<V;++x)if(z.lab[x]==q){alive=true;break;}if(alive)return true;if(z.fl[q]!=3)return false;for(int x=0;x<V;++x)if(z.lab[x]>=0)return false;z.done=true;return true;}
 void canon(S&z,int nx){std::vector<int16_t>mp(z.fl.size(),-1);std::vector<uint8_t>nf;int k=0;for(int v:active[nx]){int q=z.lab[v];if(q<0)continue;if(mp[q]<0){mp[q]=k++;nf.push_back(z.fl[q]);}z.lab[v]=mp[q];}z.fl=std::move(nf);}
 bool events(S&z,int st){for(int v:at[st]){int need=(z.d[v]&1)^((v==s||v==t)?1:0);if(need&&!add(z,v,par[v]))return false;if(!fin(z,v))return false;}canon(z,st+1);return true;}
 bool finish(S&z){int need=(z.d[root]&1)^((root==s||root==t)?1:0);if(need)return false;if(!fin(z,root))return false;return z.done;}
 std::string key(S const&z,int st)const{std::string k;k.push_back(char(z.done));for(int v:active[st]){k.push_back(char(z.d[v]));k.push_back(char(z.lab[v]+1));}k.push_back(char(z.fl.size()));for(auto f:z.fl)k.push_back(char(f));return k;}
 Zdd solve(int st,S const&in){++calls;if(st==Q){S z=in;return finish(z)?m.unit():m.empty();}auto k=key(in,st);auto&mm=memo[st];if(auto it=mm.find(k);it!=mm.end()){++hits;return it->second;}S a=in;Zdd lo=m.empty();if(events(a,st))lo=solve(st+1,a);else++pruned;S b=in;Zdd hi=m.empty();auto e=vars[st];if(add(b,e.u,e.v)&&events(b,st))hi=solve(st+1,b);else++pruned;auto z=m.make_node(Q-st,lo,hi);mm.emplace(std::move(k),z);return z;}
 Zdd build(){S z;z.d.assign(V,0);z.lab.assign(V,-1);return solve(0,z);}
 void write(const char*path,Zdd rootz){
   std::ofstream os(path);if(!os)throw std::runtime_error("write");
   std::unordered_map<uint64_t,int>eid;for(int i=0;i<(int)all.size();++i)eid[pk(all[i].u,all[i].v)]=i;
   os<<"ONEESAN_ZDD_V1\n"<<"grid_n "<<n<<"\nvertices "<<V<<"\nvariables "<<Q<<"\nsource 0\ntarget "<<(V-1)<<"\nroot "<<rootz.raw()<<"\nprojection_cotree 1\n";
   os<<"# level edge_index u v\n";for(int st=0;st<Q;++st){auto e=vars[st];os<<"var "<<(Q-st)<<' '<<eid.at(pk(e.u,e.v))<<' '<<e.u<<' '<<e.v<<"\n";}
   std::unordered_set<uint32_t>seen;std::vector<Zdd>q{rootz};std::vector<std::array<uint32_t,4>>rr;
   while(!q.empty()){auto z=q.back();q.pop_back();if(z.raw()<=1||!seen.insert(z.raw()).second)continue;auto[lo,hi]=m.split(z);rr.push_back({z.raw(),m.top_level(z),lo.raw(),hi.raw()});q.push_back(lo);q.push_back(hi);}
   std::sort(rr.begin(),rr.end(),[](auto&a,auto&b){return a[0]<b[0];});os<<"# id level low high\n";for(auto&r:rr)os<<"node "<<r[0]<<' '<<r[1]<<' '<<r[2]<<' '<<r[3]<<"\n";os<<"end\n";
 }
};
int main(int ac,char**av){try{if(ac<4){std::cerr<<"usage n basis order [cap]\n";return 2;}int n=std::stoi(av[1]);uint32_t cap=ac>4?std::stoul(av[4]):32000000;auto t=std::chrono::steady_clock::now();B b(n,av[2],av[3],cap);auto r=b.build();if(const char*o=std::getenv("ZDD_GENERIC_OUT"))b.write(o,r);double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();std::cout<<"n="<<n<<" basis="<<av[2]<<" order="<<av[3]<<" nodes="<<r.node_count()<<" calls="<<b.calls<<" hits="<<b.hits<<" maxf="<<b.maxf<<" avgf="<<double(b.sumf)/(b.Q+1)<<" card="<<r.cardinality()<<" sec="<<sec<<"\n";}catch(std::exception const&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}
