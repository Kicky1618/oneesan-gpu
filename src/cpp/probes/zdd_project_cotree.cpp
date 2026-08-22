#include <rapidd/rapidd.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
using rapidd::Zdd; using rapidd::ZddManager;
struct Rec{uint32_t id,lev,lo,hi;};
struct Loaded{int n=0,vars=0;uint32_t rootid=0;std::vector<int> edge;std::unique_ptr<ZddManager> mgr;Zdd root;};
static Loaded load(const std::string&p,uint32_t cap){
 std::ifstream f(p);if(!f)throw std::runtime_error("open");std::string line;getline(f,line);if(line!="ONEESAN_ZDD_V1")throw std::runtime_error("magic");Loaded z;std::vector<Rec> rs;std::vector<std::pair<int,int>> vs;
 while(getline(f,line)){if(line.empty()||line[0]=='#')continue;std::istringstream is(line);std::string t;is>>t;if(t=="grid_n")is>>z.n;else if(t=="variables")is>>z.vars;else if(t=="root")is>>z.rootid;else if(t=="var"){int l,e,u,v;is>>l>>e>>u>>v;vs.push_back({l,e});}else if(t=="node"){Rec r;is>>r.id>>r.lev>>r.lo>>r.hi;rs.push_back(r);}else if(t=="end")break;}
 z.edge.assign(z.vars+1,-1);for(auto [l,e]:vs)z.edge[l]=e;z.mgr=std::make_unique<ZddManager>(ZddManager::Config{.max_nodes=cap,.unique_shards=256,.thread_safe=false,.apply_cache_slots=1u<<20,.unary_cache_slots=1u<<18});z.mgr->ensure_variables(z.vars);
 std::sort(rs.begin(),rs.end(),[](auto&a,auto&b){return a.lev==b.lev?a.id<b.id:a.lev<b.lev;});std::unordered_map<uint32_t,Zdd> m;m.reserve(rs.size()*2+3);m.emplace(0,z.mgr->empty());m.emplace(1,z.mgr->unit());
 for(auto&r:rs)m.emplace(r.id,z.mgr->make_node(r.lev,m.at(r.lo),m.at(r.hi)));z.root=m.at(z.rootid);return z;
}
static std::vector<std::pair<int,int>> canon(int n){int W=n+1;std::vector<std::pair<int,int>> e;auto v=[&](int r,int c){return r*W+c;};for(int r=0;r<W;++r)for(int c=0;c<W;++c){if(c+1<W)e.push_back({v(r,c),v(r,c+1)});if(r+1<W)e.push_back({v(r,c),v(r+1,c)});}return e;}
int main(int ac,char**av){try{if(ac<2){std::cerr<<"usage input [cap]\n";return 2;}uint32_t cap=ac>2?std::stoul(av[2]):32000000u;auto z=load(av[1],cap);int W=z.n+1,P=z.n*z.n;auto ce=canon(z.n);std::vector<int> nl(z.vars+1),outedge(P+1,-1);int q=0;
 for(int lev=z.vars;lev>=1;--lev){int eid=z.edge[lev];auto [u,v]=ce.at(eid);int ru=u/W,rv=v/W;bool horizontal=ru==rv;if(!horizontal)throw std::runtime_error("input is not horizontal-only");bool keep=ru!=0;if(keep){int x=P-q++;nl[lev]=x;outedge[x]=eid;}}
 if(q!=P)throw std::runtime_error("kept count");ZddManager d(ZddManager::Config{.max_nodes=cap,.unique_shards=256,.thread_safe=false,.apply_cache_slots=1u<<20,.unary_cache_slots=1u<<18});d.ensure_variables(P);std::unordered_map<uint32_t,Zdd> mm;mm.reserve(z.root.node_count()*2+3);mm.emplace(0,d.empty());mm.emplace(1,d.unit());
 std::function<Zdd(Zdd)> go=[&](Zdd x)->Zdd{if(auto it=mm.find(x.raw());it!=mm.end())return it->second;auto l=z.mgr->top_level(x);auto [a,b]=z.mgr->split(x);auto A=go(a),B=go(b);Zdd y=nl[l]?d.make_node(nl[l],A,B):d.apply(ZddManager::Operation::Union,A,B);mm.emplace(x.raw(),y);return y;};auto r=go(z.root);std::cout<<"n="<<z.n<<" source_vars="<<z.vars<<" cotree_vars="<<P<<" source_nodes="<<z.root.node_count()<<" cotree_nodes="<<r.node_count()<<" allocated="<<d.allocated_nodes()<<" card="<<r.cardinality()<<"\n";
}catch(std::exception const&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}
