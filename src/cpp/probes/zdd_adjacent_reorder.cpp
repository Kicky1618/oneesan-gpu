#include <rapidd/rapidd.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using rapidd::Zdd;
using rapidd::ZddManager;

struct NodeRec { std::uint32_t id, level, low, high; };

struct Loaded {
    int n=0, variables=0;
    std::uint32_t root_id=0;
    std::vector<int> edge_by_level;
    std::unique_ptr<ZddManager> mgr;
    Zdd root;
};

static Loaded load_zdd(std::string const& path, std::uint32_t cap){
    std::ifstream f(path); if(!f) throw std::runtime_error("cannot open "+path);
    std::string line; std::getline(f,line); if(line!="ONEESAN_ZDD_V1") throw std::runtime_error("bad magic");
    Loaded z; std::vector<NodeRec> recs; std::vector<std::pair<int,int>> vars;
    while(std::getline(f,line)){
        if(line.empty()||line[0]=='#')continue;
        std::istringstream is(line); std::string tag; is>>tag;
        if(tag=="grid_n")is>>z.n;
        else if(tag=="variables")is>>z.variables;
        else if(tag=="root")is>>z.root_id;
        else if(tag=="var"){int lev,eid,u,v;is>>lev>>eid>>u>>v;vars.push_back({lev,eid});}
        else if(tag=="node"){NodeRec r;is>>r.id>>r.level>>r.low>>r.high;recs.push_back(r);}
        else if(tag=="end")break;
    }
    if(!z.variables)throw std::runtime_error("missing variables");
    z.edge_by_level.assign(z.variables+1,-1); for(auto [l,e]:vars)z.edge_by_level.at(l)=e;
    z.mgr=std::make_unique<ZddManager>(ZddManager::Config{.max_nodes=cap,.unique_shards=256,.thread_safe=false,
        .apply_cache_slots=1u<<18,.unary_cache_slots=1u<<16});
    z.mgr->ensure_variables(z.variables);
    std::sort(recs.begin(),recs.end(),[](auto const&a,auto const&b){if(a.level!=b.level)return a.level<b.level;return a.id<b.id;});
    std::unordered_map<std::uint32_t,Zdd> m; m.reserve(recs.size()*2+3);m.emplace(0,z.mgr->empty());m.emplace(1,z.mgr->unit());
    for(auto const&r:recs){
        auto il=m.find(r.low), ih=m.find(r.high); if(il==m.end()||ih==m.end())throw std::runtime_error("child not built");
        m.emplace(r.id,z.mgr->make_node(r.level,il->second,ih->second));
    }
    auto it=m.find(z.root_id);if(it==m.end())throw std::runtime_error("root not built");z.root=it->second;
    return z;
}

struct Swapped {
    std::unique_ptr<ZddManager> mgr;
    Zdd root;
};

static Swapped swap_adjacent(ZddManager& src,Zdd root,int variables,int k,std::uint32_t cap){
    // Swap the variable currently at numeric levels k and k-1.  The numeric
    // order remains 1..variables; callers swap their external level labels.
    if(k<=1||k>variables)throw std::invalid_argument("bad swap level");
    auto dst=std::make_unique<ZddManager>(ZddManager::Config{.max_nodes=cap,.unique_shards=256,.thread_safe=false,
        .apply_cache_slots=1u<<18,.unary_cache_slots=1u<<16});
    dst->ensure_variables(variables);
    std::unordered_map<std::uint32_t,Zdd> memo; memo.reserve(root.node_count()*2+3);
    memo.emplace(0,dst->empty());memo.emplace(1,dst->unit());

    std::function<Zdd(Zdd)> tr=[&](Zdd q)->Zdd{
        if(auto it=memo.find(q.raw());it!=memo.end())return it->second;
        std::uint32_t lev=src.top_level(q); auto [ol,oh]=src.split(q); Zdd out;
        if((int)lev==k){
            auto cof=[&](Zdd c)->std::pair<Zdd,Zdd>{
                if((int)src.top_level(c)==k-1)return src.split(c);
                return {c,Zdd{}}; // placeholder terminal empty handled below
            };
            auto [a0,a1raw]=cof(ol); auto [b0,b1raw]=cof(oh);
            bool a_has=((int)src.top_level(ol)==k-1), b_has=((int)src.top_level(oh)==k-1);
            Zdd a1=a_has?a1raw:src.empty(); Zdd b1=b_has?b1raw:src.empty();
            Zdd x0=dst->make_node(k-1,tr(a0),tr(b0));
            Zdd x1=dst->make_node(k-1,tr(a1),tr(b1));
            out=dst->make_node(k,x0,x1);
        }else if((int)lev==k-1){
            out=dst->make_node(k,tr(ol),tr(oh));
        }else{
            out=dst->make_node(lev,tr(ol),tr(oh));
        }
        memo.emplace(q.raw(),out);return out;
    };
    Zdd nr=tr(root);
    return {std::move(dst),nr};
}

int main(int argc,char**argv){
    try{
        if(argc<2){std::cerr<<"usage: "<<argv[0]<<" input.zdd [max_nodes] [passes]\n";return 2;}
        std::uint32_t cap=argc>=3?(std::uint32_t)std::stoul(argv[2]):32u*1024u*1024u;
        int passes=argc>=4?std::stoi(argv[3]):4;
        Loaded z=load_zdd(argv[1],cap);
        std::uint64_t cur=z.root.node_count();
        std::cout<<"start nodes="<<cur<<" variables="<<z.variables<<" card="<<z.root.cardinality()<<"\n";
        for(int pass=0;pass<passes;++pass){
            bool improved=false;
            // Try from high to low and low to high on alternating passes.
            std::vector<int> ks;for(int k=2;k<=z.variables;++k)ks.push_back(k);if(pass&1)std::reverse(ks.begin(),ks.end());
            for(int k:ks){
                auto s=swap_adjacent(*z.mgr,z.root,z.variables,k,cap);
                auto nn=s.root.node_count();
                if(nn<cur){
                    std::swap(z.edge_by_level[k],z.edge_by_level[k-1]);
                    z.mgr=std::move(s.mgr);z.root=s.root;cur=nn;improved=true;
                    std::cout<<"pass="<<pass<<" swap="<<k<<" nodes="<<cur<<"\n";
                }
            }
            std::cout<<"pass_end="<<pass<<" nodes="<<cur<<" improved="<<improved<<"\n";
            if(!improved)break;
        }
        std::cout<<"final nodes="<<cur<<" card="<<z.root.cardinality()<<" order_edge_ids=";
        for(int lev=z.variables;lev>=1;--lev)std::cout<<z.edge_by_level[lev]<<',';
        std::cout<<"\n";
        if(const char* outp=std::getenv("ZDD_REORDER_OUT")){
            const int W=z.n+1,V=W*W;
            std::vector<std::pair<int,int>> canon;canon.reserve(2*W*(W-1));
            auto vid=[&](int r,int c){return r*W+c;};
            for(int r=0;r<W;++r)for(int c=0;c<W;++c){if(c+1<W)canon.push_back({vid(r,c),vid(r,c+1)});if(r+1<W)canon.push_back({vid(r,c),vid(r+1,c)});}
            std::ofstream os(outp);if(!os)throw std::runtime_error("cannot write output");
            os<<"ONEESAN_ZDD_V1\n"<<"grid_n "<<z.n<<"\nvertices "<<V<<"\nvariables "<<z.variables
              <<"\nsource 0\ntarget "<<(V-1)<<"\nroot "<<z.root.raw()<<"\nprojection_horizontal 1\n";
            os<<"# level edge_index u v\n";
            for(int lev=1;lev<=z.variables;++lev){int eid=z.edge_by_level[lev];auto [u,v]=canon.at(eid);os<<"var "<<lev<<' '<<eid<<' '<<u<<' '<<v<<"\n";}
            std::unordered_set<uint32_t> seen;std::vector<Zdd> st{z.root};std::vector<std::array<uint32_t,4>> rr;
            while(!st.empty()){auto q=st.back();st.pop_back();if(q.raw()<=1||!seen.insert(q.raw()).second)continue;auto [lo,hi]=z.mgr->split(q);rr.push_back({q.raw(),z.mgr->top_level(q),lo.raw(),hi.raw()});st.push_back(lo);st.push_back(hi);}
            std::sort(rr.begin(),rr.end(),[](auto const&a,auto const&b){return a[0]<b[0];});os<<"# id level low high\n";for(auto const&r:rr)os<<"node "<<r[0]<<' '<<r[1]<<' '<<r[2]<<' '<<r[3]<<"\n";os<<"end\n";
        }
    }catch(std::exception const&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
