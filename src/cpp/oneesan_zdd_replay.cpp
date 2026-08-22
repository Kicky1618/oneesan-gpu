#include <rapidd/rapidd.hpp>
#include "../common/gridfp_transition.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using rapidd::Zdd;
using rapidd::ZddManager;
using namespace oneesan::gridfp;

struct Edge { int u, v; };
struct ReplayState { MateID mate=0; bool blocked=false; };

class ReplayBuilder {
    int n_, W_, V_, E_;
    std::vector<Edge> canonical_edges_;
    std::unordered_map<std::uint64_t,int> edge_id_;
    std::vector<int> hlevel_, vlevel_; // h: W*(W-1), v: (W-1)*W
    ZddManager manager_;
    std::vector<std::unordered_map<std::uint64_t,Zdd>> memo_;
    std::uint64_t calls_=0, hits_=0;

    static std::uint64_t pairkey(int a,int b){ if(a>b)std::swap(a,b); return (std::uint64_t(std::uint32_t(a))<<32)|std::uint32_t(b); }
    int vid(int r,int c) const { return r*W_+c; }
    int hidx(int r,int c) const { return r*(W_-1)+c; }
    int vidx(int r,int c) const { return r*W_+c; } // r is upper row

    std::uint64_t state_key(ReplayState s) const { return s.mate | (std::uint64_t(s.blocked)<<63); }

    MateID full_source(ReplayState s,int p) const {
        return s.blocked ? minsert(s.mate,p,X) : s.mate;
    }

    bool endpoint_at(ReplayState s,int p,int codepos) const {
        return is_endpoint(mget(full_source(s,p),codepos));
    }

    Zdd forced_edge(int level,bool included,Zdd child){
        if(!included) return child;
        return manager_.make_node(std::uint32_t(level),manager_.empty(),child);
    }

    Zdd advance(int row,int j,ReplayState dst,bool vertical_included){
        int nr=row,nj=j+1;
        if(nj==W_-1){++nr;nj=0;}
        Zdd z=solve(nr,nj,dst);
        if(row>0){
            // The transition from S_j to S_{j+1} also performs the forced
            // vertical decision at column j+1.
            z=forced_edge(vlevel_[vidx(row-1,j+1)],vertical_included,z);
        }
        return z;
    }

    Zdd solve_horizontal(int row,int j,ReplayState s){
        const int p=W_-j-1;
        const MateID full=full_source(s,p);
        // Vertical line into the *next* column is forced by whether the old
        // frontier vertex there is an endpoint.
        const bool vnext=(row>0) && is_endpoint(mget(full,p-1));

        ReplayState lo;
        if(s.blocked) lo={blocked_exclude(s.mate,p),false};
        else lo=s;
        Zdd low=advance(row,j,lo,vnext);

        Zdd high=manager_.empty();
        if(!s.blocked){
            auto z=include_horizontal(s.mate,W_,p);
            if(z.valid) high=advance(row,j,{z.mate,z.blocked},vnext);
        }
        return manager_.make_node(std::uint32_t(hlevel_[hidx(row,j)]),low,high);
    }

    Zdd solve(int row,int j,ReplayState s){
        ++calls_;
        if(row==W_){
            return (!s.blocked && s.mate==MateID(R)) ? manager_.unit() : manager_.empty();
        }
        const int step=row*(W_-1)+j;
        auto &mm=memo_[step];
        auto k=state_key(s);
        if(auto it=mm.find(k);it!=mm.end()){++hits_;return it->second;}

        Zdd out=solve_horizontal(row,j,s);
        if(row>0 && j==0){
            // At row rollover, the forced vertical line in column 0 is
            // represented implicitly by carrying the endpoint symbol down.
            const int p=W_-1;
            const bool v0=endpoint_at(s,p,p);
            out=forced_edge(vlevel_[vidx(row-1,0)],v0,out);
        }
        mm.emplace(k,out);
        return out;
    }

public:
    ReplayBuilder(int n,std::uint32_t max_nodes):n_(n),W_(n+1),V_(W_*W_),E_(2*W_*(W_-1)),
      manager_(ZddManager::Config{.max_nodes=max_nodes,.unique_shards=256,.thread_safe=false,
                                 .apply_cache_slots=1u<<18,.unary_cache_slots=1u<<16}){
        if(n<1||W_>28)throw std::invalid_argument("n must be 1..27");
        canonical_edges_.reserve(E_);
        for(int r=0;r<W_;++r)for(int c=0;c<W_;++c){
            if(c+1<W_){int id=(int)canonical_edges_.size();canonical_edges_.push_back({vid(r,c),vid(r,c+1)});edge_id_[pairkey(vid(r,c),vid(r,c+1))]=id;}
            if(r+1<W_){int id=(int)canonical_edges_.size();canonical_edges_.push_back({vid(r,c),vid(r+1,c)});edge_id_[pairkey(vid(r,c),vid(r+1,c))]=id;}
        }
        if((int)canonical_edges_.size()!=E_)throw std::runtime_error("edge count mismatch");
        hlevel_.assign(W_*(W_-1),0);vlevel_.assign((W_-1)*W_,0);
        int seq=0;
        for(int r=0;r<W_;++r){
            if(r>0) vlevel_[vidx(r-1,0)]=E_-seq++;
            for(int c=0;c<W_-1;++c){
                hlevel_[hidx(r,c)]=E_-seq++;
                if(r>0)vlevel_[vidx(r-1,c+1)]=E_-seq++;
            }
        }
        if(seq!=E_)throw std::runtime_error("variable order mismatch");
        manager_.ensure_variables(E_);
        memo_.resize(W_*(W_-1));
    }

    Zdd build(){
        ReplayState init{MateID(R)<<(2*(W_-1)),false};
        return solve(0,0,init);
    }

    std::vector<std::array<std::uint32_t,4>> rows(Zdd root) const {
        std::unordered_set<std::uint32_t> seen;
        std::vector<Zdd> st{root};std::vector<std::array<std::uint32_t,4>> out;
        while(!st.empty()){
            Zdd z=st.back();st.pop_back();if(z.raw()<=1||!seen.insert(z.raw()).second)continue;
            auto [lo,hi]=manager_.split(z);out.push_back({z.raw(),manager_.top_level(z),lo.raw(),hi.raw()});st.push_back(lo);st.push_back(hi);
        }
        return out;
    }

    void write_text(std::string const&path,Zdd root) const {
        std::ofstream os(path);if(!os)throw std::runtime_error("cannot open "+path);
        os<<"ONEESAN_ZDD_V1\n"<<"grid_n "<<n_<<"\nvertices "<<V_<<"\nvariables "<<E_<<"\nsource 0\ntarget "<<(V_-1)<<"\nroot "<<root.raw()<<"\n";
        std::vector<std::array<int,5>> vars;vars.reserve(E_);
        for(int r=0;r<W_;++r)for(int c=0;c<W_-1;++c){int u=vid(r,c),v=vid(r,c+1);vars.push_back({hlevel_[hidx(r,c)],edge_id_.at(pairkey(u,v)),u,v,0});}
        for(int r=0;r<W_-1;++r)for(int c=0;c<W_;++c){int u=vid(r,c),v=vid(r+1,c);vars.push_back({vlevel_[vidx(r,c)],edge_id_.at(pairkey(u,v)),u,v,0});}
        std::sort(vars.begin(),vars.end(),[](auto const&a,auto const&b){return a[0]<b[0];});
        os<<"# level edge_index u v\n";for(auto const&x:vars)os<<"var "<<x[0]<<' '<<x[1]<<' '<<x[2]<<' '<<x[3]<<"\n";
        auto rr=rows(root);std::sort(rr.begin(),rr.end(),[](auto const&a,auto const&b){return a[0]<b[0];});
        os<<"# id level low high\n";for(auto const&r:rr)os<<"node "<<r[0]<<' '<<r[1]<<' '<<r[2]<<' '<<r[3]<<"\n";os<<"end\n";
    }

    void write_sapporo(std::string const&path,Zdd root) const {
        std::ofstream os(path);if(!os)throw std::runtime_error("cannot open "+path);
        auto rr=rows(root);std::sort(rr.begin(),rr.end(),[](auto const&a,auto const&b){if(a[1]!=b[1])return a[1]<b[1];return a[0]<b[0];});
        std::unordered_map<std::uint32_t,std::uint64_t> id;id.reserve(rr.size()*2+1);std::uint64_t nx=2;for(auto const&r:rr){id[r[0]]=nx;nx+=2;}
        auto ref=[&](std::uint32_t x){if(x==0)return std::string("F");if(x==1)return std::string("T");return std::to_string(id.at(x));};
        os<<"_i "<<E_<<"\n_o 1\n_n "<<rr.size()<<"\n";for(auto const&r:rr)os<<id.at(r[0])<<' '<<r[1]<<' '<<ref(r[2])<<' '<<ref(r[3])<<"\n";os<<ref(root.raw())<<"\n";
    }

    std::uint64_t calls()const{return calls_;}std::uint64_t hits()const{return hits_;}
    std::uint32_t allocated()const{return manager_.allocated_nodes();}
};

int main(int argc,char**argv){
    try{
        if(argc<3){std::cerr<<"usage: "<<argv[0]<<" <n> <output.zdd> [max_nodes] [--sapporo file]\n";return 2;}
        int n=std::stoi(argv[1]);std::string out=argv[2],sapp;std::uint32_t cap=32u*1024u*1024u;int a=3;if(a<argc&&argv[a][0]!='-')cap=(std::uint32_t)std::stoul(argv[a++]);while(a<argc){std::string q=argv[a++];if(q=="--sapporo"&&a<argc)sapp=argv[a++];else throw std::invalid_argument("bad arg "+q);}
        ReplayBuilder b(n,cap);auto root=b.build();b.write_text(out,root);if(!sapp.empty())b.write_sapporo(sapp,root);
        std::cout<<"backend=gridfp-replay-zdd n="<<n<<" cardinality_u64="<<root.cardinality()<<" nodes="<<root.node_count()<<" allocated="<<b.allocated()<<" calls="<<b.calls()<<" memo_hits="<<b.hits()<<" output="<<out;if(!sapp.empty())std::cout<<" sapporo_output="<<sapp;std::cout<<"\n";
    }catch(std::exception const&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
