#include <rapidd/rapidd.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using rapidd::Zdd;
using rapidd::ZddManager;

struct Edge { int u,v; };
struct State {
    std::vector<uint8_t> deg,hdeg;
    std::vector<int16_t> label;
    std::vector<uint8_t> flags;
    bool done=false;
};

class Builder {
public:
    int n,W,V,H,s,t;
    std::string order_name;
    std::vector<Edge> horder;
    std::vector<int> hedge_id;
    std::vector<int> event_step,first_action;
    std::vector<std::vector<int>> events_at,active_at;
    ZddManager manager;
    std::vector<std::unordered_map<std::string,Zdd>> memo;
    uint64_t calls=0,hits=0,pruned=0;
    int max_frontier=0; uint64_t sum_frontier=0;

    int vid(int r,int c) const{return r*W+c;}
    int hcanon(int r,int c) const{
        int id=0;
        for(int rr=0;rr<W;++rr)for(int cc=0;cc<W;++cc){
            if(cc+1<W){if(rr==r&&cc==c)return id;++id;}
            if(rr+1<W)++id;
        }
        return -1;
    }

    Builder(int nn,std::string ord,uint32_t cap):n(nn),W(nn+1),V(W*W),H(W*(W-1)),s(0),t(V-1),order_name(std::move(ord)),
      manager(ZddManager::Config{.max_nodes=cap,.unique_shards=256,.thread_safe=false,
                                .apply_cache_slots=1u<<18,.unary_cache_slots=1u<<16}){
        if(n<1)throw std::invalid_argument("n>=1");
        build_order();
        build_schedule();
        manager.ensure_variables(H);
        memo.resize(H+1);
    }

    void build_order(){
        auto push=[&](int r,int c){horder.push_back({vid(r,c),vid(r,c+1)});hedge_id.push_back(hcanon(r,c));};
        if(order_name=="row"){
            for(int r=0;r<W;++r)for(int c=0;c<W-1;++c)push(r,c);
        }else if(order_name=="rtl"){
            for(int r=0;r<W;++r)for(int c=W-2;c>=0;--c)push(r,c);
        }else if(order_name=="formula"){
            // Empirical horizontal-only optimum family discovered for n=7..9:
            // special two-row prefix, then RTL rows, while the bottom two rows
            // retain the interval-order zipper.
            if(n<4)throw std::invalid_argument("formula order requires n>=4");
            int m=n;
            push(0,m-1);push(0,m-2);push(1,m-1);push(0,m-3);push(1,m-2);push(1,m-3);
            for(int k=m-4;k>=0;--k){push(0,k);push(1,k);}
            for(int r=2;r<W-2;++r)for(int c=W-2;c>=0;--c)push(r,c);
            int a=W-2,b=W-1;
            push(a,W-2);push(a,W-3);push(b,W-2);
            for(int c=W-4;c>=0;--c){push(a,c);push(b,c+1);}
            push(b,0);
        }else throw std::invalid_argument("order row|rtl|formula");
        if((int)horder.size()!=H)throw std::runtime_error("horizontal order size mismatch");
        std::unordered_set<int> seen;
        for(int id:hedge_id)if(!seen.insert(id).second)throw std::runtime_error("duplicate horizontal edge");
    }

    void build_schedule(){
        std::vector<int> hfirst(V,H), hlast(V,-1);
        for(int i=0;i<H;++i)for(int v:{horder[i].u,horder[i].v}){hfirst[v]=std::min(hfirst[v],i);hlast[v]=std::max(hlast[v],i);}
        event_step.assign(V,-1); first_action.assign(V,H);
        events_at.assign(H,{});
        for(int c=0;c<W;++c){
            int above=-1;
            for(int r=0;r<W;++r){
                int v=vid(r,c);
                int ev=std::max(hlast[v],above);
                if(ev<0||ev>=H)throw std::runtime_error("bad event step");
                event_step[v]=ev;
                first_action[v]=std::min(hfirst[v], r?above:H);
                events_at[ev].push_back(v);
                above=ev;
            }
        }
        for(auto &q:events_at)std::sort(q.begin(),q.end(),[&](int a,int b){return a/W<b/W;});
        active_at.assign(H+1,{});
        for(int i=0;i<=H;++i){
            for(int v=0;v<V;++v)if(first_action[v]<i && i<=event_step[v])active_at[i].push_back(v);
            max_frontier=std::max(max_frontier,(int)active_at[i].size());sum_frontier+=active_at[i].size();
        }
    }

    uint8_t terminal_flag(int v)const{return uint8_t((v==s?1:0)|(v==t?2:0));}
    bool degree_final_ok(int v,int d)const{return v==s||v==t?d==1:(d==0||d==2);}

    int new_label(State& st,uint8_t f){int z=st.flags.size();st.flags.push_back(f);return z;}
    bool include_edge(State&st,int u,int v,bool horizontal){
        if(st.done)return false;
        int lu=(u==s||u==t)?1:2,lv=(v==s||v==t)?1:2;
        if(st.deg[u]>=lu||st.deg[v]>=lv)return false;
        int a=st.label[u],b=st.label[v];
        if(a>=0&&b>=0&&a==b)return false;
        if(a<0&&b<0){int z=new_label(st,terminal_flag(u)|terminal_flag(v));st.label[u]=st.label[v]=z;}
        else if(a<0){st.label[u]=b;st.flags[b]|=terminal_flag(u);}
        else if(b<0){st.label[v]=a;st.flags[a]|=terminal_flag(v);}
        else {st.flags[a]|=st.flags[b];for(int x=0;x<V;++x)if(st.label[x]==b)st.label[x]=a;st.flags[b]=0;}
        ++st.deg[u];++st.deg[v];if(horizontal){++st.hdeg[u];++st.hdeg[v];}
        return true;
    }

    bool finalize_vertex(State&st,int v){
        if(!degree_final_ok(v,st.deg[v]))return false;
        int16_t lab=st.label[v];st.deg[v]=st.hdeg[v]=0;st.label[v]=-1;
        if(lab<0)return true;
        bool alive=false;for(int x=0;x<V;++x)if(st.label[x]==lab){alive=true;break;}
        if(alive)return true;
        uint8_t f=lab<(int)st.flags.size()?st.flags[lab]:0;
        if(f!=3)return false;
        for(int x=0;x<V;++x)if(st.label[x]>=0)return false;
        st.done=true;return true;
    }

    void canonicalize(State&st,int next){
        std::vector<int16_t> map(st.flags.size(),-1);std::vector<uint8_t> nf;int16_t k=0;
        for(int v:active_at[next]){int16_t x=st.label[v];if(x<0)continue;if(map[x]<0){map[x]=k++;nf.push_back(st.flags[x]);}st.label[v]=map[x];}
        st.flags=std::move(nf);
    }

    bool process_events(State&st,int step){
        for(int v:events_at[step]){
            int r=v/W,c=v%W;
            // Before this vertex event, the only vertical edge incident from
            // below is not decided yet. deg-hdeg is therefore the known above bit.
            int above=int(st.deg[v])-int(st.hdeg[v]);
            if(above<0||above>1)return false;
            int below=above ^ (st.hdeg[v]&1) ^ ((v==s||v==t)?1:0);
            if(r+1<W){
                if(below && !include_edge(st,v,vid(r+1,c),false))return false;
            }else if(below)return false;
            if(!finalize_vertex(st,v))return false;
        }
        canonicalize(st,step+1);
        return true;
    }

    std::string key(State const&st,int step)const{
        std::string k;k.reserve(active_at[step].size()*3+st.flags.size()+2);k.push_back(char(st.done));
        for(int v:active_at[step]){k.push_back(char(st.deg[v]));k.push_back(char(st.hdeg[v]));k.push_back(char(st.label[v]+1));}
        k.push_back(char(st.flags.size()));for(auto f:st.flags)k.push_back(char(f));return k;
    }

    Zdd solve(int step,State const&in){
        ++calls;if(step==H)return in.done?manager.unit():manager.empty();
        std::string k=key(in,step);auto &mm=memo[step];if(auto it=mm.find(k);it!=mm.end()){++hits;return it->second;}
        State a=in;Zdd lo=manager.empty();if(process_events(a,step))lo=solve(step+1,a);else ++pruned;
        State b=in;Zdd hi=manager.empty();auto e=horder[step];if(include_edge(b,e.u,e.v,true)&&process_events(b,step))hi=solve(step+1,b);else ++pruned;
        Zdd z=manager.make_node(H-step,lo,hi);mm.emplace(std::move(k),z);return z;
    }

    Zdd build(){State st;st.deg.assign(V,0);st.hdeg.assign(V,0);st.label.assign(V,-1);return solve(0,st);}
};

int main(int argc,char**argv){
    try{
        if(argc<3){std::cerr<<"usage: "<<argv[0]<<" n row|rtl|formula [max_nodes]\n";return 2;}
        int n=std::stoi(argv[1]);std::string o=argv[2];uint32_t cap=argc>3?std::stoul(argv[3]):32u*1024u*1024u;
        auto t0=std::chrono::steady_clock::now();Builder b(n,o,cap);auto root=b.build();double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
        std::cout<<"n="<<n<<" order="<<o<<" variables="<<b.H<<" nodes="<<root.node_count()<<" allocated="<<b.manager.allocated_nodes()
                 <<" calls="<<b.calls<<" hits="<<b.hits<<" pruned="<<b.pruned<<" max_frontier="<<b.max_frontier
                 <<" avg_frontier="<<double(b.sum_frontier)/(b.H+1)<<" card="<<root.cardinality()<<" sec="<<sec<<"\n";
    }catch(std::exception const&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}
}
