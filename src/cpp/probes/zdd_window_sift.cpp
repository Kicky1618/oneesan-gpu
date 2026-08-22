#define main zdd_adjacent_original_main
#include "zdd_adjacent_reorder.cpp"
#undef main

static int find_level(Loaded const&z,int eid){for(int l=1;l<=z.variables;++l)if(z.edge_by_level[l]==eid)return l;return -1;}
static std::uint64_t explore_dir(Loaded const&z,int pos,int dir,int window,std::uint32_t cap,int&best_steps){
    ZddManager const* dummy=nullptr;(void)dummy;
    ZddManager* sm=z.mgr.get(); Zdd sr=z.root;
    std::unique_ptr<ZddManager> hold;
    std::uint64_t best=z.root.node_count();best_steps=0;int p=pos;
    for(int step=1;step<=window;++step){
        int k=(dir>0)?p+1:p; if(k<=1||k>z.variables)break;
        try{
            auto q=swap_adjacent(*sm,sr,z.variables,k,cap);auto nn=q.root.node_count();
            if(nn<best){best=nn;best_steps=step;}
            hold=std::move(q.mgr);sr=q.root;sm=hold.get();p+=dir;
        }catch(std::exception const&){break;}
    }
    return best;
}
static void commit_move(Loaded&z,int pos,int dir,int steps,std::uint32_t cap){int p=pos;for(int st=0;st<steps;++st){int k=dir>0?p+1:p;auto q=swap_adjacent(*z.mgr,z.root,z.variables,k,cap);std::swap(z.edge_by_level[k],z.edge_by_level[k-1]);z.mgr=std::move(q.mgr);z.root=q.root;p+=dir;}}
int main(int ac,char**av){try{if(ac<2){std::cerr<<"usage input.zdd [cap] [window] [passes]\n";return 2;}uint32_t cap=ac>2?std::stoul(av[2]):32000000u;int win=ac>3?std::stoi(av[3]):6;int passes=ac>4?std::stoi(av[4]):2;Loaded z=load_zdd(av[1],cap);auto cur=z.root.node_count();std::cout<<"start="<<cur<<" vars="<<z.variables<<" window="<<win<<"\n";
 for(int pass=0;pass<passes;++pass){bool imp=false;std::vector<int> ids;for(int l=z.variables;l>=1;--l)ids.push_back(z.edge_by_level[l]);if(pass&1)std::reverse(ids.begin(),ids.end());
   for(int eid:ids){int pos=find_level(z,eid),bu=0,bd=0;auto nu=explore_dir(z,pos,+1,win,cap,bu);auto nd=explore_dir(z,pos,-1,win,cap,bd);int dir=0,steps=0;auto best=cur;if(nu<best){best=nu;dir=+1;steps=bu;}if(nd<best){best=nd;dir=-1;steps=bd;}if(dir){commit_move(z,pos,dir,steps,cap);cur=z.root.node_count();imp=true;std::cout<<"pass="<<pass<<" eid="<<eid<<" dir="<<dir<<" steps="<<steps<<" nodes="<<cur<<"\n";}}
   std::cout<<"pass_end="<<pass<<" nodes="<<cur<<" improved="<<imp<<"\n";if(!imp)break;}
 std::cout<<"final="<<cur<<" card="<<z.root.cardinality()<<" order_edge_ids=";for(int l=z.variables;l>=1;--l)std::cout<<z.edge_by_level[l]<<',';std::cout<<"\n";
 }catch(std::exception const&e){std::cerr<<"error: "<<e.what()<<"\n";return 1;}}
