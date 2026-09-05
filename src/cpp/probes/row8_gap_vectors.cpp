#define ROW8_GAP_TRANSITION_NO_MAIN 1
#include "row8_gap_transition_exact.cpp"
#include <array>
#include <iostream>
#include <memory>
#include <vector>

int main(){
    MODP=Q; Vec all; int col=0; if(!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all)) return 2;
    std::array<std::vector<Packed>,9> H; for(auto const&p:all)H[unpack(p).sp].push_back(p);
    std::array<std::unique_ptr<Space>,9>S;
    S[0]=std::make_unique<Space>(makeSpace(0,H[0],loadA01(0,H[0].size())));
    S[1]=std::make_unique<Space>(makeSpace(1,H[1],loadA01(1,H[1].size())));
    S[2]=std::make_unique<Space>(makeSpace(2,H[2],loadA2(H[2])));
    for(int h=3;h<=8;++h)S[h]=std::make_unique<Space>(makeSpace(h,H[h],makeAupper(h,H[h])));
    std::array<std::unique_ptr<GSpace>,9>G;for(int h=0;h<=8;++h)G[h]=std::make_unique<GSpace>(make_gspace(*S[h]));
    static constexpr int DEL[3]={0,-1,1};
    State z{};z.n=8;
    for(int a=0;a<3;++a){int h=1+DEL[a];WVec v{{pack(z),1}};auto w=wcolumn(std::move(v),8,true,a);auto o=project_raw_combo(*G[h],w);size_t bad=0;for(auto[j,x]:o)bad+=x!=1;std::cout<<"alpha a="<<a<<" h="<<h<<" nz="<<o.size()<<" nonunit="<<bad<<"\n";}
    for(auto [h,a]:{std::pair<int,int>{0,0},{1,1}}){size_t nz=0,bad=0;std::vector<std::pair<int,int>>src;for(int si=0;si<(int)S[h]->states.size();++si)if(G[h]->raw_gap_global[si]>=0)src.push_back({G[h]->raw_gap_global[si],si});std::sort(src.begin(),src.end());for(auto[gi,si]:src){WVec v{{S[h]->states[si],1}};auto x=wlast_value(std::move(v),8,a);if(x){++nz;if(x!=1)++bad;}}std::cout<<"beta h="<<h<<" a="<<a<<" nz="<<nz<<" nonunit="<<bad<<"\n";}
}
