#define ROW8_GAP_TRANSITION_NO_MAIN 1
#include "row8_gap_transition_exact.cpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

struct VH { char magic[8]; uint32_t version,r,count; };
struct GapVecEntry { uint32_t tag,sym,h,nz; };

int main(int ac,char**av){
    std::string path=ac>1?av[1]:"work/row8_gap/row8_gap_vectors_v1.bin";
    MODP=Q;
    Vec all; int col=0; if(!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all)) return 2;
    std::array<std::vector<Packed>,9> H; for(auto const&p:all) H[unpack(p).sp].push_back(p);
    std::array<std::unique_ptr<Space>,9>S;
    S[0]=std::make_unique<Space>(makeSpace(0,H[0],loadA01(0,H[0].size())));
    S[1]=std::make_unique<Space>(makeSpace(1,H[1],loadA01(1,H[1].size())));
    S[2]=std::make_unique<Space>(makeSpace(2,H[2],loadA2(H[2])));
    for(int h=3;h<=8;++h)S[h]=std::make_unique<Space>(makeSpace(h,H[h],makeAupper(h,H[h])));
    std::array<std::unique_ptr<GSpace>,9>G; for(int h=0;h<=8;++h)G[h]=std::make_unique<GSpace>(make_gspace(*S[h]));

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path,std::ios::binary|std::ios::trunc); if(!out) throw std::runtime_error("open output");
    VH vh{{'G','A','P','8','V','E','C','\0'},1,8,5}; out.write((char*)&vh,sizeof(vh));
    static constexpr int DEL[3]={0,-1,1}; State z{}; z.n=8;
    for(int a=0;a<3;++a){
        int h=1+DEL[a]; WVec v{{pack(z),1}}; auto w=wcolumn(std::move(v),8,true,a); auto q=project_raw_combo(*G[h],w);
        std::vector<uint16_t> idx; idx.reserve(q.size()); for(auto[j,x]:q){if(x!=1)throw std::runtime_error("alpha nonunit");idx.push_back((uint16_t)j);}
        GapVecEntry e{1u,(uint32_t)a,(uint32_t)h,(uint32_t)idx.size()}; out.write((char*)&e,sizeof(e)); out.write((char*)idx.data(),idx.size()*2);
        std::cout<<"alpha a="<<a<<" h="<<h<<" nz="<<idx.size()<<"\n";
    }
    for(auto[h,a]:{std::pair<int,int>{0,0},{1,1}}){
        std::vector<uint16_t> idx; std::vector<std::pair<int,int>> src;
        for(int si=0;si<(int)S[h]->states.size();++si) if(G[h]->raw_gap_global[si]>=0) src.push_back({G[h]->raw_gap_global[si],si});
        std::sort(src.begin(),src.end());
        for(auto[gi,si]:src){ WVec v{{S[h]->states[si],1}}; auto x=wlast_value(std::move(v),8,a); if(x){if(x!=1)throw std::runtime_error("beta nonunit"); idx.push_back((uint16_t)gi);} }
        GapVecEntry e{2u,(uint32_t)a,(uint32_t)h,(uint32_t)idx.size()}; out.write((char*)&e,sizeof(e)); out.write((char*)idx.data(),idx.size()*2);
        std::cout<<"beta a="<<a<<" h="<<h<<" nz="<<idx.size()<<"\n";
    }
    out.close(); std::cout<<"path="<<path<<" bytes="<<std::filesystem::file_size(path)<<"\n";
}
