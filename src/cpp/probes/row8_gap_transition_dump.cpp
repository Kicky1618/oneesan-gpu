#define ROW8_GAP_TRANSITION_NO_MAIN 1
#include "row8_gap_transition_exact.cpp"
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

struct FileHdr {
    char magic[8];
    uint32_t version;
    uint32_t r;
    uint32_t dims[9];
    uint64_t total_nnz;
};
struct BlockHdr {
    uint32_t sym,h,h2,src_dim,dst_dim;
    uint64_t nnz;
};

int main(int argc,char**argv){
    std::string path=argc>1?argv[1]:"work/row8_gap/row8_gap_u01_v1.bin";
    MODP=Q;
    Vec all; int col=0;
    if(!load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all)) return 2;
    std::array<std::vector<Packed>,9> H;
    for(auto const&p:all) H[unpack(p).sp].push_back(p);
    std::array<std::unique_ptr<Space>,9> S;
    S[0]=std::make_unique<Space>(makeSpace(0,H[0],loadA01(0,H[0].size())));
    S[1]=std::make_unique<Space>(makeSpace(1,H[1],loadA01(1,H[1].size())));
    S[2]=std::make_unique<Space>(makeSpace(2,H[2],loadA2(H[2])));
    for(int h=3;h<=8;++h) S[h]=std::make_unique<Space>(makeSpace(h,H[h],makeAupper(h,H[h])));
    std::array<std::unique_ptr<GSpace>,9> G;
    for(int h=0;h<=8;++h) G[h]=std::make_unique<GSpace>(make_gspace(*S[h]));

    struct DumpBlock { BlockHdr h{}; std::vector<uint32_t> rp; std::vector<uint16_t> ci; };
    std::vector<DumpBlock> blocks;
    static constexpr int DEL[3]={0,-1,1};
    uint64_t total=0,nonunit=0;
    for(int a=0;a<3;++a) for(int h=0;h<=8;++h){
        int h2=h+DEL[a]; if(h2<0||h2>8) continue;
        DumpBlock d; d.h={uint32_t(a),uint32_t(h),uint32_t(h2),uint32_t(S[h]->dim),uint32_t(S[h2]->dim),0};
        d.rp.reserve(S[h]->dim+1); d.rp.push_back(0);
        std::vector<std::pair<int,int>> src;
        for(int si=0;si<(int)S[h]->states.size();++si) if(G[h]->raw_gap_global[si]>=0)
            src.push_back({G[h]->raw_gap_global[si],si});
        std::sort(src.begin(),src.end());
        for(auto [gi,si]:src){
            WVec v{{S[h]->states[si],1}};
            auto z=wcolumn(std::move(v),8,false,a);
            auto o=project_raw_combo(*G[h2],z);
            for(auto [j,x]:o){ if(x!=1) ++nonunit; d.ci.push_back(uint16_t(j)); }
            d.rp.push_back(d.ci.size());
        }
        d.h.nnz=d.ci.size(); total+=d.ci.size(); blocks.push_back(std::move(d));
    }
    if(nonunit) throw std::runtime_error("non-unit gap transition");
    FileHdr fh{{'G','A','P','8','U','0','1','\0'},1,8,{0},total};
    for(int h=0;h<=8;++h) fh.dims[h]=S[h]->dim;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path,std::ios::binary|std::ios::trunc); if(!out) throw std::runtime_error("open output");
    out.write((char*)&fh,sizeof(fh));
    uint32_t nb=blocks.size(); out.write((char*)&nb,sizeof(nb));
    for(auto const&d:blocks){
        out.write((char*)&d.h,sizeof(d.h));
        out.write((char*)d.rp.data(),d.rp.size()*sizeof(uint32_t));
        out.write((char*)d.ci.data(),d.ci.size()*sizeof(uint16_t));
    }
    out.close();
    std::cout<<"path="<<path<<" blocks="<<blocks.size()<<" total_nnz="<<total
             <<" bytes="<<std::filesystem::file_size(path)<<"\n";
    for(auto const&d:blocks) std::cout<<"  a="<<d.h.sym<<" h="<<d.h.h<<" h2="<<d.h.h2
        <<" src="<<d.h.src_dim<<" dst="<<d.h.dst_dim<<" nnz="<<d.h.nnz<<"\n";
}
