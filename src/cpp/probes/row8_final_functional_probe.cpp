#define ROW8_RAW_PREFIX_NO_MAIN 1
#include "row8_raw_prefix_vector.cpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
using i64=int64_t;
struct IH{char m[8];uint32_t ver,h,rows,states;uint64_t maxabs;};
static std::vector<i64> loadI(std::string p,int h,int rows,int states){std::ifstream in(p,std::ios::binary);IH x{};in.read((char*)&x,sizeof(x));if(!in||x.h!=(uint32_t)h||x.rows!=(uint32_t)rows||x.states!=(uint32_t)states)throw std::runtime_error("hdr");std::vector<i64>A((size_t)rows*states);in.read((char*)A.data(),A.size()*8);return A;}
int main(){MODP=4294967291u;Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::array<std::vector<Packed>,2>H;for(auto&p:all){int h=unpack(p).sp;if(h<=1)H[h].push_back(p);}std::array<std::vector<uint32_t>,2>g;g[0].resize(H[0].size());g[1].resize(H[1].size());for(int h=0;h<=1;++h){int sym=h==0?0:1;uint64_t nz=0,mx=0,sum=0;for(int i=0;i<(int)H[h].size();++i){WVec v{{H[h][i],1}};uint32_t x=wlast_value(std::move(v),8,sym);g[h][i]=x;if(x){++nz;mx=std::max<uint64_t>(mx,x);sum+=x;}}std::cout<<"final h="<<h<<" sym="<<sym<<" states="<<H[h].size()<<" nz="<<nz<<" max="<<mx<<" sum="<<sum<<"\n";}
 auto P0=loadI("work/formal-probes/dual-basis/Phi_h0_integer.bin",0,1107,H[0].size());auto P1=loadI("work/formal-probes/dual-basis/Phi_h1_integer.bin",1,1640,H[1].size());for(int h=0;h<=1;++h){auto const&A=h?P1:P0;int rows=h?1640:1107,states=H[h].size();int exact=-1,neg=-1;for(int r=0;r<rows;++r){bool eq=true,ne=true;for(int i=0;i<states;++i){i64 a=A[(size_t)r*states+i];if(a!=(i64)g[h][i])eq=false;if(a!=-(i64)g[h][i])ne=false;if(!eq&&!ne)break;}if(eq)exact=r;if(ne)neg=r;}std::cout<<"final_basis_match h="<<h<<" exact_row="<<exact<<" neg_row="<<neg<<"\n";}
}
