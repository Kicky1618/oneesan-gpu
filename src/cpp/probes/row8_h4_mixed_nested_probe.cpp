#define ROW8_RAW_PREFIX_NO_MAIN 1
#include "row8_raw_prefix_vector.cpp"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>
static std::array<std::vector<uint32_t>,9> loadpc(){std::ifstream in("src/cuda/b300/row8_pivots_w19.bin",std::ios::binary);char m[8];uint32_t v,ds[9];in.read(m,8);in.read((char*)&v,4);in.read((char*)ds,36);std::array<std::vector<uint32_t>,9>p;for(int h=0;h<9;++h){p[h].resize(ds[h]);for(auto&x:p[h]){uint32_t sc;in.read((char*)&x,4);in.read((char*)&sc,4);}}return p;}
static Packed mixed(int gap,int outerStatus,int innerStatus){State s{};s.n=8;s.sp=4;int q=1,sk=0;std::vector<int>free;for(int i=0;i<8;++i){if(i>=gap&&i<gap+4)free.push_back(i);else{int qc=q++;s.deg[i]=1;s.comp[i]=qc;s.stack[sk++]=qc;}}int a=free[0],b=free[1],c=free[2],d=free[3];int qo=q++,qi=q++;s.deg[a]=s.deg[d]=1;s.comp[a]=s.comp[d]=qo;s.status[qo]=outerStatus;s.deg[b]=s.deg[c]=1;s.comp[b]=s.comp[c]=qi;s.status[qi]=innerStatus;s.ns=q;return pack(s);}
static uint32_t coeff(WVec const&v,Packed p){auto it=std::lower_bound(v.begin(),v.end(),WEntry{p,0},[](auto const&a,auto const&b){return a.p<b.p;});return it!=v.end()&&it->p==p?it->v:0;}
int main(){MODP=1000000007u;auto P=loadpc();constexpr int S=8;std::array<WVec,S>E;for(int i=0;i<S;++i){auto v=prefix_vec(P[5][i]);E[i]=wcolumn(v,8,false,1);}for(int g=0;g<=4;++g)for(auto st: {std::pair<int,int>{0,1},std::pair<int,int>{1,0}}){auto p=mixed(g,st.first,st.second);std::cout<<"gap="<<g<<" outer="<<st.first<<" inner="<<st.second;for(int i=0;i<S;++i)std::cout<<' '<<coeff(E[i],p);std::cout<<'\n';}}
