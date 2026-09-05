#define ROW8_H2_PARTITION_NO_MAIN 1
#include "row8_h2_relation_partition.cpp"
#include <fstream>
#include <iostream>
#include <map>
#include <set>
struct HH{char magic[8];uint32_t ver,mod,h,rows,states;};
static uint32_t mask(Packed p){State s=unpack(p);uint32_t m=0;for(int i=0;i<8;++i)if(s.deg[i])m|=1u<<i;return m;}
int main(){Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::vector<Packed>h2;for(auto&p:all)if(unpack(p).sp==2)h2.push_back(p);std::ifstream in("work/formal-probes/dual-basis/Phi_h2_extra_mod1000000007.bin",std::ios::binary);HH hh{};in.read((char*)&hh,sizeof(hh));std::vector<uint32_t>v(hh.states);std::map<int,int> maskCountHist,jCountHist,nzHist;int single=0;for(uint32_t r=0;r<hh.rows;++r){in.read((char*)v.data(),v.size()*4);std::set<uint32_t>ms;int nz=0;for(size_t i=0;i<v.size();++i)if(v[i]){++nz;ms.insert(mask(h2[i]));}++maskCountHist[ms.size()];std::set<int>js;for(auto m:ms)js.insert(__builtin_popcount(m));++jCountHist[js.size()];++nzHist[nz];if(ms.size()==1)++single;if(r<10){std::cout<<"r="<<r<<" nz="<<nz<<" masks="<<ms.size()<<" js=";for(int j:js)std::cout<<j<<',';std::cout<<"\n";}}
std::cout<<"rows="<<hh.rows<<" singleMask="<<single<<" maskHist";for(auto[a,b]:maskCountHist)std::cout<<' '<<a<<':'<<b;std::cout<<" jsetHist";for(auto[a,b]:jCountHist)std::cout<<' '<<a<<':'<<b;std::cout<<"\n";}
