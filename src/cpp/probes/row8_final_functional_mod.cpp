#define ROW8_RAW_PREFIX_NO_MAIN 1
#include "row8_raw_prefix_vector.cpp"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>
using i64=int64_t;
static uint32_t P;
static uint32_t invp(uint32_t a){uint64_t e=(uint64_t)P-2,x=a,r=1;while(e){if(e&1)r=(__uint128_t)r*x%P;x=(__uint128_t)x*x%P;e>>=1;}return (uint32_t)r;}
struct RR{int p;std::vector<std::pair<int,uint32_t>>z;};
struct Red{int n;std::vector<int>at;std::vector<RR>b;Red(int n):n(n),at(n,-1){}void reduce(std::vector<uint32_t>&y)const{for(int p=0;p<n;++p)if(y[p]&&at[p]>=0){uint32_t f=y[p];for(auto[k,v]:b[at[p]].z){uint32_t s=(uint32_t)((__uint128_t)f*v%P);y[k]=y[k]>=s?y[k]-s:y[k]+P-s;}}}bool add(std::vector<uint32_t>y){reduce(y);int p=-1;for(int i=0;i<n;++i)if(y[i]){p=i;break;}if(p<0)return false;uint32_t iv=invp(y[p]);RR r{p,{}};for(int i=p;i<n;++i)if(y[i])r.z.push_back({i,(uint32_t)((__uint128_t)y[i]*iv%P)});at[p]=b.size();b.push_back(std::move(r));return true;}bool inSpan(std::vector<uint32_t>y)const{reduce(y);for(auto x:y)if(x)return false;return true;}};
struct IH{char m[8];uint32_t ver,h,rows,states;uint64_t maxabs;};
static Red loadRed(std::string path,int h,int rows,int states){std::ifstream in(path,std::ios::binary);IH q{};in.read((char*)&q,sizeof(q));if(!in||q.h!=(uint32_t)h||q.rows!=(uint32_t)rows||q.states!=(uint32_t)states)throw std::runtime_error("hdr");Red R(states);std::vector<i64>x(states);for(int r=0;r<rows;++r){in.read((char*)x.data(),x.size()*8);std::vector<uint32_t>y(states);for(int i=0;i<states;++i){i64 z=x[i]%(i64)P;if(z<0)z+=P;y[i]=(uint32_t)z;}if(!R.add(std::move(y)))throw std::runtime_error("rank drop");}return R;}
int main(int argc,char**argv){P=argc>1?std::strtoul(argv[1],nullptr,10):4294967291u;MODP=4294967291u;Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::vector<Packed>h0,h1;for(auto&p:all){int h=unpack(p).sp;if(h==0)h0.push_back(p);else if(h==1)h1.push_back(p);}auto R0=loadRed("work/formal-probes/dual-basis/Phi_h0_integer.bin",0,1107,h0.size());auto R1=loadRed("work/formal-probes/dual-basis/Phi_h1_integer.bin",1,1640,h1.size());std::vector<uint32_t>g0(h0.size()),g1(h1.size());for(int i=0;i<(int)h0.size();++i){WVec v{{h0[i],1}};g0[i]=wlast_value(std::move(v),8,0)%P;}for(int i=0;i<(int)h1.size();++i){WVec v{{h1[i],1}};g1[i]=wlast_value(std::move(v),8,1)%P;}bool a=R0.inSpan(g0),b=R1.inSpan(g1);std::cout<<"final_functional_mod p="<<P<<" h0N="<<a<<" h1R="<<b<<" exact="<<(a&&b)<<"\n";return a&&b?0:1;}
