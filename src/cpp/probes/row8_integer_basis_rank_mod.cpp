#define ROW8_H2_PARTITION_NO_MAIN 1
#include "row8_h2_relation_partition.cpp"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <utility>
#include <vector>
using i64=int64_t;
static uint32_t P;
static uint32_t invp(uint32_t a){uint64_t e=(uint64_t)P-2,x=a,r=1;while(e){if(e&1)r=(__uint128_t)r*x%P;x=(__uint128_t)x*x%P;e>>=1;}return (uint32_t)r;}
struct RR{int p;std::vector<std::pair<int,uint32_t>>z;};
struct Red{int n;std::vector<int>at;std::vector<RR>b;Red(int n):n(n),at(n,-1){}bool add_i64(i64 const*x){std::vector<uint32_t>y(n);for(int i=0;i<n;++i){i64 q=x[i]%(i64)P;if(q<0)q+=P;y[i]=q;}return add(std::move(y));}bool add(std::vector<uint32_t>y){for(int p=0;p<n;++p)if(y[p]&&at[p]>=0){uint32_t f=y[p];for(auto[k,v]:b[at[p]].z){uint32_t s=(uint32_t)((__uint128_t)f*v%P);y[k]=y[k]>=s?y[k]-s:y[k]+P-s;}}int p=-1;for(int i=0;i<n;++i)if(y[i]){p=i;break;}if(p<0)return false;uint32_t iv=invp(y[p]);RR r{p,{}};for(int i=p;i<n;++i)if(y[i])r.z.push_back({i,(uint32_t)((__uint128_t)y[i]*iv%P)});at[p]=b.size();b.push_back(std::move(r));return true;}};
struct IH{char m[8];uint32_t ver,h,rows,states;uint64_t maxabs;};
static int rank_file(std::string path){std::ifstream in(path,std::ios::binary);IH h{};in.read((char*)&h,sizeof(h));if(!in)throw std::runtime_error("hdr");Red R(h.states);std::vector<i64>x(h.states);auto t=std::chrono::steady_clock::now();for(uint32_t r=0;r<h.rows;++r){in.read((char*)x.data(),x.size()*8);R.add_i64(x.data());if((r&127)==0)std::cerr<<"h="<<h.h<<" row="<<r<<" rank="<<R.b.size()<<"\n";}double s=std::chrono::duration<double>(std::chrono::steady_clock::now()-t).count();std::cout<<"rank h="<<h.h<<" p="<<P<<" rank="<<R.b.size()<<"/"<<h.rows<<" sec="<<s<<"\n";return R.b.size();}
static int rank_h2(){MODP=P;Vec all;int col=0;load_ck("work/formal-probes/raw_wfa_r8.ck",8,col,all);std::vector<Packed>h2;for(auto&p:all)if(unpack(p).sp==2)h2.push_back(p);auto ws=words2();std::set<int>bad;{std::ifstream q("work/formal-probes/canonical-matrix/h2_actual_bad_indices.txt");int x;while(q>>x)bad.insert(x);}std::vector<char>good(h2.size());int g=0;for(int j=0;j<(int)ws.size();++j)if(!bad.count(j)){int i=std::lower_bound(h2.begin(),h2.end(),ws[j].first)-h2.begin();good[i]=1;++g;}std::vector<int>non;for(int i=0;i<(int)h2.size();++i)if(!good[i])non.push_back(i);Red R(non.size());struct HH{char m[8];uint32_t ver,mod,h,rows,states;}hh{};std::ifstream in("work/formal-probes/dual-basis/Phi_h2_extra_mod1000000007.bin",std::ios::binary);in.read((char*)&hh,sizeof(hh));std::vector<uint32_t>row(h2.size());for(int r=0;r<120;++r){in.read((char*)row.data(),row.size()*4);std::vector<uint32_t>y(non.size());for(int k=0;k<(int)non.size();++k){uint32_t x=row[non[k]];int64_t z=x<=500000003u?(int64_t)x:(int64_t)x-1000000007LL;z%=P;if(z<0)z+=P;y[k]=(uint32_t)z;}R.add(std::move(y));}std::cout<<"rank h=2 p="<<P<<" rank="<<g+R.b.size()<<"/1428 extra="<<R.b.size()<<"/120\n";return g+R.b.size();}
int main(int argc,char**argv){P=argc>1?std::strtoul(argv[1],nullptr,10):4294967291u;int r1=rank_file("work/formal-probes/dual-basis/Phi_h1_integer.bin");int r0=rank_file("work/formal-probes/dual-basis/Phi_h0_integer.bin");int r2=rank_h2();return (r0==1107&&r1==1640&&r2==1428)?0:1;}
