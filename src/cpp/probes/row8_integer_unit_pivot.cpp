#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>
using i128=__int128_t;using i64=int64_t;
struct H{char magic[8];uint32_t ver,h,rows,states;uint64_t maxabs;};
static std::string s128(i128 x){if(!x)return"0";bool n=x<0;if(n)x=-x;std::string s;while(x){s.push_back('0'+x%10);x/=10;}if(n)s.push_back('-');std::reverse(s.begin(),s.end());return s;}
static int run(std::string path){std::ifstream in(path,std::ios::binary);H h{};in.read((char*)&h,sizeof(h));if(!in)throw std::runtime_error("hdr");std::vector<std::vector<std::pair<int,i128>>> B;std::vector<int>at(h.states,-1);i128 mx=0;for(uint32_t r=0;r<h.rows;++r){std::vector<i64>a(h.states);in.read((char*)a.data(),a.size()*8);std::vector<i128>y(h.states);for(uint32_t i=0;i<h.states;++i)y[i]=a[i];for(uint32_t p=0;p<h.states;++p)if(y[p]&&at[p]>=0){auto const&z=B[at[p]];i128 piv=0;for(auto[k,v]:z)if(k==(int)p){piv=v;break;}if(piv!=1&&piv!=-1)throw std::runtime_error("stored nonunit");i128 f=y[p]/piv;for(auto[k,v]:z){y[k]-=f*v;i128 q=y[k]<0?-y[k]:y[k];if(q>mx)mx=q;}}int p=-1;for(uint32_t i=0;i<h.states;++i)if(y[i]){p=i;break;}if(p<0){std::cout<<"dependent row="<<r<<"\n";return 2;}if(y[p]!=1&&y[p]!=-1){std::cout<<"first_nonunit h="<<h.h<<" row="<<r<<" pivot_col="<<p<<" pivot="<<s128(y[p])<<" current_rank="<<B.size()<<" maxabs="<<s128(mx)<<"\n";return 1;}std::vector<std::pair<int,i128>>z;for(uint32_t k=p;k<h.states;++k)if(y[k])z.push_back({(int)k,y[k]});at[p]=B.size();B.push_back(std::move(z));if((r&127)==0)std::cerr<<"h="<<h.h<<" row="<<r<<" nnz="<<B.back().size()<<" mx="<<s128(mx)<<"\n";}std::cout<<"unit_pivot_exact h="<<h.h<<" rank="<<B.size()<<" states="<<h.states<<" maxabs="<<s128(mx)<<"\n";return 0;}
int main(){int a=run("work/formal-probes/dual-basis/Phi_h1_integer.bin");int b=run("work/formal-probes/dual-basis/Phi_h0_integer.bin");return a||b;}
