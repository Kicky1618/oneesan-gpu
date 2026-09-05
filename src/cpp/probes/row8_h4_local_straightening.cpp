#define ROW8_RAW_PREFIX_NO_MAIN 1
#include "row8_raw_prefix_vector.cpp"
#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <vector>
static std::array<std::vector<uint32_t>,9> loadpc(){std::ifstream in("src/cuda/b300/row8_pivots_w19.bin",std::ios::binary);char m[8];uint32_t v,ds[9];in.read(m,8);in.read((char*)&v,4);in.read((char*)ds,36);std::array<std::vector<uint32_t>,9>p;for(int h=0;h<9;++h){p[h].resize(ds[h]);for(auto&x:p[h]){uint32_t sc;in.read((char*)&x,4);in.read((char*)&sc,4);}}return p;}
static std::string sig(State s){std::set<int> st;for(int i=0;i<s.sp;++i)st.insert(s.stack[i]);std::ostringstream o;o<<"through=";for(int i=0;i<8;++i)if(s.deg[i]&&st.count(s.comp[i]))o<<i;o<<" pairs=";std::map<int,std::vector<int>> pos;for(int i=0;i<8;++i)if(s.deg[i]&&!st.count(s.comp[i]))pos[s.comp[i]].push_back(i);for(auto const&[q,v]:pos){if(v.size()==2)o<<v[0]<<'-'<<v[1]<<':'<<(int)(s.status[q]&1)<<',';}return o.str();}
int main(){MODP=1000000007u;auto P=loadpc();constexpr int S=8;std::array<WVec,S>E;for(int i=0;i<S;++i){auto v=prefix_vec(P[5][i]);E[i]=wcolumn(v,8,false,1);}std::map<std::string,std::array<uint32_t,S>> cols;for(int i=0;i<S;++i)for(auto const&e:E[i]){State s=unpack(e.p);if(s.sp!=4)continue;int mask=0;for(int q=0;q<8;++q)if(s.deg[q])mask|=1<<q;if(mask!=255)continue;cols[sig(s)][i]=e.v;}for(int gap=0;gap<=4;++gap){std::string t="";for(int i=0;i<8;++i)if(i<gap||i>=gap+4)t+=char('0'+i);std::cout<<"=== gap="<<gap<<" targetThrough="<<t<<" ===\n";for(auto const&[k,v]:cols)if(k.find("through="+t+" ")==0){std::cout<<k;for(auto x:v)std::cout<<' '<<x;std::cout<<'\n';}}
}
